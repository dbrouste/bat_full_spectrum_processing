# Experimental Flattop reassignment (branch Reassignment)

This experimental module does not change `adaptive_v4` or the default public detector.
Unlike the current detection pipeline (Hann STFT), this computes a separate
Flattop STFT and time/frequency reassigned energy.

```python
import soundfile as sf
from bat_analysis.detection import detect_chirps
from bat_analysis.reassignment import (
    reassigned_spectrogram, reassigned_db, refine_detection_peaks,
)

y, sr = sf.read("example.wav")
if y.ndim == 2:
    y = y.mean(axis=1)

r = reassigned_spectrogram(
    y, sr, n_fft=512, hop=64, fmin=20000, fmax=150000,
    noise_percentile=50, snr_min_db=4, snr_full_db=16,
)
# r["reassigned_power"] and r["stft_power"] share the same axes:
# r["times_s"], r["freqs_hz"]
# IMPORTANT: reassigned cells have summed power and are not spectrally
# calibrated against STFT cell values.
v4 = detect_chirps(y, sr)   # unchanged baseline
hybrid = refine_detection_peaks(
    y, sr, v4, n_fft=512, hop=64, snr_min_db=4, snr_full_db=16
)
```

Theory: STFT with g, centered t*g and discrete derivative g'.
Reassigned positions are calculated from complex STFT ratios. Nearest-neighbour
power accumulation makes thin ridges but can fragment weak calls. The progressive
4–16 dB mask weights original bin energy before relocation; it is NOT applied
to the reassigned image. Out-of-band estimates are discarded.

Caveats: 1) reassignment cannot exceed the waveform's information bandwidth;
2) relative and absolute energy calibration differ across windows;
3) zero-padded edges are not analyzed; 4) `noise_percentile` depends heavily
on call occupancy; 5) conservative hybrid mode only adjusts PEAK positions,
not onset/offset, blob topology or ridge modelling.

Next benchmark against manual annotation: compare recall, precision,
frequency/time errors and runtime for Hann adaptive_v4 vs Flattop reassignment
512/1024; specifically inspect weak calls and crossing chirps before
integrating reassigned blobs into candidate generation.

## Experimental trajectory reconstruction

`process_full_spectrum` now accepts `tracking_method="reassignment_flattop"`.
On the `Reassignment` branch, the default is now `"reassignment_flattop"`.
The new mode replaces the **trajectory tracking stage**, not the SNR blob detector. It follows reassigned Flattop
power in both directions from the supplied seed using a continuity-penalized
local peak selector with a short gap allowance. It does not yet use the legacy
Gaussian line-fit iterations, so these two reconstruction methods must be
benchmarked independently. The result is a three-column array
`[time_seconds, frequency_hz, normalized_amplitude]` in the same 15-ms
chunk-relative temporal convention as the legacy model.

```python
from bat_analysis.modelling import process_full_spectrum

diagnostics = {}
curve = process_full_spectrum(
    y, sr,
    time_mid=candidate["time_mid"],
    duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    broad_fm_seed_freq_hz=candidate.get("broad_fm_seed_freq_hz"),
    tracking_method="reassignment_flattop",
    reassignment_n_fft=512,
    reassignment_hop=64,
    diagnostics=diagnostics,
)
```

Limitations: when the seed is incorrect or separated harmonics overlap, the
greedy path can jump or stop. Reassigned power is sparse and is not a Gaussian
amplitude measure; curve amplitudes are normalized. No claim of improved
accuracy is made before testing against manually annotated WAV files.

## Global trajectory tracker (second iteration)

The former greedy ridge follower is replaced by a second-order dynamic
program with two states: frequency bin and frequency change per frame.
Negative/zero frequency slope (forward in time) is enforced, and changes in
slope are penalized. The evidence map blends smoothed reassigned power (65%)
and regular Flattop STFT power (35%) after per-frequency noise subtraction.
This reduces jumps between disconnected reassigned pixels.

The optional entry point remains:

```python
curve = process_full_spectrum(
    y, sr, time_mid=candidate["time_mid"], duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    tracking_method="reassignment_flattop",
    reassignment_n_fft=512, reassignment_hop=48,
)
```

**Important:** the preceding example uses the detector's seed. The available
manual-label benchmark is *oracle seeded*, meaning seed time and seed frequency
are computed from manual chirp annotations, not from the detector. It therefore
evaluates trajectory reconstruction separately from detector reliability.

For a repeatable annotated benchmark:

```bash
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs
```

Development run (140 manually annotated calls, four WAVs, 384 kHz),
computed on the prototype prior to GitHub integration:

| WAV | annotated | trajectories returned | median absolute frequency error |
|---|---:|---:|---:|
| 20261005_191936.wav | 61 | 57 | 0.92 kHz |
| 20261005_191947.wav | 38 | 26 | 1.47 kHz |
| 20261005_192441.wav | 27 | 24 | 1.05 kHz |
| 20261005_192501.wav | 14 | 14 | 0.80 kHz |

Errors measured only over extracted points inside the annotated temporal
bounds against PCHIP annotations (linear interpolation for two-point labels).
The failures are disproportionately short/high-frequency harmonic traces.
These results are not a benchmark of autonomous seed placement, nor a
side-by-side comparison against the legacy tracker.

## November 2026 — trajectory optimization cycles

The earlier *greedy* and first global-tracker results above are historical
baselines. The current experimental branch uses the following defaults:

- Flattop FFT 512, **hop 24** (62.5 µs at Fs=384 kHz).
- Global frequency/velocity dynamic program with **jump_penalty=0.11**
  and **gap_penalty=0.65**, compared with 0.035 and 1.15 previously.
- For chirps shorter than 1.2 ms: shorter required trajectory and a
  larger permitted frequency rate (up to 110 kHz/ms).
- **Sub-bin frequency refinement** from a reassigned-power centroid within
  three bins of the selected path; can be disabled using
  `reassignment_subbin_refine=False` in `process_full_spectrum`
  or `subbin_refine=False` in `track_reassigned_ridge`.

Example:

```python
curve = process_full_spectrum(
    y, sr, time_mid=candidate["time_mid"], duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    tracking_method="reassignment_flattop",
    reassignment_n_fft=512,
    reassignment_hop=24,
    reassignment_subbin_refine=True,
)
```

The short-harmonic profile may be evaluated explicitly with
`reassignment_n_fft=192, reassignment_hop=16`. This setting is **not** the
default, since the four-file validation set contains no annotated calls
shorter than 1 ms.

### Local prototype measurements

The optimization was evaluated against 140 validated manual chirps in four
recordings; 99 annotations in `191936/191947` were used for development
and 41 in `192441/192501` were held aside for validation. These are
correlated recordings, not a separate field-site or species cohort.

| Metric (oracle-seeded) | Prior global tracker | Tuned global tracker | Tuned + sub-bin |
|---|---:|---:|---:|
| Track returned (140 labels) | 121 | 140 | 140 |
| Mean full-interval error, development (kHz) | not measured in same protocol | 1.72 | 1.56 |
| Mean full-interval error, validation (kHz) | not measured in same protocol | 1.35 | 1.13 |
| Mean temporal coverage, validation | not measured in same protocol | 95.8% | 95.8% |

For the intermediate 512/24 prototype with previous penalties, the
validation mean error was 2.75 kHz with 81% temporal coverage. This is the
direct comparator for the tuned configuration in the same evaluation.

**Limitations:** Seeds are generated from the manual annotations, so these
figures do not establish autonomous detection recall or robustness to an
incorrect detector candidate. Deterministic seed-jitter stress tests with
up to +/-0.5 ms time, +/-10 kHz frequency and +/-30% duration were run,
but they do not replace end-to-end detector testing. In 100 negative
windows far from annotated calls, the provisional track rate was 1/100
before penalty tuning and 3/100 after. A proposed track does not by itself
imply that a bat chirp was present.

To reproduce the documented protocol with local WAVs and annotations:

```bash
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs \
  --jitter-time-ms 0.5 --jitter-freq-khz 10 \
  --jitter-duration-fraction 0.3
```

The GitHub integration and the local prototype have the same algorithmic
parameters, but the complete GitHub test suite has **not yet been verified**
in this work session.

## Detector-seeded tracking and short-harmonic study

Following the oracle-seeded benchmark, we evaluated the upstream
`adaptive_v4` logic on four labelled WAVs using a local faithful
reimplementation of its Hann-SNR/blob candidate calculations. This
proxy experiment must be rerun with the upstream library using
`benchmark.reassignment_detector_seeds` before claiming a final detector
recall figure.

There are 140 validated manual trajectories in the four files.
The local detector proxy yielded 90 candidates, of which 83 could be
matched one-to-one to annotated trajectories using peak-time and
peak-frequency constraints. **Unmatched candidates are not automatically
false positives:** missing annotations or harmonics may explain some.

### Anchoring close to the detector peak

The older tracker searched for the strongest seed within **+/-1.5 ms**,
which could be a different nearby call or harmonic. It now searches
within **+/-0.5 ms** by default (at least half a frame hop) and keeps
the corresponding frequency near the detector's peak (+/-12 kHz).

In the 83 matched candidate examples, the full-interval mean
frequency error improved from **1.29 kHz to 1.15 kHz**, while matched
paths exceeding 5 kHz mean error decreased from **2 to 0**.
With random seed perturbations of +/-0.3 ms, +/-3 kHz and
+/-20% duration, error improved from 1.34 to 1.19 kHz (166 trials).
With larger perturbations (+/-0.75 ms, +/-10 kHz, +/-30% duration),
the benefit was less pronounced; seed placement is still a key limitation.

The wrapper `track_reassigned_candidate(y, sr, candidate)` accepts
the full detector candidate and uses its *actual bbox start/end* for
analysis rather than assuming the peak is the chirp midpoint. If
`broad_fm_seed_freq_hz` is available, its paired
`coarse_seed_time_s` is required before using it. The alternative
`process_full_spectrum(..., tracking_method="reassignment_flattop")`
accepts `chirp_start_s`, `chirp_end_s` and
`broad_fm_seed_time_s` as optional keyword arguments.

### Automatic frequency/time resolution

A separate oracle-seeded evaluation of **21 annotations under 1.2 ms**
(including harmonics) compared various FFT sizes on the same WAVs:

| Flattop | Mean full-interval error (kHz) | Tracked |
|---|---:|---:|
| FFT 512 / hop 24 | 2.48 | 21/21 |
| FFT 256 / hop 16 | 1.74 | 21/21 |
| **FFT 192 / hop 16** | **1.38** | **21/21** |

The experimental default now chooses **192/16** below 1.2 ms,
otherwise **512/24**. Explicit `reassignment_n_fft=512` and
`reassignment_hop=24` override this selection. This is a
time-frequency trade-off; a short window broadens the frequency
main lobe and should not be assumed optimal for all species.

### Reproducing the exact pipeline benchmark

```bash
python -m benchmark.reassignment_detector_seeds \
  Nycto-Thomas.json /path/to/wavs \
  --output detector_reassignment_report.json

# Fixed-resolution control:
python -m benchmark.reassignment_detector_seeds \
  Nycto-Thomas.json /path/to/wavs --n-fft 512 --hop 24
```

The output separates annotated chirps, detector candidates, one-to-one
matches, missed annotations, unmatched candidate counts, completed tracks,
per-chirp frequency errors, and coverage. Neither manual time nor manual
frequency is passed to the tracker during this benchmark.

All these changes remain on the `Reassignment` branch. The `main` branch retains the legacy default; only `Reassignment`
uses the reassigned tracker by default. GitHub CI execution is still
pending verification.

## Cross-seed slope continuity

The global dynamic program previously optimized left and right sides of a
ridge independently. Near a crossing harmonic, the right-hand side could
select a spuriously flat, high-energy branch despite a steep left-hand
approach. The left-hand slope is now summarized over its final eight
time-ordered points and supplied as a **soft prior** for the first
0.55 ms of rightward tracking.

Defaults:
- `cross_seed_velocity_penalty=0.6`
- `cross_seed_position_penalty=0.01`
- `prior_horizon_ms=0.55`

This is not a hard global straight-line assumption: the prior expires,
and the dynamic program remains free to select chirp curvature. Both
penalties can be set to zero to reproduce the previous behavior.

### Incremental local validation

The final configuration (automatic 192/16 below 1.2 ms; otherwise
512/24, +/-0.5 ms anchoring and the new cross-seed slope prior) was
tested against the same four WAVs and manual PCHIP curves:

| Measurement | Before cross-seed prior | After |
|---|---:|---:|
| Oracle seeded: mean full-interval error, 140 calls | 1.18 kHz | 1.07 kHz |
| Oracle seeded: mean full-interval error, 41 hold-out calls | 1.00 kHz | 1.01 kHz |
| Detector-proxy seeded: mean full-interval error, 83 matched calls | 1.14 kHz | 1.10 kHz |
| Negative windows returning a track, 100 sampled windows | 2 | 2 |

All 140 oracle-seeded calls and all 83 matched proxy-detector calls
produced a path. These results use a local implementation of the
reassignment formulas and a close reproduction of the V4 detector
where stated; the exact installed GitHub pipeline still needs its
CI/integration benchmark run. Error can vary with the matching
tolerance; unmatched candidates are listed separately.

One notably incorrect call, `20261005_191947.wav` chirp #37,
improved from approximately 7 kHz mean error to around 1 kHz
when the seed's approach slope was carried through the transition.
This is a diagnostic example, not evidence of generalized recovery
of overlapping harmonics.

The small ~0.01 kHz degradation on the 41-call validation group
means this prior needs evaluation on additional independent WAV
recordings before it can replace the baseline in `main`.

## Experimental second-harmonic recovery and observed-tail correction

**Reassignment tracking is now the default on this branch.**
To use the historical mode, pass `tracking_method="legacy"` explicitly.
The `main` branch is unchanged.

### Fundamental tracking and tail correction

The experimental tracker now calls `refine_ridge_endpoint` by default.
It fits the declining FM slope on the first 65% of the *observed*
trajectory, anchors a linear continuation at the splice, and limits
anomalously flat, high-frequency tail points to the continuation.
The correction is bounded to 16 kHz, does not modify timestamps or
amplitudes and **never extrapolates a missing time interval**.
It is disabled for CF and upward-sloping calls. The helper also
returns observed endpoint amplitude diagnostics.

To compare old versus new:

```python
raw = process_full_spectrum(
    y, sr, time_mid=candidate["time_mid"], duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    tracking_method="reassignment_flattop",
    reassignment_refine_endpoints=False,
)
corrected = process_full_spectrum(
    y, sr, time_mid=candidate["time_mid"], duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    tracking_method="reassignment_flattop",
    reassignment_refine_endpoints=True,
)
```

Local prototype evaluation with oracle seeds on the 140 validated
manual chirps, errors computed at 101 uniformly sampled annotation times:

| Mean absolute frequency error | Before | After |
|---|---:|---:|
| All 140 manual calls | 1.059 kHz | 0.984 kHz |
| 41 manual calls held aside for validation | 0.996 kHz | 0.963 kHz |
| Last third of each call | 1.434 kHz | 1.210 kHz |

The improvement applies only to measured points; the shortfall of
temporal coverage is unchanged. The annotations were used for seed
initialization in this experiment, not in the inference algorithm.

### Second-harmonic candidate retrieval

After obtaining a fundamental from an Adaptive V4 candidate, call:

```python
from bat_analysis.detection import detect_candidates_snr_blobs
from bat_analysis.harmonics import track_candidate_with_harmonics

candidates = detect_candidates_snr_blobs(y, sr)
for index, candidate in enumerate(candidates):
    fundamental, harmonics = track_candidate_with_harmonics(
        y, sr, candidate, parent_id=index,
    )
    for harmonic in harmonics:
        print(harmonic.harmonic_order, harmonic.parent_id,
              harmonic.points, harmonic.mean_snr_db)
```

The harmonic helper uses Flattop FFT 192 / hop 12, searches near
`2*f_fundamental(t)` with up to 20 kHz of tolerance, and requires a
short coherent segment with a 4-16 dB compatible SNR-style threshold
(local detection threshold 8 dB; mean raw SNR at least 14 dB).
It reports real observed time, frequency and local SNR points.
The first coherent segment is preferred to later echo-like segments.
The parent's `parent_id` is retained and an associated harmonic
must **not** be counted as an independent bat passage.

In a local prototype evaluation of the 14 manually labelled second
harmonics in `20261005_191947.wav`, all 14 had overlapping detected
harmonic segments when the fundamental ridge was oracle seeded.
The mean frequency error inside each annotated harmonic span was
approximately 1.66 kHz. Three additional candidates were found
around fundamentals without an annotated second harmonic; these
are **unreviewed proposals**, not confirmed false positives.
A synthetic negative check with 40 noise seeds and fundamental-only
calls returned zero proposed harmonics; 40 synthetic positive cases
returned one segment each. Synthetic checks do not measure specificity
on environmental field noise.

### Reproducible benchmarks

```bash
# Oracle-seeded tracker A/B on the annotations.
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs \
  --no-endpoint-refinement

# Matched harmonic pairs and parent endpoint errors.
python -m benchmark.harmonic_annotations Nycto-Thomas.json /path/to/wavs

# Exact V4 candidates instead of manual seeds.
python -m benchmark.reassignment_detector_seeds \
  Nycto-Thomas.json /path/to/wavs
```

The automated detector can still miss fundamentals and the harmonic
algorithm cannot recover a missing parent call without another seed.
The exact integrated repository suite and CI checks are pending and
should be run before replacing production defaults.

## November 5 — missing-ridge-flank recovery

A spectrogram screenshot showed three fitted points close to the trailing
part of a conspicuous diagonal FM trace, without its earlier section.
Two independent issues were identified:

1. The detector's `time_mid` is its strongest-energy bin, **not** the
   temporal midpoint of the call. The old centered duration window could
   omit the left side before tracking had even started.
2. The global tracker can end a path when local evidence becomes sparse.

### Energy-peak-aligned analysis window

When no detector bbox is supplied, the experimental
`process_full_spectrum` entry point now uses an **asymmetric search window**:
`[peak_time - 1.15 * duration, peak_time + 0.85 * duration]`.
When the detector provides actual `t_start` and `t_end`,
those bounds take precedence. The direct low-level
`track_reassigned_ridge` API retains a centered window unless
`peak_anchored_window=True` is passed.

The location of the strongest Flattop-STFT energy was checked within 131
manual chirp intervals of at least 0.6 ms on the four uploaded WAVs.
55/131 had their strongest bin after 65% of the manually annotated call
duration. The *geometric* coverage of those annotated intervals by a
duration-width window around that measured peak averaged 81.3%.
The proposed asymmetric window had 99.96% geometric coverage.
This is a controlled crop calculation, **not a detector-seeded recall
or ridge accuracy benchmark**.

### High-confidence predictive completion

`bat_analysis/ridge_completion.py` fits an initial local slope at the
observed ridge's edge and extrapolates the frequency for the next
STFT frame. It accepts a proposed point only if a Flattop STFT
within a 10 kHz search band provides sufficiently strong noise-relative
energy and a plausible frequency direction. The local slope is updated
gradually; two weak frames may be bridged, but a group shorter than
five added points is discarded. It never synthesizes points without
corresponding signal energy or beyond the analysis bounds.

Experimental defaults:
- SNR threshold: **17 dB** (noise floor: 20th time percentile/bin).
- FFT/hop follow the ridge-tracker resolution (typically 512/24).
- Search radius: **10 kHz**, frequency-prior penalty: **0.14**.
- Maximum extension per side: **5 ms**.
- Original ridge samples remain exactly unchanged.
- `complete_ends=True` in the experimental tracker and
  `reassignment_complete_ends=True` in its modelling wrapper.

The completion is separate from `refine_ridge_endpoint`, the existing
frequency-only tail plateau correction. It runs **before** that correction.

### Local validation (separate from the end-to-end benchmark)

The prototype was tested on real WAVs using a controlled artificial
truncation: three points were interpolated from each annotation at
64%, 75% and 86% of its duration, and the extension searched only
for energy before those points. This stress test is not automatic
detection, as the initial points were obtained from annotations.

- 119 long annotated calls (duration >= 1.2 ms) eligible.
- Extension returned >=1 point for **116/119** calls.
- Mean recovery of the portion before the first seed: **93.1%**
  among those 116 calls, median frequency error **1.13 kHz**.
- With 100 unrelated windows at least 5 ms away from the nearest
  annotation, the 17 dB gate yielded **1** apparent extension of at
  least five frames. These are negative **controls**, not proven
  false positives (unannotated signals may remain).
- A separate right-flank artificial-truncation experiment returned
  extension for **114** eligible calls, mean missing-portion recovery
  **93.7%**, median error **1.36 kHz**.

Short (<1.2 ms), crossing, low-SNR and CF calls still need
dedicated evaluation. The added STFT has a runtime cost.

### Reproducible A/B

```python
from bat_analysis.modelling import process_full_spectrum

ridge = process_full_spectrum(
    y, sr, time_mid=candidate["time_mid"],
    duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    tracking_method="reassignment_flattop",
    reassignment_peak_anchored_window=True,
    reassignment_complete_ends=True,
)
```

To compare against the old experimental mode, pass
`reassignment_complete_ends=False` and
`reassignment_peak_anchored_window=False`. Prefer providing the
detector's original `chirp_start_s`/`chirp_end_s` when known.

```bash
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs --no-completion
python -m benchmark.reassignment_detector_seeds Nycto-Thomas.json /path/to/wavs
python -m benchmark.reassignment_detector_seeds Nycto-Thomas.json /path/to/wavs --no-completion
```

The documented evaluations above used an isolated implementation of
the added completion algorithm with the WAVs. The complete repository
test suite and the exact V4-to-tracker combination are **not yet
validated in CI**, and the source screenshot had no displayed time or
frequency coordinates with which to identify a specific WAV event.

## November 5 — overlapping chirps: group or stitch

Two simultaneously visible spectrogram ridges do **not** necessarily
represent fragments of the same call. In particular, two parallel descending
FM tracks can overlap in time yet be separated by many kHz. Joining their
frequency values into one `f(t)` curve would be incorrect: both frequencies
may genuinely coexist at the same instant.

The experimental API therefore distinguishes:

- **Overlap grouping**: one `OverlappingChirps` object contains several
  distinct ridges, with each source identity preserved. This is the
  appropriate representation for parallel calls like the supplied example.
- **Coincident ridge stitching**: two fragments can be represented as a
  single ridge only if their shared duration is at least 0.20 ms, their
  median and 90th percentile frequency disagreements are at most
  2.2 and 4.0 kHz, and their estimated slopes disagree by at most
  7 kHz/ms. Stitching is conservative and optional.
- **Possible harmonics**: concurrent tracks with a median frequency
  ratio within 2.0 +/- 0.10 are marked as *possible* fundamental/H2
  relationships. These are not automatically confirmed.

```python
from bat_analysis.chirp_union import group_overlapping_chirps

# Input: list of separate Nx3 arrays:
# [absolute_time_s, frequency_hz, normalized_amplitude].
groups = group_overlapping_chirps(
    [curve_1, curve_2], source_ids=["c1", "c2"], stitch=True
)
for group in groups:
    print(group.kind, group.source_ids)
    for separate_curve in group.ridges:
        print(separate_curve.shape)
    print(group.possible_harmonics)
```

The same behavior can be invoked from Adaptive V4 candidates without
manual tracks:

```python
from bat_analysis.reassignment_tracking import track_and_group_candidates

report = track_and_group_candidates(
    y, sr, candidates=None,  # detect with Adaptive V4
    tracking_kwargs={"complete_ends": True},
    grouping_kwargs={"stitch": True, "min_overlap_ms": 0.20},
)
for group in report["groups"]:
    # Keep two or more independent curves when they merely overlap in time.
    print(group.kind, group.stitched_sources)
print(report["untracked_candidate_indices"])
```

The `kind` is `"single"` (one original track),
`"stitched"` (coincident fragments joined), or `"overlapping"`
(multiple independent ridges remain inside the group). In a group
containing both stitched fragments and a distinct parallel call,
`stitched_sources` still maps each final ridge to its original inputs.

**Important**: an overlap group does not establish that two calls came
from one bat. Do not count groups as passages, and do not count an
associated harmonic as an independent call. For the desktop application,
display the group's ridges as separate polylines, rather than
averaging two different simultaneous frequencies together.

### Annotation cross-check

Among the four manually annotated Nycto WAVs, 14 pairs overlap by at
least 0.20 ms; all 14 pairs occur in `20261005_191947.wav`.
Their shared frequencies are consistent with a roughly 2:1 relation,
and **none** meets the conservative coincident-ridge stitch criteria
when evaluated against PCHIP interpolated manual points.

The exact tracking-to-grouping integration has not yet been measured
end-to-end; these checks validate overlap semantics rather than
precision, recall, or full detector performance.

## Reassignment-v2 — preventing late noise tracking

**This section applies to the `Reassignment-v2` branch only.**
`Reassignment` and `main` have not been changed.

### Root causes

In the desktop application's `ridge_worker.py`, the detection duration
was passed as milliseconds into `process_full_spectrum()`, which expects
seconds. The application also passed the bbox midpoint while the
experimental model expects the energy-peak time and can use explicit
detection start/end bounds. The companion
`Bat-Acoustic-Analyser/Reassignment-v2` branch fixes both and reconstructs
the WAV timestamp using the same 15 ms chunk offset as the model.

The prior `extend_ridge_edges` kept up to two below-threshold SNR frames
**in its actual returned curve**, allowing a weak ending to bridge into a
separate noise feature. The new implementation skips those frames (without
returning synthetic values) and only bridges them internally when coherent
above-threshold energy resumes promptly.

A second conservative guard, `trim_weak_right_tail`, examines the **entire
tracked ridge** after completion and before the endpoint correction. It
trims a trailing suffix when the following conditions both fail for at least
four consecutive points:

- Flattop STFT SNR at the ridge frequency: `>= 10 dB`.
- Power relative to the strongest point **on that same ridge**:
  `>= -22 dB`.

The guard preserves the first eight points and, by default, trims no more
than 45% of a curve in one operation. It does not alter timestamps or
frequencies of retained points. Small/short chirps are treated
conservatively. An additional late-gap rule now stops tracking at the
beginning of a run of at least four consecutive points with power below
`-22 dB` relative to the trajectory peak, even if stronger energy resumes
afterwards on a potentially unrelated ridge. It only applies after 60% of
the observed trajectory, after its power peak, and within the same 45%
maximum trimming budget; earlier or shorter gaps are preserved.

The final endpoint-frequency refinement is now gated by the actual STFT
energy: it only moves a point to a lower frequency when the proposed
location retains at least 50% of the spectral power of the original
location. The preceding confidence filter does not move retained points.
Diagnostics include `tail_confidence.trim_reason`,
`tail_confidence.late_gap_start_index`,
`tail_confidence.trimmed_right_points` and
`endpoint_refinement.rejected_corrections`.

### Experimental parameters

The `Reassignment-v2` defaults are:

```python
curve = process_full_spectrum(
    y, sr, time_mid=peak_time_seconds,
    duration=duration_seconds,
    seed_freq_hz=peak_frequency_hz,
    chirp_start_s=detector_start_seconds,
    chirp_end_s=detector_end_seconds,
    tracking_method="reassignment_flattop",
    reassignment_complete_ends=True,
    reassignment_extension_min_snr_db=17.0,
    reassignment_trim_weak_tail=True,
    reassignment_tail_min_snr_db=10.0,
    reassignment_tail_min_relative_db=-22.0,
    diagnostics={},
)
```

To A/B test the noise-tail filter without changing the rest of the
tracking pipeline, use `reassignment_trim_weak_tail=False`. The
extension itself can be disabled separately using
`reassignment_complete_ends=False`.

### Reproduction on the annotated event

In `20261005_191936.wav`, manually annotated chirp #4 runs from about
795.83 to 797.19 ms. The raw Flattop STFT around its annotated FM
trajectory has strong energy at 797.15 ms (~23 dB local SNR),
but the extrapolated direction drops to ~1.5 dB at 797.60 ms.
A more distant low-frequency feature can have more SNR again (~11.8 dB
at 798.00 ms), which explains why a permissive gap mechanism could
follow it. These numbers come from a local STFT diagnostic on the
real WAV, not a complete CI-validated tracker benchmark.

On this WAV, the updated tracker keeps 52 successful trajectories among
54 candidates, identical to the previous version. It shortens three
late-rejoining paths (#4, #43 and #44). Chirp #4 now ends at 797.582 ms
instead of 798.332 ms (30 rather than 42 points). The endpoint spectral
gate also prevents unsupported late downward-frequency corrections.

The filtering adds a safeguard against those mistakes. Its true recall
effect on faint chirps and its impact on more varied field recordings
still need verification against manual annotation. No claim is made
that every visible dotted segment from the screenshot is the direct
output of this library; the desktop's curve drawing/simplification
is a separate stage.

## November 6 — performance optimization of Reassignment-v2

The experimental tracker keeps the original FFT sizes (192/512), hop sizes,
Flattop window, SNR thresholds, quadratic slope penalties and all
confidence checks. The following computational kernels were accelerated
without intentionally changing any signal-processing decisions:

- **Reassignment STFT:** use NumPy stride views instead of assembling
  overlapping frames with Python list comprehensions and `np.stack`.
  Three separate FFTs, window definitions, output arrays, and chunking
  are retained.
- **Power redistribution:** replace repeated `np.add.at` updates with a
  flat-indexed weighted `np.bincount` histogram. This performs the same
  weighted accumulation on the same grid.
- **Ridge dynamic programming:** precompute valid predecessor-bin indices
  and velocity-change penalties. Evaluate all candidate-velocity
  transitions in one NumPy operation per time frame. The original
  score equation, argmax tie handling, slope continuity, right/left
  conditions and traceback are retained.
- **Waveform memory:** only the required local audio interval is checked
  and converted to `float64`. The old code copied and scanned
  the entire multi-second WAV for *every* chirp, including a second
  full copy inside the ridge-completion STFT.
- **Tail-confidence sampling:** gather the three nearby STFT frequency
  bins for every tracked point in one vectorized operation instead
  of a Python loop.

### Reproducible microbenchmarks

Local NumPy/SciPy timings on a 384 kHz WAV excerpt
(`20261005_191936.wav`, approximately 25 ms, Flattop FFT 512,
hop 24, 379 frames, 174 frequency bins) were:

| Kernel | Before | After | Speedup |
|---|---:|---:|---:|
| Three windowed FFTs | 2.12 ms | 1.83 ms | 1.2x |
| Reassignment energy scatter | 0.535 ms | 0.062 ms | 8.6x |
| Synthetic 174x220x5-state dynamic programming | 25.88 ms | 8.18 ms | 3.2x |
| 160-point confidence sampling | 0.785 ms | 0.034 ms | 23x |

FFT and weighted-scatter arrays were numerically identical in
local head-to-head checks. The dynamic-programming transition scores
and predecessor trace arrays were identical in the synthetic DP
comparison. Microbenchmarks do **not** measure complete-file speed,
end-to-end detector performance, or Windows portable-build latency.

For an approximately 5-second 384 kHz WAV loaded as float32, the full
float64 cast measured roughly 0.66 ms per operation; casting one local
25 ms interval took 0.0022 ms. The two paths evaluate identical audio
samples within the selected interval.

### End-to-end speed benchmark

Run the **same benchmark on both checkouts**, on the same machine and WAV
files. Ensure the analysis environment includes scipy, numpy, soundfile,
and other bat-analysis requirements.

```bash
python -m benchmark.reassignment_speed /path/to/wavs \
  --repeat 3 --max-candidates 20 --output speed_v2.json
```

The benchmark reports detector time separately from ridge-modelling time,
per-file candidate and returned-track counts, median/90th-percentile
runtime and a SHA-256 fingerprint of the generated tracks. This helps
detect discrepancies when comparing different branches, but a hash
difference can also reflect harmless floating-point rounding; inspect
frequencies/times before drawing conclusions about accuracy.

The new numerical regression tests are in
`tests/test_reassignment_fastpaths.py`: they compare the optimized
Flattop reassignment to the old `np.add.at` reference and exercise
deterministic ridge reconstruction.

**Status:** Local kernel checks have been run. Full repository CI,
full-WAV before/after timing and annotation-quality comparison of the
complete pipeline remain to be verified; no overall speedup figure
is claimed yet.

## Lightweight Reassignment-v2 entrypoint for the desktop CPU workers

`bat_analysis.fast_modelling.process_reassigned_fast` is equivalent to
`bat_analysis.modelling.process_full_spectrum(...,
tracking_method="reassignment_flattop")` for Reassignment-v2 default
parameters. It preserves the original 15-ms chunk-relative time convention,
modelled ridge coordinates, detector bbox, and seed handling.

Unlike `modelling.py`, importing the fast module does not import
`librosa` or the legacy Gaussian tracking code. This is particularly
important for multiprocessing on Windows, where every spawned Python
interpreter must import its modelling function separately.

The desktop application `Bat-Acoustic-Analyser/Reassignment-v2`
detects `fast_modelling.py` in the selected local checkout and
uses it automatically; the public modelling function remains unchanged
for backwards compatibility. Tests in `tests/test_fast_modelling.py`
compare the lightweight wrapper's parameters, outputs and diagnostics
against the full public entrypoint.

The ridge tracker also supplies optional
`diagnostics["timings_ms"]` when passed a diagnostics dictionary.
The breakdown includes `reassignment_fft`, `evidence_and_setup`,
`global_dp`, `ridge_sampling`, `completion`,
`confidence`, and `endpoint`. Disabled by default when no
diagnostics dictionary is passed.

The desktop worker's new log reports timings for every chirp and a
summary of the slowest five; the full application benchmark is
`benchmark_ridge_parallel.py`. Speed figures measured only on
isolated kernels or manually interpolated trajectories are **not**
valid substitutes for full-WAV timing in the Windows portable build.
