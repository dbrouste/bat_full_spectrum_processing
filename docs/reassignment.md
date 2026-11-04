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
The default remains `"legacy"`. The new mode replaces the **trajectory
tracking stage**, not the SNR blob detector. It follows reassigned Flattop
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

All these changes remain on the `Reassignment` branch. The main branch
and default legacy tracker are unmodified. GitHub CI execution is still
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

**Opt-in to experimental ridge tracking**, `tracking_method="reassignment_flattop"`.
The main branch and the default legacy tracker are unchanged.

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
