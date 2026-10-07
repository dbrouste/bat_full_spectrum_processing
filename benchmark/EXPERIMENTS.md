# Processing benchmark experiments

Ground truth used for this round: 8 validated WAV files, 158 manual chirps, no `no_chirp` negative WAVs yet.

## Baseline — legacy detector + no amplitude gate

| metric | value |
|---|---:|
| detections | 141 |
| TP | 113 |
| FP | 28 |
| FN | 45 |
| precision | 0.8014 |
| recall | 0.7152 |
| F1 | 0.7559 |
| model success / TP | 113 / 113 |
| end-to-end recall | 0.7152 |
| mean chirp median abs curve error | 2.275 kHz |
| mean curve RMSE | 2.787 kHz |
| mean curve P95 abs error | 4.134 kHz |
| mean curve coverage | 0.8486 |

## Detector experiments

The truth set contains several call families, including long shallow calls around 30–40 kHz and simultaneous frequency-separated bands/harmonics. Two hard assumptions in the historical detector therefore caused avoidable misses:

- general blob slope <= -2000 Hz/ms;
- temporal-only echo NMS, which suppresses a second candidate even when it is far away in frequency.

### adaptive_v2

A dual-threshold / multiband detector was tested. The balanced configuration is:

```python
{
    "slope_filter_mode": "adaptive_v2",
    "snr_threshold_db": 10.0,
    "lowfreq_snr_threshold_db": 9.0,
    "min_blob_size": 10,
    "min_blob_height_hz": 5000.0,
    "general_max_blob_slope_hz_per_ms": -500.0,
    "lowfreq_max_hz": 45000.0,
    "lowfreq_min_blob_height_hz": 2000.0,
    "lowfreq_max_blob_slope_hz_per_ms": 0.0,
    "lowfreq_min_width_ms": 1.0,
    "echo_suppression_window_ms": 16.0,
    "echo_suppression_freq_window_hz": 25000.0,
}
```

| metric | legacy | adaptive_v2 |
|---|---:|---:|
| TP | 113 | 145 |
| FP | 28 | 25 |
| FN | 45 | 13 |
| precision | 0.8014 | 0.8529 |
| recall | 0.7152 | 0.9177 |
| F1 | 0.7559 | 0.8841 |

A slightly more recall-oriented low-frequency height of 1500 Hz produced TP=146, FP=27, FN=12, recall=0.9241, F1=0.8822. The 2000 Hz setting was retained as the better balanced v2 operating point.

### adaptive_v3 — current experimental development mode

Further error analysis showed two smaller opportunities that can be added without globally relaxing the detector:

1. widening the time+frequency NMS from 16 to 20 ms removes one duplicate/echo FP without losing a TP;
2. a narrow high-frequency recovery branch at 9 dB recovers one additional true call whose connected component does not satisfy the 10 dB general-branch height rule;
3. a conservative 2-D bbox overlap deduplication removes one overlapping fragment while preserving simultaneous frequency-separated calls.

Current v3 defaults:

```python
{
    "slope_filter_mode": "adaptive_v3",
    "snr_threshold_db": 10.0,
    "lowfreq_snr_threshold_db": 9.0,
    "general_max_blob_slope_hz_per_ms": -500.0,
    "lowfreq_max_hz": 45000.0,
    "lowfreq_min_blob_height_hz": 2000.0,
    "lowfreq_min_width_ms": 1.0,
    "highfreq_snr_threshold_db": 9.0,
    "highfreq_min_hz": 50000.0,
    "highfreq_min_blob_height_hz": 3000.0,
    "highfreq_min_width_ms": 4.0,
    "highfreq_min_blob_size": 10,
    "highfreq_max_blob_slope_hz_per_ms": -1000.0,
    "echo_suppression_window_ms": 20.0,
    "echo_suppression_freq_window_hz": 25000.0,
    "bbox_dedup_time_iou": 0.20,
    "bbox_dedup_freq_iou": 0.40,
}
```

Detection-only development result:

| metric | adaptive_v2 | adaptive_v3 |
|---|---:|---:|
| TP | 145 | 146 |
| FP | 25 | 24 |
| FN | 13 | 12 |
| precision | 0.8529 | 0.8588 |
| recall | 0.9177 | 0.9241 |
| F1 | 0.8841 | 0.8902 |

The gain is deliberately small and conservative. v3 remains opt-in pending independent WAV/no_chirp validation.

### Remaining FN after adaptive_v3

The 12 remaining misses are concentrated rather than random:

- 9 are weak/shallow calls around roughly 31–43 kHz, mostly 9–15 ms long; several fragment into tiny 0.75–1.5 kHz-high components even at 7–9 dB SNR;
- 2 are moderate/high-frequency calls around 50–72 kHz whose SNR components remain fragmented or too small for the current blob rules;
- 1 is a broad 63–120 kHz event occurring at the same time as a very strong lower-frequency connected component, so simple connected-component logic does not isolate the annotated high-frequency ridge.

A low-frequency morphology/closing branch was tested. It recovered up to two extra TP but added enough FP that F1 did not improve (best tested result TP=147, FP=28, FN=11, F1=0.8829). It is therefore not retained.

A more aggressive low-frequency SNR branch can reach TP=150 / FN=8, but FP rises to 37 and F1 falls to 0.8696. This is also rejected for now. The next useful step is to validate v3 on newly annotated/no_chirp WAVs before further relaxing weak-call detection.

## Frequency-seeded modelling

With the improved detector, unseeded modelling can lock simultaneous frequency-separated candidates onto the same strongest spectral band. The modeller was therefore tested with the detector peak frequency as an initialization seed. The initial maximum is searched within +/-8 kHz of the seed, while the rest of the legacy ridge pipeline is unchanged.

For the 145 matched TP of the v2 detector configuration:

| metric | seeded result |
|---|---:|
| model success | 145 / 145 |
| end-to-end recall | 0.9177 |
| mean chirp median abs curve error | ~1.976 kHz |
| mean curve RMSE | ~2.469 kHz |
| mean curve P95 abs error | ~3.722 kHz |
| mean curve coverage | ~0.873 |

The largest improvement was on WAVs containing simultaneous bands/harmonics: frequency seeding prevents all candidates from initializing on the globally strongest component.

## Important limitation

These values are development-set results, not an unbiased estimate of generalization. The detector parameters were selected on only 8 WAVs / 158 chirps and there are currently no validated `no_chirp` WAVs. Keep `adaptive_v2` and `adaptive_v3` opt-in until more annotations are available and a hold-out validation set can be reserved.


## Independent validation dataset — 2026-10-07

A larger exported dataset was evaluated without retuning detector parameters first.

Dataset:
- 61 validated WAV files
- 21 annotated positive WAV files
- 40 no_chirp negative WAV files
- 149 manually annotated chirps

Historical temporal Hungarian matching was kept for detector comparability.

| metric | legacy | adaptive_v2 | adaptive_v3 |
|---|---:|---:|---:|
| TP | 72 | 123 | 123 |
| FP | 156 | 254 | 250 |
| FN | 77 | 26 | 26 |
| precision | 0.3158 | 0.3263 | 0.3298 |
| recall | 0.4832 | 0.8255 | 0.8255 |
| F1 | 0.3820 | 0.4677 | 0.4713 |
| FP on 40 no_chirp WAV | 120 | 203 | 202 |
| no_chirp files with >=1 FP | 25 | 29 | 29 |
| no_chirp file specificity | 0.375 | 0.275 | 0.275 |
| FP / no_chirp WAV | 3.000 | 5.075 | 5.050 |

Interpretation:
- The adaptive detectors generalize strongly for recall: 82.6% vs 48.3% legacy.
- The apparent precision gains seen on the original 8-WAV development set do not generalize.
- Most remaining error is now false-positive rejection, especially on true negative recordings.
- v3 is only marginally better than v2 on this independent set, so further threshold relaxation is not justified.
- Parameters must not be tuned directly against this validation set without reserving a new hold-out subset.

### False-positive structure (adaptive_v3)

Of the 250 FP:
- 202 occur in no_chirp WAVs and 48 in positive WAVs.
- no_chirp FP by detector branch: general=127, lowfreq=69, highfreq=6.
- Median TP blob width is ~4.30 ms; median no_chirp FP width is ~2.00 ms.
- Median TP peak frequency is ~46.5 kHz; median no_chirp FP peak frequency is ~34.5 kHz.
- Median TP blob size is ~61 pixels; median no_chirp FP size is ~28 pixels.

These are descriptive diagnostics, not yet candidate rejection thresholds.

### False-negative structure (adaptive_v3)

26 / 149 manual chirps are missed. Median FN duration is ~5.67 ms and median mean frequency is ~34.25 kHz. Misses are concentrated in a small number of recordings and include both long shallow low-frequency calls and some short high-frequency FM calls.

### Frequency-aware matching check

Optional time+frequency matching preserves the same total TP count (123) on this dataset but changes two ambiguous assignments in one WAV (20250615_185828.wav). This confirms that frequency-aware assignment is useful for curve-error evaluation without inflating detector recall.


## Grouped development / hold-out split and adaptive_v4

After the first independent validation pass, the 61-WAV dataset was frozen into
a deterministic grouped split by source folder/session. This prevents files from
the same recording session leaking between development and hold-out.

Split rule:
- group = relative parent folder of each WAV
- deterministic SHA-256 assignment using seed `bfsp-holdout-v1`
- target hold-out fraction = 30%

Result:
- development: 46 WAV, 15 positive, 31 no_chirp, 111 manual chirps
- hold-out: 15 WAV, 6 positive, 9 no_chirp, 38 manual chirps

The hold-out was not used to select candidate thresholds.

### Development diagnostics

On adaptive_v3, false positives were dominated by short blobs:
- median matched TP width: ~4.30 ms overall
- median no_chirp FP width: ~2.00 ms overall
- low-frequency TP median width on development: ~8.33 ms
- low-frequency FP median width on development: ~2.00 ms
- low-frequency TP median slope: ~-570 Hz/ms
- low-frequency FP median slope: ~-4.3 kHz/ms
- the v3 high-frequency recovery branch produced no additional TP on the new development split, only FP.

This motivated adaptive_v4:
- general branch minimum blob width: 1.5 ms
- low-frequency branch minimum width: 3.0 ms
- low-frequency slope constrained to [-1500, 0) Hz/ms
- 20 ms time+frequency NMS
- no high-frequency recovery branch

### adaptive_v4 result

| split | TP | FP | FN | precision | recall | F1 | no_chirp FP | no_chirp specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 88 | 92 | 23 | 0.4889 | 0.7928 | 0.6048 | 70 | 0.387 |
| hold-out | 28 | 23 | 10 | 0.5490 | 0.7368 | 0.6292 | 22 | 0.444 |
| all 61 WAV | 116 | 115 | 33 | 0.5022 | 0.7785 | 0.6105 | 92 | 0.400 |

Compared with adaptive_v3 on the full 61-WAV dataset:
- precision: 0.3298 -> 0.5022
- recall: 0.8255 -> 0.7785
- F1: 0.4713 -> 0.6105
- no_chirp FP: 202 -> 92

The improvement reproduces on the grouped hold-out, so it is unlikely to be
only an in-sample threshold effect. adaptive_v4 remains opt-in until the
modelling/ridge stage is revalidated end-to-end.


## Modelling runtime/convergence diagnostics — 2026-10-07

The larger dataset exposed a second bottleneck after detection: ridge modelling can
be pathologically slow on some calls, especially long shallow low-frequency
chirps.

Observed hotspots:
- `sample_line_from_max_amp_dynamic()` repeatedly constructs/interrogates a
  spline interpolator during tracking.
- `fit_gaussian()` uses unconstrained `curve_fit` and can spend a long time
  on difficult amplitude profiles.
- `process_side()` advances with a fixed nominal 40-us tracking step. Long,
  shallow calls can therefore require hundreds of iterations per side.
- segmented extension can repeat the same expensive machinery several times.

A concrete bug was fixed on main in commit
`ba45810e5ab36b7ccd365153f51a8aefe8d19dcb`:
the first negative-direction sampling step was not normalized by slope, unlike
the positive direction. Sampling is also stopped outside the spectrogram bounds,
and stalled side tracking now has tighter guards.

A more aggressive local experiment combined:
- shared spectrogram interpolator,
- bounded Gaussian fit,
- side iteration cap,
- larger tracking step.

This made successful calls fast (typically a few tenths of a second), but a hard
iteration cap of 120 reduced model completeness. On 116 adaptive_v4 matched TP,
75 completed and 41 returned no curve. Failures were strongly concentrated at
low frequency:
- candidate peak <35 kHz: 31 failures / 39 matched calls,
- 35-45 kHz: 3 failures / 12,
- 45-60 kHz: 7 failures / 57,
- >=60 kHz: 0 failures / 8.

For the 75 completed curves in that experiment:
- median frequency error median: ~0.58 kHz,
- RMSE median: ~0.98 kHz,
- P95 error median: ~1.81 kHz,
- coverage median: ~0.68.

The low-frequency failures are mainly long, shallow calls. Their median manual
duration is much longer than successful calls, and side-level diagnostics show
the convergence failure is usually in `process_side()` itself.

A targeted step-size experiment is promising: with the same 120-iteration
diagnostic cap, increasing the default tracking step from 40 to 60 (nominal
microseconds; historical variable name says ns) raised side-level convergence
from 75/116 to 107/116. The remaining 9 side failures were all below 35 kHz.

Conclusion:
- do not simply lower thresholds or raise the iteration limit;
- make tracking step adaptive to call duration/slope;
- keep Gaussian fitting bounded;
- reuse one interpolator per chirp;
- retain a hard runtime/iteration guard;
- validate each change against curve error and coverage, not only model success.


## Adaptive ridge tracking step — implementation

The fixed 40-us ridge extrapolation step has been replaced in the canonical
modelling API by a conservative adaptive rule:

- duration <= 4 ms: 40 us
- duration 4-8 ms: linear ramp 40 -> 60 us
- duration >= 8 ms: 60 us
- detected peak < 40 kHz and duration >= 5 ms: force at least 60 us

The rule is deliberately capped at 60 us because 60 us was the largest step
already validated in the side-level convergence experiment. It is applied to
both initial left/right tracking and segmented extensions. The historical
internal variable name `step_ns` in ridge extrapolation was corrected to
`step_us`; the actual arithmetic had always been microseconds.

Main commits:
- `0773c63044f3abac55ccddb1b54f1ac756f4cbcc`: expose ridge step
- `b58da98c32dc687314ad362d957c0f770c41bad0`: fix process_side signature
- `07e59c7817597b1d96e647c6db98aa04f9c91ecc`: adaptive step in canonical modeller

The benchmark runner now records `tracking_step_us` per matched chirp so
runtime, coverage and curve error can be stratified by the chosen step.


## Adaptive-step full modelling benchmark — 116 adaptive_v4 TP

The adaptive tracking rule was run end-to-end on all 116 frequency-aware
adaptive_v4 true-positive assignments.

Result after guarding failed segmented Gaussian extension points:
- model success: **116 / 116**
- median runtime: **0.642 s / chirp**
- median absolute frequency error: **0.423 kHz**
- median RMSE: **0.652 kHz**
- median P95 absolute error: **1.335 kHz**
- median temporal coverage: **0.870**
- median start error: **+0.230 ms**
- median end error: **+0.078 ms**

Frequency-band breakdown:

| detector peak | n | model success | median error | median RMSE | median coverage |
|---|---:|---:|---:|---:|---:|
| <35 kHz | 39 | 39/39 | 0.181 kHz | 0.266 kHz | 1.000 |
| 35-45 kHz | 12 | 12/12 | 0.438 kHz | 0.623 kHz | 0.808 |
| 45-60 kHz | 57 | 57/57 | 0.600 kHz | 1.107 kHz | 0.785 |
| >=60 kHz | 8 | 8/8 | 1.222 kHz | 1.985 kHz | 0.662 |

Compared on the same 75 chirps that already succeeded in the earlier fixed-step
experiment, curve accuracy is effectively unchanged (median error 0.581 -> 0.588
kHz; median RMSE 0.982 -> 1.050 kHz). The key gain is recovery of all **41**
previous modelling failures. Those recovered calls are mostly long,
low-frequency calls and are modelled very accurately:
- recovered-call median error: **0.185 kHz**
- median RMSE: **0.317 kHz**
- median P95 error: **0.606 kHz**
- median coverage: **1.000**

Two low-frequency calls initially raised
`TypeError: cannot unpack non-iterable NoneType object` in
`initial_call_trend_segmented()` when a Gaussian extension candidate had no
valid corrected point. Main commit
`f0b7c6e591a0650f11ee0008ed47c05192c1f979` now treats that extension candidate
as invalid rather than aborting the chirp. Both calls then model successfully.

Remaining quality outliers are no longer convergence failures:
- 8/116 calls have median frequency error >2 kHz
- 1/116 is >5 kHz
- 11/116 have temporal coverage <0.5

The largest concentration is in `20250615_185828.wav`, followed by
`rec_20260103_201256.wav` and `rec_20260103_202314.wav`. These should be
treated as ridge-assignment/extension-quality problems, not detector or
convergence problems.


## Remaining modelling quality outliers

With adaptive tracking, all 116 matched adaptive_v4 true positives now return a
curve. Residual problems are quality rather than convergence:
- 8 calls have median frequency error above 2 kHz
- 1 call is above 5 kHz
- 11 calls have coverage below 0.5
- 44 calls have coverage below 0.75

The largest outlier is 20250615_185828.wav, chirp 11. The manual annotation spans
a broad downward FM from about 111 kHz to 53 kHz, while the detector peak is
near 55 kHz. The current seed can therefore lock onto the strong low-frequency
tail instead of the full sweep.

Exploratory alternatives using detector bbox centre, upper-quartile frequency,
or detector-blob slope did not generalize across the worst eight calls, so none
was promoted to the canonical modeller.

The next candidate is a coarse time-frequency ridge derived from the detector
blob itself, used only to initialize orientation and frequency band before the
existing Gaussian ridge refinement.
