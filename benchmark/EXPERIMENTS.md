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
