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

A dual-threshold / multiband detector was tested. The best balanced configuration in this small dataset was:

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

Detection result:

| metric | legacy | adaptive_v2 |
|---|---:|---:|
| TP | 113 | 145 |
| FP | 28 | 25 |
| FN | 45 | 13 |
| precision | 0.8014 | 0.8529 |
| recall | 0.7152 | 0.9177 |
| F1 | 0.7559 | 0.8841 |

A slightly more recall-oriented low-frequency height of 1500 Hz produced TP=146, FP=27, FN=12, recall=0.9241, F1=0.8822. The 2000 Hz setting was retained as the better balanced operating point.

## Frequency-seeded modelling

With the improved detector, unseeded modelling can lock simultaneous frequency-separated candidates onto the same strongest spectral band. The modeller was therefore tested with the detector peak frequency as an initialization seed. The initial maximum is searched within +/-8 kHz of the seed, while the rest of the legacy ridge pipeline is unchanged.

For the 145 matched TP of the balanced detector configuration:

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

These values are development-set results, not an unbiased estimate of generalization. The detector parameters were selected on only 8 WAVs / 158 chirps and there are currently no validated `no_chirp` WAVs. Keep `adaptive_v2` opt-in until more annotations are available and a hold-out validation set can be reserved.
