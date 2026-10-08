# Dual-ridge initialization experiment

This branch is based on `main`; the baseline behaviour remains `ridge_mode="default"`.

## Modes

- `default`: retain the existing selective broad-FM fallback.
- `peak`: use only the detector peak-frequency seed.
- `coarse`: force the coarse frequency/slope seed (requires a finite coarse frequency).

`compare_ridge_hypotheses(...)` executes `peak` and `coarse` independently, returning a dict with `curve`, `diagnostics` and `error` per hypothesis. A missing coarse seed is represented as an error for that hypothesis; it does not abort the peak run. No automatic winner is chosen.

### Minimal example

```python
from bat_analysis.modelling import compare_ridge_hypotheses

result = compare_ridge_hypotheses(
    y_filtered, sr, time_mid, duration,
    seed_freq_hz=candidate["peak_freq_hz"],
    coarse_seed_freq_hz=candidate.get("coarse_seed_freq_hz"),
    coarse_seed_slope_hz_per_ms=candidate.get("coarse_seed_slope_hz_per_ms"),
)
peak_curve = result["peak"]["curve"]
coarse_curve = result["coarse"]["curve"]
```

Curves share the legacy output coordinates (relative chunk time in seconds, frequency in Hz). Before comparing to manual annotations, convert to absolute time using the same chunk-offset logic as the current benchmark.

## Validation still required

Run `pytest -q tests/test_dual_ridge_hypotheses.py` in an environment with project dependencies and pytest installed. This command has **not yet been executed**.

The 61 WAV / 149 annotated chirp reference set is not present in this branch. The benchmark runner and ground-truth dataset must be made available from the annotation branch/local working copy before quantitative comparison is possible. Preserve the frequency-aware matching and compare identical 116 TP detections, including the 7 >2 kHz error outliers and the remaining 109 chirps. Track median error, coverage, regression count, and processing time.

Avoid developing a quality score or promoting dual hypothesis mode until those measurements are available.

## Full benchmark command

From the repository root with all 61 original WAV in their preserved subdirectories:

```python
from benchmark.runner import run_benchmark

result = run_benchmark(
    "bat_chirp_annotations.json",
    "bat_analysis/modelling.py",
    wav_root="wav",
    detector_kwargs={"slope_filter_mode": "adaptive_v4"},
    compare_hypotheses=True,
)
result.save_csv("results/dual_ridge")
```

Inspect `results/dual_ridge/benchmark_chirps.csv` for
`median_abs_error_khz`, `coverage`, `hyp_peak_median_abs_error_khz`,
`hyp_peak_coverage`, `hyp_coarse_median_abs_error_khz`,
`hyp_coarse_coverage`, and both `hyp_*_model_success` flags.

The optional comparison does **not** influence baseline matching or the
normal output curve. It compares alternative seeds on the same matched
detector candidate, using the raw `coarse_seed_freq_hz` when available.
A missing coarse frequency yields a failed coarse hypothesis, not a new
prediction. Compare paired results on exactly the same 116 TP matches.

**Caution:** The 61-file benchmark has not been executed in this
environment. Initial independent STFT diagnostics on the four problematic
WAV are exploratory and are not the dual-ridge benchmark results.
