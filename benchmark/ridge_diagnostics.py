from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def classify_ridge_quality(row: pd.Series) -> str:
    """Classify one matched chirp for algorithmic triage."""
    if not bool(row.get("model_success", True)):
        return "model_failure"

    coverage = float(row.get("coverage", np.nan))
    error = float(row.get("median_abs_error_khz", np.nan))
    bias = float(row.get("bias_khz", np.nan))
    match_freq_error = float(row.get("detection_frequency_error_khz", np.nan))

    if np.isfinite(coverage) and coverage < 0.10:
        if np.isfinite(match_freq_error) and match_freq_error > 5.0:
            return "seed_or_localization"
        return "severe_truncation"
    if np.isfinite(coverage) and coverage < 0.50:
        return "partial_truncation"
    if np.isfinite(error) and error > 2.0:
        if np.isfinite(bias) and abs(bias) > 2.0:
            return "frequency_offset"
        return "shape_mismatch"
    if np.isfinite(coverage) and coverage < 0.75:
        return "partial_coverage"
    return "ok"


def add_ridge_quality_class(chirps_df: pd.DataFrame) -> pd.DataFrame:
    out = chirps_df.copy()
    out["ridge_quality_class"] = out.apply(classify_ridge_quality, axis=1)
    return out


def parse_stop_reason_counts(value: Any) -> Counter:
    if isinstance(value, dict):
        return Counter({str(k): int(v) for k, v in value.items()})
    if not isinstance(value, str) or not value.strip():
        return Counter()
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return Counter()
    return Counter({str(k): int(v) for k, v in data.items()})


def summarize_stop_reasons(chirps_df: pd.DataFrame) -> pd.DataFrame:
    df = add_ridge_quality_class(chirps_df)
    rows = []
    for quality, group in df.groupby("ridge_quality_class", dropna=False):
        counter = Counter()
        if "ridge_stop_reason_counts" in group:
            for value in group["ridge_stop_reason_counts"]:
                counter.update(parse_stop_reason_counts(value))
        n = max(len(group), 1)
        for reason, count in counter.most_common():
            rows.append({
                "ridge_quality_class": quality,
                "chirps": len(group),
                "stop_reason": reason,
                "event_count": int(count),
                "events_per_chirp": float(count / n),
            })
    return pd.DataFrame(rows)


def select_ridge_outliers(
    chirps_df: pd.DataFrame,
    *,
    error_threshold_khz: float = 2.0,
    coverage_threshold: float = 0.50,
) -> pd.DataFrame:
    df = add_ridge_quality_class(chirps_df)
    err = pd.to_numeric(df.get("median_abs_error_khz"), errors="coerce")
    cov = pd.to_numeric(df.get("coverage"), errors="coerce")
    mask = (err > error_threshold_khz) | (cov < coverage_threshold)
    columns = [
        c for c in [
            "relative_path", "chirp_id", "ridge_quality_class",
            "median_abs_error_khz", "rmse_khz", "p95_abs_error_khz",
            "coverage", "bias_khz", "start_error_ms", "end_error_ms",
            "detection_frequency_error_khz", "candidate_peak_freq_khz",
            "candidate_duration_ms", "reference_duration_ms",
            "tracking_step_us", "broad_fm_seed_used",
            "ridge_total_iterations", "ridge_max_side_iterations",
            "ridge_stop_reasons", "ridge_stop_reason_counts",
        ] if c in df.columns
    ]
    return df.loc[mask, columns].sort_values(
        ["median_abs_error_khz", "coverage"],
        ascending=[False, True],
    )
