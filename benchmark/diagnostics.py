from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def modelling_diagnostics(result: Any) -> dict[str, pd.DataFrame]:
    """Build diagnostic tables explaining where the current pipeline loses chirps.

    This does not modify the analysis target. It combines the per-chirp matching
    results with detector candidate properties, then compares successfully
    modelled true positives with true positives for which modelling failed.
    """
    chirps = result.chirps.copy()
    detections = result.detections.copy()

    if chirps.empty:
        return {
            "failure_stages": pd.DataFrame(),
            "model_groups": pd.DataFrame(),
            "failed_chirps": pd.DataFrame(),
        }

    failure_stages = (
        chirps.assign(
            failure_stage=chirps.get("failure_stage", pd.Series(index=chirps.index, dtype=object))
            .fillna("")
            .replace("", "success")
        )
        .groupby("failure_stage", dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    failure_stages["percent_of_manual"] = 100.0 * failure_stages["count"] / len(chirps)

    tp = chirps[chirps.get("detected", False) == True].copy()
    if tp.empty:
        return {
            "failure_stages": failure_stages,
            "model_groups": pd.DataFrame(),
            "failed_chirps": pd.DataFrame(),
        }

    if not detections.empty and "detected_index" in detections.columns:
        det_cols = [
            c for c in [
                "relative_path", "detected_index", "time_mid_ms", "t_start_ms",
                "t_end_ms", "duration_ms", "peak_freq_khz", "peak_db"
            ] if c in detections.columns
        ]
        tp = tp.merge(
            detections[det_cols],
            on=["relative_path", "detected_index"],
            how="left",
            suffixes=("", "_det"),
        )

    tp["model_group"] = np.where(tp["model_success"].fillna(False), "success", "failure")

    numeric_candidates = [
        "manual_duration_ms",
        "detection_iou",
        "detection_center_error_ms",
        "duration_ms",
        "peak_freq_khz",
        "peak_db",
        "model_time_s",
    ]
    rows: list[dict[str, Any]] = []
    for col in numeric_candidates:
        if col not in tp.columns:
            continue
        values = pd.to_numeric(tp[col], errors="coerce")
        for group in ("success", "failure"):
            mask = tp["model_group"] == group
            x = values[mask].dropna()
            if x.empty:
                continue
            rows.append({
                "metric": col,
                "group": group,
                "n": int(len(x)),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "p10": float(x.quantile(0.10)),
                "p90": float(x.quantile(0.90)),
            })
    model_groups = pd.DataFrame(rows)

    failed = tp[tp["model_group"] == "failure"].copy()
    preferred = [
        "relative_path", "chirp_id", "failure_stage", "error",
        "manual_start_ms", "manual_end_ms", "manual_duration_ms",
        "candidate_time_mid_ms", "detection_center_error_ms", "detection_iou",
        "duration_ms", "peak_freq_khz", "peak_db", "model_time_s",
    ]
    failed_chirps = failed[[c for c in preferred if c in failed.columns]].copy()
    if "detection_center_error_ms" in failed_chirps.columns:
        failed_chirps = failed_chirps.sort_values(
            "detection_center_error_ms", ascending=False, na_position="last"
        )

    return {
        "failure_stages": failure_stages,
        "model_groups": model_groups,
        "failed_chirps": failed_chirps,
    }


def print_modelling_diagnostics(result: Any) -> dict[str, pd.DataFrame]:
    """Print a compact diagnostic report and return its underlying tables."""
    diag = modelling_diagnostics(result)

    print("=== Pipeline failure stages ===")
    if diag["failure_stages"].empty:
        print("No chirp rows available.")
    else:
        print(diag["failure_stages"].to_string(index=False))

    print("\n=== Modelling success vs failure ===")
    groups = diag["model_groups"]
    if groups.empty:
        print("No matched detections available.")
    else:
        pivot = groups.pivot(index="metric", columns="group", values="median")
        print("Median values:")
        print(pivot.to_string())

    print("\n=== Failed matched chirps ===")
    print(f"{len(diag['failed_chirps'])} matched chirp(s) failed modelling.")
    return diag
