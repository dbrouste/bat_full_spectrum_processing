from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return y, int(sr)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten(name, child, out)
    else:
        out[prefix] = value


def diagnose_modelling_failures(
    result,
    annotation_json: str | Path,
    *,
    wav_root: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-run matched modelling failures with structured processing diagnostics.

    Parameters
    ----------
    result:
        ``BenchmarkResult`` returned by ``benchmark.runner.run_benchmark``.
    annotation_json:
        Path to ``bat_chirp_annotations.json``.
    wav_root:
        Optional override for the WAV root. By default use ``root`` stored in
        the annotation JSON, or the JSON parent directory.

    Returns
    -------
    pandas.DataFrame
        One row per chirp whose detector match succeeded but whose production
        modelling returned ``None``. Nested left/right diagnostics are flattened
        into columns such as ``left.reason`` and ``right.reason``.
    """
    try:
        from bat_analysis import processing_diagnostics as procdiag
    except ImportError as exc:
        raise ImportError(
            "bat_analysis.processing_diagnostics is not available. Merge/pull "
            "the main branch containing the processing diagnostics first."
        ) from exc

    annotation_json = Path(annotation_json).expanduser().resolve()
    with annotation_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    root = Path(
        wav_root or annotations.get("root", annotation_json.parent)
    ).expanduser().resolve()

    chirps = result.chirps.copy()
    if chirps.empty:
        return pd.DataFrame()

    failed = chirps[
        (chirps["detected"] == True)
        & (chirps["model_success"] == False)
    ].copy()
    if failed.empty:
        return pd.DataFrame()

    detections = result.detections.copy()
    rows: list[dict[str, Any]] = []

    total = len(failed)
    for n, (_, chirp) in enumerate(failed.iterrows(), start=1):
        rel = str(chirp["relative_path"])
        det_index = int(chirp["detected_index"])

        det_match = detections[
            (detections["relative_path"] == rel)
            & (detections["detected_index"] == det_index)
        ]
        if det_match.empty:
            rows.append({
                "relative_path": rel,
                "chirp_id": chirp.get("chirp_id"),
                "detected_index": det_index,
                "diagnostic_failure": "matching detection row not found",
            })
            continue

        det = det_match.iloc[0]
        wav_path = root / rel
        if verbose:
            print(
                f"[{n}/{total}] {rel} — chirp {chirp.get('chirp_id')} "
                f"@ {float(det['time_mid_ms']):.3f} ms"
            )

        base_row: dict[str, Any] = {
            "relative_path": rel,
            "chirp_id": chirp.get("chirp_id"),
            "detected_index": det_index,
            "manual_start_ms": chirp.get("manual_start_ms"),
            "manual_end_ms": chirp.get("manual_end_ms"),
            "manual_duration_ms": chirp.get("manual_duration_ms"),
            "detection_center_error_ms": chirp.get("detection_center_error_ms"),
            "detection_iou": chirp.get("detection_iou"),
            "candidate_time_mid_ms": float(det["time_mid_ms"]),
            "candidate_duration_ms": float(det["duration_ms"]),
            "peak_freq_khz": det.get("peak_freq_khz"),
            "peak_db": det.get("peak_db"),
        }

        try:
            y, sr = _read_mono(wav_path)
            y_filtered = procdiag.high_pass_filter(y, sr)
            duration_s = float(np.clip(float(det["duration_ms"]) / 1000.0, 0.005, 0.08))
            curve, diagnostic = procdiag.process_full_spectrum_diagnostic(
                y_filtered,
                sr,
                time_mid=float(det["time_mid_ms"]) / 1000.0,
                duration=duration_s,
            )
            flat: dict[str, Any] = {}
            _flatten("", diagnostic, flat)
            base_row.update(flat)
            base_row["diagnostic_curve_returned"] = curve is not None
        except Exception as exc:
            base_row["diagnostic_failure"] = f"{type(exc).__name__}: {exc}"

        rows.append(base_row)

    return pd.DataFrame(rows)


def summarise_failure_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Count structured failure stage/reason pairs from deep diagnostics."""
    if df.empty:
        return pd.DataFrame(columns=["failure_stage", "failure_reason", "count", "percent"])

    stage_col = "failure_stage"
    reason_col = "failure_reason"
    if stage_col not in df.columns or reason_col not in df.columns:
        return pd.DataFrame(columns=["failure_stage", "failure_reason", "count", "percent"])

    out = (
        df.groupby([stage_col, reason_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    out["percent"] = 100.0 * out["count"] / len(df)
    return out


def print_deep_diagnostics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Print compact summaries and return useful diagnostic tables."""
    summary = summarise_failure_reasons(df)

    print("=== Deep modelling failure reasons ===")
    if summary.empty:
        print("No structured failures found.")
    else:
        print(summary.to_string(index=False))

    side_tables: dict[str, pd.DataFrame] = {}
    for side in ("left", "right"):
        col = f"{side}.reason"
        if col in df.columns:
            table = (
                df[col]
                .fillna("<not reached>")
                .value_counts(dropna=False)
                .rename_axis("reason")
                .reset_index(name="count")
            )
            table["percent"] = 100.0 * table["count"] / len(df)
            side_tables[side] = table
            print(f"\n=== {side.capitalize()} side stop reasons ===")
            print(table.to_string(index=False))

    numeric_cols = [
        "initial_max_amplitude",
        "left.initial_gaussian_variation_percent",
        "left.initial_distance_ms",
        "left.initial_freq_correction_hz",
        "left.amplitude_ratio",
        "right.initial_gaussian_variation_percent",
        "right.initial_distance_ms",
        "right.initial_freq_correction_hz",
        "right.amplitude_ratio",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    numeric = pd.DataFrame()
    if available:
        numeric = df[available].apply(pd.to_numeric, errors="coerce").describe().T
        print("\n=== Diagnostic numeric distributions ===")
        print(numeric.to_string())

    return {
        "failure_reasons": summary,
        "left_reasons": side_tables.get("left", pd.DataFrame()),
        "right_reasons": side_tables.get("right", pd.DataFrame()),
        "numeric": numeric,
    }
