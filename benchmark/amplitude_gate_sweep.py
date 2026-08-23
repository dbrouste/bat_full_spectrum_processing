from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .curve_metrics import compare_curves


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return y, int(sr)


def _absolute_model_curve(curve: Any, candidate_time_s: float, file_duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("Invalid model curve")
    chunk_start_s = max(0.0, candidate_time_s - 0.0075)
    chunk_start_s = min(chunk_start_s, file_duration_s)
    t_ms = (chunk_start_s + arr[:, 0]) * 1000.0
    f_khz = arr[:, 1] / 1000.0
    return t_ms, f_khz


def _manual_curve(annotations: dict[str, Any], rel: str, chirp_id: Any) -> tuple[np.ndarray, np.ndarray]:
    record = annotations["files"][rel]
    for chirp in record.get("chirps", []) or []:
        if chirp.get("chirp_id") == chirp_id:
            pts = sorted(chirp.get("points", []) or [], key=lambda p: float(p["t_ms"]))
            if len(pts) < 2:
                break
            return (
                np.asarray([float(p["t_ms"]) for p in pts], dtype=float),
                np.asarray([float(p["f_khz"]) for p in pts], dtype=float),
            )
    raise KeyError(f"Manual chirp not found: {rel} chirp_id={chirp_id}")


def _model_after_gate(y_filtered, sr: int, time_mid_s: float, duration_s: float):
    """Run the current production modelling pipeline while bypassing only max_value < 0.7.

    Returns (curve_or_none, diagnostic_dict). All process-side and extension
    thresholds remain identical to the current processing code.
    """
    from bat_analysis import bfsp_clean_patched as base
    from bat_analysis import processing_diagnostics as procdiag

    diag: dict[str, Any] = {"success": False, "failure_stage": None, "failure_reason": None}
    y_chunk = base.Extract_chunk_of_audio(y_filtered, sr, time_mid_s)
    curve_all = base.initial_call_trend(y_chunk, sr, duration_s)
    if curve_all is None or len(curve_all) == 0:
        diag.update(failure_stage="initial_call_trend", failure_reason="initial_curve_empty")
        return None, diag

    max_value = max(entry[2] for entry in curve_all)
    diag["initial_max_amplitude"] = float(max_value)

    S = np.abs(librosa.stft(y_chunk, n_fft=1024, hop_length=100, window="flattop"))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=100)

    curve_all, left_diag = procdiag.process_side_diagnostic(curve_all, S, freqs, times, sr, 0, max_value)
    diag["left_reason"] = left_diag.get("reason")
    if curve_all is None:
        diag.update(failure_stage="process_side_left", failure_reason=left_diag.get("reason"))
        return None, diag

    curve_all, right_diag = procdiag.process_side_diagnostic(curve_all, S, freqs, times, sr, 1, max_value)
    diag["right_reason"] = right_diag.get("reason")
    if curve_all is None:
        diag.update(failure_stage="process_side_right", failure_reason=right_diag.get("reason"))
        return None, diag

    max_curve_segmented = 0
    curve_segmented = base.extend_trend_left(y_chunk, sr, curve_all, S, freqs, times, max_value)
    if curve_segmented is not None:
        max_curve_segmented = max(entry[2] for entry in curve_segmented)
        while (
            base.calculate_angle_between_trends(curve_all, curve_segmented) < 70
            and base.get_number_of_points(curve_segmented) > 11
            and max_curve_segmented > max_value / 20
        ):
            curve_all = base.concatenate_trends(curve_segmented, curve_all)
            curve_segmented = base.extend_trend_left(y_chunk, sr, curve_all, S, freqs, times, max_value)
            if curve_segmented is None:
                break
            max_curve_segmented = max(entry[2] for entry in curve_segmented)

    curve_segmented = base.extend_trend_right(y_chunk, sr, curve_all, S, freqs, times, max_value)
    if curve_segmented is not None:
        max_curve_segmented = max(entry[2] for entry in curve_segmented)
        while (
            base.calculate_angle_between_trends(curve_all, curve_segmented) < 50
            and base.get_number_of_points(curve_segmented) > 11
            and base.validate_slope(curve_segmented, nb_points=25, LeftRight=0)
            and base.get_max_frequency(curve_segmented) < (base.get_min_frequency(curve_all) + 500)
            and max_curve_segmented > max_value / 10
        ):
            curve_all = base.concatenate_trends(curve_all, curve_segmented)
            curve_segmented = base.extend_trend_right(y_chunk, sr, curve_all, S, freqs, times, max_value)
            if curve_segmented is None:
                break
            max_curve_segmented = max(entry[2] for entry in curve_segmented)

    try:
        curve_all = base.interpolate_trend_results(curve_all)
        curve_all = base.fit_spline_with_smoothness(curve_all)
        curve_all = base.sample_curve_equally(curve_all)
    except Exception as exc:
        diag.update(failure_stage="final_curve_processing", failure_reason=f"{type(exc).__name__}: {exc}")
        return None, diag

    diag.update(success=True, failure_stage="", failure_reason="")
    return curve_all, diag


def run_amplitude_gate_sweep(
    result,
    annotation_json: str | Path,
    *,
    wav_root: str | Path | None = None,
    thresholds: Iterable[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the 0.7 initial-amplitude gate on matched modelling failures.

    The expensive no-gate model is run once per previously failed TP. Then each
    candidate threshold is evaluated from the same result, so the sweep is fast.

    Returns
    -------
    summary_df, detail_df
        ``summary_df`` has one row per threshold. ``detail_df`` has one row per
        previously failed matched chirp, including no-gate model quality.
    """
    from bat_analysis import bfsp_clean_patched as base

    annotation_json = Path(annotation_json).expanduser().resolve()
    with annotation_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    root = Path(wav_root or annotations.get("root", annotation_json.parent)).expanduser().resolve()

    failed = result.chirps[(result.chirps["detected"] == True) & (result.chirps["model_success"] == False)].copy()
    detections = result.detections.copy()
    rows: list[dict[str, Any]] = []

    for n, (_, chirp) in enumerate(failed.iterrows(), start=1):
        rel = str(chirp["relative_path"])
        det_index = int(chirp["detected_index"])
        det_match = detections[(detections["relative_path"] == rel) & (detections["detected_index"] == det_index)]
        if det_match.empty:
            continue
        det = det_match.iloc[0]
        if verbose:
            print(f"[{n}/{len(failed)}] {rel} — chirp {chirp.get('chirp_id')}")

        row: dict[str, Any] = {
            "relative_path": rel,
            "chirp_id": chirp.get("chirp_id"),
            "detected_index": det_index,
            "peak_db": det.get("peak_db"),
            "manual_duration_ms": chirp.get("manual_duration_ms"),
        }
        try:
            y, sr = _read_mono(root / rel)
            y_filtered = base.high_pass_filter(y, sr)
            time_mid_s = float(det["time_mid_ms"]) / 1000.0
            duration_s = float(np.clip(float(det["duration_ms"]) / 1000.0, 0.005, 0.08))
            curve, diag = _model_after_gate(y_filtered, sr, time_mid_s, duration_s)
            row["initial_max_amplitude"] = diag.get("initial_max_amplitude", np.nan)
            row["no_gate_model_success"] = curve is not None
            row["failure_stage"] = diag.get("failure_stage")
            row["failure_reason"] = diag.get("failure_reason")
            row["left_reason"] = diag.get("left_reason")
            row["right_reason"] = diag.get("right_reason")

            if curve is not None:
                ref_t, ref_f = _manual_curve(annotations, rel, chirp.get("chirp_id"))
                pred_t, pred_f = _absolute_model_curve(curve, time_mid_s, len(y_filtered) / sr)
                metrics = compare_curves(ref_t, ref_f, pred_t, pred_f)
                row.update({
                    "median_abs_error_khz": metrics.median_abs_error_khz,
                    "rmse_khz": metrics.rmse_khz,
                    "p95_abs_error_khz": metrics.p95_abs_error_khz,
                    "coverage": metrics.coverage,
                })
        except Exception as exc:
            row["no_gate_model_success"] = False
            row["failure_stage"] = "exception"
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    detail = pd.DataFrame(rows)
    thresholds = sorted(float(x) for x in thresholds)
    baseline_success = int(result.summary.get("model_success", 0))
    manual_total = int(result.summary.get("manual_chirps", 0))
    tp_total = int(result.summary.get("true_positives", 0))

    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        if detail.empty:
            eligible = detail
        else:
            eligible = detail[
                (pd.to_numeric(detail["initial_max_amplitude"], errors="coerce") >= threshold)
                & (detail["no_gate_model_success"] == True)
            ]
        recovered = len(eligible)
        total_model_success = baseline_success + recovered
        summaries.append({
            "threshold": threshold,
            "recovered_from_previous_failures": int(recovered),
            "total_model_success": int(total_model_success),
            "model_success_rate_on_tp": float(total_model_success / tp_total) if tp_total else 0.0,
            "end_to_end_recall": float(total_model_success / manual_total) if manual_total else 0.0,
            "median_abs_error_khz_mean_recovered": float(pd.to_numeric(eligible.get("median_abs_error_khz"), errors="coerce").mean()) if recovered else np.nan,
            "rmse_khz_mean_recovered": float(pd.to_numeric(eligible.get("rmse_khz"), errors="coerce").mean()) if recovered else np.nan,
            "p95_khz_mean_recovered": float(pd.to_numeric(eligible.get("p95_abs_error_khz"), errors="coerce").mean()) if recovered else np.nan,
            "coverage_mean_recovered": float(pd.to_numeric(eligible.get("coverage"), errors="coerce").mean()) if recovered else np.nan,
        })

    return pd.DataFrame(summaries), detail


def print_amplitude_gate_sweep(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    print("=== Initial amplitude gate sweep ===")
    if summary.empty:
        print("No results.")
        return
    cols = [
        "threshold",
        "recovered_from_previous_failures",
        "total_model_success",
        "model_success_rate_on_tp",
        "end_to_end_recall",
        "median_abs_error_khz_mean_recovered",
        "rmse_khz_mean_recovered",
        "coverage_mean_recovered",
    ]
    print(summary[cols].to_string(index=False))

    if not detail.empty:
        print("\n=== No-gate outcomes for the previously failed matched chirps ===")
        success = int((detail["no_gate_model_success"] == True).sum())
        print(f"No-gate model succeeded for {success}/{len(detail)} previously failed TP.")
        if "failure_reason" in detail.columns:
            failed = detail[detail["no_gate_model_success"] != True]
            if not failed.empty:
                print("\nRemaining failure reasons after bypassing the gate:")
                print(failed["failure_reason"].fillna("<none>").value_counts().to_string())
