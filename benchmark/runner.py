from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import pandas as pd
import soundfile as sf

from .curve_metrics import compare_curves
from .matching import match_chirps


DEFAULT_DETECTOR_KWARGS = {
    "snr_threshold_db": 10.0,
    "percentile_q": 96.0,
    "fmin": 20000,
    "fmax": 150000,
    "n_fft": 512,
    "hop": 128,
    "min_blob_size": 10,
    "min_blob_height_hz": 5000.0,
    "max_blob_slope_hz_per_ms": -2000.0,
    "echo_suppression_window_ms": 10.0,
}

VALID_BENCHMARK_STATUSES = {"annotated", "no_chirp"}


@dataclass
class BenchmarkResult:
    summary: dict[str, Any]
    files: pd.DataFrame
    chirps: pd.DataFrame
    detections: pd.DataFrame

    def save_csv(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.files.to_csv(output_dir / "benchmark_files.csv", index=False)
        self.chirps.to_csv(output_dir / "benchmark_chirps.csv", index=False)
        self.detections.to_csv(output_dir / "benchmark_detections.csv", index=False)
        with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)


def load_analysis_module(path: str | Path) -> ModuleType:
    """Load the processing script to benchmark without installing/copying it."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    module_name = "bfsp_benchmark_target"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import analysis module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    required = ["high_pass_filter", "detect_candidates_snr_blobs", "process_full_spectrum"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Analysis module is missing: {', '.join(missing)}")
    return module


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return y, int(sr)


def _manual_chirps(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for chirp in record.get("chirps", []) or []:
        pts = chirp.get("points", []) or []
        if len(pts) < 2:
            continue
        pts = sorted(pts, key=lambda p: float(p["t_ms"]))
        t = np.asarray([float(p["t_ms"]) for p in pts], dtype=float)
        f = np.asarray([float(p["f_khz"]) for p in pts], dtype=float)
        out.append({
            "chirp_id": chirp.get("chirp_id"),
            "t_ms": t,
            "f_khz": f,
            "interval_s": (float(t[0] / 1000.0), float(t[-1] / 1000.0)),
            "center_s": float((t[0] + t[-1]) / 2000.0),
        })
    return out


def _candidate_interval(candidate: dict[str, Any]) -> tuple[float, float]:
    if "t_start" in candidate and "t_end" in candidate:
        a, b = float(candidate["t_start"]), float(candidate["t_end"])
        if b > a:
            return a, b
    mid = float(candidate["time_mid"])
    dur = max(float(candidate.get("duration", 0.0)), 1e-6)
    return mid - dur / 2.0, mid + dur / 2.0


def _absolute_model_curve(
    curve: Any,
    candidate_time_s: float,
    file_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("process_full_spectrum returned an invalid curve")

    chunk_start_s = max(0.0, candidate_time_s - 0.0075)
    chunk_start_s = min(chunk_start_s, file_duration_s)
    t_ms = (chunk_start_s + arr[:, 0]) * 1000.0
    f_khz = arr[:, 1] / 1000.0
    return t_ms, f_khz


def run_benchmark(
    annotation_json: str | Path,
    analysis_py: str | Path,
    *,
    wav_root: Optional[str | Path] = None,
    detector_kwargs: Optional[dict[str, Any]] = None,
    min_iou: float = 0.05,
    max_center_error_ms: float = 4.0,
    verbose: bool = True,
) -> BenchmarkResult:
    """Benchmark detection and curve modelling against validated annotations.

    Only records with status ``annotated`` or ``no_chirp`` are ground truth.
    If the processing module accepts ``seed_freq_hz`` in
    ``process_full_spectrum``, the matched detector peak frequency is passed to
    the modeller. Older processing modules remain compatible.
    """
    annotation_json = Path(annotation_json).expanduser().resolve()
    with annotation_json.open("r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    root = Path(
        wav_root or annotation_data.get("root", annotation_json.parent)
    ).expanduser().resolve()
    module = load_analysis_module(analysis_py)
    model_signature = inspect.signature(module.process_full_spectrum)
    supports_seed_freq = "seed_freq_hz" in model_signature.parameters

    det_kwargs = dict(DEFAULT_DETECTOR_KWARGS)
    if detector_kwargs:
        det_kwargs.update(detector_kwargs)

    file_rows: list[dict[str, Any]] = []
    chirp_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []

    total_ref = 0
    total_det = 0
    total_tp = 0
    total_model_ok = 0

    files_data = annotation_data.get("files", {}) or {}
    benchmark_records = [
        (rel, rec)
        for rel, rec in files_data.items()
        if rec.get("status") in VALID_BENCHMARK_STATUSES
    ]

    if not benchmark_records:
        raise ValueError(
            "No validated annotation records found. Expected status 'annotated' or 'no_chirp'."
        )

    for file_no, (relative_path, record) in enumerate(benchmark_records, start=1):
        wav_path = root / relative_path
        status = record.get("status")
        refs = _manual_chirps(record)

        if status == "annotated" and not refs:
            if verbose:
                print(
                    f"[{file_no}/{len(benchmark_records)}] {relative_path} — "
                    "SKIPPED: status=annotated but no complete chirp"
                )
            continue

        if not wav_path.exists():
            raise FileNotFoundError(f"Annotated WAV not found: {wav_path}")

        if verbose:
            label = f"{len(refs)} manual chirp(s)" if refs else "no_chirp ground truth"
            print(f"[{file_no}/{len(benchmark_records)}] {relative_path} — {label}")

        t0 = time.perf_counter()
        y, sr = _read_mono(wav_path)
        y_filtered = module.high_pass_filter(y, sr)
        candidates = module.detect_candidates_snr_blobs(y_filtered, sr, **det_kwargs)
        detection_s = time.perf_counter() - t0

        ref_intervals = [r["interval_s"] for r in refs]
        det_intervals = [_candidate_interval(c) for c in candidates]
        matches, unmatched_refs, unmatched_dets = match_chirps(
            ref_intervals,
            det_intervals,
            min_iou=min_iou,
            max_center_error_ms=max_center_error_ms,
        )
        match_by_ref = {m.reference_index: m for m in matches}
        matched_det = {m.detected_index for m in matches}

        for j, cand in enumerate(candidates):
            interval = det_intervals[j]
            detection_rows.append({
                "relative_path": relative_path,
                "ground_truth_status": status,
                "detected_index": j,
                "matched": j in matched_det,
                "time_mid_ms": float(cand["time_mid"]) * 1000.0,
                "t_start_ms": interval[0] * 1000.0,
                "t_end_ms": interval[1] * 1000.0,
                "duration_ms": (interval[1] - interval[0]) * 1000.0,
                "peak_freq_khz": float(cand.get("peak_freq_hz", np.nan)) / 1000.0,
                "peak_db": float(cand.get("peak_db", np.nan)),
                "detector_branch": cand.get("detector_branch", "legacy"),
            })

        model_ok_file = 0
        model_time_file = 0.0
        file_duration_s = len(y_filtered) / sr

        for i, ref in enumerate(refs):
            row: dict[str, Any] = {
                "relative_path": relative_path,
                "ground_truth_status": status,
                "chirp_id": ref["chirp_id"],
                "manual_start_ms": ref["t_ms"][0],
                "manual_end_ms": ref["t_ms"][-1],
                "manual_duration_ms": ref["t_ms"][-1] - ref["t_ms"][0],
                "detected": i in match_by_ref,
                "model_success": False,
            }

            if i not in match_by_ref:
                row["failure_stage"] = "detection"
                chirp_rows.append(row)
                continue

            m = match_by_ref[i]
            cand = candidates[m.detected_index]
            row.update({
                "detected_index": m.detected_index,
                "detection_iou": m.iou,
                "detection_center_error_ms": m.center_error_ms,
                "candidate_time_mid_ms": float(cand["time_mid"]) * 1000.0,
                "candidate_peak_freq_khz": float(cand.get("peak_freq_hz", np.nan)) / 1000.0,
                "detector_branch": cand.get("detector_branch", "legacy"),
            })

            mt0 = time.perf_counter()
            try:
                dur = float(np.clip(float(cand.get("duration", 0.005)), 0.005, 0.08))
                model_kwargs = {
                    "time_mid": float(cand["time_mid"]),
                    "duration": dur,
                }
                if supports_seed_freq:
                    seed = float(cand.get("peak_freq_hz", np.nan))
                    if np.isfinite(seed):
                        model_kwargs["seed_freq_hz"] = seed

                curve = module.process_full_spectrum(y_filtered, sr, **model_kwargs)
                model_time = time.perf_counter() - mt0
                model_time_file += model_time
                row["model_time_s"] = model_time

                if curve is None:
                    row["failure_stage"] = "modelling"
                    chirp_rows.append(row)
                    continue

                pred_t, pred_f = _absolute_model_curve(
                    curve, float(cand["time_mid"]), file_duration_s
                )
                metrics = compare_curves(ref["t_ms"], ref["f_khz"], pred_t, pred_f)
                row.update(asdict(metrics))
                row.update({
                    "model_success": True,
                    "failure_stage": "",
                    "pred_start_ms": float(pred_t[0]),
                    "pred_end_ms": float(pred_t[-1]),
                })
                model_ok_file += 1

            except Exception as exc:
                row["model_time_s"] = time.perf_counter() - mt0
                row["failure_stage"] = "modelling_exception"
                row["error"] = f"{type(exc).__name__}: {exc}"

            chirp_rows.append(row)

        tp = len(matches)
        fp = len(unmatched_dets)
        fn = len(unmatched_refs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        file_rows.append({
            "relative_path": relative_path,
            "ground_truth_status": status,
            "manual_chirps": len(refs),
            "detections": len(candidates),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "model_success": model_ok_file,
            "model_success_rate_on_tp": model_ok_file / tp if tp else 0.0,
            "end_to_end_recall": model_ok_file / len(refs) if refs else np.nan,
            "detection_time_s": detection_s,
            "model_time_s": model_time_file,
        })

        total_ref += len(refs)
        total_det += len(candidates)
        total_tp += tp
        total_model_ok += model_ok_file

    files_df = pd.DataFrame(file_rows)
    chirps_df = pd.DataFrame(chirp_rows)
    detections_df = pd.DataFrame(detection_rows)

    fp_total = total_det - total_tp
    fn_total = total_ref - total_tp
    precision = total_tp / (total_tp + fp_total) if total_tp + fp_total else 0.0
    recall = total_tp / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    modeled = (
        chirps_df[chirps_df["model_success"] == True]
        if not chirps_df.empty and "model_success" in chirps_df
        else chirps_df
    )

    def mean_col(name: str) -> float:
        if modeled.empty or name not in modeled:
            return float("nan")
        return float(pd.to_numeric(modeled[name], errors="coerce").mean())

    no_chirp_files = 0
    if not files_df.empty and "ground_truth_status" in files_df:
        no_chirp_files = int((files_df["ground_truth_status"] == "no_chirp").sum())

    summary = {
        "wav_count": int(len(files_df)),
        "annotated_wav_count": int(len(files_df) - no_chirp_files),
        "no_chirp_wav_count": no_chirp_files,
        "manual_chirps": int(total_ref),
        "detections": int(total_det),
        "true_positives": int(total_tp),
        "false_positives": int(fp_total),
        "false_negatives": int(fn_total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "model_success": int(total_model_ok),
        "model_success_rate_on_tp": float(total_model_ok / total_tp) if total_tp else 0.0,
        "end_to_end_recall": float(total_model_ok / total_ref) if total_ref else 0.0,
        "curve_median_abs_error_khz_mean": mean_col("median_abs_error_khz"),
        "curve_rmse_khz_mean": mean_col("rmse_khz"),
        "curve_p95_khz_mean": mean_col("p95_abs_error_khz"),
        "curve_coverage_mean": mean_col("coverage"),
        "detector_kwargs": det_kwargs,
        "model_frequency_seeded": supports_seed_freq,
        "ground_truth_statuses": sorted(VALID_BENCHMARK_STATUSES),
    }

    return BenchmarkResult(summary, files_df, chirps_df, detections_df)
