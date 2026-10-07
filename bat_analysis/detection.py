"""Chirp detection algorithms.

Stable wrapper around the legacy SNR/blob detector.

Modes
-----
legacy
    Exact historical detector.
adaptive_lowfreq
    Compatibility mode adding one shallow low-frequency branch at the same SNR.
adaptive_v2
    Development detector benchmarked on 8 validated WAV / 158 chirps. Uses a
    10 dB general branch, a 9 dB low-frequency branch, relaxed general slope,
    and time+frequency NMS.
adaptive_v3
    Conservative extension of v2: 20 ms time+frequency NMS, a narrow high-
    frequency recovery branch, and bbox overlap deduplication.
adaptive_v4
    Independent-dataset refinement: removes the unhelpful high-frequency
    recovery branch, rejects very short general blobs, and keeps only long,
    shallow low-frequency blobs. Developed after freezing a grouped
    development/hold-out split of the 61-WAV validation dataset.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from bat_analysis import bfsp_clean_patched as base

@dataclass
class Detection:
    t_start_ms: float
    t_end_ms: float
    peak_time_ms: float
    peak_freq_khz: float
    score: float | None = None


def _coarse_ridge_seed(
    pd_b_blob: np.ndarray,
    blob_mask: np.ndarray,
    blob_slice,
    freqs_b: np.ndarray,
    times: np.ndarray,
) -> dict:
    """Extract a coarse downward ridge seed from one connected detector blob.

    This seed is not the final ridge. For each time column we keep the strongest
    in-blob frequency, lightly median-filter that path, then find the most
    coherent short downward segment. The result is used only to initialize
    difficult broad-FM calls where a single global peak can sit on the tail.
    """
    ridge_t = []
    ridge_f = []
    for j in range(blob_mask.shape[1]):
        rows = np.flatnonzero(blob_mask[:, j])
        if rows.size == 0:
            continue
        k = int(rows[np.argmax(pd_b_blob[rows, j])])
        ridge_t.append(float(times[blob_slice[1].start + j]))
        ridge_f.append(float(freqs_b[blob_slice[0].start + k]))

    if len(ridge_t) < 3:
        return {}

    ridge_t = np.asarray(ridge_t, dtype=float)
    ridge_f = np.asarray(ridge_f, dtype=float)
    ridge_f_smooth = base.scipy.ndimage.median_filter(
        ridge_f, size=3, mode="nearest"
    )

    best = None
    max_window = min(len(ridge_t), 12)
    for window in range(3, max_window + 1):
        for start in range(0, len(ridge_t) - window + 1):
            x = ridge_t[start:start + window]
            y = ridge_f_smooth[start:start + window]
            slope_hz_s, intercept = np.polyfit(x, y, 1)
            slope_hz_ms = slope_hz_s / 1000.0
            if slope_hz_ms >= -500.0:
                continue

            pred = slope_hz_s * x + intercept
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            drop_khz = max(0.0, float(y[0] - y[-1]) / 1000.0)
            score = drop_khz * max(r2, 0.05) * np.sqrt(window)

            if best is None or score > best["coarse_seed_score"]:
                mid_t = 0.5 * float(x[0] + x[-1])
                mid_f = float(slope_hz_s * mid_t + intercept)
                best = {
                    "coarse_seed_freq_hz": mid_f,
                    "coarse_seed_time_s": mid_t,
                    "coarse_seed_slope_hz_per_ms": float(slope_hz_ms),
                    "coarse_seed_score": float(score),
                    "coarse_seed_r2": float(r2),
                    "coarse_seed_drop_khz": float(drop_khz),
                }

    return best or {}


def _extract_candidates(y, sr, *, snr_threshold_db, percentile_q, fmin, fmax, n_fft, hop):
    snr_map, freqs_b, times, dbg = base.compute_snr_map(
        y, sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
        noise_q=percentile_q, noise_mode="percentile",
    )
    pd_b = dbg["PdB"]
    mask = (snr_map >= snr_threshold_db).astype(np.uint8)
    blobs = base._get_filtered_blobs_info(
        mask, times, freqs_b,
        min_blob_size=0, min_blob_height_hz=0,
        max_blob_slope_hz_per_ms=np.inf,
    )
    candidates = []
    for blob in blobs:
        slc = blob["slice"]
        blob_mask = blob["binary_mask_slice"]
        pd_b_masked = np.where(blob_mask == 1, pd_b[slc], -np.inf)
        flat = int(np.argmax(pd_b_masked))
        f_rel, t_rel = np.unravel_index(flat, pd_b_masked.shape)
        f_idx = slc[0].start + f_rel
        t_idx = slc[1].start + t_rel
        t_start = float(blob["t_start"])
        t_end = float(blob["t_end"])
        coarse = _coarse_ridge_seed(
            pd_b[slc], blob_mask, slc, freqs_b, times
        )
        f_span = max(float(blob["f_high"] - blob["f_low"]), 1e-9)
        peak_position = float(
            (float(freqs_b[f_idx]) - float(blob["f_low"])) / f_span
        )
        use_coarse = (
            float(blob["width_ms"]) >= 8.0
            and float(blob["height_hz"]) >= 20000.0
            and peak_position <= 0.15
            and bool(coarse)
        )

        candidates.append({
            **blob,
            "time_mid": float(times[t_idx]),
            "duration": max(1e-6, t_end - t_start),
            "peak_freq_hz": float(freqs_b[f_idx]),
            "peak_db": float(pd_b[f_idx, t_idx]),
            "peak_position_in_bbox": peak_position,
            "broad_fm_seed_freq_hz": (
                float(coarse["coarse_seed_freq_hz"]) if use_coarse else np.nan
            ),
            **coarse,
        })
    return candidates


def _time_freq_nms(candidates: list[dict], window_ms: float, freq_window_hz: float) -> list[dict]:
    if not candidates:
        return []
    selected = []
    half_s = float(window_ms) / 2000.0
    for candidate in sorted(candidates, key=lambda d: d["peak_db"], reverse=True):
        if any(
            abs(float(candidate["time_mid"]) - float(other["time_mid"])) <= half_s
            and abs(float(candidate["peak_freq_hz"]) - float(other["peak_freq_hz"])) <= freq_window_hz
            for other in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda d: d["time_mid"])
    return selected


def _bbox_overlap_dedup(candidates: list[dict], time_iou_threshold: float, freq_iou_threshold: float) -> list[dict]:
    """Suppress weaker near-duplicate blobs only when both bboxes overlap strongly."""
    if not candidates:
        return []
    selected = []
    for candidate in sorted(candidates, key=lambda d: d["peak_db"], reverse=True):
        suppress = False
        for other in selected:
            t_inter = max(0.0, min(candidate["t_end"], other["t_end"]) - max(candidate["t_start"], other["t_start"]))
            t_union = max(candidate["t_end"], other["t_end"]) - min(candidate["t_start"], other["t_start"])
            t_iou = t_inter / t_union if t_union > 0 else 0.0
            f_inter = max(0.0, min(candidate["f_high"], other["f_high"]) - max(candidate["f_low"], other["f_low"]))
            f_union = max(candidate["f_high"], other["f_high"]) - min(candidate["f_low"], other["f_low"])
            f_iou = f_inter / f_union if f_union > 0 else 0.0
            if t_iou >= time_iou_threshold and f_iou >= freq_iou_threshold:
                suppress = True
                break
        if not suppress:
            selected.append(candidate)
    selected.sort(key=lambda d: d["time_mid"])
    return selected


def _legacy_pass(c, min_blob_size, min_blob_height_hz, max_blob_slope_hz_per_ms):
    return (
        (min_blob_size <= 0 or int(c["size"]) > min_blob_size)
        and (min_blob_height_hz <= 0 or float(c["height_hz"]) > min_blob_height_hz)
        and (max_blob_slope_hz_per_ms == np.inf or float(c["slope_hz_per_ms"]) <= max_blob_slope_hz_per_ms)
    )


def detect_candidates_snr_blobs(
    y: np.ndarray,
    sr: int,
    *,
    snr_threshold_db: float = 10.0,
    percentile_q: float = 96.0,
    fmin: float = 20000,
    fmax: float = 150000,
    n_fft: int = 512,
    hop: int | None = 128,
    min_blob_size: int = 10,
    min_blob_height_hz: float = 5000.0,
    max_blob_slope_hz_per_ms: float = -2000.0,
    echo_suppression_window_ms: float = 10.0,
    slope_filter_mode: str = "legacy",
    lowfreq_max_hz: float = 45000.0,
    lowfreq_min_blob_height_hz: float = 2000.0,
    lowfreq_max_blob_slope_hz_per_ms: float = 0.0,
    lowfreq_min_width_ms: float = 1.0,
    lowfreq_snr_threshold_db: float = 9.0,
    general_max_blob_slope_hz_per_ms: float = -500.0,
    echo_suppression_freq_window_hz: float = 25000.0,
    highfreq_snr_threshold_db: float = 9.0,
    highfreq_min_hz: float = 50000.0,
    highfreq_min_blob_height_hz: float = 3000.0,
    highfreq_min_width_ms: float = 4.0,
    highfreq_min_blob_size: int = 10,
    highfreq_max_blob_slope_hz_per_ms: float = -1000.0,
    bbox_dedup_time_iou: float = 0.20,
    bbox_dedup_freq_iou: float = 0.40,
    v4_general_min_width_ms: float = 1.5,
    v4_lowfreq_min_width_ms: float = 3.0,
    v4_lowfreq_min_slope_hz_per_ms: float = -1500.0,
):
    if slope_filter_mode == "legacy":
        return base.detect_candidates_snr_blobs(
            y, sr, snr_threshold_db=snr_threshold_db, percentile_q=percentile_q,
            fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
            min_blob_size=min_blob_size, min_blob_height_hz=min_blob_height_hz,
            max_blob_slope_hz_per_ms=max_blob_slope_hz_per_ms,
            echo_suppression_window_ms=echo_suppression_window_ms,
        )

    if slope_filter_mode not in {"adaptive_lowfreq", "adaptive_v2", "adaptive_v3", "adaptive_v4"}:
        raise ValueError(
            "slope_filter_mode must be 'legacy', 'adaptive_lowfreq', "
            "'adaptive_v2', 'adaptive_v3', or 'adaptive_v4'"
        )

    general_raw = _extract_candidates(
        y, sr, snr_threshold_db=snr_threshold_db, percentile_q=percentile_q,
        fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
    )

    if slope_filter_mode == "adaptive_lowfreq":
        candidates = []
        for c in general_raw:
            if _legacy_pass(c, min_blob_size, min_blob_height_hz, max_blob_slope_hz_per_ms):
                candidates.append({**c, "detector_branch": "legacy"})
                continue
            low_ok = (
                float(c["f_high"]) <= lowfreq_max_hz
                and (min_blob_size <= 0 or int(c["size"]) > min_blob_size)
                and float(c["height_hz"]) >= lowfreq_min_blob_height_hz
                and float(c["width_ms"]) >= lowfreq_min_width_ms
                and float(c["slope_hz_per_ms"]) < lowfreq_max_blob_slope_hz_per_ms
            )
            if low_ok:
                candidates.append({**c, "detector_branch": "lowfreq"})
        return _time_freq_nms(candidates, echo_suppression_window_ms, np.inf)

    general = [
        {**c, "detector_branch": "general"}
        for c in general_raw
        if (min_blob_size <= 0 or int(c["size"]) > min_blob_size)
        and (min_blob_height_hz <= 0 or float(c["height_hz"]) > min_blob_height_hz)
        and float(c["slope_hz_per_ms"]) <= general_max_blob_slope_hz_per_ms
    ]

    low_raw = _extract_candidates(
        y, sr, snr_threshold_db=lowfreq_snr_threshold_db, percentile_q=percentile_q,
        fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
    )
    low = [
        {**c, "detector_branch": "lowfreq"}
        for c in low_raw
        if float(c["f_high"]) <= lowfreq_max_hz
        and (min_blob_size <= 0 or int(c["size"]) > min_blob_size)
        and float(c["height_hz"]) >= lowfreq_min_blob_height_hz
        and float(c["width_ms"]) >= lowfreq_min_width_ms
        and float(c["slope_hz_per_ms"]) < lowfreq_max_blob_slope_hz_per_ms
    ]

    nms_window = 16.0 if slope_filter_mode == "adaptive_v2" and echo_suppression_window_ms == 10.0 else echo_suppression_window_ms
    if slope_filter_mode == "adaptive_v2":
        return _time_freq_nms(general + low, nms_window, echo_suppression_freq_window_hz)

    if slope_filter_mode == "adaptive_v4":
        general_v4 = [
            c for c in general
            if float(c["width_ms"]) >= v4_general_min_width_ms
        ]
        low_v4 = [
            c for c in low_raw
            if float(c["f_high"]) <= lowfreq_max_hz
            and (min_blob_size <= 0 or int(c["size"]) > min_blob_size)
            and float(c["height_hz"]) >= lowfreq_min_blob_height_hz
            and float(c["width_ms"]) >= v4_lowfreq_min_width_ms
            and v4_lowfreq_min_slope_hz_per_ms <= float(c["slope_hz_per_ms"]) < 0.0
        ]
        low_v4 = [{**c, "detector_branch": "lowfreq"} for c in low_v4]
        v4_window = 20.0 if echo_suppression_window_ms == 10.0 else echo_suppression_window_ms
        return _time_freq_nms(
            general_v4 + low_v4,
            v4_window,
            echo_suppression_freq_window_hz,
        )

    high_raw = _extract_candidates(
        y, sr, snr_threshold_db=highfreq_snr_threshold_db, percentile_q=percentile_q,
        fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
    )
    high = [
        {**c, "detector_branch": "highfreq"}
        for c in high_raw
        if float(c["f_low"]) >= highfreq_min_hz
        and int(c["size"]) > highfreq_min_blob_size
        and float(c["height_hz"]) >= highfreq_min_blob_height_hz
        and float(c["width_ms"]) >= highfreq_min_width_ms
        and float(c["slope_hz_per_ms"]) <= highfreq_max_blob_slope_hz_per_ms
    ]
    v3_window = 20.0 if echo_suppression_window_ms == 10.0 else echo_suppression_window_ms
    selected = _time_freq_nms(general + low + high, v3_window, echo_suppression_freq_window_hz)
    return _bbox_overlap_dedup(selected, bbox_dedup_time_iou, bbox_dedup_freq_iou)


def detect_chirps(y: np.ndarray, sr: int, **kwargs) -> List[Detection]:
    candidates = detect_candidates_snr_blobs(y, sr, **kwargs)
    return [
        Detection(
            t_start_ms=1000.0 * float(c["t_start"]),
            t_end_ms=1000.0 * float(c["t_end"]),
            peak_time_ms=1000.0 * float(c["time_mid"]),
            peak_freq_khz=float(c["peak_freq_hz"]) / 1000.0,
            score=float(c["peak_db"]),
        )
        for c in candidates
    ]
