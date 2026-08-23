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
    frequency recovery branch, and bbox overlap deduplication. Keep opt-in until
    validated on independent annotated/no_chirp WAVs.
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
        candidates.append({
            **blob,
            "time_mid": float(times[t_idx]),
            "duration": max(1e-6, t_end - t_start),
            "peak_freq_hz": float(freqs_b[f_idx]),
            "peak_db": float(pd_b[f_idx, t_idx]),
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
):
    if slope_filter_mode == "legacy":
        return base.detect_candidates_snr_blobs(
            y, sr, snr_threshold_db=snr_threshold_db, percentile_q=percentile_q,
            fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
            min_blob_size=min_blob_size, min_blob_height_hz=min_blob_height_hz,
            max_blob_slope_hz_per_ms=max_blob_slope_hz_per_ms,
            echo_suppression_window_ms=echo_suppression_window_ms,
        )

    if slope_filter_mode not in {"adaptive_lowfreq", "adaptive_v2", "adaptive_v3"}:
        raise ValueError("slope_filter_mode must be 'legacy', 'adaptive_lowfreq', 'adaptive_v2', or 'adaptive_v3'")

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
