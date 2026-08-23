"""Chirp detection algorithms.

The legacy mode remains bit-for-bit compatible with the historical SNR/blob
candidate detector. Experimental modes are opt-in so benchmark baselines stay
reproducible.
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


def _candidate_from_blob(blob: dict, pd_b: np.ndarray, freqs_b: np.ndarray, times: np.ndarray, branch: str) -> dict:
    slc = blob["slice"]
    blob_mask = blob["binary_mask_slice"]
    local = np.where(blob_mask == 1, pd_b[slc], -np.inf)
    f_rel, t_rel = np.unravel_index(int(np.argmax(local)), local.shape)
    f_idx = slc[0].start + f_rel
    t_idx = slc[1].start + t_rel
    t_start = float(blob["t_start"])
    t_end = float(blob["t_end"])
    return {
        **blob,
        "time_mid": float(times[t_idx]),
        "duration": max(1e-6, t_end - t_start),
        "peak_freq_hz": float(freqs_b[f_idx]),
        "peak_db": float(pd_b[f_idx, t_idx]),
        "detector_branch": branch,
    }


def _time_nms(candidates: list[dict], window_ms: float) -> list[dict]:
    """Historical strongest-first temporal NMS."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda d: d["peak_db"], reverse=True)
    selected: list[dict] = []
    suppressed: list[tuple[float, float]] = []
    w = float(window_ms) / 1000.0
    for candidate in ordered:
        t = float(candidate["time_mid"])
        if any(a <= t <= b for a, b in suppressed):
            continue
        selected.append(candidate)
        suppressed.append((t - w / 2.0, t + w / 2.0))
    selected.sort(key=lambda d: d["time_mid"])
    return selected


def _time_frequency_nms(candidates: list[dict], window_ms: float, freq_window_hz: float) -> list[dict]:
    """Suppress nearby echoes but preserve simultaneous frequency-separated calls.

    A weaker candidate is suppressed only when it is close in *both* time and
    frequency to a stronger candidate. This is important for recordings where
    a fundamental and harmonics (or two calls) occur at the same time.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda d: d["peak_db"], reverse=True)
    selected: list[dict] = []
    half_window_s = float(window_ms) / 2000.0
    f_window = float(freq_window_hz)
    for cand in ordered:
        t = float(cand["time_mid"])
        f = float(cand["peak_freq_hz"])
        if any(
            abs(t - float(kept["time_mid"])) <= half_window_s
            and abs(f - float(kept["peak_freq_hz"])) <= f_window
            for kept in selected
        ):
            continue
        selected.append(cand)
    selected.sort(key=lambda d: d["time_mid"])
    return selected


def _all_blobs(mask: np.ndarray, times: np.ndarray, freqs_b: np.ndarray) -> list[dict]:
    return base._get_filtered_blobs_info(
        mask, times, freqs_b,
        min_blob_size=0,
        min_blob_height_hz=0,
        max_blob_slope_hz_per_ms=np.inf,
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
    lowfreq_min_blob_height_hz: float = 1500.0,
    lowfreq_max_blob_slope_hz_per_ms: float = 0.0,
    lowfreq_min_width_ms: float = 2.0,
    # adaptive_v2 parameters (benchmark-derived, opt-in)
    lowfreq_snr_threshold_db: float = 9.0,
    general_max_blob_slope_hz_per_ms: float = -500.0,
    echo_suppression_freq_window_hz: float = 25000.0,
):
    """Return candidate bat calls from an SNR-thresholded spectrogram.

    ``legacy`` reproduces the historical implementation exactly.

    ``adaptive_lowfreq`` is the first experimental low-frequency branch.

    ``adaptive_v2`` is a multiband detector developed against the annotated
    benchmark. It uses the normal 10 dB SNR mask for ordinary FM calls, a
    separate (default 9 dB) low-frequency mask for shallow calls <=45 kHz,
    relaxes the general downhill-slope gate to -500 Hz/ms, and uses a
    time+frequency NMS so simultaneous separated bands are not discarded.
    Its defaults are experimental until validated on a larger hold-out set.
    """
    if slope_filter_mode == "legacy":
        return base.detect_candidates_snr_blobs(
            y, sr,
            snr_threshold_db=snr_threshold_db,
            percentile_q=percentile_q,
            fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
            min_blob_size=min_blob_size,
            min_blob_height_hz=min_blob_height_hz,
            max_blob_slope_hz_per_ms=max_blob_slope_hz_per_ms,
            echo_suppression_window_ms=echo_suppression_window_ms,
        )

    snr_map, freqs_b, times, dbg = base.compute_snr_map(
        y, sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop,
        noise_q=percentile_q, noise_mode="percentile",
    )
    pd_b = dbg["PdB"]

    if slope_filter_mode == "adaptive_lowfreq":
        mask = (snr_map >= snr_threshold_db).astype(np.uint8)
        candidates: list[dict] = []
        for blob in _all_blobs(mask, times, freqs_b):
            size = int(blob["size"])
            height = float(blob["height_hz"])
            slope = float(blob["slope_hz_per_ms"])
            width = float(blob["width_ms"])
            f_high = float(blob["f_high"])
            legacy_ok = (
                (min_blob_size <= 0 or size > min_blob_size)
                and (min_blob_height_hz <= 0 or height > min_blob_height_hz)
                and (max_blob_slope_hz_per_ms == np.inf or slope <= max_blob_slope_hz_per_ms)
            )
            low_ok = (
                f_high <= lowfreq_max_hz
                and (min_blob_size <= 0 or size > min_blob_size)
                and height >= lowfreq_min_blob_height_hz
                and width >= lowfreq_min_width_ms
                and slope < lowfreq_max_blob_slope_hz_per_ms
            )
            if legacy_ok or low_ok:
                candidates.append(_candidate_from_blob(blob, pd_b, freqs_b, times, "legacy" if legacy_ok else "lowfreq"))
        return _time_nms(candidates, echo_suppression_window_ms)

    if slope_filter_mode != "adaptive_v2":
        raise ValueError("slope_filter_mode must be 'legacy', 'adaptive_lowfreq', or 'adaptive_v2'")

    candidates: list[dict] = []

    # General FM branch.
    general_mask = (snr_map >= snr_threshold_db).astype(np.uint8)
    for blob in _all_blobs(general_mask, times, freqs_b):
        if (
            (min_blob_size <= 0 or int(blob["size"]) > min_blob_size)
            and (min_blob_height_hz <= 0 or float(blob["height_hz"]) > min_blob_height_hz)
            and float(blob["slope_hz_per_ms"]) <= general_max_blob_slope_hz_per_ms
        ):
            candidates.append(_candidate_from_blob(blob, pd_b, freqs_b, times, "general"))

    # Lower-threshold shallow low-frequency branch.
    low_mask = (snr_map >= lowfreq_snr_threshold_db).astype(np.uint8)
    for blob in _all_blobs(low_mask, times, freqs_b):
        if (
            float(blob["f_high"]) <= lowfreq_max_hz
            and (min_blob_size <= 0 or int(blob["size"]) > min_blob_size)
            and float(blob["height_hz"]) >= lowfreq_min_blob_height_hz
            and float(blob["width_ms"]) >= lowfreq_min_width_ms
            and float(blob["slope_hz_per_ms"]) < lowfreq_max_blob_slope_hz_per_ms
        ):
            candidates.append(_candidate_from_blob(blob, pd_b, freqs_b, times, "lowfreq"))

    return _time_frequency_nms(
        candidates,
        echo_suppression_window_ms,
        echo_suppression_freq_window_hz,
    )


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
