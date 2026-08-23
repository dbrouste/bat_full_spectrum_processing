"""Chirp detection algorithms.

This module provides a stable public detector API around the legacy SNR/blob
implementation.  The default ``legacy`` mode is intentionally identical to the
historical detector so benchmark baselines remain reproducible.

An experimental ``adaptive_lowfreq`` mode is also available.  It keeps the
legacy blob rules for ordinary steep FM calls, but adds a second low-frequency
branch for shallow downward calls that the historical hard slope/height gates
cannot represent.
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


def _passes_adaptive_blob_filter(
    blob: dict,
    *,
    min_blob_size: int,
    min_blob_height_hz: float,
    max_blob_slope_hz_per_ms: float,
    lowfreq_max_hz: float,
    lowfreq_min_blob_height_hz: float,
    lowfreq_max_blob_slope_hz_per_ms: float,
    lowfreq_min_width_ms: float,
) -> bool:
    """Return True for either the legacy branch or the low-frequency branch."""

    size = int(blob["size"])
    height_hz = float(blob["height_hz"])
    slope = float(blob["slope_hz_per_ms"])
    width_ms = float(blob["width_ms"])
    f_high = float(blob["f_high"])

    legacy_ok = (
        (min_blob_size <= 0 or size > min_blob_size)
        and (min_blob_height_hz <= 0 or height_hz > min_blob_height_hz)
        and (
            max_blob_slope_hz_per_ms == np.inf
            or slope <= max_blob_slope_hz_per_ms
        )
    )

    # The ground-truth set contains a distinct family of long, shallow,
    # low-frequency calls around 30--40 kHz.  Their total frequency excursion
    # can be only a few kHz, so the historical >5 kHz height and <-2 kHz/ms
    # slope requirements reject them structurally.  Keep this branch narrow so
    # it does not relax the ordinary FM detector everywhere.
    lowfreq_ok = (
        f_high <= lowfreq_max_hz
        and (min_blob_size <= 0 or size > min_blob_size)
        and height_hz >= lowfreq_min_blob_height_hz
        and width_ms >= lowfreq_min_width_ms
        and slope < lowfreq_max_blob_slope_hz_per_ms
    )

    return bool(legacy_ok or lowfreq_ok)


def _time_nms(candidates: list[dict], window_ms: float) -> list[dict]:
    """Apply the same strongest-first temporal NMS as the legacy detector."""
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda d: d["peak_db"], reverse=True)
    selected: list[dict] = []
    suppressed_intervals: list[tuple[float, float]] = []
    w = float(window_ms) / 1000.0

    for candidate in ordered:
        t = float(candidate["time_mid"])
        if any(start <= t <= end for start, end in suppressed_intervals):
            continue
        selected.append(candidate)
        suppressed_intervals.append((t - w / 2.0, t + w / 2.0))

    selected.sort(key=lambda d: d["time_mid"])
    return selected


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
):
    """Detect candidate bat calls from an SNR map.

    Parameters added by this wrapper
    --------------------------------
    slope_filter_mode:
        ``"legacy"`` reproduces the historical detector exactly.
        ``"adaptive_lowfreq"`` keeps the legacy branch and additionally admits
        shallow downward blobs whose upper frequency is <= ``lowfreq_max_hz``.

    The adaptive mode is deliberately opt-in until it has been benchmarked on
    the annotated WAV set.
    """

    if slope_filter_mode == "legacy":
        return base.detect_candidates_snr_blobs(
            y,
            sr,
            snr_threshold_db=snr_threshold_db,
            percentile_q=percentile_q,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            hop=hop,
            min_blob_size=min_blob_size,
            min_blob_height_hz=min_blob_height_hz,
            max_blob_slope_hz_per_ms=max_blob_slope_hz_per_ms,
            echo_suppression_window_ms=echo_suppression_window_ms,
        )

    if slope_filter_mode != "adaptive_lowfreq":
        raise ValueError(
            "slope_filter_mode must be 'legacy' or 'adaptive_lowfreq'"
        )

    snr_map, freqs_b, times, dbg = base.compute_snr_map(
        y,
        sr,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        hop=hop,
        noise_q=percentile_q,
        noise_mode="percentile",
    )
    pd_b = dbg["PdB"]
    mask = (snr_map >= snr_threshold_db).astype(np.uint8)

    # Get all connected components first; filtering is performed below so the
    # low-frequency branch can recover blobs rejected by the legacy height and
    # slope thresholds.
    blobs = base._get_filtered_blobs_info(
        mask,
        times,
        freqs_b,
        min_blob_size=0,
        min_blob_height_hz=0,
        max_blob_slope_hz_per_ms=np.inf,
    )

    candidates: list[dict] = []
    for blob in blobs:
        if not _passes_adaptive_blob_filter(
            blob,
            min_blob_size=min_blob_size,
            min_blob_height_hz=min_blob_height_hz,
            max_blob_slope_hz_per_ms=max_blob_slope_hz_per_ms,
            lowfreq_max_hz=lowfreq_max_hz,
            lowfreq_min_blob_height_hz=lowfreq_min_blob_height_hz,
            lowfreq_max_blob_slope_hz_per_ms=lowfreq_max_blob_slope_hz_per_ms,
            lowfreq_min_width_ms=lowfreq_min_width_ms,
        ):
            continue

        slc = blob["slice"]
        blob_mask = blob["binary_mask_slice"]
        pd_b_blob = pd_b[slc]
        pd_b_masked = np.where(blob_mask == 1, pd_b_blob, -np.inf)
        flat = int(np.argmax(pd_b_masked))
        f_rel, t_rel = np.unravel_index(flat, pd_b_masked.shape)

        f_idx = slc[0].start + f_rel
        t_idx = slc[1].start + t_rel
        t_start = float(blob["t_start"])
        t_end = float(blob["t_end"])

        candidate = {
            **blob,
            "time_mid": float(times[t_idx]),
            "duration": max(1e-6, t_end - t_start),
            "peak_freq_hz": float(freqs_b[f_idx]),
            "peak_db": float(pd_b[f_idx, t_idx]),
            "detector_branch": (
                "legacy"
                if (
                    float(blob["height_hz"]) > min_blob_height_hz
                    and float(blob["slope_hz_per_ms"])
                    <= max_blob_slope_hz_per_ms
                )
                else "lowfreq"
            ),
        }
        candidates.append(candidate)

    return _time_nms(candidates, echo_suppression_window_ms)


def detect_chirps(y: np.ndarray, sr: int, **kwargs) -> List[Detection]:
    """Public typed detector API."""
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
