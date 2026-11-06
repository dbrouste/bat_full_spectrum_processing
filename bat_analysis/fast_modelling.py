"""Lightweight entry point for the Reassignment-v2 ridge model.

Unlike modelling.py, this module does NOT import librosa or the legacy
Gaussian modelling pipeline. Worker processes which only reconstruct
Flattop-reassigned ridges can import this directly, saving expensive
library initialization at every Windows 'spawn'.
"""
from __future__ import annotations

import numpy as np


def process_reassigned_fast(
    y_use,
    sr,
    time_mid,
    duration,
    seed_freq_hz=None,
    broad_fm_seed_freq_hz=None,
    diagnostics=None,
    chirp_start_s=None,
    chirp_end_s=None,
    broad_fm_seed_time_s=None,
    reassignment_n_fft=None,
    reassignment_hop=None,
    reassignment_subbin_refine=True,
    reassignment_refine_endpoints=True,
    reassignment_complete_ends=True,
    reassignment_extension_min_snr_db=17.0,
    reassignment_trim_weak_tail=True,
    reassignment_tail_min_snr_db=10.0,
    reassignment_tail_min_relative_db=-22.0,
    reassignment_peak_anchored_window=True,
    reassignment_cross_seed_velocity_penalty=0.6,
    reassignment_cross_seed_position_penalty=0.01,
):
    """Model a ridge with the same convention as modelling.process_full_spectrum.

    Returns [chunk-relative time_s, frequency_hz, normalized amplitude].
    The 15 ms chunk start is *only* a time-origin convention; the underlying
    Reassignment tracker analyses its own local acoustic window.
    """
    from .reassignment_tracking import track_reassigned_ridge

    if diagnostics is not None:
        diagnostics.clear()
    chunk_start = max(0, int((time_mid - 0.0075) * sr)) / sr
    use_coarse = (
        broad_fm_seed_freq_hz is not None
        and np.isfinite(broad_fm_seed_freq_hz)
        and broad_fm_seed_time_s is not None
        and np.isfinite(broad_fm_seed_time_s)
    )
    chosen_seed_time = float(broad_fm_seed_time_s) if use_coarse else float(time_mid)
    chosen_seed_freq = (
        float(broad_fm_seed_freq_hz) if use_coarse else seed_freq_hz
    )
    bounds = (
        (float(chirp_start_s), float(chirp_end_s))
        if chirp_start_s is not None and chirp_end_s is not None
        else None
    )
    curve = track_reassigned_ridge(
        y_use, sr,
        time_mid=chosen_seed_time, duration=duration,
        seed_freq_hz=chosen_seed_freq, time_bounds_s=bounds,
        n_fft=reassignment_n_fft, hop=reassignment_hop,
        subbin_refine=reassignment_subbin_refine,
        refine_endpoints=reassignment_refine_endpoints,
        complete_ends=reassignment_complete_ends,
        extension_min_snr_db=reassignment_extension_min_snr_db,
        trim_weak_tail=reassignment_trim_weak_tail,
        tail_min_snr_db=reassignment_tail_min_snr_db,
        tail_min_relative_db=reassignment_tail_min_relative_db,
        peak_anchored_window=reassignment_peak_anchored_window,
        cross_seed_velocity_penalty=reassignment_cross_seed_velocity_penalty,
        cross_seed_position_penalty=reassignment_cross_seed_position_penalty,
        diagnostics=diagnostics,
    )
    if curve is not None:
        curve[:, 0] -= chunk_start
    elif diagnostics is not None:
        diagnostics["model_success"] = False
        diagnostics["final_points"] = 0
    return curve
