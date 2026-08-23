"""Chirp modelling entry point used by the benchmark and future pipeline.

This module wraps the current legacy processing helpers while removing the
absolute ``max_value >= 0.7`` gate from ``process_full_spectrum``.  The gate was
shown by the annotated benchmark to reject 63/63 otherwise modelable true
positive chirps, so it is intentionally not applied here.

No other modelling thresholds or processing steps are changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import librosa
import numpy as np

from bat_analysis import bfsp_clean_patched as base


# Re-export the detector/filter API so this file can be passed directly to
# benchmark.runner.run_benchmark(..., analysis_py=...).
high_pass_filter = base.high_pass_filter
detect_candidates_snr_blobs = base.detect_candidates_snr_blobs


@dataclass
class ChirpModelResult:
    model_name: str
    parameters: Dict[str, float]
    diagnostics: Dict[str, Any]


def process_full_spectrum(y_use, sr, time_mid, duration):
    """Model a detected chirp without the legacy absolute-amplitude gate.

    This is deliberately the same algorithm as
    ``bfsp_clean_patched.process_full_spectrum`` except for removal of::

        if max_value < 0.7:
            return None

    Relative amplitude criteria inside the tracking/extension stages are kept
    unchanged.
    """
    y_chun = base.Extract_chunk_of_audio(y_use, sr, time_mid)

    curve_all = base.initial_call_trend(y_chun, sr, duration)
    if curve_all is None or len(curve_all) == 0:
        return None

    max_value = max(entry[2] for entry in curve_all)

    S = np.abs(
        librosa.stft(y_chun, n_fft=1024, hop_length=100, window="flattop")
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=sr, hop_length=100
    )

    curve_all = base.process_side(
        curve_all, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value
    )
    if curve_all is None:
        return None

    curve_all = base.process_side(
        curve_all, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value
    )
    if curve_all is None:
        return None

    max_curve_segmented = 0

    curve_segmented = base.extend_trend_left(
        y_chun, sr, curve_all, S, freqs, times, max_value
    )
    if curve_segmented is not None:
        max_curve_segmented = max(entry[2] for entry in curve_segmented)
        while (
            base.calculate_angle_between_trends(curve_all, curve_segmented) < 70
            and base.get_number_of_points(curve_segmented) > 11
            and max_curve_segmented > max_value / 20
        ):
            curve_all = base.concatenate_trends(curve_segmented, curve_all)
            curve_segmented = base.extend_trend_left(
                y_chun, sr, curve_all, S, freqs, times, max_value
            )
            if curve_segmented is None:
                break
            max_curve_segmented = max(entry[2] for entry in curve_segmented)

    curve_segmented = base.extend_trend_right(
        y_chun, sr, curve_all, S, freqs, times, max_value
    )
    if curve_segmented is not None:
        max_curve_segmented = max(entry[2] for entry in curve_segmented)
        while (
            base.calculate_angle_between_trends(curve_all, curve_segmented) < 50
            and base.get_number_of_points(curve_segmented) > 11
            and base.validate_slope(curve_segmented, nb_points=25, LeftRight=0)
            and base.get_max_frequency(curve_segmented)
            < (base.get_min_frequency(curve_all) + 500)
            and max_curve_segmented > max_value / 10
        ):
            curve_all = base.concatenate_trends(curve_all, curve_segmented)
            curve_segmented = base.extend_trend_right(
                y_chun, sr, curve_all, S, freqs, times, max_value
            )
            if curve_segmented is None:
                break
            max_curve_segmented = max(entry[2] for entry in curve_segmented)

    if curve_all is None:
        return None

    curve_all = base.interpolate_trend_results(curve_all)
    curve_all = base.fit_spline_with_smoothness(curve_all)
    curve_all = base.sample_curve_equally(curve_all)
    return curve_all


def fit_chirp_model(*args, **kwargs) -> ChirpModelResult:
    """Public API placeholder for later compact model fitting."""
    raise NotImplementedError("Compact model fitting will be selected after ridge benchmarking.")
