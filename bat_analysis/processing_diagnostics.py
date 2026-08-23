from __future__ import annotations

"""Diagnostic wrappers for the current chirp modelling pipeline.

This module deliberately does not change the production thresholds or return
values in ``bfsp_clean_patched.py``.  It mirrors the early part of
``process_full_spectrum`` and ``process_side`` so benchmark runs can identify
why a correctly detected chirp is rejected.

Once the dominant failure reason is known, the corresponding production code
can be changed and benchmarked objectively.
"""

from typing import Any

import librosa
import numpy as np

from . import bfsp_clean_patched as base


# Re-export the detector entry points so this module can also be selected as a
# benchmark target if desired.
high_pass_filter = base.high_pass_filter
detect_candidates_snr_blobs = base.detect_candidates_snr_blobs


def _stop_reason(
    *,
    gaussian_variation_percent: float,
    distance_from_max_gauss: float,
    gauss_z_value: float,
    max_amplitude: float,
    freq_from_max_gauss: float,
    angle_of_trend: float,
    gauss_x_value: float,
    left_right: int,
    min_previous_x: float,
    max_previous_x: float,
    limit_gauss_x_value: float,
    previous_coordinate: Any,
    current_coordinate: Any,
    iteration: int,
    max_iterations: int,
) -> str:
    """Return the first production while-condition that prevents continuation."""
    if gaussian_variation_percent >= 30:
        return "gaussian_variation_ge_30"
    if abs(distance_from_max_gauss) >= 0.00069:
        return "gaussian_offset_ge_0_69ms"
    if gauss_z_value <= max_amplitude / 20:
        return "amplitude_below_max_over_20"
    if iteration >= max_iterations:
        return "max_iterations"
    if abs(freq_from_max_gauss) >= 1300:
        return "gaussian_frequency_correction_ge_1300hz"
    if angle_of_trend >= 50:
        return "trend_angle_ge_50deg"
    if previous_coordinate is not None and current_coordinate == previous_coordinate and iteration >= 2:
        return "coordinate_not_changing"
    if left_right == 0 and not (gauss_x_value < min_previous_x):
        return "overlaps_previous_curve_left"
    if left_right == 1 and not (gauss_x_value > max_previous_x):
        return "overlaps_previous_curve_right"
    if left_right == 0 and not (gauss_x_value <= limit_gauss_x_value + 0.00002):
        return "time_backtrack_left"
    if left_right == 1 and not (gauss_x_value >= limit_gauss_x_value - 0.00002):
        return "time_backtrack_right"
    if gauss_x_value <= 0:
        return "chunk_left_boundary"
    if gauss_x_value >= 0.015:
        return "chunk_right_boundary"
    return "loop_stopped_other"


def process_side_diagnostic(
    seven_points,
    spectro,
    freqs,
    times,
    sr,
    left_right,
    max_amplitude,
    previous_curve=None,
    max_iterations=500,
    min_time_progress=1e-9,
):
    """Diagnostic equivalent of ``base.process_side``.

    Returns ``(curve_or_none, diagnostic_dict)``. Production thresholds and
    branch ordering are intentionally kept identical to the current code.
    """
    side_name = "left" if left_right == 0 else "right"
    diag: dict[str, Any] = {
        "side": side_name,
        "success": False,
        "reason": None,
        "iterations": 0,
        "points_initial": 0 if seven_points is None else int(len(seven_points)),
    }

    if seven_points is None:
        diag["reason"] = "input_curve_none"
        return None, diag

    if previous_curve is not None:
        min_previous_x = min(point[0] for point in previous_curve)
        max_previous_x = max(point[0] for point in previous_curve)
    else:
        min_previous_x, max_previous_x = float("inf"), float("-inf")

    new_point, new_slope = base.get_extrapolated_points(
        seven_points, LeftRight=left_right, nb_point_to_consider=20
    )
    if new_point is None:
        diag["reason"] = "initial_extrapolation_failed"
        return None, diag

    t_line, amplitude_line = base.sample_line_from_max_amp_dynamic(
        spectro, freqs, times, new_point, sr, slope=new_slope
    )
    if t_line is None:
        diag["reason"] = "initial_sample_line_failed"
        return None, diag

    if t_line.size < 10:
        diag.update({
            "success": True,
            "reason": "sample_line_too_short_but_curve_kept",
            "points_final": int(len(seven_points)),
        })
        return seven_points, diag

    (
        gaussian_variation,
        gauss_z,
        distance,
        gaussian_function,
        freq_correction,
    ) = base.calculate_percentage_variation(t_line, amplitude_line, slope=new_slope)

    diag.update({
        "initial_gaussian_variation_percent": float(gaussian_variation),
        "initial_gauss_z": float(gauss_z),
        "initial_distance_ms": float(distance * 1000.0) if distance is not None else np.nan,
        "initial_freq_correction_hz": float(freq_correction) if freq_correction is not None else np.nan,
        "max_amplitude": float(max_amplitude),
        "amplitude_ratio": float(gauss_z / max_amplitude) if max_amplitude else np.nan,
    })

    if distance is None:
        diag["reason"] = "gaussian_distance_none"
        return None, diag

    max_coordinate = base.point_along_line(new_point, new_slope, distance)
    if max_coordinate is None:
        diag["reason"] = "initial_gaussian_coordinate_none"
        return None, diag

    gauss_y, gauss_x = max_coordinate
    limit_gauss_x = gauss_x + 1 if left_right == 0 else gauss_x - 1
    i = 0
    angle = 0
    previous_coordinate = None
    internal_break_reason = None

    while (
        gaussian_variation < 30
        and abs(distance) < 0.00069
        and gauss_z > max_amplitude / 20
        and i < max_iterations
        and abs(freq_correction) < 1300
        and angle < 50
        and (previous_coordinate is None or max_coordinate != previous_coordinate or i < 2)
        and ((left_right == 0 and gauss_x < min_previous_x) or (left_right == 1 and gauss_x > max_previous_x))
        and (
            (left_right == 0 and gauss_x <= limit_gauss_x + 0.00002)
            or (left_right == 1 and gauss_x >= limit_gauss_x - 0.00002)
        )
        and gauss_x > 0
        and gauss_x < 0.015
    ):
        i += 1
        previous_coordinate = max_coordinate

        if (gauss_x < limit_gauss_x and left_right == 0) or (gauss_x > limit_gauss_x and left_right == 1):
            limit_gauss_x = gauss_x

        max_coordinate = base.point_along_line(new_point, new_slope, distance)
        if max_coordinate is None:
            internal_break_reason = "gaussian_coordinate_none"
            break

        gauss_y, gauss_x = max_coordinate
        edge_x = float(seven_points[0][0] if left_right == 0 else seven_points[-1][0])
        made_progress = (
            gauss_x < edge_x - min_time_progress
            if left_right == 0
            else gauss_x > edge_x + min_time_progress
        )
        if not made_progress:
            internal_break_reason = "no_monotonic_time_progress"
            break

        # Keep exactly the same payload as production process_side.
        new_result = [
            gauss_x,
            gauss_y,
            gauss_z,
            gaussian_variation,
            distance,
            gaussian_function[0],
            gaussian_function[1],
            gaussian_function[2],
            new_slope,
        ]
        seven_points = base.add_new_point_to_results(seven_points, new_result, LeftRight=left_right)

        new_point, new_slope = base.get_extrapolated_points(
            seven_points, LeftRight=left_right, nb_point_to_consider=20
        )
        if new_point is None or new_slope is None:
            internal_break_reason = "extrapolation_failed_after_progress"
            break

        t_line, amplitude_line = base.sample_line_from_max_amp_dynamic(
            spectro, freqs, times, new_point, sr, slope=new_slope
        )
        if t_line is None or amplitude_line is None:
            internal_break_reason = "sample_line_failed_after_progress"
            break
        if t_line.size < 10:
            internal_break_reason = "sample_line_too_short_after_progress"
            break

        (
            gaussian_variation,
            gauss_z,
            distance,
            gaussian_function,
            freq_correction,
        ) = base.calculate_percentage_variation(t_line, amplitude_line, slope=new_slope)
        angle = base.calculate_angle_between_lines(seven_points, left_right)

    diag["iterations"] = int(i)
    diag["points_final"] = int(len(seven_points))
    diag["final_gaussian_variation_percent"] = float(gaussian_variation)
    diag["final_gauss_z"] = float(gauss_z)
    diag["final_distance_ms"] = float(distance * 1000.0) if distance is not None else np.nan
    diag["final_freq_correction_hz"] = float(freq_correction) if freq_correction is not None else np.nan
    diag["final_angle_deg"] = float(angle)
    diag["final_amplitude_ratio"] = float(gauss_z / max_amplitude) if max_amplitude else np.nan

    if i >= max_iterations:
        diag["reason"] = "max_iterations"
        return None, diag

    if internal_break_reason is not None:
        # These break conditions are not failures in production: process_side
        # returns the partial curve. Record the reason, but preserve success.
        diag["success"] = True
        diag["reason"] = internal_break_reason
        return seven_points, diag

    diag["success"] = True
    diag["reason"] = _stop_reason(
        gaussian_variation_percent=gaussian_variation,
        distance_from_max_gauss=distance,
        gauss_z_value=gauss_z,
        max_amplitude=max_amplitude,
        freq_from_max_gauss=freq_correction,
        angle_of_trend=angle,
        gauss_x_value=gauss_x,
        left_right=left_right,
        min_previous_x=min_previous_x,
        max_previous_x=max_previous_x,
        limit_gauss_x_value=limit_gauss_x,
        previous_coordinate=previous_coordinate,
        current_coordinate=max_coordinate,
        iteration=i,
        max_iterations=max_iterations,
    )
    return seven_points, diag


def process_full_spectrum_diagnostic(y_use, sr, time_mid, duration):
    """Run the modelling pipeline and return ``(curve, diagnostics)``.

    The diagnostic clone focuses on the stages that can make the production
    function return ``None``. Extension/smoothing is delegated to the current
    production helpers once the initial left/right tracking has succeeded.
    """
    diagnostics: dict[str, Any] = {
        "success": False,
        "failure_stage": None,
        "failure_reason": None,
        "time_mid_s": float(time_mid),
        "duration_s": float(duration),
    }

    y_chunk = base.Extract_chunk_of_audio(y_use, sr, time_mid)
    diagnostics["chunk_samples"] = int(len(y_chunk))
    diagnostics["chunk_duration_ms"] = float(len(y_chunk) / sr * 1000.0)

    curve_all = base.initial_call_trend(y_chunk, sr, duration)
    if curve_all is None or len(curve_all) == 0:
        diagnostics.update({
            "failure_stage": "initial_call_trend",
            "failure_reason": "initial_curve_empty",
        })
        return None, diagnostics

    diagnostics["initial_points"] = int(len(curve_all))
    max_value = max(entry[2] for entry in curve_all)
    diagnostics["initial_max_amplitude"] = float(max_value)
    if max_value < 0.7:
        diagnostics.update({
            "failure_stage": "initial_amplitude_gate",
            "failure_reason": "max_amplitude_below_0_7",
        })
        return None, diagnostics

    S = np.abs(librosa.stft(y_chunk, n_fft=1024, hop_length=100, window="flattop"))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=100)

    curve_all, left_diag = process_side_diagnostic(
        curve_all, S, freqs, times, sr, 0, max_value
    )
    diagnostics["left"] = left_diag
    if curve_all is None:
        diagnostics.update({
            "failure_stage": "process_side_left",
            "failure_reason": left_diag.get("reason"),
        })
        return None, diagnostics

    curve_all, right_diag = process_side_diagnostic(
        curve_all, S, freqs, times, sr, 1, max_value
    )
    diagnostics["right"] = right_diag
    if curve_all is None:
        diagnostics.update({
            "failure_stage": "process_side_right",
            "failure_reason": right_diag.get("reason"),
        })
        return None, diagnostics

    # From here onward production process_full_spectrum has no intentional
    # failure return until the final curve. Reuse the same extension logic.
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

    if curve_all is None:
        diagnostics.update({
            "failure_stage": "post_extension",
            "failure_reason": "curve_none_after_extensions",
        })
        return None, diagnostics

    try:
        curve_all = base.interpolate_trend_results(curve_all)
        curve_all = base.fit_spline_with_smoothness(curve_all)
        curve_all = base.sample_curve_equally(curve_all)
    except Exception as exc:
        diagnostics.update({
            "failure_stage": "final_curve_processing",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        })
        return None, diagnostics

    diagnostics.update({
        "success": True,
        "failure_stage": "",
        "failure_reason": "",
        "final_points": int(len(curve_all)),
    })
    return curve_all, diagnostics


def process_full_spectrum(y_use, sr, time_mid, duration):
    """Compatibility entry point returning only the curve."""
    curve, _ = process_full_spectrum_diagnostic(y_use, sr, time_mid, duration)
    return curve
