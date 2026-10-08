"""Chirp modelling entry point used by the benchmark and future pipeline.

The historical absolute-amplitude gate is intentionally removed.  When the
detector provides a peak frequency, modelling can additionally seed the initial
ridge near that band so simultaneous fundamentals/harmonics do not all lock to
the strongest spectral component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import librosa
import numpy as np

from bat_analysis import bfsp_clean_patched as base
from bat_analysis import detection as detection_api

high_pass_filter = base.high_pass_filter
detect_candidates_snr_blobs = detection_api.detect_candidates_snr_blobs


@dataclass
class ChirpModelResult:
    model_name: str
    parameters: Dict[str, float]
    diagnostics: Dict[str, Any]


def _initial_call_trend_seeded(
    yt,
    sr,
    duration,
    seed_freq_hz: float,
    *,
    seed_half_band_hz: float = 8000.0,
    step_ns: float = 50.0,
    seed_slope_hz_per_ms: float | None = None,
):
    """Build the legacy seven-point initial trend near a detector frequency.

    Only selection of the initial maximum is changed. Gaussian line fitting and
    all subsequent ridge processing are the legacy implementation.
    """
    sp = np.abs(librosa.stft(yt, n_fft=1024, hop_length=100, window="flattop"))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    times = librosa.frames_to_time(np.arange(sp.shape[1]), sr=sr, hop_length=100)
    if sp.size == 0 or len(times) == 0:
        return None

    center_s = (len(yt) / sr) / 2.0
    duration = max(float(duration), 1e-6)
    half_time_s = max(0.0015, min(duration / 2.0, 0.004))

    fmask = (freqs >= float(seed_freq_hz) - seed_half_band_hz) & (
        freqs <= float(seed_freq_hz) + seed_half_band_hz
    )
    tmask = (times >= center_s - half_time_s) & (times <= center_s + half_time_s)
    fi = np.where(fmask)[0]
    ti = np.where(tmask)[0]
    if fi.size == 0 or ti.size == 0:
        return base.initial_call_trend(yt, sr, duration, step_ns=step_ns)

    local = sp[np.ix_(fi, ti)]
    f_rel, t_rel = np.unravel_index(int(np.argmax(local)), local.shape)
    max_freq = float(freqs[fi[f_rel]])
    max_time = float(times[ti[t_rel]])

    step = float(step_ns) / 1_000_000.0
    candidate_times = max_time + np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=float) * step

    sample_slope = 1.0
    candidate_freqs = np.full_like(candidate_times, max_freq)
    if (
        seed_slope_hz_per_ms is not None
        and np.isfinite(seed_slope_hz_per_ms)
        and abs(float(seed_slope_hz_per_ms)) > 100.0
    ):
        chirp_slope_normalized = (
            float(seed_slope_hz_per_ms) * 1000.0 / 3730000.0
        )
        if abs(chirp_slope_normalized) > 1e-9:
            sample_slope = -1.0 / chirp_slope_normalized
            candidate_freqs = max_freq + (
                candidate_times - max_time
            ) * float(seed_slope_hz_per_ms) * 1000.0

    results = np.zeros((len(candidate_times), 9), dtype=float)

    for i, (candidate_time, candidate_freq) in enumerate(zip(candidate_times, candidate_freqs)):
        candidate_point = (float(candidate_freq), float(candidate_time))
        try:
            t_line, amplitude_line = base.sample_line_from_max_amp_dynamic(
                sp, freqs, times, candidate_point, sr, slope=sample_slope
            )
            variation, max_gauss_z, max_dist, popt, _ = base.calculate_percentage_variation(
                t_line, amplitude_line, slope=sample_slope
            )
            max_gauss_freq, max_gauss_time = base.point_along_line(
                candidate_point, sample_slope, max_dist
            )
            if (
                popt is not None
                and len(popt) >= 3
                and None not in (
                    max_gauss_time,
                    max_gauss_freq,
                    max_gauss_z,
                    variation,
                    max_dist,
                )
            ):
                results[i] = [
                    max_gauss_time,
                    max_gauss_freq,
                    max_gauss_z,
                    variation,
                    max_dist,
                    popt[0],
                    popt[1],
                    popt[2],
                    sample_slope,
                ]
        except Exception:
            # Preserve the legacy convention: an invalid candidate remains a
            # zero row and is handled by downstream validation/tracking.
            continue

    return results



def _adaptive_tracking_step_us(duration_s: float, seed_freq_hz: float | None) -> float:
    """Choose a conservative ridge-tracking step from call morphology.

    Historical tracking used a fixed 40 us step. Diagnostics on the grouped
    validation set showed that long, shallow low-frequency calls often hit the
    iteration guard, while short/steep calls were already well behaved.

    The adaptive rule is intentionally conservative and bounded to 40-60 us:
    - <=4 ms: 40 us
    - 4-8 ms: linear ramp from 40 to 60 us
    - >=8 ms: 60 us
    - low-frequency calls (<40 kHz) of at least 5 ms use at least 60 us
    """
    duration_ms = max(0.0, float(duration_s)) * 1000.0
    if duration_ms <= 4.0:
        step = 40.0
    elif duration_ms >= 8.0:
        step = 60.0
    else:
        step = 40.0 + 5.0 * (duration_ms - 4.0)

    if (
        seed_freq_hz is not None
        and np.isfinite(seed_freq_hz)
        and float(seed_freq_hz) < 40000.0
        and duration_ms >= 5.0
    ):
        step = max(step, 60.0)

    return float(np.clip(step, 40.0, 60.0))

def process_full_spectrum(
    y_use,
    sr,
    time_mid,
    duration,
    seed_freq_hz: float | None = None,
    broad_fm_seed_freq_hz: float | None = None,
    broad_fm_seed_slope_hz_per_ms: float | None = None,
    diagnostics: dict | None = None,
    ridge_mode: str = "default",
):
    """Model a detected chirp without the historical 0.7 amplitude gate.

    If ``seed_freq_hz`` is supplied, the initial ridge maximum is searched only
    near that detected band (default +/-8 kHz). This prevents simultaneous
    frequency-separated calls/harmonics from all being initialized on the
    globally strongest component. Omitting it preserves the previous no-gate
    behaviour.
    """
    if ridge_mode not in {"default", "peak", "coarse"}:
        raise ValueError("ridge_mode must be 'default', 'peak', or 'coarse'")
    if ridge_mode == "peak":
        broad_fm_seed_freq_hz = None
        broad_fm_seed_slope_hz_per_ms = None
    elif ridge_mode == "coarse":
        if broad_fm_seed_freq_hz is None or not np.isfinite(broad_fm_seed_freq_hz):
            raise ValueError("coarse mode requires a finite broad_fm_seed_freq_hz")

    y_chun = base.Extract_chunk_of_audio(y_use, sr, time_mid)

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "side_events": [],
            "extension_events": [],
            "input_time_mid_s": float(time_mid),
            "input_duration_s": float(duration),
            "seed_freq_hz": (
                float(seed_freq_hz)
                if seed_freq_hz is not None and np.isfinite(seed_freq_hz)
                else None
            ),
            "broad_fm_seed_freq_hz": (
                float(broad_fm_seed_freq_hz)
                if broad_fm_seed_freq_hz is not None and np.isfinite(broad_fm_seed_freq_hz)
                else None
            ),
            "broad_fm_seed_slope_hz_per_ms": (
                float(broad_fm_seed_slope_hz_per_ms)
                if broad_fm_seed_slope_hz_per_ms is not None
                and np.isfinite(broad_fm_seed_slope_hz_per_ms)
                else None
            ),
        })

    effective_seed_freq_hz = seed_freq_hz
    if broad_fm_seed_freq_hz is not None and np.isfinite(broad_fm_seed_freq_hz):
        effective_seed_freq_hz = float(broad_fm_seed_freq_hz)

    tracking_step_us = _adaptive_tracking_step_us(duration, effective_seed_freq_hz)
    if diagnostics is not None:
        diagnostics["effective_seed_freq_hz"] = (
            float(effective_seed_freq_hz)
            if effective_seed_freq_hz is not None and np.isfinite(effective_seed_freq_hz)
            else None
        )
        diagnostics["tracking_step_us"] = float(tracking_step_us)

    if effective_seed_freq_hz is None or not np.isfinite(effective_seed_freq_hz):
        curve_all = base.initial_call_trend(y_chun, sr, duration)
    else:
        curve_all = _initial_call_trend_seeded(
            y_chun,
            sr,
            duration,
            float(effective_seed_freq_hz),
            seed_slope_hz_per_ms=(
                broad_fm_seed_slope_hz_per_ms
                if broad_fm_seed_freq_hz is not None
                and np.isfinite(broad_fm_seed_freq_hz)
                else None
            ),
        )
    if curve_all is None or len(curve_all) == 0:
        return None

    max_value = max(entry[2] for entry in curve_all)

    s = np.abs(librosa.stft(y_chun, n_fft=1024, hop_length=100, window="flattop"))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    times = librosa.frames_to_time(np.arange(s.shape[1]), sr=sr, hop_length=100)

    curve_all = base.process_side(
        curve_all, s, freqs, times, sr, LeftRight=0, max_amplitude=max_value,
        tracking_step_us=tracking_step_us,
        diagnostics=(diagnostics["side_events"] if diagnostics is not None else None),
        diagnostic_label="initial_left",
    )
    if curve_all is None:
        return None

    curve_all = base.process_side(
        curve_all, s, freqs, times, sr, LeftRight=1, max_amplitude=max_value,
        tracking_step_us=tracking_step_us,
        diagnostics=(diagnostics["side_events"] if diagnostics is not None else None),
        diagnostic_label="initial_right",
    )
    if curve_all is None:
        return None

    curve_segmented = base.extend_trend_left(
        y_chun, sr, curve_all, s, freqs, times, max_value,
        tracking_step_us=tracking_step_us,
        diagnostics=(diagnostics["side_events"] if diagnostics is not None else None),
        diagnostic_prefix="extend_left_0",
    )
    if curve_segmented is not None:
        max_segment = max(entry[2] for entry in curve_segmented)
        while (
            base.calculate_angle_between_trends(curve_all, curve_segmented) < 70
            and base.get_number_of_points(curve_segmented) > 11
            and max_segment > max_value / 20
        ):
            curve_all = base.concatenate_trends(curve_segmented, curve_all)
            curve_segmented = base.extend_trend_left(
                y_chun, sr, curve_all, s, freqs, times, max_value,
                tracking_step_us=tracking_step_us,
                diagnostics=(diagnostics["side_events"] if diagnostics is not None else None),
                diagnostic_prefix=f"extend_left_{len(diagnostics['extension_events']) + 1 if diagnostics is not None else 1}",
            )
            if curve_segmented is None:
                break
            max_segment = max(entry[2] for entry in curve_segmented)

    curve_segmented = base.extend_trend_right(
        y_chun, sr, curve_all, s, freqs, times, max_value,
        tracking_step_us=tracking_step_us,
        diagnostics=(diagnostics["side_events"] if diagnostics is not None else None),
        diagnostic_prefix="extend_right_0",
    )
    if curve_segmented is not None:
        max_segment = max(entry[2] for entry in curve_segmented)
        while (
            base.calculate_angle_between_trends(curve_all, curve_segmented) < 50
            and base.get_number_of_points(curve_segmented) > 11
            and base.validate_slope(curve_segmented, nb_points=25, LeftRight=0)
            and base.get_max_frequency(curve_segmented) < (base.get_min_frequency(curve_all) + 500)
            and max_segment > max_value / 10
        ):
            curve_all = base.concatenate_trends(curve_all, curve_segmented)
            curve_segmented = base.extend_trend_right(
                y_chun, sr, curve_all, s, freqs, times, max_value,
                tracking_step_us=tracking_step_us,
                diagnostics=(diagnostics["side_events"] if diagnostics is not None else None),
                diagnostic_prefix=f"extend_right_{len(diagnostics['extension_events']) + 1 if diagnostics is not None else 1}",
            )
            if curve_segmented is None:
                break
            max_segment = max(entry[2] for entry in curve_segmented)

    curve_all = base.interpolate_trend_results(curve_all)
    curve_all = base.fit_spline_with_smoothness(curve_all)
    curve_all = base.sample_curve_equally(curve_all)

    if diagnostics is not None:
        diagnostics["model_success"] = curve_all is not None
        diagnostics["final_points"] = int(len(curve_all)) if curve_all is not None else 0
        reasons = [
            event.get("stop_reason")
            for event in diagnostics.get("side_events", [])
            if event.get("stop_reason")
        ]
        diagnostics["stop_reason_counts"] = {
            reason: int(reasons.count(reason)) for reason in sorted(set(reasons))
        }

    return curve_all


def compare_ridge_hypotheses(
    y_use, sr, time_mid, duration, *, seed_freq_hz,
    coarse_seed_freq_hz, coarse_seed_slope_hz_per_ms=None,
):
    """Run independent seed hypotheses; do not automatically choose a winner.

    Returned curves use the same time/frequency coordinates as
    process_full_spectrum. Ground-truth-free quality selection is intentionally
    left out until a benchmark establishes a reliable criterion.
    """
    results = {}
    for mode in ("peak", "coarse"):
        details = {}
        try:
            curve = process_full_spectrum(
                y_use, sr, time_mid, duration,
                seed_freq_hz=seed_freq_hz,
                broad_fm_seed_freq_hz=coarse_seed_freq_hz,
                broad_fm_seed_slope_hz_per_ms=coarse_seed_slope_hz_per_ms,
                diagnostics=details,
                ridge_mode=mode,
            )
            results[mode] = {"curve": curve, "diagnostics": details, "error": None}
        except Exception as exc:
            results[mode] = {"curve": None, "diagnostics": details,
                             "error": f"{type(exc).__name__}: {exc}"}
    return results


def fit_chirp_model(*args, **kwargs) -> ChirpModelResult:
    raise NotImplementedError("Compact model fitting will be selected after ridge benchmarking.")
