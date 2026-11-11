"""SNR-aware comparison of predicted chirps to independent manual ridges.

The reference SNR is computed on a fixed, ordinary Flattop STFT. It is never
computed on, or fed back into, an automatic candidate's trajectory. This keeps
the benchmark independent of changes in reassigned tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import stft


def tolerance_khz(snr_db):
    """Linear 5 kHz (<=15 dB) to 3 kHz (>=35 dB) tolerance, clipped."""
    values = np.asarray(snr_db, dtype=float)
    return np.interp(values, [15.0, 35.0], [5.0, 3.0])


def detection_importance(snr_max_db, *, min_weight=0.2):
    """Importance of finding a manual call, weighted by its peak reference SNR.

    A quiet call retains a nonzero contribution so the detector is never
    rewarded for ignoring low-SNR data altogether. Default endpoints:
    <=15 dB: 0.2, >=35 dB: 1.0; linear interpolation between them.
    """
    if not 0 < min_weight <= 1:
        raise ValueError("min_weight must be in (0, 1]")
    return np.interp(np.asarray(snr_max_db, dtype=float),
                     [15.0, 35.0], [min_weight, 1.0])


@dataclass
class ReferenceSNR:
    """A fixed SNR image, using the same units for every benchmark run."""

    freqs_hz: np.ndarray
    times_s: np.ndarray
    snr_db: np.ndarray

    @classmethod
    def from_audio(cls, y, sample_rate, *, n_fft=512, hop=64,
                   noise_percentile=50.0):
        y = np.asarray(y)
        if y.ndim != 1 or len(y) < n_fft + hop or not np.isfinite(y).all():
            raise ValueError("reference audio needs >= n_fft + hop finite mono samples")
        if sample_rate <= 0 or not 1 <= hop <= n_fft:
            raise ValueError("invalid sample rate or STFT hop")
        freqs, times, z = stft(
            y, fs=sample_rate, window="flattop", nperseg=n_fft,
            noverlap=n_fft - hop, boundary=None, padded=False,
        )
        power_db = 20 * np.log10(np.maximum(np.abs(z), 1e-15))
        noise_db = np.percentile(power_db, noise_percentile, axis=1)
        return cls(freqs, times, (power_db - noise_db[:, None]).astype(np.float32))

    def along_reference(self, times_s, freqs_hz):
        """Sample local SNR at the manual f(t), allowing +/- one FFT bin.

        Bilinear interpolation avoids nearest-frame errors on very steep
        frequency sweeps. Frequency spreading provides modest robustness to
        the manual curve's interpolation/simplification uncertainty.
        """
        t = np.asarray(times_s, dtype=float)
        f = np.asarray(freqs_hz, dtype=float)
        if t.shape != f.shape or not np.isfinite(t).all() or not np.isfinite(f).all():
            raise ValueError("time/frequency samples must be finite and aligned")
        if np.any(f < 0) or np.any(f > self.freqs_hz[-1]):
            raise ValueError("reference frequency outside Nyquist")
        fi = f / (self.freqs_hz[1] - self.freqs_hz[0])
        ti = (t - self.times_s[0]) / (self.times_s[1] - self.times_s[0])
        samples = [
            map_coordinates(self.snr_db, np.stack((fi + offset, ti)),
                            order=1, mode="nearest")
            for offset in (-1, 0, 1)
        ]
        return np.maximum.reduce(samples)


def compare_trajectory(reference, t_start_s, t_end_s, predicted_curve,
                       reference_snr: ReferenceSNR, *, grid_points=50,
                       geometry_mode="vertical_legacy", manual_points=None):
    """Score a prediction using explicit vertical or square-normal geometry.

    Input times are seconds and frequencies Hz. "vertical_legacy" preserves
    historical frequency errors. "square_normals_v1" uses the external
    canonical chirp_geometry package and requires manual_points (s, Hz).
    The vertical errors remain separate diagnostics even in square mode.
    Uncovered normals count as failures; extrapolated vertical errors
    remain available for historical comparisons.
    """
    if grid_points < 2 or t_end_s <= t_start_s:
        raise ValueError("invalid benchmark time grid")
    if geometry_mode not in ("vertical_legacy", "square_normals_v1"):
        raise ValueError("unknown geometry_mode")
    if geometry_mode == "square_normals_v1":
        if manual_points is None:
            raise ValueError("square_normals_v1 requires manual_points in (s, Hz)")
        try:
            from chirp_geometry import PROTOCOL, compare_normal_curves
        except ImportError as exc:
            raise ImportError(
                "square_normals_v1 requires chirp_geometry; install the sibling "
                "Bat-Chirp-Laboratory via 'pip install -e ../Bat-Chirp-Laboratory'"
            ) from exc
        if PROTOCOL != geometry_mode:
            raise ValueError("incompatible chirp_geometry protocol")
    grid = np.linspace(t_start_s, t_end_s, grid_points)
    truth = np.asarray(reference(grid), dtype=float)
    local_snr = reference_snr.along_reference(grid, truth)
    tolerance = tolerance_khz(local_snr)
    result = {
        "snr_max_db": float(np.max(local_snr)),
        "snr_p90_db": float(np.percentile(local_snr, 90)),
        "snr_median_db": float(np.median(local_snr)),
        "tolerance_min_khz": float(np.min(tolerance)),
        "tolerance_max_khz": float(np.max(tolerance)),
        "scoring_protocol": geometry_mode,
        "tolerance_units": ("kHz_eq" if geometry_mode == "square_normals_v1"
                            else "kHz_vertical"),
    }
    if predicted_curve is None:
        return result
    curve = np.asarray(predicted_curve, dtype=float)
    if (curve.ndim != 2 or curve.shape[1] < 2 or len(curve) < 2
            or not np.isfinite(curve[:, :2]).all()):
        raise ValueError("predicted curve must have >=2 finite (time, frequency) rows")
    curve = curve[np.argsort(curve[:, 0], kind="stable")]
    time, unique_idx = np.unique(curve[:, 0], return_index=True)
    if len(time) < 2:
        raise ValueError("predicted curve needs distinct times")
    freq = curve[unique_idx, 1]
    error = np.abs(np.interp(grid, time, freq) - truth) / 1000.0
    vertical_normalized = error / tolerance
    overlap = max(0., min(time[-1], t_end_s) - max(time[0], t_start_s))
    result.update({
        # Historical vertical-frequency diagnostics, never relabel as geometry.
        "mean_full_interval_error_khz": float(np.mean(error)),
        "p90_error_khz": float(np.percentile(error, 90)),
        "vertical_mean_normalized_error": float(np.mean(vertical_normalized)),
        "vertical_within_snr_tolerance_fraction": float(np.mean(vertical_normalized <= 1.0)),
        "coverage_fraction": float(overlap / (t_end_s - t_start_s)),
        "start_error_ms": float(abs(time[0] - t_start_s) * 1000),
        "end_error_ms": float(abs(time[-1] - t_end_s) * 1000),
    })
    if geometry_mode == "vertical_legacy":
        result.update({
            "mean_normalized_error": float(np.mean(vertical_normalized)),
            "p90_normalized_error": float(np.percentile(vertical_normalized, 90)),
            "within_snr_tolerance_fraction": float(np.mean(vertical_normalized <= 1.0)),
        })
        return result

    manual = np.asarray(manual_points, dtype=float)
    if (manual.ndim != 2 or manual.shape[1] != 2 or len(manual) < 2
            or not np.isfinite(manual).all()
            or np.any(np.diff(manual[:, 0]) <= 0)):
        raise ValueError("manual_points must be ordered finite (s, Hz) rows")
    geometry = compare_normal_curves(
        manual[:, 0] * 1000, manual[:, 1] / 1000,
        time * 1000, freq / 1000, grid * 1000)
    covered = geometry["covered"]
    distances = geometry["abs_normal_khz_eq"]
    normalized = np.full(len(grid), 2.0)
    normalized[covered] = distances[covered] / tolerance[covered]
    # Uncovered normals contribute zero to the within-tolerance fraction.
    within = float(np.mean(covered & (normalized <= 1.0)))
    result.update({
        "mean_geometric_error_khz_eq": (float(np.mean(distances[covered]))
                                        if np.any(covered) else None),
        "p90_geometric_error_khz_eq": (float(np.percentile(distances[covered], 90))
                                       if np.any(covered) else None),
        "geometric_coverage_fraction": geometry["coverage_fraction"],
        "normalization_mode": geometry["normalization_mode"],
        "time_scale_khz_per_ms": geometry["time_scale_khz_per_ms"],
        "geometric_uncovered_penalty_multiple": 2.0,
        "mean_normalized_error": float(np.mean(normalized)),
        "p90_normalized_error": float(np.percentile(normalized, 90)),
        "within_snr_tolerance_fraction": within,
        "within_geometric_tolerance_fraction": within,
    })
    return result
