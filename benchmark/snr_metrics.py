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
                       reference_snr: ReferenceSNR, *, grid_points=50):
    """Score full annotated duration, including a prediction's missing ends.

    Frequency units: reference and predicted_curve are in Hz, times in s.
    Extrapolated missing ends contribute an error; time coverage is reported
    separately so a short, accurate fragment cannot hide its incompleteness.
    """
    if grid_points < 2 or t_end_s <= t_start_s:
        raise ValueError("invalid benchmark time grid")
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
    normalized = error / tolerance
    overlap = max(0., min(time[-1], t_end_s) - max(time[0], t_start_s))
    result.update({
        "mean_full_interval_error_khz": float(np.mean(error)),
        "p90_error_khz": float(np.percentile(error, 90)),
        "mean_normalized_error": float(np.mean(normalized)),
        "p90_normalized_error": float(np.percentile(normalized, 90)),
        "within_snr_tolerance_fraction": float(np.mean(normalized <= 1.0)),
        "coverage_fraction": float(overlap / (t_end_s - t_start_s)),
        "start_error_ms": float(abs(time[0] - t_start_s) * 1000),
        "end_error_ms": float(abs(time[-1] - t_end_s) * 1000),
    })
    return result
