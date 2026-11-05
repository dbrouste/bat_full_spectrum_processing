"""Conservative confidence-based stopping for reassigned FM ridge tails.

Trimming only removes observed points at the far right whose STFT evidence
has collapsed; it never extrapolates, moves frequencies, or changes the
high-confidence interior of a trajectory.
"""
from __future__ import annotations
import numpy as np


def trim_weak_right_tail(
    curve, *, times_s, freqs_hz, stft_power, snr_db,
    min_snr_db=10.0, min_relative_db=-22.0,
    min_weak_points=4, min_keep_points=8, max_trim_fraction=0.45,
    return_diagnostics=False,
):
    """Remove a sustained low-confidence suffix, preserving real FM endpoints.

    A point has sufficient support only if its local STFT SNR exceeds the
    threshold *and* its spectral power is not excessively lower than the
    strongest power seen along this particular trajectory. Using both
    prevents narrow, high-SNR noise islands far after the real chirp from
    being accepted merely because the noise baseline is extremely low.
    """
    x = np.asarray(curve, dtype=float)
    t = np.asarray(times_s, dtype=float)
    f = np.asarray(freqs_hz, dtype=float)
    power = np.asarray(stft_power, dtype=float)
    snr = np.asarray(snr_db, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 3 or len(x) < 2
            or not np.isfinite(x).all()
            or not np.all(np.diff(x[:, 0]) > 0)):
        raise ValueError("curve must be a finite, time-sorted Nx3 matrix")
    if (t.ndim != 1 or f.ndim != 1 or len(t) < 1 or len(f) < 1
            or power.shape != (len(f), len(t)) or snr.shape != power.shape):
        raise ValueError("STFT grids and matrices have inconsistent shapes")
    if (not np.isfinite(power).all() or not np.isfinite(snr).all()
            or np.any(power < 0)):
        raise ValueError("STFT power and SNR must be finite and valid")
    if (min_snr_db < 0 or min_relative_db >= 0
            or min_weak_points < 1 or min_keep_points < 2
            or not 0 < max_trim_fraction < 1):
        raise ValueError("invalid right-tail confidence thresholds")
    diag = {
        "trimmed_right_points": 0,
        "trailing_weak_points": 0,
        "min_snr_db": float(min_snr_db),
        "min_relative_db": float(min_relative_db),
    }
    if len(x) < max(12, min_keep_points + min_weak_points):
        return (x.copy(), diag) if return_diagnostics else x.copy()

    def nearest_indices(grid, positions):
        idx = np.searchsorted(grid, positions)
        idx = np.clip(idx, 0, len(grid) - 1)
        previous = np.maximum(0, idx - 1)
        return np.where(
            np.abs(grid[previous] - positions)
            < np.abs(grid[idx] - positions), previous, idx
        )

    rows = nearest_indices(f, x[:, 1])
    cols = nearest_indices(t, x[:, 0])
    local_power = np.empty(len(x), dtype=float)
    local_snr = np.empty(len(x), dtype=float)
    for k, (ri, ci) in enumerate(zip(rows, cols)):
        near = slice(max(ri - 1, 0), min(ri + 2, len(f)))
        local_power[k] = float(np.max(power[near, ci]))
        local_snr[k] = float(np.max(snr[near, ci]))
    path_peak = max(float(np.max(local_power)), 1e-30)
    relative_db = 10 * np.log10(np.maximum(local_power, 1e-30) / path_peak)
    supported = (local_snr >= min_snr_db) & (relative_db >= min_relative_db)
    weak_suffix = 0
    for ok in supported[::-1]:
        if ok:
            break
        weak_suffix += 1
    diag["trailing_weak_points"] = int(weak_suffix)
    if weak_suffix < min_weak_points:
        return (x.copy(), diag) if return_diagnostics else x.copy()
    keep = max(
        len(x) - weak_suffix,
        min_keep_points,
        int(np.ceil(len(x) * (1.0 - max_trim_fraction))),
    )
    diag["trimmed_right_points"] = len(x) - keep
    result = x[:keep].copy()
    return (result, diag) if return_diagnostics else result
