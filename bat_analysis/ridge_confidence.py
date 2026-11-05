"""Conservative confidence-based stopping for reassigned FM ridge tails.

Trimming removes unsupported right-hand points, including a late sustained
power gap followed by a misleading recovery on a different ridge. It never
extrapolates, moves frequencies, or changes the high-confidence interior.
"""
from __future__ import annotations
import numpy as np


def trim_weak_right_tail(
    curve, *, times_s, freqs_hz, stft_power, snr_db,
    min_snr_db=10.0, min_relative_db=-22.0,
    min_weak_points=4, min_keep_points=8, max_trim_fraction=0.45,
    late_gap_min_fraction=0.60,
    return_diagnostics=False,
):
    """Remove a weak suffix or a late power collapse preceding a false rejoin.

    A point has sufficient support only if its local STFT SNR exceeds the
    threshold *and* its spectral power is not excessively lower than the
    strongest power seen along this particular trajectory. Using both
    prevents narrow, high-SNR noise islands far after the real chirp from
    being accepted merely because the noise baseline is extremely low.
    Unlike a suffix-only check, a late run of weak-power frames also stops
    the curve if it subsequently joins another, seemingly strong ridge.
    Early gaps and short late gaps remain untouched.
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
            or not 0 < max_trim_fraction < 1
            or not 0 < late_gap_min_fraction < 1):
        raise ValueError("invalid right-tail confidence thresholds")
    diag = {
        "trimmed_right_points": 0,
        "trailing_weak_points": 0,
        "trim_reason": None,
        "late_gap_start_index": None,
        "late_gap_length": 0,
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
    minimum_keep = max(
        min_keep_points,
        int(np.ceil(len(x) * (1.0 - max_trim_fraction))),
    )
    keep = len(x)
    if weak_suffix >= min_weak_points:
        keep = max(len(x) - weak_suffix, minimum_keep)
        if keep < len(x):
            diag["trim_reason"] = "weak_right_suffix"

    # An apparently strong terminal ridge can hide a weak gap immediately
    # before it. The suffix-only rule misses this exact failure mode. Use
    # power, rather than SNR alone, to avoid cutting genuine calls with a
    # temporary noise-floor fluctuation. Do not touch early internal gaps.
    weak_power = relative_db < min_relative_db
    late_start = max(
        minimum_keep,
        int(np.ceil(len(x) * late_gap_min_fraction)),
        int(np.argmax(local_power)) + 1,
    )
    for pos in range(late_start, len(x) - min_weak_points):
        if (np.all(weak_power[pos:pos + min_weak_points])
                and np.any(~weak_power[pos + min_weak_points:])):
            end = pos
            while end < len(x) and weak_power[end]:
                end += 1
            diag["late_gap_start_index"] = int(pos)
            diag["late_gap_length"] = int(end - pos)
            if pos < keep:
                keep = pos
                diag["trim_reason"] = "late_power_gap"
            break

    diag["trimmed_right_points"] = len(x) - keep
    result = x[:keep].copy()
    return (result, diag) if return_diagnostics else result
