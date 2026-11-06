"""Conservative correction of spuriously horizontal chirp tails.

Frequency-only correction of observed points. Never invents temporal coverage
or extrapolates beyond the actual tracked call.
"""
from __future__ import annotations
import numpy as np


def refine_ridge_endpoint(curve, *, tail_fraction=0.35,
                          max_correction_hz=16000.,
                          times_s=None, freqs_hz=None, stft_power=None,
                          spectral_support_min_ratio=0.5,
                          return_diagnostics=False):
    """Limit late FM plateaus to the earlier decline's linear continuation.

    The continuation is anchored to the last observation preceding the tail,
    preventing a discontinuity at the splice. Corrections only decrease
    frequency and cannot exceed max_correction_hz at any one point. If the
    local STFT power is supplied, a correction is accepted only where the
    proposed frequency has comparable observed spectral support.
    """
    source = np.asarray(curve, dtype=float)
    if source.ndim != 2 or source.shape[1] < 3 or not np.isfinite(source).all():
        raise ValueError('curve must be finite N x 3 [time, frequency, amplitude]')
    if (not 0 < tail_fraction < 0.8 or max_correction_hz < 0
            or not 0 < spectral_support_min_ratio <= 1):
        raise ValueError('invalid tail correction parameters')
    has_spectral_support = any(v is not None for v in (times_s, freqs_hz, stft_power))
    if has_spectral_support:
        if any(v is None for v in (times_s, freqs_hz, stft_power)):
            raise ValueError('times_s, freqs_hz and stft_power must be supplied together')
        times_s = np.asarray(times_s, dtype=float)
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        stft_power = np.asarray(stft_power, dtype=float)
        if (times_s.ndim != 1 or freqs_hz.ndim != 1
                or stft_power.shape != (len(freqs_hz), len(times_s))
                or not np.isfinite(stft_power).all()
                or np.any(stft_power < 0)
                or len(times_s) < 2 or len(freqs_hz) < 2
                or not np.all(np.diff(times_s) > 0)
                or not np.all(np.diff(freqs_hz) > 0)):
            raise ValueError('invalid endpoint STFT support grid')
    out = source.copy()
    diagnostics = {'corrected_points': 0, 'max_correction_hz': 0.,
                   'start_amplitude': None, 'end_amplitude': None,
                   'endpoint_amplitude_ratio': None,
                   'spectral_support_gated': bool(has_spectral_support),
                   'rejected_corrections': 0}
    if len(out) >= 2:
        diagnostics['start_amplitude'] = float(out[0, 2])
        diagnostics['end_amplitude'] = float(out[-1, 2])
        diagnostics['endpoint_amplitude_ratio'] = float(
            out[-1, 2] / max(float(np.max(out[:, 2])), 1e-12))
    if len(out) < 12 or not np.all(np.diff(out[:, 0]) > 0):
        return (out, diagnostics) if return_diagnostics else out
    t = out[:, 0]
    f = out[:, 1]
    tspan = t[-1] - t[0]
    if tspan <= 0:
        return (out, diagnostics) if return_diagnostics else out
    x = (t - t[0]) / tspan
    edge_start = max(5, min(len(out) - 5, int(len(out) * (1 - tail_fraction))))
    slope = float(np.polyfit(x[:edge_start], f[:edge_start], 1)[0])
    if slope >= 0:  # Do not force a CF or upward chirp to be FM downward.
        return (out, diagnostics) if return_diagnostics else out
    expected = f[edge_start - 1] + slope * (x[edge_start:] - x[edge_start - 1])
    adjustment = np.minimum(
        np.maximum(f[edge_start:] - expected, 0.0), max_correction_hz)
    if has_spectral_support:
        def nearest(grid, value):
            idx = np.clip(np.searchsorted(grid, value), 0, len(grid) - 1)
            previous = np.maximum(0, idx - 1)
            return np.where(
                np.abs(grid[previous] - value) < np.abs(grid[idx] - value),
                previous, idx,
            )

        cols = nearest(times_s, t[edge_start:])
        original_rows = nearest(freqs_hz, f[edge_start:])
        proposed_rows = nearest(freqs_hz, f[edge_start:] - adjustment)
        for j, (ci, ri, pi) in enumerate(zip(cols, original_rows, proposed_rows)):
            if adjustment[j] <= 0:
                continue
            original = np.max(stft_power[max(0, ri - 1):min(len(freqs_hz), ri + 2), ci])
            proposed = np.max(stft_power[max(0, pi - 1):min(len(freqs_hz), pi + 2), ci])
            if proposed < spectral_support_min_ratio * max(original, 1e-30):
                adjustment[j] = 0.0
                diagnostics['rejected_corrections'] += 1
    out[edge_start:, 1] = f[edge_start:] - adjustment
    diagnostics['corrected_points'] = int(np.count_nonzero(adjustment > 0))
    diagnostics['max_correction_hz'] = float(np.max(adjustment))
    return (out, diagnostics) if return_diagnostics else out
