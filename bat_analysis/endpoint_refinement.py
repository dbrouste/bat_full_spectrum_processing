"""Conservative correction of spuriously horizontal chirp tails.

Frequency-only correction of observed points. Never invents temporal coverage
or extrapolates beyond the actual tracked call.
"""
from __future__ import annotations
import numpy as np


def refine_ridge_endpoint(curve, *, tail_fraction=0.35,
                          max_correction_hz=16000., return_diagnostics=False):
    """Limit late FM plateaus to the earlier decline's linear continuation.

    The continuation is anchored to the last observation preceding the tail,
    preventing a discontinuity at the splice. Corrections only decrease
    frequency and cannot exceed max_correction_hz at any one point.
    """
    source = np.asarray(curve, dtype=float)
    if source.ndim != 2 or source.shape[1] < 3 or not np.isfinite(source).all():
        raise ValueError('curve must be finite N x 3 [time, frequency, amplitude]')
    if not 0 < tail_fraction < 0.8 or max_correction_hz < 0:
        raise ValueError('invalid tail correction parameters')
    out = source.copy()
    diagnostics = {'corrected_points': 0, 'max_correction_hz': 0.,
                   'start_amplitude': None, 'end_amplitude': None,
                   'endpoint_amplitude_ratio': None}
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
    out[edge_start:, 1] = f[edge_start:] - adjustment
    diagnostics['corrected_points'] = int(np.count_nonzero(adjustment > 0))
    diagnostics['max_correction_hz'] = float(np.max(adjustment))
    return (out, diagnostics) if return_diagnostics else out
