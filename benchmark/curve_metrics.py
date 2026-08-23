from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CurveMetrics:
    mae_khz: float
    median_abs_error_khz: float
    p95_abs_error_khz: float
    rmse_khz: float
    bias_khz: float
    coverage: float
    start_error_ms: float
    end_error_ms: float


def compare_curves(
    reference_t_ms: Sequence[float],
    reference_f_khz: Sequence[float],
    predicted_t_ms: Sequence[float],
    predicted_f_khz: Sequence[float],
) -> CurveMetrics:
    """Compare predicted f(t) with manual ground truth over their common span."""
    rt = np.asarray(reference_t_ms, dtype=float)
    rf = np.asarray(reference_f_khz, dtype=float)
    pt = np.asarray(predicted_t_ms, dtype=float)
    pf = np.asarray(predicted_f_khz, dtype=float)

    bad = CurveMetrics(*(float("nan"),) * 5, 0.0, float("nan"), float("nan"))
    if rt.size < 2 or pt.size < 2:
        return bad

    rmask = np.isfinite(rt) & np.isfinite(rf)
    pmask = np.isfinite(pt) & np.isfinite(pf)
    rt, rf, pt, pf = rt[rmask], rf[rmask], pt[pmask], pf[pmask]
    if rt.size < 2 or pt.size < 2:
        return bad

    order_r = np.argsort(rt)
    order_p = np.argsort(pt)
    rt, rf = rt[order_r], rf[order_r]
    pt, pf = pt[order_p], pf[order_p]

    # Collapse duplicate time coordinates before interpolation.
    ru, ri = np.unique(rt, return_index=True)
    pu, pi = np.unique(pt, return_index=True)
    rt, rf, pt, pf = ru, rf[ri], pu, pf[pi]
    if rt.size < 2 or pt.size < 2:
        return bad

    lo = max(rt[0], pt[0])
    hi = min(rt[-1], pt[-1])
    ref_duration = max(rt[-1] - rt[0], np.finfo(float).eps)
    coverage = float(np.clip(max(0.0, hi - lo) / ref_duration, 0.0, 1.0))
    if hi <= lo:
        return CurveMetrics(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), coverage,
                            float(pt[0] - rt[0]), float(pt[-1] - rt[-1]))

    # Evaluate on a dense common grid so sparse manual control points do not
    # dominate the score.
    n = max(100, min(1000, int(np.ceil((hi - lo) / 0.01)) + 1))
    grid = np.linspace(lo, hi, n)
    ref_f = np.interp(grid, rt, rf)
    pred_f = np.interp(grid, pt, pf)
    err = pred_f - ref_f
    abs_err = np.abs(err)

    return CurveMetrics(
        mae_khz=float(np.mean(abs_err)),
        median_abs_error_khz=float(np.median(abs_err)),
        p95_abs_error_khz=float(np.percentile(abs_err, 95)),
        rmse_khz=float(np.sqrt(np.mean(err**2))),
        bias_khz=float(np.mean(err)),
        coverage=coverage,
        start_error_ms=float(pt[0] - rt[0]),
        end_error_ms=float(pt[-1] - rt[-1]),
    )
