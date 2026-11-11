from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class CurveMetrics:
    median_abs_error_khz: float
    p95_abs_error_khz: float
    rmse_khz: float
    coverage: float
    median_abs_error_khz_eq: float | None = None
    p95_abs_error_khz_eq: float | None = None
    geometric_coverage_fraction: float | None = None
    geometry_protocol: str = "vertical_legacy"


def compare_curves(
    reference_t_ms: Sequence[float],
    reference_f_khz: Sequence[float],
    predicted_t_ms: Sequence[float],
    predicted_f_khz: Sequence[float],
    *, geometry_mode: str = "vertical_legacy",
) -> CurveMetrics:
    """Compare predicted f(t) with manual ground truth over the common time span."""
    rt = np.asarray(reference_t_ms, dtype=float)
    rf = np.asarray(reference_f_khz, dtype=float)
    pt = np.asarray(predicted_t_ms, dtype=float)
    pf = np.asarray(predicted_f_khz, dtype=float)

    if rt.size < 2 or pt.size < 2:
        return CurveMetrics(float("nan"), float("nan"), float("nan"), 0.0)

    order_r = np.argsort(rt)
    order_p = np.argsort(pt)
    rt, rf = rt[order_r], rf[order_r]
    pt, pf = pt[order_p], pf[order_p]

    lo = max(rt[0], pt[0])
    hi = min(rt[-1], pt[-1])
    if hi <= lo:
        return CurveMetrics(float("nan"), float("nan"), float("nan"), 0.0)

    mask = (rt >= lo) & (rt <= hi)
    if not np.any(mask):
        return CurveMetrics(float("nan"), float("nan"), float("nan"), 0.0)

    pred_on_ref = np.interp(rt[mask], pt, pf)
    err = pred_on_ref - rf[mask]
    abs_err = np.abs(err)
    ref_duration = max(rt[-1] - rt[0], np.finfo(float).eps)
    coverage = float((hi - lo) / ref_duration)

    geometric = {}
    if geometry_mode == "square_normals_v1":
        from chirp_geometry import PROTOCOL, compare_normal_curves
        geom = compare_normal_curves(rt, rf, pt, pf, np.linspace(rt[0], rt[-1], 50))
        distances = geom["abs_normal_khz_eq"]
        valid = geom["covered"]
        geometric = {
            "median_abs_error_khz_eq": (float(np.median(distances[valid]))
                                        if np.any(valid) else None),
            "p95_abs_error_khz_eq": (float(np.percentile(distances[valid], 95))
                                     if np.any(valid) else None),
            "geometric_coverage_fraction": geom["coverage_fraction"],
            "geometry_protocol": PROTOCOL,
        }
    elif geometry_mode != "vertical_legacy":
        raise ValueError("unknown geometry_mode")
    return CurveMetrics(
        median_abs_error_khz=float(np.median(abs_err)),
        p95_abs_error_khz=float(np.percentile(abs_err, 95)),
        rmse_khz=float(np.sqrt(np.mean(err**2))),
        coverage=float(np.clip(coverage, 0.0, 1.0)),
        **geometric,
    )
