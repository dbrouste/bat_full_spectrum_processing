from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection_metrics import interval_iou


@dataclass(frozen=True)
class Match:
    reference_index: int
    detected_index: int
    iou: float
    center_error_ms: float


def match_chirps(
    reference_intervals_s: Sequence[tuple[float, float]],
    detected_intervals_s: Sequence[tuple[float, float]],
    *,
    min_iou: float = 0.05,
    max_center_error_ms: float = 4.0,
) -> tuple[list[Match], list[int], list[int]]:
    """One-to-one Hungarian matching between manual and detected chirps.

    A pair is admissible when intervals overlap enough OR when their temporal
    centres are close enough. The cost favours overlap first, then centre error.
    """
    nr, nd = len(reference_intervals_s), len(detected_intervals_s)
    if nr == 0 or nd == 0:
        return [], list(range(nr)), list(range(nd))

    invalid_cost = 1e6
    cost = np.full((nr, nd), invalid_cost, dtype=float)
    ious = np.zeros((nr, nd), dtype=float)
    center_ms = np.full((nr, nd), np.inf, dtype=float)

    for i, ref in enumerate(reference_intervals_s):
        rc = 0.5 * (ref[0] + ref[1])
        for j, det in enumerate(detected_intervals_s):
            dc = 0.5 * (det[0] + det[1])
            iou = interval_iou(ref, det)
            ce = abs(rc - dc) * 1000.0
            ious[i, j] = iou
            center_ms[i, j] = ce
            if iou >= min_iou or ce <= max_center_error_ms:
                # overlap dominates; centre distance breaks ambiguous cases
                cost[i, j] = (1.0 - iou) + 0.25 * min(ce / max_center_error_ms, 1.0)

    rows, cols = linear_sum_assignment(cost)
    matches: list[Match] = []
    used_r, used_d = set(), set()
    for i, j in zip(rows.tolist(), cols.tolist()):
        if cost[i, j] >= invalid_cost:
            continue
        matches.append(Match(i, j, float(ious[i, j]), float(center_ms[i, j])))
        used_r.add(i)
        used_d.add(j)

    unmatched_r = [i for i in range(nr) if i not in used_r]
    unmatched_d = [j for j in range(nd) if j not in used_d]
    return matches, unmatched_r, unmatched_d
