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
    frequency_error_khz: float | None = None


def _finish_assignment(
    cost: np.ndarray,
    ious: np.ndarray,
    center_ms: np.ndarray,
    frequency_error_khz: np.ndarray | None = None,
) -> tuple[list[Match], list[int], list[int]]:
    nr, nd = cost.shape
    invalid_cost = 1e6
    rows, cols = linear_sum_assignment(cost)
    matches: list[Match] = []
    used_r, used_d = set(), set()

    for i, j in zip(rows.tolist(), cols.tolist()):
        if cost[i, j] >= invalid_cost:
            continue
        freq_error = None
        if frequency_error_khz is not None and np.isfinite(frequency_error_khz[i, j]):
            freq_error = float(frequency_error_khz[i, j])
        matches.append(
            Match(
                i,
                j,
                float(ious[i, j]),
                float(center_ms[i, j]),
                freq_error,
            )
        )
        used_r.add(i)
        used_d.add(j)

    unmatched_r = [i for i in range(nr) if i not in used_r]
    unmatched_d = [j for j in range(nd) if j not in used_d]
    return matches, unmatched_r, unmatched_d


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

    This is the historical benchmark matcher and remains the default so old and
    new detector results stay directly comparable.
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
                cost[i, j] = (1.0 - iou) + 0.25 * min(
                    ce / max_center_error_ms, 1.0
                )

    return _finish_assignment(cost, ious, center_ms)


def match_chirps_frequency_aware(
    reference_intervals_s: Sequence[tuple[float, float]],
    detected_intervals_s: Sequence[tuple[float, float]],
    reference_times_s: Sequence[np.ndarray],
    reference_freqs_khz: Sequence[np.ndarray],
    detected_peak_times_s: Sequence[float],
    detected_peak_freqs_khz: Sequence[float],
    *,
    min_iou: float = 0.05,
    max_center_error_ms: float = 4.0,
    frequency_scale_khz: float = 10.0,
    frequency_weight: float = 0.5,
) -> tuple[list[Match], list[int], list[int]]:
    """Hungarian matching with frequency used only to disambiguate time matches.

    Temporal admissibility is identical to match_chirps; frequency does not
    create a match that would otherwise be rejected. It only changes the cost
    among temporally admissible pairs, useful for simultaneous harmonics.

    Manual frequency is linearly interpolated at the detected peak time.
    """
    nr, nd = len(reference_intervals_s), len(detected_intervals_s)
    if nr == 0 or nd == 0:
        return [], list(range(nr)), list(range(nd))

    if not (
        len(reference_times_s) == nr
        and len(reference_freqs_khz) == nr
        and len(detected_peak_times_s) == nd
        and len(detected_peak_freqs_khz) == nd
    ):
        raise ValueError("Frequency-aware matching inputs have inconsistent lengths")

    invalid_cost = 1e6
    cost = np.full((nr, nd), invalid_cost, dtype=float)
    ious = np.zeros((nr, nd), dtype=float)
    center_ms = np.full((nr, nd), np.inf, dtype=float)
    freq_err = np.full((nr, nd), np.nan, dtype=float)

    freq_scale = max(float(frequency_scale_khz), 1e-9)
    freq_weight = max(float(frequency_weight), 0.0)

    for i, ref in enumerate(reference_intervals_s):
        rc = 0.5 * (ref[0] + ref[1])
        rt = np.asarray(reference_times_s[i], dtype=float)
        rf = np.asarray(reference_freqs_khz[i], dtype=float)

        for j, det in enumerate(detected_intervals_s):
            dc = 0.5 * (det[0] + det[1])
            iou = interval_iou(ref, det)
            ce = abs(rc - dc) * 1000.0
            ious[i, j] = iou
            center_ms[i, j] = ce

            if not (iou >= min_iou or ce <= max_center_error_ms):
                continue

            temporal_cost = (1.0 - iou) + 0.25 * min(
                ce / max_center_error_ms, 1.0
            )

            dp_t = float(detected_peak_times_s[j])
            dp_f = float(detected_peak_freqs_khz[j])
            if rt.size >= 2 and rf.size == rt.size and np.isfinite(dp_t) and np.isfinite(dp_f):
                manual_f = float(np.interp(dp_t, rt, rf))
                ferr = abs(manual_f - dp_f)
                freq_err[i, j] = ferr
                frequency_cost = freq_weight * min(ferr / freq_scale, 3.0)
            else:
                frequency_cost = 0.0

            cost[i, j] = temporal_cost + frequency_cost

    return _finish_assignment(cost, ious, center_ms, freq_err)
