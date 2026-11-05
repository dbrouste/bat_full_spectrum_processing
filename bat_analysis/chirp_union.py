"""Group overlapping bat-call ridges without mixing distinct frequencies.

Two ridge fragments may be stitched only if frequency AND slope agree through
their common interval. Simultaneous distinct calls (including harmonics) share
an overlap group but remain separate tracks; a group is not a bat passage.
All input/output curves use [absolute time in seconds, frequency in Hz,
relative amplitude] with strictly increasing timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass
class OverlappingChirps:
    """Related concurrent tracks, retaining source identities.

    `ridges` may have one entry when coincident ridge fragments were stitched.
    The union of their timestamps does not imply a single bat or echolocation
    emission when `ridges` contains two or more independent tracks.
    """
    source_ids: tuple[Hashable, ...]
    ridges: tuple[np.ndarray, ...]
    stitched_sources: tuple[tuple[Hashable, ...], ...]
    possible_harmonics: tuple[tuple[Hashable, Hashable], ...]
    kind: str

    @property
    def t_start_s(self) -> float:
        return min(float(r[0, 0]) for r in self.ridges)

    @property
    def t_end_s(self) -> float:
        return max(float(r[-1, 0]) for r in self.ridges)


def _validate(ridge) -> np.ndarray:
    r = np.asarray(ridge, dtype=float)
    if (r.ndim != 2 or r.shape[1] != 3 or len(r) < 2
            or not np.isfinite(r).all()
            or not np.all(np.diff(r[:, 0]) > 0)
            or np.any(r[:, 1] <= 0)):
        raise ValueError("each ridge must be a finite, time-sorted Nx3 array")
    return r.copy()


def _overlap(a, b) -> tuple[float, float]:
    return max(float(a[0, 0]), float(b[0, 0])), min(
        float(a[-1, 0]), float(b[-1, 0])
    )


def _comparison(a, b, *, minimum_overlap_ms):
    start, stop = _overlap(a, b)
    if (stop - start) * 1000.0 < minimum_overlap_ms:
        return None
    # Evaluate the full shared interval rather than only the crossing point.
    ts = np.linspace(start, stop, 25)
    fa = np.interp(ts, a[:, 0], a[:, 1])
    fb = np.interp(ts, b[:, 0], b[:, 1])
    delta = np.abs(fa - fb)
    # Linear trends are robust to noisy/jagged reassigned bins.
    dt = (stop - start) * 1000.0
    slope_a = float((fa[-1] - fa[0]) / dt)  # Hz per ms
    slope_b = float((fb[-1] - fb[0]) / dt)
    hi = np.maximum(fa, fb)
    lo = np.minimum(fa, fb)
    ratio = float(np.median(hi / lo))
    return float(np.median(delta)), float(np.percentile(delta, 90)), (
        abs(slope_a - slope_b)
    ), ratio


def can_stitch_ridges(a, b, *, min_overlap_ms=0.20,
                      median_freq_tolerance_hz=2200.0,
                      p90_freq_tolerance_hz=4000.0,
                      slope_tolerance_hz_per_ms=7_000.0) -> bool:
    """Require a consistent shared ridge rather than mere time overlap.

    A near crossing of two independent trajectories alone is not sufficient.
    """
    x, y = _validate(a), _validate(b)
    m = _comparison(x, y, minimum_overlap_ms=min_overlap_ms)
    if m is None:
        return False
    median, p90, slope_difference, _ = m
    return (median <= median_freq_tolerance_hz
            and p90 <= p90_freq_tolerance_hz
            and slope_difference <= slope_tolerance_hz_per_ms)


def stitch_coincident_ridges(a, b, *, min_overlap_ms=0.20,
                             median_freq_tolerance_hz=2200.0,
                             p90_freq_tolerance_hz=4000.0,
                             slope_tolerance_hz_per_ms=7_000.0):
    """Combine coincident fragments without inventing points in time gaps.

    Returns None when they are simultaneous but distinct signals. If one
    fragment is fully contained in another, the longer curve is retained.
    """
    x, y = _validate(a), _validate(b)
    if not can_stitch_ridges(
        x, y, min_overlap_ms=min_overlap_ms,
        median_freq_tolerance_hz=median_freq_tolerance_hz,
        p90_freq_tolerance_hz=p90_freq_tolerance_hz,
        slope_tolerance_hz_per_ms=slope_tolerance_hz_per_ms,
    ):
        return None
    if (x[0, 0] >= y[0, 0] and x[-1, 0] <= y[-1, 0]):
        return y
    if (y[0, 0] >= x[0, 0] and y[-1, 0] <= x[-1, 0]):
        return x
    # Earlier fragment feeds the overlap; later fragment takes over smoothly.
    if (x[0, 0], x[-1, 0]) > (y[0, 0], y[-1, 0]):
        x, y = y, x
    start, stop = _overlap(x, y)
    ts = np.unique(np.concatenate((x[:, 0], y[:, 0])))
    present_x = (ts >= x[0, 0]) & (ts <= x[-1, 0])
    present_y = (ts >= y[0, 0]) & (ts <= y[-1, 0])
    fx = np.interp(ts, x[:, 0], x[:, 1])
    fy = np.interp(ts, y[:, 0], y[:, 1])
    ax = np.interp(ts, x[:, 0], x[:, 2])
    ay = np.interp(ts, y[:, 0], y[:, 2])
    z = np.empty((len(ts), 3), dtype=float)
    z[:, 0] = ts
    z[:, 1] = np.where(present_x, fx, fy)
    z[:, 2] = np.where(present_x, ax, ay)
    both = present_x & present_y
    u = np.clip((ts[both] - start) / max(stop - start, 1e-15), 0, 1)
    w = u * u * (3 - 2 * u)  # smoothstep; no step at overlap endpoints
    z[both, 1] = (1 - w) * fx[both] + w * fy[both]
    z[both, 2] = (1 - w) * ax[both] + w * ay[both]
    return z


def group_overlapping_chirps(
    ridges: Sequence[np.ndarray],
    *, source_ids=None, min_overlap_ms=0.20, stitch=True,
    median_freq_tolerance_hz=2200.0,
    p90_freq_tolerance_hz=4000.0,
    slope_tolerance_hz_per_ms=7_000.0,
    harmonic_ratio_tolerance=0.10,
) -> list[OverlappingChirps]:
    """Group every pair with temporal overlap, preserving distinct ridges.

    The optional 'stitch' operation is conservative and uses time, frequency
    and slope compatibility. Candidate harmonic links are only suggestions,
    not confirmed associations and not species/passages.
    """
    if min_overlap_ms <= 0 or harmonic_ratio_tolerance <= 0:
        raise ValueError("overlap and harmonic tolerances must be positive")
    if (median_freq_tolerance_hz <= 0 or p90_freq_tolerance_hz <= 0
            or slope_tolerance_hz_per_ms <= 0):
        raise ValueError("stitch tolerances must be positive")
    curves = [_validate(x) for x in ridges]
    count = len(curves)
    ids = list(range(count)) if source_ids is None else list(source_ids)
    if len(ids) != count or len(set(ids)) != len(ids):
        raise ValueError("source_ids must be unique and match ridges length")
    parents = list(range(count))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parents[b] = a

    for i in range(count):
        for j in range(i + 1, count):
            start, stop = _overlap(curves[i], curves[j])
            if (stop - start) * 1000.0 >= min_overlap_ms:
                union(i, j)
    clusters = {}
    for i in range(count):
        clusters.setdefault(find(i), []).append(i)

    results = []
    for indices in clusters.values():
        # Tracks are kept as separate lines even when in the same group.
        tracks = [(curves[i], (ids[i],)) for i in indices]
        merged_any = False
        if stitch:
            while True:
                choice = None
                for i in range(len(tracks)):
                    for j in range(i + 1, len(tracks)):
                        m = _comparison(
                            tracks[i][0], tracks[j][0],
                            minimum_overlap_ms=min_overlap_ms)
                        if m is None:
                            continue
                        merged = stitch_coincident_ridges(
                            tracks[i][0], tracks[j][0],
                            min_overlap_ms=min_overlap_ms,
                            median_freq_tolerance_hz=median_freq_tolerance_hz,
                            p90_freq_tolerance_hz=p90_freq_tolerance_hz,
                            slope_tolerance_hz_per_ms=slope_tolerance_hz_per_ms,
                        )
                        if merged is None:
                            continue
                        if choice is None or m[0] < choice[0]:
                            choice = (m[0], i, j, merged)
                if choice is None:
                    break
                _, i, j, joined = choice
                _, provenance_i = tracks[i]
                _, provenance_j = tracks[j]
                tracks[i] = (joined, provenance_i + provenance_j)
                tracks.pop(j)
                merged_any = True

        possible_harmonics = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                m = _comparison(tracks[i][0], tracks[j][0],
                                minimum_overlap_ms=min_overlap_ms)
                if m is not None and abs(m[3] - 2.0) <= harmonic_ratio_tolerance:
                    # Lower mean frequency is the candidate parent.
                    f_i = float(np.mean(tracks[i][0][:, 1]))
                    f_j = float(np.mean(tracks[j][0][:, 1]))
                    lower, higher = ((i, j) if f_i <= f_j else (j, i))
                    possible_harmonics.append((
                        tracks[lower][1][0], tracks[higher][1][0]
                    ))
        tracks.sort(key=lambda item: (item[0][0, 0], item[0][0, 1]))
        results.append(OverlappingChirps(
            source_ids=tuple(ids[i] for i in indices),
            ridges=tuple(x for x, _ in tracks),
            stitched_sources=tuple(tuple(src) for _, src in tracks),
            possible_harmonics=tuple(possible_harmonics),
            kind=("single" if len(indices) == 1 else
                  "stitched" if len(tracks) == 1 and merged_any else
                  "overlapping"),
        ))
    return sorted(results, key=lambda group: group.t_start_s)
