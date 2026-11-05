"""Overlap semantics: same ridge can stitch, parallel calls must remain distinct."""
import numpy as np
import pytest

from bat_analysis.chirp_union import (
    can_stitch_ridges,
    stitch_coincident_ridges,
    group_overlapping_chirps,
)
from bat_analysis import reassignment_tracking


def ridge(start_ms, end_ms, initial_khz, slope_khz_ms, count=26):
    t_ms = np.linspace(start_ms, end_ms, count)
    f_khz = initial_khz + slope_khz_ms * (t_ms - start_ms)
    return np.column_stack([
        t_ms / 1000, f_khz * 1000, np.linspace(.5, .85, count)
    ])


def test_stitches_two_fragments_of_same_ridge():
    a = ridge(1037, 1040, 120, -20)
    b = ridge(1039, 1042, 80, -20)
    orig_a = a.copy()
    assert can_stitch_ridges(a, b)
    joined = stitch_coincident_ridges(a, b)
    assert joined is not None
    assert joined[0, 0] == pytest.approx(1.037)
    assert joined[-1, 0] == pytest.approx(1.042)
    assert np.all(np.diff(joined[:, 0]) > 0)
    np.testing.assert_array_equal(a, orig_a)
    group = group_overlapping_chirps(
        [a, b], source_ids=["fragmentA", "fragmentB"]
    )
    assert len(group) == 1
    assert group[0].kind == "stitched"
    assert len(group[0].ridges) == 1
    assert set(group[0].stitched_sources[0]) == {"fragmentA", "fragmentB"}


def test_parallel_overlapping_chirps_kept_as_two_tracks():
    # Mimics the screenshot: two descending FM calls overlap in time but
    # their simultaneous frequency ridges are not the same.
    left = ridge(1037, 1040.4, 120, -20)
    right = ridge(1038.8, 1042.4, 125, -20)
    assert not can_stitch_ridges(left, right)
    groups = group_overlapping_chirps([left, right])
    assert len(groups) == 1
    group = groups[0]
    assert group.kind == "overlapping"
    assert len(group.ridges) == 2
    assert group.source_ids == (0, 1)
    assert group.t_start_s == pytest.approx(1.037)
    assert group.t_end_s == pytest.approx(1.0424)


def test_harmonic_link_without_fusing_tracks():
    parent = ridge(10, 13, 61000 / 1000, -4)
    harmonic = ridge(10.6, 12.8, 2 * (61 - 4 * .6), -8)
    groups = group_overlapping_chirps(
        [parent, harmonic], source_ids=["fundamental", "H2"]
    )
    assert len(groups) == 1
    assert len(groups[0].ridges) == 2
    assert groups[0].possible_harmonics == (("fundamental", "H2"),)


def test_crossing_at_one_point_not_stitched():
    a = ridge(10, 13, 120, -20)
    b = ridge(10, 13, 60, +20)
    assert not can_stitch_ridges(a, b)
    g = group_overlapping_chirps([a, b])
    assert len(g) == 1 and len(g[0].ridges) == 2


def test_no_temporal_overlap_stays_separate():
    a = ridge(10, 12, 120, -20)
    b = ridge(12.5, 14, 90, -20)
    assert len(group_overlapping_chirps([a, b])) == 2


def test_transitive_overlap_preserves_three_independent_tracks():
    a = ridge(10, 12, 120, -10)
    b = ridge(11, 13, 90, -10)
    c = ridge(12, 14, 65, -10)
    result = group_overlapping_chirps([a, b, c])
    assert len(result) == 1 and len(result[0].ridges) == 3


def test_contained_duplicate_uses_outer_trajectory():
    a = ridge(10, 13, 120, -10)
    b = ridge(11, 12, 110, -10)
    joined = stitch_coincident_ridges(a, b)
    np.testing.assert_array_equal(joined, a)


def test_input_validation():
    a = ridge(10, 13, 120, -10)
    with pytest.raises(ValueError, match="unique"):
        group_overlapping_chirps([a, a], source_ids=[1, 1])
    with pytest.raises(ValueError, match="finite"):
        group_overlapping_chirps([a[::-1]])
    with pytest.raises(ValueError, match="positive"):
        group_overlapping_chirps([a], min_overlap_ms=0)


def test_candidate_to_group_wrapper_with_diagnostic_failures(monkeypatch):
    def mock_tracking(y, sr, candidate, *, diagnostics, **kwargs):
        if candidate.get("skip"):
            return None
        diagnostics["model_success"] = True
        return candidate["test_curve"]
    monkeypatch.setattr(
        reassignment_tracking, "track_reassigned_candidate", mock_tracking
    )
    a = ridge(10, 13, 120, -20)
    b = ridge(11, 14, 100, -20)
    result = reassignment_tracking.track_and_group_candidates(
        np.zeros(5000), 384000,
        [{"test_curve": a}, {"test_curve": b}, {"skip": True}]
    )
    assert result["candidate_count"] == 3
    assert result["untracked_candidate_indices"] == [2]
    assert len(result["groups"]) == 1
    assert result["groups"][0].kind == "stitched"
    assert len(result["tracking_diagnostics"]) == 3


def test_near_1038ms_same_call_fragments_join_but_following_call_does_not():
    """Representative geometry from 20261005_191936: distinct later FM stays separate."""
    fragment10 = ridge(1037.6, 1040.05, 112, -19)
    fragment11 = ridge(1038.2, 1040.35, 112 - 19 * .6, -19)
    following = ridge(1039.90, 1042.4, 108, -19)
    groups = group_overlapping_chirps(
        [fragment10, fragment11, following],
        source_ids=[10, 11, 12],
        min_overlap_ms=.20,
        median_freq_tolerance_hz=4000.,
        p90_freq_tolerance_hz=7000.,
        slope_tolerance_hz_per_ms=7000.,
    )
    tracks = [sources for group in groups for sources in group.stitched_sources]
    assert (10, 11) in tracks
    assert (12,) in tracks
    merged_group = [group for group in groups if 10 in group.source_ids][0]
    assert len(merged_group.ridges) == 2
    first = merged_group.ridges[merged_group.stitched_sources.index((10, 11))]
    assert first[0, 0] == pytest.approx(1.0376)
    assert first[-1, 0] == pytest.approx(1.04035)
    assert np.all(np.diff(first[:, 0]) > 0)


def test_similar_timing_but_separate_frequency_rejected_at_five_khz_tolerance():
    """The simplification tolerance is not a license to join distinct calls."""
    a = ridge(1037.6, 1040.0, 112, -19)
    b = ridge(1038.7, 1042.3, 105, -19)
    # Simultaneous frequency difference ~28 kHz.
    assert not can_stitch_ridges(
        a, b, median_freq_tolerance_hz=4000.,
        p90_freq_tolerance_hz=7000.,
    )
