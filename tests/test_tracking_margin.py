"""Regression tests for detector-bounded search-window recovery."""

import numpy as np
import pytest

from bat_analysis.modelling import process_full_spectrum
from bat_analysis.reassignment_tracking import (
    track_reassigned_candidate, track_reassigned_ridge,
)


def fm_chirp_with_narrow_detector_bounds():
    sr = 384000
    t = np.arange(int(0.02 * sr)) / sr
    onset, duration = 0.005, 0.003
    elapsed = t - onset
    active = (elapsed >= 0) & (elapsed <= duration)
    y = np.zeros_like(t)
    e = elapsed[active]
    y[active] = (
        0.7 * np.sin(2 * np.pi * (120000 * e - .5 * (65000 / duration) * e ** 2))
        * np.sin(np.pi * e / duration) ** 2
    )
    candidate = {"time_mid": .0069, "duration": .0015,
                 "peak_freq_hz": 79000, "t_start": .0063, "t_end": .0074}
    return y, sr, onset, candidate


def test_bounded_candidate_recovers_early_fm_start_from_narrow_bbox():
    y, sr, onset, candidate = fm_chirp_with_narrow_detector_bounds()
    diag = {}
    improved = track_reassigned_candidate(
        y, sr, candidate, n_fft=512, hop=24, diagnostics=diag)
    previous = track_reassigned_candidate(
        y, sr, candidate, n_fft=512, hop=24, search_margin_ms=1.0)
    assert improved is not None and previous is not None
    assert diag["search_margin_ms"] == 1.5
    assert improved[0, 0] < previous[0, 0]
    assert abs(improved[0, 0] - onset) < abs(previous[0, 0] - onset)


def test_application_modelling_path_also_uses_new_search_margin():
    y, sr, _, candidate = fm_chirp_with_narrow_detector_bounds()
    diag = {}
    curve = process_full_spectrum(
        y, sr, time_mid=candidate["time_mid"],
        duration=candidate["duration"], seed_freq_hz=candidate["peak_freq_hz"],
        chirp_start_s=candidate["t_start"], chirp_end_s=candidate["t_end"],
        diagnostics=diag,
    )
    assert curve is not None
    assert diag["search_margin_ms"] == 1.5


def test_short_bounded_candidate_gets_adaptive_margin_with_explicit_override():
    y, sr, _, candidate = fm_chirp_with_narrow_detector_bounds()
    candidate["duration"] = .001
    diag = {}
    track_reassigned_candidate(y, sr, candidate, diagnostics=diag)
    assert diag["search_margin_ms"] == 2.0
    candidate["duration"] = .00125
    track_reassigned_candidate(y, sr, candidate, diagnostics=diag)
    assert diag["search_margin_ms"] == 1.75
    track_reassigned_candidate(
        y, sr, candidate, search_margin_ms=1.5, diagnostics=diag)
    assert diag["search_margin_ms"] == 1.5


def test_unbounded_tracking_keeps_previous_margin_and_validates_override():
    sr = 384000
    x = np.sin(2 * np.pi * 70000 * np.arange(3840) / sr)
    diag = {}
    track_reassigned_ridge(x, sr, time_mid=.005, duration=.002,
                           seed_freq_hz=70000, diagnostics=diag)
    if diag.get("model_success"):
        assert diag["search_margin_ms"] == 1.0
    with pytest.raises(ValueError, match="search_margin_ms"):
        track_reassigned_ridge(x, sr, time_mid=.005, duration=.002,
                               seed_freq_hz=70000, search_margin_ms=-.5)
