"""Tests for proposed second harmonics and frequency-only endpoint correction."""
import numpy as np
import pytest

from bat_analysis.harmonics import (
    detect_harmonic_segments, track_candidate_with_harmonics,
)
from bat_analysis.endpoint_refinement import refine_ridge_endpoint


def synthetic_call(with_harmonic=True):
    sr = 384000
    t = np.arange(int(.030 * sr)) / sr
    t0, duration = .010, .004
    active = (t >= t0) & (t <= t0 + duration)
    tau = t[active] - t0
    phase = 2 * np.pi * (
        110000 * tau - .5 * (58000 / duration) * tau**2
    )
    y = np.random.default_rng(10).normal(0, .002, size=len(t))
    y[active] += .65 * np.sin(phase) * np.sin(np.pi * tau / duration)**2
    if with_harmonic:
        h = np.zeros_like(tau)
        keep = (tau > .0022) & (tau < .0038)
        h[keep] = .08 * np.sin(2 * phase[keep]) * (
            np.sin(np.pi * (tau[keep] - .0022) / .0016)**2
        )
        y[active] += h
    tt = np.linspace(t0, t0 + duration, 80)
    fundamental = np.column_stack([
        tt, np.linspace(110000, 52000, len(tt)), np.ones(len(tt))
    ])
    return y, sr, fundamental


def test_guided_second_harmonic_and_parent_link():
    y, sr, parent = synthetic_call(True)
    segments = detect_harmonic_segments(y, sr, parent, parent_id=42)
    assert len(segments) == 1
    harmonic = segments[0]
    assert harmonic.harmonic_order == 2
    assert harmonic.parent_id == 42
    assert 1.7 < harmonic.median_frequency_ratio < 2.3
    assert harmonic.t_start_s < .0138
    assert harmonic.t_end_s > .0125
    assert harmonic.mean_snr_db >= 14
    assert np.all(np.diff(harmonic.points[:, 0]) > 0)


def test_no_harmonic_on_fundamental_only_with_noise():
    y, sr, parent = synthetic_call(False)
    assert detect_harmonic_segments(y, sr, parent) == []


def test_invalid_harmonic_input():
    y, sr, parent = synthetic_call()
    with pytest.raises(ValueError, match="second harmonic"):
        detect_harmonic_segments(y, sr, parent, harmonic_order=3)
    with pytest.raises(ValueError, match="strictly time"):
        detect_harmonic_segments(y, sr, parent[::-1])


def test_tail_correction_keeps_time_amplitude_and_input():
    t = np.linspace(.01, .013, 70)
    f = 118000 - 22000000 * (t - t[0])
    f[48:] = f[47]  # spurious plateau
    amp = np.linspace(.8, .2, len(t))
    curve = np.column_stack([t, f, amp])
    unchanged = curve.copy()
    refined, diag = refine_ridge_endpoint(curve, return_diagnostics=True)
    np.testing.assert_array_equal(curve, unchanged)
    np.testing.assert_array_equal(curve[:, 0], refined[:, 0])
    np.testing.assert_array_equal(curve[:, 2], refined[:, 2])
    assert refined[-1, 1] < curve[-1, 1]
    assert diag["corrected_points"] > 0
    assert diag["max_correction_hz"] <= 16000


def test_endpoint_correction_rejects_unsupported_frequency_shift():
    t = np.linspace(.010, .013, 70)
    f = 118000 - 22000000 * (t - t[0])
    f[48:] = f[47]
    curve = np.column_stack([t, f, np.ones(len(t))])
    frequency_grid = np.linspace(20000., 150000., 261)
    power = np.full((len(frequency_grid), len(t)), 1e-9)
    for j, observed_freq in enumerate(f):
        ri = int(np.argmin(abs(frequency_grid - observed_freq)))
        power[ri, j] = 1.0

    ungated = refine_ridge_endpoint(curve)
    guarded, diag = refine_ridge_endpoint(
        curve, times_s=t, freqs_hz=frequency_grid,
        stft_power=power, return_diagnostics=True)
    assert ungated[-1, 1] < curve[-1, 1]
    assert guarded[-1, 1] == curve[-1, 1]
    assert diag["spectral_support_gated"]
    assert diag["rejected_corrections"] > 0


def test_endpoint_correction_accepts_supported_frequency_shift():
    t = np.linspace(.010, .013, 70)
    f = 118000 - 22000000 * (t - t[0])
    f[48:] = f[47]
    curve = np.column_stack([t, f, np.ones(len(t))])
    ungated = refine_ridge_endpoint(curve)
    frequency_grid = np.linspace(20000., 150000., 261)
    power = np.full((len(frequency_grid), len(t)), 1e-9)
    for j, (observed, corrected) in enumerate(zip(f, ungated[:, 1])):
        for freq in (observed, corrected):
            ri = int(np.argmin(abs(frequency_grid - freq)))
            power[ri, j] = 1.0
    guarded = refine_ridge_endpoint(
        curve, times_s=t, freqs_hz=frequency_grid, stft_power=power)
    np.testing.assert_allclose(guarded, ungated)


def test_cf_trajectory_not_forced_down():
    t = np.linspace(.010, .013, 70)
    curve = np.column_stack([t, np.full_like(t, 80000.),
                             np.full_like(t, .7)])
    out = refine_ridge_endpoint(curve)
    np.testing.assert_allclose(curve, out)


def test_candidate_wrapper_links_harmonic_to_parent(monkeypatch):
    from bat_analysis import reassignment_tracking as tracker
    y, sr, parent = synthetic_call(True)
    monkeypatch.setattr(tracker, "track_reassigned_candidate",
                        lambda *args, **kwargs: parent)
    found, segments = track_candidate_with_harmonics(
        y, sr, {"time_mid": .012}, parent_id="fundamental-1"
    )
    np.testing.assert_allclose(found, parent)
    assert len(segments) == 1
    assert segments[0].parent_id == "fundamental-1"
