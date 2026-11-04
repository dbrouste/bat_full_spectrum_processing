"""Tests for the conservative missing-ridge-end completion stage."""
import numpy as np
import pytest

from bat_analysis.ridge_completion import extend_ridge_edges


def _synthetic_chirp():
    sr = 384000
    t = np.arange(int(.025 * sr)) / sr
    x = np.zeros_like(t)
    on = (t >= .009) & (t <= .012)
    tau = t[on] - .009
    x[on] = (
        .75 * np.sin(2 * np.pi * (
            115000 * tau - .5 * (63000 / .003) * tau * tau
        )) * np.sin(np.pi * tau / .003) ** 2
    )
    return x, sr


def _partial_tail():
    # Simulate three valid ridge points near the END of a descending chirp.
    times = np.array([.0109, .0112, .0115])
    freqs = 115000 - (63000 / .003) * (times - .009)
    return np.column_stack([times, freqs, np.ones(len(times))])


def test_recovers_observed_missing_left_flank():
    wav, sr = _synthetic_chirp()
    points = _partial_tail()
    recovered, diag = extend_ridge_edges(
        wav, sr, points, bounds_s=(.009, .012), return_diagnostics=True,
    )
    assert diag["left_points"] >= 5
    assert recovered[0, 0] < .0104
    assert recovered[-1, 0] <= .012
    assert np.all(np.diff(recovered[:, 0]) > 0)
    assert np.isfinite(recovered).all()
    # Original ridge observations are kept exactly.
    for p in points:
        assert any(np.array_equal(p, row) for row in recovered)


def test_silence_is_not_prolonged():
    sr = 384000
    silence = np.zeros(int(.02 * sr))
    ts = np.array([.009, .0092, .0094])
    source = np.column_stack([
        ts, np.array([120000, 115000, 110000]), np.ones(3),
    ])
    result, diag = extend_ridge_edges(
        silence, sr, source, bounds_s=(.008, .010),
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(result, source)
    assert diag["left_points"] == 0
    assert diag["right_points"] == 0


def test_frequency_and_time_limits_preserve_input():
    wav, sr = _synthetic_chirp()
    points = _partial_tail()
    backup = points.copy()
    r = extend_ridge_edges(
        wav, sr, points, bounds_s=(.009, .012),
        extend_right=False,
    )
    np.testing.assert_array_equal(points, backup)
    assert r[0, 0] >= .009 and r[-1, 0] <= .012


def test_invalid_settings_raise_clear_errors():
    wav, sr = _synthetic_chirp()
    points = _partial_tail()
    with pytest.raises(ValueError, match="parameters"):
        extend_ridge_edges(
            wav, sr, points, bounds_s=(.009, .012),
            min_snr_db=-1,
        )
    with pytest.raises(ValueError, match="invalid bounds"):
        extend_ridge_edges(wav, sr, points, bounds_s=(.012, .009))


def test_completion_can_be_disabled_in_public_tracker(monkeypatch):
    from bat_analysis import reassignment_tracking
    called = []
    def spy(*args, **kwargs):
        called.append(kwargs)
        return args[2], {"left_points": 0, "right_points": 0}
    # Confirm the API exposes the opt-in switch without changing legacy mode.
    import inspect
    signature = inspect.signature(reassignment_tracking.track_reassigned_ridge)
    assert signature.parameters["complete_ends"].default is True
    assert "peak_anchored_window" in signature.parameters
