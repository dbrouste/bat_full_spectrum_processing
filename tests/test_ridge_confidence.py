"""Regression tests for conservative SNR-gated right-edge stopping."""
import numpy as np
import pytest

from bat_analysis.ridge_confidence import trim_weak_right_tail


def sample_track(weak_tail=0, *, all_strong=False):
    times = np.linspace(.005, .009, 61)
    freqs = np.linspace(25000, 140000, 231)
    columns = np.arange(4, 57)
    track_times = times[columns]
    track_freqs = np.linspace(117000., 55000., len(columns))
    curve = np.column_stack([
        track_times, track_freqs, np.ones(len(columns))
    ])
    power = np.full((len(freqs), len(times)), 1e-12)
    snr = np.zeros_like(power)
    for k, (ci, fi) in enumerate(zip(columns, track_freqs)):
        ri = int(np.argmin(abs(freqs - fi)))
        good = (k < len(columns) - weak_tail or all_strong)
        power[ri, ci] = 1.0 if good else 1e-5
        snr[ri, ci] = 24.0 if good else 3.0
    return curve, times, freqs, power, snr


def test_trims_only_sustained_weak_right_suffix():
    curve, t, f, power, snr = sample_track(weak_tail=12)
    unchanged = curve.copy()
    result, d = trim_weak_right_tail(
        curve, times_s=t, freqs_hz=f,
        stft_power=power, snr_db=snr, return_diagnostics=True)
    assert d["trimmed_right_points"] >= 11
    assert d["trailing_weak_points"] >= 12
    assert np.array_equal(result, unchanged[:len(result)])
    assert result[-1, 0] < curve[-1, 0]


def test_preserves_true_low_amplitude_tail_with_snr_support():
    curve, t, f, power, snr = sample_track(all_strong=True)
    # A true chirp often fades: do not trim a coherent 12 dB tail.
    for ci, fi in zip(range(45, 57), curve[-12:, 1]):
        ri = int(np.argmin(abs(f - fi)))
        power[ri, ci] = .07
        snr[ri, ci] = 15.
    result, d = trim_weak_right_tail(
        curve, times_s=t, freqs_hz=f,
        stft_power=power, snr_db=snr, return_diagnostics=True)
    np.testing.assert_array_equal(result, curve)
    assert d["trimmed_right_points"] == 0


def test_short_track_and_inner_weak_patch_unchanged():
    curve, t, f, power, snr = sample_track(all_strong=True)
    snr[:, 20:25] = 0
    original = curve.copy()
    assert np.array_equal(
        trim_weak_right_tail(
            curve, times_s=t, freqs_hz=f,
            stft_power=power, snr_db=snr), original
    )
    np.testing.assert_array_equal(
        trim_weak_right_tail(
            curve[:8], times_s=t, freqs_hz=f,
            stft_power=power, snr_db=snr), curve[:8]
    )


def test_invalid_matrices_and_thresholds():
    curve, t, f, power, snr = sample_track()
    with pytest.raises(ValueError, match="inconsistent shapes"):
        trim_weak_right_tail(
            curve, times_s=t, freqs_hz=f,
            stft_power=power[:-1], snr_db=snr)
    with pytest.raises(ValueError, match="thresholds"):
        trim_weak_right_tail(
            curve, times_s=t, freqs_hz=f,
            stft_power=power, snr_db=snr, min_snr_db=-1)
