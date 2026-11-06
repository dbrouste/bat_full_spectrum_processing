"""Numerical regression checks for the performance-sensitive FFT kernels."""
import numpy as np
import pytest
from scipy.signal import get_window

from bat_analysis.reassignment import reassigned_spectrogram


def slow_reference_reassignment(y, sr, *, n_fft, hop,
                                fmin=20000.0, fmax=150000.0,
                                noise_percentile=20.0,
                                snr_min_db=4.0, snr_full_db=16.0):
    """Original frame-stacking and np.add.at reference implementation."""
    x = np.asarray(y, dtype=np.float64)
    g = get_window("flattop", n_fft, fftbins=True).astype(np.float64)
    u = np.arange(n_fft, dtype=np.float64) - (n_fft - 1) / 2
    starts = np.arange(0, len(x) - n_fft + 1, hop, dtype=np.int64)
    times = (starts + (n_fft - 1) / 2.0) / sr
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = np.flatnonzero((f >= fmin) & (f <= min(fmax, sr / 2)))
    frames = np.stack([x[k:k + n_fft] for k in starts])
    z = np.fft.rfft(frames * g, axis=1)[:, sel].T
    zt = np.fft.rfft(frames * (u * g), axis=1)[:, sel].T
    zd = np.fft.rfft(frames * np.gradient(g), axis=1)[:, sel].T
    power = np.abs(z) ** 2
    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    noise_db = np.percentile(power_db, noise_percentile, axis=1)
    snr = power_db - noise_db[:, None]
    fade = np.clip((snr - snr_min_db) / (snr_full_db - snr_min_db), 0, 1)
    stable = (fade > 0) & (
        power > max(float(power.max()), 1e-30) * 1e-14)
    rt = np.zeros_like(z)
    rd = np.zeros_like(z)
    np.divide(zt, z, out=rt, where=stable)
    np.divide(zd, z, out=rd, where=stable)
    that = times[None, :] + np.real(rt) / sr
    fhat = f[sel, None] - (sr / (2 * np.pi)) * np.imag(rd)
    fi = np.rint((fhat - f[sel[0]]) / (sr / n_fft)).astype(np.int64)
    ti = np.rint((that - times[0]) * sr / hop).astype(np.int64)
    valid = (stable & np.isfinite(that) & np.isfinite(fhat)
             & (fi >= 0) & (fi < len(sel))
             & (ti >= 0) & (ti < len(times)))
    assigned = np.zeros_like(power)
    np.add.at(assigned, (fi[valid], ti[valid]), (power * fade)[valid])
    return {"stft_power": power, "reassigned_power": assigned,
            "snr_db": snr, "noise_db": noise_db,
            "times_s": times, "freqs_hz": f[sel],
            "valid_fraction": float(np.mean(valid))}


@pytest.mark.parametrize("n_fft,hop", [(192, 16), (512, 24), (512, 64)])
def test_strided_fft_and_bincount_match_original(n_fft, hop):
    sr = 384000
    rng = np.random.default_rng(20261106)
    t = np.arange(4096) / sr
    x = (.4 * np.sin(2 * np.pi * (110000 * t - .5 * 5e6 * t**2))
         + .06 * rng.standard_normal(len(t)))
    reference = slow_reference_reassignment(
        x, sr, n_fft=n_fft, hop=hop
    )
    result = reassigned_spectrogram(
        x, sr, n_fft=n_fft, hop=hop,
        noise_percentile=20, snr_min_db=4, snr_full_db=16,
    )
    for key in reference:
        np.testing.assert_allclose(result[key], reference[key],
                                   rtol=1e-12, atol=1e-12)


def test_noncontiguous_input_matches_same_samples():
    sr = 384000
    rng = np.random.default_rng(55)
    interleaved = rng.standard_normal(8192)
    x = interleaved[::2]  # noncontiguous float64 input
    a = reassigned_spectrogram(x, sr, n_fft=192, hop=16)
    b = reassigned_spectrogram(x.copy(), sr, n_fft=192, hop=16)
    for key in ("stft_power", "reassigned_power", "snr_db"):
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0)


def test_vectorized_dp_tracking_is_deterministic():
    from bat_analysis.reassignment_tracking import track_reassigned_ridge
    sr = 384000
    t = np.arange(int(.018 * sr)) / sr
    signal = np.zeros_like(t)
    mask = (t >= .006) & (t <= .011)
    local = t[mask] - .006
    signal[mask] = .7 * np.sin(
        2 * np.pi * (110000 * local - .5 * 70000 / .005 * local**2)
    )
    kwargs = dict(
        time_mid=.0085, duration=.005, seed_freq_hz=80000,
        complete_ends=False, trim_weak_tail=False,
    )
    a = track_reassigned_ridge(signal, sr, **kwargs)
    b = track_reassigned_ridge(signal, sr, **kwargs)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a, b)
