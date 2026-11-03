"""Experimental Flattop time-frequency reassignment for bat ultrasound.

The input STFT and both auxiliary STFTs use the SAME window and frame grid.
Only energy above a frequency-wise SNR mask is scattered. No interpolation of
the waveform, no claimed increase in physical bandwidth or resolution.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def reassigned_spectrogram(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 512,
    hop: int | None = None,
    fmin: float = 20000.0,
    fmax: float = 150000.0,
    noise_percentile: float = 50.0,
    snr_min_db: float = 4.0,
    snr_full_db: float = 16.0,
    chunk_frames: int = 256,
) -> dict:
    """Return regular and reassigned power on the same (frequency,time) grid.

    Time reassignment: t_hat = t + Re(STFT(t*g) / STFT(g)).
    Frequency reassignment: f_hat = f - sr/(2*pi)*Im(STFT(g')/STFT(g)).
    The derivatives are with respect to sample position; the centered time
    window is measured in samples. Near-zero complex coefficients are ignored.

    Power is *summed* into the nearest cell rather than replacing it.
    Frequency masking is performed BEFORE reassignment to prevent out-of-band
    noise being deposited inside the target range.
    """
    x = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("y must be a finite 1-D mono waveform")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sr must be a positive integer")
    n_fft = int(n_fft)
    hop = n_fft // 4 if hop is None else int(hop)
    if n_fft < 32 or hop < 1 or hop > n_fft or len(x) < n_fft:
        raise ValueError("require n_fft >= 32, 1 <= hop <= n_fft and len(y) >= n_fft")
    if not 0 <= noise_percentile <= 100:
        raise ValueError("noise_percentile must be between 0 and 100")
    if not 0 <= snr_min_db < snr_full_db:
        raise ValueError("require 0 <= snr_min_db < snr_full_db")
    if chunk_frames < 1 or fmin < 0 or fmax <= fmin:
        raise ValueError("invalid chunk size or frequency band")

    g = get_window("flattop", n_fft, fftbins=True).astype(np.float64)
    u = np.arange(n_fft, dtype=np.float64) - (n_fft - 1) / 2.0
    gt = u * g
    gd = np.gradient(g)  # discrete approximation to dg/d(sample)
    starts = np.arange(0, len(x) - n_fft + 1, hop, dtype=np.int64)
    times = (starts + (n_fft - 1) / 2.0) / sr
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= min(fmax, sr / 2))
    if not np.any(band):
        raise ValueError("frequency band has no FFT bins")

    # Store complex STFT coefficients only in the requested band.
    selected = np.flatnonzero(band)
    z = np.empty((len(selected), len(starts)), dtype=np.complex128)
    zt = np.empty_like(z)
    zd = np.empty_like(z)
    for k in range(0, len(starts), chunk_frames):
        end = min(k + chunk_frames, len(starts))
        frames = np.stack([x[s:s + n_fft] for s in starts[k:end]])
        z[:, k:end] = np.fft.rfft(frames * g, axis=1)[:, selected].T
        zt[:, k:end] = np.fft.rfft(frames * gt, axis=1)[:, selected].T
        zd[:, k:end] = np.fft.rfft(frames * gd, axis=1)[:, selected].T

    power = np.abs(z) ** 2
    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    noise_db = np.percentile(power_db, noise_percentile, axis=1)
    snr_db = power_db - noise_db[:, None]
    fade = np.clip((snr_db - snr_min_db) / (snr_full_db - snr_min_db), 0, 1)
    # Avoid division by zero and unstable reassignment for tiny coefficients.
    stable = (fade > 0) & (power > max(float(power.max()), 1e-30) * 1e-14)
    ratio_t = np.zeros_like(z)
    ratio_d = np.zeros_like(z)
    np.divide(zt, z, out=ratio_t, where=stable)
    np.divide(zd, z, out=ratio_d, where=stable)
    t_hat = times[None, :] + np.real(ratio_t) / sr
    f_hat = freqs[selected, None] - (sr / (2 * np.pi)) * np.imag(ratio_d)

    # Reject estimates outside the target band / record rather than clipping
    # them to the edge and creating artificial ridges.
    fi = np.rint((f_hat - freqs[selected[0]]) / (sr / n_fft)).astype(np.int64)
    ti = np.rint((t_hat - times[0]) * sr / hop).astype(np.int64)
    valid = (stable & np.isfinite(t_hat) & np.isfinite(f_hat)
             & (fi >= 0) & (fi < len(selected))
             & (ti >= 0) & (ti < len(times)))
    assigned = np.zeros_like(power)
    np.add.at(assigned, (fi[valid], ti[valid]), (power * fade)[valid])
    return {
        "freqs_hz": freqs[selected],
        "times_s": times,
        "stft_power": power,
        "reassigned_power": assigned,
        "snr_db": snr_db,
        "noise_db": noise_db,
        "valid_fraction": float(np.mean(valid)),
        "n_fft": n_fft,
        "hop": hop,
        "window": "flattop",
    }


def reassigned_db(result: dict, *, reference: float | None = None) -> np.ndarray:
    """Reassigned spectrogram in dB relative to the *regular* STFT peak."""
    peak = (float(np.max(result["stft_power"])) if reference is None
            else float(reference))
    if peak <= 0:
        peak = 1e-30
    return 10.0 * np.log10(np.maximum(result["reassigned_power"], 1e-30) / peak)


def refine_detection_peaks(y: np.ndarray, sr: int, detections, *,
                           search_time_ms: float = 2.0,
                           search_freq_khz: float = 12.0, **kwargs):
    """Optional hybrid experiment: preserve detector timing/bounds/count.

    Returns NEW Detection objects with peak coordinates updated from local
    reassigned power; falls back to original coordinates when no valid energy.
    This does not claim to improve recall and does not change adaptive_v4.
    """
    from dataclasses import replace

    result = reassigned_spectrogram(y, sr, **kwargs)
    t = result["times_s"] * 1000.0
    f = result["freqs_hz"] / 1000.0
    p = result["reassigned_power"]
    refined = []
    for d in detections:
        tm = (np.abs(t - d.peak_time_ms) <= search_time_ms)
        fm = (np.abs(f - d.peak_freq_khz) <= search_freq_khz)
        if not np.any(tm) or not np.any(fm):
            refined.append(d)
            continue
        rows = np.flatnonzero(fm)
        cols = np.flatnonzero(tm)
        local = p[np.ix_(rows, cols)]
        if not np.any(local > 0):
            refined.append(d)
            continue
        i, j = np.unravel_index(int(np.argmax(local)), local.shape)
        refined.append(replace(d, peak_time_ms=float(t[cols[j]]),
                               peak_freq_khz=float(f[rows[i]])))
    return refined
