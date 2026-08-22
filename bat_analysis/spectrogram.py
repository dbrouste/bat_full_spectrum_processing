from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


def read_wav(path: Path | str) -> Tuple[np.ndarray, int]:
    """Read a WAV as mono float32."""
    path = Path(path)
    x, sr = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError(f"Empty WAV: {path}")
    return x, int(sr)


def compute_spectrogram(
    path: Path | str,
    fmin_khz: float = 20.0,
    fmax_khz: float = 180.0,
    nperseg: int = 1024,
    noverlap: int = 896,
) -> Dict[str, Any]:
    """Compute a relative-dB STFT spectrogram for annotation and inspection."""
    path = Path(path)
    x, sr = read_wav(path)

    nperseg_eff = min(int(nperseg), len(x))
    if nperseg_eff < 32:
        raise ValueError(f"WAV too short for spectrogram: {path}")
    noverlap_eff = min(int(noverlap), nperseg_eff - 1)

    freqs, times, z = signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        boundary=None,
        padded=False,
    )

    power = np.abs(z) ** 2
    # Avoid forming power/ref before log10: with float32 STFT data, very small
    # ratios can underflow to exactly zero and emit a divide-by-zero warning.
    # log10(P) - log10(Pref) is mathematically equivalent and numerically safer.
    tiny = np.finfo(power.dtype).tiny
    power_safe = np.maximum(power, tiny)
    ref = max(float(np.max(power_safe)), float(tiny))
    db = 10.0 * (np.log10(power_safe) - np.log10(ref))

    f_khz = freqs / 1000.0
    t_ms = times * 1000.0
    nyquist_khz = sr / 2000.0
    keep = (f_khz >= fmin_khz) & (f_khz <= min(fmax_khz, nyquist_khz))
    if not np.any(keep):
        raise ValueError(
            f"No spectrogram bins in requested band {fmin_khz:.1f}-{fmax_khz:.1f} kHz "
            f"for {path.name} (Fs={sr} Hz)"
        )

    return {
        "sr": sr,
        "duration_ms": len(x) / sr * 1000.0,
        "times_ms": t_ms.astype(float),
        "freqs_khz": f_khz[keep].astype(float),
        "db": db[keep].astype(float),
        "path": str(path),
    }
