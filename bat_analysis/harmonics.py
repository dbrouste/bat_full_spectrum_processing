"""Find second-harmonic segments guided by a previously tracked fundamental.

Input times are absolute WAV seconds. Harmonics are linked to their parent,
not independent bat calls. This mode is experimental and opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window


@dataclass
class HarmonicSegment:
    harmonic_order: int
    parent_id: str | int | None
    points: np.ndarray  # Nx3: time (s), frequency (Hz), local SNR (dB)
    mean_snr_db: float
    peak_snr_db: float
    median_frequency_ratio: float

    @property
    def t_start_s(self):
        return float(self.points[0, 0])

    @property
    def t_end_s(self):
        return float(self.points[-1, 0])


def detect_harmonic_segments(y, sr, fundamental_curve, *, parent_id=None,
                             harmonic_order=2, n_fft=192, hop=12,
                             fmax_hz=150000., frequency_tolerance_khz=20.,
                             min_snr_db=8., min_mean_snr_db=14.,
                             min_duration_ms=.18,
                             frequency_penalty_db_khz=.8, max_segments=1,
                             noise_context_ms=6.):
    """Recover an observed second harmonic close to 2x an existing ridge.

    A short Flattop STFT is calculated around the fundamental, with a noise
    percentile estimated from surrounding time. The two-to-one relation is a
    soft frequency prior, not a hard equality, so errors in the fundamental
    tail do not prevent finding a real harmonic. Output contains only observed
    high-SNR points; there is no artificial temporal extension.
    """
    y = np.asarray(y, dtype=float)
    curve = np.asarray(fundamental_curve, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all() or sr <= 0:
        raise ValueError("expected finite mono audio and positive sample rate")
    if curve.ndim != 2 or curve.shape[1] < 2 or len(curve) < 3:
        raise ValueError("fundamental_curve must have >=3 [time, frequency] rows")
    if not np.isfinite(curve).all() or not np.all(np.diff(curve[:, 0]) > 0):
        raise ValueError("curve must be finite and strictly time ordered")
    if harmonic_order != 2:
        raise ValueError("only the second harmonic is supported")
    n_fft = int(n_fft)
    hop = int(hop)
    if n_fft < 32 or not 1 <= hop <= n_fft or len(y) < n_fft:
        raise ValueError("invalid STFT size/hop")
    if (frequency_tolerance_khz <= 0 or min_duration_ms <= 0
            or max_segments < 1 or noise_context_ms <= 0
            or min_snr_db < 0 or min_mean_snr_db < 0):
        raise ValueError("invalid harmonic search parameters")
    lo = max(0, int((curve[0, 0] - noise_context_ms / 1000.) * sr))
    hi = min(len(y), int((curve[-1, 0] + noise_context_ms / 1000.) * sr))
    if hi - lo < n_fft:
        return []
    signal = y[lo:hi]
    window = get_window("flattop", n_fft, fftbins=True)
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames * window, axis=1).T) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (lo + np.arange(power.shape[1]) * hop + (n_fft - 1) / 2) / sr
    db = 10.0 * np.log10(np.maximum(power, 1e-30))
    noise_db = np.percentile(db, 22, axis=1)
    max_frequency = min(fmax_hz, sr / 2. - 3000.)
    candidates = []
    for j in np.flatnonzero((times >= curve[0, 0]) & (times <= curve[-1, 0])):
        expected = harmonic_order * np.interp(times[j], curve[:, 0], curve[:, 1])
        if not 35000 <= expected <= max_frequency:
            continue
        possible = np.flatnonzero(
            (np.abs(freqs - expected) < frequency_tolerance_khz * 1000.)
            & (freqs <= max_frequency))
        if not len(possible):
            continue
        scores = (
            db[possible, j] - noise_db[possible]
            - frequency_penalty_db_khz *
            np.abs(freqs[possible] - expected) / 1000.
        )
        index = int(possible[np.argmax(scores)])
        candidates.append((
            j, index, float(scores.max()),
            float(db[index, j] - noise_db[index]),
        ))
    if len(candidates) < 3:
        return []
    score_series = median_filter(
        np.asarray([c[2] for c in candidates]), size=3)
    active = score_series >= min_snr_db
    groups = []
    beginning = None
    for k in range(len(candidates) + 1):
        on = k < len(candidates) and bool(active[k])
        if on and beginning is None:
            beginning = k
        if beginning is not None and not on:
            group = candidates[beginning:k]
            if ((group[-1][0] - group[0][0] + 1) * hop / sr * 1000.
                    >= min_duration_ms):
                groups.append(group)
            beginning = None
    segments = []
    for group in groups:
        points = []
        for frame, index, _penalized_score, raw_snr in group:
            near = np.flatnonzero(np.abs(freqs - freqs[index]) <= 4000.)
            weights = np.maximum(power[near, frame], 1e-30)
            weights *= np.exp(
                -0.5 * ((freqs[near] - freqs[index]) / 2500.) ** 2)
            refined_freq = float(
                np.dot(freqs[near], weights) / np.sum(weights))
            points.append([times[frame], refined_freq, raw_snr])
        points = np.asarray(points, dtype=float)
        expected_parent = np.interp(
            points[:, 0], curve[:, 0], curve[:, 1])
        ratio = float(np.median(points[:, 1] / expected_parent))
        segments.append(HarmonicSegment(
            harmonic_order=harmonic_order,
            parent_id=parent_id,
            points=points,
            mean_snr_db=float(np.mean(points[:, 2])),
            peak_snr_db=float(np.max(points[:, 2])),
            median_frequency_ratio=ratio,
        ))
    segments = [s for s in segments if s.mean_snr_db >= min_mean_snr_db]
    # Keep the earliest coherent segment, not a louder delayed echo.
    segments.sort(key=lambda s: s.t_start_s)
    return segments[:max_segments]


def track_candidate_with_harmonics(y, sr, candidate, *, parent_id=None,
                                   tracking_kwargs=None, harmonic_kwargs=None):
    """Track a V4 candidate and find its second harmonic without manual seeds.

    Returns (fundamental_curve, harmonic_segments). Absolute times throughout.
    Harmonics must not be counted as independent calls or passages.
    """
    from .reassignment_tracking import track_reassigned_candidate
    parent = track_reassigned_candidate(
        y, sr, candidate, **(tracking_kwargs or {}))
    if parent is None:
        return None, []
    harmonics = detect_harmonic_segments(
        y, sr, parent, parent_id=parent_id, **(harmonic_kwargs or {}))
    return parent, harmonics
