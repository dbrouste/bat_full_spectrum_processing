"""Conservative STFT-guided completion for reassigned bat-call ridges.

The completion recovers observed ridge energy rather than extrapolating points
outside the recording. All times are absolute (seconds).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import stft


def extend_ridge_edges(y, sr, curve, *, bounds_s, n_fft=512, hop=24,
                       max_extension_ms=5.0, search_radius_khz=10.0,
                       frequency_penalty=0.14, min_snr_db=17.0,
                       max_weak_frames=2, slope_momentum=0.75,
                       min_extension_points=5,
                       extend_left=True, extend_right=True,
                       return_diagnostics=False):
    """Complete a partial descending-FM ridge from a bounded signal region.

    Candidate points must follow an extrapolated local slope and match a
    strong local STFT SNR. The original points and amplitudes are unchanged.
    Small isolated apparent extensions are discarded. Optional diagnostics
    report the number of points added on each side.
    """
    x = np.asarray(y)
    path = np.asarray(curve, dtype=float)
    if x.ndim != 1 or sr <= 0:
        raise ValueError("expected finite mono recording and positive sr")
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 3:
        raise ValueError("curve must be Nx3 with at least three observations")
    if not np.isfinite(path).all() or not np.all(np.diff(path[:, 0]) > 0):
        raise ValueError("curve must be finite and strictly ordered by time")
    if len(bounds_s) != 2 or not np.isfinite(bounds_s).all():
        raise ValueError("bounds_s must contain two finite values")
    lower, upper = map(float, bounds_s)
    if lower >= upper or n_fft < 32 or not 1 <= hop <= n_fft or len(x) < n_fft:
        raise ValueError("invalid bounds, FFT size or hop")
    if not (max_extension_ms > 0 and search_radius_khz > 0
            and frequency_penalty >= 0 and min_snr_db >= 0
            and max_weak_frames >= 0 and 0 <= slope_momentum < 1
            and min_extension_points >= 1):
        raise ValueError("invalid extension parameters")

    left_bound = max(lower, path[0, 0] - max_extension_ms / 1000)
    right_bound = min(upper, path[-1, 0] + max_extension_ms / 1000)
    pad = 0.003
    s0 = max(0, int((left_bound - pad) * sr))
    s1 = min(len(x), int((right_bound + pad) * sr))
    if s1 - s0 < n_fft:
        result = path.copy()
        diag = {"left_points": 0, "right_points": 0}
        return (result, diag) if return_diagnostics else result
    # Keep the original float64 numerical path without copying an entire
    # multi-second WAV at every detection. Validate only the local segment.
    segment = np.asarray(x[s0:s1], dtype=np.float64)
    if not np.isfinite(segment).all():
        raise ValueError("expected finite mono recording and positive sr")
    freqs, times_local, coeff = stft(
        segment, fs=sr, window="flattop", nperseg=n_fft,
        noverlap=n_fft-hop, boundary=None, padded=False,
    )
    times = times_local + s0 / sr
    use = (freqs >= 20000) & (freqs <= min(150000, sr / 2))
    freqs = freqs[use]
    power = abs(coeff[use, :]) ** 2
    if len(freqs) < 2 or power.size == 0:
        result = path.copy()
        diag = {"left_points": 0, "right_points": 0}
        return (result, diag) if return_diagnostics else result
    log_power = 10 * np.log10(np.maximum(power, 1e-30))
    baseline = np.percentile(log_power, 20, axis=1)
    snr = gaussian_filter(
        np.maximum(0, log_power - baseline[:, None]), (.6, .35)
    )
    scale = max(float(np.max(power)), 1e-30)
    weak_skipped = {"left": 0, "right": 0}

    def side(direction):
        if direction == -1:
            seed = path[:min(6, len(path))]
            edge = path[0]
            stop = left_bound
        else:
            seed = path[-min(6, len(path)):]
            edge = path[-1]
            stop = right_bound
        fit_t = seed[:, 0] * 1000
        slope = float(
            np.polyfit(fit_t - fit_t[0], seed[:, 1], 1)[0] * 1000
        )
        # Do not impose downward FM on a CF or ascending call.
        if not np.isfinite(slope) or slope >= -1e5:
            return []
        reference_frame = int(np.argmin(abs(times - edge[0])))
        last_time, last_freq = float(edge[0]), float(edge[1])
        missing = 0
        saved = []
        for j in range(
            reference_frame + direction,
            len(times) if direction == 1 else -1, direction
        ):
            tj = float(times[j])
            if (direction < 0 and tj < stop) or (direction > 0 and tj > stop):
                break
            if (tj - last_time) * direction <= 0:
                continue
            pred = last_freq + slope * (tj - last_time)
            candidate = np.flatnonzero(
                abs(freqs - pred) <= search_radius_khz * 1000
            )
            if not len(candidate):
                break
            scores = (
                snr[candidate, j] - frequency_penalty *
                ((freqs[candidate] - pred) / 1000) ** 2
            )
            best = int(candidate[int(np.argmax(scores))])
            if snr[best, j] < min_snr_db:
                missing += 1
                weak_skipped["left" if direction == -1 else "right"] += 1
                if missing > max_weak_frames:
                    break
                # A weak frame can be BRIDGED when a coherent ridge resumes,
                # but must never become a returned trajectory point itself.
                continue
            missing = 0
            near = np.arange(max(0, best - 2), min(len(freqs), best + 3))
            weights = np.maximum(0, snr[near, j] - 1) * np.exp(
                -.5 * ((freqs[near] - freqs[best]) / 2500) ** 2
            )
            next_freq = (
                float(freqs[best]) if not weights.sum()
                else float(np.dot(freqs[near], weights) / weights.sum())
            )
            # Enforce the chirp's direction but permit small local jitter.
            if direction < 0 and next_freq < last_freq - 750:
                break
            if direction > 0 and next_freq > last_freq + 750:
                break
            observed_slope = (
                (next_freq - last_freq) / (tj - last_time)
            )
            slope = (
                slope_momentum * slope + (1 - slope_momentum) *
                np.clip(observed_slope, -120e6, -.1e6)
            )
            saved.append([
                tj, next_freq, float(np.sqrt(power[best, j] / scale))
            ])
            last_time, last_freq = tj, next_freq
        return saved if len(saved) >= min_extension_points else []

    before = side(-1) if extend_left else []
    after = side(1) if extend_right else []
    completed = np.concatenate([
        np.asarray(before[::-1], dtype=float).reshape(-1, 3),
        path.copy(),
        np.asarray(after, dtype=float).reshape(-1, 3),
    ])
    # Suppress rounding-induced duplicate timestamps.
    keep = np.concatenate(([True], np.diff(completed[:, 0]) > 0))
    completed = completed[keep]
    diag = {
        "left_points": len(before),
        "right_points": len(after),
        "weak_skipped_left": weak_skipped["left"],
        "weak_skipped_right": weak_skipped["right"],
        "min_snr_db": float(min_snr_db),
        "start_time_s": float(completed[0, 0]),
        "end_time_s": float(completed[-1, 0]),
        "coverage_fraction": float(np.clip(
            (completed[-1, 0] - completed[0, 0]) / (upper - lower), 0, 1
        )),
    }
    return (completed, diag) if return_diagnostics else completed
