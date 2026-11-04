"""Experimental global ridge tracker combining Flattop STFT and reassignment.

The second-order dynamic program retains slope continuity, in contrast to the
previous greedy maximum follower. Detector seeds remain optional.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .reassignment import reassigned_spectrogram


def track_reassigned_ridge(
    y, sr, *, time_mid, duration, seed_freq_hz=None,
    n_fft=512, hop=24, search_margin_ms=1.0,
    max_step_hz=None, jump_penalty=0.11, gap_penalty=0.65,
    max_gap_frames=3, snr_min_db=4.0, snr_full_db=16.0,
    subbin_refine=True, diagnostics=None,
):
    """Return absolute-time ridge [time_s, freq_hz, normalized amplitude].

    The optimization uses frequency and velocity as states. The forward-time
    chirp is non-increasing in frequency. No annotations are used at inference.
    Input time_mid and duration are in seconds.
    """
    if duration <= 0 or sr <= 0 or n_fft < 32 or hop < 1 or hop > n_fft:
        raise ValueError("invalid sample rate, duration or STFT configuration")
    if jump_penalty < 0 or gap_penalty < 0:
        raise ValueError("penalties must be nonnegative")
    if max_step_hz is not None and max_step_hz <= 0:
        raise ValueError("max_step_hz must be positive")
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("expected finite mono audio")
    start = max(0.0, time_mid - duration / 2 - search_margin_ms / 1000)
    stop = min(len(y) / sr, time_mid + duration / 2 + search_margin_ms / 1000)
    first = max(0, int(start * sr) - n_fft)
    last = min(len(y), int(stop * sr) + n_fft)
    if last - first < n_fft:
        return None
    # Limit reassignment computation to the local window. Keep the full
    # 20-150 kHz band so large FM sweeps do not get cut off by a seed.
    r = reassigned_spectrogram(
        y[first:last], sr, n_fft=n_fft, hop=hop,
        fmin=20000, fmax=min(150000, sr / 2),
        noise_percentile=20, snr_min_db=snr_min_db,
        snr_full_db=snr_full_db,
    )
    freqs = r["freqs_hz"]
    times_all = r["times_s"] + first / sr
    keep = (times_all >= start) & (times_all <= stop)
    t = times_all[keep]
    regular = r["stft_power"][:, keep]
    reassigned = r["reassigned_power"][:, keep]
    snr = r["snr_db"][:, keep]
    if len(t) < 4 or len(freqs) < 2 or not reassigned.any():
        return None

    # A smooth STFT contribution bridges sparse reassigned pixels. Both
    # components use a frequency-wise noise floor to suppress fixed tones.
    focused = gaussian_filter(reassigned, (1.2, 0.7), mode="constant")
    broad = gaussian_filter(regular, (1.0, 0.5), mode="constant")
    focused_db = 10 * np.log10(np.maximum(focused, 1e-30))
    broad_db = 10 * np.log10(np.maximum(broad, 1e-30))
    floor = np.percentile(broad_db, 20, axis=1)
    evidence = np.clip(
        0.65 * np.maximum(0, focused_db - floor[:, None] - 2)
        + 0.35 * np.maximum(0, broad_db - floor[:, None] - 3),
        0, 32,
    ) / 32
    evidence[snr < 2] *= 0.3

    time_candidates = np.flatnonzero(abs(t - time_mid) <= 0.0015)
    if seed_freq_hz is not None and np.isfinite(seed_freq_hz):
        freq_candidates = np.flatnonzero(abs(freqs - seed_freq_hz) <= 12000)
    else:
        freq_candidates = np.arange(len(freqs))
    if not len(time_candidates) or not len(freq_candidates):
        return None
    seeds = evidence[np.ix_(freq_candidates, time_candidates)]
    if seeds.max() < (0.08 if duration < 0.0012 else 0.13):
        if diagnostics is not None:
            diagnostics.update({"model_success": False, "stop_reason": "weak_seed"})
        return None
    a, b = np.unravel_index(np.argmax(seeds), seeds.shape)
    anchor_row = int(freq_candidates[a])
    anchor_col = int(time_candidates[b])

    df = sr / n_fft
    dt = hop / sr
    # Default upper bound 45 kHz/ms; physical steep FM calls otherwise
    # exceed the previous 7.5 kHz/frame cap.
    upper_step = (110000000 if duration < 0.0012 else 45000000) * dt if max_step_hz is None else max_step_hz
    max_velocity = max(2, min(int(round(upper_step / df)), len(freqs) - 1))
    velocities = np.arange(max_velocity + 1)
    rows, columns = evidence.shape

    def follow(direction):
        score = np.full((rows, len(velocities)), -1e9)
        score[anchor_row, :] = 0.0
        backtrace = []
        best_score = -1e9
        best_depth = 0
        best_row = anchor_row
        best_velocity = 0
        row_indices = np.arange(rows)
        for depth in range(1, columns):
            col = anchor_col + direction * depth
            if col < 0 or col >= columns:
                break
            new = np.full_like(score, -1e9)
            prev_rows = np.full(score.shape, -1, dtype=np.int32)
            prev_vel = np.full(score.shape, -1, dtype=np.int16)
            for v in velocities:
                # Frequency declines as time increases, in both directions.
                previous = row_indices + direction * v
                valid = (previous >= 0) & (previous < rows)
                if not valid.any():
                    continue
                delta_v = velocities - v
                transition = score[previous[valid], :] - jump_penalty * delta_v[None, :] ** 2
                choices = np.argmax(transition, axis=1)
                rr = row_indices[valid]
                new[rr, v] = (
                    transition[np.arange(len(rr)), choices]
                    + 2.5 * evidence[rr, col] - gap_penalty
                )
                prev_rows[rr, v] = previous[valid]
                prev_vel[rr, v] = choices
            backtrace.append((prev_rows, prev_vel))
            score = new
            row, vel = np.unravel_index(np.argmax(score), score.shape)
            if score[row, vel] > best_score:
                best_score = float(score[row, vel])
                best_depth = depth
                best_row = int(row)
                best_velocity = int(vel)
        if best_depth == 0:
            return []
        row, vel = best_row, best_velocity
        path = []
        for depth in range(best_depth, 0, -1):
            path.append((anchor_col + direction * depth, row))
            previous_rows, previous_vel = backtrace[depth - 1]
            row, vel = int(previous_rows[row, vel]), int(previous_vel[row, vel])
            if row < 0:
                break
        return path

    path = sorted(set(follow(-1) + [(anchor_col, anchor_row)] + follow(1)))
    if len(path) < (3 if duration < 0.0012 else 4):
        if diagnostics is not None:
            diagnostics.update({"model_success": False, "stop_reason": "too_few_points"})
        return None
    col_idx = np.asarray([p[0] for p in path], dtype=int)
    row_idx = np.asarray([p[1] for p in path], dtype=int)
    minimum_duration = (min(0.00025, 0.3 * duration) if duration < 0.0012 else 0.0005)
    if t[col_idx[-1]] - t[col_idx[0]] < minimum_duration:
        if diagnostics is not None:
            diagnostics.update({"model_success": False, "stop_reason": "too_short"})
        return None
    # The reassigned image is already concentrated around the ridge.
    # Refine the FFT-cell centre with its original (unsmoothed) power in a
    # bounded +/-3-bin neighborhood. A Gaussian proximity kernel stops a
    # distant harmonic pulling the centroid away from the selected path.
    ridge_frequencies = freqs[row_idx].astype(float).copy()
    if subbin_refine:
        for j, (ri, ci) in enumerate(zip(row_idx, col_idx)):
            neighbors = np.arange(max(0, ri - 3), min(rows, ri + 4))
            weights = np.maximum(0.0, reassigned[neighbors, ci]) * np.exp(
                -0.5 * ((neighbors - ri) / 2.25) ** 2
            )
            if np.sum(weights) > 1e-30:
                ridge_frequencies[j] = float(
                    np.dot(freqs[neighbors], weights) / np.sum(weights)
                )
    amplitude = np.sqrt(regular[row_idx, col_idx] / max(regular.max(), 1e-30))
    curve = np.column_stack([t[col_idx], ridge_frequencies, amplitude])
    if diagnostics is not None:
        diagnostics.update({
            "tracking_method": "reassignment_flattop",
            "tracking_algorithm": "second_order_global_dp",
            "subbin_refine": bool(subbin_refine),
            "reassignment_n_fft": int(n_fft),
            "reassignment_hop": int(hop),
            "reassignment_valid_fraction": r["valid_fraction"],
            "model_success": True,
            "final_points": int(len(curve)),
        })
    return curve
