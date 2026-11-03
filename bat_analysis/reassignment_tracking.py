"""Experimental ridge tracking on a Flattop reassigned spectrogram.

Independent from the Gaussian legacy tracker. The path is extracted with a
seed-constrained, continuity-penalized tracking rather than independent column maxima.
"""
from __future__ import annotations

import numpy as np
from .reassignment import reassigned_spectrogram


def track_reassigned_ridge(y, sr, *, time_mid, duration,
                           seed_freq_hz=None, n_fft=512, hop=64,
                           search_margin_ms=1.5, max_step_hz=7500.0,
                           jump_penalty=1.5, gap_penalty=2.0,
                           max_gap_frames=3, snr_min_db=4.0,
                           snr_full_db=16.0, diagnostics=None):
    """Extract ridge as N x 3 [relative time (s), frequency (Hz), amplitude].

    time_mid, duration: seconds relative to supplied audio buffer.
    Returns None when insufficient seeded coherent energy is available.
    Amplitude is normalized to [0,1], NOT legacy Gaussian amplitude.
    """
    if duration <= 0 or max_step_hz <= 0 or max_gap_frames < 0:
        raise ValueError("invalid tracking parameters")
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("expected mono audio")
    center = float(time_mid)
    start = max(0.0, center - float(duration)/2 - search_margin_ms/1000)
    stop = min(len(y)/sr, center + float(duration)/2 + search_margin_ms/1000)
    # Work on the short excerpt, not the entire recording.
    first = max(0, int(np.floor(start*sr)) - n_fft)
    last = min(len(y), int(np.ceil(stop*sr)) + n_fft)
    if last - first < n_fft:
        return None
    clip = y[first:last]
    fmin = 20000.0
    fmax = min(150000.0, sr/2)
    if seed_freq_hz is not None and np.isfinite(seed_freq_hz):
        # Allow large chirp excursions while excluding distant harmonics.
        fmin = max(fmin, float(seed_freq_hz) - 50000)
        fmax = min(fmax, float(seed_freq_hz) + 50000)
    if fmax <= fmin:
        return None
    data = reassigned_spectrogram(
        clip, sr, n_fft=n_fft, hop=hop, fmin=fmin, fmax=fmax,
        noise_percentile=20, snr_min_db=snr_min_db, snr_full_db=snr_full_db,
    )
    ts = data["times_s"] + first/sr
    fs = data["freqs_hz"]
    power = data["reassigned_power"]
    keep_t = (ts >= start) & (ts <= stop)
    if not np.any(keep_t):
        return None
    ts = ts[keep_t]
    power = power[:, keep_t]
    rows, cols = power.shape
    if cols < 3 or rows < 2 or not np.any(power > 0):
        return None

    # Keep only bins with real assigned energy; all zero bins are forbidden.
    peak = float(power.max())
    weights = np.full(power.shape, -np.inf, dtype=float)
    active = power > 0
    weights[active] = np.log1p(power[active] / peak * 100)
    # Seed in a narrow time window; fallback to best available near center.
    seed_cols = np.where(np.abs(ts - center) <= max(2*hop/sr, 0.001))[0]
    if seed_cols.size == 0:
        seed_cols = np.array([int(np.argmin(np.abs(ts-center)))])
    if seed_freq_hz is not None and np.isfinite(seed_freq_hz):
        seed_rows = np.where(np.abs(fs - seed_freq_hz) <= 12000)[0]
    else:
        seed_rows = np.arange(rows)
    if seed_rows.size == 0:
        return None
    sub = weights[np.ix_(seed_rows,seed_cols)]
    if not np.isfinite(sub).any():
        return None
    seed = np.unravel_index(np.argmax(sub), sub.shape)
    anchor_r, anchor_c = int(seed_rows[seed[0]]), int(seed_cols[seed[1]])

    step_bins = max(1, int(np.ceil(max_step_hz / (sr/n_fft))))
    # Greedy continuity-penalized tracking in both directions. Bridges up to max_gap_frames
    # missing columns; no unrelated components can start new paths.
    points = [(anchor_c, anchor_r)]
    for direction in (-1, 1):
        current_c, current_r = anchor_c, anchor_r
        gap = 0
        history = []
        while 0 <= current_c + direction < cols:
            target_c = current_c + direction
            candidate = np.arange(max(0,current_r-step_bins),
                                  min(rows,current_r+step_bins+1))
            valid = candidate[np.isfinite(weights[candidate,target_c])]
            if valid.size:
                # Small penalty for large frequency jumps.
                scores = weights[valid,target_c] - jump_penalty * (
                    np.abs(valid-current_r)/max(step_bins,1))**2 - gap_penalty*gap
                nr = int(valid[int(np.argmax(scores))])
                history.append((target_c,nr))
                current_r = nr
                gap = 0
            else:
                gap += 1
                if gap > max_gap_frames:
                    break
            current_c = target_c
        points.extend(history)
    points = sorted(set(points))
    if len(points) < 3:
        return None
    col_idx = np.asarray([p[0] for p in points],dtype=int)
    row_idx = np.asarray([p[1] for p in points],dtype=int)
    tt = ts[col_idx]
    ff = fs[row_idx]
    amp = np.sqrt(power[row_idx,col_idx]/peak)
    curve = np.column_stack([tt,ff,amp])
    if diagnostics is not None:
        diagnostics.update({
            "tracking_method": "reassignment_flattop",
            "reassignment_n_fft": int(n_fft),
            "reassignment_hop": int(hop),
            "reassignment_points": int(len(curve)),
            "reassignment_valid_fraction": data["valid_fraction"],
            "model_success": True,
            "final_points": int(len(curve)),
        })
    return curve
