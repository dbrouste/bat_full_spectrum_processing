"""Experimental global ridge tracker combining Flattop STFT and reassignment.

The second-order dynamic program retains slope continuity, in contrast to the
previous greedy maximum follower. Detector seeds remain optional.
"""
from __future__ import annotations
import numpy as np
import time
from scipy.ndimage import gaussian_filter
from .reassignment import reassigned_spectrogram


def choose_reassignment_resolution(duration_s, n_fft=None, hop=None):
    """Select temporal resolution for steep sub-millisecond chirps.

    Explicit n_fft/hop values override automatic selection. The 1.2 ms
    boundary is experimental and should be benchmarked on additional WAVs.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    if n_fft is None:
        n_fft = 192 if duration_s < 0.0012 else 512
    n_fft = int(n_fft)
    if hop is None:
        hop = 16 if n_fft <= 256 else 24
    hop = int(hop)
    if n_fft < 32 or hop < 1 or hop > n_fft:
        raise ValueError("invalid FFT size or hop")
    return n_fft, hop


def track_reassigned_ridge(
    y, sr, *, time_mid, duration, seed_freq_hz=None,
    n_fft=None, hop=None, search_margin_ms=1.0,
    max_step_hz=None, jump_penalty=0.11, gap_penalty=0.65,
    max_gap_frames=3, snr_min_db=4.0, snr_full_db=16.0,
    subbin_refine=True, anchor_time_radius_ms=0.5,
    anchor_freq_radius_khz=12.0, time_bounds_s=None,
    cross_seed_velocity_penalty=0.6, cross_seed_position_penalty=0.01,
    prior_horizon_ms=0.55, refine_endpoints=True,
    endpoint_tail_fraction=0.35, complete_ends=True,
    extension_min_snr_db=17.0, peak_anchored_window=False,
    trim_weak_tail=True, tail_min_snr_db=10.0,
    tail_min_relative_db=-22.0, diagnostics=None,
):
    """Return absolute-time ridge [time_s, freq_hz, normalized amplitude].

    The optimization uses frequency and velocity as states. The forward-time
    chirp is non-increasing in frequency. No annotations are used at inference.
    Input time_mid and duration are in seconds.
    """
    _profile_start = time.perf_counter() if diagnostics is not None else None
    n_fft, hop = choose_reassignment_resolution(duration, n_fft, hop)
    if sr <= 0:
        raise ValueError("invalid sample rate")
    if jump_penalty < 0 or gap_penalty < 0:
        raise ValueError("penalties must be nonnegative")
    if max_step_hz is not None and max_step_hz <= 0:
        raise ValueError("max_step_hz must be positive")
    if extension_min_snr_db < 0 or tail_min_snr_db < 0:
        raise ValueError("SNR thresholds must be nonnegative")
    if tail_min_relative_db >= 0:
        raise ValueError("tail_min_relative_db must be negative")
    if anchor_time_radius_ms <= 0 or anchor_freq_radius_khz <= 0:
        raise ValueError("anchor radii must be positive")
    if (cross_seed_velocity_penalty < 0 or cross_seed_position_penalty < 0
            or prior_horizon_ms < 0):
        raise ValueError("cross-seed penalties and horizon must be nonnegative")
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("expected finite mono audio")
    if time_bounds_s is None:
        if peak_anchored_window:
            # V4 time_mid is the ENERGY PEAK, not the bbox centre.
            # Descending FM calls frequently peak close to their end.
            start = time_mid - 1.15 * duration
            stop = time_mid + 0.85 * duration
        else:
            start = time_mid - duration / 2
            stop = time_mid + duration / 2
    else:
        if len(time_bounds_s) != 2:
            raise ValueError("time_bounds_s must be (start_s, end_s)")
        start, stop = (float(time_bounds_s[0]), float(time_bounds_s[1]))
        if not np.isfinite([start, stop]).all() or start >= stop:
            raise ValueError("invalid time_bounds_s")
    start = max(0.0, start - search_margin_ms / 1000)
    stop = min(len(y) / sr, stop + search_margin_ms / 1000)
    first = max(0, int(start * sr) - n_fft)
    last = min(len(y), int(stop * sr) + n_fft)
    if last - first < n_fft:
        return None
    # Limit reassignment computation to the local window. Keep the full
    # 20-150 kHz band so large FM sweeps do not get cut off by a seed.
    # Only this local window is used; avoid scanning the entire recording
    # for NaN/Inf values on every chirp in a multi-second WAV.
    segment = np.asarray(y[first:last], dtype=np.float64)
    if not np.isfinite(segment).all():
        raise ValueError("expected finite mono audio")
    r = reassigned_spectrogram(
        segment, sr, n_fft=n_fft, hop=hop,
        fmin=20000, fmax=min(150000, sr / 2),
        noise_percentile=20, snr_min_db=snr_min_db,
        snr_full_db=snr_full_db,
    )
    _profile_fft = time.perf_counter() if diagnostics is not None else None
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

    time_candidates = np.flatnonzero(
        abs(t - time_mid) <= max(anchor_time_radius_ms / 1000.0, hop / (2 * sr))
    )
    if seed_freq_hz is not None and np.isfinite(seed_freq_hz):
        freq_candidates = np.flatnonzero(
            abs(freqs - seed_freq_hz) <= anchor_freq_radius_khz * 1000.0
        )
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

    def follow(direction, prior_velocity=None):
        """Exact vectorized DP over frequency-bin and slope-bin states.

        Precompute the admissible predecessor frequencies for every
        possible velocity, then maximize across predecessor velocities in
        one NumPy operation per frame. This preserves the old transition
        scores and tie-breaking, without the Python loop over velocities.
        """
        score = np.full((rows, len(velocities)), -1e9)
        if prior_velocity is None:
            score[anchor_row, :] = 0.0
        else:
            score[anchor_row, :] = (
                -cross_seed_velocity_penalty * (velocities - prior_velocity) ** 2
            )

        row_indices = np.arange(rows)
        prev = row_indices[:, None] + direction * velocities[None, :]
        admissible = (prev >= 0) & (prev < rows)
        safe_prev = np.clip(prev, 0, rows - 1)
        previous_rows = np.where(admissible, prev, -1).astype(np.int32)
        # Layout: [candidate velocity, previous velocity].
        transition_cost = jump_penalty * (
            velocities[:, None] - velocities[None, :]
        ) ** 2
        backtrace = []
        best_score = -1e9
        best_depth = 0
        best_row = anchor_row
        best_velocity = 0

        for depth in range(1, columns):
            col = anchor_col + direction * depth
            if col < 0 or col >= columns:
                break

            transitions = score[safe_prev, :] - transition_cost[None, :, :]
            choices = np.argmax(transitions, axis=2)
            best_transition = np.take_along_axis(
                transitions, choices[..., None], axis=2
            )[..., 0]
            new = best_transition + 2.5 * evidence[:, col, None] - gap_penalty
            if (prior_velocity is not None
                    and depth * dt < prior_horizon_ms / 1000.0):
                predicted_row = anchor_row - prior_velocity * depth
                new -= cross_seed_position_penalty * (
                    row_indices[:, None] - predicted_row
                ) ** 2
            new[~admissible] = -1e9
            previous_velocities = np.where(
                admissible, choices, -1
            ).astype(np.int16)
            backtrace.append((previous_rows, previous_velocities))
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

    _profile_dp_begin = time.perf_counter() if diagnostics is not None else None
    left = follow(-1)
    prior_velocity = None
    if len(left) >= 3:
        ordered = sorted(left + [(anchor_col, anchor_row)])
        last_rows = np.asarray([row for _, row in ordered[-8:]])
        prior_velocity = float(np.clip(
            np.median(-np.diff(last_rows)), 0, max_velocity
        ))
    path = sorted(set(left + [(anchor_col, anchor_row)]
                      + follow(1, prior_velocity)))
    _profile_dp_end = time.perf_counter() if diagnostics is not None else None
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
    _profile_path = time.perf_counter() if diagnostics is not None else None
    completion_diag = None
    if complete_ends and len(curve) >= 3:
        from .ridge_completion import extend_ridge_edges
        curve, completion_diag = extend_ridge_edges(
            y, sr, curve, bounds_s=(start, stop),
            n_fft=n_fft, hop=hop,
            min_snr_db=extension_min_snr_db,
            return_diagnostics=True,
        )
    _profile_completed = time.perf_counter() if diagnostics is not None else None
    tail_diag = None
    if trim_weak_tail:
        # Apply confidence checks to the ENTIRE observed ridge (global DP
        # plus any completion), before the frequency-only tail correction.
        # Original STFT power and SNR are compared at each ridge location.
        from .ridge_confidence import trim_weak_right_tail
        curve, tail_diag = trim_weak_right_tail(
            curve, times_s=t, freqs_hz=freqs,
            stft_power=regular, snr_db=snr,
            min_snr_db=tail_min_snr_db,
            min_relative_db=tail_min_relative_db,
            return_diagnostics=True,
        )
    _profile_trimmed = time.perf_counter() if diagnostics is not None else None
    endpoint_diag = None
    if refine_endpoints:
        from .endpoint_refinement import refine_ridge_endpoint
        curve, endpoint_diag = refine_ridge_endpoint(
            curve, tail_fraction=endpoint_tail_fraction,
            times_s=t, freqs_hz=freqs, stft_power=regular,
            return_diagnostics=True,
        )
    if diagnostics is not None:
        _profile_end = time.perf_counter()
        diagnostics["timings_ms"] = {
            "reassignment_fft": round(1000 * (_profile_fft - _profile_start), 3),
            "evidence_and_setup": round(1000 * (_profile_dp_begin - _profile_fft), 3),
            "global_dp": round(1000 * (_profile_dp_end - _profile_dp_begin), 3),
            "ridge_sampling": round(1000 * (_profile_path - _profile_dp_end), 3),
            "completion": round(1000 * (_profile_completed - _profile_path), 3),
            "confidence": round(1000 * (_profile_trimmed - _profile_completed), 3),
            "endpoint": round(1000 * (_profile_end - _profile_trimmed), 3),
            "total": round(1000 * (_profile_end - _profile_start), 3),
        }
        diagnostics.update({
            "tracking_method": "reassignment_flattop",
            "tracking_algorithm": "second_order_global_dp",
            "cross_seed_velocity_bins_per_frame": prior_velocity,
            "cross_seed_position_penalty": float(cross_seed_position_penalty),
            "subbin_refine": bool(subbin_refine),
            "endpoint_refinement": endpoint_diag,
            "ridge_completion": completion_diag,
            "tail_confidence": tail_diag,
            "trim_weak_tail": bool(trim_weak_tail),
            "complete_ends": bool(complete_ends),
            "peak_anchored_window": bool(peak_anchored_window),
            "refine_endpoints": bool(refine_endpoints),
            "reassignment_n_fft": int(n_fft),
            "reassignment_hop": int(hop),
            "anchor_time_offset_ms": float((t[anchor_col] - time_mid) * 1000.0),
            "anchor_freq_offset_khz": (
                float((freqs[anchor_row] - seed_freq_hz) / 1000.0)
                if seed_freq_hz is not None and np.isfinite(seed_freq_hz) else None
            ),
            "reassignment_valid_fraction": r["valid_fraction"],
            "model_success": True,
            "final_points": int(len(curve)),
        })
    return curve


def track_reassigned_candidate(y, sr, candidate: dict, *, use_broad_seed=True,
                               diagnostics=None, **kwargs):
    """Track directly from a detector candidate using its true time bounds.

    Adaptive V4 reports `time_mid` at its strongest peak, not necessarily
    the middle of the chirp. Use the *bbox* to set the analysis limits, while
    keeping the peak time and peak frequency together as the seed. When a
    valid coarse broad-FM seed is present, use its own matched time coordinate.
    """
    seed_time = float(candidate["time_mid"])
    seed_freq = float(candidate["peak_freq_hz"])
    if use_broad_seed:
        coarse_freq = candidate.get("broad_fm_seed_freq_hz")
        coarse_time = candidate.get("coarse_seed_time_s")
        if (coarse_freq is not None and coarse_time is not None
                and np.isfinite(coarse_freq) and np.isfinite(coarse_time)):
            seed_time = float(coarse_time)
            seed_freq = float(coarse_freq)
    bounds = None
    if candidate.get("t_start") is not None and candidate.get("t_end") is not None:
        a, b = float(candidate["t_start"]), float(candidate["t_end"])
        if np.isfinite([a, b]).all() and a < b:
            bounds = (a, b)
    return track_reassigned_ridge(
        y, sr,
        time_mid=seed_time,
        duration=float(candidate["duration"]),
        seed_freq_hz=seed_freq,
        time_bounds_s=bounds,
        diagnostics=diagnostics,
        **kwargs,
    )


def track_and_group_candidates(
    y, sr, candidates=None, *, tracking_kwargs=None, grouping_kwargs=None,
):
    """Track Adaptive V4 candidates and group their overlapping trajectories.

    Returns a dict with 'groups', 'untracked_candidate_indices',
    'candidate_count' and 'tracking_diagnostics'. Each group contains
    independent trajectories unless the shared frequency/slope supports a
    conservative stitch. Grouping is NOT evidence of a single bat passage.

    If candidates is None, the stable Adaptive V4 candidate detector is used.
    None of the annotated/manual tracks is required.
    """
    from .chirp_union import group_overlapping_chirps
    if candidates is None:
        from .detection import detect_candidates_snr_blobs
        candidates = detect_candidates_snr_blobs(
            y, sr, slope_filter_mode="adaptive_v4"
        )
    candidates = list(candidates)
    kwargs = dict(tracking_kwargs or {})
    groups_kwargs = dict(grouping_kwargs or {})
    successes = []
    indices = []
    failed = []
    diagnostics = []
    for index, candidate in enumerate(candidates):
        detail = {}
        curve = track_reassigned_candidate(
            y, sr, candidate, diagnostics=detail, **kwargs
        )
        diagnostics.append(detail)
        if curve is None:
            failed.append(index)
        else:
            successes.append(curve)
            indices.append(index)
    return {
        "groups": group_overlapping_chirps(
            successes, source_ids=indices, **groups_kwargs
        ),
        "untracked_candidate_indices": failed,
        "candidate_count": len(candidates),
        "tracking_diagnostics": diagnostics,
    }
