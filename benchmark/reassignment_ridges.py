"""Evaluate reassigned ridge tracking against manual annotations.

This is ORACLE SEEDED: the annotated midpoint, duration, and frequency are
used to initialize tracking, so this benchmark does NOT evaluate detection.
Reports full-interval error (including missed endpoints) and time coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator

from bat_analysis.reassignment_tracking import track_reassigned_ridge
from benchmark.manual_annotations import validated_manual_chirps


def evaluate(
    annotation_file, wav_dir, *, max_per_file=None,
    n_fft=512, hop=24, jump_penalty=0.11, gap_penalty=0.65,
    refine_endpoints=True, complete_ends=True, peak_anchored_window=False,
    jitter_time_ms=0.0, jitter_freq_khz=0.0, jitter_duration_fraction=0.0,
    random_seed=20261104,
):
    """Evaluate with reproducible optional seed perturbations.

    A trajectory counts as 'tracked' whenever the algorithm returns an
    array, even when its frequency error is large. For fair summary of
    failures use the optional 20 kHz penalty in summarize().
    """
    annotations = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    rng = np.random.default_rng(random_seed)
    reports = []
    for wav_name, record in annotations["files"].items():
        path = Path(wav_dir) / wav_name
        if not path.is_file():
            continue
        y, sr = sf.read(path, always_2d=False)
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        chirps = validated_manual_chirps(record)
        if max_per_file is not None:
            chirps = chirps[:max_per_file]
        for chirp in chirps:
            pts = sorted(chirp.get("points", []), key=lambda p: p["t_ms"])
            if len(pts) < 2:
                continue
            t = np.asarray([p["t_ms"] / 1000.0 for p in pts])
            f = np.asarray([p["f_khz"] * 1000.0 for p in pts])
            if np.any(np.diff(t) <= 0):
                continue
            midpoint = 0.5 * (t[0] + t[-1])
            duration = t[-1] - t[0]
            seed = float(np.interp(midpoint, t, f))
            midpoint += rng.uniform(-jitter_time_ms, jitter_time_ms) / 1000.0
            seed += rng.uniform(-jitter_freq_khz, jitter_freq_khz) * 1000.0
            duration *= 1.0 + rng.uniform(-jitter_duration_fraction, jitter_duration_fraction)
            curve = track_reassigned_ridge(
                y, sr, time_mid=midpoint, duration=duration,
                seed_freq_hz=seed, n_fft=n_fft, hop=hop,
                jump_penalty=jump_penalty, gap_penalty=gap_penalty,
                refine_endpoints=refine_endpoints,
                complete_ends=complete_ends,
                peak_anchored_window=peak_anchored_window,
            )
            record_result = {"wav": wav_name, "chirp_id": chirp.get("chirp_id"),
                             "tracked": curve is not None}
            if curve is None:
                reports.append(record_result)
                continue
            reference = (PchipInterpolator(t, f) if len(t) > 2
                         else lambda u: np.interp(u, t, f))
            # 50 uniformly spaced time points: endpoint truncation is
            # explicitly penalized via constant extrapolation by np.interp.
            grid = np.linspace(t[0], t[-1], 50)
            errors = np.abs(np.interp(grid, curve[:, 0], curve[:, 1])
                            - reference(grid)) / 1000.0
            t_in = (curve[:, 0] >= t[0]) & (curve[:, 0] <= t[-1])
            local_err = (np.abs(curve[t_in, 1] - reference(curve[t_in, 0]))
                         / 1000.0)
            coverage = max(0.0, min(curve[-1, 0], t[-1])
                           - max(curve[0, 0], t[0])) / (t[-1] - t[0])
            record_result.update({
                "median_in_span_error_khz": (
                    float(np.median(local_err)) if local_err.size else None),
                "mean_full_interval_error_khz": float(np.mean(errors)),
                "coverage_fraction": float(coverage),
                "trajectory_points": int(len(curve)),
            })
            reports.append(record_result)
    return reports


def summarize(reports, *, failure_penalty_khz=20.0):
    good = [r for r in reports if r["tracked"]]
    errors = [r["mean_full_interval_error_khz"] for r in good]
    penalized = [r.get("mean_full_interval_error_khz", failure_penalty_khz)
                 for r in reports]
    return {
        "n_annotated": len(reports),
        "n_tracked": len(good),
        "median_frequency_error_khz": (
            float(np.median(errors)) if errors else None),
        "mean_full_interval_error_khz": (
            float(np.mean(errors)) if errors else None),
        "mean_coverage_fraction": (
            float(np.mean([r["coverage_fraction"] for r in good]))
            if good else None),
        "mean_error_with_failures_khz": (
            float(np.mean(penalized)) if penalized else None),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("annotations")
    p.add_argument("wav_dir")
    p.add_argument("--max-per-file", type=int, default=None)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop", type=int, default=24)
    p.add_argument("--jump-penalty", type=float, default=0.11)
    p.add_argument("--no-endpoint-refinement", action="store_true")
    p.add_argument("--no-completion", action="store_true")
    p.add_argument("--peak-anchored-window", action="store_true")
    p.add_argument("--gap-penalty", type=float, default=0.65)
    p.add_argument("--jitter-time-ms", type=float, default=0.0)
    p.add_argument("--jitter-freq-khz", type=float, default=0.0)
    p.add_argument("--jitter-duration-fraction", type=float, default=0.0)
    p.add_argument("--random-seed", type=int, default=20261104)
    args = p.parse_args()
    reports = evaluate(
        args.annotations, args.wav_dir, max_per_file=args.max_per_file,
        n_fft=args.n_fft, hop=args.hop, jump_penalty=args.jump_penalty,
        refine_endpoints=not args.no_endpoint_refinement,
        complete_ends=not args.no_completion,
        peak_anchored_window=args.peak_anchored_window,
        gap_penalty=args.gap_penalty, jitter_time_ms=args.jitter_time_ms,
        jitter_freq_khz=args.jitter_freq_khz,
        jitter_duration_fraction=args.jitter_duration_fraction,
        random_seed=args.random_seed,
    )
    print(json.dumps({"summary": summarize(reports), "results": reports}, indent=2))


if __name__ == "__main__":
    main()
