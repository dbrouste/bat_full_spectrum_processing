"""Evaluate reassigned ridge against manually annotated PCHIP ground truth.

Usage:
 python -m benchmark.reassignment_ridges /path/to/Nycto-Thomas.json /path/to/wavs

This is an oracle-seeded *tracking-only* diagnostic, not detection recall.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from bat_analysis.reassignment_tracking import track_reassigned_ridge


def evaluate(annotation_file, wav_dir, *, max_per_file=None):
    annotations = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    reports = []
    for wav_name, record in annotations["files"].items():
        path = Path(wav_dir) / wav_name
        if not path.is_file():
            continue
        y, sr = sf.read(path)
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        chirps = record.get("chirps", [])
        if max_per_file is not None:
            chirps = chirps[:max_per_file]
        for chirp in chirps:
            pts = sorted(chirp.get("points", []), key=lambda p: p["t_ms"])
            if len(pts) < 2:
                continue
            t = np.array([p["t_ms"] / 1000 for p in pts])
            f = np.array([p["f_khz"] * 1000 for p in pts])
            if np.any(np.diff(t) <= 0):
                continue
            mid = (t[0] + t[-1]) / 2
            reference = PchipInterpolator(t, f) if len(t) > 2 else None
            seed = np.interp(mid, t, f)
            curve = track_reassigned_ridge(
                y, sr, time_mid=mid, duration=t[-1] - t[0],
                seed_freq_hz=float(seed),
            )
            if curve is None:
                reports.append({"wav": wav_name, "chirp_id": chirp.get("chirp_id"),
                                "detected": False})
                continue
            inside = (curve[:, 0] >= t[0]) & (curve[:, 0] <= t[-1])
            target = (reference(curve[inside, 0]) if reference is not None
                      else np.interp(curve[inside, 0], t, f))
            error = np.abs(curve[inside, 1] - target) / 1000
            coverage = max(0., min(curve[-1, 0], t[-1])
                           - max(curve[0, 0], t[0])) / (t[-1] - t[0])
            reports.append({
                "wav": wav_name, "chirp_id": chirp.get("chirp_id"),
                "detected": True, "median_error_khz": (
                    float(np.median(error)) if error.size else None),
                "coverage_fraction": float(coverage),
            })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations")
    parser.add_argument("wav_dir")
    parser.add_argument("--max-per-file", type=int, default=None)
    args = parser.parse_args()
    results = evaluate(args.annotations, args.wav_dir,
                       max_per_file=args.max_per_file)
    errors = [r["median_error_khz"] for r in results
              if r.get("median_error_khz") is not None]
    print(json.dumps({
        "tested": len(results),
        "tracked": sum(r["detected"] for r in results),
        "median_error_khz": float(np.median(errors)) if errors else None,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
