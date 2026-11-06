"""Benchmark detector + Reassignment-v2 tracking wall time on real WAVs.

Use identical WAVs, hardware, Python versions and candidate counts in each
checkout of Reassignment and Reassignment-v2 for a fair comparison.

python -m benchmark.reassignment_speed /path/to/wavs --repeat 3 --max-candidates 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import soundfile as sf

from bat_analysis.detection import detect_candidates_snr_blobs
from bat_analysis.reassignment_tracking import track_reassigned_candidate


def _percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def benchmark(directory, *, repeat=3, max_candidates=20):
    if repeat < 1 or max_candidates < 1:
        raise ValueError("repeat and max_candidates must be >= 1")
    paths = sorted(Path(directory).glob("*.wav"))
    if not paths:
        raise FileNotFoundError("No WAV files found")
    result = {"files": [], "configuration": {
        "repeat": repeat, "max_candidates_per_file": max_candidates,
        "method": "adaptive_v4 -> reassignment_flattop",
    }}
    all_tracking_ms = []
    all_detection_ms = []
    total_candidates = total_tracked = 0
    digest = hashlib.sha256()

    for path in paths:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        detection_times = []
        candidates = None
        for iteration in range(repeat):
            start = time.perf_counter()
            current = detect_candidates_snr_blobs(
                y, int(sr), slope_filter_mode="adaptive_v4"
            )
            detection_times.append((time.perf_counter() - start) * 1000)
            if candidates is None:
                candidates = current
        all_detection_ms.extend(detection_times)
        chosen = candidates[:max_candidates]
        durations = []
        tracked = 0
        failures = []
        for candidate_index, candidate in enumerate(chosen):
            for iteration in range(repeat):
                start = time.perf_counter()
                try:
                    curve = track_reassigned_candidate(y, int(sr), candidate)
                except Exception as exc:
                    curve = None
                    if iteration == 0:
                        failures.append({
                            "candidate_index": candidate_index,
                            "exception": f"{type(exc).__name__}: {exc}",
                        })
                durations.append((time.perf_counter() - start) * 1000)
                if iteration == 0 and curve is not None:
                    tracked += 1
                    # A fingerprint helps spot regressions between checkouts.
                    digest.update(path.name.encode("utf-8"))
                    digest.update(np.asarray(
                        curve, dtype="<f8"
                    ).tobytes())
        all_tracking_ms.extend(durations)
        total_candidates += len(chosen)
        total_tracked += tracked
        result["files"].append({
            "wav": path.name, "sample_rate": int(sr),
            "duration_s": float(len(y) / sr),
            "candidates_total": len(candidates),
            "candidates_benchmarked": len(chosen),
            "tracks_returned": tracked,
            "detection_median_ms": float(statistics.median(detection_times)),
            "tracking_median_ms": (
                float(statistics.median(durations)) if durations else None
            ),
            "tracking_p90_ms": _percentile(durations, 90),
            "errors": failures,
        })
    result["summary"] = {
        "files": len(paths),
        "candidates_benchmarked": total_candidates,
        "tracks_returned": total_tracked,
        "detection_median_ms": _percentile(all_detection_ms, 50),
        "tracking_median_ms": _percentile(all_tracking_ms, 50),
        "tracking_p90_ms": _percentile(all_tracking_ms, 90),
        "tracking_mean_ms": (
            float(np.mean(all_tracking_ms)) if all_tracking_ms else None
        ),
        "tracking_result_sha256": digest.hexdigest(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_directory")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = benchmark(
        args.wav_directory, repeat=args.repeat,
        max_candidates=args.max_candidates,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
