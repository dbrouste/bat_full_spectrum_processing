"""End-to-end Adaptive V4 candidate -> reassigned Flattop ridge benchmark.

Unlike benchmark.reassignment_ridges, this DOES NOT use manual annotations to
seed the tracker. Annotations are used only to match and score predictions.
It measures candidate recall separately from trajectory quality.

python -m benchmark.reassignment_detector_seeds Nycto-Thomas.json /path/to/wavs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment

from bat_analysis.detection import detect_candidates_snr_blobs
from bat_analysis.reassignment_tracking import track_reassigned_candidate
from bat_analysis.harmonics import detect_harmonic_segments


def _manual_annotations(record):
    valid = []
    for chirp in record.get("chirps", []):
        if chirp.get("status") not in (None, "validated"):
            continue
        points = sorted(chirp.get("points", []), key=lambda p: p["t_ms"])
        if len(points) < 2:
            continue
        t = np.asarray([float(p["t_ms"]) / 1000.0 for p in points])
        f = np.asarray([float(p["f_khz"]) * 1000.0 for p in points])
        if not np.isfinite(t).all() or not np.isfinite(f).all():
            continue
        if np.any(np.diff(t) <= 0):
            continue
        reference = PchipInterpolator(t, f, extrapolate=False)
        valid.append((chirp.get("chirp_id"), t, f, reference))
    return valid


def match_candidates(candidates, annotations, *, time_slack_ms=1.5,
                     max_seed_error_khz=18.0):
    """One-to-one matching for evaluating detector candidates.

    A match requires peak time near the annotated span and peak frequency
    within a tolerance of the manually interpolated ridge. Matching uses
    annotations ONLY for scoring and never feeds them into tracking.
    """
    if not candidates or not annotations:
        return []
    cost = np.full((len(candidates), len(annotations)), 1e6)
    for i, c in enumerate(candidates):
        peak_t = float(c["time_mid"])
        peak_f = float(c["peak_freq_hz"])
        for j, (_, t, _, curve) in enumerate(annotations):
            duration = t[-1] - t[0]
            if not t[0] - time_slack_ms / 1000 <= peak_t <= t[-1] + time_slack_ms / 1000:
                continue
            time_error_ms = abs(peak_t - 0.5 * (t[0] + t[-1])) * 1000
            if time_error_ms > max(3.0, 1.5 * duration * 1000):
                continue
            clipped_t = np.clip(peak_t, t[0], t[-1])
            freq_error_khz = abs(peak_f - float(curve(clipped_t))) / 1000
            if freq_error_khz > max_seed_error_khz:
                continue
            cost[i, j] = time_error_ms / 3.0 + freq_error_khz / max_seed_error_khz
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e6]


def evaluate(annotation_file, wav_dir, *, n_fft=None, hop=None,
             max_per_file=None, use_broad_seed=True, evaluate_harmonics=False,
             complete_ends=True):
    annotations = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    per_file = []
    details = []
    for wav_name, record in annotations["files"].items():
        path = Path(wav_dir) / wav_name
        if not path.is_file():
            per_file.append({"file": wav_name, "error": "WAV not found"})
            continue
        y, sr = sf.read(path, always_2d=False)
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        manual = _manual_annotations(record)
        if max_per_file is not None:
            manual = manual[:max_per_file]
        # Run the exact library detector, not a locally reimplemented proxy.
        candidates = detect_candidates_snr_blobs(
            np.asarray(y), int(sr), slope_filter_mode="adaptive_v4"
        )
        matches = match_candidates(candidates, manual)
        summary = dict(file=wav_name, annotated=len(manual),
                       candidates=len(candidates), matched=len(matches),
                       unmatched_annotations=len(manual) - len(matches),
                       unmatched_candidates=len(candidates) - len(matches))
        errors = []
        covers = []
        for candidate_index, manual_index in matches:
            candidate = candidates[candidate_index]
            chirp_id, t, _, reference = manual[manual_index]
            diag = {}
            try:
                curve = track_reassigned_candidate(
                    y, int(sr), candidate,
                    n_fft=n_fft, hop=hop, use_broad_seed=use_broad_seed,
                    complete_ends=complete_ends,
                    diagnostics=diag,
                )
            except Exception as e:
                curve = None
                diag["error"] = f"{type(e).__name__}: {e}"
            item = dict(file=wav_name, chirp_id=chirp_id,
                        candidate_index=candidate_index,
                        peak_time_ms=1000 * float(candidate["time_mid"]),
                        peak_freq_khz=float(candidate["peak_freq_hz"]) / 1000,
                        tracking_success=curve is not None)
            if curve is not None:
                # Uniform temporal grid penalizes missing endpoints instead of
                # measuring error only where tracking chose to return points.
                grid = np.linspace(t[0], t[-1], 50)
                diff = np.abs(np.interp(grid, curve[:, 0], curve[:, 1])
                              - reference(grid)) / 1000
                coverage = max(
                    0., min(curve[-1, 0], t[-1])
                    - max(curve[0, 0], t[0])
                ) / (t[-1] - t[0])
                item.update(mean_full_interval_error_khz=float(diff.mean()),
                            p90_error_khz=float(np.percentile(diff, 90)),
                            coverage_fraction=float(coverage),
                            points=int(len(curve)))
                errors.append(item["mean_full_interval_error_khz"])
                covers.append(float(coverage))
                if evaluate_harmonics:
                    harmonic_segments = detect_harmonic_segments(
                        y, int(sr), curve, parent_id=candidate_index)
                    item['harmonic_segments'] = [
                        {'harmonic_order': h.harmonic_order,
                         'parent_id': h.parent_id,
                         't_start_ms': h.t_start_s * 1000,
                         't_end_ms': h.t_end_s * 1000,
                         'mean_snr_db': h.mean_snr_db,
                         'median_frequency_ratio': h.median_frequency_ratio}
                        for h in harmonic_segments
                    ]
            item["diagnostics"] = diag
            details.append(item)
        summary.update(
            tracking_succeeded=len(errors),
            detector_match_recall=(len(matches) / len(manual) if manual else None),
            detector_match_precision=(len(matches) / len(candidates)
                                      if candidates else None),
            mean_tracking_error_khz=(float(np.mean(errors)) if errors else None),
            mean_tracking_coverage=(float(np.mean(covers)) if covers else None),
        )
        per_file.append(summary)
    totals = dict(
        files=len(per_file),
        annotated=sum(x.get("annotated", 0) for x in per_file),
        candidates=sum(x.get("candidates", 0) for x in per_file),
        matched=sum(x.get("matched", 0) for x in per_file),
        tracked=sum(x.get("tracking_succeeded", 0) for x in per_file),
    )
    all_err = [x["mean_full_interval_error_khz"] for x in details
               if x.get("mean_full_interval_error_khz") is not None]
    totals["mean_tracking_error_khz"] = (
        float(np.mean(all_err)) if all_err else None
    )
    return {"totals": totals, "per_file": per_file, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations")
    parser.add_argument("wav_dir")
    parser.add_argument("--max-per-file", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--no-broad-seed", action="store_true")
    parser.add_argument("--no-completion", action="store_true")
    parser.add_argument("--harmonics", action="store_true",
                        help="Report secondary harmonic segments linked to V4 candidates")
    parser.add_argument("--output", default=None,
                        help="Optional output JSON report")
    args = parser.parse_args()
    result = evaluate(
        args.annotations, args.wav_dir, n_fft=args.n_fft, hop=args.hop,
        max_per_file=args.max_per_file, use_broad_seed=not args.no_broad_seed,
        evaluate_harmonics=args.harmonics,
        complete_ends=not args.no_completion
    )
    report = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
