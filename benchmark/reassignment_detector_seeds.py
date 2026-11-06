"""End-to-end Adaptive V4 candidate -> reassigned Flattop ridge benchmark.

Unlike benchmark.reassignment_ridges, this DOES NOT use manual annotations to
seed the tracker. Annotations are used only to match and score predictions.
It measures candidate recall separately from trajectory quality.

python -m benchmark.reassignment_detector_seeds Nycto-Thomas.json /path/to/wavs
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment

from bat_analysis.detection import detect_candidates_snr_blobs
from bat_analysis.reassignment_tracking import track_reassigned_candidate
from bat_analysis.harmonics import detect_harmonic_segments
from benchmark.manual_annotations import validated_manual_chirps
from benchmark.snr_metrics import (
    ReferenceSNR, compare_trajectory, detection_importance,
)


def summarize_importance(details):
    """High-SNR misses contribute more to recall than low-SNR misses."""
    if not details:
        return {"weighted_detector_recall": None,
                "weighted_full_pipeline_within_tolerance_fraction": None,
                "weighted_missed_chirp_mass": 0.0,
                "total_chirp_importance": 0.0}
    weights = np.asarray([x["detection_importance_weight"] for x in details])
    matched = np.asarray([bool(x.get("detector_matched")) for x in details])
    within = np.asarray([x.get("within_snr_tolerance_fraction", 0.0)
                         for x in details])
    total = float(np.sum(weights))
    return {
        "weighted_detector_recall": float(np.dot(weights, matched) / total),
        "weighted_full_pipeline_within_tolerance_fraction": (
            float(np.dot(weights, within) / total)),
        "weighted_missed_chirp_mass": float(np.sum(weights[~matched])),
        "total_chirp_importance": total,
    }


def summarize_by_snr(details):
    """Report detection and full-pipeline tolerance by reference peak SNR."""
    bounds = (
        ("low_15_db_or_less", lambda snr: snr <= 15.0),
        ("medium_15_to_35_db", lambda snr: 15.0 < snr < 35.0),
        ("high_35_db_or_more", lambda snr: snr >= 35.0),
    )
    result = {}
    for label, keep in bounds:
        group = [x for x in details if x.get("snr_max_db") is not None
                 and keep(x["snr_max_db"])]
        n = len(group)
        tracked = [x for x in group if x.get("tracking_success")
                   and x.get("mean_full_interval_error_khz") is not None]

        def mean_tracked(key):
            return (float(np.mean([x[key] for x in tracked])) if tracked else None)

        result[label] = {
            "annotated": n,
            "detector_matched": sum(x.get("detector_matched", False) for x in group),
            "tracking_succeeded": sum(x.get("tracking_success", False) for x in group),
            "mean_tracking_error_khz": mean_tracked("mean_full_interval_error_khz"),
            "mean_normalized_error": mean_tracked("mean_normalized_error"),
            "mean_coverage_fraction": mean_tracked("coverage_fraction"),
            "mean_start_error_ms": mean_tracked("start_error_ms"),
            "mean_end_error_ms": mean_tracked("end_error_ms"),
            "detector_recall": (sum(x.get("detector_matched", False) for x in group) / n
                                if n else None),
            # Missing tracks count as 0. This is deliberately different from
            # the within-tolerance fraction on successfully tracked curves.
            "full_pipeline_within_tolerance_fraction": (
                float(np.mean([x.get("within_snr_tolerance_fraction", 0.0)
                               for x in group])) if n else None),
        }
    return result


def _manual_annotations(record):
    valid = []
    for chirp in validated_manual_chirps(record):
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
             complete_ends=True, v4_general_min_width_ms=None,
             low_snr_importance_weight=0.2, search_margin_ms=None):
    if not 0 < low_snr_importance_weight <= 1:
        raise ValueError("low_snr_importance_weight must be in (0, 1]")
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
        # Fixed STFT reference: the SNR never depends on a predicted trajectory.
        snr_reference = (ReferenceSNR.from_audio(y, int(sr)) if manual else None)
        file_details = []
        for chirp_id, t, _, reference in manual:
            item = dict(file=wav_name, chirp_id=chirp_id,
                        detector_matched=False, tracking_success=False)
            item.update(compare_trajectory(
                reference, t[0], t[-1], None, snr_reference))
            item["detection_importance_weight"] = float(
                detection_importance(
                    item["snr_max_db"], min_weight=low_snr_importance_weight))
            file_details.append(item)
        details.extend(file_details)
        # Run the exact library detector, not a locally reimplemented proxy.
        detector_kwargs = ({"v4_general_min_width_ms": v4_general_min_width_ms}
                           if v4_general_min_width_ms is not None else {})
        candidates = detect_candidates_snr_blobs(
            np.asarray(y), int(sr), slope_filter_mode="adaptive_v4",
            **detector_kwargs,
        )
        matches = match_candidates(candidates, manual)
        summary = dict(file=wav_name, annotated=len(manual),
                       candidates=len(candidates), matched=len(matches),
                       unmatched_annotations=len(manual) - len(matches),
                       # With only a subset of manual chirps, other candidates
                       # cannot fairly be counted as unmatched proposals.
                       unmatched_candidates=(len(candidates) - len(matches)
                                             if max_per_file is None else None),
                       ignored_automatic_proposals=sum(
                           chirp.get("source") == "automatic"
                           for chirp in record.get("chirps", [])))
        errors = []
        covers = []
        normalized_errors = []
        tolerance_fractions = []
        for candidate_index, manual_index in matches:
            candidate = candidates[candidate_index]
            chirp_id, t, _, reference = manual[manual_index]
            item = file_details[manual_index]
            item.update(detector_matched=True, candidate_index=candidate_index,
                        peak_time_ms=1000 * float(candidate["time_mid"]),
                        peak_freq_khz=float(candidate["peak_freq_hz"]) / 1000)
            diag = {}
            try:
                tracking_kwargs = ({"search_margin_ms": search_margin_ms}
                                   if search_margin_ms is not None else {})
                curve = track_reassigned_candidate(
                    y, int(sr), candidate,
                    n_fft=n_fft, hop=hop, use_broad_seed=use_broad_seed,
                    complete_ends=complete_ends,
                    diagnostics=diag, **tracking_kwargs,
                )
            except Exception as e:
                curve = None
                diag["error"] = f"{type(e).__name__}: {e}"
            item["tracking_success"] = curve is not None
            if curve is not None:
                item.update(compare_trajectory(
                    reference, t[0], t[-1], curve, snr_reference))
                item["points"] = int(len(curve))
                errors.append(item["mean_full_interval_error_khz"])
                covers.append(item["coverage_fraction"])
                normalized_errors.append(item["mean_normalized_error"])
                tolerance_fractions.append(item["within_snr_tolerance_fraction"])
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
        summary.update(
            tracking_succeeded=len(errors),
            detector_match_recall=(len(matches) / len(manual) if manual else None),
            detector_match_precision=(len(matches) / len(candidates)
                                      if candidates and max_per_file is None else None),
            mean_tracking_error_khz=(float(np.mean(errors)) if errors else None),
            mean_tracking_coverage=(float(np.mean(covers)) if covers else None),
            mean_normalized_error=(float(np.mean(normalized_errors))
                                   if normalized_errors else None),
            mean_within_snr_tolerance_fraction=(float(np.mean(tolerance_fractions))
                                                if tolerance_fractions else None),
            full_pipeline_within_tolerance_fraction=(
                float(np.mean([x.get("within_snr_tolerance_fraction", 0.0)
                               for x in file_details])) if file_details else None),
            by_snr=summarize_by_snr(file_details),
            **summarize_importance(file_details),
        )
        per_file.append(summary)
    totals = dict(
        files=len(per_file),
        files_evaluated=sum("error" not in x for x in per_file),
        files_missing=sum("error" in x for x in per_file),
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
    all_normalized = [x["mean_normalized_error"] for x in details
                      if x.get("mean_normalized_error") is not None]
    all_tolerance = [x["within_snr_tolerance_fraction"] for x in details
                     if x.get("within_snr_tolerance_fraction") is not None]
    totals["mean_normalized_error"] = (
        float(np.mean(all_normalized)) if all_normalized else None)
    totals["mean_within_snr_tolerance_fraction"] = (
        float(np.mean(all_tolerance)) if all_tolerance else None)
    totals["full_pipeline_within_tolerance_fraction"] = (
        float(np.mean([x.get("within_snr_tolerance_fraction", 0.0)
                       for x in details])) if details else None)
    totals["by_snr"] = summarize_by_snr(details)
    totals.update(summarize_importance(details))
    totals["ignored_automatic_proposals"] = sum(
        x.get("ignored_automatic_proposals", 0) for x in per_file)
    totals["snr_tolerance_khz"] = {"at_or_below_15_db": 5.0,
                                    "at_or_above_35_db": 3.0}
    totals["detection_importance"] = {
        "at_or_below_15_db": float(low_snr_importance_weight),
        "at_or_above_35_db": 1.0,
    }
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL, text=True).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root,
            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        git_dirty = None
    return {
        "metadata": {
            "method": "adaptive_v4 -> reassignment_flattop",
            "git_commit": commit,
            "git_dirty": git_dirty,
            "snr_reference": "STFT flattop 512, hop 64; median dB noise per frequency",
            "ground_truth": "manual + validated only",
            "annotations": str(annotation_file),
            "n_fft": n_fft,
            "hop": hop,
            "max_per_file": max_per_file,
            "v4_general_min_width_ms": (
                1.33 if v4_general_min_width_ms is None
                else v4_general_min_width_ms),
            "low_snr_importance_weight": float(low_snr_importance_weight),
            "search_margin_ms": search_margin_ms,
        },
        "totals": totals, "per_file": per_file, "details": details,
    }


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
    parser.add_argument("--general-min-width-ms", type=float, default=None,
                        help="Override Adaptive V4 minimum general blob width; "
                             "1.5 reproduces the previous Reassignment-v2 setting")
    parser.add_argument("--low-snr-weight", type=float, default=0.2,
                        help="Importance for a manually annotated chirp at <=15 dB; "
                             "defaults to 0.2, rising linearly to 1 at >=35 dB")
    parser.add_argument("--search-margin-ms", type=float, default=None,
                        help="Override tracking search margin in ms; 1.5 "
                             "reproduces the previous fixed bounded margin")
    parser.add_argument("--output", default=None,
                        help="Optional output JSON report")
    args = parser.parse_args()
    result = evaluate(
        args.annotations, args.wav_dir, n_fft=args.n_fft, hop=args.hop,
        max_per_file=args.max_per_file, use_broad_seed=not args.no_broad_seed,
        evaluate_harmonics=args.harmonics,
        complete_ends=not args.no_completion,
        v4_general_min_width_ms=args.general_min_width_ms,
        low_snr_importance_weight=args.low_snr_weight,
        search_margin_ms=args.search_margin_ms,
    )
    report = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
