"""Measure endpoint and second-harmonic tracking against manual annotations.

Annotation times are used ONLY for parent-seed setup and evaluation; harmonic
annotations are never used as input to the automatic harmonic proposal.
The output also lists unreviewed harmonic proposals (not false positives).

python -m benchmark.harmonic_annotations Nycto-Thomas.json /path/to/wavs
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator

from bat_analysis.reassignment_tracking import track_reassigned_ridge
from bat_analysis.harmonics import detect_harmonic_segments
from benchmark.manual_annotations import validated_manual_chirps


def _valid_chirps(record):
    out = []
    for c in validated_manual_chirps(record):
        pts = sorted(c.get("points", []), key=lambda p: p["t_ms"])
        if len(pts) < 2:
            continue
        t = np.asarray([float(p["t_ms"]) / 1000 for p in pts])
        f = np.asarray([float(p["f_khz"]) * 1000 for p in pts])
        if not np.isfinite(t).all() or not np.isfinite(f).all():
            continue
        if np.any(np.diff(t) <= 0):
            continue
        out.append((c.get("chirp_id"), t, f))
    return out


def _pairs(chirps):
    """Discover annotated second-harmonic pairs for EVALUATION ONLY."""
    pairs = {}
    for hid, ht, hf in chirps:
        if ht[-1] - ht[0] >= .0012:
            continue
        mid = .5 * (ht[0] + ht[-1])
        reference_freq = np.interp(mid, ht, hf)
        options = []
        for pid, pt, pf in chirps:
            if (pid == hid or pt[-1] - pt[0] < .0012
                    or not pt[0] <= mid <= pt[-1]):
                continue
            base_ref = (PchipInterpolator(pt, pf)(mid)
                        if len(pt) > 2 else np.interp(mid, pt, pf))
            ratio = float(reference_freq / base_ref)
            if 1.7 <= ratio <= 2.3:
                options.append((abs(ratio - 2), pid))
        if options:
            pairs[min(options)[1]] = (hid, ht, hf)
    return pairs


def evaluate(annotation_json, wav_dir):
    labels = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    report = []
    for name, record in labels["files"].items():
        wav = Path(wav_dir) / name
        if not wav.exists():
            continue
        y, sr = sf.read(wav)
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        chirps = _valid_chirps(record)
        pairs = _pairs(chirps)
        all_detected = []
        for pid, pt, pf in chirps:
            if pt[-1] - pt[0] < .0012:
                continue
            mid = .5 * (pt[0] + pt[-1])
            seed = np.interp(mid, pt, pf)
            common = dict(time_mid=mid, duration=pt[-1] - pt[0],
                          seed_freq_hz=float(seed), n_fft=512, hop=24)
            before = track_reassigned_ridge(
                y, sr, refine_endpoints=False, **common)
            after = track_reassigned_ridge(
                y, sr, refine_endpoints=True, **common)
            row = {"file": name, "fundamental_id": pid,
                   "tracked": after is not None,
                   "annotated_harmonic_id": pairs[pid][0] if pid in pairs else None}
            if before is not None and after is not None:
                target = (PchipInterpolator(pt, pf) if len(pt) > 2
                          else lambda x: np.interp(x, pt, pf))
                grid = np.linspace(pt[0], pt[-1], 101)
                row["mae_before_khz"] = float(np.mean(np.abs(
                    np.interp(grid, before[:, 0], before[:, 1]) - target(grid)
                )) / 1000)
                row["mae_after_khz"] = float(np.mean(np.abs(
                    np.interp(grid, after[:, 0], after[:, 1]) - target(grid)
                )) / 1000)
                row["tail_mae_before_khz"] = float(np.mean(np.abs(
                    np.interp(grid[-34:], before[:, 0], before[:, 1])
                    - target(grid[-34:])
                )) / 1000)
                row["tail_mae_after_khz"] = float(np.mean(np.abs(
                    np.interp(grid[-34:], after[:, 0], after[:, 1])
                    - target(grid[-34:])
                )) / 1000)
            if after is not None:
                proposals = detect_harmonic_segments(
                    y, sr, after, parent_id=pid)
                row["harmonic_proposals"] = len(proposals)
                if proposals:
                    all_detected.append(pid)
                if pid in pairs:
                    hid, ht, hf = pairs[pid]
                    observed = [p for p in proposals
                                if p.t_start_s <= ht[-1] and
                                   p.t_end_s >= ht[0]]
                    row["harmonic_matched"] = bool(observed)
                    if observed:
                        p = observed[0]
                        h_ref = (PchipInterpolator(ht, hf)
                                 if len(ht) > 2 else
                                 lambda x: np.interp(x, ht, hf))
                        covered = ((p.points[:, 0] >= ht[0])
                                   & (p.points[:, 0] <= ht[-1]))
                        if np.any(covered):
                            row["harmonic_frequency_mae_khz"] = float(
                                np.mean(np.abs(
                                    p.points[covered, 1]
                                    - h_ref(p.points[covered, 0])
                                )) / 1000)
                            row["harmonic_coverage"] = float(
                                max(0, min(p.t_end_s, ht[-1])
                                    - max(p.t_start_s, ht[0]))
                                / (ht[-1] - ht[0]))
            report.append(row)
    eligible = [r for r in report if r.get("annotated_harmonic_id") is not None]
    matched = [r for r in eligible if r.get("harmonic_matched")]
    before = [r["mae_before_khz"] for r in report if "mae_before_khz" in r]
    after = [r["mae_after_khz"] for r in report if "mae_after_khz" in r]
    return {
        "summary": {
            "annotated_harmonics": len(eligible),
            "harmonics_found": len(matched),
            "unreviewed_proposals": sum(
                r.get("harmonic_proposals", 0) for r in report
                if r.get("annotated_harmonic_id") is None),
            "fundamentals_evaluated": len(report),
            "mean_parent_error_before_khz": (
                float(np.mean(before)) if before else None),
            "mean_parent_error_after_khz": (
                float(np.mean(after)) if after else None),
        },
        "detail": report,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("annotations")
    p.add_argument("wav_dir")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    result = evaluate(args.annotations, args.wav_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
