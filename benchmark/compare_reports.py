"""Compare two independent benchmark JSON reports, including SNR-weighted error.

Run benchmark.reassignment_detector_seeds in each code version, with the same
WAV/annotation dataset, then compare the exported JSON results. This command
never reads automatic annotation proposals as ground truth.

python -m benchmark.compare_reports reassignment.json reassignment_v2.json \
    --output comparison.json --html comparison.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "mean_normalized_error",
    "mean_full_interval_error_khz",
    "coverage_fraction",
    "within_snr_tolerance_fraction",
)


def _index(report):
    return {(row["file"], str(row["chirp_id"])): row
            for row in report.get("details", []) if row.get("chirp_id") is not None}


def compare_reports(before, after, *, significant_delta=0.05):
    """Align by WAV and manual chirp ID; negative normalized delta improves."""
    first, second = _index(before), _index(after)
    rows = []
    for wav, chirp_id in sorted(first.keys() & second.keys()):
        a, b = first[(wav, chirp_id)], second[(wav, chirp_id)]
        row = {"file": wav, "chirp_id": a["chirp_id"],
               "before_matched": bool(a.get("detector_matched")),
               "after_matched": bool(b.get("detector_matched")),
               "before_tracked": bool(a.get("tracking_success")),
               "after_tracked": bool(b.get("tracking_success")),
               "snr_max_db": b.get("snr_max_db", a.get("snr_max_db"))}
        for metric in METRICS:
            x, y = a.get(metric), b.get(metric)
            row[f"before_{metric}"] = x
            row[f"after_{metric}"] = y
            row[f"delta_{metric}"] = (float(y - x) if x is not None
                                       and y is not None else None)
        rows.append(row)
    comparable = [r for r in rows if r["delta_mean_normalized_error"] is not None]
    changes = [r["delta_mean_normalized_error"] for r in comparable]
    summary = {
        "common_manual_chirps": len(rows),
        "only_before": len(first.keys() - second.keys()),
        "only_after": len(second.keys() - first.keys()),
        "comparable_tracked_chirps": len(comparable),
        "improved": sum(delta < -significant_delta for delta in changes),
        "regressed": sum(delta > significant_delta for delta in changes),
        "essentially_unchanged": sum(abs(delta) <= significant_delta for delta in changes),
        "mean_change_normalized_error": (float(np.mean(changes)) if changes
                                         else None),
        "newly_matched": sum(not r["before_matched"] and r["after_matched"]
                             for r in rows),
        "lost_matches": sum(r["before_matched"] and not r["after_matched"]
                            for r in rows),
        "newly_tracked": sum(not r["before_tracked"] and r["after_tracked"]
                             for r in rows),
        "lost_tracks": sum(r["before_tracked"] and not r["after_tracked"]
                           for r in rows),
    }
    return {
        "before": before.get("metadata", {}),
        "after": after.get("metadata", {}),
        "before_totals": before.get("totals", {}),
        "after_totals": after.get("totals", {}),
        "summary": summary,
        "per_chirp": rows,
    }


def write_html_comparison(report, output):
    """Interactive SNR-vs-error plot, one point per shared manual chirp."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for prefix, label in (("before", "Avant"), ("after", "Après")):
        rows = [r for r in report["per_chirp"]
                if r.get("snr_max_db") is not None and
                r.get(f"{prefix}_mean_normalized_error") is not None]
        fig.add_trace(go.Scattergl(
            x=[r["snr_max_db"] for r in rows],
            y=[r[f"{prefix}_mean_normalized_error"] for r in rows],
            text=[f'{r["file"]} · chirp {r["chirp_id"]}' for r in rows],
            mode="markers", name=label,
            hovertemplate="%{text}<br>SNR max %{x:.1f} dB"
                          "<br>Erreur normalisée %{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Comparaison des trajectoires · annotations manuelles uniquement",
        xaxis_title="SNR max estimé (dB)",
        yaxis_title="Erreur fréquentielle / tolérance SNR (plus bas = mieux)",
        hovermode="closest",
    )
    fig.write_html(str(output), include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", help="First JSON benchmark report")
    parser.add_argument("after", help="Second JSON benchmark report")
    parser.add_argument("--output", help="Write full comparison to JSON")
    parser.add_argument("--html", help="Write an interactive Plotly comparison")
    args = parser.parse_args()
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    report = compare_reports(before, after)
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    if args.html:
        write_html_comparison(report, args.html)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
