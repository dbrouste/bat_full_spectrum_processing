import json

import numpy as np
import pytest
import soundfile as sf

from benchmark.manual_annotations import validated_manual_chirps
from benchmark.snr_metrics import (
    ReferenceSNR, compare_trajectory, tolerance_khz, detection_importance,
)
from benchmark import reassignment_detector_seeds as detector_benchmark
from benchmark.compare_reports import compare_reports


def _chirp(chirp_id, *, source="manual", status="validated", t0=20.0):
    return {
        "chirp_id": chirp_id, "source": source, "status": status,
        "points": [{"t_ms": t0, "f_khz": 30.0},
                   {"t_ms": t0 + 2.0, "f_khz": 25.0}],
    }


def test_manual_annotations_exclude_every_automatic_proposal_and_draft():
    record = {"chirps": [
        _chirp(1), _chirp(2, source="automatic"),
        _chirp(3, source="automatic", status="draft"),
        _chirp(4, status="draft"),
        {"chirp_id": 5, "points": _chirp(5)["points"]},
    ]}
    assert [c["chirp_id"] for c in validated_manual_chirps(record)] == [1]
    assert [c[0] for c in detector_benchmark._manual_annotations(record)] == [1]


def test_requested_snr_tolerances_and_linear_interpolation():
    result = tolerance_khz([-10, 15, 20, 25, 30, 35, 80])
    np.testing.assert_allclose(result, [5, 5, 4.5, 4, 3.5, 3, 3])


def test_high_snr_missed_calls_are_more_costly_than_low_snr_misses():
    from benchmark.reassignment_detector_seeds import summarize_importance

    np.testing.assert_allclose(
        detection_importance([5, 15, 25, 35, 50]),
        [.2, .2, .6, 1., 1.])
    strong_found = [
        {"detection_importance_weight": 1., "detector_matched": True,
         "within_snr_tolerance_fraction": 1.},
        {"detection_importance_weight": .2, "detector_matched": False},
    ]
    weak_found = [
        {"detection_importance_weight": 1., "detector_matched": False},
        {"detection_importance_weight": .2, "detector_matched": True,
         "within_snr_tolerance_fraction": 1.},
    ]
    assert summarize_importance(strong_found)["weighted_detector_recall"] == pytest.approx(1 / 1.2)
    assert summarize_importance(weak_found)["weighted_detector_recall"] == pytest.approx(.2 / 1.2)


def test_adaptive_v4_recovers_steep_short_but_not_tiny_blobs(monkeypatch):
    from bat_analysis import detection

    blob = {"size": 40, "height_hz": 18000.0,
            "slope_hz_per_ms": -20000.0, "width_ms": 1.3333,
            "peak_db": -52., "time_mid": .010, "peak_freq_hz": 85000.,
            "f_high": 100000., "f_low": 82000.,
            "t_start": .0095, "t_end": .01083}
    tiny = {**blob, "width_ms": .67, "time_mid": .025}
    monkeypatch.setattr(detection, "_extract_candidates",
                        lambda *args, **kwargs: [blob, tiny])
    y = np.zeros(2048)
    current = detection.detect_candidates_snr_blobs(y, 384000)
    previous = detection.detect_candidates_snr_blobs(
        y, 384000, v4_general_min_width_ms=1.5)
    assert len(current) == 1
    assert current[0]["width_ms"] == pytest.approx(1.3333)
    assert previous == []


class _ConstantSNR:
    def __init__(self, value):
        self.value = value

    def along_reference(self, times_s, freqs_hz):
        return np.full(len(times_s), self.value)


def test_same_frequency_error_is_stricter_for_strong_chirp():
    reference = lambda t: np.full_like(t, 70_000)
    predicted = np.array([[0.0, 74_000], [0.01, 74_000]])
    strong = compare_trajectory(reference, 0, 0.01, predicted, _ConstantSNR(40))
    weak = compare_trajectory(reference, 0, 0.01, predicted, _ConstantSNR(10))
    assert strong["mean_full_interval_error_khz"] == pytest.approx(4)
    assert weak["mean_full_interval_error_khz"] == pytest.approx(4)
    assert strong["mean_normalized_error"] == pytest.approx(4 / 3)
    assert weak["mean_normalized_error"] == pytest.approx(4 / 5)
    assert strong["within_snr_tolerance_fraction"] == 0
    assert weak["within_snr_tolerance_fraction"] == 1


def test_missing_ends_are_penalized_and_coverage_reported():
    reference = lambda t: 80_000 - 2_000_000 * t
    # Exact middle segment, but missing both ends.
    predicted = np.array([[0.002, 76_000], [0.008, 64_000]])
    score = compare_trajectory(reference, 0, 0.01, predicted, _ConstantSNR(35))
    assert score["coverage_fraction"] == pytest.approx(0.6)
    assert score["mean_full_interval_error_khz"] > 0
    assert score["start_error_ms"] == pytest.approx(2)
    assert score["end_error_ms"] == pytest.approx(2)



def test_square_normal_benchmark_uses_manual_calibration_and_separates_vertical():
    pytest.importorskip("chirp_geometry")
    from chirp_geometry import reference_normals

    t = np.array([0.020, 0.024])
    f = np.array([120_000.0, 40_000.0])
    manual = np.column_stack((t, f))
    reference = lambda x: np.interp(x, t, f)
    sample = reference_normals(t * 1000, f / 1000, t * 1000)
    offset = 3.0
    scale = sample["time_scale_khz_per_ms"]
    auto_t = t + offset * sample["normal_u"] / (scale * 1000)
    auto_f = f + offset * sample["normal_f"] * 1000
    scores = compare_trajectory(
        reference, t[0], t[-1], np.column_stack((auto_t, auto_f)),
        _ConstantSNR(35), manual_points=manual,
        geometry_mode="square_normals_v1")
    assert scores["mean_geometric_error_khz_eq"] == pytest.approx(3, abs=1e-7)
    assert scores["geometric_coverage_fraction"] == 1
    assert scores["scoring_protocol"] == "square_normals_v1"
    assert scores["mean_full_interval_error_khz"] > 0
    assert scores["tolerance_units"] == "kHz_eq"


def test_square_normal_missing_ends_are_uncovered_not_extrapolated():
    pytest.importorskip("chirp_geometry")
    t = np.array([0.020, 0.024])
    f = np.array([120_000., 40_000.])
    reference = lambda x: np.interp(x, t, f)
    pred_t = np.array([0.0208, 0.0232])
    pred_f = reference(pred_t)
    score = compare_trajectory(
        reference, t[0], t[-1], np.column_stack((pred_t, pred_f)),
        _ConstantSNR(35), manual_points=np.column_stack((t, f)),
        geometry_mode="square_normals_v1")
    assert .5 < score["geometric_coverage_fraction"] < .7
    assert score["within_snr_tolerance_fraction"] < .7
    assert score["mean_geometric_error_khz_eq"] == pytest.approx(0, abs=1e-7)


def test_reference_snr_follows_manual_frequency_and_is_finite():
    sr = 96_000
    t = np.arange(int(0.05 * sr)) / sr
    audio = 0.0005 * np.random.default_rng(0).standard_normal(len(t))
    audio += np.where((t >= 0.02) & (t <= 0.03),
                      0.4 * np.sin(2 * np.pi * 25_000 * t), 0)
    snr = ReferenceSNR.from_audio(audio, sr)
    on = snr.along_reference(np.array([0.022, 0.025]),
                             np.array([25_000, 25_000]))
    off = snr.along_reference(np.array([0.012, 0.015]),
                              np.array([25_000, 25_000]))
    assert np.all(np.isfinite(on))
    assert np.mean(on) > np.mean(off) + 20


def test_end_to_end_benchmark_counts_only_manual_and_scores_unmatched(tmp_path, monkeypatch):
    sr = 96_000
    sf.write(tmp_path / "sample.wav", np.zeros(4800), sr)
    record = {"chirps": [
        _chirp(1), _chirp(2, t0=30),
        _chirp(3, source="automatic", status="validated"),
        _chirp(4, source="automatic", status="draft"),
        _chirp(5, status="draft"),
    ]}
    json_path = tmp_path / "manual.json"
    json_path.write_text(json.dumps({"files": {"sample.wav": record}}))
    candidate = {"time_mid": 0.021, "peak_freq_hz": 27_500.0}
    monkeypatch.setattr(detector_benchmark, "detect_candidates_snr_blobs",
                        lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(detector_benchmark, "track_reassigned_candidate",
                        lambda *args, **kwargs:
                        np.array([[0.02, 30_000], [0.022, 25_000]]))
    report = detector_benchmark.evaluate(
        json_path, tmp_path, low_snr_importance_weight=.35)
    assert report["totals"]["annotated"] == 2
    assert report["totals"]["ignored_automatic_proposals"] == 2
    assert report["totals"]["matched"] == 1
    assert report["totals"]["tracked"] == 1
    assert report["totals"]["detection_importance"]["at_or_below_15_db"] == .35
    assert report["details"][0]["detection_importance_weight"] == pytest.approx(.35)
    assert report["totals"]["full_pipeline_within_tolerance_fraction"] <= 0.5
    assert sum(x["annotated"] for x in report["totals"]["by_snr"].values()) == 2
    active_groups = [x for x in report["totals"]["by_snr"].values()
                     if x["tracking_succeeded"]]
    assert len(active_groups) == 1
    assert active_groups[0]["mean_tracking_error_khz"] is not None
    assert active_groups[0]["mean_coverage_fraction"] is not None
    assert [item["chirp_id"] for item in report["details"]] == [1, 2]
    assert report["details"][0]["detector_matched"]
    assert not report["details"][1]["detector_matched"]
    assert "snr_max_db" in report["details"][1]
    assert "mean_normalized_error" in report["details"][0]
    assert "mean_normalized_error" not in report["details"][1]


def test_compare_reports_aligns_manual_chirps_and_reports_regressions():
    before = {"details": [
        {"file": "a.wav", "chirp_id": 7, "detector_matched": True,
         "tracking_success": True, "snr_max_db": 38,
         "mean_normalized_error": 1.1},
        {"file": "a.wav", "chirp_id": 8, "detector_matched": False,
         "tracking_success": False, "snr_max_db": 17},
    ]}
    after = {"details": [
        {"file": "a.wav", "chirp_id": 7, "detector_matched": True,
         "tracking_success": True, "snr_max_db": 38,
         "mean_normalized_error": 0.8},
        {"file": "a.wav", "chirp_id": 8, "detector_matched": True,
         "tracking_success": True, "snr_max_db": 17,
         "mean_normalized_error": 0.5},
    ]}
    result = compare_reports(before, after)
    assert result["summary"]["improved"] == 1
    assert result["summary"]["regressed"] == 0
    assert result["summary"]["newly_matched"] == 1
    assert result["summary"]["newly_tracked"] == 1
    assert result["per_chirp"][0]["delta_mean_normalized_error"] == pytest.approx(-0.3)
    assert result["per_chirp"][1]["delta_mean_normalized_error"] is None
