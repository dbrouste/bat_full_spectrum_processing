"""Non-regression and experimental trajectory tracking tests."""
import numpy as np
import pytest
from bat_analysis.reassignment_tracking import track_reassigned_ridge
from bat_analysis.modelling import process_full_spectrum


def test_modelling_defaults_to_reassignment_flattop(monkeypatch):
    called = []

    def fake_tracker(*args, **kwargs):
        called.append(kwargs["time_mid"])
        return np.array([[0.010, 45000.0, 0.5], [0.011, 43000.0, 0.4]])

    monkeypatch.setattr(
        "bat_analysis.reassignment_tracking.track_reassigned_ridge",
        fake_tracker,
    )
    result = process_full_spectrum(
        np.zeros(8000), 384000, time_mid=0.010, duration=0.004,
        seed_freq_hz=45000.0,
    )
    assert called == [0.010]
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0], [0.0075, 0.0085])


def test_reassignment_tracks_downward_sweep():
    sr = 384000
    t = np.arange(int(0.020 * sr)) / sr
    # 100 -> 70 kHz from 4 to 16 ms, surrounded by silence.
    k = (t >= .004) & (t <= .016)
    phase = 2 * np.pi * (100000 * t - .5 * (30000/.012) * (t-.004)**2)
    x = np.zeros(t.size)
    x[k] = .8 * np.sin(phase[k])
    r = track_reassigned_ridge(x, sr, time_mid=.010, duration=.012,
                               seed_freq_hz=85000, n_fft=512, hop=64)
    assert r is not None and len(r) >= 3
    assert r.shape[1] == 3
    assert np.all(np.diff(r[:,0]) > 0)
    assert 20000 <= r[:,1].min() and r[:,1].max() <= 150000
    assert np.all((r[:,2] >= 0) & (r[:,2] <= 1 + 1e-10))


def test_modelling_explicit_reassignment_mode():
    sr = 384000
    t = np.arange(int(.02 * sr)) / sr
    x = np.sin(2*np.pi*(85000*t - 2e6*t*t))
    diag = {}
    result = process_full_spectrum(
        x, sr, time_mid=.01, duration=.008,
        seed_freq_hz=45000, tracking_method="reassignment_flattop",
        reassignment_n_fft=512, reassignment_hop=64, diagnostics=diag,
    )
    assert result is None or result.shape[1] == 3
    assert "model_success" in diag


def test_unknown_tracking_method_rejected():
    with pytest.raises(ValueError, match="tracking_method"):
        process_full_spectrum(np.zeros(1024), 384000,
                              time_mid=.001, duration=.001,
                              tracking_method="unknown")


def test_subbin_refinement_keeps_path_but_refines_frequency():
    sr = 384000
    tt = np.arange(int(0.015 * sr)) / sr
    x = np.zeros_like(tt)
    on = (tt >= .005) & (tt <= .008)
    tau = tt[on] - .005
    x[on] = (np.sin(2 * np.pi * (110000 * tau
                                - .5 * (50000 / .003) * tau ** 2))
             * np.sin(np.pi * tau / .003) ** 2)
    args = dict(time_mid=.0065, duration=.003, seed_freq_hz=85000,
                n_fft=512, hop=24)
    raw = track_reassigned_ridge(x, sr, subbin_refine=False, **args)
    precise = track_reassigned_ridge(x, sr, subbin_refine=True, **args)
    assert raw is not None and precise is not None
    np.testing.assert_allclose(raw[:, 0], precise[:, 0])
    np.testing.assert_allclose(raw[:, 2], precise[:, 2])
    assert np.isfinite(precise).all()
    assert np.max(np.abs(raw[:, 1] - precise[:, 1])) <= 3 * sr / 512 + 1e-9


def test_short_harmonic_sweep_can_be_tracked():
    sr = 384000
    tt = np.arange(int(.015 * sr)) / sr
    x = np.zeros_like(tt)
    on = (tt >= .006) & (tt < .00655)
    tau = tt[on] - .006
    freq0, delta = 137000, 30000
    x[on] = np.sin(2 * np.pi * (freq0 * tau
                                  - .5 * (delta / .00055) * tau ** 2))
    result = track_reassigned_ridge(
        x, sr, time_mid=.006275, duration=.00055, seed_freq_hz=122000,
        n_fft=192, hop=16,
    )
    assert result is not None
    assert len(result) >= 3
    assert np.all(np.diff(result[:, 0]) > 0)


def test_resolution_auto_and_explicit_override():
    from bat_analysis.reassignment_tracking import choose_reassignment_resolution
    assert choose_reassignment_resolution(0.00055) == (192, 16)
    assert choose_reassignment_resolution(0.003) == (512, 24)
    assert choose_reassignment_resolution(0.00055, n_fft=512, hop=24) == (512, 24)
    assert choose_reassignment_resolution(0.00055, n_fft=256) == (256, 16)
    with pytest.raises(ValueError, match="positive"):
        choose_reassignment_resolution(0.0)


def test_detector_candidate_pairs_time_frequency_and_bbox(monkeypatch):
    from bat_analysis import reassignment_tracking as tracking
    captured = {}
    def fake_track(y, sr, **kwargs):
        captured.update(kwargs)
        return np.ones((4, 3))
    monkeypatch.setattr(tracking, "track_reassigned_ridge", fake_track)
    signal = np.zeros(1024)
    candidate = dict(time_mid=0.011, duration=0.003,
                     peak_freq_hz=90000., t_start=0.010, t_end=0.013,
                     broad_fm_seed_freq_hz=108000.,
                     coarse_seed_time_s=0.0104)
    tracking.track_reassigned_candidate(signal, 384000, candidate)
    assert captured["time_bounds_s"] == (0.010, 0.013)
    assert captured["time_mid"] == 0.0104
    assert captured["seed_freq_hz"] == 108000.
    candidate["broad_fm_seed_freq_hz"] = float("nan")
    tracking.track_reassigned_candidate(signal, 384000, candidate)
    assert captured["time_mid"] == 0.011
    assert captured["seed_freq_hz"] == 90000.
    assert captured["duration"] == 0.003


def test_seed_radius_and_bounds_in_diagnostics():
    sr = 384000
    t = np.arange(int(0.020 * sr)) / sr
    y = np.zeros_like(t)
    on = (t >= 0.008) & (t <= 0.011)
    tau = t[on] - 0.008
    y[on] = np.sin(2 * np.pi * (110000 * tau -
                                0.5 * (45000 / .003) * tau**2))
    diag = {}
    curve = track_reassigned_ridge(
        y, sr, time_mid=0.0095, duration=0.003,
        seed_freq_hz=85000., time_bounds_s=(.008, .011),
        diagnostics=diag,
    )
    assert curve is not None
    assert abs(diag["anchor_time_offset_ms"]) <= 0.5 + 1e-6
    assert diag["reassignment_n_fft"] == 512
    assert diag["reassignment_hop"] == 24


def test_modelling_uses_coarse_time_only_with_its_frequency(monkeypatch):
    from bat_analysis import reassignment_tracking as trk
    seen = []
    def fake(y, sr, **kwargs):
        seen.append(kwargs)
        return np.array([[0.0105, 100000.0, 0.8],
                         [0.0110, 90000.0, 0.7],
                         [0.0115, 80000.0, 0.5]])
    monkeypatch.setattr(trk, "track_reassigned_ridge", fake)
    signal = np.zeros(10000)
    common = dict(time_mid=.011, duration=.003, seed_freq_hz=92000.,
                  broad_fm_seed_freq_hz=108000.,
                  chirp_start_s=.009, chirp_end_s=.013,
                  tracking_method="reassignment_flattop")
    process_full_spectrum(
        signal, 384000, **common,
        broad_fm_seed_time_s=.0104,
    )
    assert seen[-1]["time_mid"] == .0104
    assert seen[-1]["seed_freq_hz"] == 108000.
    assert seen[-1]["time_bounds_s"] == (.009, .013)
    process_full_spectrum(signal, 384000, **common)
    assert seen[-1]["time_mid"] == .011
    assert seen[-1]["seed_freq_hz"] == 92000.


def test_cross_seed_penalty_validation_and_diagnostics():
    sr = 384000
    t = np.arange(int(0.020 * sr)) / sr
    y = np.zeros_like(t)
    active = (t >= .007) & (t <= .011)
    tau = t[active] - .007
    y[active] = np.sin(2 * np.pi * (120000 * tau -
                                    .5 * (70000 / .004) * tau**2))
    kwargs = dict(time_mid=.009, duration=.004,
                  seed_freq_hz=85000)
    with pytest.raises(ValueError, match="cross-seed"):
        track_reassigned_ridge(y, sr, cross_seed_position_penalty=-1, **kwargs)
    diag = {}
    curve = track_reassigned_ridge(y, sr, diagnostics=diag, **kwargs)
    assert curve is not None
    assert diag["cross_seed_position_penalty"] == 0.01
    assert "cross_seed_velocity_bins_per_frame" in diag
    legacy_slope = track_reassigned_ridge(
        y, sr, cross_seed_velocity_penalty=0.0,
        cross_seed_position_penalty=0.0, **kwargs)
    assert legacy_slope is not None
