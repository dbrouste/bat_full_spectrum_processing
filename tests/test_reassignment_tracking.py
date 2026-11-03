"""Non-regression and experimental trajectory tracking tests."""
import numpy as np
import pytest
from bat_analysis.reassignment_tracking import track_reassigned_ridge
from bat_analysis.modelling import process_full_spectrum


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


def test_modelling_new_opt_in_mode():
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
