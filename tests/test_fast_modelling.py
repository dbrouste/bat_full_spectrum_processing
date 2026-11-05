"""Check the lightweight Windows multiprocessing entrypoint against modelling.py."""
import numpy as np
import pytest

from bat_analysis.fast_modelling import process_reassigned_fast
from bat_analysis.modelling import process_full_spectrum
from bat_analysis import reassignment_tracking as tracking


def test_fast_model_matches_default_reassignment_convention(monkeypatch):
    received = []
    def fake_track(y, sr, *, diagnostics=None, **kwargs):
        received.append(kwargs)
        if diagnostics is not None:
            diagnostics.update(model_success=True, final_points=3)
        at = kwargs["time_mid"]
        return np.array([
            [at-.0004, 105000., .8],
            [at, 85000., .9],
            [at+.0004, 72000., .5],
        ], dtype=float)
    monkeypatch.setattr(tracking, "track_reassigned_ridge", fake_track)
    wave = np.zeros(19200, dtype=np.float32)
    kw = dict(
        time_mid=.016, duration=.0022, seed_freq_hz=85000.,
        chirp_start_s=.015, chirp_end_s=.0172,
        reassignment_extension_min_snr_db=17.,
        reassignment_cross_seed_velocity_penalty=.6,
    )
    diag_a = {}
    diag_b = {}
    a = process_full_spectrum(wave, 384000, diagnostics=diag_a, **kw)
    b = process_reassigned_fast(wave, 384000, diagnostics=diag_b, **kw)
    np.testing.assert_array_equal(a, b)
    assert received[0] == received[1]
    assert diag_a == diag_b
    assert a[0, 0] == pytest.approx(.016-.0004 - .0085)


def test_coarse_seed_is_paired_with_time(monkeypatch):
    records = []
    def fake_track(y, sr, *, diagnostics=None, **kwargs):
        records.append(kwargs)
        return None
    monkeypatch.setattr(tracking, "track_reassigned_ridge", fake_track)
    x = np.zeros(10000, dtype=np.float32)
    a = dict(
        time_mid=.010, duration=.003,
        seed_freq_hz=90000., broad_fm_seed_freq_hz=110000.,
        broad_fm_seed_time_s=.0097,
    )
    process_reassigned_fast(x, 384000, **a)
    assert records[-1]["time_mid"] == .0097
    assert records[-1]["seed_freq_hz"] == 110000.
    a["broad_fm_seed_time_s"] = None
    process_reassigned_fast(x, 384000, **a)
    assert records[-1]["time_mid"] == .010
    assert records[-1]["seed_freq_hz"] == 90000.
