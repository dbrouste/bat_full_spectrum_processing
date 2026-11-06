import numpy as np
import pytest
from bat_analysis.reassignment import reassigned_spectrogram, reassigned_db, refine_detection_peaks
from bat_analysis.detection import Detection


def test_tone_reassigned_near_true_frequency():
    sr = 384000
    t = np.arange(8192) / sr
    y = np.sin(2 * np.pi * 73500 * t)
    result = reassigned_spectrogram(y, sr, n_fft=512, hop=64,
                                    fmin=20000, fmax=150000,
                                    noise_percentile=20)
    i, j = np.unravel_index(np.argmax(result["reassigned_power"]),
                            result["reassigned_power"].shape)
    assert abs(result["freqs_hz"][i] - 73500) <= sr / 512
    assert result["times_s"].shape[0] == result["stft_power"].shape[1]
    assert np.isfinite(reassigned_db(result)).all()


def test_chirp_downward_and_hybrid_preserves_bounds():
    sr = 384000
    t = np.arange(10000) / sr
    f = 110000 - 40000 * t / t[-1]
    y = np.sin(2 * np.pi * np.cumsum(f) / sr)
    r = reassigned_spectrogram(y, sr, n_fft=512, hop=64,
                               noise_percentile=20)
    assert np.count_nonzero(r["reassigned_power"]) > 0
    ds = [Detection(0, 26, 12, 90)]
    refined = refine_detection_peaks(y, sr, ds, n_fft=512, hop=64,
                                     noise_percentile=20)
    assert refined[0].t_start_ms == 0
    assert refined[0].t_end_ms == 26
    assert len(refined) == 1


def test_bad_inputs():
    with pytest.raises(ValueError):
        reassigned_spectrogram(np.zeros(20), 384000)
    with pytest.raises(ValueError):
        reassigned_spectrogram(np.zeros(1024), 384000, hop=0)
