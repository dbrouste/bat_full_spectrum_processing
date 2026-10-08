"""Focused regression tests for optional ridge hypothesis dispatch."""
from unittest.mock import patch

import numpy as np
import pytest

from bat_analysis import modelling


def test_default_mode_preserves_seed_dispatch():
    # Mock the expensive ridge engine while checking the unchanged default path.
    with patch.object(modelling.base, "Extract_chunk_of_audio", return_value=np.ones(64)), \
         patch.object(modelling, "_initial_call_trend_seeded", return_value=None) as seed:
        result = modelling.process_full_spectrum(
            np.ones(64), 384000, 0.01, 0.003,
            seed_freq_hz=70000, broad_fm_seed_freq_hz=55000,
            broad_fm_seed_slope_hz_per_ms=-2000,
        )
    assert result is None
    assert seed.call_args.args[3] == 55000
    assert seed.call_args.kwargs["seed_slope_hz_per_ms"] == -2000


def test_peak_mode_suppresses_coarse_seed():
    with patch.object(modelling.base, "Extract_chunk_of_audio", return_value=np.ones(64)), \
         patch.object(modelling, "_initial_call_trend_seeded", return_value=None) as seed:
        modelling.process_full_spectrum(
            np.ones(64), 384000, 0.01, 0.003,
            seed_freq_hz=70000, broad_fm_seed_freq_hz=55000,
            broad_fm_seed_slope_hz_per_ms=-2000, ridge_mode="peak",
        )
    assert seed.call_args.args[3] == 70000
    assert seed.call_args.kwargs["seed_slope_hz_per_ms"] is None


def test_coarse_mode_requires_seed():
    with pytest.raises(ValueError, match="requires a finite"):
        modelling.process_full_spectrum(
            np.ones(64), 384000, 0.01, 0.003,
            seed_freq_hz=70000, ridge_mode="coarse",
        )


def test_comparison_retains_both_outcomes():
    def mock_process(*args, **kwargs):
        if kwargs["ridge_mode"] == "coarse":
            raise RuntimeError("invalid coarse curve")
        kwargs["diagnostics"]["tracking_step_us"] = 40
        return np.array([[0., 70000.], [0.001, 60000.]])

    with patch.object(modelling, "process_full_spectrum", side_effect=mock_process):
        result = modelling.compare_ridge_hypotheses(
            np.ones(64), 384000, 0.01, 0.003,
            seed_freq_hz=70000, coarse_seed_freq_hz=60000,
        )
    assert result["peak"]["curve"].shape == (2, 2)
    assert result["peak"]["diagnostics"]["tracking_step_us"] == 40
    assert result["coarse"]["curve"] is None
    assert "invalid coarse curve" in result["coarse"]["error"]
