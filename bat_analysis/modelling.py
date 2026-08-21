"""Compact models fitted to extracted bat chirp ridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ChirpModelResult:
    model_name: str
    parameters: Dict[str, float]
    diagnostics: Dict[str, Any]


def fit_chirp_model(*args, **kwargs) -> ChirpModelResult:
    """Public API placeholder for model fitting."""
    raise NotImplementedError("Model fitting will be selected after ridge benchmarking.")
