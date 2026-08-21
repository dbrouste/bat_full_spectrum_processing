"""Frequency-ridge extraction from detected chirps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RidgePoint:
    t_ms: float
    f_khz: float
    db: float | None = None


def extract_ridge(*args, **kwargs) -> List[RidgePoint]:
    """Public API placeholder for automatic f(t) reconstruction."""
    raise NotImplementedError("Ridge extraction will be implemented against manual annotations.")
