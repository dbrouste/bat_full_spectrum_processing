"""Chirp detection algorithms.

The current repository foundation keeps detection separate from annotation so
manual ground truth can benchmark and evolve the detector independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    t_start_ms: float
    t_end_ms: float
    peak_time_ms: float
    peak_freq_khz: float
    score: float | None = None


def detect_chirps(*args, **kwargs) -> List[Detection]:
    """Placeholder public API for the production detector.

    The existing library detector will be migrated here after the annotation
    ground-truth workflow is validated.
    """
    raise NotImplementedError("Detector migration is the next development step.")
