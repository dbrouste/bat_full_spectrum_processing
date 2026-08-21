from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    inter = max(0.0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_intervals(
    reference: Sequence[Tuple[float, float]],
    detected: Sequence[Tuple[float, float]],
    min_iou: float = 0.1,
) -> DetectionMetrics:
    """Greedy one-to-one interval matching for detection benchmarking."""
    unmatched = set(range(len(detected)))
    tp = 0
    for ref in reference:
        best_j = None
        best_iou = min_iou
        for j in unmatched:
            score = interval_iou(ref, detected[j])
            if score >= best_iou:
                best_iou = score
                best_j = j
        if best_j is not None:
            tp += 1
            unmatched.remove(best_j)
    fp = len(unmatched)
    fn = len(reference) - tp
    return DetectionMetrics(tp, fp, fn)
