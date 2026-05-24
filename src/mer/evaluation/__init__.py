"""Evaluation metrics, reports, and comparison utilities."""

from mer.evaluation.metrics import (
    classification_metrics,
    mcnemar_from_two_preds,
    reliability_ece,
    selective_accuracy_curve,
)

__all__ = [
    "classification_metrics",
    "mcnemar_from_two_preds",
    "reliability_ece",
    "selective_accuracy_curve",
]
