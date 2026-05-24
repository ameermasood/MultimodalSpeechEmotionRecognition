"""Reusable evaluation metrics for emotion recognition experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)


def reliability_ece(confidence: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected calibration error from confidence and correctness arrays."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    mask = np.isfinite(conf)
    conf, corr = conf[mask], corr[mask]
    if len(conf) == 0:
        return float("nan"), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        in_bin = bin_ids == bin_idx
        if in_bin.sum() > 0:
            bin_acc[bin_idx] = corr[in_bin].mean()
            bin_conf[bin_idx] = conf[in_bin].mean()
            bin_count[bin_idx] = in_bin.sum()

    ece = float(np.sum((bin_count / max(1, len(conf))) * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf, bin_count


def selective_accuracy_curve(confidence: Sequence[float], correct: Sequence[float], n_points: int = 20) -> tuple[list[float], list[float], float]:
    """Compute selective accuracy over descending confidence thresholds."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    mask = np.isfinite(conf)
    conf, corr = conf[mask], corr[mask]
    if len(conf) == 0:
        return [], [], float("nan")

    order = np.argsort(-conf)
    correct_sorted = corr[order]
    coverages = np.linspace(0.1, 1.0, n_points)
    coverage_values: list[float] = []
    accuracy_values: list[float] = []

    for coverage in coverages:
        keep = max(1, int(round(coverage * len(correct_sorted))))
        accuracy = float(correct_sorted[:keep].mean())
        coverage_values.append(float(keep / len(correct_sorted)))
        accuracy_values.append(accuracy)

    risk_values = np.asarray([1.0 - acc for acc in accuracy_values], dtype=np.float64)
    risk = float(np.sum((risk_values[:-1] + risk_values[1:]) * np.diff(coverage_values) / 2.0))
    return coverage_values, accuracy_values, risk


def mcnemar_from_two_preds(y_true: Sequence[Any], y_pred_a: Sequence[Any], y_pred_b: Sequence[Any]) -> dict[str, Any]:
    """Compute McNemar's continuity-corrected chi-square statistic."""
    truth = np.asarray(y_true)
    a_correct = np.asarray(y_pred_a) == truth
    b_correct = np.asarray(y_pred_b) == truth

    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    divergent = n01 + n10
    statistic = float(((abs(n01 - n10) - 1) ** 2) / divergent) if divergent > 0 else 0.0

    return {
        "n01_a_wrong_b_right": n01,
        "n10_a_right_b_wrong": n10,
        "total_divergent": int(divergent),
        "chi_sq_stat": statistic,
        "significant_p05": bool(statistic > 3.841),
    }


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    confidence: Sequence[float] | None = None,
    latencies_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute common classification, calibration, and latency metrics."""
    metrics: dict[str, Any] = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "true_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
    }

    if latencies_ms is not None:
        latencies = np.asarray(latencies_ms, dtype=np.float64)
        latencies = latencies[np.isfinite(latencies)]
        if len(latencies) > 0:
            metrics.update(
                {
                    "latency_ms_p50": float(np.percentile(latencies, 50)),
                    "latency_ms_p90": float(np.percentile(latencies, 90)),
                    "latency_ms_p99": float(np.percentile(latencies, 99)),
                    "latency_ms_mean": float(np.mean(latencies)),
                }
            )

    if confidence is not None:
        correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(np.float64)
        ece, _, _, _ = reliability_ece(confidence, correct)
        _, _, risk = selective_accuracy_curve(confidence, correct)
        metrics["ece_10bins"] = float(ece) if np.isfinite(ece) else float("nan")
        metrics["selective_risk_area"] = float(risk) if np.isfinite(risk) else float("nan")

    return metrics
