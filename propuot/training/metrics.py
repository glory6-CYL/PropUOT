from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if np.unique(labels).size < 2:
        return {"auroc": 0.5, "auprc": float(labels.mean())}
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    samples: int = 1000,
    seed: int = 42,
) -> dict:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    point = binary_metrics(labels, probabilities)
    if samples <= 0:
        return point
    generator = np.random.default_rng(seed)
    auroc: list[float] = []
    auprc: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled = binary_metrics(sampled_labels, probabilities[indices])
        auroc.append(sampled["auroc"])
        auprc.append(sampled["auprc"])
    if not auroc:
        return {
            **point,
            "auroc_ci95": [point["auroc"], point["auroc"]],
            "auprc_ci95": [point["auprc"], point["auprc"]],
        }
    return {
        **point,
        "auroc_ci95": [float(np.percentile(auroc, 2.5)), float(np.percentile(auroc, 97.5))],
        "auprc_ci95": [float(np.percentile(auprc, 2.5)), float(np.percentile(auprc, 97.5))],
        "bootstrap_samples_used": len(auroc),
    }

