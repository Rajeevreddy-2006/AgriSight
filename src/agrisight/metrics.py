"""Evaluation metrics in plain NumPy (accuracy, calibration, selective prediction)."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(1) == labels).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: gap between confidence and accuracy, averaged over bins."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())


def risk_coverage(probs: np.ndarray, labels: np.ndarray):
    """Error rate when keeping only the most confident predictions.

    Returns (coverage, risk, AURC). Lower AURC = confidence ranks errors better.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    n = len(c)
    counts = np.arange(1, n + 1)
    coverage = counts / n
    risk = 1.0 - np.cumsum(c) / counts
    return coverage, risk, float(risk.mean())


def selective(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Coverage and accuracy when the model abstains below `threshold` confidence."""
    conf = probs.max(1)
    keep = conf >= threshold
    correct = probs.argmax(1) == labels
    return {
        "threshold": threshold,
        "coverage": float(keep.mean()),
        "accuracy_on_kept": float(correct[keep].mean()) if keep.any() else float("nan"),
        "errors_caught": float((~correct & ~keep).sum() / max(1, (~correct).sum())),
    }
