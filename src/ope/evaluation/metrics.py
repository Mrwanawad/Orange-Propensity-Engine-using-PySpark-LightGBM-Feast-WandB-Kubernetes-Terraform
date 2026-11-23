"""Evaluation metrics for ranked, imbalanced binary targets."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def ece(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected calibration error. Quantile bins, since scores pile up near zero."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability(y, p, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean predicted, observed rate and count per quantile bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ids = np.unique(bins)
    pred = np.array([p[bins == b].mean() for b in ids])
    obs = np.array([y[bins == b].mean() for b in ids])
    cnt = np.array([(bins == b).sum() for b in ids])
    return pred, obs, cnt


def lift_table(y, p, deciles=(1, 2, 3)) -> dict[str, float]:
    """Lift and capture rate in the top k deciles (cumulative)."""
    y = np.asarray(y, float)
    order = np.argsort(-np.asarray(p), kind="stable")
    base = y.mean()
    out = {}
    for d in deciles:
        n = int(round(len(y) * d / 10))
        top = y[order[:n]]
        out[f"lift_d{d}"] = float(top.mean() / base)
        out[f"capture_d{d}"] = float(top.sum() / y.sum())
    return out


def precision_at(y, p, share: float = 0.05) -> float:
    n = max(1, int(round(len(y) * share)))
    order = np.argsort(-np.asarray(p), kind="stable")
    return float(np.asarray(y)[order[:n]].mean())


def summary(y, p) -> dict[str, float]:
    return {
        "auc": auc(y, p),
        "pr_auc": pr_auc(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "precision_top5": precision_at(y, p, 0.05),
        **lift_table(y, p),
    }


def profit_curve(y, p, value_tp: float, cost_contact: float, steps: int = 50):
    """Realised value when contacting the top x% by score (labels known: evaluation only)."""
    y = np.asarray(y, float)
    order = np.argsort(-np.asarray(p), kind="stable")
    shares = np.linspace(0, 1, steps + 1)
    values = []
    for s in shares:
        n = int(round(len(y) * s))
        values.append(y[order[:n]].sum() * value_tp - n * cost_contact)
    return shares, np.array(values)
