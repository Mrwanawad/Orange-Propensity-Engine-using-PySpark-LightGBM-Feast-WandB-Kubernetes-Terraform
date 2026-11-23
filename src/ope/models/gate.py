"""Promotion gate (section 6.4). All checks must pass for a model to become `challenger`."""

from __future__ import annotations

import numpy as np

from ope import config


def check(
    fold_auc: list[float],
    baseline_auc: float,
    calibration_ece: float,
    champion: dict | None = None,
) -> dict:
    g = config.load()["gate"]
    mean = float(np.mean(fold_auc))
    se = float(np.std(fold_auc, ddof=1) / np.sqrt(len(fold_auc)))
    checks = {
        "beats_baseline": mean - baseline_auc >= g["min_gain_over_baseline"],
        "not_worse_than_champion": champion is None
        or mean >= champion["cv_auc"] - champion.get("cv_se", se),
        "calibrated": calibration_ece <= g["max_ece"],
        "stable_folds": min(fold_auc) >= mean - g["max_fold_drop"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cv_auc": mean,
        "cv_se": se,
        "baseline_auc": baseline_auc,
        "calibration_ece": calibration_ece,
        "champion_auc": None if champion is None else champion["cv_auc"],
        "worst_fold_gap": mean - min(fold_auc),
    }
