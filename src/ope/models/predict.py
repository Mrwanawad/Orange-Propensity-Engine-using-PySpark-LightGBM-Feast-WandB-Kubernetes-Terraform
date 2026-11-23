"""Score raw subscriber rows with a frozen bundle. Used by batch scoring, the API and reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ope.data.contracts import FEATURES


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def raw_score(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Uncalibrated score. X holds the raw feature columns (missing ones are treated as NaN)."""
    X = X.reindex(columns=FEATURES)
    base = []
    for part in bundle["members"]:
        Xp = part["prep"].transform(X)
        if part.get("columns"):
            Xp = Xp[part["columns"]]
        base.append(part["model"].predict(Xp))
    if bundle.get("meta") is None:
        return base[0]
    return bundle["meta"].predict_proba(_logit(np.column_stack(base)))[:, 1]


def score(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Calibrated probability."""
    return np.clip(bundle["calibrator"](raw_score(bundle, X)), 0.0, 1.0)
