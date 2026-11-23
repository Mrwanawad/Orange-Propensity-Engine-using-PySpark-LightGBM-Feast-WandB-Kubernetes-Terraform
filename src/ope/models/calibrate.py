"""Isotonic vs Platt calibration on the calibration split.

The method is chosen by cross-validated Brier score inside the calibration split (fitting and
scoring on the same rows would always favour isotonic), then refitted on the whole split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ope.evaluation.metrics import brier, ece


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class Calibrator:
    method: str
    model: object = None

    def fit(self, p, y) -> Calibrator:
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        elif self.method == "platt":
            self.model = LogisticRegression(C=1e6).fit(_logit(p), y)
        else:
            raise ValueError(self.method)
        return self

    def __call__(self, p) -> np.ndarray:
        if self.method == "isotonic":
            return self.model.predict(np.asarray(p, float))
        return self.model.predict_proba(_logit(p))[:, 1]


def choose(p, y, seed: int = 42) -> tuple[Calibrator, dict]:
    """Pick isotonic or Platt by 5-fold Brier on (p, y); report out-of-fold Brier and ECE."""
    p, y = np.asarray(p, float), np.asarray(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    report = {"raw": {"brier": brier(y, p), "ece": ece(y, p)}}
    for method in ("isotonic", "platt"):
        oof = np.zeros_like(p)
        for tr, va in skf.split(p, y):
            oof[va] = Calibrator(method).fit(p[tr], y[tr])(p[va])
        report[method] = {"brier": brier(y, oof), "ece": ece(y, oof)}
    best = min(("isotonic", "platt"), key=lambda m: report[m]["brier"])
    report["chosen"] = best
    return Calibrator(best).fit(p, y), report
