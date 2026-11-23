"""Uplift estimation: designed, NOT trained (section 6.6). Needs campaign A/B data.

Today the policy uses propensity x assumed response rate. Once a campaign runs with a random
holdout, `TwoModelUplift` (T-learner) estimates P(outcome | contacted) - P(outcome | not
contacted) per subscriber and replaces `p * r` in policy/value.py. Qini measures ranking quality.

The tests use a clearly SYNTHETIC randomised treatment. No uplift number is reported anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class TwoModelUplift:
    base: object = field(default_factory=lambda: HistGradientBoostingClassifier(max_iter=200))
    m_treated: object = None
    m_control: object = None

    def fit(self, X, y, treated) -> TwoModelUplift:
        t = np.asarray(treated, bool)
        self.m_treated = clone(self.base).fit(X[t], y[t])
        self.m_control = clone(self.base).fit(X[~t], y[~t])
        return self

    def predict(self, X) -> np.ndarray:
        """Estimated uplift: positive means contacting raises the outcome rate.
        For retention the outcome is 'stayed', so uplift = reduction in churn."""
        return self.m_treated.predict_proba(X)[:, 1] - self.m_control.predict_proba(X)[:, 1]


def qini_curve(y, treated, uplift, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcomes when targeting the top x% by predicted uplift."""
    y, t = np.asarray(y, float), np.asarray(treated, bool)
    order = np.argsort(-np.asarray(uplift), kind="stable")
    y, t = y[order], t[order]
    shares = np.linspace(0, 1, steps + 1)
    q = []
    for s in shares:
        n = int(round(len(y) * s))
        yt, yc = y[:n][t[:n]].sum(), y[:n][~t[:n]].sum()
        nt, nc = t[:n].sum(), (~t[:n]).sum()
        q.append(yt - yc * (nt / nc) if nc else 0.0)
    return shares, np.array(q)


def qini_coefficient(y, treated, uplift) -> float:
    """Area between the model Qini curve and the random-targeting line, normalised by n."""
    shares, q = qini_curve(y, treated, uplift)
    random_line = shares * q[-1]
    return float(np.trapezoid(q - random_line, shares))
