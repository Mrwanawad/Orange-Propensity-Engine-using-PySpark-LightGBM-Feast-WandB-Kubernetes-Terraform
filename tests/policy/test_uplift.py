"""Uplift interface on a SYNTHETIC randomised treatment. No real uplift data exists."""

import numpy as np

from ope.policy.uplift import TwoModelUplift, qini_coefficient


def synthetic_campaign(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    treated = rng.uniform(size=n) < 0.5  # randomised
    base = 1 / (1 + np.exp(-(X[:, 0] - 1)))
    true_uplift = 0.25 * (X[:, 1] > 0)  # only half the base responds to the offer
    p = np.clip(base + treated * true_uplift, 0, 1)
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, treated, true_uplift


def test_two_model_recovers_responders():
    X, y, t, true_u = synthetic_campaign()
    m = TwoModelUplift().fit(X, y, t)
    pred = m.predict(X)
    assert pred[true_u > 0].mean() - pred[true_u == 0].mean() > 0.15


def test_qini_beats_random():
    X, y, t, _ = synthetic_campaign(seed=1)
    m = TwoModelUplift().fit(X[:10_000], y[:10_000], t[:10_000])
    model_q = qini_coefficient(y[10_000:], t[10_000:], m.predict(X[10_000:]))
    rand_q = qini_coefficient(y[10_000:], t[10_000:], np.random.default_rng(2).uniform(size=10_000))
    assert model_q > rand_q
