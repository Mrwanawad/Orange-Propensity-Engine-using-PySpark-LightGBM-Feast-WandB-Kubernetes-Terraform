import numpy as np
import pytest

from ope.evaluation import metrics as m
from ope.evaluation.bootstrap import bootstrap_ci


def test_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert m.auc(y, p) == 1.0
    assert m.precision_at(y, p, 0.4) == 1.0


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert m.ece(y, p) < 0.01


def test_ece_detects_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, p.size) < p / 2).astype(int)
    assert m.ece(y, p) > 0.2


def test_lift_random_is_about_one():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=100_000) < 0.1).astype(int)
    lt = m.lift_table(y, rng.uniform(size=y.size))
    assert lt["lift_d1"] == pytest.approx(1.0, abs=0.1)
    assert lt["capture_d3"] == pytest.approx(0.3, abs=0.03)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    p = y * 0.3 + rng.uniform(size=y.size)
    point, lo, hi = bootstrap_ci(y, p, m.auc, n=200)
    assert lo < point < hi


def test_profit_curve_starts_at_zero():
    shares, values = m.profit_curve([1, 0, 1], [0.9, 0.1, 0.8], 10, 1, steps=3)
    assert values[0] == 0 and shares[-1] == 1
