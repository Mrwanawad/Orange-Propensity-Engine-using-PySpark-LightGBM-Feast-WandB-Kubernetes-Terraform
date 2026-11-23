import numpy as np
import pandas as pd

from ope.monitoring.psi import psi_categorical, psi_frame, psi_numeric


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=50_000), rng.normal(size=50_000)) < 0.01


def test_shift_is_detected():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=50_000), rng.normal(0.5, 1, size=50_000)) > 0.2


def test_missing_share_change_counts():
    ref = np.r_[np.ones(900), [np.nan] * 100]
    cur = np.r_[np.ones(500), [np.nan] * 500]
    assert psi_numeric(ref, cur) > 0.2


def test_categorical_new_levels_pool_into_other():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi_categorical(ref, ref) < 1e-9
    cur = pd.Series(["a"] * 500 + ["z"] * 500)
    assert psi_categorical(ref, cur) > 1


def test_psi_is_symmetric_in_sign():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=10_000), rng.normal(0.3, 1, size=10_000)
    assert psi_numeric(a, b) > 0 and psi_numeric(b, a) > 0


def test_frame_sorted():
    ref = pd.DataFrame({"x": np.arange(1000.0), "c": ["a", "b"] * 500})
    cur = pd.DataFrame({"x": np.arange(1000.0) + 500, "c": ["a", "b"] * 500})
    s = psi_frame(ref, cur, ["x", "c"])
    assert s.index[0] == "x"
