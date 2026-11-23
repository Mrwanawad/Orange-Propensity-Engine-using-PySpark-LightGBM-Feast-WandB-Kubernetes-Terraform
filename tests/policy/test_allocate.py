"""Property tests (section 12): one action per subscriber, never over budget or capacity,
suppressed subscribers never contacted, EV always positive."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from ope import config
from ope.policy.allocate import allocate, summary
from ope.policy.optimal_check import gap_report
from ope.policy.value import breakeven, expected_values


def make_scores(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "subscriber_id": [f"{i:016d}" for i in range(n)],
            "p_churn": rng.beta(1, 8, n),
            "p_appetency": rng.beta(1, 30, n),
            "p_upselling": rng.beta(1, 8, n),
        }
    )


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(1, 400),
    seed=st.integers(0, 10_000),
    budget=st.floats(0, 5000),
    caps=st.tuples(st.integers(0, 200), st.integers(0, 200), st.integers(0, 200)),
    supp_share=st.floats(0, 1),
)
def test_allocation_invariants(n, seed, budget, caps, supp_share):
    biz = config.load("business")
    scores = make_scores(n, seed)
    capacity = dict(zip(["call_centre", "sms", "app_push"], caps, strict=True))
    suppressed = np.random.default_rng(seed + 1).uniform(size=n) < supp_share
    out = allocate(scores, biz, suppressed=suppressed, budget=budget, capacity=capacity)
    assert (
        len(out) == n and out["subscriber_id"].is_unique
    )  # one row, so one action, per subscriber
    used = out[out["action"] != "none"]
    assert used["cost"].sum() <= budget + 1e-6
    for ch, cap in capacity.items():
        assert (used["channel"] == ch).sum() <= cap
    assert not used["subscriber_id"].isin(out.loc[suppressed, "subscriber_id"]).any()
    assert (used["ev"] > 0).all()


def test_ev_matches_breakeven():
    biz = config.load("business")
    be = breakeven(biz)
    s = pd.DataFrame(
        {"p_churn": [be["retain"]], "p_appetency": [be["crosssell"]], "p_upselling": [be["upsell"]]}
    )
    ev = expected_values(s, biz)
    np.testing.assert_allclose(
        ev[["ev_retain", "ev_crosssell", "ev_upsell"]].to_numpy(), 0, atol=1e-9
    )


def test_unconstrained_takes_every_positive_best_action():
    biz = config.load("business")
    s = make_scores(2000, 3)
    out = allocate(
        s, biz, budget=1e12, capacity={"call_centre": 10**9, "sms": 10**9, "app_push": 10**9}
    )
    ev = expected_values(s, biz)[["ev_retain", "ev_crosssell", "ev_upsell"]]
    assert (out["action"] != "none").sum() == (ev.max(axis=1) > 0).sum()


def test_refill_uses_second_best_channel():
    biz = config.load("business")
    # everyone prefers retain, but the call centre takes only one: others should fall back
    s = pd.DataFrame(
        {
            "subscriber_id": ["a" * 16, "b" * 16, "c" * 16],
            "p_churn": [0.9, 0.9, 0.9],
            "p_appetency": [0.0, 0.0, 0.0],
            "p_upselling": [0.8, 0.8, 0.8],
        }
    )
    out = allocate(s, biz, budget=1e6, capacity={"call_centre": 1, "sms": 0, "app_push": 10})
    assert sorted(out["action"]) == ["retain", "upsell", "upsell"]


def test_greedy_gap_small_on_sample():
    biz = config.load("business")
    s = make_scores(3000, 7)
    rep = gap_report(s, biz, time_limit=30, budget=2000.0)
    assert rep["gap_vs_bound"] < 0.01
    assert summary(allocate(s, biz, budget=2000.0))["spend"] <= 2000.0
