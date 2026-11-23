"""Greedy next-best-action allocation (section 7.2).

Constraints: at most one action per subscriber, total budget, contacts per channel, suppression
list, EV > 0. Every step is a sort, a rank within a group or a cumulative sum, so the same logic
runs as Spark window functions at 100M rows (see scoring/batch_spark.py).

Pass 1: each subscriber's best action; keep the top `capacity` per channel by EV per unit cost;
        then fill the budget in EV-per-cost order.
Pass 2: subscribers cut only by a full channel get their next-best action in channels that
        still have room, with the budget that is left.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ope import config
from ope.policy.value import ACTIONS, expected_values

POLICY_VERSION = "greedy-v1"


def _channels(biz) -> dict[str, str]:
    return {a: biz["actions"][a]["channel"] for a in ACTIONS}


def _fill(cand: pd.DataFrame, capacity: dict[str, int], budget: float) -> pd.DataFrame:
    """cand: one row per subscriber (idx, action, ev, cost, channel, ratio). Returns the chosen rows
    plus a flag for rows cut by channel capacity."""
    cand = cand.sort_values(["ratio", "ev"], ascending=False, kind="stable")
    rank = cand.groupby("channel").cumcount()
    cap = cand["channel"].map(capacity).fillna(0).to_numpy()
    within_cap = rank.to_numpy() < cap
    kept = cand[within_cap]
    fits = kept["cost"].cumsum().to_numpy() <= budget + 1e-9
    chosen = kept[fits]
    capacity_cut = cand[~within_cap]
    return chosen, capacity_cut


def allocate(
    scores: pd.DataFrame,
    biz: dict[str, Any] | None = None,
    suppressed: np.ndarray | None = None,
    budget: float | None = None,
    capacity: dict[str, int] | None = None,
) -> pd.DataFrame:
    """scores: subscriber_id, p_churn, p_appetency, p_upselling. Returns one row per subscriber."""
    biz = biz or config.load("business")
    budget = float(biz["budget"] if budget is None else budget)
    capacity = dict(biz["channel_capacity"] if capacity is None else capacity)
    chan = _channels(biz)
    ev = expected_values(scores, biz)
    n = len(scores)
    ev_mat = ev[[f"ev_{a}" for a in ACTIONS]].to_numpy()
    cost_mat = ev[[f"cost_{a}" for a in ACTIONS]].to_numpy()
    allowed = ev_mat > 0
    if suppressed is not None:
        allowed &= ~np.asarray(suppressed, bool)[:, None]

    def candidates(mask: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
        masked = np.where(mask[rows], ev_mat[rows], -np.inf)
        best = masked.argmax(axis=1)
        ok = np.isfinite(masked[np.arange(len(rows)), best])
        r, b = rows[ok], best[ok]
        e, c = ev_mat[r, b], cost_mat[r, b]
        acts = np.array(ACTIONS)[b]
        return pd.DataFrame(
            {
                "idx": r,
                "action": acts,
                "ev": e,
                "cost": c,
                "channel": [chan[a] for a in acts],
                "ratio": e / np.maximum(c, 1e-9),
            }
        )

    # pass 1
    chosen1, cut = _fill(candidates(allowed, np.arange(n)), capacity, budget)
    left_cap = {ch: capacity.get(ch, 0) - int((chosen1["channel"] == ch).sum()) for ch in capacity}
    left_budget = budget - float(chosen1["cost"].sum())
    # pass 2: only subscribers cut by capacity, only channels with room
    chosen = [chosen1]
    if len(cut) and left_budget > 0:
        rows = cut["idx"].to_numpy()
        mask2 = allowed.copy()
        for j, a in enumerate(ACTIONS):
            if left_cap.get(chan[a], 0) <= 0:
                mask2[:, j] = False
        chosen2, _ = _fill(candidates(mask2, rows), left_cap, left_budget)
        chosen.append(chosen2)
    sel = pd.concat(chosen)

    out = pd.DataFrame(
        {
            "subscriber_id": scores["subscriber_id"].to_numpy(),
            "action": "none",
            "ev": 0.0,
            "cost": 0.0,
            "channel": None,
        }
    )
    out.loc[sel["idx"].to_numpy(), ["action", "ev", "cost", "channel"]] = sel[
        ["action", "ev", "cost", "channel"]
    ].to_numpy()
    for a in ACTIONS:
        out[f"ev_{a}"] = ev[f"ev_{a}"].to_numpy()
    out["ev"] = out["ev"].astype(float)
    out["cost"] = out["cost"].astype(float)
    out["policy_version"] = POLICY_VERSION
    return out


def summary(actions: pd.DataFrame) -> dict[str, Any]:
    used = actions[actions["action"] != "none"]
    return {
        "contacts": int(len(used)),
        "expected_value": float(used["ev"].sum()),
        "spend": float(used["cost"].sum()),
        "per_action": used.groupby("action")
        .agg(n=("ev", "size"), ev=("ev", "sum"), cost=("cost", "sum"))
        .to_dict(orient="index"),
        "per_channel": used.groupby("channel").size().to_dict(),
    }


def profit_curve(
    scores: pd.DataFrame, budgets: np.ndarray, biz: dict[str, Any] | None = None, **kw
) -> pd.DataFrame:
    rows = []
    for b in budgets:
        s = summary(allocate(scores, biz, budget=float(b), **kw))
        rows.append(
            {
                "budget": float(b),
                "expected_value": s["expected_value"],
                "spend": s["spend"],
                "contacts": s["contacts"],
            }
        )
    return pd.DataFrame(rows)
