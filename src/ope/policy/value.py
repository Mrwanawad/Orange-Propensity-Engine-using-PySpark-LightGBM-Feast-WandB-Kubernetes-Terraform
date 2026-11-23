"""Expected value per subscriber and action (section 7.1). All inputs from configs/business.yaml.

    EV_retain    = p_churn * r_retain * CLV_saved              - cost_retain
    EV_crosssell = p_app   * r_cross  * margin_new             - cost_cross
    EV_upsell    = p_up    * r_up     * delta_ARPU * m * H     - cost_up
    EV_none      = 0

CLV_saved = ARPU * m * extra tenure. Up-sell uses margin (m), not revenue, so the three actions
are compared in the same unit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ope import config

ACTIONS = ["retain", "crosssell", "upsell"]
SCORE_COL = {"retain": "p_churn", "crosssell": "p_appetency", "upsell": "p_upselling"}


def value_per_success(biz: dict[str, Any]) -> dict[str, float]:
    a, m = biz["actions"], biz["margin_rate"]
    return {
        "retain": a["retain"]["response_rate"]
        * biz["arpu_monthly"]
        * m
        * a["retain"]["extra_tenure_months"],
        "crosssell": a["crosssell"]["response_rate"] * a["crosssell"]["margin_new"],
        "upsell": a["upsell"]["response_rate"]
        * a["upsell"]["delta_arpu"]
        * m
        * a["upsell"]["horizon_months"],
    }


def expected_values(scores: pd.DataFrame, biz: dict[str, Any] | None = None) -> pd.DataFrame:
    """scores: columns p_churn, p_appetency, p_upselling. Returns EV_<action> and cost_<action>."""
    biz = biz or config.load("business")
    v = value_per_success(biz)
    out = pd.DataFrame(index=scores.index)
    for act in ACTIONS:
        cost = float(biz["actions"][act]["cost"])
        out[f"ev_{act}"] = scores[SCORE_COL[act]].to_numpy(float) * v[act] - cost
        out[f"cost_{act}"] = cost
    return out


def breakeven(biz: dict[str, Any] | None = None) -> dict[str, float]:
    """Probability above which each action has positive EV."""
    biz = biz or config.load("business")
    v = value_per_success(biz)
    return {a: float(np.clip(biz["actions"][a]["cost"] / v[a], 0, 1)) for a in ACTIONS}
