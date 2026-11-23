"""Single-node weekly run for the 50k base: score -> policy -> reasons -> parquet, campaign CSV, Redis.
Same steps as batch_spark.py, in pandas. Used by the local pipeline, the API demo and the console.

    python -m ope.scoring.batch_local [--no-redis]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from ope import config
from ope.data.contracts import FEATURES, ID, TARGETS
from ope.data.split import load_table
from ope.explain.shap_reasons import reasons
from ope.models import registry
from ope.models.predict import score
from ope.policy.allocate import allocate, profit_curve, summary

log = logging.getLogger(__name__)
TARGET_OF = {"retain": "churn", "crosssell": "appetency", "upsell": "upselling"}


def suppression(ids: pd.Series) -> np.ndarray:
    """Suppression list (recent contact, opt-out). data/processed/suppression.csv if present."""
    p = config.path("processed") / "suppression.csv"
    if not p.exists():
        return np.zeros(len(ids), bool)
    return ids.isin(set(pd.read_csv(p)[ID].astype(str))).to_numpy()


def run(write_redis: bool = True, snapshot_date: str | None = None) -> dict:
    snapshot_date = snapshot_date or date.today().isoformat()
    df = load_table()
    bundles = {t: registry.load_bundle(t, "champion") for t in TARGETS}
    versions = {t: b["version"] for t, b in bundles.items()}
    t0 = time.time()
    scores = pd.DataFrame({ID: df[ID]})
    for t in TARGETS:
        scores[f"p_{t}"] = score(bundles[t], df[FEATURES])
    score_s = time.time() - t0
    actions = allocate(scores, suppressed=suppression(df[ID]))
    acted = actions["action"] != "none"
    sample = (~acted) & (np.random.default_rng(1).uniform(size=len(df)) < 0.001)
    reason_map: dict[str, list] = {}
    for act, tgt in TARGET_OF.items():
        m = (actions["action"] == act).to_numpy()
        if m.any():
            for sid, r in zip(df.loc[m, ID], reasons(bundles[tgt], df.loc[m, FEATURES].reset_index(drop=True)), strict=True):
                reason_map[sid] = r
    if sample.any():
        for sid, r in zip(df.loc[sample, ID], reasons(bundles["churn"], df.loc[sample, FEATURES].reset_index(drop=True)), strict=True):
            reason_map[sid] = r

    out = config.path("processed")
    hist = out / "history"
    hist.mkdir(exist_ok=True)
    scores.assign(snapshot_date=snapshot_date).to_parquet(hist / f"scores_{snapshot_date}.parquet", index=False)
    scores.assign(snapshot_date=snapshot_date, **{f"model_version_{t}": v for t, v in versions.items()}).to_parquet(out / "scores.parquet", index=False)
    actions.assign(
        reasons=actions[ID].map(lambda s: json.dumps(reason_map[s]) if s in reason_map else None),
        snapshot_date=snapshot_date,
    ).to_parquet(out / "actions.parquet", index=False)
    camp = actions[acted][[ID, "action", "channel", "ev", "cost"]].sort_values("ev", ascending=False)
    camp.to_csv(out / f"campaign_{snapshot_date}.csv", index=False)
    biz = config.load("business")
    curve = profit_curve(scores, np.linspace(0, biz["budget"] * 2, 41))
    curve.to_parquet(out / "profit_curve.parquet", index=False)
    n_redis = 0
    if write_redis:
        from ope.scoring import online

        n_redis = online.write(scores, actions, reason_map, versions, snapshot_date)
    info = {"rows": len(df), "score_seconds": round(score_s, 2), "redis_keys": n_redis, "policy": summary(actions), "model_versions": versions}
    (config.path("reports") / "last_batch_run.json").write_text(json.dumps(info, indent=2, default=float))
    log.info("local batch: %s", info)
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-redis", action="store_true")
    ap.add_argument("--snapshot-date")
    a = ap.parse_args()
    run(write_redis=not a.no_redis, snapshot_date=a.snapshot_date)
