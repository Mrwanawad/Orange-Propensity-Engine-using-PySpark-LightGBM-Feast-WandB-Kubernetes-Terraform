"""Weekly batch scoring in Spark: score -> policy -> SHAP for actioned -> Iceberg gold + Redis.

Scoring is `mapInPandas` with the frozen bundles broadcast once and Arrow batches of ~50k rows.
The policy is the same greedy as policy/allocate.py, made distributed: instead of a global sort
with a running sum (one partition at 100M rows), each pass aggregates cost and count per
(channel, EV-per-cost bucket), the driver finds the cut-off bucket per channel and for the
budget, and executors filter. Rows in the boundary bucket are left out, so it never exceeds
budget or capacity and is at most one bucket below the pandas version.

    docker compose run --rm jobs python -m ope.scoring.batch_spark --table lake.silver.subscribers
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from datetime import date

import numpy as np
import pandas as pd

from ope import config
from ope.data.contracts import FEATURES, ID, TARGETS
from ope.models import registry
from ope.policy.allocate import POLICY_VERSION
from ope.policy.value import ACTIONS, SCORE_COL, value_per_success

log = logging.getLogger(__name__)
BUCKET = 1e-3  # EV-per-cost bucket width used for the distributed cut-offs


def score_fn(bundles_bc):
    def fn(batches):
        from ope.models.predict import score

        bundles = pickle.loads(bundles_bc.value)
        for b in batches:
            out = pd.DataFrame({ID: b[ID].to_numpy()})
            for t in TARGETS:
                out[f"p_{t}"] = score(bundles[t], b[FEATURES])
            yield out

    return fn


def _cutoffs(df, capacity: dict[str, int], budget: float, F) -> tuple[dict[str, float], float]:
    """Per-channel ratio cut-off for capacity, then one global cut-off for the budget."""
    hist = (
        df.withColumn("bucket", F.floor(F.col("ratio") / BUCKET))
        .groupBy("channel", "bucket")
        .agg(F.count("*").alias("n"), F.sum("cost").alias("cost"))
        .toPandas()
        .sort_values("bucket", ascending=False)
    )
    chan_cut = {}
    kept = []
    for ch, g in hist.groupby("channel"):
        cum = g["n"].cumsum()
        ok = g[cum <= capacity.get(ch, 0)]
        chan_cut[ch] = float(ok["bucket"].min()) if len(ok) else np.inf
        kept.append(ok)
    kept_df = pd.concat(kept) if kept else hist.iloc[:0]
    kept_df = kept_df.groupby("bucket")["cost"].sum().sort_index(ascending=False)
    cum = kept_df.cumsum()
    ok = kept_df[cum <= budget + 1e-9]
    budget_cut = float(ok.index.min()) if len(ok) else np.inf
    return chan_cut, budget_cut


def _apply(df, chan_cut, budget_cut, F):
    cond = F.lit(False)
    for ch, cut in chan_cut.items():
        if np.isfinite(cut):
            cond = cond | ((F.col("channel") == ch) & (F.col("bucket") >= F.lit(cut)))
    if not np.isfinite(budget_cut):
        return df.where(F.lit(False))
    return df.withColumn("bucket", F.floor(F.col("ratio") / BUCKET)).where(cond & (F.col("bucket") >= F.lit(budget_cut)))


def best_action(df, biz, F, allowed: dict[str, bool] | None = None):
    """Adds ev_<a>, cost_<a>, then action/ev/cost/channel/ratio of the best positive action."""
    v = value_per_success(biz)
    allowed = allowed or dict.fromkeys(ACTIONS, True)
    for a in ACTIONS:
        df = df.withColumn(f"ev_{a}", F.col(SCORE_COL[a]) * F.lit(v[a]) - F.lit(float(biz["actions"][a]["cost"])))
    best = None
    for a in ACTIONS:
        if not allowed[a]:
            continue
        ev = F.col(f"ev_{a}")
        cand = F.struct(ev.alias("ev"), F.lit(a).alias("action"), F.lit(float(biz["actions"][a]["cost"])).alias("cost"), F.lit(biz["actions"][a]["channel"]).alias("channel"))
        best = cand if best is None else F.when(ev > best["ev"], cand).otherwise(best)
    df = df.withColumn("best", best)
    return (
        df.withColumn("action", F.col("best.action"))
        .withColumn("ev", F.col("best.ev"))
        .withColumn("cost", F.col("best.cost"))
        .withColumn("channel", F.col("best.channel"))
        .withColumn("ratio", F.col("ev") / F.col("cost"))
        .drop("best")
        .where(F.col("ev") > 0)
    )


def allocate_spark(scores, biz, suppressed=None):
    from pyspark.sql import functions as F

    capacity = dict(biz["channel_capacity"])
    budget = float(biz["budget"])
    base = scores if suppressed is None else scores.join(suppressed, ID, "left_anti")
    cand = best_action(base, biz, F).withColumn("bucket", F.floor(F.col("ratio") / BUCKET)).cache()
    chan_cut, budget_cut = _cutoffs(cand, capacity, budget, F)
    chosen1 = _apply(cand, chan_cut, budget_cut, F).cache()
    used = {r["channel"]: (r["n"], r["c"]) for r in chosen1.groupBy("channel").agg(F.count("*").alias("n"), F.sum("cost").alias("c")).collect()}
    left_cap = {ch: capacity[ch] - used.get(ch, (0, 0))[0] for ch in capacity}
    left_budget = budget - sum(c for _, c in used.values())
    # pass 2: rows cut by a full channel, next-best action in channels with room
    cond = F.lit(False)
    for ch, cut in chan_cut.items():
        cond = cond | ((F.col("channel") == ch) & ((F.col("bucket") < F.lit(cut)) if np.isfinite(cut) else F.lit(True)))
    cut_by_cap = cand.join(chosen1.select(ID), ID, "left_anti").where(cond).select(ID, *[SCORE_COL[a] for a in ACTIONS])
    allowed = {a: left_cap.get(biz["actions"][a]["channel"], 0) > 0 for a in ACTIONS}
    chosen = chosen1
    if any(allowed.values()) and left_budget > 0:
        cand2 = best_action(cut_by_cap, biz, F, allowed).withColumn("bucket", F.floor(F.col("ratio") / BUCKET))
        c2, b2 = _cutoffs(cand2, left_cap, left_budget, F)
        chosen = chosen1.unionByName(_apply(cand2, c2, b2, F), allowMissingColumns=True)
    out = scores.join(chosen.select(ID, "action", "ev", "cost", "channel"), ID, "left")
    return out.fillna({"action": "none", "ev": 0.0, "cost": 0.0}).withColumn("policy_version", F.lit(POLICY_VERSION))


def reasons_fn(bundles_bc):
    def fn(batches):
        from ope.explain.shap_reasons import reasons

        bundles = pickle.loads(bundles_bc.value)
        target_of = {"retain": "churn", "crosssell": "appetency", "upsell": "upselling", "none": "churn"}
        for b in batches:
            out = []
            for action, g in b.groupby("action"):
                rs = reasons(bundles[target_of[action]], g[FEATURES].reset_index(drop=True))
                out.append(pd.DataFrame({ID: g[ID].to_numpy(), "reasons": [json.dumps(r) for r in rs]}))
            yield pd.concat(out) if out else pd.DataFrame({ID: [], "reasons": []})

    return fn


def run(table: str, snapshot_date: str | None = None, write_redis: bool = True, limit: int | None = None) -> dict:
    from pyspark.sql import functions as F

    from ope.data.iceberg import CATALOG, spark_session

    snapshot_date = snapshot_date or date.today().isoformat()
    biz = config.load("business")
    spark = spark_session("ope-batch-scoring")
    bundles = {t: registry.load_bundle(t, "champion") for t in TARGETS}
    versions = {t: b["version"] for t, b in bundles.items()}
    bc = spark.sparkContext.broadcast(pickle.dumps(bundles))
    src = spark.table(table)
    if "snapshot_date" in src.columns:
        latest = src.agg(F.max("snapshot_date")).collect()[0][0]
        src = src.where(F.col("snapshot_date") == latest)
    if limit:
        src = src.limit(limit)
    src = src.select(ID, *FEATURES).cache()
    n = src.count()

    t0 = time.time()
    schema = f"{ID} string, " + ", ".join(f"p_{t} double" for t in TARGETS)
    scores = src.mapInPandas(score_fn(bc), schema=schema).cache()
    scores.count()
    score_secs = time.time() - t0

    t1 = time.time()
    actions = allocate_spark(scores, biz).cache()
    summary = {r["action"]: (r["n"], r["ev"]) for r in actions.groupBy("action").agg(F.count("*").alias("n"), F.sum("ev").alias("ev")).collect()}
    policy_secs = time.time() - t1

    # SHAP only for actioned subscribers plus a 0.1% monitoring sample
    t2 = time.time()
    to_explain = actions.where("action != 'none'").select(ID, "action").unionByName(
        actions.where("action = 'none'").sample(0.001, seed=1).select(ID, "action")
    )
    reason_df = to_explain.join(src, ID).mapInPandas(reasons_fn(bc), schema=f"{ID} string, reasons string").cache()
    n_reasons = reason_df.count()
    shap_secs = time.time() - t2

    gold = actions.join(reason_df, ID, "left").withColumn("snapshot_date", F.lit(snapshot_date).cast("date"))
    for t in TARGETS:
        gold = gold.withColumn(f"model_version_{t}", F.lit(versions[t]))
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.gold")
    target_table = f"{CATALOG}.gold.actions"
    if not spark.catalog.tableExists(target_table):
        gold.limit(0).writeTo(target_table).using("iceberg").partitionedBy(F.col("snapshot_date")).create()
    gold.writeTo(target_table).overwritePartitions()

    if write_redis:
        from ope.scoring import online

        def push(rows):
            pdf = pd.DataFrame([r.asDict() for r in rows])
            if len(pdf):
                reasons = {r[ID]: json.loads(r["reasons"]) for _, r in pdf.dropna(subset=["reasons"]).iterrows()}
                online.write(pdf, pdf, reasons, versions, snapshot_date)

        gold.foreachPartition(push)

    info = {
        "rows": n,
        "score_seconds": round(score_secs, 1),
        "policy_seconds": round(policy_secs, 1),
        "shap_rows": n_reasons,
        "shap_seconds": round(shap_secs, 1),
        "actions": {k: {"n": v[0], "ev": round(v[1] or 0, 2)} for k, v in summary.items()},
        "executor_cores": spark.sparkContext.defaultParallelism,
        "model_versions": versions,
    }
    log.info("batch scoring %s", info)
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="lake.silver.subscribers")
    ap.add_argument("--no-redis", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(run(a.table, write_redis=not a.no_redis, limit=a.limit), indent=2))
