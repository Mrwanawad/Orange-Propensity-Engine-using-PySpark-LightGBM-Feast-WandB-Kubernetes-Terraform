"""Throughput benchmark of the Spark scoring + policy on the SYNTHETIC 10M replica.

Measures rows/sec per core for scoring (3 calibrated models, mapInPandas, parquet read included)
and the time of the distributed policy. Results go to reports/bench_spark.json; docs/capacity.md
turns them into the 100M sizing. Features are streamed from parquet, not cached: 10M rows x 230
features would not fit in a laptop's Docker memory, and a production run reads from Iceberg the
same way.

    docker compose run --rm jobs python scripts/bench_spark.py --rows 10000000
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from datetime import UTC, datetime

from ope import config
from ope.data.contracts import FEATURES, ID, TARGETS
from ope.data.iceberg import spark_session
from ope.models import registry
from ope.scoring.batch_spark import allocate_spark, score_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000_000)
    a = ap.parse_args()
    spark = spark_session("ope-bench")
    cores = spark.sparkContext.defaultParallelism
    bundles = {t: registry.load_bundle(t, "champion") for t in TARGETS}
    bc = spark.sparkContext.broadcast(pickle.dumps(bundles))
    # whole parquet parts (500k rows each) rather than .limit(), which would serialise the read
    parts = sorted((config.ROOT / "data" / "replica").glob("part-*.parquet"))
    parts = parts[: max(1, math.ceil(a.rows / 500_000))]
    src = spark.read.parquet(*[str(p) for p in parts]).select(ID, *FEATURES)
    n = src.count()

    schema = f"{ID} string, " + ", ".join(f"p_{t} double" for t in TARGETS)
    t0 = time.time()
    scores = src.mapInPandas(score_fn(bc), schema=schema).cache()
    scores.count()
    score_s = time.time() - t0

    biz = dict(config.load("business"))
    scale = n / 50_000  # budget and capacities are set for the 50k base; scale them linearly
    biz["budget"] = biz["budget"] * scale
    biz["channel_capacity"] = {k: int(v * scale) for k, v in biz["channel_capacity"].items()}
    t1 = time.time()
    acts = allocate_spark(scores, biz).cache()
    planned = acts.where("action != 'none'").count()
    policy_s = time.time() - t1

    rps_core = n / score_s / cores
    res = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": n,
        "cores": cores,
        "score_seconds": round(score_s, 1),
        "rows_per_sec_per_core_3_models": round(rps_core, 1),
        "policy_seconds": round(policy_s, 1),
        "planned_contacts": planned,
        "machine": "laptop i7-11370H (4C/8T), Spark local mode in Docker",
        "data": "SYNTHETIC replica (scripts/make_scale_replica.py)",
    }
    (config.path("reports") / "bench_spark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
