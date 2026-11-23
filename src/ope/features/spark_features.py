"""Spark feature job: silver.subscribers -> gold.subscriber_features (the Feast offline source in
production, section 4). Model-specific preparation (missing indicators, rare levels, encodings)
lives inside each model bundle (features/prep.py), so it is fitted per CV fold and travels with
the model. This job only shapes rows for point-in-time joins: entity key, event timestamp,
created timestamp, and the raw features.

Domain features (tenure, top-ups, dropped calls) would be added here as extra columns and a new
Feast view; the public data is anonymised, so none exist yet.

    docker compose run --rm jobs python -m ope.features.spark_features --snapshot-date 2026-11-09
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from ope.data.contracts import FEATURES, ID
from ope.data.iceberg import CATALOG, _create, spark_session

log = logging.getLogger(__name__)
GOLD = f"{CATALOG}.gold.subscriber_features"


def build(snapshot_date: str | None = None) -> int:
    from pyspark.sql import functions as F

    snapshot_date = snapshot_date or date.today().isoformat()
    spark = spark_session("ope-features")
    silver = spark.table(f"{CATALOG}.silver.subscribers").where(F.col("snapshot_date") == snapshot_date)
    gold = silver.select(
        ID,
        *FEATURES,
        F.col("snapshot_date"),
        F.to_timestamp(F.col("snapshot_date")).alias("event_timestamp"),
        F.current_timestamp().alias("created_at"),
    )
    _create(spark, GOLD, gold)
    gold.writeTo(GOLD).overwritePartitions()  # idempotent per snapshot_date
    n = spark.table(GOLD).where(F.col("snapshot_date") == snapshot_date).count()
    log.info("%s rows=%d snapshot_date=%s", GOLD, n, snapshot_date)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-date")
    build(ap.parse_args().snapshot_date)
