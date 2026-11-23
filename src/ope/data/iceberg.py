"""Iceberg bronze / silver tables, written with Spark through the REST catalog on S3.

bronze.subscribers_raw  rows as joined from the raw ARFF files, plus ingestion metadata
silver.subscribers      the same rows after the Pandera contract passed

Both are partitioned by snapshot_date and bucket(64, subscriber_id). Training reads a
pinned silver snapshot id (export_silver) and logs it to W&B.

Run inside the jobs image:  docker compose run --rm jobs python -m ope.data.iceberg
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date

import pandas as pd

from ope import config
from ope.data.contracts import ID, pandas_schema

log = logging.getLogger(__name__)
CATALOG = "lake"


def spark_session(app: str = "ope"):
    from pyspark.sql import SparkSession

    s3 = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
    builder = (
        SparkSession.builder.appName(app)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "rest")
        .config(
            f"spark.sql.catalog.{CATALOG}.uri",
            os.environ.get("ICEBERG_REST_URI", "http://localhost:8181"),
        )
        .config(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"spark.sql.catalog.{CATALOG}.s3.endpoint", s3)
        .config(f"spark.sql.catalog.{CATALOG}.s3.path-style-access", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "64"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "6g"))
    )
    return builder.getOrCreate()


def _create(spark, table: str, df) -> None:
    ns = table.rsplit(".", 1)[0]
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ns}")
    if not spark.catalog.tableExists(table):
        (
            df.limit(0)
            .writeTo(table)
            .using("iceberg")
            .partitionedBy(df["snapshot_date"], _bucket(df))
            .create()
        )


def _bucket(df):
    from pyspark.sql.functions.partitioning import bucket

    return bucket(64, df[ID])


def write_bronze_silver(snapshot_date: str | None = None) -> dict:
    from pyspark.sql import functions as F

    snapshot_date = snapshot_date or date.today().isoformat()
    spark = spark_session("ope-ingest")
    manifest = json.loads((config.path("raw") / "MANIFEST.json").read_text())
    pdf = pd.read_parquet(config.path("interim") / "subscribers.parquet")
    sdf = spark.createDataFrame(pdf).withColumn("snapshot_date", F.lit(snapshot_date).cast("date"))

    bronze = f"{CATALOG}.bronze.subscribers_raw"
    meta = sdf.withColumn("ingested_at", F.current_timestamp()).withColumn(
        "source_sha256", F.lit(",".join(m["sha256"][:12] for m in manifest.values()))
    )
    _create(spark, bronze, meta)
    # idempotent per snapshot_date: overwrite that partition only
    meta.writeTo(bronze).overwritePartitions()

    # contract: per Arrow batch with Pandera, plus a global uniqueness check in Spark
    schema = pandas_schema()

    def validate(batches):
        for b in batches:
            schema.validate(b.drop(columns=["snapshot_date"]), lazy=True)
            yield b

    checked = sdf.mapInPandas(validate, schema=sdf.schema)
    dupes = checked.groupBy(ID).count().filter("count > 1").count()
    if dupes:
        raise ValueError(f"contract failed: {dupes} duplicate subscriber ids")
    silver = f"{CATALOG}.silver.subscribers"
    _create(spark, silver, sdf)
    checked.writeTo(silver).overwritePartitions()

    snap = spark.sql(
        f"SELECT snapshot_id, committed_at FROM {silver}.snapshots ORDER BY committed_at DESC LIMIT 1"
    ).collect()[0]
    info = {
        "table": silver,
        "snapshot_id": int(snap["snapshot_id"]),
        "committed_at": str(snap["committed_at"]),
        "snapshot_date": snapshot_date,
        "rows": spark.table(silver).where(F.col("snapshot_date") == snapshot_date).count(),
    }
    (config.path("processed") / "silver_snapshot.json").write_text(json.dumps(info, indent=2))
    log.info("silver snapshot %s", info)
    return info


def export_silver(snapshot_id: int | None = None) -> pd.DataFrame:
    """Read a pinned silver snapshot with PyIceberg (no Spark) for the single-node training path."""
    from pyiceberg.catalog import load_catalog

    cat = load_catalog(
        CATALOG,
        type="rest",
        uri=os.environ.get("ICEBERG_REST_URI", "http://localhost:8181"),
        **{
            "s3.endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:8333"),
            "s3.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID", "localadmin"),
            "s3.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY", "localadmin"),
            "s3.path-style-access": "true",
            "s3.region": "us-east-1",
        },
    )
    table = cat.load_table("silver.subscribers")
    if snapshot_id is None:
        snapshot_id = json.loads((config.path("processed") / "silver_snapshot.json").read_text())[
            "snapshot_id"
        ]
    return table.scan(snapshot_id=snapshot_id).to_pandas()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-date")
    write_bronze_silver(ap.parse_args().snapshot_date)
