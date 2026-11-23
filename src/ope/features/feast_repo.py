"""Feast feature store: offline = parquet (local) / Spark over Iceberg (production), online = Redis.

Feature views
  subscriber_numeric      192 numeric Orange variables
  subscriber_categorical  38 categorical Orange variables
  (domain features)       placeholder: tenure, top-ups, dropped calls ... would plug in here as a
                          third view. The public data is anonymised, so none exist yet.

    python -m ope.features.feast_repo apply        # build offline parquet, register
    python -m ope.features.feast_repo materialize  # load the latest values into Redis
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ope import config
from ope.data.contracts import CATEGORICAL, ID, NUMERIC
from ope.data.split import load_table

REPO = config.ROOT / "configs" / "feast"
OFFLINE = config.ROOT / "data" / "feast" / "subscriber_features.parquet"


def definitions():
    from feast import Entity, FeatureView, Field, FileSource, ValueType
    from feast.types import Float64, String

    subscriber = Entity(name="subscriber", join_keys=[ID], value_type=ValueType.STRING)
    source = FileSource(path=str(OFFLINE), timestamp_field="event_timestamp", created_timestamp_column="created_at")
    ttl = timedelta(days=8)
    numeric = FeatureView(
        name="subscriber_numeric", entities=[subscriber], ttl=ttl, source=source,
        schema=[Field(name=c, dtype=Float64) for c in NUMERIC],
    )
    categorical = FeatureView(
        name="subscriber_categorical", entities=[subscriber], ttl=ttl, source=source,
        schema=[Field(name=c, dtype=String) for c in CATEGORICAL],
    )
    return [subscriber, numeric, categorical]


def store():
    from feast import FeatureStore

    os.environ.setdefault("REDIS_CONNECTION", "localhost:6379")
    return FeatureStore(repo_path=str(REPO))


def build_offline(snapshot: datetime | None = None) -> Path:
    snapshot = snapshot or datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    df = load_table().drop(columns=["churn", "appetency", "upselling"])
    df["event_timestamp"] = pd.Timestamp(snapshot)
    df["created_at"] = pd.Timestamp(datetime.now(UTC))
    for c in CATEGORICAL:
        df[c] = df[c].astype("object")
    OFFLINE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OFFLINE, index=False)
    return OFFLINE


def apply() -> None:
    build_offline()
    store().apply(definitions())


def materialize() -> None:
    # explicit window = the online TTL; both views, idempotent for a weekly rerun
    end = datetime.now(UTC)
    store().materialize(start_date=end - timedelta(days=8), end_date=end)


def online_features(ids: list[str]) -> pd.DataFrame:
    refs = [f"subscriber_numeric:{c}" for c in NUMERIC] + [f"subscriber_categorical:{c}" for c in CATEGORICAL]
    return store().get_online_features(features=refs, entity_rows=[{ID: i} for i in ids]).to_df()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["apply", "materialize"])
    a = ap.parse_args()
    apply() if a.cmd == "apply" else materialize()
