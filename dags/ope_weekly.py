"""Weekly scoring run. Every task is idempotent and keyed by the logical date ({{ ds }}).

ingest -> contract check -> features -> feast materialize -> score + calibrate + policy + SHAP
-> write Iceberg / Redis -> NannyML -> report

A failed contract check stops the run (and Airflow alerts on task failure).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import DAG
from ope_common import EXECUTION, step

default_args = {"retries": 1, "retry_delay": timedelta(minutes=10)}

with DAG(
    dag_id="ope_weekly",
    schedule="0 3 * * 1",  # Mondays 03:00
    start_date=datetime(2026, 11, 2),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["ope", "scoring"],
) as dag:
    ingest = step("ingest", "python -m ope.data.ingest")
    contract = step("contract_check", "python -m ope.data.iceberg --snapshot-date {{ ds }}")
    if EXECUTION == "k8s":
        features = step(
            "features",
            "python -m ope.features.spark_features --snapshot-date {{ ds }} && python -m ope.features.feast_repo apply",
        )
    else:
        features = step("features", "python -m ope.features.feast_repo apply")
    materialize = step("feast_materialize", "python -m ope.features.feast_repo materialize")
    if EXECUTION == "k8s":
        score = step("score_policy_shap", "python -m ope.scoring.batch_spark --table lake.silver.subscribers")
    else:
        score = step("score_policy_shap", "python -m ope.scoring.batch_local --snapshot-date {{ ds }}")
    monitor = step("nannyml", "python -m ope.monitoring.nannyml_jobs --label weekly_{{ ds }}")
    report = step("report", "python -m ope.monitoring.weekly_report --snapshot-date {{ ds }}")

    ingest >> contract >> features >> materialize >> score >> monitor >> report
