"""Retraining, opened by a monitoring alert (alerts.emit) or by hand.

Produces `challenger` models only. Promotion to `champion` still needs the gate to pass and a
person to run the promote task (it is skipped unless the run conf has "promote": true).
"""

from __future__ import annotations

from datetime import datetime

from airflow.sdk import DAG
from ope_common import step

with DAG(
    dag_id="ope_retrain",
    schedule=None,
    start_date=datetime(2026, 11, 2),
    catchup=False,
    max_active_runs=1,
    tags=["ope", "training"],
    params={"trials_lightgbm": 40, "promote": False},
) as dag:
    split = step("split", "python -m ope.data.split")
    baseline = step("baseline", "python -m ope.models.baseline")
    tune = step("tune", "python -m ope.models.tune tune --family lightgbm --trials {{ params.trials_lightgbm }} --fresh")
    freeze = step("freeze_and_gate", "python -m ope.models.train freeze")
    promote = step(
        "promote",
        "if [ '{{ params.promote }}' = 'True' ]; then python -m ope.models.train promote; else echo 'promotion skipped: needs promote=true'; fi",
    )
    split >> baseline >> tune >> freeze >> promote
