"""Retrain triggers (section 11) and the alert sink.

Retrain when any of:
  - estimated AUC (CBPE) falls more than 0.02 below the reference AUC, for any target;
  - PSI above 0.2 on any of the top-10 features;
  - 8 weeks or more since the champion was trained.
The trigger opens the `ope_retrain` Airflow DAG; promotion still needs the gate.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import httpx

from ope import config

log = logging.getLogger(__name__)
AUC_DROP = 0.02
PSI_LIMIT = 0.2
MAX_AGE_WEEKS = 8


def decide(report: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    reasons = []
    for t, c in report["cbpe"].items():
        drop = c["reference_auc"] - c["estimated_auc"]
        if drop > AUC_DROP:
            reasons.append(f"{t}: estimated AUC {c['estimated_auc']:.4f} is {drop:.4f} below reference {c['reference_auc']:.4f}")
    for f in report["top_features"][:10]:
        v = report["psi"].get(f, 0.0)
        if v > PSI_LIMIT:
            reasons.append(f"PSI {v:.3f} on {f}")
    if report.get("trained_at"):
        age = (now - datetime.fromisoformat(report["trained_at"])).days / 7
        if age >= MAX_AGE_WEEKS:
            reasons.append(f"champion is {age:.1f} weeks old")
    return {"retrain": bool(reasons), "reasons": reasons}


def emit(report: dict) -> None:
    """Append to the alert log; open the retrain DAG when Airflow is reachable."""
    d = report["decision"]
    line = {"at": report["computed_at"], "label": report["label"], **d}
    with open(config.path("reports") / "monitoring" / "alerts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    url = os.environ.get("AIRFLOW_API_URL")
    if d["retrain"] and url:
        try:
            r = httpx.post(
                f"{url}/dags/ope_retrain/dagRuns",
                json={"conf": {"reasons": d["reasons"], "label": report["label"]}},
                auth=(os.environ.get("AIRFLOW_USER", "airflow"), os.environ.get("AIRFLOW_PASSWORD", "airflow")),
                timeout=10,
            )
            log.info("retrain DAG triggered: %s", r.status_code)
        except httpx.HTTPError as e:
            log.error("could not trigger retrain DAG: %s", e)
