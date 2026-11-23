"""Online API. Reads precomputed scores and actions from Redis; never calls Spark.
POST /v1/score runs the in-process champion models for new or changed subscribers.

    uvicorn ope.api.main:app --workers 4
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ope import config
from ope.api import telemetry
from ope.api.routes import analytics, subscribers
from ope.data.contracts import TARGETS
from ope.models import registry

log = logging.getLogger("ope.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    telemetry.setup_logging()
    app.state.bundles = {t: registry.load_bundle(t, "champion") for t in TARGETS}
    app.state.model_version = {t: b["version"] for t, b in app.state.bundles.items()}
    scores_path = config.path("processed") / "scores.parquet"
    app.state.scores = pd.read_parquet(scores_path) if scores_path.exists() else None
    try:
        from ope.scoring.online import client

        app.state.redis = client()
        app.state.redis.ping()
    except Exception as e:
        log.warning("redis unavailable: %s", e)
        app.state.redis = None
    yield


app = FastAPI(
    title="Orange Propensity Engine API",
    version="1.0.0",
    description="Calibrated churn / appetency / up-sell scores and next-best actions. "
    "Not affiliated with Orange; money values are illustrative assumptions.",
    lifespan=lifespan,
)
app.add_middleware(telemetry.RequestContext)
telemetry.setup_tracing(app)
app.include_router(subscribers.router)
app.include_router(analytics.router)


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readyz(response: Response):
    ready = bool(getattr(app.state, "bundles", None))
    redis_ok = False
    if getattr(app.state, "redis", None) is not None:
        try:
            redis_ok = bool(app.state.redis.ping())
        except Exception:
            redis_ok = False
    if not ready:
        response.status_code = 503
    return {"models_loaded": ready, "redis": redis_ok, "model_version": getattr(app.state, "model_version", {})}


def _refresh_batch_gauges() -> None:
    """Batch and monitoring gauges come from the latest report files."""
    import json

    rep = config.path("reports")
    run = rep / "last_batch_run.json"
    if run.exists():
        telemetry.BATCH_ROWS.set(json.loads(run.read_text())["rows"])
    mon = sorted((rep / "monitoring").glob("*.json"), key=lambda p: p.stat().st_mtime) if (rep / "monitoring").exists() else []
    if mon:
        m = json.loads(mon[-1].read_text())
        telemetry.DRIFT_FLAG.set(int(m["decision"]["retrain"]))
        for t, c in m["cbpe"].items():
            telemetry.EST_AUC.labels(t).set(c["estimated_auc"])


@app.get("/metrics", include_in_schema=False)
def metrics():
    try:
        _refresh_batch_gauges()
    except Exception as e:  # metrics must never fail the scrape
        log.warning("gauge refresh failed: %s", e)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
