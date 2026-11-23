"""Structured JSON logs (no feature values), request ids, Prometheus metrics, OpenTelemetry traces."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

REQUESTS = Counter("ope_requests_total", "HTTP requests", ["route", "method", "status"])
LATENCY = Histogram(
    "ope_request_seconds",
    "Request latency",
    ["route"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2, 5),
)
CACHE = Counter("ope_redis_lookups_total", "Precomputed score lookups", ["result"])
BATCH_ROWS = Gauge("ope_batch_rows_scored", "Rows scored by the last batch run")
DRIFT_FLAG = Gauge("ope_drift_retrain_flag", "1 when the last monitoring run asked for retraining")
EST_AUC = Gauge("ope_estimated_auc", "CBPE estimated AUC from the last monitoring run", ["target"])

log = logging.getLogger("ope.api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "client_id", "route", "status", "ms"):
            if hasattr(record, k):
                out[k] = getattr(record, k)
        return json.dumps(out)


def setup_logging() -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [h]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


class RequestContext(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        t = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            ms = (time.perf_counter() - t) * 1000
            REQUESTS.labels(path, request.method, str(status)).inc()
            LATENCY.labels(path).observe(ms / 1000)
            if path not in ("/metrics", "/healthz", "/readyz"):
                log.info(
                    "request",
                    extra={"request_id": rid, "client_id": getattr(request.state, "client_id", None), "route": path, "status": status, "ms": round(ms, 2)},
                )


def setup_tracing(app) -> None:
    """OTLP exporter when OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise tracing stays off."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "ope-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
