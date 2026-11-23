"""FastMCP server exposing the Retention Analyst's tools. Every tool calls the OPE API, so the
agent sees exactly what the console and the care screens see.

    python -m ope.analyst.mcp_server            # stdio (spawned by the agent)
    python -m ope.analyst.mcp_server --http     # streamable HTTP on :8765/mcp
"""

from __future__ import annotations

import os
import sys
from typing import Literal

import httpx
from fastmcp import FastMCP

API = os.environ.get("OPE_API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ.get("OPE_API_KEY", "local-dev-key")}

mcp = FastMCP("ope-retention-analyst", instructions="Read-only tools over the Orange Propensity Engine API.")


def _get(path: str, **params) -> dict:
    r = httpx.get(f"{API}{path}", headers=HEADERS, params={k: v for k, v in params.items() if v is not None}, timeout=30)
    if r.status_code == 404:
        return {"error": "not found", "detail": r.json().get("detail")}
    r.raise_for_status()
    return r.json()


@mcp.tool
def get_segment_summary(group_by: str = "risk_band", snapshot: str | None = None) -> dict:
    """Subscribers and mean calibrated churn / appetency / up-sell probability per segment.
    group_by: 'risk_band', 'action', or a categorical feature such as 'Var218'.
    snapshot: an older batch date (YYYY-MM-DD) to compare with; omit for the current one."""
    return _get("/v1/segments/summary", group_by=group_by, snapshot=snapshot)


@mcp.tool
def explain_subscriber(subscriber_id: str) -> dict:
    """Scores, next best action, expected value and top SHAP reasons for one subscriber."""
    s = _get(f"/v1/subscribers/{subscriber_id}/scores")
    if "error" in s:
        return s
    a = _get(f"/v1/subscribers/{subscriber_id}/next-best-action")
    return {"scores": s["scores"], "next_best_action": a, "model_version": s["model_version"]}


@mcp.tool
def simulate_campaign(
    budget: float,
    retain_cost: float | None = None,
    crosssell_cost: float | None = None,
    upsell_cost: float | None = None,
    call_centre_capacity: int | None = None,
    sms_capacity: int | None = None,
    app_push_capacity: int | None = None,
) -> dict:
    """Run the next-best-action policy for a budget (EGP) and optional per-contact costs and
    channel capacities. Returns contacts, spend and expected value per action and channel."""
    actions = {}
    for name, cost in (("retain", retain_cost), ("crosssell", crosssell_cost), ("upsell", upsell_cost)):
        if cost is not None:
            actions[name] = {"cost": cost}
    caps = {k: v for k, v in (("call_centre", call_centre_capacity), ("sms", sms_capacity), ("app_push", app_push_capacity)) if v is not None}
    body: dict = {"budget": budget, "actions": actions or None}
    if caps:
        body["channel_capacity"] = caps  # channels left out keep their configured capacity
    r = httpx.post(f"{API}/v1/campaigns/simulate", headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


@mcp.tool
def get_model_health() -> dict:
    """CV and test AUC, calibration error and gate status of the champion model per target."""
    return _get("/v1/model/health")


@mcp.tool
def get_drift_report(label: str | None = None) -> dict:
    """Latest data-drift and estimated-performance (CBPE) report, or one by label."""
    return _get("/v1/monitoring/drift", label=label)


def main(transport: Literal["stdio", "http"] = "stdio") -> None:
    if transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "8765")))
    else:
        mcp.run()


if __name__ == "__main__":
    main("http" if "--http" in sys.argv else "stdio")
