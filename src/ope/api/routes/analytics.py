"""Campaign simulation, segment summaries, model health and drift. Used by the console and the
Retention Analyst's MCP tools."""

from __future__ import annotations

import copy
import json
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ope import config
from ope.api.auth import client_id
from ope.api.schemas import SimulateRequest, SimulateResponse
from ope.data.contracts import CATEGORICAL, ID
from ope.data.split import load_table
from ope.policy.allocate import allocate, summary

router = APIRouter(prefix="/v1", dependencies=[Depends(client_id)])
RISK_BANDS = [0, 0.05, 0.10, 0.20, 1.0001]
RISK_LABELS = ["low (<5%)", "medium (5-10%)", "high (10-20%)", "very high (>=20%)"]


def _scores(request: Request) -> pd.DataFrame:
    s = request.app.state.scores
    if s is None:
        raise HTTPException(503, "no batch scores yet; run the weekly batch")
    return s


@router.post("/campaigns/simulate", response_model=SimulateResponse)
def simulate(body: SimulateRequest, request: Request):
    s = _scores(request)
    biz = copy.deepcopy(config.load("business"))
    for a, v in (body.actions or {}).items():
        if v.cost is not None:
            biz["actions"][a]["cost"] = v.cost
        if v.response_rate is not None:
            biz["actions"][a]["response_rate"] = v.response_rate
    cap = {**biz["channel_capacity"], **(body.channel_capacity or {})}
    unknown = set(cap) - set(biz["channel_capacity"])
    if unknown:
        raise HTTPException(422, f"unknown channels: {sorted(unknown)}")
    out = summary(allocate(s, biz, budget=body.budget, capacity=cap))
    return SimulateResponse(budget=body.budget, rows_scored=len(s), model_version=request.app.state.model_version, **out)


@router.post("/campaigns/profit-curve")
def profit_curve(body: SimulateRequest, request: Request, points: int = Query(default=16, ge=2, le=41)):
    """Expected value against budget, from zero to twice the requested budget, same costs."""
    s = _scores(request)
    rows = []
    for b in np.linspace(0, max(body.budget, 1.0) * 2, points):
        r = simulate(SimulateRequest(budget=float(b), channel_capacity=body.channel_capacity, actions=body.actions), request)
        rows.append({"budget": float(b), "expected_value": r.expected_value, "spend": r.spend, "contacts": r.contacts})
    return {"points": rows, "rows_scored": len(s)}


_features_cache: dict[str, pd.DataFrame] = {}


def _segment_column(name: str, s: pd.DataFrame) -> pd.Series:
    if name == "risk_band":
        return pd.cut(s["p_churn"], RISK_BANDS, labels=RISK_LABELS, right=False).astype(str)
    if name == "action":
        p = config.path("processed") / "actions.parquet"
        acts = pd.read_parquet(p, columns=[ID, "action"]).set_index(ID)["action"]
        return s[ID].map(acts).fillna("none")
    if name in CATEGORICAL:
        if "table" not in _features_cache:
            _features_cache["table"] = load_table()[[ID, *CATEGORICAL]].set_index(ID)
        col = _features_cache["table"][name]
        top = col.value_counts().index[:8]
        return s[ID].map(col.where(col.isin(top), "other").fillna("missing")).astype(str)
    raise HTTPException(422, "group_by must be risk_band, action or a categorical feature (Var191..Var229)")


@router.get("/segments/summary")
def segment_summary(request: Request, group_by: str = "risk_band", snapshot: str | None = None):
    """Per segment: subscribers, mean calibrated scores, and planned contacts / EV for the
    current snapshot. `snapshot` picks an older batch run from data/processed/history/."""
    if snapshot:
        p = config.path("processed") / "history" / f"scores_{snapshot}.parquet"
        if not p.exists():
            raise HTTPException(404, f"no snapshot {snapshot}")
        s = pd.read_parquet(p)
    else:
        s = _scores(request)
    seg = _segment_column(group_by, s)
    g = s.assign(segment=seg).groupby("segment", observed=True)
    out = g.agg(
        subscribers=(ID, "size"),
        mean_p_churn=("p_churn", "mean"),
        mean_p_appetency=("p_appetency", "mean"),
        mean_p_upselling=("p_upselling", "mean"),
    ).reset_index()
    return {
        "group_by": group_by,
        "snapshot_date": str(s["snapshot_date"].iloc[0]) if "snapshot_date" in s else None,
        "segments": json.loads(out.round(5).to_json(orient="records")),
        "available_snapshots": sorted(p.stem.removeprefix("scores_") for p in (config.path("processed") / "history").glob("scores_*.parquet")),
    }


@router.get("/model/health")
def model_health(request: Request):
    """CV, calibration and test metrics of the current champions (from generated reports)."""
    rep = config.path("reports")
    ev = json.loads((rep / "evaluation.json").read_text()) if (rep / "evaluation.json").exists() else {}
    fr = json.loads((rep / "cv" / "freeze.json").read_text()) if (rep / "cv" / "freeze.json").exists() else {}
    out = {}
    for t, version in request.app.state.model_version.items():
        e, f = ev.get(t, {}), fr.get(t, {})
        out[t] = {
            "model_version": version,
            "candidate": f.get("candidate"),
            "cv_auc": f.get("cv_auc"),
            "test_auc": e.get("auc"),
            "test_auc_ci95": e.get("auc_ci95"),
            "test_pr_auc": e.get("pr_auc"),
            "test_ece_calibrated": e.get("ece_calibrated"),
            "calibration_method": e.get("calibration_method"),
            "gate_passed": (f.get("gate") or {}).get("passed"),
        }
    return out


@router.get("/monitoring/drift")
def drift(label: str | None = Query(default=None, description="monitoring run label; latest if omitted")):
    d = config.path("reports") / "monitoring"
    files = sorted((p for p in d.glob("*.json")), key=lambda p: p.stat().st_mtime)
    if label:
        files = [p for p in files if p.stem == label]
    if not files:
        raise HTTPException(404, "no monitoring report")
    rep = json.loads(files[-1].read_text())
    top_psi = dict(list(rep["psi"].items())[:10])
    return {
        "label": rep["label"],
        "computed_at": rep["computed_at"],
        "synthetic": rep.get("synthetic"),
        "decision": rep["decision"],
        "cbpe": {t: {k: v for k, v in c.items() if k != "estimated_auc_chunks"} for t, c in rep["cbpe"].items()},
        "top_psi": top_psi,
        "univariate_alert_chunks": rep["univariate_alert_chunks"],
        "multivariate": rep["multivariate"],
        "available_labels": [p.stem for p in files] if not label else None,
    }


@router.get("/profit-curve")
def curve(kind: Literal["policy"] = "policy"):
    p = config.path("processed") / "profit_curve.parquet"
    if not p.exists():
        raise HTTPException(404, "no profit curve yet")
    c = pd.read_parquet(p)
    return {"points": json.loads(c.replace({np.nan: None}).to_json(orient="records"))}
