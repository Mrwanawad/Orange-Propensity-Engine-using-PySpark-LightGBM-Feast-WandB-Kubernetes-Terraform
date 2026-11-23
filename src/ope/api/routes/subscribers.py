from __future__ import annotations

import pandas as pd
import pandera.errors as pe
from fastapi import APIRouter, Depends, HTTPException, Request

from ope.api.auth import client_id
from ope.api.schemas import ActionResponse, ScoreRequest, ScoreResult, ScoresResponse
from ope.api.telemetry import CACHE
from ope.data.contracts import FEATURES, TARGETS, feature_payload_schema
from ope.explain.shap_reasons import reasons
from ope.models.predict import score
from ope.policy.allocate import POLICY_VERSION
from ope.policy.value import ACTIONS, expected_values
from ope.scoring import online

router = APIRouter(prefix="/v1", dependencies=[Depends(client_id)])
TARGET_OF = {"retain": "churn", "crosssell": "appetency", "upsell": "upselling"}
_payload_schema = feature_payload_schema()


def _lookup(request: Request, subscriber_id: str) -> dict:
    r = request.app.state.redis
    if r is None:
        raise HTTPException(503, "online store unavailable")
    rec = online.read(subscriber_id, r)
    CACHE.labels("hit" if rec else "miss").inc()
    if rec is None:
        raise HTTPException(404, "no precomputed scores for this subscriber; use POST /v1/score")
    return rec


@router.get("/subscribers/{subscriber_id}/scores", response_model=ScoresResponse)
def get_scores(subscriber_id: str, request: Request):
    rec = _lookup(request, subscriber_id)
    return ScoresResponse(
        subscriber_id=subscriber_id,
        scores=rec["scores"],
        model_version=rec["model_version"],
        snapshot_date=rec["snapshot_date"],
        source="precomputed",
    )


@router.get("/subscribers/{subscriber_id}/next-best-action", response_model=ActionResponse)
def get_action(subscriber_id: str, request: Request):
    rec = _lookup(request, subscriber_id)
    act = rec["action"] or {"action": "none", "ev": 0.0, "cost": 0.0, "channel": None, "policy_version": POLICY_VERSION}
    return ActionResponse(
        subscriber_id=subscriber_id,
        action=act["action"],
        expected_value=act["ev"],
        cost=act["cost"],
        channel=act["channel"],
        reasons=rec["reasons"] or [],
        policy_version=act["policy_version"],
        model_version=rec["model_version"],
        snapshot_date=rec["snapshot_date"],
    )


@router.post("/score", response_model=ScoreResult)
def score_on_demand(body: ScoreRequest, request: Request):
    unknown = set(body.features) - set(FEATURES)
    if unknown:
        raise HTTPException(422, f"unknown features: {sorted(unknown)[:5]}")
    row = pd.DataFrame([{"subscriber_id": body.subscriber_id, **body.features}])
    try:
        row = _payload_schema.validate(row)
    except (pe.SchemaError, pe.SchemaErrors) as e:
        raise HTTPException(422, f"payload failed the data contract: {str(e)[:300]}") from None
    X = row.reindex(columns=FEATURES)
    bundles = request.app.state.bundles
    p = {f"p_{t}": float(score(bundles[t], X)[0]) for t in TARGETS}
    ev = expected_values(pd.DataFrame([p])).iloc[0]
    best = max(ACTIONS, key=lambda a: ev[f"ev_{a}"])
    action = best if ev[f"ev_{best}"] > 0 else "none"
    why = None
    if body.explain and action != "none":
        why = reasons(bundles[TARGET_OF[action]], X)[0]
    return ScoreResult(
        subscriber_id=body.subscriber_id,
        scores=p,
        action=action,
        expected_value=round(float(ev[f"ev_{best}"]) if action != "none" else 0.0, 2),
        reasons=why,
        model_version=request.app.state.model_version,
    )
