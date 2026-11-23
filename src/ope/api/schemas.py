from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Scores(BaseModel):
    p_churn: float
    p_appetency: float
    p_upselling: float


class ScoresResponse(BaseModel):
    subscriber_id: str
    scores: Scores
    model_version: dict[str, str]
    snapshot_date: str | None
    source: Literal["precomputed", "on_demand"]


class Reason(BaseModel):
    feature: str
    signal: str
    value: float | str | None
    shap: float


class ActionResponse(BaseModel):
    subscriber_id: str
    action: Literal["retain", "crosssell", "upsell", "none"]
    expected_value: float
    cost: float
    channel: str | None
    reasons: list[Reason]
    policy_version: str
    model_version: dict[str, str]
    snapshot_date: str | None


class ScoreRequest(BaseModel):
    subscriber_id: str = Field(min_length=16, max_length=16)
    features: dict[str, float | str | None] = Field(
        description="Raw features Var1..Var230; missing keys are treated as missing values"
    )
    explain: bool = False


class ScoreResult(BaseModel):
    subscriber_id: str
    scores: Scores
    action: str
    expected_value: float
    reasons: list[Reason] | None
    model_version: dict[str, str]
    source: Literal["on_demand"] = "on_demand"


class ActionCost(BaseModel):
    cost: float | None = Field(default=None, ge=0)
    response_rate: float | None = Field(default=None, ge=0, le=1)


class SimulateRequest(BaseModel):
    budget: float = Field(ge=0)
    channel_capacity: dict[str, int] | None = None
    actions: dict[Literal["retain", "crosssell", "upsell"], ActionCost] | None = None


class SimulateResponse(BaseModel):
    budget: float
    contacts: int
    expected_value: float
    spend: float
    per_action: dict[str, dict[str, Any]]
    per_channel: dict[str, int]
    rows_scored: int
    assumptions_note: str = "All money values are illustrative assumptions (configs/business.yaml)."
    model_version: dict[str, str]
