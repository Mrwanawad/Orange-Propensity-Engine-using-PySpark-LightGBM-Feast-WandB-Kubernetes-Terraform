"""API tests with a tiny model trained on the synthetic fixture and fakeredis."""

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ope.data.contracts import FEATURES, TARGETS
from ope.features.prep import Prep
from ope.models.calibrate import Calibrator
from ope.models.gbm import Model
from ope.policy.allocate import allocate
from ope.scoring import online
from tests.conftest import make_fixture

KEY = {"X-API-Key": "test-key"}


def tiny_bundle(df, target):
    prep = Prep(cat_mode="te")
    X = prep.fit_transform(df[FEATURES], df[target].to_numpy())
    m = Model("lightgbm", {"n_estimators": 30, "learning_rate": 0.1, "min_child_samples": 10}, weighted=False).fit(X, df[target].to_numpy())
    raw = m.predict(X)
    return {
        "target": target,
        "members": [{"family": "lightgbm", "prep": prep, "model": m, "columns": None}],
        "meta": None,
        "calibrator": Calibrator("platt").fit(raw, df[target].to_numpy()),
        "version": f"{target}-test",
        "metadata": {"candidate": "lightgbm", "cv_auc": 0.7},
    }


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    df = make_fixture(1500, seed=4)
    bundles = {t: tiny_bundle(df, t) for t in TARGETS}
    from ope.models.predict import score

    scores = pd.DataFrame({"subscriber_id": df["subscriber_id"]})
    for t in TARGETS:
        scores[f"p_{t}"] = score(bundles[t], df[FEATURES])
    actions = allocate(scores)
    r = fakeredis.FakeRedis(decode_responses=True)
    sid = df["subscriber_id"].iloc[0]
    online.write(scores, actions, {sid: [{"feature": "Var1", "signal": "value", "value": 1.0, "shap": 0.2}]}, {t: f"{t}-test" for t in TARGETS}, "2026-11-09", r=r)
    return df, bundles, scores, r


@pytest.fixture
def client(setup, monkeypatch):
    df, bundles, scores, r = setup
    monkeypatch.setenv("OPE_API_KEYS", "test-key")
    monkeypatch.setattr("ope.models.registry.load_bundle", lambda t, alias="champion": bundles[t])
    monkeypatch.setattr("ope.scoring.online.client", lambda: r)
    from ope.api.main import app

    with TestClient(app) as c:
        app.state.scores = scores
        yield c


def test_auth_required(client):
    assert client.get("/v1/subscribers/x/scores").status_code == 401
    assert client.get("/v1/subscribers/x/scores", headers={"X-API-Key": "wrong"}).status_code == 401


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz").json()
    assert r["models_loaded"] and r["redis"]


def test_precomputed_scores_and_action(client, setup):
    sid = setup[0]["subscriber_id"].iloc[0]
    r = client.get(f"/v1/subscribers/{sid}/scores", headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert set(body["scores"]) == {"p_churn", "p_appetency", "p_upselling"}
    assert body["model_version"]["churn"] == "churn-test"
    assert "x-request-id" in r.headers
    a = client.get(f"/v1/subscribers/{sid}/next-best-action", headers=KEY).json()
    assert a["action"] in {"retain", "crosssell", "upsell", "none"}
    assert a["reasons"][0]["feature"] == "Var1"


def test_unknown_subscriber_404(client):
    assert client.get("/v1/subscribers/ffffffffffffffff/scores", headers=KEY).status_code == 404


def test_on_demand_score(client, setup):
    row = setup[0].iloc[3]
    feats = {c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], float | np.floating) else str(row[c]))) for c in FEATURES[:20] + ["Var192", "Var218"]}
    r = client.post("/v1/score", headers=KEY, json={"subscriber_id": "a" * 16, "features": feats, "explain": True})
    assert r.status_code == 200, r.text
    p = r.json()["scores"]
    assert all(0 <= v <= 1 for v in p.values())


def test_on_demand_rejects_bad_payload(client):
    r = client.post("/v1/score", headers=KEY, json={"subscriber_id": "a" * 16, "features": {"Var999": 1.0}})
    assert r.status_code == 422
    r = client.post("/v1/score", headers=KEY, json={"subscriber_id": "short", "features": {}})
    assert r.status_code == 422


def test_simulate_respects_budget(client):
    r = client.post("/v1/campaigns/simulate", headers=KEY, json={"budget": 500, "actions": {"retain": {"cost": 10}}})
    assert r.status_code == 200
    body = r.json()
    assert body["spend"] <= 500 and body["rows_scored"] == 1500


def test_rate_limit(client, monkeypatch):
    from ope.api import auth

    monkeypatch.setattr(auth.limiter, "limit", 3)
    codes = [client.get("/v1/subscribers/x/scores", headers=KEY).status_code for _ in range(5)]
    assert 429 in codes


def test_metrics_exposed(client):
    client.get("/healthz")
    assert "ope_requests_total" in client.get("/metrics").text
