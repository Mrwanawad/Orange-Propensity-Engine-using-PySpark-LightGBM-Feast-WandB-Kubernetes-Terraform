"""API load test (section 12): p50 / p95 / p99 at 200 concurrent users.

Start the API with a high rate limit and scores in Redis, then:
    OPE_RATE_LIMIT_PER_MINUTE=1000000 uvicorn ope.api.main:app --workers 4
    locust -f load/locustfile.py --host http://localhost:8000 -u 200 -r 20 -t 3m \
        --headless --csv reports/locust/run

Traffic mix mirrors the care-screen pattern: mostly precomputed reads, some on-demand scoring,
a few campaign simulations.
"""

from __future__ import annotations

import os
import random

import pandas as pd
from locust import HttpUser, between, task

ROOT = os.path.join(os.path.dirname(__file__), "..")
HEADERS = {"X-API-Key": os.environ.get("OPE_API_KEY", "local-dev-key")}
_table = pd.read_parquet(os.path.join(ROOT, "data", "interim", "subscribers.parquet"))
IDS = _table["subscriber_id"].tolist()
FEATURE_ROWS = [
    {k: (None if pd.isna(v) else v) for k, v in row.items() if k.startswith("Var")}
    for row in _table.sample(500, random_state=0).to_dict(orient="records")
]
del _table


class CareScreen(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(10)
    def scores(self):
        self.client.get(f"/v1/subscribers/{random.choice(IDS)}/scores", headers=HEADERS, name="/v1/subscribers/{id}/scores")

    @task(10)
    def next_best_action(self):
        self.client.get(f"/v1/subscribers/{random.choice(IDS)}/next-best-action", headers=HEADERS, name="/v1/subscribers/{id}/next-best-action")

    @task(3)
    def score_on_demand(self):
        body = {"subscriber_id": f"{random.getrandbits(64):016x}", "features": random.choice(FEATURE_ROWS)}
        self.client.post("/v1/score", json=body, headers=HEADERS, name="/v1/score")

    @task(1)
    def simulate(self):
        body = {"budget": random.choice([20000, 60000, 120000])}
        self.client.post("/v1/campaigns/simulate", json=body, headers=HEADERS, name="/v1/campaigns/simulate")
