"""Redis online store for precomputed scores and actions (section 4.1).

One hash per subscriber, key `ope:sub:{subscriber_id}`, fields:
  scores         JSON {p_churn, p_appetency, p_upselling}
  action         JSON {action, ev, cost, channel, policy_version}
  reasons        JSON list of reason codes (only for actioned subscribers + monitoring sample)
  model_version  JSON {churn: ..., appetency: ..., upselling: ...}
  snapshot_date  ISO date of the scoring run
TTL 8 days, so a missed weekly run shows up as cache misses rather than stale answers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import pandas as pd

KEY = "ope:sub:{}"
TTL_SECONDS = 8 * 24 * 3600


def client():
    import redis

    return redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)


def _records(scores: pd.DataFrame, actions: pd.DataFrame | None, reasons: dict[str, list] | None,
             model_version: dict[str, str], snapshot_date: str) -> Iterable[tuple[str, dict[str, str]]]:
    acts = actions.set_index("subscriber_id") if actions is not None else None
    mv = json.dumps(model_version)
    for row in scores.itertuples(index=False):
        sid = row.subscriber_id
        h = {
            "scores": json.dumps({
                "p_churn": round(float(row.p_churn), 6),
                "p_appetency": round(float(row.p_appetency), 6),
                "p_upselling": round(float(row.p_upselling), 6),
            }),
            "model_version": mv,
            "snapshot_date": snapshot_date,
        }
        if acts is not None and sid in acts.index:
            a = acts.loc[sid]
            h["action"] = json.dumps({
                "action": a["action"], "ev": round(float(a["ev"]), 2), "cost": float(a["cost"]),
                "channel": a["channel"], "policy_version": a["policy_version"],
            })
        if reasons and sid in reasons:
            h["reasons"] = json.dumps(reasons[sid])
        yield KEY.format(sid), h


def write(scores, actions=None, reasons=None, model_version=None, snapshot_date="", r=None, batch: int = 5000) -> int:
    r = r or client()
    n = 0
    pipe = r.pipeline(transaction=False)
    for key, h in _records(scores, actions, reasons, model_version or {}, snapshot_date):
        pipe.delete(key)
        pipe.hset(key, mapping=h)
        pipe.expire(key, TTL_SECONDS)
        n += 1
        if n % batch == 0:
            pipe.execute()
    pipe.execute()
    return n


def read(subscriber_id: str, r=None) -> dict[str, Any] | None:
    r = r or client()
    h = r.hgetall(KEY.format(subscriber_id))
    if not h:
        return None
    out: dict[str, Any] = {"subscriber_id": subscriber_id, "snapshot_date": h.get("snapshot_date")}
    for f in ("scores", "action", "reasons", "model_version"):
        out[f] = json.loads(h[f]) if f in h else None
    return out
