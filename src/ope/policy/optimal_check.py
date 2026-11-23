"""Exact solution of the allocation problem with OR-Tools CP-SAT, to measure the greedy gap.

    max  sum EV[i,a] x[i,a]
    s.t. sum_a x[i,a] <= 1                 each subscriber at most one action
         sum cost[i,a] x[i,a] <= budget
         sum_{i, a in channel c} x[i,a] <= capacity[c]
         x[i,a] = 0 if suppressed or EV <= 0

Money is scaled to integer cents. Reports the solver's best bound too, so the gap is honest
even if the time limit stops it before proving optimality.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from ope import config
from ope.policy.allocate import allocate, summary
from ope.policy.value import ACTIONS, expected_values


def solve(
    scores: pd.DataFrame,
    biz=None,
    budget=None,
    capacity=None,
    suppressed=None,
    time_limit: float = 120.0,
) -> dict:
    biz = biz or config.load("business")
    budget = float(biz["budget"] if budget is None else budget)
    capacity = dict(biz["channel_capacity"] if capacity is None else capacity)
    ev = expected_values(scores, biz)
    evm = np.round(ev[[f"ev_{a}" for a in ACTIONS]].to_numpy() * 100).astype(np.int64)
    cm = np.round(ev[[f"cost_{a}" for a in ACTIONS]].to_numpy() * 100).astype(np.int64)
    allowed = evm > 0
    if suppressed is not None:
        allowed &= ~np.asarray(suppressed, bool)[:, None]
    m = cp_model.CpModel()
    x = {}
    for i, j in zip(*np.nonzero(allowed), strict=True):
        x[i, j] = m.NewBoolVar(f"x{i}_{j}")
    by_sub: dict[int, list] = {}
    for (i, _j), v in x.items():
        by_sub.setdefault(i, []).append(v)
    for vs in by_sub.values():
        if len(vs) > 1:
            m.AddAtMostOne(vs)
    m.Add(sum(int(cm[i, j]) * v for (i, j), v in x.items()) <= int(round(budget * 100)))
    for c, cap in capacity.items():
        vs = [v for (i, j), v in x.items() if biz["actions"][ACTIONS[j]]["channel"] == c]
        if vs:
            m.Add(sum(vs) <= int(cap))
    m.Maximize(sum(int(evm[i, j]) * v for (i, j), v in x.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    t = time.time()
    status = solver.Solve(m)
    return {
        "status": solver.StatusName(status),
        "objective": solver.ObjectiveValue() / 100,
        "best_bound": solver.BestObjectiveBound() / 100,
        "seconds": round(time.time() - t, 1),
        "n_vars": len(x),
    }


def gap_report(scores: pd.DataFrame, biz=None, time_limit: float = 120.0, **kw) -> dict:
    t = time.time()
    greedy = summary(allocate(scores, biz, **kw))
    greedy_s = time.time() - t
    exact = solve(scores, biz, time_limit=time_limit, **kw)
    g = greedy["expected_value"]
    return {
        "n_subscribers": len(scores),
        "greedy_ev": g,
        "greedy_seconds": round(greedy_s, 2),
        "cpsat_ev": exact["objective"],
        "cpsat_bound": exact["best_bound"],
        "cpsat_status": exact["status"],
        "cpsat_seconds": exact["seconds"],
        "gap_vs_cpsat": (exact["objective"] - g) / exact["objective"]
        if exact["objective"]
        else 0.0,
        "gap_vs_bound": (exact["best_bound"] - g) / exact["best_bound"]
        if exact["best_bound"]
        else 0.0,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(config.path("processed") / "scores.parquet"))
    ap.add_argument("--time-limit", type=float, default=300.0)
    a = ap.parse_args()
    s = pd.read_parquet(a.scores)
    rep = {"budget_default": gap_report(s, time_limit=a.time_limit)}
    biz = config.load("business")
    rep["budget_tight"] = gap_report(s, time_limit=a.time_limit, budget=biz["budget"] / 4)
    out = config.path("reports") / "policy_gap.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
