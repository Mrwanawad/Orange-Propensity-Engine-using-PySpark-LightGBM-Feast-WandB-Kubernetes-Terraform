"""Gold-set evaluation of the Retention Analyst: tool choice, numeric grounding, refusals.

Needs the API running (scores in Redis). Writes reports/analyst_eval.json. When
LANGSMITH_API_KEY is set, the gold set is also pushed as a LangSmith dataset and every answer is
logged as an experiment with the three evaluators.

    python -m ope.analyst.evals.run_evals
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from ope import config
from ope.analyst.agent import ask, setup_tracing

GOLD = Path(__file__).with_name("gold.jsonl")


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def grade(case: dict, out: dict) -> dict:
    called = [c["name"] for c in out["tool_calls"]]
    expected = case["expected_tool"]
    return {
        "tool_choice": (expected in called) if expected else not called,
        "grounded": out["grounding"]["grounded"],
        "refused": not called if expected is None else None,
    }


async def run_all(gold: list[dict]) -> list[dict]:
    rows = []
    for case in gold:
        out = await ask(case["question"])
        rows.append({**case, **grade(case, out), "answer": out["answer"], "tool_calls": out["tool_calls"], "model": out["model"]})
    return rows


def push_langsmith(gold: list[dict], rows: list[dict]) -> str | None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        return None
    from langsmith import Client

    client = Client()
    name = "ope-retention-analyst-gold"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="20 gold questions (10 English, 10 Arabic)")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": g["question"]} for g in gold],
            outputs=[{"expected_tool": g["expected_tool"]} for g in gold],
        )
    by_q = {r["question"]: r for r in rows}

    def target(inputs: dict) -> dict:
        r = by_q[inputs["question"]]
        return {"answer": r["answer"], "tool_choice": r["tool_choice"], "grounded": r["grounded"]}

    def tool_choice(run, example) -> dict:
        return {"key": "tool_choice", "score": int(run.outputs["tool_choice"])}

    def numeric_grounding(run, example) -> dict:
        return {"key": "numeric_grounding", "score": int(run.outputs["grounded"])}

    res = client.evaluate(target, data=name, evaluators=[tool_choice, numeric_grounding], experiment_prefix="ope-analyst")
    return res.experiment_name


def main() -> None:
    setup_tracing()
    gold = load_gold()
    rows = asyncio.run(run_all(gold))
    refusals = [r for r in rows if r["expected_tool"] is None]
    summary = {
        "model": rows[0]["model"],
        "cases": len(rows),
        "tool_choice_accuracy": sum(r["tool_choice"] for r in rows) / len(rows),
        "numeric_grounding": sum(r["grounded"] for r in rows) / len(rows),
        "refusal_accuracy": sum(r["refused"] for r in refusals) / len(refusals),
        "langsmith_experiment": push_langsmith(gold, rows),
    }
    out = {"summary": summary, "cases": rows}
    (config.path("reports") / "analyst_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
