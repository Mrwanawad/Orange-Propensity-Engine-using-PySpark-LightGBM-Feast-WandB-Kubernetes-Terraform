"""Retention Analyst: a small Microsoft Agent Framework agent over the FastMCP tools.

Model: Anthropic claude-sonnet-5-5 when ANTHROPIC_API_KEY is set; otherwise a scripted model
(deterministic tool routing and template answers) so tests and evals run without keys.
Traces go to LangSmith over OpenTelemetry when LANGSMITH_API_KEY is set.

    python -m ope.analyst.agent "What if the budget is 2M EGP and SMS costs 0.3?"
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from typing import Any

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    MCPStdioTool,
    Message,
)

from ope.analyst.grounding import enforce

MODEL = "claude-sonnet-5-5"
PROMPT = (os.path.join(os.path.dirname(__file__), "prompts", "system.md"))
ARABIC = re.compile(r"[؀-ۿ]")


def instructions() -> str:
    with open(PROMPT, encoding="utf-8") as f:
        return f.read()


# ---------- scripted model (no API key) ----------

def _route(q: str) -> tuple[str, dict] | None:
    ql = q.lower()
    sid = re.search(r"\b[0-9a-f]{16}\b", ql)
    if sid:
        return "explain_subscriber", {"subscriber_id": sid.group(0)}
    if re.search(r"budget|simulat|campaign|ميزانية|حملة|محاكاة", ql):
        args: dict[str, Any] = {"budget": 60000.0}
        m = re.search(r"(\d+(?:\.\d+)?)\s*(m|million|k|مليون|ألف)?\s*(?:egp|جنيه)?", ql.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
        if m:
            mult = {"m": 1e6, "million": 1e6, "مليون": 1e6, "k": 1e3, "ألف": 1e3}.get(m.group(2) or "", 1)
            args["budget"] = float(m.group(1)) * mult
        sms = re.search(r"(?:sms|رسالة|الرسائل)[^\d]{0,25}(\d+(?:\.\d+)?)", ql)
        if sms:
            args["crosssell_cost"] = float(sms.group(1))
        return "simulate_campaign", args
    if re.search(r"drift|shift|انحراف|تغير|تحول", ql):
        return "get_drift_report", {}
    if re.search(r"health|auc|accura|calibrat|model|أداء|دقة|النموذج", ql):
        return "get_model_health", {}
    if re.search(r"segment|risk|band|group|شريحة|شرائح|مخاطر", ql):
        return "get_segment_summary", {"group_by": "risk_band"}
    return None


def _answer(q: str, tool: str, result: dict) -> str:
    ar = bool(ARABIC.search(q))
    if "error" in result:
        return "لم أجد هذا المشترك." if ar else "I could not find that subscriber."
    if tool == "simulate_campaign":
        acts = ", ".join(f"{a}: {v['n']} contacts" for a, v in result["per_action"].items())
        if ar:
            return f"بميزانية {result['budget']:.0f} جنيه: {result['contacts']} تواصل، إنفاق {result['spend']:.0f}، قيمة متوقعة {result['expected_value']:.0f} جنيه (افتراضات توضيحية). {acts}"
        return f"With a budget of {result['budget']:.0f} EGP the policy contacts {result['contacts']} subscribers, spends {result['spend']:.0f} EGP and expects {result['expected_value']:.0f} EGP of value (illustrative assumptions). {acts}."
    if tool == "explain_subscriber":
        s, a = result["scores"], result["next_best_action"]
        why = ", ".join(r["feature"] for r in a["reasons"][:3]) or "none recorded"
        if ar:
            return f"احتمال المغادرة {s['p_churn']:.4f}، الإجراء المقترح {a['action']} بقيمة متوقعة {a['expected_value']:.2f}. أهم الأسباب: {why}."
        return f"Churn probability {s['p_churn']:.4f}, appetency {s['p_appetency']:.4f}, up-sell {s['p_upselling']:.4f}. Next best action: {a['action']} (expected value {a['expected_value']:.2f} EGP). Top reasons: {why}."
    if tool == "get_model_health":
        parts = [f"{t}: test AUC {m['test_auc']:.4f}" for t, m in result.items() if m.get("test_auc")]
        return ("أداء النماذج: " if ar else "Champion models: ") + "; ".join(parts)
    if tool == "get_drift_report":
        d = result["decision"]
        est = "; ".join(f"{t} estimated AUC {c['estimated_auc']:.4f} vs reference {c['reference_auc']:.4f}" for t, c in result["cbpe"].items())
        head = ("طلب إعادة تدريب" if ar else "Retrain requested") if d["retrain"] else ("لا يوجد انحراف يستدعي إعادة التدريب" if ar else "No retrain trigger")
        return f"{head}. {est}."
    seg = "; ".join(f"{s['segment']}: {s['subscribers']} subscribers, mean churn {s['mean_p_churn']:.4f}" for s in result["segments"])
    return seg


class ScriptedChatClient(FunctionInvocationLayer, BaseChatClient):
    """Deterministic stand-in for the LLM: one tool call, then a template answer."""

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        await self._validate_options(options)
        question = next((m.text for m in messages if m.role == "user"), "")
        results = [c for m in messages for c in m.contents if c.type == "function_result"]
        if not results:
            route = _route(question)
            if route is None:
                text = "هذا خارج نطاقي. أستطيع الإجابة عن الشرائح والمشتركين والحملات وصحة النماذج والانحراف." if ARABIC.search(question) else (
                    "That is outside what I can answer. I can help with segments, subscribers, campaign simulations, model health and drift."
                )
                return ChatResponse(messages=[Message("assistant", [Content.from_text(text)])])
            name, args = route
            call = Content.from_function_call(call_id=uuid.uuid4().hex[:8], name=name, arguments=args)
            return ChatResponse(messages=[Message("assistant", [call])])
        calls = [c for m in messages for c in m.contents if c.type == "function_call"]
        result = _parse(results[-1].result)
        return ChatResponse(messages=[Message("assistant", [Content.from_text(_answer(question, calls[-1].name, result))])])


def _parse(r: Any) -> Any:
    if isinstance(r, list):
        r = r[0] if r else {}
    if hasattr(r, "text"):
        r = r.text
    if isinstance(r, str):
        try:
            return json.loads(r)
        except ValueError:
            return {"text": r}
    return r


# ---------- real agent ----------

def build_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from agent_framework_anthropic import AnthropicClient

        return AnthropicClient(model=MODEL), MODEL
    return ScriptedChatClient(), "scripted"


def setup_tracing() -> None:
    """LangSmith accepts OTLP traces at /otel with the API key and project as headers."""
    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        return
    from agent_framework.observability import configure_otel_providers

    configure_otel_providers(
        service_name="ope-retention-analyst",
        otlp_endpoint=os.environ.get("LANGSMITH_OTEL_ENDPOINT", "https://api.smith.langchain.com/otel"),
        otlp_protocol="http/protobuf",
        otlp_headers={"x-api-key": key, "Langsmith-Project": os.environ.get("LANGSMITH_PROJECT", "ope-retention-analyst")},
    )


async def ask(question: str) -> dict[str, Any]:
    client, model = build_client()
    tools = MCPStdioTool(
        name="ope",
        command=sys.executable,
        args=["-m", "ope.analyst.mcp_server"],
        env={k: v for k, v in os.environ.items() if k.startswith(("OPE_", "PATH", "SYSTEMROOT", "PYTHON"))},
    )
    async with tools:
        agent = Agent(client, instructions(), name="retention-analyst", tools=tools)
        resp = await agent.run(question)
    calls = [c for m in resp.messages for c in m.contents if c.type == "function_call"]
    results = [_parse(c.result) for m in resp.messages for c in m.contents if c.type == "function_result"]
    answer, grounding = enforce(resp.text, results, question)
    return {
        "question": question,
        "answer": answer,
        "model": model,
        "tool_calls": [{"name": c.name, "arguments": c.parse_arguments() if hasattr(c, "parse_arguments") else c.arguments} for c in calls],
        "grounding": grounding,
    }


if __name__ == "__main__":
    setup_tracing()
    out = asyncio.run(ask(" ".join(sys.argv[1:]) or "How are the models doing?"))
    print(json.dumps(out, indent=2, ensure_ascii=False))
