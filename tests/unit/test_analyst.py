import pytest

from ope.analyst.grounding import check, enforce

pytest.importorskip("agent_framework")
from ope.analyst.agent import _route  # noqa: E402
from ope.analyst.evals.run_evals import grade, load_gold  # noqa: E402

RESULT = {"budget": 60000.0, "contacts": 1834, "spend": 59988.0, "expected_value": 412345.67, "p_churn": 0.12345}


def test_grounded_numbers_pass():
    reply = "The plan contacts 1,834 subscribers, spends 59988 EGP and expects 412,345.67 EGP."
    assert check(reply, [RESULT])["grounded"]


def test_display_rounding_percent_and_thousands_pass():
    reply = "Churn probability is 12.3%, spend about 60K, value 0.4M, budget 60000."
    assert check(reply, [RESULT])["grounded"], check(reply, [RESULT])


def test_invented_number_is_flagged():
    reply, res = enforce("The plan saves 999,999 EGP.", [RESULT])
    assert not res["grounded"]
    assert "999,999" in res["untraced"]
    assert "[grounding check]" in reply


def test_numbers_from_the_question_are_allowed():
    q = "What if the budget is 2000000 EGP and SMS costs 0.3?"
    assert check("With 2,000,000 EGP and SMS at 0.3 ...", [], q)["grounded"]


def test_arabic_digits_are_checked():
    assert check("عدد التواصل ١٨٣٤", [RESULT])["grounded"]
    assert not check("عدد التواصل ٩٩٩٩", [RESULT])["grounded"]


@pytest.mark.parametrize(
    "q,tool",
    [
        ("What if the budget is 2M EGP and SMS costs 0.3?", "simulate_campaign"),
        ("Explain subscriber 6fa0bd4335e31582", "explain_subscriber"),
        ("Is there drift this week?", "get_drift_report"),
        ("What is the test AUC of the models?", "get_model_health"),
        ("ماذا لو كانت الميزانية 2 مليون جنيه؟", "simulate_campaign"),
        ("Write me a poem", None),
    ],
)
def test_scripted_routing(q, tool):
    r = _route(q)
    assert (r[0] if r else None) == tool


def test_scripted_budget_parsing():
    name, args = _route("What if the budget is 2M EGP and SMS costs 0.3?")
    assert args == {"budget": 2_000_000.0, "crosssell_cost": 0.3}


def test_gold_set_shape_and_grading():
    gold = load_gold()
    assert len(gold) == 20
    assert sum(g["lang"] == "ar" for g in gold) == 10
    ok = {"tool_calls": [{"name": "simulate_campaign"}], "grounding": {"grounded": True}}
    assert grade({"expected_tool": "simulate_campaign"}, ok)["tool_choice"]
    refused = {"tool_calls": [], "grounding": {"grounded": True}}
    assert grade({"expected_tool": None}, refused) == {"tool_choice": True, "grounded": True, "refused": True}
