from datetime import UTC, datetime

from ope.monitoring.alerts import decide


def report(est=0.75, ref=0.75, psi=0.05, trained="2026-11-01T00:00:00+00:00"):
    return {
        "cbpe": {"churn": {"estimated_auc": est, "reference_auc": ref}},
        "top_features": ["Var126"],
        "psi": {"Var126": psi},
        "trained_at": trained,
    }


NOW = datetime(2026, 11, 12, tzinfo=UTC)


def test_quiet_when_stable():
    assert decide(report(), NOW) == {"retrain": False, "reasons": []}


def test_auc_drop_triggers():
    assert decide(report(est=0.72), NOW)["retrain"]


def test_psi_triggers():
    assert decide(report(psi=0.25), NOW)["retrain"]


def test_age_triggers():
    assert decide(report(trained="2026-09-01T00:00:00+00:00"), NOW)["retrain"]
