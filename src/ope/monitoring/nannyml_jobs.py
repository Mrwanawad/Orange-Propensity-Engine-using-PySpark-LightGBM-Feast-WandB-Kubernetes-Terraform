"""Drift and estimated performance before labels arrive (section 11).

Reference = calibration split: features, calibrated champion scores and labels.
Analysis  = a scoring snapshot: features and scores, no labels (churn labels arrive 30-60 days late).

  univariate drift   Kolmogorov-Smirnov (numeric), Jensen-Shannon (categorical) on the top features
  multivariate drift PCA reconstruction error
  CBPE               estimated ROC AUC per target
  PSI                our own psi.py on the top 20 features and the three scores
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ope import config
from ope.data.contracts import CATEGORICAL, FEATURES, ID, TARGETS
from ope.data.split import load_split
from ope.models import registry
from ope.models.predict import score
from ope.monitoring.psi import psi_frame

log = logging.getLogger(__name__)
CHUNK = 500


def mon_dir():
    p = config.path("reports") / "monitoring"
    p.mkdir(exist_ok=True)
    return p


def top_features(n: int = 20) -> list[str]:
    ev = config.path("reports") / "evaluation.json"
    if ev.exists():
        rows = json.loads(ev.read_text())
        ranked: dict[str, float] = {}
        for r in rows.values():
            for s in r["shap_top"]:
                ranked[s["feature"]] = ranked.get(s["feature"], 0) + s["mean_abs_shap"]
        return [f for f, _ in sorted(ranked.items(), key=lambda kv: -kv[1])][:n]
    from ope.explain.shap_reasons import global_importance

    ref = load_split("calibration")
    return global_importance(registry.load_bundle("churn"), ref.sample(2000, random_state=0), top=n)["feature"].tolist()


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[ID] + FEATURES].copy()
    for t in TARGETS:
        out[f"p_{t}"] = score(registry.load_bundle(t), df[FEATURES])
    return out


def reference() -> pd.DataFrame:
    p = mon_dir() / "reference.parquet"
    reg = registry.load_registry()
    versions = {t: reg[t]["champion"]["version"] for t in TARGETS}
    if p.exists():
        ref = pd.read_parquet(p)
        if all(ref[f"model_version_{t}"].iloc[0] == v for t, v in versions.items()):
            return ref
    cal = load_split("calibration")
    ref = score_frame(cal)
    for t in TARGETS:
        ref[t] = cal[t].to_numpy()
        ref[f"model_version_{t}"] = versions[t]
    ref.to_parquet(p, index=False)
    return ref


def _cbpe(ref: pd.DataFrame, cur: pd.DataFrame, target: str) -> dict:
    import nannyml as nml

    thr = float(np.quantile(ref[f"p_{target}"], 0.95))
    r = ref.assign(y_pred=(ref[f"p_{target}"] >= thr).astype(int))
    c = cur.assign(y_pred=(cur[f"p_{target}"] >= thr).astype(int))
    est = nml.CBPE(
        metrics=["roc_auc"],
        y_pred_proba=f"p_{target}",
        y_pred="y_pred",
        y_true=target,
        problem_type="classification_binary",
        chunk_size=CHUNK,
    )
    est.fit(r)
    res = est.estimate(c).filter(period="analysis").to_df()
    values = res[("roc_auc", "value")].astype(float)
    alerts = res[("roc_auc", "alert")].astype(bool)
    from sklearn.metrics import roc_auc_score

    return {
        "reference_auc": float(roc_auc_score(ref[target], ref[f"p_{target}"])),
        "estimated_auc": float(values.mean()),
        "estimated_auc_chunks": [round(float(v), 4) for v in values],
        "alert_chunks": int(alerts.sum()),
    }


def run(current: pd.DataFrame, label: str, trained_at: str | None = None) -> dict:
    """current: raw rows (subscriber_id + features). Returns and saves the monitoring report."""
    import nannyml as nml

    ref = reference()
    cur = score_frame(current)
    feats = top_features(20)
    cats = [f for f in feats if f in CATEGORICAL]
    cols = feats + [f"p_{t}" for t in TARGETS]

    def prep(df):
        out = df[cols].copy()
        for c in cats:
            out[c] = out[c].astype("object").fillna("__missing__")
        return out

    uni = nml.UnivariateDriftCalculator(
        column_names=cols,
        treat_as_categorical=cats,
        continuous_methods=["kolmogorov_smirnov"],
        categorical_methods=["jensen_shannon"],
        chunk_size=CHUNK,
    ).fit(prep(ref))
    u = uni.calculate(prep(cur)).filter(period="analysis").to_df()
    uni_alerts = {}
    for c in cols:
        alert_cols = [k for k in u.columns if k[0] == c and k[-1] == "alert"]
        uni_alerts[c] = int(u[alert_cols].astype(bool).any(axis=1).sum()) if alert_cols else 0

    num_feats = [f for f in feats if f not in cats]
    mv = nml.DataReconstructionDriftCalculator(column_names=num_feats + cats, chunk_size=CHUNK).fit(prep(ref)[num_feats + cats])
    m = mv.calculate(prep(cur)[num_feats + cats]).filter(period="analysis").to_df()
    mv_err = m[("reconstruction_error", "value")].astype(float)
    mv_alerts = int(m[("reconstruction_error", "alert")].astype(bool).sum())

    psi = psi_frame(ref, cur, cols)
    cbpe = {t: _cbpe(ref, cur, t) for t in TARGETS}

    report = {
        "label": label,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": int(len(cur)),
        "chunk_size": CHUNK,
        "top_features": feats,
        "univariate_alert_chunks": {k: v for k, v in uni_alerts.items() if v},
        "multivariate": {"mean_reconstruction_error": float(mv_err.mean()), "alert_chunks": mv_alerts},
        "psi": {k: round(float(v), 4) for k, v in psi.items()},
        "cbpe": cbpe,
        "trained_at": trained_at,
    }
    from ope.monitoring.alerts import decide

    report["decision"] = decide(report)
    (mon_dir() / f"{label}.json").write_text(json.dumps(report, indent=2))
    log.info("monitoring %s: %s", label, report["decision"])
    return report


if __name__ == "__main__":
    import argparse

    from ope.data.split import load_table
    from ope.monitoring.alerts import emit

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    trained = registry.load_registry()["churn"]["champion"]["set_at"]
    emit(run(load_table(), a.label, trained_at=trained))
