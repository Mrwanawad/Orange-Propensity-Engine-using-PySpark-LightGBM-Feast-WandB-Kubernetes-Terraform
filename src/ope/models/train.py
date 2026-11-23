"""P4: pick the candidate per target, test stacking, fit on the full training split, calibrate,
run the gate, register as `challenger`. Promotion to `champion` is a separate command.

    python -m ope.models.train freeze
    python -m ope.models.train promote
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ope import config
from ope.data.contracts import FEATURES, TARGETS
from ope.data.split import load_split
from ope.evaluation.metrics import auc
from ope.features.prep import Prep
from ope.models import gate, registry
from ope.models.baseline import cv_dir
from ope.models.calibrate import choose
from ope.models.cv import run_cv
from ope.models.gbm import Model
from ope.models.predict import _logit, raw_score

log = logging.getLogger(__name__)
FAMILIES = ["lightgbm", "xgboost", "catboost"]


def _read(name):
    p = cv_dir() / name
    return json.loads(p.read_text()) if p.exists() else {}


def _member_spec(target: str, fam: str) -> dict:
    spec = dict(_read("tuned.json")[f"{target}/{fam}"])
    sel = _read("selection.json").get(target, {})
    spec["columns"] = sel["kept"] if fam == "lightgbm" and sel.get("adopted") else None
    return spec


def _stack_cv(train: pd.DataFrame, target: str, oofs: dict[str, np.ndarray]) -> list[float]:
    """Logistic meta-model over out-of-fold logits, evaluated on the same folds."""
    X = _logit(np.column_stack([oofs[f] for f in FAMILIES]))
    y = train[target].to_numpy()
    aucs = []
    for k in sorted(train["fold"].unique()):
        tr, va = (train["fold"] != k).to_numpy(), (train["fold"] == k).to_numpy()
        meta = LogisticRegression(C=1.0).fit(X[tr], y[tr])
        aucs.append(auc(y[va], meta.predict_proba(X[va])[:, 1]))
    return aucs


def _fit_member(train: pd.DataFrame, target: str, spec: dict) -> dict:
    y = train[target].to_numpy()
    prep = Prep(cat_mode=spec["cat_mode"])
    X = prep.fit_transform(train[FEATURES], y)
    if spec.get("columns"):
        X = X[spec["columns"]]
    cats = [c for c in prep.cat_features if c in X.columns]
    m = Model(spec["family"], dict(spec["params"]), weighted=spec["weighted"]).fit(X, y, cats)
    return {"family": spec["family"], "prep": prep, "model": m, "columns": spec.get("columns")}


def freeze() -> None:
    train, cal = load_split("train"), load_split("calibration")
    baseline = _read("baseline.json")
    reg = registry.load_registry()
    report = {}
    for target in TARGETS:
        cands, oofs = {}, {}
        for fam in FAMILIES:
            spec = _member_spec(target, fam)
            r = run_cv(
                train,
                target,
                fam,
                spec["params"],
                cat_mode=spec["cat_mode"],
                weighted=spec["weighted"],
                columns=spec["columns"],
            )
            cands[fam] = {"fold_auc": r.fold_auc, "mean": r.mean, "se": r.se}
            oofs[fam] = r.oof
            log.info("%s %s cv auc %.4f", target, fam, r.mean)
        stack_auc = _stack_cv(train, target, oofs)
        best_single = max(FAMILIES, key=lambda f: cands[f]["mean"])
        gain = float(np.mean(stack_auc)) - cands[best_single]["mean"]
        use_stack = gain > cands[best_single]["se"]
        cands["stack"] = {
            "fold_auc": stack_auc,
            "mean": float(np.mean(stack_auc)),
            "gain_over_best": gain,
            "kept": use_stack,
        }
        log.info("%s stack auc %.4f (gain %.4f, kept=%s)", target, np.mean(stack_auc), gain, use_stack)

        fams = FAMILIES if use_stack else [best_single]
        members = [_fit_member(train, target, _member_spec(target, f)) for f in fams]
        meta = None
        if use_stack:
            X = _logit(np.column_stack([oofs[f] for f in FAMILIES]))
            meta = LogisticRegression(C=1.0).fit(X, train[target].to_numpy())
        bundle = {"target": target, "members": members, "meta": meta}
        p_cal = raw_score(bundle, cal)
        y_cal = cal[target].to_numpy()
        calibrator, cal_report = choose(p_cal, y_cal)
        bundle["calibrator"] = calibrator
        chosen = "stack" if use_stack else best_single
        fold_auc = cands[chosen]["fold_auc"]
        g = gate.check(
            fold_auc,
            baseline_auc=baseline[target]["mean"],
            calibration_ece=cal_report[cal_report["chosen"]]["ece"],
            champion=reg.get(target, {}).get("champion"),
        )
        meta_info = {
            "target": target,
            "candidate": chosen,
            "cv_auc": g["cv_auc"],
            "cv_se": g["cv_se"],
            "calibration_auc_raw": auc(y_cal, p_cal),
            "calibration": cal_report,
            "gate": g,
            "snapshot": _snapshot_info(),
        }
        bundle["metadata"] = meta_info
        path = registry.save_bundle(target, bundle)
        if g["passed"]:
            registry.set_alias(target, "challenger", path, {k: meta_info[k] for k in ("candidate", "cv_auc", "cv_se")})
        report[target] = {"candidates": cands, **meta_info, "bundle": path.name}
        log.info("%s %s gate=%s %s", target, chosen, g["passed"], g["checks"])
        with registry.run(f"freeze-{target}", "freeze", {"target": target, "candidate": chosen}, group="freeze") as run:
            run.summary.update({"cv_auc": g["cv_auc"], "gate_passed": g["passed"], "cal_ece": g["calibration_ece"]})
    (cv_dir() / "freeze.json").write_text(json.dumps(report, indent=2, default=str))


def _snapshot_info() -> dict:
    p = config.path("processed") / "silver_snapshot.json"
    return json.loads(p.read_text()) if p.exists() else {"source": "data/interim/subscribers.parquet"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "promote"])
    a = ap.parse_args()
    if a.stage == "freeze":
        freeze()
    else:
        for t in TARGETS:
            registry.promote(t)


if __name__ == "__main__":
    main()
