"""P3: encoding comparison, class weighting, Optuna tuning, feature selection, cross-target test.

    python -m ope.models.tune encodings
    python -m ope.models.tune tune [--family lightgbm] [--target churn] [--trials N]
    python -m ope.models.tune select
    python -m ope.models.tune crosstarget

Every stage writes JSON under reports/cv/ and logs to W&B. Nothing here touches the test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from ope import config
from ope.data.contracts import FEATURES, TARGETS
from ope.data.split import load_split
from ope.features.prep import Prep
from ope.models import registry
from ope.models.baseline import cv_dir
from ope.models.cv import run_cv
from ope.models.gbm import Model

log = logging.getLogger(__name__)
FAMILIES = ["lightgbm", "xgboost", "catboost"]
ENCODINGS = {"lightgbm": ["native", "te"], "xgboost": ["native", "te"], "catboost": ["raw", "te"]}
# moderate defaults for the comparison stage, before tuning
DEFAULTS = {
    "lightgbm": {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "colsample_bytree": 0.5,
    },
    "xgboost": {
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 10,
        "colsample_bytree": 0.5,
    },
    "catboost": {"learning_rate": 0.05, "depth": 6},
}


def _read(name: str) -> dict:
    p = cv_dir() / name
    return json.loads(p.read_text()) if p.exists() else {}


def _write(name: str, obj) -> None:
    (cv_dir() / name).write_text(json.dumps(obj, indent=2))


def _row(r) -> dict:
    return {
        "fold_auc": r.fold_auc,
        "mean": r.mean,
        "std": r.std,
        "se": r.se,
        "best_iterations": r.best_iterations,
    }


def encodings(train: pd.DataFrame) -> None:
    """Encoding comparison (unweighted), then weighted vs unweighted on the best encoding."""
    out = _read("encodings.json")
    for target in TARGETS:
        for fam in FAMILIES:
            key = f"{target}/{fam}"
            if key in out:
                continue
            rows = {}
            for enc in ENCODINGS[fam]:
                r = run_cv(train, target, fam, DEFAULTS[fam], cat_mode=enc, weighted=False)
                rows[f"{enc}/unweighted"] = _row(r)
                log.info("%s %s %s unweighted auc=%.4f", target, fam, enc, r.mean)
            best_enc = max(ENCODINGS[fam], key=lambda e: rows[f"{e}/unweighted"]["mean"])
            r = run_cv(train, target, fam, DEFAULTS[fam], cat_mode=best_enc, weighted=True)
            rows[f"{best_enc}/weighted"] = _row(r)
            log.info("%s %s %s weighted auc=%.4f", target, fam, best_enc, r.mean)
            best = max(rows, key=lambda k: rows[k]["mean"])
            out[key] = {"runs": rows, "best": best}
            with registry.run(
                f"enc-{target}-{fam}",
                "encoding-comparison",
                {"target": target, "family": fam},
                group="encodings",
            ) as run:
                run.summary.update({k.replace("/", "_"): v["mean"] for k, v in rows.items()})
            _write("encodings.json", out)


def _suggest(trial: optuna.Trial, family: str) -> dict:
    params = {}
    for name, (lo, hi, kind) in config.load()["search_space"][family].items():
        if kind == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi), log=kind == "log")
    return params


def _objective(train: pd.DataFrame, target: str, fam: str, cat_mode: str, weighted: bool):
    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, fam)
        scores: list[float] = []

        def on_fold(k, a):
            scores.append(a)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        r = run_cv(
            train, target, fam, params, cat_mode=cat_mode, weighted=weighted, on_fold=on_fold
        )
        trial.set_user_attr("fold_auc", r.fold_auc)
        trial.set_user_attr("best_iterations", r.best_iterations)
        return r.mean

    return objective


def tune(train: pd.DataFrame, targets: list[str], families: list[str], trials: int | None) -> None:
    enc = _read("encodings.json")
    out = _read("tuned.json")
    cfg = config.load()
    for target in targets:
        for fam in families:
            if f"{target}/{fam}" in out:
                continue
            cat_mode, weighting = enc[f"{target}/{fam}"]["best"].split("/")
            weighted = weighting == "weighted"
            n = trials or cfg["tuning"]["n_trials"][fam]
            start = DEFAULTS[fam] | ({} if fam != "lightgbm" else {"subsample_freq": 1})

            objective = _objective(train, target, fam, cat_mode, weighted)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                study_name=f"{target}-{fam}",
            )
            study.enqueue_trial({k: v for k, v in start.items() if k in cfg["search_space"][fam]})
            with registry.run(
                f"tune-{target}-{fam}",
                "tuning",
                {
                    "target": target,
                    "family": fam,
                    "cat_mode": cat_mode,
                    "weighted": weighted,
                    "n_trials": n,
                },
                group="tuning",
            ) as run:

                def cb(st, tr):
                    if tr.value is not None:
                        run.log(
                            {"trial": tr.number, "cv_auc": tr.value, "best_cv_auc": st.best_value}
                        )

                study.optimize(
                    objective, n_trials=n, timeout=cfg["tuning"]["timeout_sec"][fam], callbacks=[cb]
                )
                best = study.best_trial
                fa = best.user_attrs["fold_auc"]
                row = {
                    "family": fam,
                    "cat_mode": cat_mode,
                    "weighted": weighted,
                    "params": best.params,
                    "fold_auc": fa,
                    "mean": float(np.mean(fa)),
                    "std": float(np.std(fa, ddof=1)),
                    "se": float(np.std(fa, ddof=1) / np.sqrt(len(fa))),
                    "best_iterations": best.user_attrs["best_iterations"],
                    "n_trials": len(study.trials),
                    "n_pruned": sum(
                        t.state == optuna.trial.TrialState.PRUNED for t in study.trials
                    ),
                }
                run.summary.update({"best_cv_auc": row["mean"], "best_cv_std": row["std"]})
            # re-read before writing: one process per target may run in parallel
            out = _read("tuned.json") | {f"{target}/{fam}": row}
            _write("tuned.json", out)
            log.info(
                "tuned %s %s auc=%.4f±%.4f (%d trials)",
                target,
                fam,
                row["mean"],
                row["std"],
                row["n_trials"],
            )


def best_tuned(target: str) -> dict:
    tuned = _read("tuned.json")
    rows = [v for k, v in tuned.items() if k.startswith(f"{target}/")]
    return max(rows, key=lambda r: r["mean"])


def select(train: pd.DataFrame) -> None:
    """Permutation importance per fold with the tuned LightGBM. Keep a feature only if its
    importance is above zero in at least 4 of 5 folds; adopt only if CV AUC does not fall."""
    tuned = _read("tuned.json")
    out = _read("selection.json")
    for target in TARGETS:
        if target in out:
            continue
        spec = tuned[f"{target}/lightgbm"]
        y = train[target].to_numpy()
        positive = {}
        for k in sorted(train["fold"].unique()):
            tr, va = train["fold"] != k, train["fold"] == k
            prep = Prep(cat_mode=spec["cat_mode"])
            Xtr = prep.fit_transform(train.loc[tr, FEATURES], y[tr])
            Xva = prep.transform(train.loc[va, FEATURES])
            m = Model("lightgbm", dict(spec["params"]), weighted=spec["weighted"]).fit(
                Xtr, y[tr], prep.cat_features
            )
            pi = permutation_importance(
                m.model, Xva, y[va], scoring="roc_auc", n_repeats=3, random_state=k, n_jobs=1
            )
            for col, imp in zip(Xva.columns, pi.importances_mean, strict=True):
                positive[col] = positive.get(col, 0) + int(imp > 0)
        keep = [c for c, n in positive.items() if n >= 4]
        r = run_cv(
            train,
            target,
            "lightgbm",
            spec["params"],
            cat_mode=spec["cat_mode"],
            weighted=spec["weighted"],
            columns=keep,
        )
        adopted = r.mean >= spec["mean"]
        out[target] = {
            "n_features_before": len(positive),
            "n_features_after": len(keep),
            "auc_before": spec["mean"],
            "auc_after": r.mean,
            "fold_auc_after": r.fold_auc,
            "adopted": adopted,
            "kept": keep,
        }
        log.info(
            "%s selection %d -> %d features, auc %.4f -> %.4f, adopted=%s",
            target,
            len(positive),
            len(keep),
            spec["mean"],
            r.mean,
            adopted,
        )
        with registry.run(
            f"select-{target}", "feature-selection", {"target": target}, group="selection"
        ) as run:
            run.summary.update(
                {k: v for k, v in out[target].items() if k != "kept" and k != "fold_auc_after"}
            )
        _write("selection.json", out)


def _fit_predict(Xtr_raw, ytr, Xte_raw, spec) -> np.ndarray:
    prep = Prep(cat_mode=spec["cat_mode"])
    Xtr = prep.fit_transform(Xtr_raw, ytr)
    m = Model("lightgbm", dict(spec["params"]), weighted=spec["weighted"]).fit(
        Xtr, ytr, prep.cat_features
    )
    return m.predict(prep.transform(Xte_raw))


def crosstarget(train: pd.DataFrame) -> None:
    """Does the score of the other targets help? Nested so nothing leaks: for outer fold k,
    other-target scores for the training rows are inner out-of-fold predictions made only
    from the training rows, and scores for the validation rows come from a model fitted on
    all training rows of that fold."""
    tuned = _read("tuned.json")
    out = _read("crosstarget.json")
    y_all = {t: train[t].to_numpy() for t in TARGETS}
    folds = sorted(train["fold"].unique())
    for target in TARGETS:
        if target in out:
            continue
        spec = tuned[f"{target}/lightgbm"]
        others = [t for t in TARGETS if t != target]
        oof = np.full(len(train), np.nan)
        aucs = []
        for k in folds:
            tr_idx = np.where(train["fold"] != k)[0]
            va_idx = np.where(train["fold"] == k)[0]
            extra_tr = pd.DataFrame(index=tr_idx, columns=[f"p_{o}" for o in others], dtype=float)
            extra_va = pd.DataFrame(index=va_idx, columns=[f"p_{o}" for o in others], dtype=float)
            inner_fold = train["fold"].to_numpy()[tr_idx]
            for o in others:
                ospec = tuned[f"{o}/lightgbm"]
                for j in [f for f in folds if f != k]:
                    a, b = tr_idx[inner_fold != j], tr_idx[inner_fold == j]
                    extra_tr.loc[b, f"p_{o}"] = _fit_predict(
                        train.loc[a, FEATURES], y_all[o][a], train.loc[b, FEATURES], ospec
                    )
                extra_va[f"p_{o}"] = _fit_predict(
                    train.loc[tr_idx, FEATURES],
                    y_all[o][tr_idx],
                    train.loc[va_idx, FEATURES],
                    ospec,
                )
            prep = Prep(cat_mode=spec["cat_mode"])
            y = y_all[target]
            Xtr = pd.concat(
                [prep.fit_transform(train.loc[tr_idx, FEATURES], y[tr_idx]), extra_tr], axis=1
            )
            Xva = pd.concat([prep.transform(train.loc[va_idx, FEATURES]), extra_va], axis=1)
            m = Model("lightgbm", dict(spec["params"]), weighted=spec["weighted"]).fit(
                Xtr, y[tr_idx], prep.cat_features
            )
            oof[va_idx] = m.predict(Xva)
            aucs.append(float(roc_auc_score(y[va_idx], oof[va_idx])))
        mean = float(np.mean(aucs))
        out[target] = {
            "features_added": [f"p_{o}" for o in others],
            "auc_without": spec["mean"],
            "auc_with": mean,
            "fold_auc_with": aucs,
            "se_without": spec["se"],
            "helps": mean - spec["mean"] > spec["se"],
        }
        log.info("%s cross-target auc %.4f -> %.4f", target, spec["mean"], mean)
        with registry.run(
            f"crosstarget-{target}", "cross-target", {"target": target}, group="crosstarget"
        ) as run:
            run.summary.update({"auc_without": spec["mean"], "auc_with": mean})
        _write("crosstarget.json", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("optuna").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encodings", "tune", "select", "crosstarget"])
    ap.add_argument("--family", action="append")
    ap.add_argument("--target", action="append")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--fresh", action="store_true", help="re-tune even if results exist")
    a = ap.parse_args()
    train = load_split("train")
    if a.stage == "encodings":
        encodings(train)
    elif a.stage == "tune":
        if a.fresh:
            tuned = _read("tuned.json")
            for t in a.target or TARGETS:
                for f in a.family or FAMILIES:
                    tuned.pop(f"{t}/{f}", None)
            _write("tuned.json", tuned)
        tune(train, a.target or TARGETS, a.family or FAMILIES, a.trials)
    elif a.stage == "select":
        select(train)
    else:
        crosstarget(train)


if __name__ == "__main__":
    main()
