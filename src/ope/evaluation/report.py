"""P5: the one-time test evaluation. Writes reports/evaluation.md and reports/evaluation.json.

Refuses to run unless configs/base.yaml has final_evaluation: true (see data/split.load_test).
    python -m ope.evaluation.report
"""

from __future__ import annotations

import json
import logging
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ope import config
from ope.data.contracts import ID, TARGETS
from ope.data.split import load_test
from ope.evaluation import metrics as M
from ope.evaluation.bootstrap import bootstrap_ci
from ope.explain.shap_reasons import global_importance
from ope.models import registry
from ope.models.baseline import cv_dir
from ope.models.predict import raw_score

log = logging.getLogger(__name__)
# house style for every chart: Times New Roman Bold, plain colours
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})
NAMES = {"churn": "Churn", "appetency": "Appetency", "upselling": "Up-selling"}


def _read(name):
    p = cv_dir() / name
    return json.loads(p.read_text()) if p.exists() else {}


def reliability_plot(target: str, y, p_raw, p_cal, path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=130)
    for p, label, style in ((p_raw, "raw model score", "--"), (p_cal, "calibrated", "-")):
        pred, obs, _ = M.reliability(y, p, n_bins=10)
        ax.plot(pred, obs, style, marker="o", ms=3.5, lw=1.4, label=label)
    hi = max(float(np.quantile(p_raw, 0.995)), float(np.quantile(p_cal, 0.995)), float(np.mean(y)) * 3)
    ax.plot([0, hi], [0, hi], color="#888", lw=0.8)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("mean predicted probability (decile)")
    ax.set_ylabel("observed rate")
    ax.set_title(f"{NAMES[target]}: reliability on test")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def comparison_table(target: str, test_auc: float) -> list[dict]:
    rows = []
    b = _read("baseline.json").get(target)
    if b:
        rows.append({"model": "Logistic regression (baseline)", "cv_auc": b["mean"], "cv_std": b["std"]})
    for fam in ("lightgbm", "xgboost", "catboost"):
        t = _read("tuned.json").get(f"{target}/{fam}")
        if t:
            rows.append({
                "model": f"{fam} (tuned, {t['cat_mode']}, {'weighted' if t['weighted'] else 'unweighted'})",
                "cv_auc": t["mean"],
                "cv_std": t["std"],
            })
    h = _read("h2o.json").get(target)
    if h:
        rows.append({"model": f"H2O AutoML ({h['leader'].split('_AutoML')[0]})", "cv_auc": h["cv_auc"], "cv_std": float(np.std(h["fold_auc"], ddof=1)) if h.get("fold_auc") else None})
    f = _read("freeze.json").get(target)
    if f:
        s = f["candidates"]["stack"]
        rows.append({"model": "Stack (logistic over the 3 GBMs)", "cv_auc": s["mean"], "cv_std": float(np.std(s["fold_auc"], ddof=1))})
        rows.append({"model": f"**Champion: {f['candidate']}** (test)", "cv_auc": f["cv_auc"], "cv_std": None, "test_auc": test_auc})
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    test = load_test()  # guarded
    img = config.ROOT / "docs" / "images"
    img.mkdir(parents=True, exist_ok=True)
    ref = json.loads((config.path("reports") / "kdd2009_reference.json").read_text())
    out, scores = {}, pd.DataFrame({ID: test[ID]})
    for target in TARGETS:
        bundle = registry.load_bundle(target, "champion")
        y = test[target].to_numpy()
        p_raw = raw_score(bundle, test)
        p_cal = np.clip(bundle["calibrator"](p_raw), 0, 1)
        scores[f"p_{target}"] = p_cal
        auc_pt, auc_lo, auc_hi = bootstrap_ci(y, p_cal, M.auc, n=1000, seed=config.load()["seed"])
        pr_pt, pr_lo, pr_hi = bootstrap_ci(y, p_cal, M.pr_auc, n=1000, seed=config.load()["seed"])
        s = M.summary(y, p_cal)
        reliability_plot(target, y, p_raw, p_cal, img / f"reliability_{target}.png")
        imp = global_importance(bundle, test.sample(min(3000, len(test)), random_state=0))
        out[target] = {
            "version": bundle["version"],
            "candidate": bundle["metadata"]["candidate"],
            "n_test": int(len(y)),
            "positives": int(y.sum()),
            "auc": auc_pt,
            "auc_ci95": [auc_lo, auc_hi],
            "pr_auc": pr_pt,
            "pr_auc_ci95": [pr_lo, pr_hi],
            "base_rate": float(y.mean()),
            "brier": s["brier"],
            "ece_raw": M.ece(y, p_raw),
            "ece_calibrated": s["ece"],
            "brier_raw": M.brier(y, p_raw),
            "precision_top5": s["precision_top5"],
            "lift": {k: v for k, v in s.items() if k.startswith(("lift", "capture"))},
            "calibration_method": bundle["calibrator"].method,
            "cv_auc": bundle["metadata"]["cv_auc"],
            "shap_top": imp.to_dict(orient="records"),
            "comparison": comparison_table(target, auc_pt),
            "kdd2009": ref["test_auc"][target],
        }
        log.info("%s test auc %.4f [%.4f, %.4f]", target, auc_pt, auc_lo, auc_hi)
    scores.to_parquet(config.path("processed") / "test_scores.parquet", index=False)
    (config.path("reports") / "evaluation.json").write_text(json.dumps(out, indent=2))
    (config.path("reports") / "evaluation.md").write_text(render(out, ref))
    with registry.run("final-evaluation", "final-evaluation", {}) as run:
        run.summary.update({f"test_auc_{t}": r["auc"] for t, r in out.items()})


def _f(x, d=4):
    return "" if x is None else f"{x:.{d}f}"


def render(out: dict, ref: dict) -> str:
    L = [
        "# Evaluation report",
        "",
        "Generated by `python -m ope.evaluation.report` from the frozen champions on the held-out",
        "test split (20%, stratified, opened once). Do not edit by hand.",
        "",
        "## Headline (test split)",
        "",
        "| Target | Model | Test AUC (95% CI) | PR-AUC (95% CI) | Base rate | Precision @ top 5% | Lift d1 | Capture d1-3 | Brier | ECE raw | ECE calibrated |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, r in out.items():
        L.append(
            f"| {NAMES[t]} | {r['candidate']} | {_f(r['auc'])} ({_f(r['auc_ci95'][0])}-{_f(r['auc_ci95'][1])}) "
            f"| {_f(r['pr_auc'])} ({_f(r['pr_auc_ci95'][0])}-{_f(r['pr_auc_ci95'][1])}) | {r['base_rate']:.2%} "
            f"| {r['precision_top5']:.2%} | {r['lift']['lift_d1']:.2f} | {r['lift']['capture_d3']:.1%} "
            f"| {_f(r['brier'])} | {_f(r['ece_raw'])} | {_f(r['ece_calibrated'])} |"
        )
    avg = np.mean([r["auc"] for r in out.values()])
    L += [
        "",
        f"Average test AUC over the three targets: **{avg:.4f}**.",
        "",
        "Bootstrap: 1,000 resamples of the test rows, percentile intervals.",
        "",
        "## External reference: KDD Cup 2009 (approximate only)",
        "",
        f"Source: {ref['citation']} {ref['source']}.",
        "",
        "| Target | This project, test AUC | KDD 2009 best overall | Orange in-house (selective NB) | Basic Naive Bayes |",
        "|---|---|---|---|---|",
    ]
    for t, r in out.items():
        k = ref["test_auc"][t]
        L.append(f"| {NAMES[t]} | {_f(r['auc'])} | {k['best_overall']} | {k['orange_in_house_snb']} | {k['basic_naive_bayes']} |")
    k = ref["test_auc"]["average"]
    L.append(f"| Average | {avg:.4f} | {k['best_overall']} | {k['orange_in_house_snb']} | {k['basic_naive_bayes']} |")
    L += ["", "Caveats:"] + [f"- {c}" for c in ref["caveats"]]
    for t, r in out.items():
        L += [
            "",
            f"## {NAMES[t]}",
            "",
            f"Champion `{r['version']}` ({r['candidate']}), calibration: {r['calibration_method']}. "
            f"CV AUC on the training split {r['cv_auc']:.4f}; test AUC {r['auc']:.4f}.",
            "",
            "Model comparison (5-fold CV on the training split; the champion row adds the test AUC):",
            "",
            "| Model | CV AUC | CV std | Test AUC |",
            "|---|---|---|---|",
        ]
        for c in r["comparison"]:
            L.append(f"| {c['model']} | {_f(c['cv_auc'])} | {_f(c.get('cv_std'))} | {_f(c.get('test_auc'))} |")
        L += [
            "",
            "Lift and capture (cumulative, by score decile):",
            "",
            "| Decile | Lift | Capture |",
            "|---|---|---|",
        ]
        for d in (1, 2, 3):
            L.append(f"| top {d * 10}% | {r['lift'][f'lift_d{d}']:.2f} | {r['lift'][f'capture_d{d}']:.1%} |")
        L += [
            "",
            f"![reliability](../docs/images/reliability_{t}.png)",
            "",
            "Top 10 features by mean |SHAP| (anonymised Orange variables):",
            "",
            "| Feature | mean abs SHAP (log-odds) |",
            "|---|---|",
        ]
        for s in r["shap_top"][:10]:
            L.append(f"| {s['feature']} | {s['mean_abs_shap']:.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
