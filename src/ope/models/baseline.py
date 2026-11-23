"""Logistic regression baseline (L2). Target + frequency encoded categoricals, median impute, scaling."""

from __future__ import annotations

import json
import logging
import warnings

from ope import config
from ope.data.contracts import TARGETS
from ope.data.split import load_split
from ope.models import registry
from ope.models.cv import run_cv

log = logging.getLogger(__name__)


def cv_dir():
    p = config.path("reports") / "cv"
    p.mkdir(exist_ok=True)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    train = load_split("train")
    results = {}
    for target in TARGETS:
        best = None
        for C in (0.01, 0.1, 1.0):
            for weighted in (True, False):
                r = run_cv(train, target, "logreg", {"C": C}, weighted=weighted)
                row = {
                    "C": C,
                    "weighted": weighted,
                    "fold_auc": r.fold_auc,
                    "mean": r.mean,
                    "std": r.std,
                    "se": r.se,
                }
                log.info(
                    "%s logreg C=%s weighted=%s auc=%.4f±%.4f", target, C, weighted, r.mean, r.std
                )
                with registry.run(
                    f"baseline-{target}-C{C}-w{int(weighted)}",
                    "cv",
                    {"target": target, "family": "logreg", **row},
                    group="baseline",
                ) as run:
                    run.summary.update({"cv_auc": r.mean, "cv_auc_std": r.std})
                if best is None or r.mean > best["mean"]:
                    best = row
        results[target] = best
    (cv_dir() / "baseline.json").write_text(json.dumps(results, indent=2))
    log.info("baseline: %s", {t: round(r["mean"], 4) for t, r in results.items()})


if __name__ == "__main__":
    main()
