"""H2O AutoML challenger, on the same CV folds as our models so the AUCs compare directly.

Needs Java 17, so it runs in the jobs image:
    docker compose run --rm jobs python -m ope.models.h2o_automl
"""

from __future__ import annotations

import json
import logging

from ope import config
from ope.data.contracts import FEATURES, TARGETS
from ope.data.split import load_split
from ope.models.baseline import cv_dir

log = logging.getLogger(__name__)


def main() -> None:
    import h2o
    from h2o.automl import H2OAutoML

    logging.basicConfig(level=logging.INFO)
    cfg = config.load()
    h2o.init(nthreads=-1, max_mem_size="8G")
    train = load_split("train")
    out_path = cv_dir() / "h2o.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for target in TARGETS:
        if target in out:
            continue
        df = train[FEATURES + ["fold", target]].copy()
        df[target] = df[target].astype(str)
        hf = h2o.H2OFrame(df)
        hf[target] = hf[target].asfactor()
        aml = H2OAutoML(
            max_runtime_secs=cfg["h2o"]["max_runtime_secs"],
            sort_metric="AUC",
            seed=cfg["seed"],
            keep_cross_validation_predictions=True,
            project_name=f"ope_{target}",
        )
        aml.train(x=FEATURES, y=target, training_frame=hf, fold_column="fold")
        lb = aml.leaderboard.as_data_frame()
        leader = aml.leader
        # per-fold AUC from the leader's cross-validation summary (columns cv_1_valid ...)
        fold_auc = []
        cv_summary = leader.cross_validation_metrics_summary().as_data_frame()
        auc_row = cv_summary[cv_summary.iloc[:, 0] == "auc"]
        if len(auc_row):
            fold_auc = [float(v) for v in auc_row.filter(like="cv_").iloc[0].tolist()]
        out[target] = {
            "leader": leader.model_id,
            "cv_auc": float(lb.loc[0, "auc"]),
            "fold_auc": fold_auc,
            "n_models": int(len(lb)),
            "top5": lb.head(5)[["model_id", "auc", "aucpr", "logloss"]].to_dict(orient="records"),
            "max_runtime_secs": cfg["h2o"]["max_runtime_secs"],
        }
        log.info("h2o %s leader %s auc=%.4f", target, leader.model_id, out[target]["cv_auc"])
        out_path.write_text(json.dumps(out, indent=2))
        h2o.remove_all()
    h2o.cluster().shutdown()


if __name__ == "__main__":
    main()
