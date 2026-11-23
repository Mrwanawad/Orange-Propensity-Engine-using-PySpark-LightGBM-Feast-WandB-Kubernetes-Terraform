"""Train / calibration / test split with a guard on the test set.

Stratified on the joint label pattern. In this data the three labels never co-occur,
so the pattern has four values: none, churn, appetency, upselling.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ope import config
from ope.data.contracts import ID, TARGETS

log = logging.getLogger(__name__)


class TestSetLocked(RuntimeError):
    pass


def label_pattern(df: pd.DataFrame) -> pd.Series:
    return df[TARGETS].astype(str).agg("".join, axis=1)


def make_splits(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    cfg = config.load()
    seed = cfg["seed"] if seed is None else seed
    s = cfg["split"]
    strat = label_pattern(df)
    idx = np.arange(len(df))
    rest, test = train_test_split(idx, test_size=s["test"], stratify=strat, random_state=seed)
    cal_share = s["calibration"] / (s["train"] + s["calibration"])
    train, cal = train_test_split(
        rest, test_size=cal_share, stratify=strat.iloc[rest], random_state=seed
    )
    split = np.empty(len(df), dtype=object)
    split[train], split[cal], split[test] = "train", "calibration", "test"
    fold = np.full(len(df), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=s["n_folds"], shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(train, strat.iloc[train])):
        fold[train[va]] = k
    return pd.DataFrame({ID: df[ID].to_numpy(), "split": split, "fold": fold})


def load_table() -> pd.DataFrame:
    return pd.read_parquet(config.path("interim") / "subscribers.parquet")


def splits_path():
    return config.path("processed") / "splits.parquet"


def load_split(name: str) -> pd.DataFrame:
    """Return rows of one split joined with their fold. 'test' is guarded."""
    if name == "test":
        raise TestSetLocked("use load_test() for the test split")
    return _load(name)


def load_test() -> pd.DataFrame:
    """The one door to the test split. Refuses unless final_evaluation is true in base.yaml."""
    if not config.load()["final_evaluation"]:
        raise TestSetLocked(
            "test split is locked: set final_evaluation: true in configs/base.yaml "
            "only after champions are frozen"
        )
    log.warning("TEST SPLIT OPENED")
    try:
        import wandb

        run = wandb.init(
            project=config.load()["wandb"]["project"],
            mode=config.load()["wandb"]["mode"],
            job_type="final-evaluation",
            name="test-split-opened",
        )
        run.log({"test_split_opened": 1})
        run.finish()
    except Exception as e:  # logging must never block the evaluation
        log.warning("could not log test access to W&B: %s", e)
    return _load("test")


def _load(name: str) -> pd.DataFrame:
    df = load_table().merge(pd.read_parquet(splits_path()), on=ID)
    return df[df["split"] == name].reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    df = load_table()
    sp = make_splits(df)
    sp.to_parquet(splits_path(), index=False)
    log.info("splits: %s", sp["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
