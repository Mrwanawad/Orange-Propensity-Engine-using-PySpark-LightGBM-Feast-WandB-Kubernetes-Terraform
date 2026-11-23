"""Cross-validation on the training split. Prep is fitted inside each fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ope.data.contracts import FEATURES
from ope.evaluation.metrics import auc
from ope.features.prep import Prep
from ope.models.gbm import CAT_MODE, Model


@dataclass
class CVResult:
    fold_auc: list[float]
    oof: np.ndarray
    best_iterations: list[int | None]

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_auc))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_auc, ddof=1))

    @property
    def se(self) -> float:
        return self.std / np.sqrt(len(self.fold_auc))


def run_cv(
    train: pd.DataFrame,
    target: str,
    family: str,
    params: dict[str, Any] | None = None,
    cat_mode: str | None = None,
    weighted: bool = True,
    extra: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    folds: list[int] | None = None,
    on_fold=None,
) -> CVResult:
    """extra: additional numeric columns (e.g. out-of-fold scores) aligned with train.
    columns: optional feature subset after prep. on_fold(k, auc) may raise to prune."""
    cat_mode = cat_mode or CAT_MODE[family]
    y = train[target].to_numpy()
    oof = np.full(len(train), np.nan)
    aucs, iters = [], []
    for k in folds if folds is not None else sorted(train["fold"].unique()):
        tr, va = train["fold"] != k, train["fold"] == k
        prep = Prep(cat_mode=cat_mode, linear=family == "logreg")
        Xtr = prep.fit_transform(train.loc[tr, FEATURES], y[tr])
        Xva = prep.transform(train.loc[va, FEATURES])
        if extra is not None:
            Xtr = pd.concat([Xtr, extra.loc[tr]], axis=1)
            Xva = pd.concat([Xva, extra.loc[va]], axis=1)
        if columns is not None:
            Xtr, Xva = Xtr[[c for c in columns if c in Xtr]], Xva[[c for c in columns if c in Xva]]
        cats = [c for c in prep.cat_features if c in Xtr.columns]
        m = Model(family, dict(params or {}), weighted=weighted).fit(Xtr, y[tr], cats)
        oof[va.to_numpy()] = m.predict(Xva)
        aucs.append(auc(y[va], oof[va.to_numpy()]))
        iters.append(m.best_iteration)
        if on_fold is not None:
            on_fold(len(aucs) - 1, aucs[-1])
    return CVResult(aucs, oof, iters)
