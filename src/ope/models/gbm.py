"""Gradient boosting models behind one small fit / predict interface.

Early stopping uses an inner 10% slice of the training fold, never the CV validation fold,
so the CV estimate stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CAT_MODE = {"lightgbm": "native", "xgboost": "native", "catboost": "raw", "logreg": "te"}


@dataclass
class Model:
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    weighted: bool = True
    seed: int = 42
    early_stopping: int = 100
    model: Any = None
    best_iteration: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, cat_features: list[str] | None = None) -> Model:
        cat_features = cat_features or []
        y = np.asarray(y)
        spw = (len(y) - y.sum()) / max(y.sum(), 1) if self.weighted else 1.0
        Xtr, Xes, ytr, yes = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=self.seed
        )
        p = dict(self.params)
        if self.family == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=p.pop("n_estimators", 3000),
                scale_pos_weight=spw,
                metric="auc",
                random_state=self.seed,
                verbose=-1,
                n_jobs=-1,
                **p,
            )
            self.model.fit(
                Xtr,
                ytr,
                eval_set=[(Xes, yes)],
                eval_metric="auc",
                categorical_feature=cat_features or "auto",
                callbacks=[lgb.early_stopping(self.early_stopping, first_metric_only=True, verbose=False)],
            )
            self.best_iteration = self.model.best_iteration_
        elif self.family == "xgboost":
            import xgboost as xgb

            self.model = xgb.XGBClassifier(
                n_estimators=p.pop("n_estimators", 3000),
                scale_pos_weight=spw,
                tree_method="hist",
                enable_categorical=True,
                max_cat_to_onehot=1,
                eval_metric="auc",
                early_stopping_rounds=self.early_stopping,
                random_state=self.seed,
                n_jobs=-1,
                **p,
            )
            self.model.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
            self.best_iteration = self.model.best_iteration
        elif self.family == "catboost":
            from catboost import CatBoostClassifier

            self.model = CatBoostClassifier(
                iterations=p.pop("iterations", 3000),
                scale_pos_weight=spw,
                eval_metric="AUC",
                random_seed=self.seed,
                od_type="Iter",
                od_wait=self.early_stopping,
                verbose=False,
                thread_count=-1,
                allow_writing_files=False,
                **p,
            )
            self.model.fit(
                Xtr, ytr, cat_features=cat_features, eval_set=(Xes, yes), use_best_model=True
            )
            self.best_iteration = self.model.get_best_iteration()
        elif self.family == "logreg":
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(
                C=p.pop("C", 0.1),
                class_weight="balanced" if self.weighted else None,
                max_iter=3000,
                **p,
            )
            self.model.fit(X, y)
        else:
            raise ValueError(self.family)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Raw (uncalibrated) score in [0, 1]."""
        return self.model.predict_proba(X)[:, 1]
