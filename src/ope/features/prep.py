"""Fold-safe feature preparation. Fit on the training fold only, then transform anything.

cat_mode:
  native  categoricals as pandas category with levels fixed at fit (LightGBM / XGBoost native)
  raw     categoricals as plain strings (CatBoost ordered target statistics)
  te      out-of-fold target encoding + frequency encoding, all numeric
linear=True adds median imputation and standard scaling (logistic baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from ope import config
from ope.data.contracts import CATEGORICAL, NUMERIC

RARE, UNSEEN, MISSING = "__rare__", "__unseen__", "__missing__"


@dataclass
class Prep:
    cat_mode: str = "native"
    linear: bool = False
    seed: int = 42
    dropped: list[str] = field(default_factory=list)
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    indicator_cols: list[str] = field(default_factory=list)
    levels: dict[str, list[str]] = field(default_factory=dict)
    freq: dict[str, dict[str, float]] = field(default_factory=dict)
    medians: pd.Series | None = None
    means: pd.Series | None = None
    stds: pd.Series | None = None
    te: TargetEncoder | None = None
    _rare_seen: dict[str, set[str]] = field(default_factory=dict)

    def _fit_columns(self, X: pd.DataFrame) -> None:
        c = config.load()["prep"]
        miss = X[NUMERIC + CATEGORICAL].isna().mean()
        nunique = X[NUMERIC + CATEGORICAL].nunique(dropna=True)
        self.dropped = sorted(
            [col for col in miss.index if miss[col] > c["max_missing_drop"] or nunique[col] <= 1],
            key=lambda s: int(s[3:]),
        )
        keep = set(miss.index) - set(self.dropped)
        self.num_cols = [col for col in NUMERIC if col in keep]
        self.cat_cols = [col for col in CATEGORICAL if col in keep]
        self.indicator_cols = [
            col
            for col in self.num_cols
            if c["indicator_min_missing"] <= miss[col] <= c["max_missing_drop"]
        ]
        min_count = c["rare_level_min_count"]
        for col in self.cat_cols:
            counts = X[col].fillna(MISSING).value_counts()
            kept = sorted(counts[counts >= min_count].index.astype(str))
            self.levels[col] = kept + [RARE, UNSEEN]

    def _clean_cats(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.cat_cols:
            s = X[col].astype("object").where(X[col].notna(), MISSING).astype(str)
            known = set(self.levels[col])
            # a level seen at fit but below the count threshold is rare; never seen is unseen
            seen_rare = self._rare_seen.get(col, set())
            s = s.where(s.isin(known), np.where(s.isin(seen_rare), RARE, UNSEEN))
            out[col] = s
        return pd.DataFrame(out, index=X.index)

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> Prep:
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        self._fit_columns(X)
        self._rare_seen = {
            col: set(X[col].dropna().astype(str)) - set(self.levels[col]) for col in self.cat_cols
        }
        cats = self._clean_cats(X)
        if self.cat_mode == "te":
            if y is None:
                raise ValueError("target encoding needs y")
            self.freq = {
                col: cats[col].value_counts(normalize=True).to_dict() for col in self.cat_cols
            }
            cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
            self.te = TargetEncoder(target_type="binary", smooth="auto", cv=cv)
            # cross-fitted: each training row is encoded by a model that never saw its label
            enc = self.te.fit_transform(cats, y)
            out = self._assemble(X, cats, enc)
        else:
            out = self._assemble(X, cats, None)
        if self.linear:
            self.medians = out.median()
            filled = out.fillna(self.medians)
            self.means, self.stds = filled.mean(), filled.std().replace(0, 1.0)
            out = (filled - self.means) / self.stds
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cats = self._clean_cats(X)
        enc = self.te.transform(cats) if self.cat_mode == "te" and self.te is not None else None
        out = self._assemble(X, cats, enc)
        if self.linear:
            out = (out.fillna(self.medians) - self.means) / self.stds
        return out

    def _assemble(self, X: pd.DataFrame, cats: pd.DataFrame, enc) -> pd.DataFrame:
        parts = [X[self.num_cols].astype("float64")]
        if self.indicator_cols:
            ind = X[self.indicator_cols].isna().astype("float64")
            ind.columns = [f"{c}_isna" for c in self.indicator_cols]
            parts.append(ind)
        if self.cat_mode == "te":
            parts.append(
                pd.DataFrame(enc, index=X.index, columns=[f"{c}_te" for c in self.cat_cols])
            )
            fr = pd.DataFrame(
                {
                    f"{c}_freq": cats[c].map(self.freq[c]).fillna(0.0).astype("float64")
                    for c in self.cat_cols
                },
                index=X.index,
            )
            parts.append(fr)
        elif self.cat_mode == "native":
            parts.append(
                pd.DataFrame(
                    {c: pd.Categorical(cats[c], categories=self.levels[c]) for c in self.cat_cols},
                    index=X.index,
                )
            )
        elif self.cat_mode == "raw":
            parts.append(cats)
        else:
            raise ValueError(self.cat_mode)
        return pd.concat(parts, axis=1)

    @property
    def cat_features(self) -> list[str]:
        return self.cat_cols if self.cat_mode in ("native", "raw") else []
