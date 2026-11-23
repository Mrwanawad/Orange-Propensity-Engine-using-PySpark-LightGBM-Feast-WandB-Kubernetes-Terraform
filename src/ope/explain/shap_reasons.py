"""SHAP reason codes per subscriber and global importance (TreeExplainer).

Reasons come from the first (strongest) tree member of a bundle. They explain the raw model
score in log-odds; calibration is monotone, so the ranking of reasons is unchanged.
Computed only for subscribers selected for an action plus a small monitoring sample (section 4.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ope.data.contracts import FEATURES


def raw_name(col: str) -> str:
    """Map an engineered column back to the source feature."""
    for suffix in ("_isna", "_te", "_freq"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def _member(bundle: dict[str, Any]) -> dict[str, Any]:
    order = {"lightgbm": 0, "xgboost": 1, "catboost": 2}
    return sorted(bundle["members"], key=lambda m: order.get(m["family"], 9))[0]


def shap_matrix(bundle: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    m = _member(bundle)
    Xp = m["prep"].transform(X.reindex(columns=FEATURES))
    if m.get("columns"):
        Xp = Xp[m["columns"]]
    model = m["model"].model
    if m["family"] == "lightgbm":
        # LightGBM's built-in TreeSHAP: same algorithm as shap.TreeExplainer, handles categoricals
        contrib = model.predict(Xp, pred_contrib=True)
        return np.asarray(contrib)[:, :-1], Xp
    if m["family"] == "catboost":
        from catboost import Pool

        pool = Pool(Xp, cat_features=[c for c in m["prep"].cat_features if c in Xp.columns])
        return model.get_feature_importance(pool, type="ShapValues")[:, :-1], Xp
    import shap

    return shap.TreeExplainer(model).shap_values(Xp), Xp


def reasons(bundle: dict[str, Any], X: pd.DataFrame, top: int = 5) -> list[list[dict[str, Any]]]:
    """Top features pushing each subscriber's score up, with their raw value."""
    sv, Xp = shap_matrix(bundle, X)
    cols = np.array(Xp.columns)
    out = []
    for i in range(len(Xp)):
        idx = np.argsort(-sv[i])[:top]
        row = []
        for j in idx:
            if sv[i, j] <= 0:
                break
            src = raw_name(cols[j])
            val = X.iloc[i][src] if src in X.columns else None
            row.append({
                "feature": src,
                "signal": "missing" if cols[j].endswith("_isna") else "value",
                "value": None if pd.isna(val) else (val if isinstance(val, str) else float(val)),
                "shap": round(float(sv[i, j]), 4),
            })
        out.append(row)
    return out


def global_importance(bundle: dict[str, Any], X: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    sv, Xp = shap_matrix(bundle, X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=[raw_name(c) for c in Xp.columns])
    imp = imp.groupby(level=0).sum().sort_values(ascending=False)
    return imp.head(top).rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
