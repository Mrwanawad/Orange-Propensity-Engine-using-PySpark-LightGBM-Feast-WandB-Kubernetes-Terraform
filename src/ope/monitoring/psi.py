"""Population stability index, our own implementation so it is unit-tested.

PSI = sum over bins of (a - e) * ln(a / e), with e the reference share and a the current share.
Numeric: 10 quantile bins from the reference, plus one bin for missing.
Categorical: reference levels, everything else pooled into one 'other' bin, plus missing.
Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant (retrain trigger).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(e: np.ndarray, a: np.ndarray) -> float:
    e = np.clip(e, EPS, None)
    a = np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref, cur, bins: int = 10) -> float:
    ref, cur = pd.Series(ref, dtype=float), pd.Series(cur, dtype=float)
    r, c = ref.dropna(), cur.dropna()
    if len(r) == 0:
        return 0.0
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:
        edges = np.array([-np.inf, float(r.iloc[0]), np.inf])
    e = np.histogram(r, edges)[0] / len(ref)
    a = np.histogram(c, edges)[0] / len(cur)
    e = np.append(e, ref.isna().mean())
    a = np.append(a, cur.isna().mean())
    return _psi(e, a)


def psi_categorical(ref, cur, max_levels: int = 50) -> float:
    ref = pd.Series(ref, dtype="object").fillna("__missing__")
    cur = pd.Series(cur, dtype="object").fillna("__missing__")
    top = ref.value_counts().index[:max_levels]
    r = ref.where(ref.isin(top), "__other__").value_counts(normalize=True)
    c = cur.where(cur.isin(top), "__other__").value_counts(normalize=True)
    levels = r.index.union(c.index)
    return _psi(r.reindex(levels, fill_value=0).to_numpy(), c.reindex(levels, fill_value=0).to_numpy())


def psi_frame(ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            out[col] = psi_numeric(ref[col], cur[col])
        else:
            out[col] = psi_categorical(ref[col], cur[col])
    return pd.Series(out).sort_values(ascending=False)
