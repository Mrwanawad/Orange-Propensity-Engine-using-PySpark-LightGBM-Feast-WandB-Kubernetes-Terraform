import numpy as np
import pandas as pd
import pytest

from ope.data.contracts import CATEGORICAL, NUMERIC, TARGETS
from ope.data.ingest import subscriber_id


def make_fixture(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Small synthetic table with the real schema. Signal lives in Var1 and Var192."""
    rng = np.random.default_rng(seed)
    data = {"subscriber_id": [subscriber_id(i) for i in range(n)]}
    for c in NUMERIC:
        x = rng.normal(size=n)
        x[rng.random(n) < 0.3] = np.nan
        data[c] = x
    data["Var2"] = np.nan  # fully missing column, must be dropped
    data["Var3"] = 1.0  # constant column, must be dropped
    for c in CATEGORICAL:
        lv = rng.choice([f"L{i}" for i in range(30)], size=n, p=None)
        s = pd.Series(lv, dtype="string")
        s[rng.random(n) < 0.2] = pd.NA
        data[c] = s
    df = pd.DataFrame(data)
    score = (
        3.0 * np.nan_to_num(df["Var1"].to_numpy())
        + 3.0 * (df["Var192"].fillna("x") == "L1").to_numpy()
    )
    p = 1 / (1 + np.exp(-(score - 3.0)))
    churn = (rng.random(n) < p).astype("int8")
    df["churn"] = churn
    df["appetency"] = ((rng.random(n) < 0.05) & (churn == 0)).astype("int8")
    df["upselling"] = ((rng.random(n) < 0.08) & (churn == 0) & (df["appetency"] == 0)).astype(
        "int8"
    )
    for t in TARGETS:
        df[t] = df[t].astype("int8")
    return df


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return make_fixture()
