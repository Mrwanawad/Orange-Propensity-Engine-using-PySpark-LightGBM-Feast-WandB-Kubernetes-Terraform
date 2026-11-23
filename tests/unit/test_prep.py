import numpy as np
import pandas as pd

from ope.data.contracts import FEATURES
from ope.features.prep import RARE, UNSEEN, Prep


def test_drops_empty_and_constant(fixture_df):
    p = Prep().fit(fixture_df[FEATURES])
    assert "Var2" in p.dropped and "Var3" in p.dropped
    assert "Var2" not in p.num_cols


def test_rare_and_unseen_levels(fixture_df):
    X = fixture_df[FEATURES].copy()
    X.loc[: len(X) - 2, "Var193"] = "common"
    X.loc[len(X) - 1, "Var193"] = "once"
    p = Prep(cat_mode="native").fit(X)
    new = X.head(2).copy()
    new["Var193"] = ["once", "never-seen"]
    out = p.transform(new)
    assert list(out["Var193"].astype(str)) == [RARE, UNSEEN]


def test_native_categories_fixed_at_fit(fixture_df):
    X = fixture_df[FEATURES]
    p = Prep(cat_mode="native")
    a = p.fit_transform(X.iloc[:400])
    b = p.transform(X.iloc[400:])
    for c in p.cat_cols:
        assert list(a[c].cat.categories) == list(b[c].cat.categories)


def test_te_mode_is_numeric_and_same_columns(fixture_df):
    X, y = fixture_df[FEATURES], fixture_df["churn"].to_numpy()
    p = Prep(cat_mode="te")
    a = p.fit_transform(X.iloc[:500], y[:500])
    b = p.transform(X.iloc[500:])
    assert list(a.columns) == list(b.columns)
    assert all(np.issubdtype(t, np.number) for t in a.dtypes)


def test_linear_mode_has_no_nan(fixture_df):
    X, y = fixture_df[FEATURES], fixture_df["churn"].to_numpy()
    p = Prep(cat_mode="te", linear=True)
    a = p.fit_transform(X.iloc[:500], y[:500])
    b = p.transform(X.iloc[500:])
    assert not a.isna().any().any() and not b.isna().any().any()


def test_missing_indicators(fixture_df):
    p = Prep().fit(fixture_df[FEATURES])
    out = p.transform(fixture_df[FEATURES].head(5))
    assert "Var1_isna" in out.columns
    pd.testing.assert_series_equal(
        out["Var1_isna"], fixture_df["Var1"].head(5).isna().astype(float), check_names=False
    )
