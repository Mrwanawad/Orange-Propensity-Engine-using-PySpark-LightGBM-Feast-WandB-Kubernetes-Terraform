"""Leakage checks (section 12): fold-fitted prep never sees validation rows, the test
split is locked, and shuffled labels collapse AUC to about 0.5."""

import numpy as np
import pytest

from ope import config
from ope.data import split
from ope.data.contracts import FEATURES
from ope.features.prep import UNSEEN, Prep
from ope.models.cv import run_cv
from tests.conftest import make_fixture


def with_folds(df):
    sp = split.make_splits(df, seed=0)
    out = df.merge(sp, on="subscriber_id")
    return out[out["split"] == "train"].reset_index(drop=True)


def test_prep_fitted_on_fold_never_sees_validation_levels(fixture_df):
    train = with_folds(fixture_df)
    tr, va = train[train["fold"] != 0].copy(), train[train["fold"] == 0].copy()
    va["Var194"] = [f"val-only-{i}" for i in range(len(va))]
    p = Prep(cat_mode="native").fit(tr[FEATURES])
    assert not any(str(v).startswith("val-only") for v in p.levels["Var194"])
    assert set(p.transform(va[FEATURES])["Var194"].astype(str)) == {UNSEEN}


def test_target_encoding_is_cross_fitted(fixture_df):
    """Training rows get out-of-fold encodings, so their own label never feeds their value."""
    train = with_folds(fixture_df)
    X, y = train[FEATURES], train["churn"].to_numpy()
    p = Prep(cat_mode="te")
    cross_fitted = p.fit_transform(X, y)
    full_fit = p.transform(X)
    te_cols = [c for c in cross_fitted.columns if c.endswith("_te")]
    assert not np.allclose(cross_fitted[te_cols].to_numpy(), full_fit[te_cols].to_numpy())


def test_test_split_is_locked():
    assert config.load()["final_evaluation"] is False
    with pytest.raises(split.TestSetLocked):
        split.load_test()
    with pytest.raises(split.TestSetLocked):
        split.load_split("test")


def test_splits_are_disjoint_and_stratified(fixture_df):
    sp = split.make_splits(fixture_df, seed=0)
    assert sp["subscriber_id"].is_unique
    assert set(sp["split"]) == {"train", "calibration", "test"}
    assert (sp.loc[sp["split"] != "train", "fold"] == -1).all()
    rates = fixture_df.merge(sp, on="subscriber_id").groupby("split")["churn"].mean()
    assert rates.max() - rates.min() < 0.05


def test_shuffled_labels_collapse_auc():
    train = with_folds(make_fixture(4000, seed=1))
    real = run_cv(train, "churn", "lightgbm", {"learning_rate": 0.1}, weighted=False, folds=[0, 1])
    shuffled = train.copy()
    shuffled["churn"] = np.random.default_rng(0).permutation(shuffled["churn"].to_numpy())
    fake = run_cv(
        shuffled, "churn", "lightgbm", {"learning_rate": 0.1}, weighted=False, folds=[0, 1]
    )
    assert real.mean > 0.7
    assert abs(fake.mean - 0.5) < 0.12
