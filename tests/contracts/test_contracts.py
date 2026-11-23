import pandas as pd
import pandera.errors as pe
import polars as pl
import pytest

from ope.data import contracts as c


def test_fixture_passes_both_backends(fixture_df):
    c.pandas_schema().validate(fixture_df)
    c.polars_schema().validate(
        pl.from_pandas(fixture_df).with_columns(pl.col(c.TARGETS).cast(pl.Int8))
    )


def test_duplicate_id_fails(fixture_df):
    bad = pd.concat([fixture_df, fixture_df.iloc[:1]])
    with pytest.raises((pe.SchemaError, pe.SchemaErrors)):
        c.pandas_schema().validate(bad)


def test_bad_label_fails(fixture_df):
    bad = fixture_df.copy()
    bad.loc[0, "churn"] = 2
    with pytest.raises((pe.SchemaError, pe.SchemaErrors)):
        c.pandas_schema().validate(bad)


def test_unknown_column_fails(fixture_df):
    bad = fixture_df.assign(Var999=1.0)
    with pytest.raises((pe.SchemaError, pe.SchemaErrors)):
        c.pandas_schema().validate(bad)


def test_payload_schema_accepts_partial_features(fixture_df):
    payload = fixture_df[["subscriber_id", "Var1", "Var192"]].head(3)
    c.feature_payload_schema().validate(payload)


def test_column_counts_match_openml_metadata():
    assert len(c.NUMERIC) == 192
    assert len(c.CATEGORICAL) == 38
    assert sorted(c.NUMERIC + c.CATEGORICAL, key=lambda s: int(s[3:])) == c.FEATURES
