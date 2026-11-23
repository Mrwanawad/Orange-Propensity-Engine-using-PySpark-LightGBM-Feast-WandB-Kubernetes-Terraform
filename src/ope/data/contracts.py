"""Data contract for the subscriber table, used by the pipeline, the API and the tests.

One column spec, two backends: pandas (training, API) and Polars (EDA, batch prep).
Spark jobs convert each Arrow batch to pandas and reuse the pandas schema.
"""

from __future__ import annotations

import pandera.pandas as pa
import pandera.polars as pap
import polars as pl

# Types as declared in the OpenML ARFF: 192 numeric, 38 nominal. Var209 and Var230 sit in the
# nominal range but are declared numeric (and are empty in the small dataset).
NUMERIC = [f"Var{i}" for i in range(1, 191)] + ["Var209", "Var230"]
CATEGORICAL = [f"Var{i}" for i in range(191, 231) if i not in (209, 230)]
FEATURES = [f"Var{i}" for i in range(1, 231)]
TARGETS = ["churn", "appetency", "upselling"]
ID = "subscriber_id"


def pandas_schema(with_labels: bool = True, required: bool = True) -> pa.DataFrameSchema:
    cols: dict[str, pa.Column] = {
        ID: pa.Column(str, pa.Check.str_length(16, 16), unique=True, nullable=False),
    }
    for c in NUMERIC:
        cols[c] = pa.Column(float, nullable=True, coerce=True, required=required)
    for c in CATEGORICAL:
        cols[c] = pa.Column("string", nullable=True, coerce=True, required=required)
    if with_labels:
        for t in TARGETS:
            cols[t] = pa.Column("int8", pa.Check.isin([0, 1]), coerce=True)
    return pa.DataFrameSchema(cols, strict=True, name="subscribers")


def feature_payload_schema() -> pa.DataFrameSchema:
    """On-demand scoring payload: id plus any subset of features (missing means NaN)."""
    return pandas_schema(with_labels=False, required=False)


def polars_schema(with_labels: bool = True) -> pap.DataFrameSchema:
    cols: dict[str, pap.Column] = {
        ID: pap.Column(pl.Utf8, pap.Check.str_length(16, 16), unique=True, nullable=False),
    }
    for c in NUMERIC:
        cols[c] = pap.Column(pl.Float64, nullable=True)
    for c in CATEGORICAL:
        cols[c] = pap.Column(pl.Utf8, nullable=True)
    if with_labels:
        for t in TARGETS:
            cols[t] = pap.Column(pl.Int8, pap.Check.isin([0, 1]))
    return pap.DataFrameSchema(cols, strict=True, name="subscribers")
