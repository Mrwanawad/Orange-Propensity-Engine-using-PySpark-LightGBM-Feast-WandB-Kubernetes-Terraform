# Data card: KDD Cup 2009 (Orange), small version

## Source

| Target | OpenML id | File | SHA-256 (first 16) | Positives | Rate |
|---|---|---|---|---|---|
| churn | 1112 | `KDDCup09_churn.arff` | `797ab790cd99a1bb` | 3,672 | 7.34% |
| appetency | 1111 | `KDDCup09_appetency.arff` | `201abbcfec1177ba` | 890 | 1.78% |
| upselling | 1114 | `KDDCup09_upselling.arff` | `978359bbb764c850` | 3,682 | 7.36% |

Full hashes, OpenML md5 and download URLs: `data/raw/MANIFEST.json`. Licence on OpenML: **Public**.
Raw files are never edited.

Public research data released by Orange for the KDD Cup 2009. This project is not affiliated with or
endorsed by Orange.

## Join

The three datasets have the same 50,000 feature rows in the same order: `ope.data.ingest` hashes
every row's features and refuses to join if any hash differs. The three labels are joined into one
table with `subscriber_id` = a stable hash of the row index (16 hex characters).

The labels never co-occur (0 rows have more than one positive label), so the stratification
pattern for the split has four values: none, churn, appetency, upselling.

## Shape

| Fact | Value |
|---|---|
| Rows | 50,000 |
| Features | 230: 192 numeric (`Var1`..`Var190`, `Var209`, `Var230`) and 38 categorical (`Var191`..`Var229` except `Var209`) |
| Missing cells | 8,024,152 (69.8% of all feature cells) |
| Rows with at least one missing value | 50,000 (all) |
| Features 100% missing | 18 |
| Features more than 99% missing | 24 (dropped by the prep step, fitted per fold) |
| Features more than 50% missing | 161 |
| Features with no missing value | 19 |
| Categoricals with more than 1,000 levels | 9; largest `Var200` and `Var214` with 15,415 levels each |

## Splits (seed 42, `data/processed/splits.parquet`)

| Split | Rows | churn | appetency | upselling |
|---|---|---|---|---|
| train (5 stratified CV folds inside) | 34,999 | 7.35% | 1.78% | 7.36% |
| calibration | 5,001 | 7.34% | 1.78% | 7.36% |
| test (opened once) | 10,000 | 7.34% | 1.78% | 7.37% |

## Lakehouse

`lake.bronze.subscribers_raw` and `lake.silver.subscribers` (Iceberg, REST catalog on S3),
partitioned by `snapshot_date` and `bucket(64, subscriber_id)`. Silver is written only after the
Pandera contract passes on every Arrow batch and the id-uniqueness check passes in Spark.
Training is pinned to silver snapshot `1751266164113702854` (committed 2026-11-12, 50,000 rows),
recorded in `data/processed/silver_snapshot.json` and in every freeze run's metadata.

## Limits

- **No timestamp**: no out-of-time validation is possible. The split is stratified random, which
  is optimistic compared with a real "train on last quarter, score next month" setup.
- **Anonymised features**: no domain feature engineering (tenure, top-ups, dropped calls). The
  Feast repo has a placeholder for such a view.
- **No treatment / control data**: no uplift model can be trained (see `policy/uplift.py`).
- **Official test labels never released**: our test AUCs are on our own 20% hold-out.

## Synthetic scale replica

`data/replica/` (10,000,000 rows in 20 parquet parts, built by `scripts/make_scale_replica.py`) resamples real
rows with 1% numeric jitter and 1% categorical swaps. It is **synthetic**, used only for throughput
benchmarks, never for training or model metrics.
