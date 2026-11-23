# Decisions

Every choice the brief left open, one line each with the reason.

## Data and methodology

- **Labels never co-occur**, so the stratification pattern has four values (none, churn, appetency, upselling); stratifying on it keeps all three rates equal across splits.
- **`subscriber_id` = first 16 hex chars of a hash of the row index**: stable across reruns and the same length the API validates.
- **Early stopping uses an inner 10% slice of the training fold**, never the CV validation fold, so CV AUC is not tuned on its own validation rows.
- **Out-of-fold target encoding uses scikit-learn `TargetEncoder`** (internal cross-fitting, smoothing) plus frequency encoding: maintained, fold-safe, no custom code.
- **Encoding comparison is run with fixed moderate defaults before tuning**; tuning then uses the winning encoding and weighting per (target, family). Tuning every encoding would triple the compute for a choice the defaults already separate clearly.
- **Weighted vs unweighted is tested only on the best encoding** per (target, family) for the same reason.
- **Optuna wall-clock caps: LightGBM 30 min, XGBoost 20 min, CatBoost 20 min per target** (`configs/base.yaml`). The trial counts (100/40/30) stay as targets; on an 8-core laptop the caps bind first. Recorded trial counts are in `reports/cv/tuned.json`.
- **MedianPruner(n_startup_trials=8, n_warmup_steps=1)** prunes on the running mean of fold AUCs: a trial must survive two folds before it can be pruned.
- **Feature selection is tested on LightGBM only** (permutation importance per fold, kept if > 0 in at least 4 of 5) and adopted only if CV AUC does not fall; GBMs already ignore weak features, so this is a check, not an expected gain.
- **Cross-target test is fully nested**: other-target scores for training rows come from inner out-of-fold models, so nothing leaks.
- **Stacking**: logistic meta-model on out-of-fold logits of the three tuned GBMs; kept only if it beats the best single model by more than that model's CV standard error.
- **Calibration method is chosen by 5-fold cross-validated Brier inside the calibration split**; fitting and scoring on the same rows would always pick isotonic.
- **Gate "not below champion minus one SE"** is skipped on the first freeze (no champion yet) and recorded as such.
- **Test split opened once** after the champions were frozen and promoted; `final_evaluation: true` stays in `configs/base.yaml` afterwards so the access is visible in history, and the W&B run `test-split-opened` records it.
- **KDD Cup 2009 reference numbers** were read from the paper PDF (Guyon et al. 2009, Table 2 and Section 5) and stored with page references in `reports/kdd2009_reference.json`. The paper's winners used the LARGE dataset; we use SMALL, so the comparison is approximate.

## Policy

- **Business numbers are illustrative assumptions for an Egyptian mobile operator** (`configs/business.yaml`), sized for the 50k base: ARPU 120 EGP, 45% margin, weekly budget 60k EGP.
- **Channel per action is fixed** (retain = call centre, cross-sell = SMS, up-sell = app push): one channel per offer keeps capacity constraints readable for the CVM team.
- **Greedy has a second pass**: subscribers cut only because their best action's channel is full get their next-best action in a channel with room. This closes most of the gap a pure single-pass greedy leaves when one channel saturates.
- **Distributed policy in Spark uses EV-per-cost buckets** instead of a global sort (a global sort with a running sum collapses to one partition at 100M rows). Rows in the boundary bucket are left out, so it never exceeds budget or capacity.
- **CP-SAT money is scaled to integer cents**; the report includes the solver's best bound so the gap is honest even if the time limit stops before proving optimality.

## Platform

- **PySpark 4.1 instead of 3.5**: 4.1 is the current Spark release line and Iceberg 1.12.0 ships a matching runtime (`iceberg-spark-runtime-4.1_2.13`); the APIs used here (`mapInPandas`, `writeTo`, partition transforms) are the same.
- **SeaweedFS instead of MinIO** for local S3: MinIO community images left Docker Hub; SeaweedFS speaks the S3 API and runs in one container.
- **Iceberg REST catalog fixture** (`apache/iceberg-rest-fixture`) with a SQLite catalog for local runs; AWS would use Glue or a managed REST catalog.
- **Airflow 3.3** (official image `apache/airflow:3.3.0-python3.12`), `airflow.sdk.DAG`. One task factory switches between `BashOperator` (local) and `KubernetesPodOperator` (production) with the same command.
- **Spark on Windows is avoided**: all Spark jobs run in the `ope-jobs` image (Java 17 inside), which also runs H2O, so nothing needs Java on the host.
- **Feast offline store is parquet locally**; in production the offline source is `lake.gold.subscriber_features` (Iceberg, written by `features/spark_features.py`).
- **W&B runs offline** (`WANDB_MODE=offline`); `wandb sync wandb/` uploads them once a login exists. The registry aliases are mirrored to `artifacts/models/registry.json` so serving never needs the W&B server.
- **API rate limit is a fixed one-minute window** in Redis (shared across replicas), memory fallback; simple and good enough for internal B2B clients.
- **Scores for the console's simulator are read from the batch parquet** loaded at API start, so `/v1/campaigns/simulate` re-runs the policy in memory on 50k rows without Redis scans.
- **Retention Analyst falls back to a scripted model** (deterministic routing + templates) when no `ANTHROPIC_API_KEY` is present, so tool wiring, grounding checks and the gold-set eval run in CI without keys.
- **LangSmith tracing goes through the Agent Framework's OpenTelemetry setup** to `https://api.smith.langchain.com/otel`.
- **Grafana, Prometheus and Redis pinned** to explicit image tags in compose for reproducible local runs.
- **Terraform is validate/plan only**; no backend configured (commented S3 backend with lockfile).
- **kind smoke test uses tiny models trained on the synthetic test fixture** (`deploy/kind/smoke_artifacts.py`) so CI checks the chart and image without the real data or the trained bundles.
- **Console pages live in one `console/app.py`** (four tabs, ~240 lines); splitting into a `pages/` package added files without adding clarity.
