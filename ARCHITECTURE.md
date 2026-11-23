# Orange Propensity Engine: Architecture and Build Brief

This file is the full brief for the Claude Code session that implements the project. Read all of it before writing code. Where it says "verify", check the current docs or package before relying on the claim. Where a decision is missing, pick the simplest option that a senior engineer would defend, write it in `DECISIONS.md`, and continue. Stop and ask the owner only for things listed under "Ask first".

---

## 1. What we are building and for whom

**Stakeholder:** a telecom operator's CVM (customer value management) and retention team. They want one answer every week: *which subscribers should we contact, with which offer, and why, within this month's budget.*

**Project:** a production-grade propensity and next-best-action engine built on Orange's own public customer data (KDD Cup 2009). For every subscriber it:

1. Scores three propensities: **churn** (will leave), **appetency** (will buy a new product or service), **up-selling** (will buy an upgrade or add-on).
2. Calibrates each score into a real probability.
3. Explains each score with the top reason codes (SHAP).
4. Picks **one action per subscriber** (retain, cross-sell, up-sell, or no contact) that maximises expected value under a campaign budget and channel capacity.
5. Monitors data drift and estimated model performance before labels arrive, and triggers retraining.
6. Serves scores and actions through a batch pipeline (all subscribers) and a low-latency API (care agent screen, IVR, app).
7. Gives the CVM team a console with a campaign simulator, and a small LLM "Retention Analyst" that answers questions using only numbers from the system's tools.

**Design target:** an operator with **100 million subscribers**. The real dataset has 50,000 rows, so the code runs on a laptop, while the architecture, partitioning, storage and serving are built for 100M. Scale is shown by a throughput benchmark (section 12), not claimed.

**Why this matters for the owner:** this is the portfolio piece for a Senior Data Scientist role with Systems Limited for its client Orange. It must read as the work of a senior data scientist who also ships to production: strong ML methodology first, then production engineering.

---

## 2. Data

### 2.1 Source (confirmed 2026-11-12)

OpenML, the "small" KDD Cup 2009 version (230 features):

| Target | OpenML id | Rows | Positives | Rate |
|---|---|---|---|---|
| churn | 1112 (`KDDCup09_churn`) | 50,000 | 3,672 | 7.3% |
| appetency | 1111 (`KDDCup09_appetency`) | 50,000 | 890 | 1.8% |
| upselling | 1114 (`KDDCup09_upselling`) | 50,000 | 3,682 | 7.4% |

Facts from OpenML metadata: 192 numeric and 38 categorical features plus the target; every row has missing values (about 8.0M missing cells); the biggest categorical has 15,415 distinct values. Features are anonymised (`Var1` ... `Var230`). There is no customer id and no timestamp.

### 2.2 Ingestion rules

- Download with `openml` (or the ARFF file URL from the OpenML API), cache the raw files under `data/raw/`, and record the SHA-256 of each file in `data/raw/MANIFEST.json`. Never edit raw files.
- **Verify** that the three datasets have identical feature rows in identical order (hash each row's features). If they match, join the three labels into one table with a generated `subscriber_id` (a stable hash of the row index). If they do not match, stop and report.
- Check the dataset licence on the OpenML page and record it in the README.
- README disclaimer: public research data released by Orange for KDD Cup 2009; the project is not affiliated with or endorsed by Orange; all money values (ARPU, offer costs) are stated assumptions.

### 2.3 Honest limits (write them into the README)

- No timestamp, so no out-of-time validation. Use a stratified split and say so.
- Anonymised features, so no domain feature engineering (tenure, top-ups, dropped calls). The README shows where such features would plug in (the feature store schema has a placeholder group).
- No treatment or control data, so no uplift model. The policy uses propensity times an assumed response rate. The uplift module is designed (section 6.6) but marked "needs campaign A/B data".
- The official KDD test labels were never released, so our test AUCs are on our own held-out split. Compare them to the published KDD Cup 2009 results as an approximate reference only (section 6.5).

### 2.4 Synthetic scale replica (benchmarks only)

`scripts/make_scale_replica.py` builds a 10M-row parquet replica by resampling real rows with small numeric jitter and remapped categorical levels. It is used only for throughput and latency benchmarks. Never train on it, never report model metrics from it, and label it synthetic everywhere.

---

## 3. Which tools, and why

### 3.1 The owner's stack versus what his public repos already show

His GitHub profile README lists his skills. All 22 of his repos were scanned on 2026-11-12 (dependency files, Python imports, workflows, infra files). The listed tools below **appear in none of his repos**. Where one is the right tool here, this project uses it so his public work backs the claim.

| Listed on profile, not in any repo | Used here? | Where |
|---|---|---|
| PySpark | **Yes** | feature pipeline and batch scoring for 100M subscribers |
| Weights & Biases | **Yes** | experiment tracking, Optuna studies, artifacts, model registry |
| H2O | **Yes** | H2O AutoML as the challenger benchmark |
| Terraform | **Yes** | AWS infrastructure as code (validate and plan only) |
| Kubernetes (Helm) | **Yes** | Helm chart and kind smoke test (his only k8s so far is raw manifests in Dentalbase-CRM) |
| Microsoft Agent Framework | **Yes** | Retention Analyst agent |
| LangSmith | **Yes** | tracing and evals for the analyst agent |
| Plotly Dash | **Yes** | CVM console |
| Amazon SageMaker | No | Spark on EKS covers batch scoring; a second platform adds cost and no signal |
| Auto-sklearn | No | Linux only and slow on 230 features with 15k-level categoricals; H2O AutoML fills the AutoML role |
| TensorFlow / Keras | No | deep nets do not beat GBMs on this kind of tabular data; no value |
| Pinecone / ChromaDB / AutoGen / Vertex AI | No | no retrieval or second agent framework needed |

Already shown in other repos and reused here because they are the best fit: LightGBM, XGBoost, CatBoost, Optuna, SHAP, imbalanced-learn (evaluation only), Pandera, Polars, Redis, FastAPI, OpenTelemetry, Prometheus, Airflow, FastMCP, GitHub Actions, Docker.

Langfuse is **not** used: it is already in AI-OS-Sales-Copilot and Market-Lead-Intelligence-Agent. LangSmith takes the LLM observability role here.

### 3.2 Final stack

| Layer | Choice | Reason |
|---|---|---|
| Language and tooling | Python 3.12, `uv`, `ruff`, `pyright`, `pytest`, `pre-commit` | standard and fast |
| Lakehouse storage | Apache Iceberg tables on S3 (MinIO locally) | engine-neutral open table format, time travel for reproducible training snapshots |
| Batch compute | PySpark 3.5 (Spark on Kubernetes in production, local mode in dev) | the default at telecom scale; partitioned scoring of 100M rows |
| Single-node work | Polars | fast EDA and the 50k-row training path |
| Data contracts | Pandera (pandas, Polars and PySpark backends) | one schema used by the pipeline and the tests |
| Feature store | Feast: offline store = parquet/Iceberg via Spark, online store = Redis | point-in-time correct training sets and millisecond online lookups |
| Models | LightGBM (candidate champion), XGBoost, CatBoost, logistic regression baseline, H2O AutoML challenger | GBMs win on tabular data; the baseline and AutoML keep claims honest |
| Tuning | Optuna (TPE plus median pruner) | logged to W&B |
| Calibration | isotonic or Platt, chosen by Brier score on a calibration fold | probabilities feed money decisions |
| Explainability | SHAP TreeExplainer | per-subscriber reason codes plus global importance |
| Decision policy | expected-value greedy allocation; OR-Tools CP-SAT for the optimality check | greedy scales to 100M; CP-SAT proves the greedy gap on a sample |
| Tracking and registry | Weights & Biases (runs, sweeps, artifacts, registry aliases `champion` and `challenger`) | one tool for lineage; `WANDB_MODE=offline` works without an account |
| Orchestration | Apache Airflow 2.10+ (or 3.x if the official Helm chart supports it; verify) with `SparkKubernetesOperator` / `KubernetesPodOperator` | the most common scheduler at operators |
| Drift and performance monitoring | NannyML (CBPE estimated AUC before labels arrive, univariate and multivariate drift) | churn labels arrive 30 to 60 days late; estimating performance without labels is the senior move |
| Online serving | FastAPI with uvicorn workers; scores precomputed into Redis; on-demand model path in process | p95 under 50 ms from Redis |
| API protection | OAuth2 client credentials (JWT) with an API-key fallback for local; rate limiting per client | enterprise integration pattern |
| Observability | OpenTelemetry traces, Prometheus metrics, Grafana dashboards | standard |
| CVM console | Plotly Dash | campaign simulator and customer cards |
| LLM analyst | Microsoft Agent Framework (Python), tools served by a FastMCP server, traced to LangSmith | see section 9 |
| LLM provider | Anthropic `claude-sonnet-5-5` through the framework's Anthropic connector (verify it exists); otherwise an OpenAI-compatible endpoint via OpenRouter | |
| Load testing | Locust | API latency under load |
| Containers and deploy | Docker, Helm chart, kind for a local cluster smoke test | |
| Infrastructure as code | Terraform for AWS: VPC, EKS, S3, ECR, ElastiCache (Redis), IAM roles for service accounts | `terraform validate` and `plan` only |
| CI | GitHub Actions | lint, types, tests, training smoke test, image build, `helm lint`, `terraform validate`, `tflint` |

Not in the core build: Kafka for real-time triggers (for example "top-up stopped for 7 days"). Section 8.3 describes it as a later extension only.

---

## 4. System architecture

```
                         +---------------------------- Airflow (weekly + on drift alert) ---------------------------+
                         |                                                                                           |
 OpenML raw ARFF  -->  ingest  -->  Iceberg: bronze.subscribers_raw  --(Pandera contract)-->  silver.subscribers   |
 (MinIO/S3)               |                                                                         |                 |
                          |                                                          Spark feature jobs               |
                          |                                                                         v                 |
                          |                                              Feast offline store (gold.features_*)        |
                          |                                                    |                         |            |
                          |                                     training set (point-in-time)     materialize          |
                          |                                                    v                         v            |
                          |                 train: baseline / LGBM / XGB / CatBoost / H2O        Redis online store    |
                          |                 Optuna -> W&B runs -> calibrate -> gate               |                    |
                          |                                  |                                    |                    |
                          |                          W&B registry (champion alias)                |                    |
                          |                                  v                                    |                    |
                          |            Spark batch scoring (mapInPandas, model broadcast)         |                    |
                          |              -> gold.scores  (p_churn, p_app, p_up, calibrated)       |                    |
                          |              -> SHAP reason codes for actioned subscribers            |                    |
                          |              -> NBA policy (budget, capacity, suppression)            |                    |
                          |              -> gold.actions + campaign file + Redis (scores/actions) |                    |
                          |                                  |                                    |                    |
                          |            NannyML: drift + CBPE estimated AUC  -> alert -> retrain DAG                    |
                          +------------------------------------------------------------------------------------------+

  Consumers:
   FastAPI  /v1/subscribers/{id}/scores, /v1/subscribers/{id}/next-best-action, /v1/score (on demand), /v1/campaigns/simulate
   Dash CVM console  (simulator, segments, customer card, model health)
   Retention Analyst (Microsoft Agent Framework) --MCP--> FastMCP tools --> FastAPI
   Prometheus / Grafana / LangSmith
```

### 4.1 Scale design for 100M subscribers

- Iceberg tables are partitioned by `bucket(64, subscriber_id)` and `snapshot_date`. Training reads a pinned Iceberg snapshot id, which is logged to W&B.
- Batch scoring: Spark `mapInPandas` with the LightGBM booster broadcast once per executor, in batches of about 50k rows per Arrow batch. Estimated work: 100M rows x 3 models. Measure rows/sec per core on the 10M replica and write the cluster sizing formula into `docs/capacity.md` (cores needed = rows x models / (rows/sec/core x time window)).
- SHAP is costly. Compute reason codes only for subscribers selected for an action (usually 1 to 5% of the base), plus a 0.1% random sample for monitoring.
- Online store: Redis hash per subscriber (`scores`, `action`, `reasons`, `model_version`), with a TTL of 8 days. Size estimate for 100M keys goes in `docs/capacity.md`.
- The API never calls Spark. It reads Redis first; the on-demand path (`POST /v1/score` with a feature payload) runs the in-process model for new or changed subscribers.

---

## 5. Repository layout

```
orange-propensity-engine/
  ARCHITECTURE.md          this file
  README.md                results first (real metrics), how to run, limits, disclaimer
  DECISIONS.md             every choice the brief left open, one line each with the reason
  CHECKPOINT.md            current state, updated at the end of every phase
  pyproject.toml           single package "ope", uv managed, optional groups: spark, h2o, agent, dev
  .env.example
  docker-compose.yml       minio, redis, spark (master + worker), airflow (standalone), api, console, prometheus, grafana
  configs/
    base.yaml              paths, seeds, split ratios, model search spaces
    business.yaml          ARPU, offer costs, response rates, margins, budget, channel capacity (ALL marked as assumptions)
    feast/                 feature_store.yaml + feature definitions
  src/ope/
    data/          ingest.py, contracts.py (Pandera), split.py, iceberg.py
    features/      prep.py (missing indicators, rare levels, encodings), spark_features.py, feast_repo.py
    models/        baseline.py, gbm.py, h2o_automl.py, tune.py (Optuna), calibrate.py, gate.py, registry.py (W&B)
    evaluation/    metrics.py (AUC, PR-AUC, lift, Brier, ECE, profit curve), bootstrap.py, report.py
    explain/       shap_reasons.py
    policy/        value.py (expected value per action), allocate.py (greedy), optimal_check.py (CP-SAT)
    scoring/       batch_spark.py, online.py (Redis writer and reader)
    monitoring/    nannyml_jobs.py, psi.py, alerts.py
    api/           main.py, routes/, auth.py, schemas.py, telemetry.py
    console/       app.py (Dash), pages/
    analyst/       agent.py (Microsoft Agent Framework), mcp_server.py (FastMCP), prompts/, evals/
  dags/            ope_weekly.py, ope_retrain.py
  scripts/         download_data.py, make_scale_replica.py, bench_spark.py, run_local_pipeline.py
  tests/           unit/, integration/, contracts/, policy/, api/
  load/            locustfile.py
  deploy/
    helm/ope/      api, console, analyst, scoring CronJob / SparkApplication, values-local.yaml, values-aws.yaml
    terraform/     modules: network, eks, s3, ecr, redis, iam; envs/dev
    kind/          kind-config.yaml + smoke script
  docs/            capacity.md, model_card.md, data_card.md, runbook.md, images/
  .github/workflows/ ci.yml
```

---

## 6. ML methodology (the core: get this right first)

### 6.1 Splits

- Fixed seed. Stratify on the combined label pattern (churn, appetency, upselling).
- 70% train (5-fold stratified CV inside it for model selection and tuning), 10% calibration, 20% test.
- The test split is opened **once**, after the champion is frozen. Write a guard in code: the test loader refuses to run unless `configs/base.yaml` has `final_evaluation: true`, and it logs the call to W&B.

### 6.2 Feature preparation (fit inside each CV fold, never on full data)

- Drop features that are constant or more than 99% missing in the training fold; log the list.
- Add a missing-indicator for numeric features with 5 to 99% missing.
- Numeric: leave NaN for GBMs; median impute plus scaling only for the logistic baseline.
- Categoricals: group rare levels (count under 20 in the fold) into `__rare__`; unseen levels go to `__unseen__`. Compare, by CV AUC:
  a) LightGBM native categorical handling (`min_data_per_group`, `cat_smooth` tuned);
  b) out-of-fold target encoding with smoothing (fitted within folds) plus frequency encoding;
  c) CatBoost ordered target statistics.
  Keep the best per model family and record the comparison table in the README.
- Feature selection: drop features whose permutation importance in CV is not above zero in at least 4 of 5 folds. Report AUC before and after; keep it only if AUC does not fall.

### 6.3 Models

For each of the three targets separately:

1. Logistic regression baseline (L2, class weights).
2. LightGBM, XGBoost, CatBoost, each tuned with Optuna (about 100 trials for the main candidate, with pruning; fewer for the others), optimising mean CV AUC.
3. H2O AutoML challenger (`max_runtime_secs` around 900 per target, `sort_metric="AUC"`), needs Java 17.
4. Optional stacking: a logistic meta-model over out-of-fold predictions. Keep it only if it beats the best single model by more than the CV standard error.

Class imbalance: use class weights or `scale_pos_weight` and compare with no weighting. Do **not** oversample (SMOTE) before calibration, because it distorts probabilities. imbalanced-learn is used only for balanced metrics in the evaluation report.

Cross-target signal: test whether adding the out-of-fold churn score as a feature helps the up-sell model (and so on). Use out-of-fold predictions only; otherwise it leaks.

### 6.4 Calibration and gates

- Fit isotonic and Platt on the calibration split; pick by Brier score; report ECE and reliability plots before and after.
- Promotion gate (`models/gate.py`), all must pass:
  - CV AUC is at least 0.01 above the logistic baseline;
  - CV AUC is not below the current champion minus one standard error;
  - ECE on calibration is at most 0.02;
  - no fold has AUC more than 0.03 below the mean (stability).
- Passing models get the W&B registry alias `challenger`; promoting to `champion` is a separate, logged step.

### 6.5 Evaluation report (`reports/evaluation.md`, generated)

Per target: ROC-AUC with a 95% bootstrap CI (1,000 resamples), PR-AUC, lift and capture at deciles 1 to 3, precision at the top 5%, Brier, ECE, reliability plot, SHAP global importance, and the model comparison table (baseline, each GBM, H2O, stack).

External reference: look up the official results in Guyon et al., "Analysis of the KDD Cup 2009" (JMLR Workshop and Conference Proceedings, volume 7). Put the winners' per-target test AUCs in the report with the citation, and a note that our split differs, so the comparison is approximate. **Do not write any reference number from memory.** If the paper cannot be reached, leave the cell as "not verified".

### 6.6 Uplift (designed, not trained)

`policy/uplift.py` defines the interface for a two-model or X-learner uplift estimator (scikit-uplift or causalml) plus Qini evaluation. Its tests use a clearly synthetic randomised treatment. The README explains that with real campaign A/B data it replaces "propensity x assumed response rate" in the policy.

---

## 7. Next-best-action policy

### 7.1 Expected value per subscriber and action (`configs/business.yaml` holds every number, each marked as an assumption)

```
EV_retain   = p_churn  * r_retain * CLV_saved      - cost_retain
EV_crosssell = p_app   * r_cross  * margin_new     - cost_cross
EV_upsell   = p_up     * r_up     * delta_ARPU * H - cost_up
EV_none     = 0
```

`r_*` are response rates given contact, `H` is the margin horizon in months, and `CLV_saved` is the monthly margin times the expected extra tenure. Defaults should be plausible for an Egyptian mobile operator and clearly labelled as illustrative.

### 7.2 Allocation

- Constraints: at most one action per subscriber, a total budget, capacity per channel (SMS, call centre, app push), a suppression list (contacted in the last N days, opted out), and minimum EV above 0.
- Algorithm: score every (subscriber, action) pair, keep the best action per subscriber, sort by EV per unit cost, fill until budget or capacity runs out. This is O(n log n) and runs in Spark at 100M.
- Optimality check (`policy/optimal_check.py`): on a 50k sample, solve the same problem exactly with OR-Tools CP-SAT and report the greedy gap. Target: under 1%. If it is larger, document why and consider a Lagrangian-relaxation price on the budget.
- Outputs: `gold.actions` (subscriber_id, action, EV, reason codes, model versions, policy version), a campaign CSV for the CVM tool, and the profit curve (EV against budget) for the console.

---

## 8. Serving

### 8.1 Batch (weekly DAG `ope_weekly`)

`ingest -> contract check -> features -> feast materialize -> score (3 models) -> calibrate -> policy -> SHAP for actioned -> write Iceberg + Redis -> NannyML -> report`

Each task is idempotent and keyed by `snapshot_date`. Failure of the contract check stops the run and alerts.

### 8.2 Online API (FastAPI)

| Endpoint | Purpose | Target |
|---|---|---|
| `GET /v1/subscribers/{id}/scores` | precomputed calibrated scores plus model version | p95 < 50 ms |
| `GET /v1/subscribers/{id}/next-best-action` | action, EV, top 5 reasons | p95 < 50 ms |
| `POST /v1/score` | on-demand scoring from a feature payload (validated by the same Pandera schema) | p95 < 150 ms |
| `POST /v1/campaigns/simulate` | budget and costs in, expected value and contacts per action out | < 2 s on 50k |
| `GET /healthz`, `/readyz`, `/metrics` | probes and Prometheus | |

Requirements: OAuth2 client-credentials JWT validation (local: static API key), per-client rate limit, request ids, OpenTelemetry traces, structured JSON logs without feature values, and the model version in every response.

### 8.3 Later extension (do not build now)

Kafka topic `subscriber-events` with a consumer that re-scores a subscriber on trigger events and updates Redis. Only describe it in `docs/runbook.md`.

---

## 9. Retention Analyst (LLM, small and strict)

Purpose: a CVM manager asks "why did predicted churn rise in segment X?", "what if the budget is 2M EGP and SMS costs 0.3?", "explain subscriber 123". The agent answers in English or Arabic.

- Built with **Microsoft Agent Framework** (Python). Verify the current package name, the API and the Anthropic connector before coding.
- Tools are exposed by a **FastMCP** server that calls the FastAPI service: `get_segment_summary`, `explain_subscriber`, `simulate_campaign`, `get_model_health`, `get_drift_report`.
- Rule: **the model never computes or invents numbers.** Every figure in a reply must come from a tool result in that turn. A post-check compares numbers in the reply with the tool outputs and appends a warning if any cannot be traced.
- Tracing and evaluation in **LangSmith**: send traces through OpenTelemetry (verify LangSmith's OTel ingestion setup). A dataset of 20 gold questions (10 English, 10 Arabic) with expected tool calls; evaluators for tool-choice accuracy, numeric grounding (must be 100%) and refusal of out-of-scope requests.
- Keep it under about 400 lines. The ML system is the product; the analyst is the interface.

---

## 10. CVM console (Plotly Dash)

Pages: Campaign simulator (budget and cost sliders, profit curve, contacts per action, per-channel load); Segments (score distributions, lift); Subscriber card (scores, action, reason codes); Model health (CV and test metrics, calibration, NannyML estimated AUC, drift).

Design rules from the owner (must follow):
- Fonts: GFS Didot for headings and display numbers; Times New Roman Bold for labels and data. Never JetBrains Mono or any monospace or letter-spaced label style.
- No neon, glow or "matrix AI" styling. Calm, plain palette. Secondary text must stay clearly readable (meet WCAG AA contrast).

---

## 11. Monitoring and retraining

- NannyML: univariate drift (Jensen-Shannon for categoricals, Kolmogorov-Smirnov for numerics) and multivariate drift (PCA reconstruction error) per scoring run, plus CBPE estimated AUC for each target.
- PSI on the top 20 features and on the score distribution, computed by our own `psi.py` so it is unit-tested.
- Demonstration: `scripts/simulate_drift.py` builds clearly labelled synthetic shifted snapshots (for example, shift the top 3 features) to show alerts and the retrain trigger working.
- Retrain triggers: estimated AUC falls by more than 0.02, PSI above 0.2 on any top-10 feature, or 8 weeks since the last training. The trigger opens the `ope_retrain` DAG, which produces a `challenger`; promotion still needs the gate.
- Prometheus metrics: request latency, error rate, Redis hit rate, batch rows scored, drift flags, estimated AUC. Grafana dashboard JSON goes in `deploy/`.

---

## 12. Verification: what proves it right (not the implementer's word)

| Claim | Proof |
|---|---|
| Model quality | `reports/evaluation.md` generated from the frozen test split, with bootstrap CIs, plus the KDD 2009 paper reference |
| No leakage | `tests/integration/test_leakage.py`: features fitted on a fold never see its validation rows; the test loader guard; shuffled-label sanity check (AUC must fall to about 0.5) |
| Calibration | reliability plots and ECE before and after |
| Policy optimality | CP-SAT gap report on 50k |
| Policy correctness | property tests (Hypothesis): never more than one action per subscriber, never over budget or capacity, suppressed subscribers never contacted |
| Data contracts | Pandera schemas run in CI on a fixture and in the DAG on real data |
| Scale | `scripts/bench_spark.py` on the 10M replica: rows/sec per core, then the 100M sizing in `docs/capacity.md` |
| API latency | Locust run: p50, p95, p99 at 200 concurrent users, saved in `docs/` |
| Agent grounding | LangSmith eval results: numeric grounding 100% on the gold set |
| Infra | `helm lint`, kind smoke test (`helm install` then `/readyz`), `terraform fmt -check`, `terraform validate`, `tflint` in CI |
| Reproducibility | a fresh clone plus `uv sync` plus `python scripts/run_local_pipeline.py` reproduces the metrics within the bootstrap CI |

Every number in the README must come from a generated report file. No placeholder or invented metrics anywhere.

---

## 13. Build phases (each ends with tests passing and CHECKPOINT.md updated)

| Phase | Scope | Done when |
|---|---|---|
| P0 Skeleton | `uv` project, ruff, pyright, pytest, pre-commit, CI, docker-compose (MinIO, Redis), config loading | CI green on an empty pipeline |
| P1 Data | download, manifest hashes, row-identity check, join labels, Pandera contracts, Iceberg bronze/silver (Spark local) | contract tests pass; data card written |
| P2 Baselines and prep | split with test guard, fold-safe prep, logistic baselines, metrics module, W&B logging | baseline CV AUCs logged |
| P3 GBMs and tuning | LightGBM, XGBoost, CatBoost, Optuna, encoding comparison, feature selection | comparison table generated |
| P4 Challenger, calibration, gate | H2O AutoML, stacking test, calibration, gate, W&B registry aliases | champion per target frozen |
| P5 Final evaluation | one-time test run, bootstrap CIs, SHAP, evaluation report, model card, KDD paper reference | `reports/evaluation.md` complete |
| P6 Policy | EV, greedy allocation, CP-SAT check, property tests, profit curve | gap report and tests pass |
| P7 Batch at scale | Spark features and scoring, Feast plus Redis materialisation, scale replica, benchmark, capacity doc | benchmark numbers in `docs/capacity.md` |
| P8 Monitoring | NannyML jobs, PSI, drift simulation, alerts, retrain DAG, Airflow DAGs | weekly DAG runs end to end locally |
| P9 API | endpoints, auth, rate limits, OTel, Prometheus, Locust | latency targets met or the gap explained |
| P10 Console | Dash pages per section 10 | screenshots in `docs/images/` |
| P11 Analyst | Agent Framework agent, FastMCP tools, grounding check, LangSmith evals | grounding 100% on gold set |
| P12 Deploy | Dockerfiles, Helm chart, kind smoke, Terraform modules, Grafana dashboard | CI infra checks green |
| P13 README and polish | results first, architecture image, how to run, limits, disclaimer | owner review |

P0 to P6 are the priority: they are what a Senior Data Scientist interview will probe. If time is short, finish P0 to P6 and P9 to a high standard before the rest.

---

## 14. Environment notes (owner's machine)

- Windows 10, i7-11370H, 32 GB RAM, RTX 3050 4 GB (not needed: all models run on CPU).
- Spark on Windows is fragile (winutils). Run Spark through Docker (`docker-compose` spark master and worker) or WSL2; Python code reaches it with `SPARK_REMOTE` / `spark://`. Keep a Polars path for the 50k training so modelling never waits on Spark.
- H2O needs Java 17. If Java is missing, ask before installing it.
- W&B: use `WANDB_MODE=offline` until the owner provides a login; never block on it.
- LLM keys: read from `.env` (`ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`, `LANGSMITH_API_KEY`). If missing, the analyst runs with a scripted fake model for tests and the README says the live run is pending.

---

## 15. Rules for the implementing session

- Write simple, readable code a senior human would write. No speculative abstractions, no framework layers that are not used.
- Never invent numbers, results, benchmarks or citations. If something was not run, say so.
- Do not run `terraform apply`, create cloud resources, or spend money.
- Git: `git init` locally. Commits are authored by **Mrwan Awad** only, with **no** Claude or AI attribution lines or co-author trailers. Short, plain commit messages.
- **Ask first** before: creating the GitHub remote, pushing, installing system software (Java, WSL), or any paid service. Proposed repo name (follow his convention, confirm before creating): `Orange-Propensity-Engine-using-PySpark-LightGBM-Feast-WandB-Kubernetes-Terraform`, public.
- Update `CHECKPOINT.md` at the end of each phase: what is done, real metrics so far, what is next, open problems.
- If a tool in section 3.2 turns out to be broken, unmaintained or incompatible (check current versions), pick the closest alternative, record the reason in `DECISIONS.md`, and continue.
