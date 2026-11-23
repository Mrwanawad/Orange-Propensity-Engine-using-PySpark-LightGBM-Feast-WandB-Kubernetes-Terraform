# Checkpoint

Updated: 2026-11-23 04:00 (in progress)

## Done

- P0 skeleton: uv project, ruff, pyright, pytest, pre-commit, docker-compose, CI workflow.
- P1 data: OpenML download + manifest hashes, row-identity check, label join, Pandera contracts,
  Iceberg bronze/silver written through Spark (silver snapshot pinned), data card.
- P2 baselines: split with test guard, fold-safe prep, logistic baselines (CV AUC churn 0.7092,
  appetency 0.7978, up-selling 0.7988), W&B offline runs.
- P3 encoding comparison complete (`reports/cv/encodings.json`).
- Synthetic 10M replica built (`data/replica/`, 20 parts).
- Analyst gold set (20 questions) and eval runner; unit tests for grounding and routing.
- kind smoke test, Locust file, docs (data card, runbook), DECISIONS.md.
- Git: local repo, first commit 2026-11-23. No remote yet (name in ARCHITECTURE.md section 15,
  needs owner's yes before creating).

## Running

- `reports/cv/chain.sh` (detached, started 2026-11-23 03:39): fresh re-tune of appetency and
  upselling, then `select`, `crosstarget`, `train freeze`. `CHAIN_DONE` in
  `reports/cv/chain.out` marks the end (ETA about 07:00).
- Churn studies are kept as they are: lightgbm 89 trials, xgboost 18, catboost 3 (CatBoost hits
  its 20-minute wall-clock cap at about 400 s per trial).

## Next (in order)

1. H2O AutoML in the jobs image (`docker compose run --rm jobs python -m ope.models.h2o_automl`).
2. `train promote`, set `final_evaluation: true`, `evaluation.report` (P5), model card.
3. `batch_local`, `optimal_check` (P6 gap), `simulate_drift`, `weekly_report` (P8).
4. `bench_spark.py` on the replica in the jobs image, `docs/capacity.md` (P7).
5. Compose app stack, Locust at 200 users (P9), console screenshots (P10), analyst eval (P11).
6. README (P13).

## Open problems

- Host has no Java, helm or kind; Spark and H2O run in the `ope-jobs` image. Docker Desktop must
  be running.
