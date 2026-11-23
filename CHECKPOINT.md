# Checkpoint

Updated: 2026-11-21 (in progress)

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

## Running

- Optuna tuning: 8/9 studies in `reports/cv/tuned.json`. Parallel run on 2026-11-21 was cut
  by a restart at 19:14 and starved CPU (1-3 trials per study). Re-tuning all but
  churn/lightgbm sequentially with `--fresh`, then `select`, `crosstarget`, `train freeze`.

## Next

P4 freeze + H2O, P5 test report, P6 policy gap, P7 Spark benchmark, P8 monitoring demo, P9
Locust, P10 screenshots, P11 analyst eval, README.
