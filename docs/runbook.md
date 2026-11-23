# Runbook

## Weekly run (`ope_weekly`, Mondays 03:00)

```
ingest -> contract_check -> features -> feast_materialize -> score_policy_shap -> nannyml -> report
```

Every task is `python -m ...` in the `ope-jobs` image and is keyed by the logical date, so a rerun
overwrites that `snapshot_date` partition only (Iceberg `overwritePartitions`, Redis keys
overwritten with a fresh 8-day TTL).

| Task | Writes | If it fails |
|---|---|---|
| ingest | `data/raw/*` (cached), `data/interim/subscribers.parquet` | OpenML down: rerun later; cached raw files are reused when hashes match |
| contract_check | `lake.bronze.subscribers_raw`, `lake.silver.subscribers` | **Run stops.** Read the Pandera error in the task log (failing column and check). Do not bypass: fix upstream data or the contract, in review |
| features | `lake.gold.subscriber_features` / Feast offline parquet | Spark job log; idempotent, rerun |
| feast_materialize | Redis feature views (8-day TTL) | Redis memory or connectivity; rerun |
| score_policy_shap | `gold.scores`, `gold.actions`, campaign CSV, Redis `ope:sub:*` | Most often a missing `champion` in `artifacts/models/registry.json`; check `python -m ope.models.train promote` was run |
| nannyml | `reports/monitoring/weekly_<date>.json`, alert | See "Drift alert" below |
| report | `reports/weekly/<date>.md` | Pure formatting; rerun |

## Drift alert and retraining

`monitoring/alerts.py` requests a retrain when any of these is true:

- CBPE estimated AUC fell by more than 0.02 against the reference (calibration split);
- PSI above 0.2 on any of the top 10 features;
- more than 8 weeks since the champion was trained.

The alert triggers `ope_retrain`: split -> baseline -> tune (LightGBM, fresh study) -> freeze ->
promote. Freeze registers a **challenger**
only if the gate passes (CV AUC at least 0.01 over the logistic baseline, not below champion minus
one SE, calibration ECE at most 0.02, no fold more than 0.03 below the mean).

Promotion is a **separate human step**: the DAG's promote task is skipped unless the run conf has
`"promote": true`, or run it by hand:

```
python -m ope.models.train promote     # challenger -> champion, logged to W&B
```

Before promoting, read `reports/cv/freeze.json` (candidate, CV AUC and SE, calibration report,
gate checks). If the drift came from a broken upstream feed rather than real behaviour change, fix
the feed and do not retrain.

## API

| Symptom | Check |
|---|---|
| `/readyz` 503 | models not loaded: `artifacts/models/registry.json` missing or bundle path wrong |
| many 404 on `/scores` | Redis cache misses: the weekly batch did not run within 8 days (TTL), or Redis was flushed. Grafana panel "Redis hit rate" |
| 429 | per-client rate limit (`OPE_RATE_LIMIT_PER_MINUTE`, default 1200/min) |
| p95 above target | Grafana latency panel by route; `/v1/score` is model inference in process, scale replicas (HPA) |

Logs are JSON, one line per request with request id and client id, never feature values.

## Rollback

Champion bundles are immutable files (`artifacts/models/<target>-<UTC timestamp>.pkl`).
To roll back, point `champion` in `registry.json` at the previous bundle (the W&B artifact
history lists every version with its aliases) and restart the API pods. The next weekly run
rescores with it.

## Later extension: real-time triggers (not built)

A Kafka topic `subscriber-events` (top-up stopped for 7 days, port-out request, many dropped
calls) with a consumer that re-scores one subscriber on a trigger event and updates its Redis hash.
The consumer would reuse `ope.models.predict.score` and `ope.policy.value.expected_values`, read
features from the Feast online store, and respect the same suppression list. The weekly batch stays
the source of truth; events only refresh single subscribers between runs.
