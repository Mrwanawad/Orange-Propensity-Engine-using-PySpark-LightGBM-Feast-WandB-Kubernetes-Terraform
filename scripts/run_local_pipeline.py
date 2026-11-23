"""End-to-end local run on the 50k real rows: data -> CV -> tuning -> freeze -> test report ->
batch scoring and policy -> CP-SAT gap -> monitoring. Every stage is `python -m ...` so it is the
same command the Airflow DAGs run. Stages whose output already exists are skipped (resumable);
pass --fresh to rerun everything.

    uv run python scripts/run_local_pipeline.py                 # full run
    uv run python scripts/run_local_pipeline.py --quick         # 5 Optuna trials per study (smoke)
    uv run python scripts/run_local_pipeline.py --from freeze   # start at a stage

H2O AutoML needs Java 17, so it is not part of this run; use
`docker compose run --rm jobs python -m ope.models.h2o_automl`.
The test split opens only when configs/base.yaml has final_evaluation: true.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "reports" / "cv"

# (name, command, output that marks the stage as done)
STAGES = [
    ("ingest", ["ope.data.ingest"], "data/interim/subscribers.parquet"),
    ("split", ["ope.data.split"], "data/processed/splits.parquet"),
    ("baseline", ["ope.models.baseline"], "reports/cv/baseline.json"),
    ("encodings", ["ope.models.tune", "encodings"], None),
    ("tune", ["ope.models.tune", "tune"], None),
    ("select", ["ope.models.tune", "select"], None),
    ("crosstarget", ["ope.models.tune", "crosstarget"], None),
    ("freeze", ["ope.models.train", "freeze"], "reports/cv/freeze.json"),
    ("promote", ["ope.models.train", "promote"], None),
    ("evaluate", ["ope.evaluation.report"], "reports/evaluation.md"),
    ("batch", ["ope.scoring.batch_local", "--no-redis"], "data/processed/scores.parquet"),
    ("policy_gap", ["ope.policy.optimal_check", "--time-limit", "300"], "reports/policy_gap.json"),
    ("drift_demo", ["scripts/simulate_drift.py"], "reports/monitoring/synthetic_week_2_shifted.json"),
    ("weekly_report", ["ope.monitoring.weekly_report"], None),
]


def run(name: str, cmd: list[str], quick: bool) -> None:
    if name == "tune" and quick:
        cmd = [*cmd, "--trials", "5"]
    argv = [sys.executable, str(ROOT / cmd[0]), *cmd[1:]] if cmd[0].endswith(".py") else [sys.executable, "-m", *cmd]
    print(f"\n=== {name}: {' '.join(argv[1:])}", flush=True)
    t = time.time()
    subprocess.run(argv, cwd=ROOT, check=True)
    print(f"=== {name} done in {time.time() - t:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete CV reports and models first")
    ap.add_argument("--quick", action="store_true", help="5 Optuna trials per study")
    ap.add_argument("--from", dest="start", choices=[s[0] for s in STAGES])
    a = ap.parse_args()
    if a.fresh:
        shutil.rmtree(CV, ignore_errors=True)
        shutil.rmtree(ROOT / "artifacts" / "models", ignore_errors=True)
    names = [s[0] for s in STAGES]
    for name, cmd, done in STAGES[names.index(a.start) if a.start else 0 :]:
        if done and (ROOT / done).exists() and not a.fresh and name != a.start:
            print(f"=== {name}: skipped ({done} exists)")
            continue
        run(name, cmd, a.quick)


if __name__ == "__main__":
    main()
