"""Build clearly SYNTHETIC weekly snapshots to show drift alerts and the retrain trigger.

week_1_stable   bootstrap resample of the calibration split, no change
week_2_shifted  same, with the top 3 numeric features shifted by +1.5 standard deviations
                and a third of one top categorical's values replaced by a new level

Labels are never used. Output: reports/monitoring/synthetic_week_*.json
    python scripts/simulate_drift.py
"""

from __future__ import annotations

import logging

import numpy as np

from ope.data.contracts import CATEGORICAL
from ope.data.split import load_split
from ope.models import registry
from ope.monitoring import alerts, nannyml_jobs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(7)
    cal = load_split("calibration")
    trained_at = registry.load_registry()["churn"]["champion"]["set_at"]
    feats = nannyml_jobs.top_features(20)
    nums = [f for f in feats if f not in CATEGORICAL][:3]
    cats = [f for f in feats if f in CATEGORICAL][:1]

    stable = cal.sample(len(cal), replace=True, random_state=1).reset_index(drop=True)
    shifted = cal.sample(len(cal), replace=True, random_state=2).reset_index(drop=True)
    for f in nums:
        shifted[f] = shifted[f] + 1.5 * cal[f].std()
    for f in cats:
        m = rng.uniform(size=len(shifted)) < 0.33
        shifted.loc[m, f] = "SYNTHETIC_NEW_LEVEL"

    for label, df in (("synthetic_week_1_stable", stable), ("synthetic_week_2_shifted", shifted)):
        rep = nannyml_jobs.run(df, label, trained_at=trained_at)
        rep["synthetic"] = {"shifted_numeric": nums if "shifted" in label else [], "shifted_categorical": cats if "shifted" in label else []}
        alerts.emit(rep)
        print(label, rep["decision"])


if __name__ == "__main__":
    main()
