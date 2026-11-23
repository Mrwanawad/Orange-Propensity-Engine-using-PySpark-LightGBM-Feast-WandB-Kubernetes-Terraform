"""SYNTHETIC scale replica for throughput benchmarks only. Never train on it, never report model
metrics from it.

Resamples real rows with replacement, multiplies numeric values by 1 + N(0, 0.01) jitter, and
swaps 1% of categorical values for another level of the same column. New ids are prefixed 'syn'.

    python scripts/make_scale_replica.py --rows 10000000
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np

from ope import config
from ope.data.contracts import CATEGORICAL, ID, NUMERIC
from ope.data.split import load_table

OUT = config.ROOT / "data" / "replica"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000_000)
    ap.add_argument("--chunk", type=int, default=500_000)
    a = ap.parse_args()
    src = load_table().drop(columns=["churn", "appetency", "upselling"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SYNTHETIC_README.txt").write_text(
        "SYNTHETIC replica of the KDD Cup 2009 small dataset for throughput benchmarks only.\n"
        "Never train on it. Never report model metrics from it.\n"
    )
    levels = {c: src[c].dropna().unique() for c in CATEGORICAL}
    done = 0
    part = 0
    while done < a.rows:
        n = min(a.chunk, a.rows - done)
        out = OUT / f"part-{part:04d}.parquet"
        if out.exists():  # resumable: each part has its own seed
            done += n
            part += 1
            continue
        rng = np.random.default_rng([2009, part])
        idx = rng.integers(0, len(src), n)
        df = src.iloc[idx].reset_index(drop=True)
        df[ID] = [hashlib.sha1(f"syn-{done + i}".encode()).hexdigest()[:16] for i in range(n)]
        num = df[NUMERIC].to_numpy()
        df[NUMERIC] = num * (1 + rng.normal(0, 0.01, num.shape))
        for c in CATEGORICAL:
            if len(levels[c]) > 1:
                m = rng.uniform(size=n) < 0.01
                df.loc[m, c] = rng.choice(levels[c], m.sum())
        df.to_parquet(out, index=False)
        done += n
        part += 1
        print(f"{done:,} rows", flush=True)


if __name__ == "__main__":
    main()
