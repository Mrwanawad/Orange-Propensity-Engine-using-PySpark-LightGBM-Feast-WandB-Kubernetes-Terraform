"""Tiny champion bundles trained on the synthetic test fixture, laid out the way the API expects,
so the kind smoke test checks the chart and the image without the real 50k data.
NOT real models: never use their scores for anything.

    python deploy/kind/smoke_artifacts.py /tmp/ope-kind
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for tests/

from tests.api.test_api import tiny_bundle  # noqa: E402
from tests.conftest import make_fixture  # noqa: E402

from ope.data.contracts import TARGETS  # noqa: E402


def main(root: Path) -> None:
    models = root / "artifacts" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df = make_fixture(1500, seed=4)
    reg = {}
    for t in TARGETS:
        path = models / f"{t}-smoke.pkl"
        path.write_bytes(pickle.dumps(tiny_bundle(df, t)))
        reg[t] = {"champion": {"path": f"artifacts/models/{path.name}", "version": path.stem}}
    (models / "registry.json").write_text(json.dumps(reg, indent=2))
    print(f"smoke artifacts in {root}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
