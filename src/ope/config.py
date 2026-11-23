"""Load the YAML configs once. Paths are resolved from the repo root."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@cache
def load(name: str = "base") -> dict[str, Any]:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    p = ROOT / load()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
