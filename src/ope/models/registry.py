"""W&B tracking and the model registry.

W&B runs offline until a login is provided (WANDB_MODE=offline); `wandb sync wandb/` uploads
later. Registry aliases (challenger, champion) are logged as W&B artifact aliases and mirrored
in artifacts/models/registry.json so the scorer and API never need the W&B server.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ope import config

log = logging.getLogger(__name__)


@contextmanager
def run(name: str, job_type: str, cfg: dict[str, Any] | None = None, group: str | None = None):
    import wandb

    os.environ.setdefault("WANDB_MODE", config.load()["wandb"]["mode"])
    os.environ.setdefault("WANDB_SILENT", "true")
    r = wandb.init(
        project=config.load()["wandb"]["project"],
        name=name,
        job_type=job_type,
        group=group,
        config=cfg or {},
        dir=str(config.ROOT),
        reinit="finish_previous",
    )
    try:
        yield r
    finally:
        r.finish()


def registry_path() -> Path:
    return config.path("models") / "registry.json"


def load_registry() -> dict:
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_bundle(target: str, bundle: dict[str, Any]) -> Path:
    """bundle: prep, model, calibrator, metadata. Returns the file path."""
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = config.path("models") / f"{target}-{version}.pkl"
    bundle = {**bundle, "version": f"{target}-{version}"}
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    return out


def load_bundle(target: str, alias: str = "champion") -> dict[str, Any]:
    entry = load_registry()[target][alias]
    with open(config.ROOT / entry["path"], "rb") as f:
        return pickle.load(f)


def set_alias(target: str, alias: str, path: Path, metadata: dict[str, Any]) -> None:
    reg = load_registry()
    reg.setdefault(target, {})[alias] = {
        "path": str(path.relative_to(config.ROOT)).replace("\\", "/"),
        "version": path.stem,
        "set_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **metadata,
    }
    registry_path().write_text(json.dumps(reg, indent=2))
    try:
        import wandb

        with run(f"registry-{target}-{alias}", "registry", {"target": target, "alias": alias}) as r:
            art = wandb.Artifact(f"ope-{target}", type="model", metadata=metadata)
            art.add_file(str(path))
            r.log_artifact(art, aliases=[alias, path.stem])
    except Exception as e:
        log.warning("W&B artifact logging failed (registry.json still updated): %s", e)


def promote(target: str) -> None:
    """challenger -> champion. A separate, logged step (section 6.4)."""
    reg = load_registry()
    ch = reg[target]["challenger"]
    set_alias(
        target,
        "champion",
        config.ROOT / ch["path"],
        {k: v for k, v in ch.items() if k not in ("path", "version", "set_at")},
    )
    log.info("promoted %s %s to champion", target, ch["version"])
