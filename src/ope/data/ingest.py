"""Download the three KDD Cup 2009 datasets from OpenML and join their labels.

Raw ARFF files are cached under data/raw and never edited. Their SHA-256, OpenML
metadata and licence go to data/raw/MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
import logging

import arff  # liac-arff, installed with openml
import httpx
import numpy as np
import pandas as pd

from ope import config

log = logging.getLogger(__name__)
API = "https://www.openml.org/api/v1/json/data/{id}"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> dict:
    raw = config.path("raw")
    manifest = {}
    for target, did in config.load()["openml_ids"].items():
        meta = httpx.get(API.format(id=did), timeout=60).json()["data_set_description"]
        out = raw / f"{meta['name']}.arff"
        if not out.exists():
            log.info("downloading %s", meta["url"])
            with httpx.stream("GET", meta["url"], timeout=600, follow_redirects=True) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        md5 = hashlib.md5(out.read_bytes()).hexdigest()
        if md5 != meta["md5_checksum"]:
            raise ValueError(f"{out.name}: md5 {md5} does not match OpenML {meta['md5_checksum']}")
        manifest[target] = {
            "openml_id": did,
            "name": meta["name"],
            "file": out.name,
            "url": meta["url"],
            "licence": meta.get("licence"),
            "version": meta.get("version"),
            "md5_openml": meta["md5_checksum"],
            "sha256": sha256(out),
            "target_attribute": meta["default_target_attribute"],
        }
    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_arff(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = arff.load(f)
    cols = [a[0] for a in data["attributes"]]
    df = pd.DataFrame(data["data"], columns=cols)
    # cast by the ARFF declared type: all-null numeric columns otherwise come back as object
    for name, kind in data["attributes"]:
        if isinstance(kind, str) and kind.upper() in ("NUMERIC", "REAL", "INTEGER"):
            df[name] = df[name].astype("float64")
        else:
            df[name] = df[name].astype("string")
    return df


def row_hashes(features: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(features, index=False).to_numpy()


def subscriber_id(i: int) -> str:
    # stable id from the row index; the dataset has no customer key
    return hashlib.sha1(f"kdd09-{i}".encode()).hexdigest()[:16]


def build_table() -> pd.DataFrame:
    """Check the three files share identical feature rows, then join the labels."""
    manifest = json.loads((config.path("raw") / "MANIFEST.json").read_text())
    features, labels, hashes = None, {}, {}
    for target, m in manifest.items():
        df = read_arff(config.path("raw") / m["file"])
        y = df.pop(m["target_attribute"])
        labels[target] = (y.astype(str).str.strip() == "1").astype("int8")
        hashes[target] = row_hashes(df)
        if features is None:
            features = df
    first = next(iter(hashes.values()))
    for target, h in hashes.items():
        if not np.array_equal(h, first):
            raise ValueError(f"feature rows of {target} differ from the others; refusing to join")
    assert features is not None
    out = features.copy()
    out.insert(0, "subscriber_id", [subscriber_id(i) for i in range(len(out))])
    for target, y in labels.items():
        out[target] = y.to_numpy()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download()
    table = build_table()
    out = config.path("interim") / "subscribers.parquet"
    table.to_parquet(out, index=False)
    log.info("wrote %s rows=%d cols=%d", out, *table.shape)


if __name__ == "__main__":
    main()
