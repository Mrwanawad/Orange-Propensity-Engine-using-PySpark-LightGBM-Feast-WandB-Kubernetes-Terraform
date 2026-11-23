"""Download the three KDD Cup 2009 (small) datasets from OpenML, record SHA-256 hashes in
data/raw/MANIFEST.json, check row identity and join the labels. Same as `python -m ope.data.ingest`.

    uv run python scripts/download_data.py
"""

from ope.data.ingest import main

if __name__ == "__main__":
    main()
