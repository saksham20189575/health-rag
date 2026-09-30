"""CLI entry point for phase 1: download the corpus and write the manifest.

Exits non-zero, naming the affected document ids, when any row is not ``ok``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.fetcher import CorpusFetchError, Fetcher  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    fetcher = Fetcher()
    try:
        rows = fetcher.fetch_all(strict=True)
    except CorpusFetchError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"\nManifest: {fetcher.manifest_path}")
    for row in rows:
        print(
            f"{row.status:<5} {row.document_id:<22} HTTP {row.http_status} "
            f"retrieved {row.retrieval_date} sha256 {(row.sha256 or '')[:12]} {row.file}"
        )


if __name__ == "__main__":
    main()
