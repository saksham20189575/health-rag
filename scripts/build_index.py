"""CLI entry point for the offline index build.

Fetches the seven documents, parses and chunks them, and embeds the chunks
into ``data/index/``. A document whose manifest row is not ``ok`` stops the
build before any embedding. A second run with the same chunks does not re-embed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.indexer import ChunkTooLongError  # noqa: E402
from src.ingestion.pipeline import StaleCorpusError, build_index  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = build_index()
    except (StaleCorpusError, ChunkTooLongError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    manifest = result.manifest
    action = "Unchanged" if result.skipped else "Indexed"
    print(
        f"{action}: {manifest.chunk_count} chunks, model {manifest.model_name}, "
        f"dimension {manifest.dimension}, max tokens {manifest.max_tokens}"
    )


if __name__ == "__main__":
    main()
