"""CLI entry point for phase 2, task 2: chunk parsed blocks for retrieval.

Reads ``data/parsed/blocks.jsonl`` and ``data/corpus_manifest.json``, writes
``data/chunks/chunks.jsonl``, and prints a per-document summary.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CORPUS  # noqa: E402
from src.ingestion.chunker import chunk_parsed, write_chunks  # noqa: E402


def main() -> None:
    try:
        chunks = chunk_parsed()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    path = write_chunks(chunks)

    print(f"\nChunks:  {path}  ({len(chunks)} total)\n")
    print(f"{'document':<24} {'chunks':>7} {'para':>6} {'list':>6} {'table':>6} {'row':>6}")
    by_doc: dict[str, list] = {document.document_id: [] for document in CORPUS}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk)
    for document_id, rows in by_doc.items():
        counts = Counter(chunk.block_type for chunk in rows)
        print(
            f"{document_id:<24} {len(rows):>7} {counts.get('paragraph', 0):>6} "
            f"{counts.get('list', 0):>6} {counts.get('table', 0):>6} {counts.get('table_row', 0):>6}"
        )


if __name__ == "__main__":
    main()
