"""CLI entry point for phase 2, task 1: parse the raw corpus into typed blocks.

Reads ``data/raw/<document_id>/source.<pdf|html>`` for every registry document,
writes ``data/parsed/blocks.jsonl`` and ``data/parsed/parse_report.json`` and
prints a per-document summary. Exits non-zero when a raw file is missing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parser import parse_corpus, write_parsed  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        parsed = parse_corpus()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    blocks_path, report_path = write_parsed(parsed)

    print(f"\nBlocks:  {blocks_path}\nReport:  {report_path}\n")
    print(f"{'document':<22} {'blocks':>6} {'head':>5} {'para':>5} {'list':>5} {'table':>5} {'pages':>9}  notes")
    for document_id, item in parsed.items():
        report = item.report
        counts = report.blocks_by_type
        notes = []
        if report.image_only_pages:
            notes.append(f"image-only pages {report.image_only_pages}")
        if report.skipped_sections:
            notes.append(f"skipped sections {report.skipped_sections}")
        if report.tables_rejected:
            notes.append(f"{report.tables_rejected} layout tables rejected")
        print(
            f"{document_id:<22} {len(item.blocks):>6} {counts.get('heading', 0):>5} "
            f"{counts.get('paragraph', 0):>5} {counts.get('list_item', 0):>5} {counts.get('table', 0):>5} "
            f"{report.pages_parsed:>4}/{report.pages_total:<4}  {'; '.join(notes)}"
        )


if __name__ == "__main__":
    main()
