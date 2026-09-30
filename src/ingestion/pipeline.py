"""Offline indexing pipeline: fetch, parse, chunk, then embed.

A corpus row that is not ``ok`` stops the pipeline before parsing and before
any embedding. The index is never built from a partial or stale download.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.ingestion.chunker import chunk_parsed, write_chunks
from src.ingestion.fetcher import ManifestRow, fetch_corpus
from src.ingestion.indexer import IndexBuild, index_chunks
from src.ingestion.parser import parse_corpus, write_parsed

logger = logging.getLogger(__name__)


class StaleCorpusError(Exception):
    """The corpus manifest has a row that is not fresh. Indexing did not start."""

    def __init__(self, document_ids: Sequence[str]) -> None:
        self.document_ids = list(document_ids)
        super().__init__(
            "Refusing to index because these corpus documents are not fresh: "
            + ", ".join(self.document_ids)
        )


def assert_corpus_fresh(rows: Sequence[ManifestRow]) -> None:
    """Raise :class:`StaleCorpusError` when any manifest row is not ``ok``."""
    stale = [row.document_id for row in rows if row.status != "ok"]
    if stale:
        raise StaleCorpusError(stale)


def build_index() -> IndexBuild:
    """Fetch the registry, rebuild chunks, and upsert the Chroma collection."""
    logger.info("Fetching corpus")
    rows = fetch_corpus(strict=False)
    assert_corpus_fresh(rows)

    logger.info("Parsing corpus")
    parsed = parse_corpus()
    write_parsed(parsed)

    logger.info("Chunking corpus")
    chunks = chunk_parsed()
    write_chunks(chunks)

    logger.info("Indexing %d chunks", len(chunks))
    return index_chunks(chunks)
