"""The persistent Chroma client, on its own so the answer path stays light.

``src.rag.retriever`` needs only this function. Importing it from
``src.ingestion.indexer`` pulled in the chunker and the parser, and with them
``pymupdf`` and ``beautifulsoup4``, which the deployed page does not install
(docs/deployment-plan.md, phase D2). This module depends on ``chromadb`` and
``src.constants`` and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from src.constants import INDEX_DIR


def chroma_client(index_dir: Path | str = INDEX_DIR) -> ClientAPI:
    """Open (or create) the persistent collection store under ``index_dir``."""
    return chromadb.PersistentClient(
        path=str(index_dir),
        settings=Settings(anonymized_telemetry=False),
    )
