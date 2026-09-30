"""Indexer tests against a temp Chroma directory and a fake embedder.

The real BGE model is not loaded. Token limits, metadata, and the query-prefix
rule are checked on fixture chunks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.constants import BGE_MODEL_NAME, BGE_QUERY_PREFIX, CHROMA_COLLECTION_NAME, CORPUS
from src.ingestion.chunker import Chunk
from src.ingestion.fetcher import ManifestRow
from src.ingestion.indexer import (
    METADATA_FIELDS,
    ChunkTooLongError,
    chroma_client,
    chunk_fingerprint,
    index_chunks,
    l2_normalize,
)
from src.ingestion.pipeline import StaleCorpusError, assert_corpus_fresh, build_index

DOC = next(document for document in CORPUS if document.document_id == "cold-food-storage")
CHICKEN = (
    "Food: Fresh poultry | Type: Chicken or turkey, whole | "
    "Refrigerator [40°F (4°C) or below]: 1 to 2 days | "
    "Freezer [0°F (-18°C) or below]: 1 year"
)


class FakeEmbedder:
    """Records the strings it embeds and returns distinct non-unit vectors."""

    def __init__(self, dimension: int = 8, tokens: dict[str, int] | None = None) -> None:
        self.model_name = BGE_MODEL_NAME
        self.dimension = dimension
        self.tokens = tokens or {}
        self.embedded: list[str] = []
        self.counted: list[str] = []

    def count_tokens(self, text: str) -> int:
        self.counted.append(text)
        return self.tokens.get(text, 10)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        vectors = []
        for index, text in enumerate(texts):
            raw = [0.0] * self.dimension
            raw[index % self.dimension] = 1.0 + (len(text) % 5)
            raw[(index + 1) % self.dimension] = 0.5
            vectors.append(raw)
        return vectors


def _chunk(
    ordinal: int,
    text: str,
    *,
    block_type: str = "table_row",
    document_id: str = DOC.document_id,
    embed_text: str | None = None,
) -> Chunk:
    document = next(item for item in CORPUS if item.document_id == document_id)
    heading = "Cold Food Storage Chart"
    return Chunk(
        chunk_id=f"{document.document_id}:{ordinal}",
        document_id=document.document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        retrieval_date="2026-09-27",
        section_heading=heading,
        block_type=block_type,
        text=text,
        embed_text=embed_text or f"{document.document_name} — {heading}\n{text}",
    )


def _stale_row() -> ManifestRow:
    return ManifestRow(
        document_id="eatwell-guide",
        document_name="The Eatwell Guide booklet",
        publisher="Public Health England (now OHID)",
        year=2018,
        source_url="https://example.invalid/eatwell",
        downloaded_url=None,
        retrieval_date="2026-09-27",
        sha256=None,
        http_status=None,
        status="stale",
        file=None,
    )


def test_metadata_round_trip_and_query_prefix_absent(tmp_path: Path):
    embedder = FakeEmbedder()
    chicken = _chunk(24, CHICKEN)
    intro = _chunk(0, "Freezer storage times are for quality only.", block_type="paragraph")
    result = index_chunks([intro, chicken], index_dir=tmp_path, embedder=embedder)

    assert result.skipped is False
    assert result.manifest.model_name == BGE_MODEL_NAME
    assert result.manifest.dimension == embedder.dimension
    assert result.manifest.max_sequence_length == 512
    assert result.manifest.max_tokens == 10
    assert result.manifest.chunk_count == 2
    assert result.manifest.fingerprint == chunk_fingerprint([intro, chicken])
    assert result.manifest.embedded_at.endswith("+00:00")

    assert embedder.embedded == [intro.embed_text, chicken.embed_text]
    assert all(not text.startswith(BGE_QUERY_PREFIX) for text in embedder.embedded)

    collection = chroma_client(tmp_path).get_collection(CHROMA_COLLECTION_NAME, embedding_function=None)
    assert collection.metadata["hnsw:space"] == "cosine"
    assert collection.count() == 2
    stored = collection.get(ids=["cold-food-storage:24"], include=["metadatas", "documents", "embeddings"])
    assert stored["documents"] == [CHICKEN]
    metadata = stored["metadatas"][0]
    assert set(metadata) == set(METADATA_FIELDS)
    assert "embed_text" not in metadata
    assert type(metadata["year"]) is int
    assert metadata["year"] == DOC.year
    assert metadata["block_type"] == "table_row"
    assert metadata["document_id"] == "cold-food-storage"
    assert metadata["source_url"] == DOC.source_url
    assert "1 to 2 days" in stored["documents"][0]
    assert "1 year" in stored["documents"][0]
    embedding = stored["embeddings"][0]
    norm = sum(value * value for value in embedding) ** 0.5
    assert norm == pytest.approx(1.0)


def test_passage_over_512_tokens_is_not_upserted(tmp_path: Path):
    chunk = _chunk(0, "too long")
    embedder = FakeEmbedder(tokens={chunk.embed_text: 513})

    with pytest.raises(ChunkTooLongError) as exc_info:
        index_chunks([chunk], index_dir=tmp_path, embedder=embedder)

    assert exc_info.value.offenders == [("cold-food-storage:0", 513)]
    assert embedder.embedded == []
    assert not (tmp_path / "chroma.sqlite3").exists()
    assert list(tmp_path.glob("index_manifest.json")) == []


def test_second_run_with_the_same_chunks_does_not_reembed(tmp_path: Path):
    embedder = FakeEmbedder()
    chunks = [_chunk(24, CHICKEN)]
    first = index_chunks(chunks, index_dir=tmp_path, embedder=embedder)
    second = index_chunks(chunks, index_dir=tmp_path, embedder=embedder)

    assert first.skipped is False
    assert second.skipped is True
    assert second.manifest == first.manifest
    assert embedder.embedded == [chunks[0].embed_text]
    assert embedder.counted == [chunks[0].embed_text]


def test_changed_embed_text_reembeds(tmp_path: Path):
    embedder = FakeEmbedder()
    original = _chunk(24, CHICKEN)
    index_chunks([original], index_dir=tmp_path, embedder=embedder)
    updated = _chunk(24, CHICKEN, embed_text=original.embed_text + " updated")
    result = index_chunks([updated], index_dir=tmp_path, embedder=embedder)

    assert result.skipped is False
    assert embedder.embedded[-1] == updated.embed_text
    assert chroma_client(tmp_path).get_collection(CHROMA_COLLECTION_NAME, embedding_function=None).count() == 1


def test_l2_normalize_rejects_a_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize([[0.0, 0.0, 0.0]])


def test_stale_corpus_refuses_to_index(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.ingestion.pipeline.fetch_corpus", lambda **kwargs: [_stale_row()])

    def fail_parse():
        raise AssertionError("parse should not run when the corpus is stale")

    monkeypatch.setattr("src.ingestion.pipeline.parse_corpus", fail_parse)
    with pytest.raises(StaleCorpusError) as exc_info:
        build_index()
    assert exc_info.value.document_ids == ["eatwell-guide"]


def test_assert_corpus_fresh_accepts_ok_rows():
    row = _stale_row()
    row.status = "ok"
    assert_corpus_fresh([row])
