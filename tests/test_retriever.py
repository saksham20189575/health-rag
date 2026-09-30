"""Retriever tests against a temp Chroma index and, when built, the real one.

Fixture vectors are unit length, so cosine similarity with the query
``[1, 0, ...]`` equals the vector's first component. The real BGE model is
loaded only for the tests that read ``data/index``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from src.constants import (
    BGE_MODEL_NAME,
    BGE_QUERY_PREFIX,
    CANDIDATE_K,
    CORPUS,
    INDEX_DIR,
    MAX_CHUNKS_PER_DOCUMENT,
    MIN_SIMILARITY,
    TOP_K_ALL,
)
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import index_chunks
from src.rag.retriever import REGISTRY_IDS, Retriever

DIMENSION = 8
QUERY = [1.0] + [0.0] * (DIMENSION - 1)

IN_CORPUS = (
    "How long can I keep cooked leftovers in the fridge?",
    "How long can I keep eggs in the fridge?",
    "How much salt per day?",
    "What is the danger zone temperature?",
    "What do the guidelines say about cooking oil?",
    "What does WHO say about limiting free sugars?",
    "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
)
OFF_TOPIC = (
    "What is the tariff on imported olive oil?",
    "How do I refinance my mortgage?",
    "How do I write a for loop in Python?",
    "What is the weather in Paris tomorrow?",
)


class FakeQueryEmbedder:
    """Returns one fixed vector and records the string it was given."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = list(vector or QUERY)
        self.texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.texts.append(text)
        return list(self.vector)


class VectorEmbedder:
    """Passage embedder for ``index_chunks``. One vector per chunk, in order."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.model_name = BGE_MODEL_NAME
        self.dimension = len(vectors[0])
        self._vectors = vectors

    def count_tokens(self, text: str) -> int:
        return 10

    def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) != len(self._vectors):
            raise AssertionError(f"expected {len(self._vectors)} texts, got {len(texts)}")
        return self._vectors


def _unit(similarity: float) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[0] = similarity
    vector[1] = math.sqrt(1.0 - similarity * similarity)
    return vector


def _chunk(document_id: str, ordinal: int, text: str) -> Chunk:
    document = next(item for item in CORPUS if item.document_id == document_id)
    heading = document.document_name
    return Chunk(
        chunk_id=f"{document_id}:{ordinal}",
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        retrieval_date="2026-09-27",
        section_heading=heading,
        block_type="paragraph",
        text=text,
        embed_text=f"{document.document_name} — {heading}\n{text}",
    )


def _index(tmp_path: Path, rows: list[tuple[str, int, str, float]]) -> None:
    chunks = [_chunk(document_id, ordinal, text) for document_id, ordinal, text, _ in rows]
    vectors = [_unit(similarity) for *_, similarity in rows]
    index_chunks(chunks, index_dir=tmp_path, embedder=VectorEmbedder(vectors))


def _retriever(tmp_path: Path) -> tuple[Retriever, FakeQueryEmbedder]:
    embedder = FakeQueryEmbedder()
    return Retriever(tmp_path, embedder=embedder), embedder


def test_unknown_document_id_raises_before_chroma_or_embedder(tmp_path: Path):
    retriever, embedder = _retriever(tmp_path)

    with pytest.raises(ValueError, match="not in the registry"):
        retriever.retrieve("How long do leftovers keep?", document_id="not-a-document")

    assert embedder.texts == []
    assert not (tmp_path / "chroma.sqlite3").exists()


def test_filtered_query_returns_only_that_document(tmp_path: Path):
    _index(
        tmp_path,
        [
            ("eatwell-guide", 0, "Fruit and vegetables", 0.90),
            ("eatwell-guide", 1, "Oils and spreads", 0.70),
            ("cold-food-storage", 0, "Cooked leftovers", 0.95),
        ],
    )
    retriever, embedder = _retriever(tmp_path)

    result = retriever.retrieve("fruit and vegetables", document_id="eatwell-guide")

    assert embedder.texts == [BGE_QUERY_PREFIX + "fruit and vegetables"]
    assert result.searched == ("eatwell-guide",)
    assert [group.document_id for group in result.groups] == ["eatwell-guide"]
    ids = [hit.chunk_id for group in result.groups for hit in group.hits]
    assert ids == ["eatwell-guide:0", "eatwell-guide:1"]
    assert "cold-food-storage:0" not in ids


def test_filtered_query_keeps_more_than_the_per_document_cap(tmp_path: Path):
    _index(
        tmp_path,
        [("eatwell-guide", ordinal, f"row {ordinal}", 0.90 - ordinal * 0.02) for ordinal in range(4)],
    )
    retriever, _embedder = _retriever(tmp_path)

    result = retriever.retrieve("eatwell", document_id="eatwell-guide")

    ids = [hit.chunk_id for hit in result.groups[0].hits]
    assert len(ids) == 4
    assert len(ids) > MAX_CHUNKS_PER_DOCUMENT


def test_unfiltered_query_caps_each_document_and_can_span_documents(tmp_path: Path):
    rows: list[tuple[str, int, str, float]] = [
        ("kitchen-companion", 0, "highest", 0.90),
        ("cold-food-storage", 0, "second", 0.88),
        ("kitchen-companion", 1, "third", 0.85),
        ("kitchen-companion", 2, "fourth", 0.80),
        ("kitchen-companion", 3, "fifth, over the cap", 0.75),
        ("cold-food-storage", 1, "sixth", 0.70),
    ]
    _index(tmp_path, rows)
    retriever, _embedder = _retriever(tmp_path)

    result = retriever.retrieve("leftovers")

    kept = [hit for group in result.groups for hit in group.hits]
    assert {hit.document_id for hit in kept} == {"cold-food-storage", "kitchen-companion"}
    assert [group.document_id for group in result.groups] == ["cold-food-storage", "kitchen-companion"]
    per_document: dict[str, int] = {}
    for hit in kept:
        per_document[hit.document_id] = per_document.get(hit.document_id, 0) + 1
    assert per_document["kitchen-companion"] == MAX_CHUNKS_PER_DOCUMENT
    assert "kitchen-companion:3" not in {hit.chunk_id for hit in kept}
    assert [hit.chunk_id for hit in result.groups[0].hits] == [
        "cold-food-storage:0",
        "cold-food-storage:1",
    ]
    assert result.candidate_count == len(rows)
    assert len(kept) <= TOP_K_ALL


def test_similarity_threshold_drops_a_hit_at_0_61_and_keeps_0_63(tmp_path: Path):
    _index(
        tmp_path,
        [
            ("eatwell-guide", 0, "above", 0.63),
            ("who-healthy-diet", 0, "below", 0.61),
        ],
    )
    retriever, _embedder = _retriever(tmp_path)

    result = retriever.retrieve("sugars")

    assert MIN_SIMILARITY == 0.62
    kept = {hit.chunk_id for group in result.groups for hit in group.hits}
    assert kept == {"eatwell-guide:0"}
    assert {hit.chunk_id for hit in result.candidates} == {"eatwell-guide:0", "who-healthy-diet:0"}
    above = next(hit for hit in result.candidates if hit.chunk_id == "eatwell-guide:0")
    below = next(hit for hit in result.candidates if hit.chunk_id == "who-healthy-diet:0")
    assert above.similarity == pytest.approx(0.63, abs=1e-3)
    assert below.similarity == pytest.approx(0.61, abs=1e-3)
    assert above.year == 2018
    assert above.document_name.startswith("The Eatwell Guide")


def test_empty_kept_list_is_a_result_with_the_full_scope(tmp_path: Path):
    _index(tmp_path, [("fsa-chill", 0, "nothing relevant", 0.40)])
    retriever, _embedder = _retriever(tmp_path)

    result = retriever.retrieve("imported olive oil tariff")

    assert result.groups == ()
    assert result.searched == REGISTRY_IDS
    assert result.candidate_count == 1
    assert len(REGISTRY_IDS) == 7


def test_query_string_starts_with_the_bge_prefix(tmp_path: Path):
    _index(tmp_path, [("fsa-chill", 0, "defrost in the fridge", 0.80)])
    retriever, embedder = _retriever(tmp_path)
    question = "How should I defrost food?"

    retriever.retrieve(question)

    assert len(embedder.texts) == 1
    assert embedder.texts[0].startswith(BGE_QUERY_PREFIX)
    assert embedder.texts[0].endswith(question)
    assert not question.startswith(BGE_QUERY_PREFIX)


@pytest.fixture(scope="module")
def real_retriever() -> Retriever:
    if not (INDEX_DIR / "chroma.sqlite3").is_file():
        pytest.skip("data/index is not built")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return Retriever()


@pytest.mark.parametrize("question", IN_CORPUS)
def test_real_index_in_corpus_question_returns_a_hit(real_retriever: Retriever, question: str):
    document_id = "eatwell-guide" if question.startswith("According to the Eatwell Guide") else None
    result = real_retriever.retrieve(question, document_id=document_id)

    assert result.groups, question
    if document_id is None:
        assert result.candidate_count == CANDIDATE_K
        assert result.searched == REGISTRY_IDS
        kept = [hit for group in result.groups for hit in group.hits]
        assert len(kept) <= TOP_K_ALL
        assert all(len(group.hits) <= MAX_CHUNKS_PER_DOCUMENT for group in result.groups)
    else:
        assert result.searched == (document_id,)
        assert {group.document_id for group in result.groups} == {document_id}


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_real_index_off_topic_question_returns_no_hits(real_retriever: Retriever, question: str):
    result = real_retriever.retrieve(question)

    assert result.groups == ()
    assert result.searched == REGISTRY_IDS
    assert result.candidate_count == CANDIDATE_K


def test_real_index_leftovers_span_both_storage_charts(real_retriever: Retriever):
    result = real_retriever.retrieve("How long can I keep cooked leftovers in the fridge?")

    found = {group.document_id for group in result.groups}
    assert {"cold-food-storage", "kitchen-companion"} <= found


def test_real_index_filtered_miss_names_only_the_filtered_document(real_retriever: Retriever):
    result = real_retriever.retrieve(
        "How do I thaw a frozen turkey?",
        document_id="who-healthy-diet",
    )

    assert result.groups == ()
    assert result.searched == ("who-healthy-diet",)
