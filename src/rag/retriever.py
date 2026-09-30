"""Vector search over the local Chroma collection.

Queries are embedded with ``BGE_QUERY_PREFIX``. Passages were embedded without
it. Similarity is ``1 - distance`` in the collection's cosine space.

Unfiltered search fetches ``CANDIDATE_K`` neighbours, drops hits below
``MIN_SIMILARITY``, keeps at most ``MAX_CHUNKS_PER_DOCUMENT`` per document,
and stops at ``TOP_K_ALL``. Filtered search fetches ``TOP_K_FILTERED`` from
one document and keeps every survivor. An empty group list is a result, not
an error. This module does not call Groq and does not write refusal text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from chromadb.errors import NotFoundError

from src.constants import (
    BGE_EMBEDDING_DIM,
    BGE_MAX_SEQUENCE_LENGTH,
    BGE_MODEL_NAME,
    BGE_QUERY_PREFIX,
    CANDIDATE_K,
    CHROMA_COLLECTION_NAME,
    CORPUS,
    DOCUMENT_IDS,
    INDEX_DIR,
    MAX_CHUNKS_PER_DOCUMENT,
    MIN_SIMILARITY,
    TOP_K_ALL,
    TOP_K_FILTERED,
)
from src.ingestion.store import chroma_client

REGISTRY_IDS: tuple[str, ...] = tuple(document.document_id for document in CORPUS)


class QueryEmbedder(Protocol):
    """Embeds one already-prefixed query. Passage embedding is the indexer's job."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Return one vector for ``text``. ``text`` includes ``BGE_QUERY_PREFIX``."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One neighbour, before or after the threshold and the per-document cap."""

    chunk_id: str
    text: str
    similarity: float
    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    section_heading: str
    block_type: str
    retrieval_date: str


@dataclass(frozen=True)
class DocumentHits:
    """Surviving chunks for one document, highest similarity first."""

    document_id: str
    hits: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """Scope that was searched, and the hits phase 6 may put in the prompt.

    ``candidates`` are the neighbours Chroma returned, in rank order, before
    the similarity threshold and the per-document cap. ``candidate_count`` is
    ``len(candidates)``. ``groups`` is empty when nothing survived; that is a
    miss, not an error.
    """

    searched: tuple[str, ...]
    groups: tuple[DocumentHits, ...]
    candidate_count: int
    candidates: tuple[RetrievedChunk, ...]


class BgeQueryEmbedder:
    """Local ``BAAI/bge-small-en-v1.5``. The caller supplies the query prefix."""

    def __init__(self, model_name: str = BGE_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        from src.hf_ssl import configure_huggingface_ssl

        configure_huggingface_ssl()
        self._model = SentenceTransformer(model_name)
        dimension = self._model.get_embedding_dimension()
        if dimension != BGE_EMBEDDING_DIM:
            raise RuntimeError(
                f"{model_name} reports dimension {dimension}, expected {BGE_EMBEDDING_DIM}"
            )
        self._model.max_seq_length = BGE_MAX_SEQUENCE_LENGTH

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return [float(value) for value in vector]


class Retriever:
    """Search ``data/index``. A passed-in embedder is used as-is; otherwise BGE."""

    def __init__(
        self,
        index_dir: Path | str = INDEX_DIR,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._embedder = embedder

    def retrieve(self, query: str, document_id: str | None = None) -> RetrievalResult:
        """Search one document, or all seven when ``document_id`` is omitted.

        A ``document_id`` outside the registry raises ``ValueError`` before
        the embedder or Chroma is called.
        """
        if document_id is not None and document_id not in DOCUMENT_IDS:
            raise ValueError(f"document_id is not in the registry: {document_id}")

        filtered = document_id is not None
        searched = (document_id,) if filtered else REGISTRY_IDS
        vector = list(self._query_embedder().embed_query(BGE_QUERY_PREFIX + query))
        candidates = _query_collection(
            self._index_dir,
            vector,
            n_results=TOP_K_FILTERED if filtered else CANDIDATE_K,
            document_id=document_id,
        )
        kept = _keep(candidates, filtered=filtered)
        return RetrievalResult(
            searched=searched,
            groups=_group(kept, searched),
            candidate_count=len(candidates),
            candidates=tuple(candidates),
        )

    def _query_embedder(self) -> QueryEmbedder:
        if self._embedder is None:
            self._embedder = BgeQueryEmbedder()
        return self._embedder


def _query_collection(
    index_dir: Path,
    vector: Sequence[float],
    *,
    n_results: int,
    document_id: str | None,
) -> list[RetrievedChunk]:
    if not index_dir.exists():
        raise FileNotFoundError(f"No index at {index_dir}")
    client = chroma_client(index_dir)
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=None)
    except NotFoundError as exc:
        raise FileNotFoundError(
            f"No collection {CHROMA_COLLECTION_NAME!r} in {index_dir}"
        ) from exc

    kwargs: dict[str, object] = {
        "query_embeddings": [list(vector)],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if document_id is not None:
        kwargs["where"] = {"document_id": document_id}
    stored = collection.query(**kwargs)

    hits: list[RetrievedChunk] = []
    ids = stored["ids"][0]
    documents = stored["documents"][0]
    metadatas = stored["metadatas"][0]
    distances = stored["distances"][0]
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        hits.append(_chunk(chunk_id, text or "", metadata or {}, float(distance)))
    return hits


def _chunk(chunk_id: str, text: str, metadata: dict[str, object], distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        similarity=1.0 - distance,
        document_id=str(metadata["document_id"]),
        document_name=str(metadata["document_name"]),
        publisher=str(metadata["publisher"]),
        year=int(metadata["year"]),
        source_url=str(metadata["source_url"]),
        section_heading=str(metadata["section_heading"]),
        block_type=str(metadata["block_type"]),
        retrieval_date=str(metadata["retrieval_date"]),
    )


def _keep(candidates: Sequence[RetrievedChunk], *, filtered: bool) -> list[RetrievedChunk]:
    survivors = [hit for hit in candidates if hit.similarity >= MIN_SIMILARITY]
    if filtered:
        return survivors
    kept: list[RetrievedChunk] = []
    per_document: Counter[str] = Counter()
    for hit in survivors:
        if per_document[hit.document_id] >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        kept.append(hit)
        per_document[hit.document_id] += 1
        if len(kept) >= TOP_K_ALL:
            break
    return kept


def _group(hits: Sequence[RetrievedChunk], searched: Sequence[str]) -> tuple[DocumentHits, ...]:
    by_document: dict[str, list[RetrievedChunk]] = {document_id: [] for document_id in searched}
    for hit in hits:
        by_document.setdefault(hit.document_id, []).append(hit)
    groups: list[DocumentHits] = []
    for document_id in searched:
        document_hits = by_document.get(document_id, [])
        if document_hits:
            groups.append(DocumentHits(document_id=document_id, hits=tuple(document_hits)))
    return tuple(groups)
