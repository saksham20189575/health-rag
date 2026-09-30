"""Embed chunks with BGE-small and store them in a local Chroma collection.

Passages are ``embed_text`` with no query prefix. The query prefix is applied
later, at retrieval time. Vectors are L2-normalized and the collection uses
cosine distance, so a hit's similarity is ``1 - distance``.

The build refuses to upsert any chunk whose ``embed_text`` exceeds
``BGE_MAX_SEQUENCE_LENGTH`` (512) tokens, counted with this model's tokenizer
including ``[CLS]`` and ``[SEP]``. On the current ``chunks.jsonl`` the longest
passage is 479 tokens.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from src.constants import (
    BGE_EMBEDDING_DIM,
    BGE_MAX_SEQUENCE_LENGTH,
    BGE_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    DOCUMENT_IDS,
    INDEX_DIR,
    INDEX_MANIFEST_PATH,
)
from src.ingestion.chunker import (
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_TABLE_ROW,
    Chunk,
)
from src.ingestion.store import chroma_client

__all__ = [
    "BgeEmbedder",
    "ChunkTooLongError",
    "IndexBuild",
    "IndexManifest",
    "chroma_client",
    "chunk_fingerprint",
    "chunk_metadata",
    "index_chunks",
    "l2_normalize",
    "load_index_manifest",
]

logger = logging.getLogger(__name__)

ALLOWED_BLOCK_TYPES = frozenset(
    {BLOCK_TYPE_PARAGRAPH, BLOCK_TYPE_LIST, BLOCK_TYPE_TABLE, BLOCK_TYPE_TABLE_ROW}
)
METADATA_FIELDS = (
    "document_id",
    "document_name",
    "publisher",
    "year",
    "source_url",
    "section_heading",
    "block_type",
    "retrieval_date",
)


class ChunkTooLongError(Exception):
    """One or more passages do not fit the embedding window. Nothing was upserted."""

    def __init__(self, offenders: Sequence[tuple[str, int]], limit: int) -> None:
        self.offenders = list(offenders)
        self.limit = limit
        listed = ", ".join(f"{chunk_id} ({tokens} tokens)" for chunk_id, tokens in self.offenders)
        super().__init__(
            f"{len(self.offenders)} chunk(s) exceed the {limit}-token embedding window: {listed}"
        )


class Embedder(Protocol):
    """Token counter and passage embedder. Queries are not this object's job."""

    model_name: str
    dimension: int

    def count_tokens(self, text: str) -> int:
        """Token count for ``text``, including special tokens."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages. ``texts`` are ``embed_text`` values, with no query prefix."""


@dataclass(frozen=True)
class IndexManifest:
    """Record of one successful embedding pass, written to ``index_manifest.json``."""

    model_name: str
    dimension: int
    max_sequence_length: int
    max_tokens: int
    chunk_count: int
    fingerprint: str
    embedded_at: str


@dataclass(frozen=True)
class IndexBuild:
    """Result of :func:`index_chunks`. ``skipped`` means the collection was left as-is."""

    manifest: IndexManifest
    skipped: bool


class BgeEmbedder:
    """Local ``BAAI/bge-small-en-v1.5``. Passages are embedded as given."""

    def __init__(self, model_name: str = BGE_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        from src.hf_ssl import configure_huggingface_ssl

        configure_huggingface_ssl()
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimension = self._model.get_embedding_dimension()
        if dimension != BGE_EMBEDDING_DIM:
            raise RuntimeError(
                f"{model_name} reports dimension {dimension}, expected {BGE_EMBEDDING_DIM}"
            )
        self.dimension = dimension
        # The check below refuses over-long passages. Keep the model's own
        # truncation point on the same 512-token window so a passage that
        # passed the check is embedded in full.
        self._model.max_seq_length = BGE_MAX_SEQUENCE_LENGTH
        self._tokenizer = self._model.tokenizer

    def count_tokens(self, text: str) -> int:
        # truncation=False so an over-long passage is counted, not clipped.
        # The special tokens are the [CLS] and [SEP] the 512 limit includes.
        return len(self._tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return [[float(value) for value in vector] for vector in vectors]


def chunk_fingerprint(chunks: Sequence[Chunk]) -> str:
    """sha256 of ``chunk_id`` and ``embed_text`` in list order."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.embed_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Scalar metadata stored beside ``text``. ``embed_text`` stays out of this dict."""
    return {
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "publisher": chunk.publisher,
        "year": int(chunk.year),
        "source_url": chunk.source_url,
        "section_heading": chunk.section_heading,
        "block_type": chunk.block_type,
        "retrieval_date": chunk.retrieval_date,
    }


def l2_normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return unit-length copies. A zero vector is rejected rather than stored."""
    normalized: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("Refusing to index a zero embedding")
        normalized.append([value / norm for value in vector])
    return normalized


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    index_dir: Path | str = INDEX_DIR,
    manifest_path: Path | str | None = None,
    embedder: Embedder | None = None,
    model_name: str | None = None,
) -> IndexBuild:
    """Upsert ``chunks`` into the persistent collection.

    Re-embedding is skipped when the manifest's fingerprint and model name
    match and the collection still holds the same number of rows. A passage
    over 512 tokens raises :class:`ChunkTooLongError` before any upsert.
    """
    if not chunks:
        raise ValueError("No chunks to index")
    _validate_chunks(chunks)

    index_dir = Path(index_dir)
    if manifest_path is None:
        manifest_path = index_dir / INDEX_MANIFEST_PATH.name
    else:
        manifest_path = Path(manifest_path)

    resolved_model = model_name or (embedder.model_name if embedder is not None else BGE_MODEL_NAME)
    fingerprint = chunk_fingerprint(chunks)
    existing = load_index_manifest(manifest_path)
    if existing is not None and _can_skip(existing, fingerprint, resolved_model, index_dir):
        logger.info(
            "Index unchanged (%d chunks, %s); skipping re-embed",
            existing.chunk_count,
            existing.model_name,
        )
        return IndexBuild(manifest=existing, skipped=True)

    if embedder is None:
        embedder = BgeEmbedder(resolved_model)
    max_tokens = _measure_max_tokens(chunks, embedder)
    logger.info(
        "Embedding %d chunks with %s (longest passage %d tokens)",
        len(chunks),
        embedder.model_name,
        max_tokens,
    )
    vectors = l2_normalize(embedder.embed([chunk.embed_text for chunk in chunks]))
    _check_vectors(vectors, chunks, embedder.dimension)

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chroma_client(index_dir)
    _upsert(client, chunks, vectors)

    manifest = IndexManifest(
        model_name=embedder.model_name,
        dimension=len(vectors[0]),
        max_sequence_length=BGE_MAX_SEQUENCE_LENGTH,
        max_tokens=max_tokens,
        chunk_count=len(chunks),
        fingerprint=fingerprint,
        embedded_at=_utc_now(),
    )
    _write_manifest(manifest_path, manifest)
    logger.info("Wrote %s (%d chunks)", manifest_path, manifest.chunk_count)
    return IndexBuild(manifest=manifest, skipped=False)


def load_index_manifest(path: Path | str = INDEX_MANIFEST_PATH) -> IndexManifest | None:
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IndexManifest(
        model_name=str(data["model_name"]),
        dimension=int(data["dimension"]),
        max_sequence_length=int(data["max_sequence_length"]),
        max_tokens=int(data["max_tokens"]),
        chunk_count=int(data["chunk_count"]),
        fingerprint=str(data["fingerprint"]),
        embedded_at=str(data["embedded_at"]),
    )


def _validate_chunks(chunks: Sequence[Chunk]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    unknown_documents: set[str] = set()
    bad_types: list[str] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            duplicates.append(chunk.chunk_id)
        seen.add(chunk.chunk_id)
        if chunk.document_id not in DOCUMENT_IDS:
            unknown_documents.add(chunk.document_id)
        if chunk.block_type not in ALLOWED_BLOCK_TYPES:
            bad_types.append(f"{chunk.chunk_id} ({chunk.block_type})")
        if not chunk.embed_text or not chunk.text:
            raise ValueError(f"{chunk.chunk_id} is missing text or embed_text")
    if duplicates:
        raise ValueError("Duplicate chunk_id values: " + ", ".join(duplicates))
    if unknown_documents:
        raise ValueError(
            "chunk document_id is not in the registry: " + ", ".join(sorted(unknown_documents))
        )
    if bad_types:
        raise ValueError("block_type is not paragraph, list, table, or table_row: " + ", ".join(bad_types))


def _measure_max_tokens(chunks: Sequence[Chunk], embedder: Embedder) -> int:
    observed = 0
    offenders: list[tuple[str, int]] = []
    for chunk in chunks:
        count = embedder.count_tokens(chunk.embed_text)
        if count > observed:
            observed = count
        if count > BGE_MAX_SEQUENCE_LENGTH:
            offenders.append((chunk.chunk_id, count))
    if offenders:
        raise ChunkTooLongError(offenders, BGE_MAX_SEQUENCE_LENGTH)
    return observed


def _check_vectors(vectors: Sequence[Sequence[float]], chunks: Sequence[Chunk], dimension: int) -> None:
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != dimension:
            raise RuntimeError(
                f"{chunk.chunk_id} embedding has dimension {len(vector)}, expected {dimension}"
            )


def _can_skip(
    existing: IndexManifest | None,
    fingerprint: str,
    model_name: str,
    index_dir: Path,
) -> bool:
    if existing is None:
        return False
    if existing.fingerprint != fingerprint or existing.model_name != model_name:
        return False
    if existing.max_sequence_length != BGE_MAX_SEQUENCE_LENGTH:
        return False
    return _collection_count(index_dir) == existing.chunk_count


def _collection_count(index_dir: Path) -> int | None:
    if not index_dir.exists():
        return None
    client = chroma_client(index_dir)
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        return None
    return collection.count()


def _upsert(client: ClientAPI, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
    collection = _collection_for_write(client)
    ids = [chunk.chunk_id for chunk in chunks]
    collection.upsert(
        ids=ids,
        embeddings=[list(vector) for vector in vectors],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk_metadata(chunk) for chunk in chunks],
    )
    kept = set(ids)
    current = collection.get(include=["metadatas"])["ids"]
    extra = [chunk_id for chunk_id in current if chunk_id not in kept]
    if extra:
        collection.delete(ids=extra)


def _collection_for_write(client: ClientAPI) -> Collection:
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        return _create_collection(client)
    space = (collection.metadata or {}).get("hnsw:space")
    if space != "cosine":
        client.delete_collection(CHROMA_COLLECTION_NAME)
        return _create_collection(client)
    return collection


def _create_collection(client: ClientAPI) -> Collection:
    # embedding_function=None: vectors are computed in this module and passed
    # in on upsert. Chroma must not run its default ONNX embedder.
    return client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def _write_manifest(path: Path, manifest: IndexManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
