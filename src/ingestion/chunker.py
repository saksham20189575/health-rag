"""Turn parsed blocks into structure-aware retrieval chunks.

Reads ``data/parsed/blocks.jsonl`` (phase 2, task 2 of the implementation plan).
Headings are context, not chunks. Tokens are ``round(word_count × 1.3)`` so
chunking does not load the embedding model.

- Paragraphs under one heading pack toward ``CHUNK_TARGET_TOKENS`` (400).
  Glossary entries (``heading_path`` ending in ``Glossary``) stay one chunk each.
- A list run under one heading packs the same way. A short paragraph immediately
  before the run (at most 80 words, same heading) is the lead-in of the first
  list chunk only.
- A table at or under the target stays one chunk. A larger table becomes one
  chunk per data row, with the caption and column headers repeated on the row.
- A single paragraph over ``MAX_CHUNK_TOKENS`` (800) is split on sentence
  boundaries with about ``CHUNK_OVERLAP_TOKENS`` of overlap. List items and
  table rows are never split and never overlapped.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.constants import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    CHUNKS_DATA_DIR,
    CORPUS,
    CORPUS_MANIFEST_PATH,
    MAX_CHUNK_TOKENS,
    CorpusDocument,
)
from src.ingestion.parser import (
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    PARSED_DATA_DIR,
    Block,
    format_table_markdown,
    format_table_row,
)

# Tokens ≈ words × 1.3, matching the sizes measured on data/parsed/blocks.jsonl.
TOKEN_ESTIMATE_PER_WORD = 1.3
# A paragraph of at most this many words, sitting immediately before a list
# under the same heading, opens that list instead of standing as prose.
LEAD_IN_MAX_WORDS = 80

BLOCK_TYPE_PARAGRAPH = "paragraph"
BLOCK_TYPE_LIST = "list"
BLOCK_TYPE_TABLE = "table"
BLOCK_TYPE_TABLE_ROW = "table_row"

BLOCKS_PATH = PARSED_DATA_DIR / "blocks.jsonl"
CHUNKS_PATH = CHUNKS_DATA_DIR / "chunks.jsonl"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit, with the citation fields the answer layer renders."""

    chunk_id: str
    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    retrieval_date: str
    section_heading: str
    block_type: str
    text: str
    embed_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Approximate BGE tokens without loading the model."""
    words = len(text.split())
    if words == 0:
        return 0
    return round(words * TOKEN_ESTIMATE_PER_WORD)


def word_count(text: str) -> int:
    return len(text.split())


def load_blocks(path: Path | str = BLOCKS_PATH) -> dict[str, list[Block]]:
    """Load ``blocks.jsonl`` grouped by ``document_id``, in file order."""
    path = Path(path)
    grouped: dict[str, list[Block]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            block = Block(
                type=row["type"],
                text=row.get("text") or "",
                heading_path=tuple(row.get("heading_path") or ()),
                page=row.get("page"),
                level=row.get("level"),
                header=row.get("header"),
                rows=row.get("rows"),
                caption=row.get("caption"),
            )
            grouped.setdefault(row["document_id"], []).append(block)
    return grouped


def load_retrieval_dates(path: Path | str = CORPUS_MANIFEST_PATH) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["document_id"]: row.get("retrieval_date") or "" for row in data["documents"]}


def chunk_blocks(blocks: Sequence[Block], document: CorpusDocument, retrieval_date: str = "") -> list[Chunk]:
    """Chunk one document's blocks, in order. Ordinals start at 0."""
    pieces = [(block_type, text.strip(), heading) for block_type, text, heading in _pieces(blocks) if text.strip()]
    chunks: list[Chunk] = []
    for ordinal, (block_type, body, heading) in enumerate(pieces):
        section = " > ".join(heading) if heading else document.document_name
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{ordinal}",
                document_id=document.document_id,
                document_name=document.document_name,
                publisher=document.publisher,
                year=document.year,
                source_url=document.source_url,
                retrieval_date=retrieval_date,
                section_heading=section,
                block_type=block_type,
                text=body,
                embed_text=f"{document.document_name} — {section}\n{body}",
            )
        )
    return chunks


def chunk_parsed(
    blocks_path: Path | str = BLOCKS_PATH,
    manifest_path: Path | str = CORPUS_MANIFEST_PATH,
) -> list[Chunk]:
    """Chunk every registry document present in ``blocks.jsonl``, in registry order."""
    grouped = load_blocks(blocks_path)
    dates = load_retrieval_dates(manifest_path)
    chunks: list[Chunk] = []
    for document in CORPUS:
        blocks = grouped.get(document.document_id)
        if not blocks:
            continue
        chunks.extend(chunk_blocks(blocks, document, dates.get(document.document_id, "")))
    return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path | str = CHUNKS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------
# Block walk
# --------------------------------------------------------------------------


def _pieces(blocks: Sequence[Block]) -> list[tuple[str, str, tuple[str, ...]]]:
    pieces: list[tuple[str, str, tuple[str, ...]]] = []
    index = 0
    total = len(blocks)
    while index < total:
        block = blocks[index]
        if block.type == BLOCK_HEADING:
            index += 1
            continue
        if block.type == BLOCK_TABLE:
            pieces.extend(_table_pieces(block))
            index += 1
            continue
        if block.type == BLOCK_PARAGRAPH:
            heading = block.heading_path
            run: list[Block] = []
            while index < total and blocks[index].type == BLOCK_PARAGRAPH and blocks[index].heading_path == heading:
                run.append(blocks[index])
                index += 1
            if _is_glossary(heading):
                for paragraph in run:
                    for part in _split_overlong_paragraph(paragraph.text):
                        pieces.append((BLOCK_TYPE_PARAGRAPH, part, heading))
                continue
            lead_in: Block | None = None
            if (
                run
                and index < total
                and blocks[index].type == BLOCK_LIST_ITEM
                and blocks[index].heading_path == heading
                and word_count(run[-1].text) <= LEAD_IN_MAX_WORDS
            ):
                lead_in = run.pop()
            if run:
                pieces.extend(_pack_paragraphs(run, heading))
            if lead_in is not None:
                items: list[Block] = []
                while index < total and blocks[index].type == BLOCK_LIST_ITEM and blocks[index].heading_path == heading:
                    items.append(blocks[index])
                    index += 1
                pieces.extend(_pack_list(items, heading, lead_in.text))
            continue
        if block.type == BLOCK_LIST_ITEM:
            heading = block.heading_path
            items = []
            while index < total and blocks[index].type == BLOCK_LIST_ITEM and blocks[index].heading_path == heading:
                items.append(blocks[index])
                index += 1
            pieces.extend(_pack_list(items, heading, None))
            continue
        index += 1
    return pieces


def _is_glossary(heading: tuple[str, ...]) -> bool:
    return bool(heading) and heading[-1] == "Glossary"


def _pack_paragraphs(paragraphs: Sequence[Block], heading: tuple[str, ...]) -> list[tuple[str, str, tuple[str, ...]]]:
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_overlong_paragraph(paragraph.text))
    return [(BLOCK_TYPE_PARAGRAPH, text, heading) for text in _pack(units, "\n\n")]


def _pack_list(
    items: Sequence[Block], heading: tuple[str, ...], lead_in: str | None
) -> list[tuple[str, str, tuple[str, ...]]]:
    texts = [item.text.strip() for item in items if item.text.strip()]
    if not texts:
        if lead_in and lead_in.strip():
            return [(BLOCK_TYPE_LIST, lead_in.strip(), heading)]
        return []
    packed: list[str] = []
    buffer: list[str] = []
    first_chunk = True
    for text in texts:
        if not buffer:
            buffer = [text]
            continue
        prefix = [lead_in.strip()] if first_chunk and lead_in and lead_in.strip() else []
        candidate = "\n".join(prefix + buffer + [text])
        if estimate_tokens(candidate) > CHUNK_TARGET_TOKENS:
            packed.append("\n".join(prefix + buffer))
            first_chunk = False
            buffer = [text]
        else:
            buffer.append(text)
    if buffer:
        prefix = [lead_in.strip()] if first_chunk and lead_in and lead_in.strip() else []
        packed.append("\n".join(prefix + buffer))
    return [(BLOCK_TYPE_LIST, text, heading) for text in packed]


def _pack(units: Sequence[str], separator: str) -> list[str]:
    """Pack whole units toward the target. A single over-target unit stays whole."""
    packed: list[str] = []
    buffer: list[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if not buffer:
            buffer = [unit]
            continue
        candidate = separator.join(buffer + [unit])
        if estimate_tokens(candidate) > CHUNK_TARGET_TOKENS:
            packed.append(separator.join(buffer))
            buffer = [unit]
        else:
            buffer.append(unit)
    if buffer:
        packed.append(separator.join(buffer))
    return packed


def _table_pieces(block: Block) -> list[tuple[str, str, tuple[str, ...]]]:
    heading = block.heading_path
    source = _table_source_text(block)
    header = block.header or []
    rows = block.rows or []
    if header and rows and estimate_tokens(source) > CHUNK_TARGET_TOKENS:
        pieces: list[tuple[str, str, tuple[str, ...]]] = []
        caption = (block.caption or "").strip()
        for row in rows:
            labeled = format_table_row(header, row)
            if not labeled:
                continue
            if caption:
                labeled = f"{caption}\n{labeled}"
            pieces.append((BLOCK_TYPE_TABLE_ROW, labeled, heading))
        if pieces:
            return pieces
    if source:
        return [(BLOCK_TYPE_TABLE, source, heading)]
    return []


def _table_source_text(block: Block) -> str:
    if block.text and block.text.strip():
        return block.text.strip()
    if block.header and block.rows is not None:
        return format_table_markdown(block.header, block.rows, block.caption)
    return ""


def _split_overlong_paragraph(text: str) -> list[str]:
    """Split one paragraph on sentence boundaries once it passes the hard cap."""
    body = text.strip()
    if not body or estimate_tokens(body) <= MAX_CHUNK_TOKENS:
        return [body] if body else []
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(body) if part.strip()]
    if len(sentences) <= 1:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if not current:
            current = [sentence]
            continue
        candidate = " ".join(current + [sentence])
        if estimate_tokens(candidate) <= MAX_CHUNK_TOKENS:
            current.append(sentence)
            continue
        pieces.append(" ".join(current))
        overlap = _overlap_sentences(current)
        current = overlap + [sentence]
        # A sentence that does not fit beside the overlap stays whole, with no partial sentence.
        if len(current) > 1 and estimate_tokens(" ".join(current)) > MAX_CHUNK_TOKENS:
            current = [sentence]
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_sentences(sentences: Sequence[str]) -> list[str]:
    """Trailing whole sentences totalling about ``CHUNK_OVERLAP_TOKENS``."""
    chosen: list[str] = []
    for sentence in reversed(sentences):
        trial = [sentence, *chosen]
        if estimate_tokens(" ".join(trial)) > CHUNK_OVERLAP_TOKENS:
            break
        chosen = trial
    return chosen
