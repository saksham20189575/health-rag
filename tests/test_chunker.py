"""Chunker tests against the parsed corpus and a few synthetic blocks."""

from __future__ import annotations

import pytest

from src.constants import CHUNK_TARGET_TOKENS, CORPUS, DOCUMENT_IDS
from src.ingestion.chunker import (
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_TABLE_ROW,
    chunk_blocks,
    chunk_parsed,
    estimate_tokens,
    load_blocks,
    write_chunks,
)
from src.ingestion.parser import BLOCK_LIST_ITEM, BLOCK_PARAGRAPH, BLOCK_TABLE, Block, format_table_row

DOCS = {document.document_id: document for document in CORPUS}


@pytest.fixture(scope="module")
def chunks():
    return chunk_parsed()


@pytest.fixture(scope="module")
def blocks_by_doc():
    return load_blocks()


def _of(chunks, document_id, block_type=None):
    rows = [chunk for chunk in chunks if chunk.document_id == document_id]
    if block_type is not None:
        rows = [chunk for chunk in rows if chunk.block_type == block_type]
    return rows


# --------------------------------------------------------------------------
# Parsed corpus
# --------------------------------------------------------------------------


def test_every_chunk_has_citation_metadata(chunks):
    assert {chunk.document_id for chunk in chunks} == DOCUMENT_IDS
    for chunk in chunks:
        document = DOCS[chunk.document_id]
        assert chunk.chunk_id.startswith(f"{chunk.document_id}:")
        assert chunk.document_name == document.document_name
        assert chunk.publisher == document.publisher
        assert chunk.year == document.year
        assert chunk.source_url == document.source_url
        assert chunk.retrieval_date
        assert chunk.section_heading
        assert chunk.block_type in {BLOCK_TYPE_PARAGRAPH, BLOCK_TYPE_LIST, BLOCK_TYPE_TABLE, BLOCK_TYPE_TABLE_ROW}
        assert chunk.text
        assert chunk.embed_text.startswith(f"{chunk.document_name} — {chunk.section_heading}\n")


def test_headings_are_not_chunks(chunks):
    assert all(chunk.block_type != "heading" for chunk in chunks)


def test_cold_food_storage_rows_stay_intact(chunks, blocks_by_doc):
    rows = _of(chunks, "cold-food-storage", BLOCK_TYPE_TABLE_ROW)
    table = next(block for block in blocks_by_doc["cold-food-storage"] if block.type == BLOCK_TABLE)
    assert len(rows) == len(table.rows) == 53
    assert all(chunk.block_type != BLOCK_TYPE_TABLE for chunk in _of(chunks, "cold-food-storage"))

    leftovers = next(chunk for chunk in rows if "Cooked meat or poultry" in chunk.text)
    assert "3 to 4 days" in leftovers.text
    assert "2 to 6 months" in leftovers.text
    assert "Refrigerator" in leftovers.text
    assert "Freezer" in leftovers.text
    assert "Chicken or turkey, whole" not in leftovers.text

    intro = _of(chunks, "cold-food-storage", BLOCK_TYPE_PARAGRAPH)
    assert len(intro) == 1
    assert "quality only" in intro[0].text
    assert "Chicken or turkey, whole" not in intro[0].text


def test_tables_follow_the_size_branch(chunks, blocks_by_doc):
    small = []
    large = []
    for document_id, blocks in blocks_by_doc.items():
        for block in blocks:
            if block.type != BLOCK_TABLE:
                continue
            if estimate_tokens(block.text) <= CHUNK_TARGET_TOKENS:
                small.append((document_id, block))
            else:
                large.append((document_id, block))
    assert len(small) == 7
    assert len(large) == 5

    for document_id, block in small:
        matches = [chunk for chunk in _of(chunks, document_id, BLOCK_TYPE_TABLE) if chunk.text == block.text.strip()]
        assert len(matches) == 1
        labeled = format_table_row(block.header, block.rows[0])
        assert not any(chunk.text == labeled or chunk.text.endswith("\n" + labeled) for chunk in _of(chunks, document_id, BLOCK_TYPE_TABLE_ROW))

    for document_id, block in large:
        assert not any(chunk.text == block.text.strip() for chunk in _of(chunks, document_id))
        for row in block.rows:
            labeled = format_table_row(block.header, row)
            if block.caption:
                labeled = f"{block.caption.strip()}\n{labeled}"
            matches = [chunk for chunk in _of(chunks, document_id, BLOCK_TYPE_TABLE_ROW) if chunk.text == labeled]
            assert len(matches) == 1


def test_eatwell_sugar_table_is_one_chunk(chunks):
    sugar = [
        chunk
        for chunk in _of(chunks, "eatwell-guide", BLOCK_TYPE_TABLE)
        if chunk.section_heading == "Cutting down on sugar" and "4-6 years" in chunk.text
    ]
    assert len(sugar) == 1
    assert "7-10 years" in sugar[0].text
    assert "From 11 years" in sugar[0].text


def test_eatwell_fruit_row_is_its_own_chunk(chunks):
    fruit = [
        chunk
        for chunk in _of(chunks, "eatwell-guide", BLOCK_TYPE_TABLE_ROW)
        if chunk.text.startswith("Food group: Fruit and vegetables")
    ]
    assert len(fruit) == 1
    assert "at least 5 portions" in fruit[0].text
    assert "Oils and spreads" not in fruit[0].text


def test_eatwell_numbered_tips_share_one_list_chunk(chunks):
    tips = [
        chunk
        for chunk in _of(chunks, "eatwell-guide", BLOCK_TYPE_LIST)
        if "1. Base your meals on starchy foods" in chunk.text
    ]
    assert len(tips) == 1
    text = tips[0].text
    for phrase in (
        "2. Eat lots of fruit and veg",
        "3. Eat more fish",
        "4. Cut down on saturated fat and sugar",
        "5. Eat less salt",
        "6. Get active and be a healthy weight",
        "7. Don",
        "8. Don",
        "skip breakfast",
    ):
        assert phrase in text
    assert tips[0].section_heading == "8 tips for eating well"


def test_five_keys_fragments_stay_with_their_list(chunks):
    packed = [chunk for chunk in _of(chunks, "who-five-keys", BLOCK_TYPE_LIST) if "Faeces;" in chunk.text]
    assert len(packed) == 1
    assert "Soil and water;" in packed[0].text
    assert packed[0].text.strip() != "Faeces;"


def test_glossary_entries_stay_separate(chunks):
    entries = [
        chunk
        for chunk in _of(chunks, "who-five-keys", BLOCK_TYPE_PARAGRAPH)
        if chunk.section_heading == "Glossary" and chunk.text.startswith("Danger zone")
    ]
    assert len(entries) == 1
    assert "5 °C to 60 °C" in entries[0].text
    assert "Cross-contamination" not in entries[0].text
    assert "Bacteria" not in entries[0].text


def test_long_list_runs_break_between_items(chunks):
    promote = [
        chunk
        for chunk in _of(chunks, "who-healthy-diet", BLOCK_TYPE_LIST)
        if chunk.section_heading.endswith("HOW TO PROMOTE HEALTHY DIETS")
    ]
    assert len(promote) >= 2
    outages = [
        chunk
        for chunk in _of(chunks, "kitchen-companion", BLOCK_TYPE_LIST)
        if chunk.section_heading.endswith("Power Outages")
    ]
    assert len(outages) >= 2


# --------------------------------------------------------------------------
# Synthetic blocks
# --------------------------------------------------------------------------


def _paragraph(text, heading=("Storage",)):
    return Block(type=BLOCK_PARAGRAPH, text=text, heading_path=heading)


def _item(text, heading=("Storage",)):
    return Block(type=BLOCK_LIST_ITEM, text=text, heading_path=heading)


def test_overlong_paragraph_splits_on_a_sentence_boundary():
    sentences = [f"Guidance sentence number {index:03d} remains intact." for index in range(120)]
    blocks = [_paragraph(" ".join(sentences))]
    chunks = chunk_blocks(blocks, DOCS["cold-food-storage"], "2026-09-27")

    assert len(chunks) >= 2
    assert all(chunk.section_heading == "Storage" for chunk in chunks)
    assert all(chunk.block_type == BLOCK_TYPE_PARAGRAPH for chunk in chunks)
    known = set(sentences)
    for chunk in chunks:
        parts = [part if part.endswith(".") else part + "." for part in chunk.text.split(". ")]
        assert parts
        assert all(part in known for part in parts)
    second_first = chunks[1].text.split(". ")[0] + "."
    assert second_first in chunks[0].text
    last = chunks[-1].text
    assert "Guidance sentence number 119 remains intact." in last
    assert "Guidance sentence number 119 remains intact." not in chunks[0].text


def test_paragraph_under_the_cap_stays_whole():
    sentences = [f"Short guidance sentence number {index:03d} stays put." for index in range(70)]
    text = " ".join(sentences)
    assert estimate_tokens(text) < 800
    chunks = chunk_blocks([_paragraph(text)], DOCS["cold-food-storage"], "2026-09-27")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_short_lead_in_joins_the_list_and_a_long_one_does_not():
    short = "Wash hands before you cook."
    blocks = [_paragraph(short), _item("Faeces;"), _item("Soil and water;")]
    chunks = chunk_blocks(blocks, DOCS["who-five-keys"], "2026-09-27")
    assert len(chunks) == 1
    assert chunks[0].block_type == BLOCK_TYPE_LIST
    assert chunks[0].text.startswith(short)
    assert "Faeces;" in chunks[0].text
    assert "Soil and water;" in chunks[0].text

    long_lead = " ".join(["background"] * 81)
    separate = chunk_blocks(
        [_paragraph(long_lead), _item("Keep the list."), _item("Second item.")],
        DOCS["who-five-keys"],
        "2026-09-27",
    )
    assert [chunk.block_type for chunk in separate] == [BLOCK_TYPE_PARAGRAPH, BLOCK_TYPE_LIST]
    assert separate[0].text == long_lead
    assert long_lead not in separate[1].text
    assert "Keep the list." in separate[1].text


def test_list_and_table_rows_are_not_overlapped():
    first = "alpha " * 200
    second = "beta " * 200
    chunks = chunk_blocks([_item(first.strip()), _item(second.strip())], DOCS["fsa-chill"], "2026-09-27")
    assert len(chunks) == 2
    assert chunks[0].text.startswith("alpha")
    assert "alpha" not in chunks[1].text
    assert chunks[1].text.startswith("beta")

    wide = "cell " * 400
    table = Block(
        type=BLOCK_TABLE,
        text=wide,
        heading_path=("Chart",),
        header=["Food", "Note"],
        rows=[["Chicken", ""], ["Pork", "Rest the meat"]],
        caption="Roasting Chart",
    )
    row_chunks = chunk_blocks([table], DOCS["kitchen-companion"], "2026-09-27")
    assert [chunk.block_type for chunk in row_chunks] == [BLOCK_TYPE_TABLE_ROW, BLOCK_TYPE_TABLE_ROW]
    assert row_chunks[0].text == "Roasting Chart\nFood: Chicken"
    assert "Note" not in row_chunks[0].text
    assert row_chunks[1].text == "Roasting Chart\nFood: Pork | Note: Rest the meat"
    assert row_chunks[0].section_heading == row_chunks[1].section_heading == "Chart"


def test_write_chunks_round_trip(tmp_path, chunks):
    path = write_chunks(chunks, tmp_path / "chunks.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    assert '"document_id": "cold-food-storage"' in lines[0]
