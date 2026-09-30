"""Citation renderer (implementation plan, phase 6; architecture §5.5, §6)."""

from __future__ import annotations

import pytest

from src.constants import CORPUS
from src.rag.citations import citation_for, order_sections, render_sections, render_text
from src.rag.models import GeneratedPayload
from src.rag.retriever import DocumentHits, RetrievedChunk

BY_ID = {document.document_id: document for document in CORPUS}


def chunk(document_id: str, ordinal: int, text: str = "text") -> RetrievedChunk:
    document = BY_ID[document_id]
    return RetrievedChunk(
        chunk_id=f"{document_id}:{ordinal}",
        text=text,
        similarity=0.8,
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        section_heading="Heading",
        block_type="paragraph",
        retrieval_date="2026-09-27",
    )


GROUPS = (
    DocumentHits("who-healthy-diet", (chunk("who-healthy-diet", 8),)),
    DocumentHits("eatwell-guide", (chunk("eatwell-guide", 34), chunk("eatwell-guide", 35))),
    DocumentHits("kitchen-companion", (chunk("kitchen-companion", 37),)),
)


def payload(*documents: tuple[str, list[tuple[str, list[str]]]]) -> GeneratedPayload:
    return GeneratedPayload.model_validate(
        {
            "documents": [
                {
                    "document_id": document_id,
                    "claims": [{"text": text, "chunk_ids": chunk_ids} for text, chunk_ids in claims],
                }
                for document_id, claims in documents
            ]
        }
    )


def test_citation_is_filled_from_the_chunk_record() -> None:
    citation = citation_for(GROUPS[1])
    hit = GROUPS[1].hits[0]
    assert citation.document_name == hit.document_name
    assert citation.publisher == hit.publisher
    assert citation.year == hit.year
    assert citation.source_url == hit.source_url


def test_citation_for_needs_a_hit() -> None:
    with pytest.raises(ValueError):
        citation_for(DocumentHits("eatwell-guide", ()))


def test_fabricated_multi_document_payload_becomes_one_section_per_document_in_registry_order() -> None:
    generated = payload(
        ("kitchen-companion", [("Store leftovers within 2 hours.", ["kitchen-companion:37"])]),
        ("eatwell-guide", [("Choose unsaturated oils.", ["eatwell-guide:34"]), ("Use small amounts.", ["eatwell-guide:35"])]),
        ("who-healthy-diet", [("Total fat under 30% of energy.", ["who-healthy-diet:8"])]),
    )

    sections = render_sections(generated, GROUPS)

    assert [section.document_id for section in sections] == ["who-healthy-diet", "eatwell-guide", "kitchen-companion"]
    for section in sections:
        document = BY_ID[section.document_id]
        assert section.document_name == document.document_name
        assert section.publisher == document.publisher
        assert section.year == document.year
        assert section.source_url == document.source_url
        for claim in section.claims:
            assert claim.citation.source_url == document.source_url
            assert claim.citation.document_name == document.document_name
            assert claim.citation.publisher == document.publisher
            assert claim.citation.year == document.year
    assert [claim.text for claim in sections[1].claims] == ["Choose unsaturated oils.", "Use small amounts."]


def test_document_with_empty_claims_is_omitted() -> None:
    generated = payload(
        ("eatwell-guide", [("Choose unsaturated oils.", ["eatwell-guide:34"])]),
        ("who-healthy-diet", []),
    )

    sections = render_sections(generated, GROUPS)

    assert [section.document_id for section in sections] == ["eatwell-guide"]


def test_document_outside_the_retrieval_scope_is_omitted() -> None:
    generated = payload(("fsa-chill", [("Defrost in the fridge.", ["fsa-chill:3"])]))

    assert render_sections(generated, GROUPS) == []


def test_repeated_document_id_is_merged_into_one_section() -> None:
    generated = payload(
        ("eatwell-guide", [("Choose unsaturated oils.", ["eatwell-guide:34"])]),
        ("eatwell-guide", [("Use small amounts.", ["eatwell-guide:35"])]),
    )

    sections = render_sections(generated, GROUPS)

    assert len(sections) == 1
    assert [claim.text for claim in sections[0].claims] == ["Choose unsaturated oils.", "Use small amounts."]


def test_claim_text_is_stripped_and_chunk_ids_kept_for_the_validator() -> None:
    generated = payload(("eatwell-guide", [("  Choose unsaturated oils.  ", ["eatwell-guide:34", "eatwell-guide:999"])]))

    sections = render_sections(generated, GROUPS)

    assert sections[0].claims[0].text == "Choose unsaturated oils."
    assert sections[0].claims[0].chunk_ids == ["eatwell-guide:34", "eatwell-guide:999"]


def test_order_sections_follows_the_registry() -> None:
    sections = render_sections(
        payload(
            ("kitchen-companion", [("a", ["kitchen-companion:37"])]),
            ("who-healthy-diet", [("b", ["who-healthy-diet:8"])]),
        ),
        GROUPS,
    )
    reordered = order_sections(list(reversed(sections)))
    assert [section.document_id for section in reordered] == ["who-healthy-diet", "kitchen-companion"]


def test_render_text_puts_the_citation_under_each_claim_and_never_merges_documents() -> None:
    sections = render_sections(
        payload(
            ("eatwell-guide", [("Choose unsaturated oils.", ["eatwell-guide:34"])]),
            ("who-healthy-diet", [("Total fat under 30% of energy.", ["who-healthy-diet:8"])]),
        ),
        GROUPS,
    )

    text = render_text(sections)

    who = BY_ID["who-healthy-diet"]
    eatwell = BY_ID["eatwell-guide"]
    assert text.startswith(f"According to {who.document_name} — {who.publisher}, {who.year}")
    assert f"— {who.document_name}, {who.publisher}, {who.year}" in text
    assert who.source_url in text
    assert f"According to {eatwell.document_name} — {eatwell.publisher}, {eatwell.year}" in text
    assert f"— {eatwell.document_name}, {eatwell.publisher}, {eatwell.year}" in text
    assert eatwell.source_url in text
    assert text.index(who.source_url) < text.index("According to The Eatwell Guide")
    assert "the guidelines say" not in text.lower()
    assert "guidelines" not in text.lower()


def test_render_text_of_no_sections_is_empty() -> None:
    assert render_text([]) == ""
