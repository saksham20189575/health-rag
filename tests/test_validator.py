"""Draft validator (implementation plan, phase 6; architecture §5.6)."""

from __future__ import annotations

import pytest

from src.constants import CORPUS
from src.rag.models import Citation, Claim, Section
from src.rag.retriever import DocumentHits, RetrievedChunk
from src.rag.validator import (
    claim_is_out_of_scope,
    drop_reason,
    support_failure,
    validate_sections,
)

BY_ID = {document.document_id: document for document in CORPUS}

LEFTOVERS_ROW = (
    "Food: Leftovers | Type: Cooked meat or poultry | "
    "Refrigerator [40°F (4°C) or below]: 3 to 4 days | Freezer [0°F (-18°C) or below]: 2 to 6 months"
)
WHOLE_CHICKEN_ROW = (
    "Food: Fresh poultry | Type: Chicken or turkey, whole | "
    "Refrigerator [40°F (4°C) or below]: 1 to 2 days | Freezer [0°F (-18°C) or below]: 1 year"
)
SALT_TEXT = "Adults should consume less than 5 g of salt (equivalent to about one teaspoon) per day."


def chunk(document_id: str, ordinal: int, text: str) -> RetrievedChunk:
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
        block_type="table_row",
        retrieval_date="2026-09-27",
    )


def citation(document_id: str, source_url: str | None = None) -> Citation:
    document = BY_ID[document_id]
    return Citation(
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=source_url or document.source_url,
    )


def claim(document_id: str, text: str, chunk_ids: list[str], source_url: str | None = None) -> Claim:
    return Claim(text=text, chunk_ids=chunk_ids, citation=citation(document_id, source_url))


def section(document_id: str, claims: list[Claim]) -> Section:
    document = BY_ID[document_id]
    return Section(
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        claims=claims,
    )


GROUPS = (
    DocumentHits(
        "cold-food-storage",
        (chunk("cold-food-storage", 51, LEFTOVERS_ROW), chunk("cold-food-storage", 24, WHOLE_CHICKEN_ROW)),
    ),
    DocumentHits("who-healthy-diet", (chunk("who-healthy-diet", 9, SALT_TEXT),)),
)
RETRIEVED = {hit.chunk_id: hit for group in GROUPS for hit in group.hits}


# --- support check -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.",
        "Store cooked leftovers in the fridge for up to 4 days.",
        "Leftover cooked meat can be frozen for 2 to 6 months.",
    ],
)
def test_supported_claims_pass(text: str) -> None:
    assert support_failure(text, [LEFTOVERS_ROW]) is None


def test_a_number_missing_from_the_cited_chunk_fails() -> None:
    assert support_failure("Cooked leftovers keep for 48 hours.", [LEFTOVERS_ROW]) == "unsupported_numbers"
    assert support_failure("Cooked leftovers keep 3 to 5 days.", [LEFTOVERS_ROW]) == "unsupported_numbers"


def test_unrelated_terms_fail() -> None:
    assert support_failure("Import tariffs on olive oil are set by customs.", [LEFTOVERS_ROW]) == "unsupported_terms"
    assert support_failure("Yes.", [LEFTOVERS_ROW]) == "unsupported_terms"
    assert support_failure("", [LEFTOVERS_ROW]) == "unsupported_terms"


def test_food_term_from_a_sibling_row_is_checked_against_the_cited_row_only() -> None:
    # The whole-chicken figures are in :24, not in the cited leftovers row.
    assert support_failure("A whole chicken keeps 1 to 2 days in the refrigerator.", [LEFTOVERS_ROW]) == "unsupported_numbers"
    assert support_failure("A whole chicken keeps 1 to 2 days in the refrigerator.", [WHOLE_CHICKEN_ROW]) is None


def test_numbers_with_thousands_separators_and_decimals() -> None:
    assert support_failure("Energy needs are about 2,000 kcal.", ["Women need around 2000 kcal a day."]) is None
    assert support_failure("Use 0.5 teaspoons.", ["Add 0.5 teaspoons of salt."]) is None
    assert support_failure("Use 0.5 teaspoons.", ["Add 1.5 teaspoons of salt."]) == "unsupported_numbers"


# --- out-of-scope claims -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "What should I weigh?",
        "You should weigh 70 kg.",
        "Your ideal weight is 65 kg.",
        "You need about 2000 calories a day.",
        "You should eat 1,800 kcal to lose weight.",
        "If you have diabetes, avoid sugar.",
        "Take insulin before meals.",
    ],
)
def test_personal_target_and_medical_claims_are_out_of_scope(text: str) -> None:
    assert claim_is_out_of_scope(text)


@pytest.mark.parametrize(
    "text",
    [
        "Adults should eat less than 5 g of salt per day.",
        "Intake of saturated fats should be less than 10% of total energy intake.",
        "Women need around 2000 kcal a day and men around 2500 kcal a day.",
        "Maintaining a healthy body weight helps prevent disease.",
        "Energy intake (calories) should be in balance with energy expenditure.",
        "Eat at least 5 portions of a variety of fruit and vegetables every day.",
        "Keep hot food above 60°C and cold food below 5°C.",
        "You should wash your hands before handling food.",
        "Protein in 100 g of chicken is not covered.",  # nutrient_lookup is not a claim drop reason
    ],
)
def test_population_guidance_claims_stay_in_scope(text: str) -> None:
    assert not claim_is_out_of_scope(text)


# --- drop reasons --------------------------------------------------------------


def test_good_claim_has_no_drop_reason() -> None:
    good = claim("cold-food-storage", "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"])
    assert drop_reason(good, "cold-food-storage", RETRIEVED) is None


def test_chunk_id_not_retrieved_for_that_document() -> None:
    bad = claim("cold-food-storage", "Leftovers keep 3 to 4 days.", ["cold-food-storage:999"])
    assert drop_reason(bad, "cold-food-storage", RETRIEVED) == "unknown_chunk_id"


def test_chunk_id_from_another_document_is_unknown_for_this_one() -> None:
    borrowed = claim("cold-food-storage", "Leftovers keep 3 to 4 days.", ["who-healthy-diet:9"])
    own_retrieved = {hit.chunk_id: hit for hit in GROUPS[0].hits}
    assert drop_reason(borrowed, "cold-food-storage", own_retrieved) == "unknown_chunk_id"


def test_no_chunk_ids_is_unknown_chunk_id() -> None:
    bare = claim("cold-food-storage", "Leftovers keep 3 to 4 days.", [])
    assert drop_reason(bare, "cold-food-storage", RETRIEVED) == "unknown_chunk_id"


def test_empty_text_is_dropped() -> None:
    blank = claim("cold-food-storage", "   ", ["cold-food-storage:51"])
    assert drop_reason(blank, "cold-food-storage", RETRIEVED) == "empty_text"


def test_citation_url_outside_the_allowlist_is_dropped() -> None:
    wrong = claim(
        "cold-food-storage",
        "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.",
        ["cold-food-storage:51"],
        source_url="https://example.com/not-the-registry",
    )
    assert drop_reason(wrong, "cold-food-storage", RETRIEVED) == "citation_url_mismatch"


def test_another_documents_registry_url_is_still_a_mismatch() -> None:
    swapped = claim(
        "cold-food-storage",
        "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.",
        ["cold-food-storage:51"],
        source_url=BY_ID["kitchen-companion"].source_url,
    )
    assert drop_reason(swapped, "cold-food-storage", RETRIEVED) == "citation_url_mismatch"


def test_out_of_scope_claim_text_is_dropped_even_when_supported() -> None:
    weight_chunk = chunk("eatwell-guide", 26, "You should weigh 70 kg according to this chart.")
    personal = claim("eatwell-guide", "You should weigh 70 kg.", ["eatwell-guide:26"])
    assert drop_reason(personal, "eatwell-guide", {weight_chunk.chunk_id: weight_chunk}) == "out_of_scope_claim"


# --- validate_sections ---------------------------------------------------------


def test_sections_keep_only_surviving_claims_and_report_drops() -> None:
    sections = [
        section(
            "cold-food-storage",
            [
                claim("cold-food-storage", "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"]),
                claim("cold-food-storage", "Leftovers keep 48 hours.", ["cold-food-storage:51"]),
            ],
        ),
        section(
            "who-healthy-diet",
            [claim("who-healthy-diet", "Adults should consume less than 5 g of salt per day.", ["who-healthy-diet:999"])],
        ),
    ]

    validated = validate_sections(sections, GROUPS)

    assert [item.document_id for item in validated.sections] == ["cold-food-storage"]
    assert len(validated.sections[0].claims) == 1
    assert {(item.document_id, item.reason) for item in validated.dropped} == {
        ("cold-food-storage", "unsupported_numbers"),
        ("who-healthy-diet", "unknown_chunk_id"),
    }


def test_section_for_a_document_not_retrieved_is_dropped_entirely() -> None:
    sections = [
        section("eatwell-guide", [claim("eatwell-guide", "Choose unsaturated oils.", ["eatwell-guide:34"])]),
    ]

    validated = validate_sections(sections, GROUPS)

    assert validated.sections == []
    assert validated.dropped[0].reason == "unknown_chunk_id"


def test_validate_does_not_mutate_the_input_sections() -> None:
    original = section(
        "cold-food-storage",
        [
            claim("cold-food-storage", "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"]),
            claim("cold-food-storage", "Leftovers keep 48 hours.", ["cold-food-storage:51"]),
        ],
    )

    validate_sections([original], GROUPS)

    assert len(original.claims) == 2
