"""Refusal templates (implementation plan, phase 5; architecture §7 and §8)."""

from __future__ import annotations

import json

import pytest

from src.constants import CORPUS
from src.rag import refusal
from src.rag.classifier import classify
from src.rag.refusal import (
    GENERATION_UNAVAILABLE_MESSAGE,
    NOT_IN_CORPUS_MESSAGE,
    NUTRIENT_LOOKUP_MESSAGE,
    PROFESSIONAL_REFERRAL_MESSAGE,
    generation_unavailable,
    not_in_corpus,
    out_of_scope,
    searched_entries,
)

ALL_IDS = tuple(document.document_id for document in CORPUS)
RESPONSE_KEYS = {"type", "reason", "message", "sections", "searched"}


def _is_json(response: dict[str, object]) -> bool:
    json.dumps(response)
    return True


# --- out_of_scope -------------------------------------------------------------


@pytest.mark.parametrize("reason", ["medical", "calorie_target", "weight_target"])
def test_out_of_scope_refers_to_a_professional(reason: str) -> None:
    response = out_of_scope(reason)
    assert set(response) == RESPONSE_KEYS
    assert response["type"] == "out_of_scope"
    assert response["reason"] == reason
    assert response["message"] == PROFESSIONAL_REFERRAL_MESSAGE
    assert "qualified health professional" in response["message"]
    assert response["sections"] == []
    assert response["searched"] == []
    assert _is_json(response)


def test_out_of_scope_nutrient_lookup_has_its_own_sentence() -> None:
    response = out_of_scope("nutrient_lookup")
    assert response["type"] == "out_of_scope"
    assert response["reason"] == "nutrient_lookup"
    assert response["message"] == NUTRIENT_LOOKUP_MESSAGE
    assert "dietary guidance documents" in response["message"]
    assert "nutrient numbers for individual foods" in response["message"]
    assert "professional" not in response["message"]
    assert response["searched"] == []


def test_out_of_scope_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError):
        out_of_scope("no_supporting_chunks")  # type: ignore[arg-type]


def test_weight_question_end_to_end_through_guard_and_template() -> None:
    """Exit criterion: "What should I weigh?" is weight_target with the referral."""
    decision = classify("What should I weigh?")
    assert decision.reason == "weight_target"
    response = out_of_scope(decision.reason)
    assert response["type"] == "out_of_scope"
    assert response["reason"] == "weight_target"
    assert "qualified health professional" in response["message"]
    assert response["searched"] == []


# --- not_in_corpus ------------------------------------------------------------


def test_not_in_corpus_lists_all_seven_by_name_publisher_year() -> None:
    response = not_in_corpus(ALL_IDS)
    assert set(response) == RESPONSE_KEYS
    assert response["type"] == "not_in_corpus"
    assert response["reason"] == "no_supporting_chunks"
    assert response["sections"] == []
    assert len(response["searched"]) == 7
    assert [entry["document_id"] for entry in response["searched"]] == list(ALL_IDS)

    message = response["message"]
    assert message.startswith(NOT_IN_CORPUS_MESSAGE)
    assert "I searched:" in message
    for document in CORPUS:
        assert f"- {document.document_name}, {document.publisher}, {document.year}" in message
    assert _is_json(response)


def test_not_in_corpus_searched_entries_carry_registry_fields() -> None:
    response = not_in_corpus(ALL_IDS)
    by_id = {document.document_id: document for document in CORPUS}
    for entry in response["searched"]:
        document = by_id[entry["document_id"]]
        assert set(entry) == {"document_id", "document_name", "publisher", "year"}
        assert entry["document_name"] == document.document_name
        assert entry["publisher"] == document.publisher
        assert entry["year"] == document.year
        assert isinstance(entry["year"], int)


def test_not_in_corpus_filtered_lists_only_that_document() -> None:
    response = not_in_corpus(("eatwell-guide",))
    assert [entry["document_id"] for entry in response["searched"]] == ["eatwell-guide"]
    message = response["message"]
    assert "The Eatwell Guide booklet, Public Health England (now OHID), 2018" in message
    assert "Kitchen Companion" not in message
    assert message.count("\n- ") == 1


def test_not_in_corpus_orders_searched_by_registry() -> None:
    response = not_in_corpus(("kitchen-companion", "cold-food-storage", "eatwell-guide"))
    assert [entry["document_id"] for entry in response["searched"]] == [
        "cold-food-storage",
        "eatwell-guide",
        "kitchen-companion",
    ]


def test_not_in_corpus_accepts_a_phase6_reason() -> None:
    response = not_in_corpus(ALL_IDS, reason="all_claims_dropped")
    assert response["reason"] == "all_claims_dropped"


def test_not_in_corpus_rejects_empty_or_unknown_scope() -> None:
    with pytest.raises(ValueError):
        not_in_corpus(())
    with pytest.raises(ValueError):
        not_in_corpus(("not-a-document",))


# --- generation_unavailable ----------------------------------------------------


def test_generation_unavailable_does_not_claim_an_empty_search() -> None:
    response = generation_unavailable(ALL_IDS)
    assert set(response) == RESPONSE_KEYS
    assert response["type"] == "generation_unavailable"
    assert response["reason"] == "groq_error"
    assert response["message"] == GENERATION_UNAVAILABLE_MESSAGE
    assert "does not cover" not in response["message"]
    assert "I searched:" not in response["message"]
    assert response["sections"] == []
    assert len(response["searched"]) == 7
    assert _is_json(response)


def test_generation_unavailable_quota_reason_and_filtered_scope() -> None:
    response = generation_unavailable(("fsa-chill",), reason="quota_exceeded")
    assert response["reason"] == "quota_exceeded"
    assert [entry["document_id"] for entry in response["searched"]] == ["fsa-chill"]


def test_generation_unavailable_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError):
        generation_unavailable(ALL_IDS, reason="no_supporting_chunks")


# --- shared -------------------------------------------------------------------


def test_searched_entries_dedupe_and_order() -> None:
    entries = searched_entries(("eatwell-guide", "eatwell-guide", "cold-food-storage"))
    assert [entry["document_id"] for entry in entries] == ["cold-food-storage", "eatwell-guide"]


def test_the_three_types_are_distinct() -> None:
    types = {
        out_of_scope("medical")["type"],
        not_in_corpus(ALL_IDS)["type"],
        generation_unavailable(ALL_IDS)["type"],
    }
    assert types == {"out_of_scope", "not_in_corpus", "generation_unavailable"}


def test_refusal_module_has_no_model_client() -> None:
    import sys

    assert "groq" not in sys.modules or "groq" not in refusal.__dict__
    assert not hasattr(refusal, "Groq")
