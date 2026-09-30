"""Fixed refusal templates (implementation plan, phase 5; architecture §7, §8).

Every builder returns the ``POST /chat`` response shape from architecture §8
as a plain dict: ``type``, ``reason``, ``message``, ``sections`` (always
empty here), and ``searched``. No builder calls a model.

Three response types live here:

- ``out_of_scope``: the scope guard matched. ``searched`` is empty because
  the corpus was not consulted.
- ``not_in_corpus``: the question was in scope and nothing in the searched
  documents answers it. ``searched`` lists every document in the scope by
  name, publisher, and year. Phase 6 emits this on an empty retrieval and
  when the validator drops every claim.
- ``generation_unavailable``: Groq errored or the quota is exhausted. The
  message does not say the guidance was searched and found empty.
"""

from __future__ import annotations

from typing import Sequence

from src.constants import CORPUS, CorpusDocument
from src.rag.classifier import (
    PROFESSIONAL_REFERRAL_REASONS,
    SCOPE_REASONS,
    ScopeReason,
    registry_order,
)

PROFESSIONAL_REFERRAL_MESSAGE = (
    "I can’t help with medical advice or with personal calorie or weight targets. "
    "Please speak with a qualified health professional."
)
NUTRIENT_LOOKUP_MESSAGE = (
    "This assistant answers from dietary guidance documents and does not look up "
    "nutrient numbers for individual foods, such as the calories or protein in a "
    "specific food."
)
NOT_IN_CORPUS_MESSAGE = "The guidance I searched does not cover that."
NOT_IN_CORPUS_REASON = "no_supporting_chunks"
GENERATION_UNAVAILABLE_MESSAGE = (
    "I found relevant guidance but could not generate an answer right now because "
    "the language model service is unavailable. Please try again in a moment."
)
GENERATION_UNAVAILABLE_REASON = "groq_error"
GENERATION_UNAVAILABLE_REASONS: tuple[str, ...] = ("groq_error", "quota_exceeded", "invalid_json")

_BY_ID: dict[str, CorpusDocument] = {document.document_id: document for document in CORPUS}


def searched_entry(document_id: str) -> dict[str, object]:
    """One ``searched`` row: id, name, publisher, year, from the registry."""
    document = _BY_ID.get(document_id)
    if document is None:
        raise ValueError(f"document_id is not in the registry: {document_id}")
    return {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "publisher": document.publisher,
        "year": document.year,
    }


def searched_entries(document_ids: Sequence[str]) -> list[dict[str, object]]:
    """``searched`` rows in registry order. Unknown ids raise ``ValueError``."""
    return [searched_entry(document_id) for document_id in registry_order(document_ids)]


def searched_line(entry: dict[str, object]) -> str:
    """``- Name, Publisher, Year`` as the not-in-corpus template lists it."""
    return f"- {entry['document_name']}, {entry['publisher']}, {entry['year']}"


def out_of_scope(reason: ScopeReason) -> dict[str, object]:
    """Decline without consulting the corpus. ``searched`` is empty."""
    if reason not in SCOPE_REASONS:
        raise ValueError(f"unknown scope reason: {reason}")
    if reason in PROFESSIONAL_REFERRAL_REASONS:
        message = PROFESSIONAL_REFERRAL_MESSAGE
    else:
        message = NUTRIENT_LOOKUP_MESSAGE
    return {
        "type": "out_of_scope",
        "reason": reason,
        "message": message,
        "sections": [],
        "searched": [],
    }


def not_in_corpus(
    searched: Sequence[str],
    reason: str = NOT_IN_CORPUS_REASON,
) -> dict[str, object]:
    """Name what was searched. ``searched`` is the retrieval scope's ids.

    Pass the one filtered ``document_id`` for a filtered miss, or all seven
    for a whole-corpus miss. The message lists each document's name,
    publisher, and year on its own line.
    """
    if not searched:
        raise ValueError("not_in_corpus needs at least one searched document_id")
    entries = searched_entries(searched)
    lines = [NOT_IN_CORPUS_MESSAGE, "I searched:"] + [searched_line(entry) for entry in entries]
    return {
        "type": "not_in_corpus",
        "reason": reason,
        "message": "\n".join(lines),
        "sections": [],
        "searched": entries,
    }


def generation_unavailable(
    searched: Sequence[str],
    reason: str = GENERATION_UNAVAILABLE_REASON,
) -> dict[str, object]:
    """Groq failed after retrieval found something. Not a corpus miss.

    ``searched`` keeps the scope so the UI can still show the boundary; the
    message does not claim the documents lacked an answer.
    """
    if reason not in GENERATION_UNAVAILABLE_REASONS:
        raise ValueError(f"unknown generation_unavailable reason: {reason}")
    return {
        "type": "generation_unavailable",
        "reason": reason,
        "message": GENERATION_UNAVAILABLE_MESSAGE,
        "sections": [],
        "searched": searched_entries(searched),
    }
