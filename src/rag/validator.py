"""Draft validator (implementation plan, phase 6; architecture §5.6).

Runs after the renderer. A claim is dropped when any check fails:

- ``unknown_chunk_id``: a cited chunk id is not in that document's retrieved
  set (this also catches a chunk id borrowed from another document, and a
  claim with no chunk ids).
- ``empty_text``: the claim text is blank.
- ``unsupported_numbers``: a number in the claim is absent from the cited
  chunk text.
- ``unsupported_terms``: fewer than ``MIN_CLAIM_TERM_OVERLAP`` of the claim's
  content words appear in the cited chunk text.
- ``citation_url_mismatch``: the rendered URL is not the registry
  ``source_url`` for that document.
- ``out_of_scope_claim``: the claim text matches the medical, calorie-target,
  or weight-target patterns from the scope guard.

A section with no surviving claim is removed. When no section survives the
pipeline emits ``not_in_corpus``; this module does not build refusals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.constants import CORPUS, MIN_CLAIM_TERM_OVERLAP
from src.rag.classifier import PROFESSIONAL_REFERRAL_REASONS, classify_scope
from src.rag.models import Claim, Section
from src.rag.retriever import DocumentHits, RetrievedChunk

_SOURCE_URL_BY_ID: dict[str, str] = {document.document_id: document.source_url for document in CORPUS}

DROP_REASONS: tuple[str, ...] = (
    "unknown_chunk_id",
    "empty_text",
    "unsupported_numbers",
    "unsupported_terms",
    "citation_url_mismatch",
    "out_of_scope_claim",
)

# Function words and generic verbs that say nothing about which chunk a
# claim came from. Content words are everything else of three letters or more.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but nor so yet of in on at to for from by with without within into onto
    as is are was were be been being am do does did done have has had having will would shall
    should can could may might must ought it its this that these those there here where when
    which who whom whose what why how all any both each few more most other some such no not
    only own same than too very just also about above after again against before below between
    during over under up down out off once then than because while until per via
    you your yours they them their theirs he him his she her hers we us our ours i me my mine
    if unless whether either neither
    should recommend recommends recommended advise advises advised say says said state states
    stated according guidance guideline guidelines document passage passages
    keep kept make makes made use used using get gets got take takes taken give gives given
    """.split()
)

_ALIASES: dict[str, str] = {
    "fridge": "refrigerator",
    "fridges": "refrigerator",
    "refrigerate": "refrigerator",
    "refrigerated": "refrigerator",
    "refrigerating": "refrigerator",
    "refrigeration": "refrigerator",
    "freeze": "freezer",
    "frozen": "freezer",
    "freezing": "freezer",
    "veg": "vegetable",
    "veggies": "vegetable",
}

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# The scope guard's patterns are written for first-person questions. A
# generated claim addresses the reader, so a personal target inside an answer
# is second person: "you should weigh 70 kg", "you need 2000 calories a day".
# Population wording ("adults need around 2000 kcal") does not match.
_CALORIE_WORD = r"(?:calories?|kcal|kilocalories?)"
_CLAIM_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\byou (?:should|ought to|need to|must|are expected to) (?:weigh|aim to weigh)\b",
        r"\byour (?:ideal|target|goal|recommended|healthy|correct|optimal) (?:body )?weight (?:is|should|would)\b",
        r"\byou should (?:lose|gain|drop|shed|put on) (?:about |around |at least )?\d",
        r"\byour bmi (?:is|should)\b",
        rf"\byou (?:should|need|need to|must|ought to|require)(?: (?:eat|consume|have|aim for|get|take in|stick to))?"
        rf"\b[^.;]{{0,25}}?\b\d[\d,]* ?{_CALORIE_WORD}\b",
        rf"\byour (?:daily )?(?:{_CALORIE_WORD}|energy) (?:target|goal|needs?|requirement|allowance|intake) (?:is|should be|would be)\b",
        rf"\b(?:a |an )?(?:deficit|surplus) of \d[\d,]* ?{_CALORIE_WORD}\b",
    )
)


@dataclass(frozen=True)
class DroppedClaim:
    """One claim the validator removed, for the phase 9 log."""

    document_id: str
    chunk_ids: tuple[str, ...]
    reason: str


@dataclass
class ValidationResult:
    sections: list[Section]
    dropped: list[DroppedClaim] = field(default_factory=list)


def _numbers(text: str) -> set[str]:
    return {match.replace(",", "") for match in _NUMBER.findall(text)}


def _stem(word: str) -> str:
    word = _ALIASES.get(word, word)
    for suffix in ("ies", "ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _terms(text: str) -> set[str]:
    words = _WORD.findall(text.lower())
    return {_stem(word) for word in words if len(word) >= 3 and word not in _STOPWORDS}


def _term_present(term: str, chunk_terms: set[str]) -> bool:
    if term in chunk_terms:
        return True
    # Loose stemming: "dishe" vs "dish", "vegetable" vs "vegetables", "cook" vs "cooked".
    if len(term) >= 4:
        return any(
            len(other) >= 4 and (other.startswith(term) or term.startswith(other))
            for other in chunk_terms
        )
    return False


def support_failure(claim_text: str, chunk_texts: Iterable[str]) -> str | None:
    """``None`` when the claim's numbers and terms appear in the cited text.

    Otherwise ``unsupported_numbers`` or ``unsupported_terms``.
    """
    cited = " ".join(chunk_texts)
    cited_numbers = _numbers(cited)
    for number in _numbers(claim_text):
        if number not in cited_numbers:
            return "unsupported_numbers"
    claim_terms = _terms(claim_text)
    if not claim_terms:
        return "unsupported_terms"
    cited_terms = _terms(cited)
    present = sum(1 for term in claim_terms if _term_present(term, cited_terms))
    if present / len(claim_terms) < MIN_CLAIM_TERM_OVERLAP:
        return "unsupported_terms"
    return None


def claim_is_out_of_scope(text: str) -> bool:
    """True when the claim reads as medical advice or a personal calorie or weight target.

    Uses the scope guard's medical, calorie-target, and weight-target patterns,
    plus the second-person forms above. The nutrient-lookup patterns are not
    applied: a cited "less than 5 g of salt per day" is population guidance.
    """
    match = classify_scope(text)
    if match is not None and match[0] in PROFESSIONAL_REFERRAL_REASONS:
        return True
    return any(pattern.search(text) for pattern in _CLAIM_TARGET_PATTERNS)


def drop_reason(claim: Claim, document_id: str, retrieved: dict[str, RetrievedChunk]) -> str | None:
    """The first failing check for ``claim`` against ``document_id``'s retrieved set."""
    if not claim.chunk_ids or any(chunk_id not in retrieved for chunk_id in claim.chunk_ids):
        return "unknown_chunk_id"
    if not claim.text.strip():
        return "empty_text"
    failure = support_failure(claim.text, (retrieved[chunk_id].text for chunk_id in claim.chunk_ids))
    if failure is not None:
        return failure
    if claim.citation.source_url != _SOURCE_URL_BY_ID.get(document_id):
        return "citation_url_mismatch"
    if claim_is_out_of_scope(claim.text):
        return "out_of_scope_claim"
    return None


def validate_sections(sections: Sequence[Section], groups: Sequence[DocumentHits]) -> ValidationResult:
    """Drop failing claims, then drop sections left with none."""
    retrieved_by_document: dict[str, dict[str, RetrievedChunk]] = {
        group.document_id: {hit.chunk_id: hit for hit in group.hits} for group in groups
    }
    kept_sections: list[Section] = []
    dropped: list[DroppedClaim] = []
    for section in sections:
        retrieved = retrieved_by_document.get(section.document_id, {})
        kept_claims: list[Claim] = []
        for claim in section.claims:
            reason = drop_reason(claim, section.document_id, retrieved)
            if reason is None:
                kept_claims.append(claim)
            else:
                dropped.append(DroppedClaim(section.document_id, tuple(claim.chunk_ids), reason))
        if kept_claims:
            kept_sections.append(section.model_copy(update={"claims": kept_claims}))
    return ValidationResult(sections=kept_sections, dropped=dropped)
