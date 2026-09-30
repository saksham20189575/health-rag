"""Answer pipeline (implementation plan, phase 6; architecture §5).

Order of operations for one message:

1. Normalize the message. An empty message raises ``ValueError``.
2. Run the scope guard. On a match, return ``out_of_scope`` without retrieval.
3. Resolve ``document_id`` from the request, else from document aliases.
4. Retrieve. On no surviving hit, return ``not_in_corpus`` without Groq.
5. Generate claims, render citations from chunk metadata, validate.
6. A Groq quota, transport, or JSON failure returns ``generation_unavailable``.

The retriever and generator are injected so tests run with fakes and no
network. Both are created lazily when omitted, so importing this module does
not load BGE or read ``GROQ_API_KEY``.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from src.rag import refusal
from src.rag.citations import render_sections, render_text
from src.rag.classifier import classify
from src.rag.generator import ClaimGenerator, GenerationError, GroqClaimGenerator
from src.rag.models import ChatResponse, SearchedEntry
from src.rag.refusal import searched_entries
from src.rag.retriever import RetrievalResult, Retriever
from src.rag.validator import validate_sections

logger = logging.getLogger(__name__)

ALL_CLAIMS_DROPPED_REASON = "all_claims_dropped"


class RetrieverLike(Protocol):
    """Structural type for the retriever: ``retrieve(query, document_id=None)``."""

    def retrieve(self, query: str, document_id: str | None = None) -> RetrievalResult: ...


def normalize_message(message: str) -> str:
    """Collapse whitespace and trim. Case is kept for the retriever and Groq."""
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    return re.sub(r"\s+", " ", message.replace("\u00a0", " ")).strip()


class AnswerPipeline:
    """Guard, retrieve, generate, cite, validate."""

    def __init__(
        self,
        retriever: RetrieverLike | None = None,
        generator: ClaimGenerator | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(self, message: str, document_id: str | None = None) -> ChatResponse:
        """The architecture §8 response for ``message``.

        ``document_id`` outside the registry raises ``ValueError`` before
        anything else runs, as does an empty message.
        """
        text = normalize_message(message)
        if not text:
            raise ValueError("message is empty")

        decision = classify(text, document_id)
        if decision.reason is not None:
            logger.info("out_of_scope reason=%s", decision.reason)
            return ChatResponse.from_refusal(refusal.out_of_scope(decision.reason))

        result = self._retriever_instance().retrieve(text, document_id=decision.document_id)
        hit_count = sum(len(group.hits) for group in result.groups)
        logger.info(
            "retrieved scope=%s candidates=%d hits=%d documents=%d",
            decision.document_id or "all",
            result.candidate_count,
            hit_count,
            len(result.groups),
        )
        if not result.groups:
            return ChatResponse.from_refusal(refusal.not_in_corpus(result.searched))

        try:
            payload = self._generator_instance().generate(text, result.groups)
        except GenerationError as exc:
            logger.warning("generation_unavailable reason=%s detail=%s", exc.reason, exc.detail)
            return ChatResponse.from_refusal(
                refusal.generation_unavailable(result.searched, reason=exc.reason)
            )

        rendered = render_sections(payload, result.groups)
        validated = validate_sections(rendered, result.groups)
        for dropped in validated.dropped:
            logger.info(
                "dropped claim document=%s reason=%s chunk_ids=%s",
                dropped.document_id,
                dropped.reason,
                ",".join(dropped.chunk_ids),
            )
        if not validated.sections:
            return ChatResponse.from_refusal(
                refusal.not_in_corpus(result.searched, reason=ALL_CLAIMS_DROPPED_REASON)
            )

        return ChatResponse(
            type="answer",
            reason=None,
            message=render_text(validated.sections),
            sections=validated.sections,
            searched=[
                SearchedEntry.model_validate(entry) for entry in searched_entries(result.searched)
            ],
        )

    def _retriever_instance(self) -> RetrieverLike:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    def _generator_instance(self) -> ClaimGenerator:
        if self._generator is None:
            self._generator = GroqClaimGenerator()
        return self._generator


_default_pipeline: AnswerPipeline | None = None


def default_pipeline() -> AnswerPipeline:
    """One process-wide pipeline so BGE and the Groq client load once."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = AnswerPipeline()
    return _default_pipeline


def answer(message: str, document_id: str | None = None) -> ChatResponse:
    """Module-level entry point used by the API: ``answer(message, document_id=None)``."""
    return default_pipeline().answer(message, document_id=document_id)
