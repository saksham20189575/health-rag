"""Groq claim generator (implementation plan, phase 6; architecture §5.4).

The prompt carries the question, each retrieved document's ``document_id``,
and that document's chunk ids and chunk text. Nothing else: no document
name, publisher, year, or URL, so the model cannot echo a citation. The
application writes citations from chunk metadata in ``src.rag.citations``.

Groq must return the JSON object in ``GeneratedPayload``. The reply is parsed
strictly: text that is not a JSON object of that shape is a generation
failure. Quota, transport, and shape failures all raise ``GenerationError``
with a reason the pipeline maps onto the ``generation_unavailable`` template.
"""

from __future__ import annotations

import json
import os
from typing import Protocol, Sequence

from pydantic import ValidationError

from src.constants import GROQ_DEFAULT_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE
from src.rag.models import GeneratedPayload
from src.rag.refusal import GENERATION_UNAVAILABLE_REASONS
from src.rag.retriever import DocumentHits

SYSTEM_PROMPT = """You answer questions about food, nutrition, and food safety using only the passages you are given.

Return one JSON object with exactly this shape and nothing else:
{"documents": [{"document_id": "<id>", "claims": [{"text": "<one factual statement>", "chunk_ids": ["<chunk id>"]}]}]}

Rules:
1. Use only the passages. Do not add facts, numbers, units, or advice that are not written in them.
2. Each claim states what one document's passages say, in one or two sentences. Copy numbers, units, temperatures, times, and food names exactly as they appear in the passage.
3. A claim's chunk_ids must be chosen from the chunk ids listed under that same document_id. Never cite a chunk id from a different document. Every claim needs at least one chunk id.
4. Write a separate entry for each document_id that answers the question. Give an empty claims list for a document whose passages do not answer it. Do not invent a document_id.
5. Do not name the document, publisher, or organisation, and do not add links. The application adds citations.
6. Do not give medical advice, a personal calorie target, or a personal weight target, even if a passage mentions weight or calories.
7. Do not merge different documents into one statement about what "the guidelines say"."""


class GenerationError(Exception):
    """Groq could not produce a usable payload. ``reason`` names why."""

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in GENERATION_UNAVAILABLE_REASONS:
            raise ValueError(f"unknown generation reason: {reason}")
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class ClaimGenerator(Protocol):
    """Turns a question and grouped chunks into a ``GeneratedPayload``."""

    def generate(self, question: str, groups: Sequence[DocumentHits]) -> GeneratedPayload:
        """Raise ``GenerationError`` when Groq fails or returns the wrong shape."""


def build_passages(groups: Sequence[DocumentHits]) -> str:
    """The passages block: ``document_id`` headers and ``[chunk_id] text`` lines only."""
    blocks: list[str] = []
    for group in groups:
        lines = [f"document_id: {group.document_id}"]
        for hit in group.hits:
            lines.append(f"[{hit.chunk_id}] {hit.text.strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(question: str, groups: Sequence[DocumentHits]) -> list[dict[str, str]]:
    """Chat messages for Groq: the fixed system prompt, then question and passages."""
    if not groups:
        raise ValueError("build_messages needs at least one retrieved group")
    user = f"Question: {question.strip()}\n\nPassages:\n\n{build_passages(groups)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_payload(content: str | None) -> GeneratedPayload:
    """Strictly parse Groq's reply. Anything but the §5.4 object is ``invalid_json``."""
    if content is None or not content.strip():
        raise GenerationError("invalid_json", "empty completion")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GenerationError("invalid_json", f"not JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise GenerationError("invalid_json", "top level is not an object")
    try:
        return GeneratedPayload.model_validate(data)
    except ValidationError as exc:
        raise GenerationError("invalid_json", f"wrong shape: {exc.error_count()} errors") from exc


class GroqClaimGenerator:
    """Calls Groq chat completions in JSON mode. A passed-in client is used as-is."""

    def __init__(
        self,
        client: object | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._api_key = api_key

    @property
    def model(self) -> str:
        """Argument, then ``GROQ_MODEL`` from the environment or ``.env``, then the default.

        ``.env`` is loaded here, not at import, so a missing key does not fail
        tests that never call Groq. An explicit ``model`` argument wins.
        """
        if self._model is None:
            from dotenv import load_dotenv

            load_dotenv()
            self._model = os.environ.get("GROQ_MODEL") or GROQ_DEFAULT_MODEL
        return self._model

    def generate(self, question: str, groups: Sequence[DocumentHits]) -> GeneratedPayload:
        import groq

        messages = build_messages(question, groups)
        try:
            completion = self._groq_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except groq.RateLimitError as exc:
            raise GenerationError("quota_exceeded", str(exc)) from exc
        except groq.GroqError as exc:
            raise GenerationError("groq_error", str(exc)) from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise GenerationError("invalid_json", "completion has no message content") from exc
        return parse_payload(content)

    def _groq_client(self) -> object:
        if self._client is None:
            import httpx
            from dotenv import load_dotenv
            from groq import Groq

            from src.hf_ssl import verified_ssl_context

            load_dotenv()
            api_key = self._api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise GenerationError("groq_error", "GROQ_API_KEY is not set")
            self._client = Groq(
                api_key=api_key,
                http_client=httpx.Client(verify=verified_ssl_context()),
            )
        return self._client
