"""Response and generator payload models (implementation plan, phase 6).

``ChatResponse`` is the ``POST /chat`` body from architecture §8. Every
response carries ``type``, ``sections``, and ``searched``; refusals also
carry ``reason`` and ``message``. An answer carries a plain-text ``message``
rendered by ``src.rag.citations`` so a client without a section renderer can
still show it.

``GeneratedPayload`` is the JSON shape Groq must return (architecture §5.4).
It is validated in strict mode: a claim's ``text`` must be a string and its
``chunk_ids`` a list of strings, or the whole payload is invalid JSON for the
purposes of the pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ResponseType = Literal["answer", "not_in_corpus", "out_of_scope", "generation_unavailable"]

RESPONSE_TYPES: tuple[str, ...] = ("answer", "not_in_corpus", "out_of_scope", "generation_unavailable")


class SearchedEntry(BaseModel):
    """One document in the retrieval scope: id, name, publisher, year."""

    document_id: str
    document_name: str
    publisher: str
    year: int


class Citation(BaseModel):
    """What the renderer attaches to a claim, filled from the chunk record."""

    document_name: str
    publisher: str
    year: int
    source_url: str


class Claim(BaseModel):
    """One statement, the chunk ids that support it, and its citation."""

    text: str
    chunk_ids: list[str]
    citation: Citation


class Section(BaseModel):
    """All surviving claims from one document. Sections are in registry order."""

    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    claims: list[Claim]


class ChatResponse(BaseModel):
    """The architecture §8 response body for every path through the pipeline."""

    type: ResponseType
    reason: str | None = None
    message: str | None = None
    sections: list[Section] = Field(default_factory=list)
    searched: list[SearchedEntry] = Field(default_factory=list)

    @classmethod
    def from_refusal(cls, refusal: dict[str, object]) -> "ChatResponse":
        """Wrap a ``src.rag.refusal`` dict. Raises ``ValidationError`` on a bad shape."""
        return cls.model_validate(refusal)


# --- Groq payload (architecture §5.4) -----------------------------------------


class GeneratedClaim(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    text: str
    chunk_ids: list[str]


class GeneratedDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    document_id: str
    claims: list[GeneratedClaim]


class GeneratedPayload(BaseModel):
    """``{"documents": [{"document_id", "claims": [{"text", "chunk_ids"}]}]}``."""

    model_config = ConfigDict(strict=True, extra="ignore")

    documents: list[GeneratedDocument]
