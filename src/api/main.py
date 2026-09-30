"""HTTP entry point for the answer pipeline (implementation plan, phase 7).

``POST /chat`` takes a ``message`` and an optional ``document_id`` and returns
the architecture §8 response body unchanged: ``type``, ``reason``,
``message``, ``sections``, and ``searched``. Request validation is the only
thing this layer adds. An empty message and a ``document_id`` outside the
registry are rejected with HTTP 422 before the pipeline runs, so neither
reaches the retriever.

``GET /health`` reports whether the Chroma directory exists. The pipeline is
resolved through a dependency so tests can override it with a stub, and it is
built lazily so importing this module does not load BGE or read
``GROQ_API_KEY``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.constants import DOCUMENT_IDS, INDEX_DIR
from src.rag.models import ChatResponse
from src.rag.pipeline import default_pipeline, normalize_message

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000

HTTP_OK = 200
HTTP_UNPROCESSABLE = 422
HTTP_UNAVAILABLE = 503


class PipelineLike(Protocol):
    """Structural type for the answer pipeline: ``answer(message, document_id=None)``."""

    def answer(self, message: str, document_id: str | None = None) -> ChatResponse: ...


class ChatRequest(BaseModel):
    """``{"message": "...", "document_id": null}`` from architecture §8."""

    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    document_id: str | None = None

    @field_validator("message")
    @classmethod
    def _message_is_not_empty(cls, value: str) -> str:
        normalized = normalize_message(value)
        if not normalized:
            raise ValueError("message must not be empty")
        return normalized

    @field_validator("document_id")
    @classmethod
    def _document_id_is_in_the_registry(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in DOCUMENT_IDS:
            raise ValueError(f"document_id is not in the registry: {value}")
        return value


def get_pipeline() -> PipelineLike:
    """The process-wide pipeline. Overridden in tests with a stub."""
    return default_pipeline()


app = FastAPI(
    title="Dietary guidance assistant",
    description="Answers questions from seven official dietary guidance documents, with citations.",
)


@app.exception_handler(ValueError)
async def _value_error_is_unprocessable(request: Request, exc: ValueError) -> JSONResponse:
    """The pipeline rejects an empty message and an unknown document_id too.

    Request validation catches both before the call, so reaching here means a
    disagreement between the two layers. Report it as a bad request rather
    than a server error.
    """
    logger.warning("rejected %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=HTTP_UNPROCESSABLE,
        content={"detail": str(exc)},
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    pipeline: Annotated[PipelineLike, Depends(get_pipeline)],
) -> ChatResponse:
    """Answer one message, or refuse. The response type carries which happened."""
    response = pipeline.answer(request.message, document_id=request.document_id)
    logger.info(
        "chat type=%s reason=%s scope=%s sections=%d",
        response.type,
        response.reason,
        request.document_id or "all",
        len(response.sections),
    )
    return response


@app.get("/health")
def health() -> JSONResponse:
    """``ok`` when the Chroma directory exists, ``unavailable`` when it does not."""
    index_exists = INDEX_DIR.is_dir()
    return JSONResponse(
        status_code=HTTP_OK if index_exists else HTTP_UNAVAILABLE,
        content={
            "status": "ok" if index_exists else "unavailable",
            "index_dir": str(INDEX_DIR),
            "index_present": index_exists,
        },
    )
