"""HTTP client for ``POST /chat`` (implementation plan, phase 8).

The chat page talks to the API through this module, so the base URL is one
setting in one place and tests can replace ``post_chat`` without a server.
Set ``CHAT_API_BASE_URL`` to point the UI at a different host.

Every transport or HTTP failure becomes a ``ChatError`` carrying a sentence
the page can show. A refusal is not a failure: ``out_of_scope`` and
``not_in_corpus`` arrive as ordinary 200 responses and are returned as they
are.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL_ENV = "CHAT_API_BASE_URL"
# A question that reaches Groq waits on generation, so this is generous.
REQUEST_TIMEOUT_SECONDS = 60.0


class ChatError(RuntimeError):
    """The API could not be reached, or it did not answer with a chat body."""


def api_base_url() -> str:
    """``CHAT_API_BASE_URL`` if set, else the local uvicorn default."""
    return os.environ.get(API_BASE_URL_ENV, DEFAULT_API_BASE_URL).strip().rstrip("/")


def post_chat(
    message: str,
    document_id: str | None = None,
    base_url: str | None = None,
) -> dict[str, object]:
    """Send one message and return the architecture §8 response body.

    ``document_id`` of ``None`` searches all seven documents.
    """
    base = (base_url or api_base_url()).rstrip("/")
    payload = {"message": message, "document_id": document_id}
    try:
        response = httpx.post(
            f"{base}/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise ChatError(
            f"Could not reach the chat API at {base}. Start it with "
            "`uvicorn src.api.main:app` and try again."
        ) from exc

    if response.status_code >= 400:
        raise ChatError(_error_detail(response))
    try:
        body = response.json()
    except ValueError as exc:
        raise ChatError("The chat API returned a body that is not JSON.") from exc
    if not isinstance(body, dict) or "type" not in body:
        raise ChatError("The chat API returned a body without a response type.")
    return body


def _error_detail(response: httpx.Response) -> str:
    """The API's ``detail`` when it sent one, else the status line."""
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict) and first.get("msg"):
            detail = str(first["msg"])
    if isinstance(detail, str) and detail:
        return f"The chat API rejected that request: {detail}"
    return f"The chat API returned HTTP {response.status_code}."
