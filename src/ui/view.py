"""Widgets and renderers for the chat page (implementation plan, phase 8).

The page is split so tests can import these names without executing a
Streamlit script: ``src/ui/app.py`` is the entry point and does nothing but
wire the pieces below together.

An answer renders one block per section with the citation under each claim,
read from the API body and never rewritten here. The two refusals are
deliberately different states: ``not_in_corpus`` is a warning that lists the
documents searched, ``out_of_scope`` is a decline with a professional
referral and no document list, because the corpus was not consulted.

Two ways to reach the pipeline (docs/deployment-plan.md, phases D0 and D3):

- Default: ``client.post_chat`` over HTTP to the FastAPI server. This is the
  laptop run, ``uvicorn`` first and then ``python scripts/run_ui.py``.
- ``CHAT_IN_PROCESS=1``: call ``AnswerPipeline.answer`` in this process and
  never open a socket. Streamlit Community Cloud runs one process, so this
  is the cloud setting. Off unless set, so the local run is unchanged.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st

from src.constants import CORPUS, INDEX_DIR
from src.ui import client

if TYPE_CHECKING:  # pragma: no cover
    from src.rag.pipeline import AnswerPipeline

PAGE_TITLE = "Dietary guidance assistant"
DISCLAIMER = (
    "Answers come only from the seven official guidance documents in the picker "
    "below, and every claim is cited. This is not medical advice, and this "
    "assistant does not set personal calorie or weight targets or look up "
    "nutrient numbers for individual foods."
)
ALL_DOCUMENTS_LABEL = "All documents"
CHAT_PLACEHOLDER = "Ask about food, nutrition, or food safety"
SPINNER_TEXT = "Searching the guidance…"
# Shown for the first in-process question, which downloads and loads BGE.
MODEL_LOADING_TEXT = "Loading the search model, then searching the guidance…"
TURNS_KEY = "turns"
PICKER_KEY = "document_choice"

IN_PROCESS_ENV = "CHAT_IN_PROCESS"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
# Copied from ``st.secrets`` into the environment when the environment lacks
# them. ``GroqClaimGenerator`` reads ``os.environ``; Community Cloud also
# injects top-level secrets as environment variables, so this covers a local
# ``.streamlit/secrets.toml`` run.
SECRET_ENV_KEYS: tuple[str, ...] = ("GROQ_API_KEY", "GROQ_MODEL", IN_PROCESS_ENV)
MISSING_INDEX_ERROR = (
    f"The search index is missing at {INDEX_DIR}. Run `python -m scripts.build_index` "
    "locally and deploy with `data/index/` in the repository."
)

# The three paths phase 8 asks the chips to demonstrate: a cross-document
# fridge-time answer, an Eatwell question the alias detector filters, and the
# out-of-scope refusal.
EXAMPLE_CHIPS: tuple[tuple[str, str], ...] = (
    ("Leftovers in the fridge", "How long can I keep cooked leftovers in the fridge?"),
    ("Oils in the Eatwell Guide", "According to the Eatwell Guide, where do oils and spreads fit?"),
    ("A weight target", "What should I weigh?"),
)

DOCUMENT_NAMES: dict[str, str] = {
    document.document_id: document.document_name for document in CORPUS
}
ID_BY_LABEL: dict[str, str | None] = {ALL_DOCUMENTS_LABEL: None} | {
    document.document_name: document.document_id for document in CORPUS
}


def disclaimer() -> None:
    """Architecture §8: the boundary of the assistant, stated on the page."""
    st.info(DISCLAIMER, icon="ℹ️")


def document_picker() -> str | None:
    """"All documents" plus the seven registry names. "All documents" sends ``None``."""
    label = st.selectbox(
        "Search in",
        tuple(ID_BY_LABEL),
        index=0,
        key=PICKER_KEY,
        help="One document restricts the search, and a miss then names only that document.",
    )
    return ID_BY_LABEL[label]


def example_chips() -> str | None:
    """The message of the chip pressed on this run, else ``None``."""
    st.caption("Examples")
    columns = st.columns(len(EXAMPLE_CHIPS))
    pressed: str | None = None
    for index, (column, (label, message)) in enumerate(zip(columns, EXAMPLE_CHIPS)):
        if column.button(label, key=f"chip-{index}", use_container_width=True):
            pressed = message
    return pressed


def load_secrets_into_environment() -> None:
    """Copy ``SECRET_ENV_KEYS`` from ``st.secrets`` into ``os.environ`` when absent.

    A missing secrets file is not an error: the local run keeps reading
    ``.env`` through ``python-dotenv`` in the generator.
    """
    for key in SECRET_ENV_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # noqa: BLE001 - StreamlitSecretNotFoundError, or no file at all
            return
        if value is not None and str(value).strip():
            os.environ[key] = str(value).strip()


def in_process() -> bool:
    """``True`` when ``CHAT_IN_PROCESS`` is set to a truthy value."""
    return os.environ.get(IN_PROCESS_ENV, "").strip().lower() in _TRUE_VALUES


@st.cache_resource(show_spinner=False)
def in_process_pipeline() -> "AnswerPipeline":
    """The process-wide pipeline, built once. BGE loads on the first retrieve, not here."""
    from src.rag.pipeline import default_pipeline

    return default_pipeline()


_pipeline_warm = False


def spinner_text() -> str:
    """The model-loading sentence until BGE has been used once in this process."""
    if in_process() and not _pipeline_warm:
        return MODEL_LOADING_TEXT
    return SPINNER_TEXT


def ask(message: str, document_id: str | None) -> dict[str, object]:
    """One transcript turn: the question, its scope, and the response body or an error."""
    turn: dict[str, object] = {
        "question": message,
        "document_id": document_id,
        "response": None,
        "error": None,
    }
    if in_process():
        _ask_in_process(turn, message, document_id)
    else:
        try:
            turn["response"] = client.post_chat(message, document_id=document_id)
        except client.ChatError as exc:
            turn["error"] = str(exc)
    return turn


def _ask_in_process(turn: dict[str, object], message: str, document_id: str | None) -> None:
    """Call the pipeline directly. ``ValueError`` is the pipeline's 422."""
    global _pipeline_warm
    if not INDEX_DIR.is_dir():
        turn["error"] = MISSING_INDEX_ERROR
        return
    try:
        response = in_process_pipeline().answer(message, document_id=document_id)
    except ValueError as exc:
        turn["error"] = f"The question was rejected: {exc}"
        return
    body = response.model_dump()
    # The guard answers out_of_scope before retrieval, so BGE is still cold.
    if body.get("type") != "out_of_scope":
        _pipeline_warm = True
    turn["response"] = body


def render_turn(turn: dict[str, object]) -> None:
    """The question with its scope, then the answer or the refusal."""
    with st.chat_message("user"):
        st.markdown(str(turn["question"]))
        document_id = turn.get("document_id")
        if document_id is not None:
            st.caption(f"Searching only {DOCUMENT_NAMES.get(str(document_id), str(document_id))}")
    with st.chat_message("assistant"):
        error = turn.get("error")
        if error is not None:
            st.error(str(error))
        else:
            render_response(turn.get("response") or {})


def render_response(body: dict[str, object]) -> None:
    """Dispatch on the response ``type`` from architecture §8."""
    response_type = body.get("type")
    if response_type == "answer":
        render_answer(body)
    elif response_type == "not_in_corpus":
        render_not_in_corpus(body)
    elif response_type == "out_of_scope":
        render_out_of_scope(body)
    elif response_type == "generation_unavailable":
        st.warning(message_of(body, "The answer service is unavailable right now."))
    else:
        st.error(f"Unexpected response type from the chat API: {response_type!r}")


def render_answer(body: dict[str, object]) -> None:
    """One block per section, with the citation under each claim."""
    sections = body.get("sections")
    if not isinstance(sections, list) or not sections:
        st.warning("The chat API returned an answer with no sections.")
        return
    for index, section in enumerate(sections):
        if index:
            st.divider()
        st.markdown(
            f"**According to {section['document_name']} — "
            f"{section['publisher']}, {section['year']}**"
        )
        for claim in section.get("claims") or []:
            st.markdown(claim["text"])
            st.caption(citation_line(claim["citation"]))
    render_searched_expander(body)


def render_not_in_corpus(body: dict[str, object]) -> None:
    """A warning, then the documents that were searched and did not cover it."""
    message = message_of(body, "The guidance I searched does not cover that.")
    st.warning(message.splitlines()[0])
    searched = body.get("searched") or []
    if not searched:
        return
    st.markdown("I searched:")
    for entry in searched:
        st.markdown(f"- {searched_line(entry)}")


def render_out_of_scope(body: dict[str, object]) -> None:
    """A decline with the referral. No document list: the corpus was not searched."""
    st.error(
        message_of(
            body,
            "I can’t help with that. Please speak with a qualified health professional.",
        )
    )


def render_searched_expander(body: dict[str, object]) -> None:
    """The retrieval scope behind an answer, so the boundary stays visible."""
    searched = body.get("searched") or []
    if not searched:
        return
    with st.expander(f"Documents searched ({len(searched)})"):
        for entry in searched:
            st.markdown(f"- {searched_line(entry)}")


def citation_line(citation: dict[str, object]) -> str:
    """``— Name, Publisher, Year`` and the source link, from architecture §5.5."""
    url = citation["source_url"]
    return (
        f"— {citation['document_name']}, {citation['publisher']}, "
        f"{citation['year']} · [{url}]({url})"
    )


def searched_line(entry: dict[str, object]) -> str:
    return f"{entry['document_name']}, {entry['publisher']}, {entry['year']}"


def message_of(body: dict[str, object], fallback: str) -> str:
    """The API's message, or a fixed sentence when the body has none."""
    message = body.get("message")
    return message if isinstance(message, str) and message.strip() else fallback
