"""Streamlit chat page (implementation plan, phase 8; architecture §8).

One page: a disclaimer, a document picker whose options are "All documents"
plus the seven registry names, three example chips, and the chat transcript.
The widgets and renderers live in ``src/ui/view.py``; the HTTP call lives in
``src/ui/client.py``.

Start the API first, then the page:

    uvicorn src.api.main:app
    python scripts/run_ui.py

On Streamlit Community Cloud there is no second process. Set
``CHAT_IN_PROCESS=1`` (a secret) and the page calls the pipeline directly;
see docs/deployment-plan.md. This file is the deploy entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script, so the repo root is not on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.ui import view  # noqa: E402


def main() -> None:
    """Draw the page. Streamlit calls this on every interaction."""
    st.set_page_config(page_title=view.PAGE_TITLE, page_icon="🥗", layout="centered")
    # Before anything reads GROQ_API_KEY or CHAT_IN_PROCESS.
    view.load_secrets_into_environment()
    st.title(view.PAGE_TITLE)
    view.disclaimer()

    document_id = view.document_picker()
    chip_message = view.example_chips()
    typed_message = st.chat_input(view.CHAT_PLACEHOLDER)

    turns: list[dict[str, object]] = st.session_state.setdefault(view.TURNS_KEY, [])
    for turn in turns:
        view.render_turn(turn)

    # A chip press and a typed message cannot arrive on the same run; the chip
    # wins if they ever do.
    message = (chip_message or typed_message or "").strip()
    if not message:
        return
    with st.spinner(view.spinner_text()):
        turn = view.ask(message, document_id)
    turns.append(turn)
    view.render_turn(turn)


main()
