"""Chat page and its HTTP client (phase 8).

The page is driven through Streamlit's ``AppTest`` with
``src.ui.client.post_chat`` replaced by a recorder, so no test starts a
server, loads BGE, or calls Groq. The client tests monkeypatch ``httpx.post``
and never open a socket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.constants import CORPUS, PROJECT_ROOT
from src.rag import refusal
from src.rag.models import ChatResponse
from src.ui import client, view

APP_PATH = str(PROJECT_ROOT / "src" / "ui" / "app.py")
BY_ID = {document.document_id: document for document in CORPUS}
REGISTRY_IDS = tuple(document.document_id for document in CORPUS)

LEFTOVERS_CLAIM = "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator."
KITCHEN_CLAIM = "Refrigerate leftovers within two hours of cooking."


def section(document_id: str, ordinal: int, text: str) -> dict[str, object]:
    document = BY_ID[document_id]
    citation = {
        "document_name": document.document_name,
        "publisher": document.publisher,
        "year": document.year,
        "source_url": document.source_url,
    }
    return {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "publisher": document.publisher,
        "year": document.year,
        "source_url": document.source_url,
        "claims": [{"text": text, "chunk_ids": [f"{document_id}:{ordinal}"], "citation": citation}],
    }


def answer_body(*sections: dict[str, object]) -> dict[str, object]:
    return {
        "type": "answer",
        "reason": None,
        "message": "an answer",
        "sections": list(sections),
        "searched": refusal.searched_entries(REGISTRY_IDS),
    }


@dataclass
class Recorder:
    """Stands in for ``client.post_chat`` and records every call."""

    body: dict[str, object]
    error: Exception | None = None
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def __call__(
        self,
        message: str,
        document_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, object]:
        self.calls.append((message, document_id))
        if self.error is not None:
            raise self.error
        return self.body


@pytest.fixture(autouse=True)
def http_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts on the HTTP path, the way ``scripts/run_ui.py`` runs the page.

    With neither variable set the page answers in-process (the Community
    Cloud shape), which would load the real pipeline here. The ``in_process``
    fixture overrides this with ``CHAT_IN_PROCESS=1``.
    """
    monkeypatch.delenv(view.IN_PROCESS_ENV, raising=False)
    monkeypatch.setenv(client.API_BASE_URL_ENV, "http://api.test")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    stub = Recorder(body=answer_body(section("cold-food-storage", 51, LEFTOVERS_CLAIM)))
    monkeypatch.setattr(client, "post_chat", stub)
    return stub


@dataclass
class FakePipeline:
    """Stands in for ``AnswerPipeline`` on the in-process path and records every call."""

    body: dict[str, object]
    error: Exception | None = None
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def answer(self, message: str, document_id: str | None = None) -> ChatResponse:
        self.calls.append((message, document_id))
        if self.error is not None:
            raise self.error
        return ChatResponse.model_validate(self.body)


@pytest.fixture
def in_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakePipeline:
    """``CHAT_IN_PROCESS=1`` with a stub pipeline and an index directory that exists."""
    stub = FakePipeline(body=answer_body(section("cold-food-storage", 51, LEFTOVERS_CLAIM)))
    monkeypatch.setenv(view.IN_PROCESS_ENV, "1")
    monkeypatch.setattr(view, "in_process_pipeline", lambda: stub)
    monkeypatch.setattr(view, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(view, "_pipeline_warm", False)
    return stub


@pytest.fixture
def page() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30)


def new_page() -> AppTest:
    """A second page with its own session state, for a fresh transcript."""
    return AppTest.from_file(APP_PATH, default_timeout=30)


def texts(elements: Iterator[object]) -> list[str]:
    return [str(element.value) for element in elements]  # type: ignore[attr-defined]


def all_text(page: AppTest) -> str:
    return "\n".join(
        texts(page.markdown) + texts(page.caption) + texts(page.warning) + texts(page.error)
    )


def ask(page: AppTest, message: str) -> AppTest:
    page.run()
    page.chat_input[0].set_value(message).run()
    return page


# --- page furniture ------------------------------------------------------------


def test_the_page_shows_the_disclaimer_and_the_chat_input(page: AppTest) -> None:
    page.run()

    assert not page.exception
    disclaimer = "\n".join(texts(page.info))
    assert "seven official guidance documents" in disclaimer
    assert "not medical advice" in disclaimer
    assert len(page.chat_input) == 1


def test_the_picker_offers_all_documents_plus_the_seven_names(page: AppTest) -> None:
    page.run()

    options = list(page.selectbox[0].options)
    assert options[0] == view.ALL_DOCUMENTS_LABEL
    assert options[1:] == [document.document_name for document in CORPUS]
    assert page.selectbox[0].value == view.ALL_DOCUMENTS_LABEL


def test_the_three_example_chips_send_the_planned_messages(recorder: Recorder) -> None:
    labels = [label for label, _ in view.EXAMPLE_CHIPS]
    messages = [message for _, message in view.EXAMPLE_CHIPS]
    assert messages == [
        "How long can I keep cooked leftovers in the fridge?",
        "According to the Eatwell Guide, where do oils and spreads fit?",
        "What should I weigh?",
    ]

    for index, message in enumerate(messages):
        page = new_page()
        page.run()
        assert [button.label for button in page.button] == labels

        page.button[index].click().run()

        assert recorder.calls[-1] == (message, None)


# --- scope --------------------------------------------------------------------


def test_all_documents_sends_a_null_document_id(page: AppTest, recorder: Recorder) -> None:
    ask(page, "How long can I keep cooked leftovers in the fridge?")

    assert recorder.calls == [("How long can I keep cooked leftovers in the fridge?", None)]


def test_the_picker_restricted_to_one_document_sends_that_document_id(
    page: AppTest, recorder: Recorder
) -> None:
    recorder.body = answer_body(
        section("eatwell-guide", 2, "Fruit and vegetables should make up over a third of the diet.")
    )
    page.run()
    page.selectbox[0].set_value(BY_ID["eatwell-guide"].document_name).run()

    page.chat_input[0].set_value("How much of the diet should be fruit and vegetables?").run()

    assert recorder.calls == [
        ("How much of the diet should be fruit and vegetables?", "eatwell-guide")
    ]
    assert "Searching only The Eatwell Guide booklet" in all_text(page)


def test_a_blank_message_is_not_sent(page: AppTest, recorder: Recorder) -> None:
    ask(page, "   ")

    assert recorder.calls == []


# --- answers ------------------------------------------------------------------


def test_an_answer_with_two_sections_shows_two_citation_blocks(
    page: AppTest, recorder: Recorder
) -> None:
    recorder.body = answer_body(
        section("cold-food-storage", 51, LEFTOVERS_CLAIM),
        section("kitchen-companion", 37, KITCHEN_CLAIM),
    )

    ask(page, "How long can I keep cooked leftovers in the fridge?")

    markdown = texts(page.markdown)
    assert LEFTOVERS_CLAIM in markdown
    assert KITCHEN_CLAIM in markdown
    citations = [caption for caption in texts(page.caption) if caption.startswith("—")]
    assert len(citations) == 2
    for document_id in ("cold-food-storage", "kitchen-companion"):
        document = BY_ID[document_id]
        expected = f"— {document.document_name}, {document.publisher}, {document.year}"
        assert any(citation.startswith(expected) for citation in citations)
        assert any(document.source_url in citation for citation in citations)


def test_each_section_is_headed_by_its_own_document(page: AppTest, recorder: Recorder) -> None:
    recorder.body = answer_body(
        section("cold-food-storage", 51, LEFTOVERS_CLAIM),
        section("kitchen-companion", 37, KITCHEN_CLAIM),
    )

    ask(page, "How long do leftovers keep?")

    headings = [text for text in texts(page.markdown) if text.startswith("**According to ")]
    assert len(headings) == 2
    assert BY_ID["cold-food-storage"].document_name in headings[0]
    assert BY_ID["kitchen-companion"].document_name in headings[1]


def test_the_transcript_keeps_earlier_turns(page: AppTest, recorder: Recorder) -> None:
    ask(page, "How long do leftovers keep?")

    page.chat_input[0].set_value("How long do eggs keep?").run()

    assert len(recorder.calls) == 2
    markdown = texts(page.markdown)
    assert "How long do leftovers keep?" in markdown
    assert "How long do eggs keep?" in markdown


# --- refusals -----------------------------------------------------------------


def test_not_in_corpus_shows_a_warning_and_the_searched_list(
    page: AppTest, recorder: Recorder
) -> None:
    recorder.body = refusal.not_in_corpus(REGISTRY_IDS)

    ask(page, "What is the tariff on imported olive oil?")

    assert any("does not cover" in warning for warning in texts(page.warning))
    assert texts(page.error) == []
    markdown = texts(page.markdown)
    assert "I searched:" in markdown
    for document in CORPUS:
        assert f"- {document.document_name}, {document.publisher}, {document.year}" in markdown


def test_a_filtered_miss_lists_only_that_document(page: AppTest, recorder: Recorder) -> None:
    recorder.body = refusal.not_in_corpus(("who-healthy-diet",))
    page.run()
    page.selectbox[0].set_value(BY_ID["who-healthy-diet"].document_name).run()

    page.chat_input[0].set_value("How do I thaw a frozen turkey?").run()

    listed = [text for text in texts(page.markdown) if text.startswith("- ")]
    assert listed == ["- Healthy diet (Fact sheet), World Health Organization, 2018"]


def test_the_weight_target_chip_shows_out_of_scope_and_no_document_list(
    page: AppTest, recorder: Recorder
) -> None:
    recorder.body = refusal.out_of_scope("weight_target")
    chip_index = [message for _, message in view.EXAMPLE_CHIPS].index("What should I weigh?")
    page.run()

    page.button[chip_index].click().run()

    assert recorder.calls == [("What should I weigh?", None)]
    assert any("qualified health professional" in error for error in texts(page.error))
    assert texts(page.warning) == []
    rendered = all_text(page)
    for document in CORPUS:
        assert document.document_name not in rendered
    assert "I searched" not in rendered
    assert not any(text.startswith("- ") for text in texts(page.markdown))


def test_the_two_refusals_use_different_states(recorder: Recorder) -> None:
    recorder.body = refusal.out_of_scope("nutrient_lookup")
    out_of_scope = ask(new_page(), "How many grams of protein are in 100 g of chicken?")

    recorder.body = refusal.not_in_corpus(REGISTRY_IDS)
    not_in_corpus = ask(new_page(), "What is the tariff on imported olive oil?")

    assert (len(out_of_scope.error), len(out_of_scope.warning)) == (1, 0)
    assert (len(not_in_corpus.error), len(not_in_corpus.warning)) == (0, 1)


def test_generation_unavailable_is_a_warning_that_does_not_claim_a_miss(
    page: AppTest, recorder: Recorder
) -> None:
    recorder.body = refusal.generation_unavailable(REGISTRY_IDS, reason="quota_exceeded")

    ask(page, "How long do leftovers keep?")

    warnings = texts(page.warning)
    assert len(warnings) == 1
    assert "does not cover" not in warnings[0]


def test_an_unreachable_api_is_shown_as_an_error(page: AppTest, recorder: Recorder) -> None:
    recorder.error = client.ChatError("Could not reach the chat API at http://127.0.0.1:8000.")

    ask(page, "How long do leftovers keep?")

    assert not page.exception
    assert any("Could not reach the chat API" in error for error in texts(page.error))


# --- in-process (deployment plan, phase D3) ------------------------------------


def test_a_configured_api_url_selects_the_http_path(
    monkeypatch: pytest.MonkeyPatch, page: AppTest, recorder: Recorder
) -> None:
    monkeypatch.delenv(view.IN_PROCESS_ENV, raising=False)
    monkeypatch.setenv(client.API_BASE_URL_ENV, "http://127.0.0.1:8000")
    stub = FakePipeline(body=answer_body())
    monkeypatch.setattr(view, "in_process_pipeline", lambda: stub)

    ask(page, "How long do leftovers keep?")

    assert recorder.calls == [("How long do leftovers keep?", None)]
    assert stub.calls == []


def test_neither_variable_set_is_in_process_the_community_cloud_shape(
    monkeypatch: pytest.MonkeyPatch, page: AppTest, recorder: Recorder, tmp_path: Path
) -> None:
    """A bare ``streamlit run`` with no mode secret must not post to 127.0.0.1:8000."""
    monkeypatch.delenv(view.IN_PROCESS_ENV, raising=False)
    monkeypatch.delenv(client.API_BASE_URL_ENV, raising=False)
    stub = FakePipeline(body=answer_body(section("cold-food-storage", 51, LEFTOVERS_CLAIM)))
    monkeypatch.setattr(view, "in_process_pipeline", lambda: stub)
    monkeypatch.setattr(view, "INDEX_DIR", tmp_path)

    ask(page, "How long can I keep cooked leftovers in the fridge?")

    assert not page.exception
    assert stub.calls == [("How long can I keep cooked leftovers in the fridge?", None)]
    assert recorder.calls == []
    assert not any("Could not reach the chat API" in error for error in texts(page.error))
    assert LEFTOVERS_CLAIM in texts(page.markdown)


@pytest.mark.parametrize(
    ("flag", "base_url", "expected"),
    [
        (None, None, True),  # bare `streamlit run`: Community Cloud
        (None, "http://127.0.0.1:8000", False),  # scripts/run_ui.py
        ("1", "http://127.0.0.1:8000", True),  # the flag wins over the URL
        ("0", None, False),  # the flag wins over the missing URL
        ("", None, True),  # an empty flag is unset
        ("", "http://127.0.0.1:8000", False),
    ],
)
def test_in_process_is_decided_by_the_flag_then_the_api_url(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, base_url: str | None, expected: bool
) -> None:
    for name, value in ((view.IN_PROCESS_ENV, flag), (client.API_BASE_URL_ENV, base_url)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert view.in_process() is expected


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_truthy_values_turn_the_in_process_path_on(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(view.IN_PROCESS_ENV, value)
    assert view.in_process()


@pytest.mark.parametrize("value", ["0", "false", "no"])
def test_falsy_values_force_the_http_path(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(view.IN_PROCESS_ENV, value)
    monkeypatch.delenv(client.API_BASE_URL_ENV, raising=False)
    assert not view.in_process()


def test_in_process_calls_the_pipeline_and_never_the_api(
    page: AppTest, recorder: Recorder, in_process: FakePipeline
) -> None:
    in_process.body = answer_body(
        section("cold-food-storage", 51, LEFTOVERS_CLAIM),
        section("kitchen-companion", 37, KITCHEN_CLAIM),
    )

    ask(page, "How long can I keep cooked leftovers in the fridge?")

    assert not page.exception
    assert in_process.calls == [("How long can I keep cooked leftovers in the fridge?", None)]
    assert recorder.calls == []
    markdown = texts(page.markdown)
    assert LEFTOVERS_CLAIM in markdown
    assert KITCHEN_CLAIM in markdown
    assert len([caption for caption in texts(page.caption) if caption.startswith("—")]) == 2


def test_in_process_sends_the_picked_document_id(
    page: AppTest, recorder: Recorder, in_process: FakePipeline
) -> None:
    in_process.body = refusal.not_in_corpus(("eatwell-guide",))
    page.run()
    page.selectbox[0].set_value(BY_ID["eatwell-guide"].document_name).run()

    page.chat_input[0].set_value("How do I thaw a frozen turkey?").run()

    assert in_process.calls == [("How do I thaw a frozen turkey?", "eatwell-guide")]
    assert recorder.calls == []
    eatwell = BY_ID["eatwell-guide"]
    listed = [text for text in texts(page.markdown) if text.startswith("- ")]
    assert listed == [f"- {eatwell.document_name}, {eatwell.publisher}, {eatwell.year}"]


def test_in_process_out_of_scope_renders_the_decline(
    page: AppTest, recorder: Recorder, in_process: FakePipeline
) -> None:
    in_process.body = refusal.out_of_scope("weight_target")

    ask(page, "What should I weigh?")

    assert any("qualified health professional" in error for error in texts(page.error))
    assert texts(page.warning) == []
    assert recorder.calls == []


def test_in_process_value_error_is_shown_not_raised(
    page: AppTest, in_process: FakePipeline
) -> None:
    in_process.error = ValueError("document_id is not in the registry: nope")

    ask(page, "How long do leftovers keep?")

    assert not page.exception
    assert any("not in the registry" in error for error in texts(page.error))


def test_in_process_reports_a_missing_index(
    monkeypatch: pytest.MonkeyPatch, page: AppTest, in_process: FakePipeline, tmp_path: Path
) -> None:
    monkeypatch.setattr(view, "INDEX_DIR", tmp_path / "absent")

    ask(page, "How long do leftovers keep?")

    assert not page.exception
    assert in_process.calls == []
    assert any("index is missing" in error for error in texts(page.error))


def test_the_spinner_says_the_model_is_loading_until_retrieval_has_run(
    in_process: FakePipeline,
) -> None:
    assert view.spinner_text() == view.MODEL_LOADING_TEXT

    in_process.body = refusal.out_of_scope("weight_target")
    view.ask("What should I weigh?", None)
    assert view.spinner_text() == view.MODEL_LOADING_TEXT  # the guard did not load BGE

    in_process.body = answer_body(section("cold-food-storage", 51, LEFTOVERS_CLAIM))
    view.ask("How long do leftovers keep?", None)
    assert view.spinner_text() == view.SPINNER_TEXT


def test_the_spinner_is_the_search_text_on_the_http_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(view.IN_PROCESS_ENV, raising=False)
    monkeypatch.setattr(view, "_pipeline_warm", False)

    assert view.spinner_text() == view.SPINNER_TEXT


# --- secrets (deployment plan, phase D4) ---------------------------------------


class NoSecretsFile:
    """``st.secrets`` when neither secrets.toml location exists."""

    def get(self, key: str, default: object = None) -> object:
        raise RuntimeError("No secrets found")


def test_secrets_are_copied_into_the_environment_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in view.SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(view.st, "secrets", {"GROQ_API_KEY": "gsk-test", "CHAT_IN_PROCESS": "1"})

    view.load_secrets_into_environment()

    assert view.in_process()
    assert view.os.environ["GROQ_API_KEY"] == "gsk-test"
    assert "GROQ_MODEL" not in view.os.environ


def test_an_existing_environment_value_wins_over_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    monkeypatch.setattr(view.st, "secrets", {"GROQ_API_KEY": "from-secrets"})

    view.load_secrets_into_environment()

    assert view.os.environ["GROQ_API_KEY"] == "from-env"


def test_a_missing_secrets_file_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in view.SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(view.st, "secrets", NoSecretsFile())

    view.load_secrets_into_environment()

    assert all(key not in view.os.environ for key in view.SECRET_ENV_KEYS)


# --- client -------------------------------------------------------------------


@dataclass
class FakePost:
    """Replaces ``httpx.post``: records the call and returns a canned response."""

    response: httpx.Response | None = None
    error: Exception | None = None
    url: str | None = None
    payload: object = None

    def __call__(self, url: str, **kwargs: object) -> httpx.Response:
        self.url = url
        self.payload = kwargs.get("json")
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def fake_post(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> FakePost:
    post = FakePost(**kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(httpx, "post", post)
    return post


def test_post_chat_sends_the_message_and_scope_to_the_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = fake_post(
        monkeypatch,
        response=httpx.Response(200, json={"type": "answer", "sections": [], "searched": []}),
    )

    body = client.post_chat(
        "How long do leftovers keep?",
        document_id="eatwell-guide",
        base_url="http://api:9000/",
    )

    assert post.url == "http://api:9000/chat"
    assert post.payload == {
        "message": "How long do leftovers keep?",
        "document_id": "eatwell-guide",
    }
    assert body["type"] == "answer"


def test_post_chat_reads_the_base_url_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(client.API_BASE_URL_ENV, "http://elsewhere:1234")
    post = fake_post(
        monkeypatch,
        response=httpx.Response(200, json={"type": "not_in_corpus", "sections": [], "searched": []}),
    )

    client.post_chat("How long do leftovers keep?")

    assert post.url == "http://elsewhere:1234/chat"


def test_post_chat_defaults_to_local_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(client.API_BASE_URL_ENV, raising=False)

    assert client.api_base_url() == client.DEFAULT_API_BASE_URL


def test_a_null_document_id_is_sent_for_the_whole_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    post = fake_post(
        monkeypatch,
        response=httpx.Response(200, json={"type": "answer", "sections": [], "searched": []}),
    )

    client.post_chat("How long do leftovers keep?")

    assert post.payload == {"message": "How long do leftovers keep?", "document_id": None}


def test_a_transport_failure_becomes_a_chat_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(monkeypatch, error=httpx.ConnectError("no route"))

    with pytest.raises(client.ChatError, match="Could not reach the chat API"):
        client.post_chat("How long do leftovers keep?")


def test_a_422_detail_is_carried_into_the_chat_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(
        monkeypatch,
        response=httpx.Response(422, json={"detail": "document_id is not in the registry: nope"}),
    )

    with pytest.raises(client.ChatError, match="not in the registry"):
        client.post_chat("How long do leftovers keep?", document_id="nope")


def test_a_pydantic_error_list_is_summarised(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(
        monkeypatch,
        response=httpx.Response(
            422, json={"detail": [{"msg": "Value error, message must not be empty"}]}
        ),
    )

    with pytest.raises(client.ChatError, match="must not be empty"):
        client.post_chat(" ")


def test_a_body_without_a_type_is_a_chat_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(monkeypatch, response=httpx.Response(200, json={"sections": []}))

    with pytest.raises(client.ChatError, match="without a response type"):
        client.post_chat("How long do leftovers keep?")


def test_a_non_json_body_is_a_chat_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(monkeypatch, response=httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(client.ChatError, match="not JSON"):
        client.post_chat("How long do leftovers keep?")
