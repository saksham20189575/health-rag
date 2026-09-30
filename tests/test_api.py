"""``POST /chat`` and ``GET /health`` with a stubbed pipeline (phase 7).

The stub records every call and returns a canned ``ChatResponse``, so no test
here loads BGE, reads ``data/index``, or calls Groq. Each response type is
round-tripped through the client to prove the API returns the pipeline body
unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_pipeline
from src.constants import CORPUS
from src.rag import refusal
from src.rag.models import ChatResponse, Citation, Claim, Section

BY_ID = {document.document_id: document for document in CORPUS}
REGISTRY_IDS = tuple(document.document_id for document in CORPUS)

LEFTOVERS_CLAIM = "Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator."
KITCHEN_CLAIM = "Refrigerate leftovers within two hours of cooking."


def section(document_id: str, ordinal: int, text: str) -> Section:
    document = BY_ID[document_id]
    citation = Citation(
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
    )
    return Section(
        document_id=document.document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        claims=[Claim(text=text, chunk_ids=[f"{document_id}:{ordinal}"], citation=citation)],
    )


def answer_response(*sections: Section) -> ChatResponse:
    return ChatResponse(
        type="answer",
        message="\n".join(claim.text for item in sections for claim in item.claims),
        sections=list(sections),
        searched=refusal.searched_entries(REGISTRY_IDS),
    )


@dataclass
class StubPipeline:
    """Mimics ``AnswerPipeline.answer`` and records every call."""

    response: ChatResponse | None = None
    error: Exception | None = None
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def answer(self, message: str, document_id: str | None = None) -> ChatResponse:
        self.calls.append((message, document_id))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

@pytest.fixture
def stub() -> StubPipeline:
    return StubPipeline(response=answer_response(section("cold-food-storage", 51, LEFTOVERS_CLAIM)))


@pytest.fixture
def client(stub: StubPipeline) -> TestClient:
    app.dependency_overrides[get_pipeline] = lambda: stub
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --- answers -------------------------------------------------------------------


def test_answer_round_trips_with_its_type_and_citations(client: TestClient, stub: StubPipeline) -> None:
    response = client.post("/chat", json={"message": "How long can I keep cooked leftovers in the fridge?"})

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "answer"
    assert body["reason"] is None
    assert [item["document_id"] for item in body["sections"]] == ["cold-food-storage"]
    claim = body["sections"][0]["claims"][0]
    assert claim["text"] == LEFTOVERS_CLAIM
    assert claim["chunk_ids"] == ["cold-food-storage:51"]
    assert claim["citation"]["source_url"] == BY_ID["cold-food-storage"].source_url
    assert len(body["searched"]) == 7
    assert stub.calls == [("How long can I keep cooked leftovers in the fridge?", None)]


def test_response_body_carries_the_architecture_keys(client: TestClient) -> None:
    body = client.post("/chat", json={"message": "How long do leftovers keep?"}).json()

    assert set(body) == {"type", "reason", "message", "sections", "searched"}
    assert set(body["sections"][0]) == {
        "document_id",
        "document_name",
        "publisher",
        "year",
        "source_url",
        "claims",
    }
    assert set(body["searched"][0]) == {"document_id", "document_name", "publisher", "year"}


def test_two_sections_survive_the_round_trip_in_order(client: TestClient, stub: StubPipeline) -> None:
    stub.response = answer_response(
        section("cold-food-storage", 51, LEFTOVERS_CLAIM),
        section("kitchen-companion", 37, KITCHEN_CLAIM),
    )

    body = client.post("/chat", json={"message": "How long do leftovers keep?"}).json()

    assert [item["document_id"] for item in body["sections"]] == ["cold-food-storage", "kitchen-companion"]
    for item in body["sections"]:
        for claim in item["claims"]:
            assert claim["citation"]["source_url"] == BY_ID[item["document_id"]].source_url


def test_document_id_in_the_registry_is_passed_to_the_pipeline(client: TestClient, stub: StubPipeline) -> None:
    stub.response = answer_response(section("eatwell-guide", 2, "Fruit and vegetables should make up over a third of the diet."))

    response = client.post(
        "/chat",
        json={"message": "How much of the diet should be fruit and vegetables?", "document_id": "eatwell-guide"},
    )

    assert response.status_code == 200
    assert stub.calls[0][1] == "eatwell-guide"


def test_message_whitespace_is_collapsed_before_the_pipeline(client: TestClient, stub: StubPipeline) -> None:
    client.post("/chat", json={"message": "  How   long\ndo leftovers   keep?  "})

    assert stub.calls == [("How long do leftovers keep?", None)]


# --- refusals ------------------------------------------------------------------


def test_not_in_corpus_round_trips_with_the_searched_list(client: TestClient, stub: StubPipeline) -> None:
    stub.response = ChatResponse.from_refusal(refusal.not_in_corpus(REGISTRY_IDS))

    body = client.post("/chat", json={"message": "What is the tariff on imported olive oil?"}).json()

    assert body["type"] == "not_in_corpus"
    assert body["reason"] == "no_supporting_chunks"
    assert body["sections"] == []
    assert [entry["document_id"] for entry in body["searched"]] == list(REGISTRY_IDS)
    for document in CORPUS:
        assert f"- {document.document_name}, {document.publisher}, {document.year}" in body["message"]


def test_filtered_not_in_corpus_lists_only_that_document(client: TestClient, stub: StubPipeline) -> None:
    stub.response = ChatResponse.from_refusal(refusal.not_in_corpus(("who-healthy-diet",)))

    body = client.post(
        "/chat",
        json={"message": "How do I thaw a frozen turkey?", "document_id": "who-healthy-diet"},
    ).json()

    assert body["type"] == "not_in_corpus"
    assert [entry["document_id"] for entry in body["searched"]] == ["who-healthy-diet"]
    assert "Kitchen Companion" not in body["message"]


def test_out_of_scope_round_trips_with_an_empty_searched_list(client: TestClient, stub: StubPipeline) -> None:
    stub.response = ChatResponse.from_refusal(refusal.out_of_scope("weight_target"))

    body = client.post("/chat", json={"message": "What should I weigh?"}).json()

    assert body["type"] == "out_of_scope"
    assert body["reason"] == "weight_target"
    assert body["sections"] == []
    assert body["searched"] == []
    assert "qualified health professional" in body["message"]

def test_generation_unavailable_round_trips(client: TestClient, stub: StubPipeline) -> None:
    stub.response = ChatResponse.from_refusal(
        refusal.generation_unavailable(REGISTRY_IDS, reason="quota_exceeded")
    )

    body = client.post("/chat", json={"message": "How long do leftovers keep?"}).json()

    assert body["type"] == "generation_unavailable"
    assert body["reason"] == "quota_exceeded"
    assert "does not cover" not in body["message"]


# --- request validation --------------------------------------------------------


def test_unknown_document_id_is_422_and_does_not_call_the_pipeline(client: TestClient, stub: StubPipeline) -> None:
    response = client.post(
        "/chat",
        json={"message": "How long do leftovers keep?", "document_id": "not-a-document"},
    )

    assert response.status_code == 422
    assert "not in the registry" in response.text
    assert stub.calls == []


@pytest.mark.parametrize("message", ["", "   ", "\n\t ", "\u00a0"])
def test_empty_message_is_422_and_does_not_call_the_pipeline(
    client: TestClient, stub: StubPipeline, message: str
) -> None:
    response = client.post("/chat", json={"message": message})

    assert response.status_code == 422
    assert stub.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"document_id": "eatwell-guide"},
        {"message": None},
        {"message": 42},
        {"message": ["How long do leftovers keep?"]},
        {"message": "How long do leftovers keep?", "document_id": 7},
    ],
)
def test_a_malformed_body_is_422(client: TestClient, stub: StubPipeline, body: dict[str, object]) -> None:
    response = client.post("/chat", json=body)
    assert response.status_code == 422
    assert stub.calls == []


def test_an_over_long_message_is_422(client: TestClient, stub: StubPipeline) -> None:
    response = client.post("/chat", json={"message": "leftovers " * 500})

    assert response.status_code == 422
    assert stub.calls == []


def test_a_pipeline_value_error_is_422_not_500(client: TestClient, stub: StubPipeline) -> None:
    stub.error = ValueError("document_id is not in the registry: eatwell-guide")

    response = client.post("/chat", json={"message": "How long do leftovers keep?"})

    assert response.status_code == 422
    assert "not in the registry" in response.json()["detail"]

# --- health --------------------------------------------------------------------


def test_health_is_ok_when_the_chroma_directory_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.api.main.INDEX_DIR", tmp_path)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["index_present"] is True


def test_health_is_unavailable_without_the_chroma_directory(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.api.main.INDEX_DIR", tmp_path / "missing")

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["index_present"] is False
