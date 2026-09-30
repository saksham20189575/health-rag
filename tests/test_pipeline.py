"""Answer pipeline with a fake retriever and a fake Groq client (phase 6).

No test here touches the network, the BGE model, or ``data/index``. The fake
Groq client returns a canned completion and records whether it was called;
the fake retriever returns a preset ``RetrievalResult`` and records its calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest

from src.constants import CORPUS
from src.rag.generator import GroqClaimGenerator
from src.rag.models import ChatResponse
from src.rag.pipeline import ALL_CLAIMS_DROPPED_REASON, AnswerPipeline
from src.rag.retriever import REGISTRY_IDS, DocumentHits, RetrievalResult, RetrievedChunk

BY_ID = {document.document_id: document for document in CORPUS}

LEFTOVERS_ROW = (
    "Food: Leftovers | Type: Cooked meat or poultry | "
    "Refrigerator [40°F (4°C) or below]: 3 to 4 days | Freezer [0°F (-18°C) or below]: 2 to 6 months"
)
KITCHEN_ROW = "Leftovers: cooked meat, meat casseroles — Refrigerator 3 to 4 days — Freezer 2 to 3 months"
OILS_TEXT = (
    "Choose unsaturated oils and spreads, and eat them in small amounts. "
    "Unsaturated fats are healthier fats, such as vegetable, rapeseed, olive and sunflower oils."
)
WHO_FAT_TEXT = (
    "Total fat should not exceed 30% of total energy intake. Intake of saturated fats "
    "should be less than 10% of total energy intake."
)


def chunk(document_id: str, ordinal: int, text: str, similarity: float = 0.8) -> RetrievedChunk:
    document = BY_ID[document_id]
    return RetrievedChunk(
        chunk_id=f"{document_id}:{ordinal}",
        text=text,
        similarity=similarity,
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        section_heading="Heading",
        block_type="table_row",
        retrieval_date="2026-09-27",
    )


def result(*groups: DocumentHits, searched: tuple[str, ...] | None = None) -> RetrievalResult:
    scope = searched if searched is not None else REGISTRY_IDS
    candidates = tuple(hit for group in groups for hit in group.hits)
    return RetrievalResult(
        searched=scope,
        groups=tuple(groups),
        candidate_count=len(candidates),
        candidates=candidates,
    )


@dataclass
class FakeRetriever:
    preset: RetrievalResult
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def retrieve(self, query: str, document_id: str | None = None) -> RetrievalResult:
        self.calls.append((query, document_id))
        return self.preset


class FakeCompletions:
    def __init__(self, owner: "FakeGroqClient") -> None:
        self._owner = owner

    def create(self, **kwargs: object) -> SimpleNamespace:
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        message = SimpleNamespace(content=self._owner.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroqClient:
    """Mimics ``groq.Groq().chat.completions.create`` and records every call."""

    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


def payload(*documents: tuple[str, list[tuple[str, list[str]]]]) -> str:
    return json.dumps(
        {
            "documents": [
                {
                    "document_id": document_id,
                    "claims": [{"text": text, "chunk_ids": chunk_ids} for text, chunk_ids in claims],
                }
                for document_id, claims in documents
            ]
        }
    )


def pipeline(
    retrieval: RetrievalResult,
    groq: FakeGroqClient,
) -> tuple[AnswerPipeline, FakeRetriever, FakeGroqClient]:
    retriever = FakeRetriever(retrieval)
    generator = GroqClaimGenerator(client=groq, model="fake-model")
    return AnswerPipeline(retriever=retriever, generator=generator), retriever, groq


def two_document_retrieval() -> RetrievalResult:
    return result(
        DocumentHits("cold-food-storage", (chunk("cold-food-storage", 51, LEFTOVERS_ROW),)),
        DocumentHits(
            "kitchen-companion",
            (chunk("kitchen-companion", 37, KITCHEN_ROW), chunk("kitchen-companion", 140, KITCHEN_ROW)),
        ),
    )


# --- answers -------------------------------------------------------------------


def test_two_supporting_documents_yield_two_sections_in_registry_order() -> None:
    groq = FakeGroqClient(
        payload(
            # Groq lists kitchen-companion first; the renderer must reorder.
            ("kitchen-companion", [("Cooked meat leftovers keep 3 to 4 days in the refrigerator.", ["kitchen-companion:37"])]),
            ("cold-food-storage", [("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"])]),
        )
    )
    answer_pipeline, retriever, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("How long can I keep cooked leftovers in the fridge?")

    assert isinstance(response, ChatResponse)
    assert response.type == "answer"
    assert response.reason is None
    assert [section.document_id for section in response.sections] == ["cold-food-storage", "kitchen-companion"]
    for section in response.sections:
        for claim in section.claims:
            assert all(chunk_id.startswith(section.document_id + ":") for chunk_id in claim.chunk_ids)
    assert len(response.searched) == 7
    assert retriever.calls == [("How long can I keep cooked leftovers in the fridge?", None)]
    assert len(groq.calls) == 1


def test_no_claim_cites_chunk_ids_from_two_documents() -> None:
    groq = FakeGroqClient(
        payload(
            (
                "cold-food-storage",
                [
                    ("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"]),
                    # Borrowed a kitchen-companion id: must be dropped.
                    ("Leftovers keep 3 to 4 days.", ["cold-food-storage:51", "kitchen-companion:37"]),
                ],
            ),
        )
    )
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "answer"
    assert [section.document_id for section in response.sections] == ["cold-food-storage"]
    claims = response.sections[0].claims
    assert len(claims) == 1
    assert claims[0].chunk_ids == ["cold-food-storage:51"]


def test_claim_whose_chunk_id_was_not_retrieved_is_dropped() -> None:
    groq = FakeGroqClient(
        payload(
            (
                "cold-food-storage",
                [
                    ("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"]),
                    ("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:999"]),
                ],
            ),
        )
    )
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "answer"
    kept = [claim.chunk_ids for section in response.sections for claim in section.claims]
    assert kept == [["cold-food-storage:51"]]


def test_rendered_citation_matches_the_chunk_record() -> None:
    groq = FakeGroqClient(
        payload(("cold-food-storage", [("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"])]))
    )
    retrieval = two_document_retrieval()
    answer_pipeline, _, _ = pipeline(retrieval, groq)

    response = answer_pipeline.answer("How long do leftovers keep?")

    section = response.sections[0]
    hit = retrieval.groups[0].hits[0]
    assert section.source_url == hit.source_url
    assert section.document_name == hit.document_name
    assert section.publisher == hit.publisher
    assert section.year == hit.year
    claim = section.claims[0]
    assert claim.citation.source_url == hit.source_url
    assert claim.citation.document_name == hit.document_name
    assert claim.citation.publisher == hit.publisher
    assert claim.citation.year == hit.year
    assert claim.citation.source_url == BY_ID["cold-food-storage"].source_url


def test_answer_message_does_not_add_the_guidelines_say() -> None:
    groq = FakeGroqClient(
        payload(
            ("eatwell-guide", [("Choose unsaturated oils and spreads and eat them in small amounts.", ["eatwell-guide:34"])]),
            ("who-healthy-diet", [("Total fat should not exceed 30% of total energy intake.", ["who-healthy-diet:8"])]),
        )
    )
    retrieval = result(
        DocumentHits("who-healthy-diet", (chunk("who-healthy-diet", 8, WHO_FAT_TEXT),)),
        DocumentHits("eatwell-guide", (chunk("eatwell-guide", 34, OILS_TEXT),)),
    )
    answer_pipeline, _, _ = pipeline(retrieval, groq)

    response = answer_pipeline.answer("What do the guidelines say about cooking oil?")

    assert response.type == "answer"
    assert [section.document_id for section in response.sections] == ["who-healthy-diet", "eatwell-guide"]
    assert response.message is not None
    assert "the guidelines say" not in response.message.lower()
    assert response.message.count("According to ") == 2
    assert BY_ID["eatwell-guide"].source_url in response.message
    assert BY_ID["who-healthy-diet"].source_url in response.message


def test_prompt_contains_only_chunk_text_and_ids() -> None:
    groq = FakeGroqClient(
        payload(("cold-food-storage", [("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"])]))
    )
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), groq)

    answer_pipeline.answer("How long do leftovers keep?")

    call = groq.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "fake-model"
    user = [message for message in call["messages"] if message["role"] == "user"][0]["content"]
    assert "[cold-food-storage:51]" in user
    assert LEFTOVERS_ROW in user
    assert "[kitchen-companion:37]" in user and "[kitchen-companion:140]" in user
    for document in ("cold-food-storage", "kitchen-companion"):
        assert BY_ID[document].publisher not in user
        assert BY_ID[document].source_url not in user
        assert BY_ID[document].document_name not in user
    assert "http" not in user


# --- not_in_corpus -------------------------------------------------------------


def test_in_scope_miss_returns_not_in_corpus_and_does_not_call_groq() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(result(), groq)

    response = answer_pipeline.answer("What is the tariff on imported olive oil?")

    assert response.type == "not_in_corpus"
    assert response.reason == "no_supporting_chunks"
    assert response.sections == []
    assert [entry.document_id for entry in response.searched] == list(REGISTRY_IDS)
    assert response.message is not None
    for document in CORPUS:
        assert f"- {document.document_name}, {document.publisher}, {document.year}" in response.message
    assert groq.calls == []
    assert len(retriever.calls) == 1


def test_filtered_miss_lists_only_that_document() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(result(searched=("who-healthy-diet",)), groq)

    response = answer_pipeline.answer("How do I thaw a frozen turkey?", document_id="who-healthy-diet")

    assert response.type == "not_in_corpus"
    assert [entry.document_id for entry in response.searched] == ["who-healthy-diet"]
    assert response.message is not None
    assert "Healthy diet (Fact sheet), World Health Organization, 2018" in response.message
    assert "Kitchen Companion" not in response.message
    assert retriever.calls == [("How do I thaw a frozen turkey?", "who-healthy-diet")]
    assert groq.calls == []


def test_all_claims_dropped_yields_not_in_corpus_naming_the_searched_documents() -> None:
    groq = FakeGroqClient(
        payload(
            ("cold-food-storage", [("Leftovers keep 48 hours.", ["cold-food-storage:51"])]),
            ("kitchen-companion", [("Leftovers keep 3 to 4 days.", ["kitchen-companion:1"])]),
        )
    )
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "not_in_corpus"
    assert response.reason == ALL_CLAIMS_DROPPED_REASON
    assert response.sections == []
    assert len(response.searched) == 7
    assert response.message is not None and "I searched:" in response.message
    assert len(groq.calls) == 1


def test_generated_document_outside_the_retrieval_scope_is_ignored() -> None:
    groq = FakeGroqClient(
        payload(("eatwell-guide", [("Choose unsaturated oils.", ["eatwell-guide:34"])]))
    )
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "not_in_corpus"
    assert response.reason == ALL_CLAIMS_DROPPED_REASON


# --- out_of_scope --------------------------------------------------------------


def test_weight_target_question_does_not_call_the_retriever_or_groq() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("What should I weigh?")

    assert response.type == "out_of_scope"
    assert response.reason == "weight_target"
    assert response.searched == []
    assert response.sections == []
    assert response.message is not None and "qualified health professional" in response.message
    assert retriever.calls == []
    assert groq.calls == []


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("How many calories should I eat a day?", "calorie_target"),
        ("I have diabetes, what should I eat?", "medical"),
        ("How many grams of protein are in 100 g of chicken?", "nutrient_lookup"),
    ],
)
def test_other_guard_reasons_have_an_empty_searched_list(question: str, reason: str) -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer(question)

    assert response.type == "out_of_scope"
    assert response.reason == reason
    assert response.searched == []
    assert retriever.calls == []


def test_guard_runs_before_an_explicit_document_id_is_used() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(two_document_retrieval(), groq)

    response = answer_pipeline.answer("What should I weigh?", document_id="eatwell-guide")

    assert response.type == "out_of_scope"
    assert response.searched == []
    assert retriever.calls == []


# --- scope resolution ----------------------------------------------------------


def test_alias_in_the_message_filters_retrieval() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(result(searched=("eatwell-guide",)), groq)

    answer_pipeline.answer("According to the Eatwell Guide, how much of the diet should be fruit and vegetables?")

    assert retriever.calls[0][1] == "eatwell-guide"


def test_explicit_document_id_overrides_aliases() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(result(searched=("fsa-chill",)), groq)

    answer_pipeline.answer("What does the Eatwell Guide say about oils?", document_id="fsa-chill")

    assert retriever.calls[0][1] == "fsa-chill"


def test_unknown_document_id_raises_before_the_retriever() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(two_document_retrieval(), groq)

    with pytest.raises(ValueError, match="not in the registry"):
        answer_pipeline.answer("How long do leftovers keep?", document_id="not-a-document")
    assert retriever.calls == []


@pytest.mark.parametrize("message", ["", "   ", "\n\t "])
def test_empty_message_is_rejected(message: str) -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(two_document_retrieval(), groq)

    with pytest.raises(ValueError, match="empty"):
        answer_pipeline.answer(message)
    assert retriever.calls == []


def test_whitespace_is_collapsed_before_retrieval() -> None:
    groq = FakeGroqClient(payload())
    answer_pipeline, retriever, _ = pipeline(result(), groq)

    answer_pipeline.answer("  How   long\ndo leftovers   keep?  ")

    assert retriever.calls == [("How long do leftovers keep?", None)]


# --- generation_unavailable ----------------------------------------------------


def _status_error(cls: type, status: int) -> Exception:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def test_rate_limit_is_generation_unavailable_quota_exceeded() -> None:
    import groq as groq_sdk

    client = FakeGroqClient(error=_status_error(groq_sdk.RateLimitError, 429))
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), client)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "generation_unavailable"
    assert response.reason == "quota_exceeded"
    assert response.sections == []
    assert len(response.searched) == 7
    assert response.message is not None
    assert "does not cover" not in response.message


def test_transport_error_is_generation_unavailable_groq_error() -> None:
    import groq as groq_sdk

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client = FakeGroqClient(error=groq_sdk.APIConnectionError(request=request))
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), client)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "generation_unavailable"
    assert response.reason == "groq_error"


def test_server_error_is_generation_unavailable_groq_error() -> None:
    import groq as groq_sdk

    client = FakeGroqClient(error=_status_error(groq_sdk.InternalServerError, 500))
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), client)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "generation_unavailable"
    assert response.reason == "groq_error"


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "",
        None,
        "[]",
        '{"claims": []}',
        '{"documents": "cold-food-storage"}',
        '{"documents": [{"document_id": "cold-food-storage", "claims": [{"text": 42, "chunk_ids": []}]}]}',
        '{"documents": [{"document_id": "cold-food-storage", "claims": [{"text": "x", "chunk_ids": "cold-food-storage:51"}]}]}',
        '```json\n{"documents": []}\n```',
    ],
)
def test_invalid_json_is_generation_unavailable(content: str | None) -> None:
    client = FakeGroqClient(content=content)
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), client)

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "generation_unavailable"
    assert response.reason == "invalid_json"
    assert len(response.searched) == 7


def test_missing_api_key_is_generation_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    answer_pipeline = AnswerPipeline(
        retriever=FakeRetriever(two_document_retrieval()),
        generator=GroqClaimGenerator(model="fake-model"),
    )

    response = answer_pipeline.answer("How long do leftovers keep?")

    assert response.type == "generation_unavailable"
    assert response.reason == "groq_error"


# --- response shape ------------------------------------------------------------


def test_every_response_type_serialises_with_the_architecture_keys() -> None:
    groq = FakeGroqClient(
        payload(("cold-food-storage", [("Cooked meat or poultry leftovers keep 3 to 4 days in the refrigerator.", ["cold-food-storage:51"])]))
    )
    answer_pipeline, _, _ = pipeline(two_document_retrieval(), groq)
    responses = [
        answer_pipeline.answer("How long do leftovers keep?"),
        answer_pipeline.answer("What should I weigh?"),
        AnswerPipeline(retriever=FakeRetriever(result()), generator=GroqClaimGenerator(client=groq)).answer(
            "What is the tariff on imported olive oil?"
        ),
    ]

    assert [response.type for response in responses] == ["answer", "out_of_scope", "not_in_corpus"]
    for response in responses:
        body = json.loads(response.model_dump_json())
        assert set(body) == {"type", "reason", "message", "sections", "searched"}
        for entry in body["searched"]:
            assert set(entry) == {"document_id", "document_name", "publisher", "year"}
    answer_body = json.loads(responses[0].model_dump_json())
    section = answer_body["sections"][0]
    assert set(section) == {"document_id", "document_name", "publisher", "year", "source_url", "claims"}
    assert set(section["claims"][0]) == {"text", "chunk_ids", "citation"}
