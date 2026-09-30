"""Prompt construction and strict payload parsing (phase 6; architecture §5.4)."""

from __future__ import annotations

import json

import pytest

from src.constants import CORPUS, GROQ_DEFAULT_MODEL
from src.rag.generator import (
    SYSTEM_PROMPT,
    GenerationError,
    GroqClaimGenerator,
    build_messages,
    build_passages,
    parse_payload,
)
from src.rag.retriever import DocumentHits, RetrievedChunk

BY_ID = {document.document_id: document for document in CORPUS}


def chunk(document_id: str, ordinal: int, text: str) -> RetrievedChunk:
    document = BY_ID[document_id]
    return RetrievedChunk(
        chunk_id=f"{document_id}:{ordinal}",
        text=text,
        similarity=0.8,
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        section_heading="Some > Heading",
        block_type="paragraph",
        retrieval_date="2026-09-27",
    )


GROUPS = (
    DocumentHits("cold-food-storage", (chunk("cold-food-storage", 51, "Leftovers: 3 to 4 days"),)),
    DocumentHits(
        "kitchen-companion",
        (chunk("kitchen-companion", 37, "Leftovers keep 3 to 4 days."), chunk("kitchen-companion", 140, "  Reheat to 165°F.  ")),
    ),
)


def test_passages_carry_document_id_chunk_id_and_text_only() -> None:
    passages = build_passages(GROUPS)

    assert passages == (
        "document_id: cold-food-storage\n"
        "[cold-food-storage:51] Leftovers: 3 to 4 days\n"
        "\n"
        "document_id: kitchen-companion\n"
        "[kitchen-companion:37] Leftovers keep 3 to 4 days.\n"
        "[kitchen-companion:140] Reheat to 165°F."
    )
    for document in CORPUS:
        assert document.publisher not in passages
        assert document.source_url not in passages
        assert document.document_name not in passages
    assert "Some > Heading" not in passages
    assert "2026-09-27" not in passages


def test_messages_are_system_then_user_with_question_and_passages() -> None:
    messages = build_messages("  How long do leftovers keep?  ", GROUPS)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"].startswith("Question: How long do leftovers keep?\n\nPassages:\n\n")
    assert build_passages(GROUPS) in messages[1]["content"]


def test_system_prompt_states_the_three_validator_rules() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "chunk_ids must be chosen from the chunk ids listed under that same document_id" in lowered
    assert "use only the passages" in lowered
    assert "empty claims list" in lowered
    assert '"documents"' in SYSTEM_PROMPT and '"chunk_ids"' in SYSTEM_PROMPT


def test_messages_need_at_least_one_group() -> None:
    with pytest.raises(ValueError):
        build_messages("question", ())


def test_parse_payload_accepts_the_architecture_shape() -> None:
    payload = parse_payload(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "who-healthy-diet",
                        "claims": [
                            {
                                "text": "Intake of saturated fats should be less than 10% of total energy intake.",
                                "chunk_ids": ["who-healthy-diet:12"],
                            }
                        ],
                    }
                ]
            }
        )
    )

    assert [document.document_id for document in payload.documents] == ["who-healthy-diet"]
    assert payload.documents[0].claims[0].chunk_ids == ["who-healthy-diet:12"]


def test_parse_payload_ignores_extra_keys_and_allows_empty_claims() -> None:
    payload = parse_payload(
        '{"documents": [{"document_id": "fsa-chill", "claims": [], "note": "x"}], "extra": 1}'
    )
    assert payload.documents[0].claims == []


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "plain prose",
        "[]",
        "null",
        '"documents"',
        '{"documents": null}',
        '{"documents": {}}',
        '{"documents": [{"claims": []}]}',
        '{"documents": [{"document_id": 7, "claims": []}]}',
        '{"documents": [{"document_id": "fsa-chill", "claims": [{"text": "x"}]}]}',
        '{"documents": [{"document_id": "fsa-chill", "claims": [{"text": "x", "chunk_ids": [1]}]}]}',
        '{"documents": [{"document_id": "fsa-chill", "claims": [{"text": None, "chunk_ids": []}]}]}',
        '```json\n{"documents": []}\n```',
        '{"documents": []} trailing',
    ],
)
def test_parse_payload_rejects_anything_else_as_invalid_json(content: str | None) -> None:
    with pytest.raises(GenerationError) as excinfo:
        parse_payload(content)
    assert excinfo.value.reason == "invalid_json"


def test_generation_error_reason_must_be_a_generation_unavailable_reason() -> None:
    assert GenerationError("groq_error").reason == "groq_error"
    assert GenerationError("quota_exceeded").reason == "quota_exceeded"
    with pytest.raises(ValueError):
        GenerationError("no_supporting_chunks")


def test_model_comes_from_argument_then_env_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    assert GroqClaimGenerator(client=object()).model == GROQ_DEFAULT_MODEL
    monkeypatch.setenv("GROQ_MODEL", "env-model")
    assert GroqClaimGenerator(client=object()).model == "env-model"
    assert GroqClaimGenerator(client=object(), model="arg-model").model == "arg-model"


def test_importing_the_generator_does_not_construct_a_groq_client() -> None:
    generator = GroqClaimGenerator(model="x")
    assert generator._client is None
