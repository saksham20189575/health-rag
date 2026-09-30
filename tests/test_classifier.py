"""Scope guard and alias detector (implementation plan, phase 5).

The guard is code, not a prompt: each reason code is a set of regular
expressions, and population guidance stays in scope.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.constants import CORPUS
from src.rag.classifier import (
    PROFESSIONAL_REFERRAL_REASONS,
    SCOPE_REASONS,
    ScopeDecision,
    classify,
    classify_scope,
    detect_document_id,
    detect_document_ids,
    normalize_message,
    registry_order,
    scope_reason,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- reason codes -----------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What should I weigh?",
        "How much should I weigh?",
        "What is my ideal weight?",
        "What weight should I be for my height?",
        "How much weight should I lose?",
        "How do I lose weight?",
        "I want to lose 5 kg, what should I eat?",
        "Am I overweight?",
        "What's my BMI?",
        "What is a healthy weight for me?",
    ],
)
def test_weight_target(message: str) -> None:
    assert scope_reason(message) == "weight_target"


@pytest.mark.parametrize(
    "message",
    [
        "How many calories should I eat a day?",
        "How many calories do I need?",
        "How many kcal should I be eating to lose weight?",
        "What should my daily calorie intake be?",
        "Should I be in a calorie deficit?",
        "Is a 500 calorie deficit safe?",
        "What is a good calorie target for me?",
        "How many calories to lose a pound a week?",
    ],
)
def test_calorie_target(message: str) -> None:
    assert scope_reason(message) == "calorie_target"


@pytest.mark.parametrize(
    "message",
    [
        "I have diabetes, what should I eat?",
        "I've been diagnosed with high cholesterol, which foods should I avoid?",
        "I am pregnant, can I eat soft cheese?",
        "Is honey safe for diabetics?",
        "Is it safe to eat soft cheese if I have a kidney condition?",
        "What are the symptoms of salmonella?",
        "Can I eat grapefruit with my statins?",
        "How do I lower my blood pressure with diet?",
        "Should I take a vitamin D supplement?",
        "What medication helps with food poisoning?",
        "My son has coeliac disease, what can he eat?",
        "How do I treat diarrhoea? What are the treatments?",
    ],
)
def test_medical(message: str) -> None:
    assert scope_reason(message) == "medical"


@pytest.mark.parametrize(
    "message",
    [
        "Protein in 100 g of chicken",
        "How many grams of protein are in 100 g of chicken?",
        "How many calories are in a banana?",
        "How much sugar is in a can of coke?",
        "How much protein does chicken breast have?",
        "how many calories in a slice of pizza",
        "how much sodium in soy sauce",
        "Nutritional value of an avocado",
        "Does spinach contain iron?",
        "How much protein per 100g of lentils?",
        "What is the calorie count of a Big Mac?",
    ],
)
def test_nutrient_lookup(message: str) -> None:
    assert scope_reason(message) == "nutrient_lookup"


@pytest.mark.parametrize(
    "message",
    [
        # The phase 4 probes and the architecture §10 worked paths.
        "How long can I keep cooked leftovers in the fridge?",
        "How long can I keep eggs in the fridge?",
        "How much salt per day?",
        "What is the danger zone temperature?",
        "What do the guidelines say about cooking oil?",
        "What does WHO say about limiting free sugars?",
        "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
        "What is the tariff on imported olive oil?",
        "How do I refinance my mortgage?",
        "How do I thaw a frozen turkey?",
        # Population guidance stays in scope.
        "What does the guidance say about free sugars?",
        "Is salt under 5 g per day the recommendation?",
        "How much salt per adult per day?",
        "How much saturated fat should adults eat per day?",
        "How much salt in a healthy diet?",
        "How much fibre in the Eatwell Guide?",
        "What does the Eatwell Guide say about maintaining a healthy weight?",
        "What is the recommended daily energy intake for adults in the Eatwell Guide?",
        "Which foods are high in fibre?",
        "Should I eat less red meat?",
        "How many portions of fruit should I eat?",
        "How much water should I drink a day?",
        # First-person food-safety questions with no condition or target.
        "Is it safe to eat leftovers after 4 days?",
        "I have leftover rice, how long does it keep?",
        "Can I refreeze thawed chicken?",
        "How should I wash my hands before cooking?",
        "How do I prevent food poisoning?",
        "What temperature should I cook chicken to?",
        "Why should I limit sugary treats?",
        "What are the five keys to safer food?",
    ],
)
def test_in_scope(message: str) -> None:
    assert classify_scope(message) is None
    assert classify(message).in_scope


def test_reason_order_prefers_personal_target_over_nutrient_lookup() -> None:
    # Contains a calorie word and a first-person target: a calorie target,
    # not a lookup of calories in a food.
    assert scope_reason("How many calories should I eat to lose weight?") == "calorie_target"
    # Calories in a named food, no personal target.
    assert scope_reason("How many calories are in a banana?") == "nutrient_lookup"


def test_classify_scope_returns_reason_and_pattern() -> None:
    match = classify_scope("What should I weigh?")
    assert match is not None
    reason, pattern = match
    assert reason == "weight_target"
    assert pattern  # the regex source, for the phase 9 log


def test_empty_message_is_not_a_guard_match() -> None:
    assert classify_scope("") is None
    assert classify_scope("   ") is None


def test_reason_constants() -> None:
    assert PROFESSIONAL_REFERRAL_REASONS == ("medical", "calorie_target", "weight_target")
    assert set(SCOPE_REASONS) == {"medical", "calorie_target", "weight_target", "nutrient_lookup"}


def test_normalize_message_unifies_quotes_and_whitespace() -> None:
    assert normalize_message("  What’s   MY  BMI? ") == "what's my bmi?"
    assert scope_reason("What’s my BMI?") == "weight_target"


# --- alias detection --------------------------------------------------------


def test_eatwell_alias_sets_eatwell_guide() -> None:
    message = "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?"
    assert detect_document_id(message) == "eatwell-guide"
    assert classify(message) == ScopeDecision(reason=None, document_id="eatwell-guide")


def test_who_fact_sheet_wording_sets_who_healthy_diet() -> None:
    assert detect_document_id("What does the WHO fact sheet say about free sugars?") == "who-healthy-diet"
    assert detect_document_id("healthy diet fact sheet on salt") == "who-healthy-diet"


def test_bare_who_is_ambiguous_and_stays_unfiltered() -> None:
    # Three registry documents are WHO publications; "WHO" alone names none.
    message = "What does WHO say about limiting free sugars?"
    assert classify_scope(message) is None
    assert detect_document_id(message) is None
    assert classify(message).document_id is None


@pytest.mark.parametrize(
    ("message", "document_id"),
    [
        ("Five keys manual: how should I wash my hands?", "who-five-keys"),
        ("What does the Food Standards Agency say about defrosting?", "fsa-chill"),
        ("What are healthy diets according to FAO and WHO?", "fao-who-healthy-diets"),
        ("What does the Kitchen Companion say about thawing turkey?", "kitchen-companion"),
        ("Cold food storage chart: how long do eggs keep?", "cold-food-storage"),
        ("eatwell's advice on oils", "eatwell-guide"),
        ("What does the Healthy diet (Fact sheet) say about salt?", "who-healthy-diet"),
        ("In 'What are healthy diets? Joint statement', what is a healthy diet?", "fao-who-healthy-diets"),
    ],
)
def test_alias_detection(message: str, document_id: str) -> None:
    assert detect_document_id(message) == document_id


def test_every_registry_name_and_alias_is_detected() -> None:
    for document in CORPUS:
        for alias in (document.document_name, *document.aliases):
            assert detect_document_id(f"What does {alias} say?") == document.document_id, alias


def test_two_named_documents_leave_scope_unfiltered() -> None:
    message = "Compare the Eatwell Guide and the Kitchen Companion on oils"
    assert detect_document_ids(message) == ("eatwell-guide", "kitchen-companion")
    assert detect_document_id(message) is None
    assert classify(message).document_id is None


def test_no_named_document_is_unfiltered() -> None:
    assert detect_document_ids("How long can I keep cooked leftovers in the fridge?") == ()
    assert classify("How long can I keep cooked leftovers in the fridge?").document_id is None


def test_explicit_document_id_overrides_aliases() -> None:
    decision = classify("According to the Eatwell Guide, how long do eggs keep?", document_id="fsa-chill")
    assert decision.document_id == "fsa-chill"
    assert decision.in_scope


def test_explicit_document_id_outside_registry_raises() -> None:
    with pytest.raises(ValueError):
        classify("How long do eggs keep?", document_id="not-a-document")


def test_guard_match_wins_over_alias() -> None:
    decision = classify("According to the Eatwell Guide, what should I weigh?")
    assert decision.reason == "weight_target"
    assert decision.document_id is None
    assert not decision.in_scope


def test_registry_order() -> None:
    ids = [document.document_id for document in CORPUS]
    assert registry_order(reversed(ids)) == tuple(ids)
    assert registry_order(["kitchen-companion", "eatwell-guide", "eatwell-guide"]) == (
        "eatwell-guide",
        "kitchen-companion",
    )
    with pytest.raises(ValueError):
        registry_order(["nope"])


# --- isolation ---------------------------------------------------------------


def test_classifier_imports_neither_retriever_nor_groq() -> None:
    """Deliverable: the guard runs without the retriever, Chroma, or Groq."""
    code = (
        "import sys\n"
        "import src.rag.classifier\n"
        "loaded = set(sys.modules)\n"
        "for name in ('src.rag.retriever', 'chromadb', 'groq', 'sentence_transformers'):\n"
        "    assert not any(m == name or m.startswith(name + '.') for m in loaded), name\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
