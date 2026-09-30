"""Scope guard and document-alias detection (implementation plan, phase 5).

The guard runs on the raw user message before retrieval and before any Groq
call. It is deterministic: a set of regular expressions per reason code, and
a fixed reason string on a match. Population guidance stays in scope ("what
does the guidance say about free sugars", "salt under 5 g per day"). The
patterns target a personal target ("how many calories should I eat", "what
should I weigh"), a personal medical situation ("I have diabetes"), or a
nutrient number for a named food ("protein in 100 g of chicken").

This module imports only ``src.constants``. It must not import the retriever,
Chroma, or the Groq client, so a test can prove the guard runs on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Pattern, Sequence

from src.constants import CORPUS, DOCUMENT_IDS

ScopeReason = Literal["medical", "calorie_target", "weight_target", "nutrient_lookup"]

# Reasons that decline and refer the person to a qualified professional.
PROFESSIONAL_REFERRAL_REASONS: tuple[ScopeReason, ...] = (
    "medical",
    "calorie_target",
    "weight_target",
)
SCOPE_REASONS: tuple[ScopeReason, ...] = PROFESSIONAL_REFERRAL_REASONS + ("nutrient_lookup",)

# First-person markers. A pattern that needs a personal target requires one of
# these so that population guidance ("adults should eat under 5 g of salt")
# is not caught.
_PERSONAL = r"\b(?:i|i'm|i've|i'd|me|my|myself|mine)\b"

# Conditions, illnesses, and clinical words used by the medical patterns.
_CONDITION = (
    r"(?:diabet\w*|pre-?diabet\w*|hypertens\w*|high blood pressure|low blood pressure|"
    r"cholesterol|kidney|renal|liver|heart (?:disease|condition|failure)|"
    r"celiac|coeliac|crohn'?s?|colitis|ibs|irritable bowel|gout|an(?:a)?emi\w*|"
    r"cancer|chemo\w*|thyroid|pcos|osteoporosis|arthritis|"
    r"allerg\w*|intoleran\w*|"
    r"food poisoning|salmonella|listeria|e\.? ?coli|norovirus|"
    r"diarrh?oea|diarrhea|nausea|vomit\w*|fever|infection|"
    r"disease|condition|disorder|syndrome|illness|ill|sick|unwell|pregnan\w*|breastfeeding)"
)

_MEDICAL: tuple[Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "I have diabetes", "I've been diagnosed with ...", "I suffer from ..."
        rf"\b(?:i have|i've got|i got|i've been diagnosed|i was diagnosed|i am diagnosed|"
        rf"i suffer from|i'm suffering from|i am suffering from|i live with|i'm|i am|"
        rf"my (?:child|son|daughter|partner|wife|husband|mother|father|mum|dad) (?:has|is))"
        rf"\b[^.?!]{{0,40}}?\b{_CONDITION}\b",
        # Diagnosis, symptoms, medicines, treatment.
        r"\b(?:diagnos\w*|symptoms?|medication\w*|medicines?|prescri\w*|treatments?|"
        r"cure|cures|curing|dosage|doses?|side[- ]effects?|antibiotic\w*|insulin|"
        r"statins?|metformin|ibuprofen|paracetamol|acetaminophen|aspirin|antacids?|"
        r"laxatives?|painkillers?)\b",
        # "Should I take a supplement / vitamin / tablet" and medicine-taking.
        rf"{_PERSONAL}\b[^.?!]{{0,30}}?\b(?:take|taking|start|stop)\b[^.?!]{{0,20}}?"
        r"\b(?:supplements?|vitamins?|tablets?|pills?|capsules?|medicine|medication)\b",
        # Personal clinical markers: "lower my cholesterol", "my blood sugar".
        r"\b(?:my|lower my|reduce my|control my|manage my|improve my|raise my)\b[^.?!]{0,20}?"
        r"\b(?:blood sugar|blood glucose|glucose|blood pressure|cholesterol|a1c|hba1c|"
        r"triglycerides?|iron levels?|haemoglobin|hemoglobin)\b",
        # Whether a food is safe for a stated illness or patient group.
        rf"\b(?:safe|ok|okay|good|bad|allowed|suitable|fine|harmful)\b[^.?!]{{0,40}}?"
        rf"\b(?:for|with|if|when)\b[^.?!]{{0,30}}?\b(?:{_CONDITION}|someone with|people with|"
        r"patients?|a person with|diabetics?)\b",
        r"\bdiabetics?\b",
        r"\b(?:for|with|if you have|if i have)\s+(?:a\s+)?(?:diabetes|hypertension|gout|"
        r"celiac|coeliac|kidney disease|heart disease|cancer|ibs|crohn'?s)\b",
    )
)

_CALORIE_WORD = r"(?:calories?|caloric|kcal|kilocalories?|energy intake)"

_CALORIE_TARGET: tuple[Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "How many calories should I eat a day?", "how many kcal do I need"
        rf"\bhow (?:many|much)\b[^.?!]{{0,20}}?\b{_CALORIE_WORD}\b[^.?!]{{0,40}}?"
        rf"\b(?:should|do|does|can|could|must|need|needs|ought|would)\b[^.?!]{{0,15}}?{_PERSONAL}",
        # "What should my calorie intake be", "my daily calorie goal"
        rf"\b(?:my|for me|for myself)\b[^.?!]{{0,25}}?\b{_CALORIE_WORD}\b",
        rf"\b{_CALORIE_WORD}\b[^.?!]{{0,25}}?\b(?:for me|for myself|i should|should i|do i need|i need)\b",
        # A deficit or a personal goal is a personal target by definition.
        r"\b(?:calori\w* (?:deficit|surplus|target|goal|budget|allowance|limit)|"
        r"(?:deficit|surplus|target|goal|budget|allowance|limit) of \d[\d,]* ?(?:calories|kcal))\b",
        # Calories tied to losing or gaining weight.
        rf"\b{_CALORIE_WORD}\b[^.?!]{{0,40}}?\b(?:lose|losing|gain|gaining|drop|shed|cut)\b[^.?!]{{0,10}}?\bweight\b",
        rf"\b(?:lose|losing|gain|gaining|drop|shed|cut)\b[^.?!]{{0,10}}?\bweight\b[^.?!]{{0,40}}?\b{_CALORIE_WORD}\b",
        rf"\bhow (?:many|much)\b[^.?!]{{0,20}}?\b{_CALORIE_WORD}\b[^.?!]{{0,20}}?\bto (?:lose|gain|maintain)\b",
    )
)

_WEIGHT_TARGET: tuple[Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "What should I weigh?", "how much should I weigh", "what weight should I be"
        r"\b(?:what|how much|how many (?:kilos|kilograms|kg|pounds|lbs|stone))\b[^.?!]{0,20}?"
        r"\b(?:should|do|must|ought)\b[^.?!]{0,6}?\bi\b[^.?!]{0,15}?\b(?:weigh|be weighing)\b",
        r"\bwhat (?:body )?weight (?:should|must|ought) i be\b",
        # "ideal weight", "target weight", "goal weight" (a personal target by wording).
        r"\b(?:ideal|target|goal|perfect|optimal|correct|right|proper|recommended)\s+(?:body\s+)?weight\b",
        r"\b(?:a |an )?(?:healthy|normal|good) (?:body )?weight for (?:me|my|someone|a person|a man|a woman|my height|my age)\b",
        r"\bmy (?:ideal |target |goal |healthy |normal |current |own )?(?:body )?weight\b",
        # "How much weight should I lose", "how many kilos should I lose"
        r"\bhow (?:much|many)\b[^.?!]{0,20}?\b(?:weight|kilos|kilograms|kg|pounds|lbs|stone)\b[^.?!]{0,10}?"
        r"\b(?:should|do|must|can)\b[^.?!]{0,6}?\bi\b[^.?!]{0,10}?\b(?:lose|gain|drop|shed|put on)\b",
        # "How do I lose weight", "help me lose weight", "I want to lose weight"
        rf"{_PERSONAL}\b[^.?!]{{0,30}}?\b(?:lose|losing|gain|gaining|drop|dropping|shed|shedding|put on)\b"
        r"[^.?!]{0,10}?\b(?:weight|kilos|kilograms|kg|pounds|lbs|stone|belly fat|body fat)\b",
        # "Am I overweight", "my BMI", "what is my bmi"
        r"\bam i (?:overweight|underweight|obese|too (?:fat|thin|heavy|light|skinny))\b",
        r"\bmy bmi\b",
        r"\bwhat (?:is|'s|should) my (?:bmi|body mass index)\b",
        r"\bbmi (?:for me|should i)\b",
    )
)

_NUTRIENT = (
    r"(?:calories?|kcal|kilocalories?|kilojoules?|kj|protein|carbs?|carbohydrates?|"
    r"fat|fats|saturated fat|saturates|trans fat|sugars?|salt|sodium|fib(?:re|er)|"
    r"iron|calcium|zinc|magnesium|potassium|phosphorus|selenium|iodine|"
    r"cholesterol|omega[- ]?3|vitamin\s*[a-k]?\d*|vitamins?|minerals?|nutrients?|"
    r"macros?|macronutrients?|micronutrients?|nutritional (?:value|content|information|facts)|"
    r"nutrition (?:facts|information|label))"
)
# A food serving, the thing that makes a nutrient question a per-food lookup.
_SERVING = (
    r"(?:a|an|one|two|three|1|2|3|100 ?g|100 ?grams?|100 ?ml|per 100|per serving|per portion|"
    r"a cup|a glass|a bowl|a slice|a serving|a portion|a piece|a tablespoon|a teaspoon|a tbsp|a tsp|"
    r"a can|a bottle|a handful|a scoop|\d+ ?(?:g|grams?|ml|oz|ounces?|cups?|slices?|servings?|pieces?|tbsp|tsp))"
)

# Words that turn "<nutrient> in a ..." or "<nutrient> per ..." back into
# population guidance: "salt in a healthy diet", "sugar in the Eatwell Guide",
# "salt per day". Up to three words may sit between the determiner and the noun.
_GUIDANCE_NOUN = (
    r"(?:diet|diets|day|days|week|month|year|meal|meals|guidance|guide|guidelines?|"
    r"recommend\w*|document|booklet|fact ?sheet|manual|handbook|statement|chart|corpus|"
    r"daily|person|adult|adults|child|children)"
)
_NOT_GUIDANCE = rf"(?!\s*(?:[\w'-]+\s+){{0,3}}?{_GUIDANCE_NOUN}\b)"

_NUTRIENT_LOOKUP: tuple[Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "protein in 100 g of chicken", "calories in a banana", "sugar in a can of coke"
        rf"\b{_NUTRIENT}\b[^.?!]{{0,15}}?\b(?:are |is )?(?:in|per)\b\s+{_SERVING}\b{_NOT_GUIDANCE}",
        # "how much protein does chicken have", "how many calories does an apple contain"
        rf"\bhow (?:many|much)\b[^.?!]{{0,15}}?\b{_NUTRIENT}\b[^.?!]{{0,40}}?"
        r"\b(?:does|do|is there|are there|has|have|contain|contains|in)\b[^.?!]{0,30}?"
        r"\b(?:have|contain|contains|got|in it|in them)\b",
        # "how many calories in chicken breast", "how much sodium in soy sauce".
        # Not "how much salt per day" or "how much sugar in the diet": those
        # are population guidance, so "per <time>" and "in the/a diet" are excluded.
        rf"\bhow (?:many|much)\b\s+{_NUTRIENT}\b\s+(?:are |is )?(?:there )?(?:in|per)\b{_NOT_GUIDANCE}",
        # "nutritional value of X", "nutrition facts for X", "macros of X"
        r"\b(?:nutritional (?:value|content|information|facts|breakdown|profile)|"
        r"nutrition (?:facts|information|label|breakdown)|macros?|macronutrient breakdown)\b"
        r"[^.?!]{0,10}?\b(?:of|for|in|on)\b",
        # "does X contain protein", "is there iron in spinach", "X calorie count"
        rf"\b(?:does|do)\b[^.?!]{{0,30}}?\b(?:contain|have)\b[^.?!]{{0,10}}?\b(?:any |much |a lot of )?{_NUTRIENT}\b",
        rf"\bis there\b[^.?!]{{0,10}}?\b{_NUTRIENT}\b[^.?!]{{0,10}}?\bin\b",
        rf"\b(?:calorie|kcal|energy) (?:count|content|value)\b",
        rf"\b{_NUTRIENT} (?:content|count|value|amount|level)s?\b[^.?!]{{0,10}}?\b(?:of|in)\b",
    )
)

_REASON_PATTERNS: tuple[tuple[ScopeReason, tuple[Pattern[str], ...]], ...] = (
    ("medical", _MEDICAL),
    ("calorie_target", _CALORIE_TARGET),
    ("weight_target", _WEIGHT_TARGET),
    ("nutrient_lookup", _NUTRIENT_LOOKUP),
)


@dataclass(frozen=True)
class ScopeDecision:
    """Outcome of the guard and the alias detector for one message.

    ``reason`` is ``None`` when the message is in scope. ``document_id`` is
    the single document the message names, an explicit override, or ``None``
    for the whole corpus. ``matched_pattern`` is the regex that fired, for
    the phase 9 log.
    """

    reason: ScopeReason | None
    document_id: str | None
    matched_pattern: str | None = None

    @property
    def in_scope(self) -> bool:
        return self.reason is None


def normalize_message(message: str) -> str:
    """Lower-case, unify quotes and dashes, and collapse whitespace."""
    text = message.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def classify_scope(message: str) -> tuple[ScopeReason, str] | None:
    """Return ``(reason, pattern)`` for the first matching guard, else ``None``.

    Reasons are tried in the order medical, calorie_target, weight_target,
    nutrient_lookup, so "how many calories should I eat to lose weight" is a
    calorie target and "calories in a banana" is a nutrient lookup.
    """
    text = normalize_message(message)
    if not text:
        return None
    for reason, patterns in _REASON_PATTERNS:
        for pattern in patterns:
            if pattern.search(text):
                return reason, pattern.pattern
    return None


def scope_reason(message: str) -> ScopeReason | None:
    """The reason code for ``message``, or ``None`` when it is in scope."""
    match = classify_scope(message)
    return match[0] if match else None


def _alias_pattern(alias: str) -> Pattern[str]:
    # Match on the alias's words; punctuation in the alias ("(Fact sheet)",
    # "diets? Joint") or the message may sit between them. A possessive may follow.
    words = [re.escape(word) for word in re.findall(r"[a-z0-9]+", normalize_message(alias))]
    return re.compile(r"\b" + r"[\W_]+".join(words) + r"(?:'s)?\b", re.IGNORECASE)


_ALIASES: tuple[tuple[str, Pattern[str]], ...] = tuple(
    (document.document_id, _alias_pattern(alias))
    for document in CORPUS
    for alias in (document.document_name, *document.aliases)
)


def detect_document_ids(message: str) -> tuple[str, ...]:
    """Every registry document the message names, in registry order."""
    text = normalize_message(message)
    found: list[str] = []
    for document_id, pattern in _ALIASES:
        if document_id not in found and pattern.search(text):
            found.append(document_id)
    return tuple(found)


def detect_document_id(message: str) -> str | None:
    """The one document the message names, or ``None``.

    ``None`` also when two or more documents are named, so retrieval stays
    unfiltered and both can be searched.
    """
    found = detect_document_ids(message)
    return found[0] if len(found) == 1 else None


def classify(message: str, document_id: str | None = None) -> ScopeDecision:
    """Run the guard, then resolve the retrieval scope.

    An explicit ``document_id`` overrides alias detection. A ``document_id``
    outside the registry raises ``ValueError``.
    """
    if document_id is not None and document_id not in DOCUMENT_IDS:
        raise ValueError(f"document_id is not in the registry: {document_id}")
    match = classify_scope(message)
    if match is not None:
        reason, pattern = match
        return ScopeDecision(reason=reason, document_id=None, matched_pattern=pattern)
    resolved = document_id if document_id is not None else detect_document_id(message)
    return ScopeDecision(reason=None, document_id=resolved)


def registry_order(document_ids: Sequence[str]) -> tuple[str, ...]:
    """``document_ids`` sorted by registry position; unknown ids raise."""
    ids = tuple(document_ids)
    order = {document.document_id: index for index, document in enumerate(CORPUS)}
    unknown = [document_id for document_id in ids if document_id not in order]
    if unknown:
        raise ValueError(f"document_id is not in the registry: {unknown[0]}")
    return tuple(sorted(set(ids), key=order.__getitem__))
