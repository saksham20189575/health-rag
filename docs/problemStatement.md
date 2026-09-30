# Dietary Guidance RAG Chatbot — Problem Statement

> Milestone 2, Week 7. Source: `docs/problemStatement.txt`.

## 1. Brief

Build a **prototype chatbot** that answers questions about **food, nutrition and food safety**. The chatbot sits on top of a **retrieval layer** (RAG — Retrieval-Augmented Generation) so that it answers **only from official, public dietary guidance documents**.

Three non-negotiable properties define the assistant's behaviour:

1. **Grounded answers only.** Every answer is produced from retrieved chunks of the corpus, never from the model's general knowledge.
2. **Every claim carries a citation.** Each statement points back to the document it came from (name, publisher, year, link).
3. **Honest refusal.** When the guidance does not cover a question, the assistant says so explicitly rather than guessing.

## 2. Motivation and Context

### Where this fits in the larger project

In the final project this chatbot becomes the service responsible for answering two families of questions:

- *"Is this a reasonable way to eat?"* — general dietary pattern guidance.
- *"How long can I keep this in the fridge?"* — food storage and safety guidance.

### Why RAG

Health authorities already publish authoritative guidance on exactly these topics — long, careful, dense PDFs written in prose. There is **no API** for this content, and almost nobody reads it end-to-end. That gap between *"the answer exists in an official document"* and *"a person can actually get that answer"* is precisely what a retrieval layer is designed to close.

### What is explicitly *not* part of this milestone

**Nutrient numbers for individual foods** (e.g. calories, grams of protein in 100 g of chicken) are a different kind of data. They come from a **structured database in Milestone 3**, not from prose guidance documents. They do not belong in this corpus or in this chatbot's answers.

## 3. The Pipeline — What You Build

The system is composed of six pieces, roughly in data-flow order: corpus → chunking → retrieval → answer layer, plus two cross-cutting behaviours (cross-document handling and refusals).

### 3.1 Corpus

Gather **5 to 7 public guidance documents** from recognised authorities. Acceptable sources include:

- National nutrition institutes
- Food safety regulators
- International health bodies

Constraints:

- **Written prose only.** Anything that has a clean API behind it does not belong here — that data is served differently (see Milestone 3).
- **Provenance metadata is mandatory.** For every document, store:
  - Publisher
  - Year
  - Source URL
  - Retrieval date (the date you actually download the file)

Use these **seven public documents** as the corpus. They cover both sides of the product: dietary pattern guidance ("is this a reasonable way to eat") and food storage and handling ("how long can I keep this in the fridge"). Nutrition institutes and food safety regulators are both represented, so a question such as cooking oil can be answered per document.

| # | Document | Publisher | Year | Source URL |
|---|----------|-----------|------|------------|
| 1 | Cold Food Storage Chart | FoodSafety.gov (U.S. Department of Health and Human Services / FDA / USDA) | 2023 | https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts |
| 2 | Healthy diet (Fact sheet) | World Health Organization | 2018 | https://www.who.int/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf |
| 3 | What are healthy diets? Joint statement | Food and Agriculture Organization of the United Nations and World Health Organization | 2024 | https://www.who.int/publications/i/item/9789240101876 |
| 4 | Five keys to safer food manual | World Health Organization | 2006 | https://www.who.int/publications/i/item/9789241594639 |
| 5 | The Eatwell Guide booklet | Public Health England (now Office for Health Improvement and Disparities) | 2018 | https://assets.publishing.service.gov.uk/media/5ba8a50540f0b605084c9501/Eatwell_Guide_booklet_2018v4.pdf |
| 6 | How to chill, freeze and defrost food safely | Food Standards Agency (UK) | 2017 | https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely |
| 7 | Kitchen Companion: Your Safe Food Handbook | USDA Food Safety and Inspection Service | 2008 | https://www.fsis.usda.gov/sites/default/files/media_file/2020-12/Kitchen-Companion.pdf |

Documents 2, 3, and 5 are nutrition guidance. Documents 1, 4, 6, and 7 are food-safety guidance. The nutrition documents discuss fats and oils, while the food-safety documents discuss storage times, chilling, and leftovers — the cross-document case in §3.5. Document 1 is largely tabular (food type → refrigerator / freezer time), so it is a direct test of the chunking constraint in §3.2: a fixed-size splitter will cut table rows apart.

### 3.2 Chunking

Split each document into chunks for indexing. Requirements:

- Every chunk must carry:
  - Document name
  - Publisher
  - Year
  - Section heading
- These documents are **full of tables and numbered recommendations**. Naive fixed-size chunking will cut these in half and destroy their meaning. The chunking strategy must account for this.
- **Document the trade-off in the README**: state which chunking approach was chosen and what it cost (e.g. complexity, uneven chunk sizes, lost context, implementation effort).

### 3.3 Retrieval

Build a **vector index** over the chunks. It must support two retrieval modes:

1. **Cross-corpus retrieval** — search across all documents at once.
2. **Filtered retrieval** — search restricted to **one named document**.

### 3.4 Answer Layer

The assistant generates answers **only from the retrieved chunks**.

Every claim in an answer must carry a citation that shows:

- Document name
- Publisher
- Year
- A link to the source

### 3.5 Cross-Document Questions

Some questions are addressed by more than one document. Example: **cooking oil**, where a nutrition institute (dietary fat recommendations) and a food safety regulator (storage, reuse, smoke point) both have something to say.

Rules for these cases:

- **Answer per document**, each with its **own separate citations**.
- **Never blend two sources into one claim** about what "the guidelines say". Sources may disagree or cover different aspects; the answer must keep them distinct.

### 3.6 Two Kinds of Refusal

The assistant needs **both** of the following refusal behaviours, and they must be distinguishable.

#### (a) Not in the corpus

When the retrieved chunks do not contain the answer, the assistant:

- States that the guidance does not cover the question, and
- **Names what it searched** (which documents / scope were consulted), so the user understands the boundary of the answer.

#### (b) Out of scope by design

Certain topics are deliberately excluded regardless of whether the corpus mentions them:

- No **medical advice**
- No **calorie or weight targets**
- Nothing about **what anyone should weigh**

For these, the assistant **declines** and **points the person to a qualified professional**.

This must be **enforced in code** (a guardrail / policy layer), not left to prompt wording alone — "same as last week", i.e. following the same enforcement pattern established in the previous milestone.

## 4. Summary of Deliverables

| # | Component | Key requirement |
|---|-----------|-----------------|
| 1 | Corpus | The seven public documents in §3.1, each stored with publisher, year, source URL, and retrieval date |
| 2 | Chunking | Structure-aware; each chunk tagged with doc name, publisher, year, section heading; trade-offs documented in README |
| 3 | Retrieval | Vector index; supports all-document and single-document filtered search |
| 4 | Answer layer | Answers only from retrieved chunks; every claim cited (name, publisher, year, link) |
| 5 | Cross-document handling | Per-document answers with separate citations; never merge sources into one claim |
| 6a | Refusal — not in corpus | Say the guidance doesn't cover it; name what was searched |
| 6b | Refusal — out of scope | Decline medical / calorie / weight questions; refer to a professional; enforced in code |

## 5. Guiding Principles

- **Groundedness over helpfulness.** A correct "the guidance doesn't cover this" beats a plausible-sounding guess.
- **Traceability.** A reader should be able to follow any claim back to a specific document, publisher, year and URL.
- **Source integrity.** Distinct authorities stay distinct; the system never manufactures a consensus.
- **Safety by construction.** Scope limits (no medical advice, no weight targets) are code-level guardrails, not suggestions to the model.
- **Right data, right place.** Prose guidance lives here; per-food nutrient numbers live in the Milestone 3 structured database.
