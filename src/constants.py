"""Corpus registry, chunking thresholds, and embedding settings.

The registry is the allowlist for fetch, filtered retrieval, and citations.
`retrieval_date` is recorded in the corpus manifest when a file is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DATA_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
CORPUS_MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"

# Structure-aware chunking (implementation plan, phase 2). Tokens are estimated
# as round(word_count × 1.3). Pack paragraphs and list runs toward the target.
# Split a single paragraph on sentence boundaries only past the cap, with overlap.
CHUNK_TARGET_TOKENS = 400
MAX_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 50

TOP_K_ALL = 8
TOP_K_FILTERED = 5
MAX_CHUNKS_PER_DOCUMENT = 3
# Unfiltered retrieval fetches this many neighbours, then applies the
# per-document cap and TOP_K_ALL. 24 is 3 chunks per document times 8 kept.
CANDIDATE_K = 24
# Measured on the 372-chunk index (phase 4). In-corpus top hits run from
# 0.669 ("what does WHO say about limiting free sugars") to 0.899. Off-topic
# questions and filtered misses peak at 0.604 (the Eatwell Guide asked how
# long raw chicken keeps). Tune inside 0.60–0.66: below 0.60 that miss is
# kept, above 0.66 the free-sugars question collapses. The band is for
# queries prefixed with BGE_QUERY_PREFIX.
MIN_SIMILARITY = 0.62

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE v1.5 asymmetric retrieval: queries use this prefix; chunk text does not.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Measured on data/chunks/chunks.jsonl: every embed_text is at most 479 tokens
# under this tokenizer, inside the 512-token window. See implementation plan phase 3.
BGE_EMBEDDING_DIM = 384
BGE_MAX_SEQUENCE_LENGTH = 512

INDEX_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"
CHROMA_COLLECTION_NAME = "guidance"

# Answer layer (implementation plan, phase 6). GROQ_MODEL in the environment
# overrides the default. Temperature 0 keeps the JSON shape stable.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.0
GROQ_MAX_TOKENS = 1024
# The validator's lexical support check: every number in a claim must appear
# in its cited chunks, and at least this share of the claim's content words
# must appear there too. A claim below either bar is dropped.
MIN_CLAIM_TERM_OVERLAP = 0.5


@dataclass(frozen=True)
class CorpusDocument:
    """One official guidance document in the corpus allowlist."""

    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    kind: str
    aliases: tuple[str, ...]
    # True when `source_url` is a publication landing page rather than the
    # document itself. The fetcher follows the English PDF link on that page
    # and stores the PDF; `source_url` stays the citation link.
    landing_page: bool = False


CORPUS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        document_id="cold-food-storage",
        document_name="Cold Food Storage Chart",
        publisher="FoodSafety.gov (HHS / FDA / USDA)",
        year=2023,
        source_url="https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts",
        kind="html",
        aliases=("cold food storage chart", "cold food storage"),
    ),
    CorpusDocument(
        document_id="who-healthy-diet",
        document_name="Healthy diet (Fact sheet)",
        publisher="World Health Organization",
        year=2018,
        source_url="https://www.who.int/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf",
        kind="pdf",
        aliases=("healthy diet fact sheet", "who fact sheet"),
    ),
    CorpusDocument(
        document_id="fao-who-healthy-diets",
        document_name="What are healthy diets? Joint statement",
        publisher="FAO and WHO",
        year=2024,
        source_url="https://www.who.int/publications/i/item/9789240101876",
        kind="html",
        aliases=("what are healthy diets", "fao and who"),
        landing_page=True,
    ),
    CorpusDocument(
        document_id="who-five-keys",
        document_name="Five keys to safer food manual",
        publisher="World Health Organization",
        year=2006,
        source_url="https://www.who.int/publications/i/item/9789241594639",
        kind="html",
        aliases=("five keys", "five keys to safer food"),
        landing_page=True,
    ),
    CorpusDocument(
        document_id="eatwell-guide",
        document_name="The Eatwell Guide booklet",
        publisher="Public Health England (now OHID)",
        year=2018,
        source_url="https://assets.publishing.service.gov.uk/media/5ba8a50540f0b605084c9501/Eatwell_Guide_booklet_2018v4.pdf",
        kind="pdf",
        aliases=("eatwell", "eatwell guide"),
    ),
    CorpusDocument(
        document_id="fsa-chill",
        document_name="How to chill, freeze and defrost food safely",
        publisher="Food Standards Agency (UK)",
        year=2017,
        source_url="https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely",
        kind="html",
        aliases=("chill freeze and defrost", "food standards agency"),
    ),
    CorpusDocument(
        document_id="kitchen-companion",
        document_name="Kitchen Companion: Your Safe Food Handbook",
        publisher="USDA Food Safety and Inspection Service",
        year=2008,
        source_url="https://www.fsis.usda.gov/sites/default/files/media_file/2020-12/Kitchen-Companion.pdf",
        kind="pdf",
        aliases=("kitchen companion",),
    ),
)

DOCUMENT_IDS: frozenset[str] = frozenset(document.document_id for document in CORPUS)
SOURCE_URLS: frozenset[str] = frozenset(document.source_url for document in CORPUS)
