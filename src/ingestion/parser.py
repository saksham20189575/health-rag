"""Parse raw corpus files (PDF and HTML) into typed blocks.

A block is a ``heading``, ``paragraph``, ``list_item``, or ``table`` with its
text and the heading path above it (architecture §4.2). Headings update the
path and are not retrieval chunks themselves. The chunker (phase 2, task 2)
turns blocks into chunks.

PDF
    Text comes from the PyMuPDF text layer. Headings are recognised from font
    size and weight relative to the document's body text. Bulleted and
    numbered items are recognised from symbol-font markers or leading
    markers, and indented continuation lines are folded into the item.
    Running headers, footers, and page numbers are removed by cross-page
    repetition. Pages whose text is drawn rotated are re-rendered upright.
    Ruled tables are extracted with PyMuPDF's table finder and validated;
    layout grids with paragraph-length cells are rejected so their text is
    parsed as prose. Text that lies inside a kept table is dropped from the
    reading order so a grid is not indexed twice.

HTML
    The content root is the first of ``.govspeak``, ``article``, ``main``,
    ``body``. Navigation, cookie banners, feedback widgets, and footers are
    stripped. ``h1``-``h3`` become headings, ``p`` paragraphs, ``li`` list
    items (numbered when inside ``ol``), and ``<table>`` tables with
    ``rowspan``/``colspan`` carried into every row.

Scope
    Only guidance content is kept. Sections whose heading is references,
    index, contents, evaluation forms, contact details, and similar are
    skipped, and :data:`DOCUMENT_HINTS` lists the front- and back-matter
    pages of each PDF that carry no guidance (covers, poster graphics,
    tables of contents, legal notices). Image-only pages have no text layer
    and are recorded in the parse report.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pymupdf
from bs4 import BeautifulSoup, NavigableString, Tag

from src.constants import CORPUS, DATA_DIR, RAW_DATA_DIR, CorpusDocument

logger = logging.getLogger(__name__)

BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_LIST_ITEM = "list_item"
BLOCK_TABLE = "table"
BLOCK_TYPES = (BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_LIST_ITEM, BLOCK_TABLE)

# HTML uses h1-h3. PDF headings are ranked by font size, and a booklet may use
# five or six sizes; deeper ranks keep "Core information" nested under
# "KEEP CLEAN" instead of replacing it.
MAX_HEADING_LEVEL = 6
PARSED_DATA_DIR = DATA_DIR / "parsed"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Block:
    """One typed unit of a document."""

    type: str
    text: str
    heading_path: tuple[str, ...]
    page: int | None = None
    # Headings only.
    level: int | None = None
    # Tables only: column headers, data rows (same width as ``header``), and
    # the caption row(s) printed above the grid, if any.
    header: list[str] | None = None
    rows: list[list[str]] | None = None
    caption: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class ParseHints:
    """Per-document overrides for content the generic rules cannot classify."""

    # 1-indexed PDF pages that carry no guidance: covers, poster graphics,
    # tables of contents, legal notices, contact pages.
    skip_pages: frozenset[int] = frozenset()
    # Regexes; a block whose text matches any of them is dropped.
    drop_patterns: tuple[str, ...] = ()
    # Level-1 section titles by 1-indexed page, for PDFs whose section titles
    # are drawn as letter-spaced decoration or images and have no outline.
    page_titles: dict[int, str] = field(default_factory=dict)


@dataclass
class ParseReport:
    """What the parser kept and dropped for one document."""

    document_id: str
    kind: str
    pages_total: int = 0
    pages_parsed: int = 0
    pages_skipped: list[int] = field(default_factory=list)
    image_only_pages: list[int] = field(default_factory=list)
    blank_pages: list[int] = field(default_factory=list)
    rotated_pages: list[int] = field(default_factory=list)
    tables_kept: int = 0
    tables_rejected: int = 0
    skipped_sections: list[str] = field(default_factory=list)
    blocks_by_type: dict[str, int] = field(default_factory=dict)
    headings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ParsedDocument:
    document_id: str
    blocks: list[Block]
    report: ParseReport


# --------------------------------------------------------------------------
# Hints and scope rules
# --------------------------------------------------------------------------

# Section headings that introduce non-guidance content. When a heading matches,
# every block is skipped until the next heading of the same or a higher level.
SKIP_SECTION_HEADING = re.compile(
    r"^\W*("
    r"references?|bibliography|further reading|index|(table of )?contents|"
    r"acknowledg(e)?ments?|corrigend(a|um)|evaluation( forms?)?|answers|"
    r"for more information|still have questions\??|food safety contacts|"
    r"how can i find out more[^?]*\??|photography credits?|"
    r"about usda|about “?ask karen”?|glossary of terms"
    r")\W*$",
    re.IGNORECASE,
)

# Lines dropped everywhere: legal, credits, and identifiers that are not guidance.
GLOBAL_DROP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^photography credit",
        r"^isbn\b",
        r"^©",
        r"^all rights reserved",
        r"^some rights reserved",
        r"^gateway number",
        r"^(please )?change it at your discretion",
    )
)

DOCUMENT_HINTS: dict[str, ParseHints] = {
    "cold-food-storage": ParseHints(
        drop_patterns=(r"^looking for a specific item\?",),
    ),
    "fao-who-healthy-diets": ParseHints(
        # 1-7: cover, blank, title page, copyright, corrigenda, contents, photo.
        # 16-18: references. 19: Annex 1, agency terminology definitions.
        # 22-23: blank and contact page.
        skip_pages=frozenset({1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 22, 23}),
        # Stock-photo alt text, and the four labels of the principles graphic
        # (the principles are covered in sections 1.1-1.4).
        drop_patterns=(
            r"^a healthy human heart made of fruits",
            r"^(adequate|balanced|moderate|diverse)$",
        ),
        page_titles={8: "What are healthy diets?"},
    ),
    "who-five-keys": ParseHints(
        # 1-4: covers and blanks. 12: image divider. 14-22: image-only pages
        # (recorded in the report). 23-27: trainer evaluation forms and
        # answers. 29-30: blank and ISBN page.
        skip_pages=frozenset({1, 2, 3, 4, 12, 23, 24, 25, 26, 27, 29, 30}),
        # Section titles are letter-spaced decoration in this PDF.
        page_titles={
            5: "Introduction",
            8: "Section One: Background material",
            13: "Section Two: Five Keys to Safer Food",
            28: "Glossary",
        },
    ),
    "eatwell-guide": ParseHints(
        # 1 and 11: the Eatwell plate poster (label fragments only).
        # 12: links, copyright, and licence.
        skip_pages=frozenset({1, 11, 12}),
    ),
    "kitchen-companion": ParseHints(
        # 1-4: cover, blank, foreword, about FSIS. 5: contents. 8-9: image-only
        # pathogen table. 45-50: Ask Karen, contacts, index, legal, back cover.
        skip_pages=frozenset({1, 2, 3, 4, 5, 8, 9, 45, 46, 47, 48, 49, 50}),
    ),
}


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"})
_SOFT_HYPHEN = re.compile(r"\u00ad\s*")
_SPACES = re.compile(r"[ \t\r\f\v\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_TERMINAL = re.compile(r"[.!?:;…]['\"”’)\]]*$")
_URL_LINE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalise a fragment of extracted text to plain single-spaced prose."""
    text = unicodedata.normalize("NFC", text).translate(_LIGATURES)
    text = _SOFT_HYPHEN.sub("", text)
    text = "".join(
        ch for ch in text if ch in "\n " or unicodedata.category(ch) not in {"Cc", "Cf", "Co", "Cn", "Cs"}
    )
    text = _SPACES.sub(" ", text)
    return " ".join(part.strip() for part in text.split("\n") if part.strip())


def _join_pieces(previous: str, nxt: str) -> str:
    """Join two consecutive text pieces.

    A trailing hyphen followed by a lowercase continuation is joined without a
    space and the hyphen is kept: compound words (``industrially-produced``)
    are far more common in these documents than line-break hyphenation, and a
    kept hyphen still tokenises well while a wrongly merged word does not.
    """
    if not previous:
        return nxt
    if not nxt:
        return previous
    if previous.endswith("-") and nxt[:1].islower():
        return previous + nxt
    return previous + " " + nxt


def _alpha_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _is_letter_spaced(text: str) -> bool:
    """True for decorative text such as ``S e c t i o n  T w o``."""
    tokens = text.split()
    if len(tokens) < 5:
        return False
    return sum(len(token) == 1 for token in tokens) / len(tokens) >= 0.6


def _is_label_fragment(text: str) -> bool:
    """True for a stray label from a figure: a date stamp ("28 OCT 13"), a
    thermometer mark ("165ºF (73.8°C)") or a shouting sticker ("BEST IF USED BY")."""
    compact = text.replace(" ", "")
    if _alpha_count(text) < 0.5 * len(compact):
        return True
    words = text.split()
    return len(words) <= 4 and text.isupper() and not text.rstrip().endswith((".", "!", "?", ":"))


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


# --------------------------------------------------------------------------
# Tables (shared by PDF and HTML)
# --------------------------------------------------------------------------

# Data tables have short cells (a storage time, a temperature, a few
# sentences of tips). Layout grids that merely arrange page text into two
# columns have paragraph-length cells and are rejected by these limits.
TABLE_MAX_MEAN_CELL_CHARS = 300
TABLE_MAX_CELL_CHARS = 700
TABLE_MIN_FILL_RATIO = 0.5
GRID_MAX_LABEL_WORDS = 12  # row labels of a data table are short
GRID_MAX_LABEL_CHARS = 80


def format_table_row(header: Sequence[str], cells: Sequence[str]) -> str:
    """Render one data row with its column headers, as in architecture §4.2.

    ``Food: Poultry | Type: Chicken, whole | Refrigerator (...): 1 to 2 days``
    Empty cells are omitted.
    """
    parts = []
    for name, cell in zip(header, cells):
        cell = cell.strip()
        if not cell:
            continue
        parts.append(f"{name}: {cell}" if name else cell)
    return " | ".join(parts)


def format_table_markdown(header: Sequence[str], rows: Sequence[Sequence[str]], caption: str | None) -> str:
    def escape(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = []
    if caption:
        lines.append(caption)
    lines.append("| " + " | ".join(escape(cell) for cell in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _clean_grid(grid: Sequence[Sequence[object]]) -> list[list[str]]:
    rows: list[list[str]] = []
    width = max((len(row) for row in grid), default=0)
    for raw in grid:
        cells = [normalize_text(str(cell)) if cell is not None else "" for cell in raw]
        cells += [""] * (width - len(cells))
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    # Drop columns that are empty in every row.
    keep = [index for index in range(width) if any(row[index] for row in rows)]
    return [[row[index] for index in keep] for row in rows]


def _composite_header(header_rows: Sequence[Sequence[str]]) -> list[str]:
    """Merge stacked header rows, carrying merged cells across their span."""
    width = len(header_rows[0])
    columns: list[list[str]] = [[] for _ in range(width)]
    for row in header_rows:
        carried = ""
        for index, cell in enumerate(row):
            if cell:
                carried = cell
            elif index == 0:
                carried = ""
            if carried and carried not in columns[index]:
                columns[index].append(carried)
    return [" – ".join(parts) for parts in columns]


def build_table(grid: Sequence[Sequence[object]], caption: str | None = None) -> tuple[str | None, list[str], list[list[str]]] | None:
    """Turn a raw cell grid into ``(caption, header, rows)`` or ``None`` if it is not a data table.

    Leading rows with a single filled cell become the caption. The first row
    with two or more filled cells is the header; following rows whose first
    cell is empty are stacked header rows. A later single-cell row is a group
    label that is prefixed onto the first cell of the rows beneath it.
    """
    rows = _clean_grid(grid)
    if len(rows) < 2 or len(rows[0]) < 2:
        return None

    captions: list[str] = [caption] if caption else []
    index = 0
    while index < len(rows) and sum(1 for cell in rows[index] if cell) == 1:
        captions.append(next(cell for cell in rows[index] if cell))
        index += 1
    if index >= len(rows):
        return None

    header_rows = [rows[index]]
    index += 1
    while index < len(rows) and not rows[index][0] and any(rows[index]):
        header_rows.append(rows[index])
        index += 1
    header = _composite_header(header_rows)
    # A header whose cells are mostly numbers (e.g. the traffic-light label
    # graphic "3.0g | 1.3g | 34g") is an illustration, not a data table.
    if sum(1 for cell in header if _alpha_count(cell) >= 2) < math.ceil(len(header) / 2):
        return None

    data: list[list[str]] = []
    labels: list[str] = []
    group = ""
    previous_label = ""
    for row in rows[index:]:
        filled = [cell for cell in row if cell]
        if len(filled) == 1 and row[0]:
            group = row[0]
            continue
        row = list(row)
        if not row[0] and previous_label:
            # A label that visually spans two ruled rows ("Egg substitutes,
            # liquid – Unopened / Opened"): repeat it so the row stays usable.
            row[0] = previous_label
        elif row[0]:
            previous_label = row[0]
        labels.append(row[0])
        if group and row[0]:
            row[0] = f"{group} – {row[0]}"
        data.append(row)

    if not data:
        return None
    # Data tables label their rows briefly; a two-column page layout whose
    # first column holds paragraphs ("Core information | Why?") is prose.
    long_labels = sum(1 for label in labels if len(label.split()) > GRID_MAX_LABEL_WORDS or len(label) > GRID_MAX_LABEL_CHARS)
    if long_labels >= math.ceil(len(labels) / 2):
        return None
    cells = [cell for row in data for cell in row]
    filled_cells = [cell for cell in cells if cell]
    if len(filled_cells) / len(cells) < TABLE_MIN_FILL_RATIO:
        return None
    if not any(sum(1 for cell in row if cell) >= 2 for row in data):
        return None
    mean_len = sum(len(cell) for cell in filled_cells) / len(filled_cells)
    if mean_len > TABLE_MAX_MEAN_CELL_CHARS or max(len(cell) for cell in filled_cells) > TABLE_MAX_CELL_CHARS:
        return None
    return (" ".join(captions) or None, header, data)


def _table_block(
    grid: Sequence[Sequence[object]],
    heading_path: tuple[str, ...],
    page: int | None,
    caption: str | None = None,
) -> Block | None:
    built = build_table(grid, caption)
    if built is None:
        return None
    caption, header, rows = built
    return Block(
        type=BLOCK_TABLE,
        text=format_table_markdown(header, rows, caption),
        heading_path=heading_path,
        page=page,
        header=header,
        rows=rows,
        caption=caption,
    )


# --------------------------------------------------------------------------
# Heading path bookkeeping
# --------------------------------------------------------------------------


class _Outline:
    """Maintains the heading stack and the skip state for non-guidance sections."""

    def __init__(self, report: ParseReport) -> None:
        self.stack: list[tuple[int, str]] = []
        self.skip_level: int | None = None
        self.report = report

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self.stack)

    @property
    def skipping(self) -> bool:
        return self.skip_level is not None

    def push(self, level: int, text: str) -> None:
        level = max(1, min(MAX_HEADING_LEVEL, level))
        if self.skip_level is not None and level <= self.skip_level:
            self.skip_level = None
        self.stack = [(lvl, txt) for lvl, txt in self.stack if lvl < level]
        self.stack.append((level, text))
        if SKIP_SECTION_HEADING.match(text):
            self.skip_level = level
            self.report.skipped_sections.append(text)
        elif len(self.report.headings) < 400:
            self.report.headings.append(" > ".join(self.path))


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

HTML_CONTENT_ROOT_SELECTORS = (".govspeak", "article", "main", "[role=main]", "body")
HTML_STRIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form",
     "button", "iframe", "svg", "canvas", "input", "select", "textarea", "label", "dialog"}
)
HTML_STRIP_ROLES = frozenset({"navigation", "banner", "contentinfo", "complementary", "search", "dialog", "alert", "alertdialog"})
HTML_STRIP_CLASS = re.compile(
    r"cookie|banner|feedback|survey|breadcrumb|navigation|\bnav\b|menu|footer|sidebar|"
    r"share|social|skip-link|back-to-top|language|translation|related-|devolved|"
    r"contents-list|usa-alert|visually-hidden|sr-only|toolbar|site-header|page-header|"
    r"gem-c-feedback|govuk-cookie|usa-identifier|usa-banner|block-ofinterest|block-languageassistance",
    re.IGNORECASE,
)
HTML_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 3, "h5": 3, "h6": 3}
HTML_PARAGRAPH_TAGS = frozenset({"p", "blockquote", "pre", "dd", "dt", "figcaption", "summary", "address"})


def _html_is_boilerplate(tag: Tag) -> bool:
    if tag.name in HTML_STRIP_TAGS:
        return True
    role = tag.get("role")
    if isinstance(role, str) and role.lower() in HTML_STRIP_ROLES:
        return True
    if tag.get("hidden") is not None or tag.get("aria-hidden") == "true":
        return True
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    identity = " ".join([*classes, str(tag.get("id") or "")])
    return bool(identity) and HTML_STRIP_CLASS.search(identity) is not None


def _html_own_text(tag: Tag, exclude: Sequence[str] = ("ul", "ol", "table")) -> str:
    parts: list[str] = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name not in exclude and not _html_is_boilerplate(child):
            parts.append(child.get_text(" "))
    return normalize_text(" ".join(parts))


def _html_table_grid(table: Tag) -> list[list[str]]:
    """Read a ``<table>`` into a grid, carrying ``rowspan`` and ``colspan`` cells."""
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}
    for tr in table.find_all("tr"):
        cells = [cell for cell in tr.find_all(["th", "td"]) if cell.find_parent("tr") is tr]
        row: list[str] = []
        col = 0

        def fill_pending() -> None:
            nonlocal col
            while col in pending:
                text, remaining = pending[col]
                row.append(text)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col] = (text, remaining - 1)
                col += 1

        for cell in cells:
            fill_pending()
            text = normalize_text(cell.get_text(" "))
            try:
                rowspan = max(1, int(cell.get("rowspan") or 1))
            except ValueError:
                rowspan = 1
            try:
                colspan = max(1, int(cell.get("colspan") or 1))
            except ValueError:
                colspan = 1
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col] = (text, rowspan - 1)
                col += 1
        fill_pending()
        grid.append(row)
    return grid


class _HtmlWalker:
    def __init__(self, hints: ParseHints, report: ParseReport) -> None:
        self.blocks: list[Block] = []
        self.outline = _Outline(report)
        self.report = report
        self.drop = [*GLOBAL_DROP_PATTERNS, *(re.compile(p, re.IGNORECASE) for p in hints.drop_patterns)]

    def emit(self, block_type: str, text: str, **extra: object) -> None:
        if not text or self.outline.skipping or _matches_any(text, self.drop):
            return
        if block_type == BLOCK_PARAGRAPH and _URL_LINE.match(text):
            return
        self.blocks.append(Block(type=block_type, text=text, heading_path=self.outline.path, **extra))  # type: ignore[arg-type]

    def heading(self, level: int, text: str) -> None:
        if not text:
            return
        self.outline.push(level, text)
        if not self.outline.skipping:
            self.blocks.append(Block(type=BLOCK_HEADING, text=text, heading_path=self.outline.path, level=level))

    def walk(self, node: Tag) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            if _html_is_boilerplate(child):
                continue
            name = child.name.lower()
            if name in HTML_HEADING_TAGS:
                self.heading(HTML_HEADING_TAGS[name], normalize_text(child.get_text(" ")))
            elif name in HTML_PARAGRAPH_TAGS:
                self.emit(BLOCK_PARAGRAPH, _html_own_text(child))
                for nested in child.find_all(["ul", "ol", "table"], recursive=False):
                    self.walk_container(nested)
            elif name in ("ul", "ol", "table"):
                self.walk_container(child)
            else:
                self.walk(child)

    def walk_container(self, tag: Tag) -> None:
        name = tag.name.lower()
        if name == "table":
            caption_tag = tag.find("caption")
            caption = normalize_text(caption_tag.get_text(" ")) if caption_tag else None
            if self.outline.skipping:
                return
            block = _table_block(_html_table_grid(tag), self.outline.path, None, caption)
            if block is None:
                self.report.tables_rejected += 1
                self.walk(tag)
            else:
                self.report.tables_kept += 1
                self.blocks.append(block)
            return
        ordered = name == "ol"
        try:
            number = int(tag.get("start") or 1)
        except ValueError:
            number = 1
        for li in tag.find_all("li", recursive=False):
            text = _html_own_text(li)
            if ordered:
                text = f"{number}. {text}" if text else ""
                number += 1
            self.emit(BLOCK_LIST_ITEM, text)
            for nested in li.find_all(["ul", "ol", "table"], recursive=False):
                self.walk_container(nested)


def parse_html(html: str | bytes, document_id: str = "", hints: ParseHints | None = None, report: ParseReport | None = None) -> list[Block]:
    """Parse an HTML page into blocks."""
    hints = hints or DOCUMENT_HINTS.get(document_id, ParseHints())
    report = report or ParseReport(document_id=document_id, kind="html")
    soup = BeautifulSoup(html, "html.parser")
    root: Tag | None = None
    for selector in HTML_CONTENT_ROOT_SELECTORS:
        root = soup.select_one(selector)
        if root is not None:
            break
    if root is None:
        root = soup
    walker = _HtmlWalker(hints, report)
    # Anchor every block to the page title when the content root has no h1
    # of its own: prefer an h1 elsewhere on the page, then <title> minus
    # the site-name suffix ("Chilling | Food Standards Agency").
    if root.find("h1") is None and root.name != "h1":
        page_title = soup.find("h1")
        title_text = normalize_text(page_title.get_text(" ")) if page_title is not None else ""
        if not title_text and soup.title is not None and soup.title.string:
            title_text = normalize_text(re.split(r"\s+[|\-–—]\s+", soup.title.string, maxsplit=1)[0])
        if title_text:
            walker.heading(1, title_text)
    walker.walk(root)
    report.pages_total = report.pages_parsed = 1
    _finish_report(report, walker.blocks)
    return walker.blocks


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|ultra|demi|-bd\b|bd$", re.IGNORECASE)
_SYMBOL_FONT = re.compile(r"dingbat|wingding|symbol|bullet|webdings", re.IGNORECASE)
_BULLET_MARKER = re.compile(r"^[•■▪●○◦‣⁃∙·\-–—*»>✓✔❑❒]\s+(?=\S)")
# "1. Text", "(2) Text", "a) Text"; also "3.Text" where a tab stop left no space.
_NUMBER_MARKER = re.compile(r"^(\(?\d{1,2}[.)])(?:\s+(?=\S)|(?=[A-Z(]))|^(\(?[a-z][.)])\s+(?=\S)")
_LEADING_SYMBOL = re.compile(r"^[•■▪●○◦‣⁃∙·✓✔❑❒]+\s*")
_LONE_MARKER = re.compile(r"^[•■▪●○◦‣⁃∙·\-–—*»>✓✔❑❒]$")
_PAGE_NUMBER = re.compile(r"^(\d{1,4}|[ivxlc]{1,6}|page \d{1,4}(?: of \d{1,4})?)$", re.IGNORECASE)
_DROP_CAP = re.compile(r"^[A-Z]$")
# A heading ending in a preposition or article ("A closer look at") runs on
# into the next heading line.
_DANGLING_END = re.compile(r"\b(at|of|to|for|in|on|and|or|the|a|an|with|about|by|from|into)$", re.IGNORECASE)

HEADER_FOOTER_ZONE = 0.15
IMAGE_ONLY_MAX_CHARS = 20
TILT_TOLERANCE = 0.005  # |sin| of the text angle still treated as horizontal
IMAGE_LABEL_SIZE_DROP = 1.5  # points below body size that mark a figure label


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    block_no: int
    marker: str | None = None
    symbol_only: bool = False
    leading_space: bool = False
    # Visual rows and characters in the whole PyMuPDF block this line is in.
    block_rows: int = 1
    block_chars: int = 0
    # Side-by-side cells this visual row was merged from: (x0, x1, text).
    cells: list[tuple[float, float, str]] = field(default_factory=list)
    # Centre lies inside an illustration (not a full-page background).
    in_image: bool = False

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class _PageData:
    number: int
    width: float
    height: float
    lines: list[_Line]
    tables: list[tuple[float, Block]]  # (y0, table block with empty heading_path)


def _text_direction(page: pymupdf.Page) -> tuple[int, int]:
    """Dominant text direction weighted by characters: (1,0) upright, (0,±1) rotated."""
    weights: Counter[tuple[int, int]] = Counter()
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            chars = sum(len(span["text"].strip()) for span in line["spans"])
            if chars:
                dx, dy = line["dir"]
                weights[(int(round(dx)), int(round(dy)))] += chars
    if not weights:
        return (1, 0)
    return weights.most_common(1)[0][0]


def _upright_page(doc: pymupdf.Document, page: pymupdf.Page, direction: tuple[int, int]) -> pymupdf.Page:
    """Re-render a page whose text is drawn rotated so it reads left to right."""
    rotate = 90 if direction == (0, 1) else -90
    scratch = pymupdf.open()
    upright = scratch.new_page(width=page.rect.height, height=page.rect.width)
    upright.show_pdf_page(upright.rect, doc, page.number, rotate=rotate)
    return upright  # ``upright.parent`` keeps ``scratch`` alive


def _collect_page(doc: pymupdf.Document, page: pymupdf.Page, number: int, report: ParseReport) -> _PageData | None:
    text_flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES
    plain = page.get_text("text").strip()
    if len(plain) < IMAGE_ONLY_MAX_CHARS:
        if page.get_images():
            report.image_only_pages.append(number)
        else:
            report.blank_pages.append(number)
        return None

    direction = _text_direction(page)
    if direction[0] == 0:
        page = _upright_page(doc, page, direction)
        report.rotated_pages.append(number)

    tables: list[tuple[float, Block]] = []
    table_rects: list[pymupdf.Rect] = []
    try:
        found = page.find_tables(strategy="lines")
    except Exception as exc:  # pragma: no cover - defensive, PyMuPDF internals
        logger.warning("page %d: table finder failed: %s", number, exc)
        found = None
    for table in (found.tables if found else []):
        block = _table_block(_ruled_table_grid(page, table), (), number)
        if block is None:
            report.tables_rejected += 1
            continue
        report.tables_kept += 1
        rect = pymupdf.Rect(table.bbox)
        table_rects.append(rect)
        tables.append((rect.y0, block))

    text_dict = page.get_text("dict", flags=text_flags)
    # Labels drawn over illustrations (a poster's food names) are not prose.
    # Full-page background images are ignored so they do not swallow text.
    page_area = page.rect.get_area() or 1.0
    image_rects = [
        rect for rect in (pymupdf.Rect(info["bbox"]) for info in page.get_image_info())
        if 0 < rect.get_area() < 0.5 * page_area
    ]

    lines: list[_Line] = []
    for block_no, block in enumerate(text_dict["blocks"]):
        if block.get("type") != 0:
            continue
        # Text that is not horizontal (labels curved around a poster's plate,
        # vertical chapter tabs) is decoration; the page itself is already upright.
        upright_lines = [line for line in block["lines"] if abs(line.get("dir", (1, 0))[1]) <= TILT_TOLERANCE]
        if not upright_lines:
            continue
        rows = _visual_rows(upright_lines)
        block_chars = sum(len(span["text"].strip()) for line in upright_lines for span in line["spans"])
        for row in rows:
            if not "".join(span["text"] for line in row for span in line["spans"]).strip():
                continue
            bbox = pymupdf.Rect(row[0]["bbox"])
            for line in row[1:]:
                bbox |= pymupdf.Rect(line["bbox"])
            centre = pymupdf.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
            if any(rect.contains(centre) for rect in table_rects):
                continue
            parsed = _line_from_row(row, bbox, number, block_no)
            parsed.block_rows = len(rows)
            parsed.block_chars = block_chars
            parsed.in_image = any(rect.contains(centre) for rect in image_rects)
            if len(parsed.cells) == 1 and _TAB_STOP_GAP.search(
                "".join(span["text"] for line in row for span in line["spans"])
            ):
                parsed.cells = _split_tab_stop_cells(page, bbox, parsed.size) or parsed.cells
            lines.append(parsed)
    return _PageData(number=number, width=page.rect.width, height=page.rect.height, lines=lines, tables=tables)


_TAB_STOP_GAP = re.compile(r"\S[ \t]{3,}\S")


def _split_tab_stop_cells(page: pymupdf.Page, bbox: pymupdf.Rect, size: float) -> list[tuple[float, float, str]]:
    """Split a single span that positions columns with runs of spaces ("Food      °F (°C)").

    Word boxes give the real x positions; a horizontal gap wider than about two
    characters starts a new cell.
    """
    words = sorted(page.get_text("words", clip=bbox), key=lambda word: word[0])
    if len(words) < 2:
        return []
    gap_limit = max(2.0 * size, 6.0)
    cells: list[list[float | str]] = []
    for x0, _y0, x1, _y1, text, *_ in words:
        if cells and x0 - float(cells[-1][1]) <= gap_limit:
            cells[-1][1] = max(float(cells[-1][1]), x1)
            cells[-1][2] = f"{cells[-1][2]} {text}"
        else:
            cells.append([x0, x1, text])
    if len(cells) < 2:
        return []
    return [(float(x0), float(x1), normalize_text(str(text))) for x0, x1, text in cells]


_STACKED_FRACTION = re.compile(r"(?<=\d)([1-3])/([2-4])(?=\D|$)")
_STRAY_GLYPHS = re.compile(r"(?:(?<=\s)|^)(?:[A-Za-z]\s+){2,}(?=[A-Z0-9])")


def _clean_cell_text(text: str) -> str:
    """Normalise a table cell and undo two typesetting artefacts.

    Stacked fractions arrive glued to the whole number ("21/4 hours"); a
    space restores "2 1/4". Rows with a hidden text layer leak isolated
    letters ("F f l f Country") ahead of the real content.
    """
    text = normalize_text(text)
    text = _STACKED_FRACTION.sub(r" \1/\2", text)
    return _STRAY_GLYPHS.sub("", text).strip()


def _ruled_table_grid(page: pymupdf.Page, table: pymupdf.table.Table) -> list[list[str]]:
    """Read cell text through the normal text extractor, cell by cell.

    ``Table.extract()`` orders characters itself and garbles ligatures
    ("Stufefd" for "Stuffed"); clipping the standard extractor to each cell
    bbox keeps the words intact. Merged cells (``None``) become empty strings.
    """
    grid: list[list[str]] = []
    try:
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                if cell is None:
                    cells.append("")
                    continue
                rect = pymupdf.Rect(cell)
                cells.append(_clean_cell_text(page.get_text("text", clip=rect)) if not rect.is_empty else "")
            grid.append(cells)
    except Exception as exc:  # pragma: no cover - defensive, PyMuPDF internals
        logger.debug("cell-wise table extraction failed, falling back: %s", exc)
        return [[("" if cell is None else str(cell)) for cell in row] for row in table.extract()]
    return grid


def _visual_rows(lines: list[dict]) -> list[list[dict]]:
    """Group a block's lines that share a baseline (side-by-side cells) into rows.

    PyMuPDF emits each cell of an unruled table row, or each column header on
    a two-column page, as its own line inside one block. Reading them as one
    row keeps ``Protein | 10-15% | (6)`` together instead of three fragments.
    """
    rows: list[list[dict]] = []
    for line in lines:
        rect = pymupdf.Rect(line["bbox"])
        if rows:
            last = pymupdf.Rect(rows[-1][-1]["bbox"])
            overlap = min(rect.y1, last.y1) - max(rect.y0, last.y0)
            if overlap > 0.5 * min(rect.height, last.height) and rect.x0 >= last.x0:
                rows[-1].append(line)
                continue
        rows.append([line])
    for row in rows:
        row.sort(key=lambda item: item["bbox"][0])
    return rows


def _line_from_row(row: list[dict], bbox: pymupdf.Rect, page: int, block_no: int) -> _Line:
    spans = [span for line in row for span in line["spans"]]
    marker: str | None = None
    symbol_only = False
    body_spans = spans
    first = next(span for span in spans if span["text"].strip())
    if _SYMBOL_FONT.search(first["font"]) and len(first["text"].strip()) <= 2:
        marker = "•"
        body_spans = spans[spans.index(first) + 1:]
        if not "".join(span["text"] for span in body_spans).strip():
            symbol_only = True
    # Spans inside one line are contiguous text; separate lines of the row
    # are separate cells or columns and need a space between them.
    skipped = set(map(id, spans)) - set(map(id, body_spans))
    cells: list[tuple[float, float, str]] = []
    for line in row:
        cell_text = normalize_text("".join(span["text"] for span in line["spans"] if id(span) not in skipped))
        if cell_text:
            cells.append((line["bbox"][0], line["bbox"][2], cell_text))
    raw = " ".join(
        "".join(span["text"] for span in line["spans"] if id(span) not in skipped) for line in row
    )
    leading_space = raw[:1].isspace() and marker is None
    text = normalize_text(raw)
    if marker is None and text:
        match = _BULLET_MARKER.match(text)
        if match:
            marker = "•"
            text = text[match.end():]
        elif _LONE_MARKER.match(text):
            # A dash or bullet set as its own line; the item text follows.
            marker = "•"
            text = ""
        else:
            match = _NUMBER_MARKER.match(text)
            if match:
                marker = match.group(1) or match.group(2)
                text = text[match.end():]
    elif marker is not None:
        text = _LEADING_SYMBOL.sub("", text)
    if not text and marker is not None:
        symbol_only = True

    chars = 0
    bold_chars = 0
    size_weights: Counter[float] = Counter()
    for span in body_spans or spans:
        count = len(span["text"].strip())
        chars += count
        size_weights[round(span["size"] * 2) / 2] += count
        if span["flags"] & 16 or _BOLD_FONT.search(span["font"]):
            bold_chars += count
    size = size_weights.most_common(1)[0][0] if size_weights else round(first["size"] * 2) / 2
    bold = chars > 0 and bold_chars / chars >= 0.75
    return _Line(
        text=text, size=size, bold=bold,
        x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1,
        page=page, block_no=block_no,
        marker=marker, symbol_only=symbol_only, leading_space=leading_space,
        cells=cells,
    )


def _body_size(pages: Sequence[_PageData]) -> float:
    weights: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            weights[line.size] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _remove_noise(pages: Sequence[_PageData], body: float) -> None:
    """Drop page numbers, running headers and footers, decorations, and image labels.

    Running text is recognised by repetition: the same words, in the same
    size, at about the same height on three or more pages. Keying on size
    and position keeps a real heading ("1. Principles of healthy diets", 13pt
    at the top of one page) apart from the 8pt running header that repeats
    its words on the pages that follow, and catches a chapter tab set in a
    large face on every page of the chapter.
    """
    def key_of(line: _Line) -> tuple[str, float, int]:
        return (re.sub(r"[\d\W_]+", "", line.text.lower()), line.size, int(line.y0 // 24))

    counts: Counter[tuple[str, float, int]] = Counter()
    for page in pages:
        seen: set[tuple[str, float, int]] = set()
        for line in page.lines:
            key = key_of(line)
            if key[0] and len(line.text.split()) <= 12 and key not in seen:
                seen.add(key)
                counts[key] += 1
    running = {key for key, count in counts.items() if count >= 3} if len(pages) >= 3 else set()

    for page in pages:
        kept: list[_Line] = []
        for line in page.lines:
            in_zone = _in_zone(line, page.height)
            if in_zone and (_PAGE_NUMBER.match(line.text) or _URL_LINE.match(line.text)):
                continue
            if key_of(line) in running:
                continue
            if _is_letter_spaced(line.text):
                continue
            if (
                line.in_image and len(line.text.split()) <= 4 and line.block_rows <= 3
                and line.size <= body - IMAGE_LABEL_SIZE_DROP
            ):
                continue  # a caption inside an illustration, not sidebar prose on a backdrop
            kept.append(line)
        page.lines = kept


def _in_zone(line: _Line, page_height: float) -> bool:
    return line.y0 < page_height * HEADER_FOOTER_ZONE or line.y1 > page_height * (1 - HEADER_FOOTER_ZONE)


# Unruled tables: a header row of short bold cells defines the columns, and
# blocks in the first column define the rows.
GRID_COLUMN_TOLERANCE_LEFT = 25.0  # column headers are often centred over their cells
GRID_COLUMN_TOLERANCE_RIGHT = 40.0
GRID_MIN_ROWS = 3


def _detect_unruled_tables(page: _PageData, body: float, report: ParseReport) -> None:
    """Rebuild tables drawn without rules from the geometry of their cells.

    A booklet's summary grid ("Food group | What's included? | How much? |
    Tips") or a temperature chart ("Food | °F (°C)") has no ruling lines, so
    the table finder misses it and its cells arrive as scattered blocks. When
    a short, emphasised row with two or more cells is followed by blocks
    aligned under those cells, the blocks are gathered into a grid. The grid
    replaces the prose only when it validates as a data table.
    """
    index = 0
    while index < len(page.lines):
        header = page.lines[index]
        if _is_grid_header(header, body):
            result = _gather_grid(page, index, body)
            if result is not None:
                consumed, block = result
                report.tables_kept += 1
                page.tables.append((header.y0, block))
                page.lines = [line for line in page.lines if id(line) not in consumed]
                continue
        index += 1


def _is_grid_header(line: _Line, body: float) -> bool:
    if len(line.cells) < 2 or line.marker is not None:
        return False
    if not (line.bold or line.size >= body + 1.0):
        return False
    return all(len(text.split()) <= 6 and len(text) <= 40 for _, _, text in line.cells)


def _gather_grid(page: _PageData, header_index: int, body: float) -> tuple[set[int], Block] | None:
    header = page.lines[header_index]
    starts = [x0 for x0, _, _ in header.cells]
    ncols = len(starts)

    def column_of(x0: float) -> int | None:
        for col in range(ncols - 1, -1, -1):
            if starts[col] - GRID_COLUMN_TOLERANCE_LEFT <= x0 <= starts[col] + GRID_COLUMN_TOLERANCE_RIGHT:
                return col
        return None

    def spans_columns(col: int, x1: float) -> bool:
        return col + 1 < ncols and x1 > starts[col + 1] + GRID_COLUMN_TOLERANCE_LEFT

    # Candidate cells below the header, in geometric order.
    below = [line for line in page.lines[header_index + 1:] if line.y0 >= header.y0 and not line.symbol_only]
    below.sort(key=lambda line: (line.y0, line.x0))

    placed: list[tuple[_Line, int, str]] = []  # (line, column, text)
    header_extra: list[list[str]] = [[] for _ in range(ncols)]
    header_extra_lines: list[_Line] = []
    row_started = False
    for line in below:
        cells = line.cells or [(line.x0, line.x1, line.text)]
        columns = [column_of(x0) for x0, _, _ in cells]
        if any(col is None for col in columns):
            if all(col is None for col in columns):
                continue  # elsewhere on the page (a sidebar); not part of the grid
            break
        if line.size >= body + 1.0 and not row_started:
            break
        wide = any(spans_columns(col, x1) for col, (_, x1, _) in zip(columns, cells))
        if wide and row_started:
            break
        if line.size >= body + 1.0:
            break
        if not row_started and columns[0] != 0:
            for col, (_, _, text) in zip(columns, cells):
                header_extra[col].append(text)
            header_extra_lines.append(line)
            continue
        row_started = True
        for col, (_, _, text) in zip(columns, cells):
            placed.append((line, col, text))

    if not placed:
        return None

    # Rows start at each first-column label. A label is a new block or a line
    # separated from the previous label by a visible gap (labels of one column
    # are often emitted as a single block); other cells join the row whose
    # vertical range contains them.
    row_starts: list[float] = []
    previous: _Line | None = None
    for line, col, _ in placed:
        if col != 0:
            continue
        if previous is None or line.block_no != previous.block_no or line.y0 - previous.y1 > 0.5 * (line.y1 - line.y0):
            row_starts.append(line.y0)
        previous = line
    if len(row_starts) < GRID_MIN_ROWS:
        return None
    grid: list[list[list[str]]] = [[[] for _ in range(ncols)] for _ in row_starts]
    for line, col, text in placed:
        centre = (line.y0 + line.y1) / 2
        row = max(0, sum(1 for y0 in row_starts if y0 - 2.0 <= centre) - 1)
        grid[row][col].append(text)

    rows = [[" ".join(cell) for cell in row] for row in grid]
    if sum(1 for row in rows if sum(1 for cell in row if cell) >= 2) < max(GRID_MIN_ROWS, math.ceil(0.5 * len(rows))):
        return None
    header_cells = [
        " – ".join([text, *extra]) if extra else text
        for (_, _, text), extra in zip(header.cells, header_extra)
    ]
    block = _table_block([header_cells, *rows], (), page.number)
    if block is None:
        return None
    consumed = {id(header)} | {id(line) for line, _, _ in placed} | {id(line) for line in header_extra_lines}
    return consumed, block


def _is_heading_line(line: _Line, body: float) -> bool:
    text = line.text
    if not text or line.symbol_only or _alpha_count(text) < 2:
        return False
    if _alpha_count(text) < 0.5 * len(text.replace(" ", "")):
        return False  # a date stamp or a figure, not a title
    if len(text) > 120 or len(text.split()) > 16:
        return False
    if text.endswith((".", ";", ",")) and not re.search(r"\b\d\.$", text):
        return False
    if line.size >= body + 1.0:
        return line.marker != "•"
    # Body-size bold text is a heading only when it is short and stands on
    # its own: documents also set recommendations and pull quotes in bold.
    return (
        line.bold
        and line.marker is None
        and line.size >= body - 0.6
        and line.block_rows <= 2
        and line.block_chars <= 130
    )


def _heading_levels(pages: Sequence[_PageData], body: float, offset: int) -> dict[float, int]:
    sizes = sorted({line.size for page in pages for line in page.lines if _is_heading_line(line, body)}, reverse=True)
    return {size: min(MAX_HEADING_LEVEL, offset + index + 1) for index, size in enumerate(sizes)}


def _outline_titles(doc: pymupdf.Document, skip_pages: frozenset[int]) -> dict[int, str]:
    """Top-level PDF bookmarks by 1-indexed page, when the outline points at real pages."""
    titles: dict[int, str] = {}
    for level, title, page in doc.get_toc(simple=True):
        if level != 1 or page < 1 or page > doc.page_count or page in skip_pages:
            continue
        title = normalize_text(re.sub(r"[\s.]*\d+\s*$", "", title))
        if title and page not in titles:
            titles[page] = title
    if len(set(titles)) < 3:
        return {}
    return titles


class _PdfAssembler:
    """Turns the ordered line/table stream into blocks."""

    def __init__(self, body: float, levels: dict[float, int], outline_titles: dict[int, str],
                 hints: ParseHints, report: ParseReport) -> None:
        self.body = body
        self.levels = levels
        self.outline_titles = outline_titles
        self.report = report
        self.drop = [*GLOBAL_DROP_PATTERNS, *(re.compile(p, re.IGNORECASE) for p in hints.drop_patterns)]
        self.outline = _Outline(report)
        self.blocks: list[Block] = []
        self.current: Block | None = None
        self.current_line: _Line | None = None
        self.current_size: float = 0.0
        self.item_x0: float = 0.0
        self.pending_prefix: str = ""
        self.pending_marker = False

    # -- emission -----------------------------------------------------------

    def flush(self) -> None:
        block, line = self.current, self.current_line
        self.current = self.current_line = None
        if block is None or line is None:
            return
        text = normalize_text(block.text)
        if block.type == BLOCK_HEADING:
            if not text or _matches_any(text, self.drop):
                return
            if text.casefold() in {title.casefold() for title in self.outline.path}:
                return  # a chapter tab repeating the chapter title ("storing food")
            level = self.levels.get(self.current_size, MAX_HEADING_LEVEL)
            self.outline.push(level, text)
            if not self.outline.skipping:
                self.blocks.append(Block(BLOCK_HEADING, text, self.outline.path, page=block.page, level=level))
            return
        if self.outline.skipping or not text:
            return
        if _alpha_count(text) < 3 or len(text.split()) < 2 and block.type == BLOCK_PARAGRAPH:
            return
        if block.type == BLOCK_PARAGRAPH and _is_label_fragment(text):
            return
        if _URL_LINE.match(text) or _matches_any(text, self.drop):
            return
        block.text = text
        block.heading_path = self.outline.path
        self.blocks.append(block)

    def add_table(self, block: Block) -> None:
        self.flush()
        self.pending_prefix = ""
        if self.outline.skipping:
            return
        block.heading_path = self.outline.path
        self._inherit_header(block)
        self.blocks.append(block)

    def _inherit_header(self, block: Block) -> None:
        """Give a table that continues onto the next page its parent's header.

        A roasting chart split over two pages does not repeat "Meat | Oven
        Temperature | ..." on the second page, so the finder promotes the
        first data row to a header. When the previous table ended on the
        preceding page, has the same width and this "header" is mostly
        measurements, reuse the earlier header and keep the row as data.
        """
        previous = next((b for b in reversed(self.blocks) if b.type == BLOCK_TABLE), None)
        if previous is None or previous.page is None or block.page is None or block.header is None:
            return
        if block.page != previous.page + 1 or previous.header is None or len(previous.header) != len(block.header):
            return
        numeric = sum(1 for cell in block.header if re.search(r"\d", cell))
        if numeric < math.ceil(len(block.header) / 2):
            return
        rows = [list(block.header), *(block.rows or [])]
        if block.caption:
            # The "caption" was really the group label of the leading rows
            # ("CHICKEN, FRESH"); later groups are already prefixed.
            for row in rows:
                if " – " in row[0]:
                    break
                if row[0]:
                    row[0] = f"{block.caption} – {row[0]}"
        block.header = list(previous.header)
        block.rows = rows
        block.caption = None
        block.text = format_table_markdown(block.header, block.rows, None)

    def start(self, block_type: str, line: _Line, text: str) -> None:
        self.flush()
        self.current = Block(block_type, text, (), page=line.page)
        self.current_line = line
        self.current_size = line.size
        self.item_x0 = line.x0

    def append(self, line: _Line, text: str) -> None:
        assert self.current is not None
        self.current.text = _join_pieces(self.current.text, text)
        self.current_line = line

    # -- classification -------------------------------------------------------

    def start_page(self, number: int) -> None:
        title = self.outline_titles.get(number)
        if title:
            self.flush()
            self.outline.push(1, title)
            if not self.outline.skipping:
                self.blocks.append(Block(BLOCK_HEADING, title, self.outline.path, page=number, level=1))

    def add_line(self, line: _Line) -> None:
        if line.symbol_only:
            self.pending_marker = True
            return
        if _DROP_CAP.match(line.text) and line.marker is None:
            self.flush()
            self.pending_prefix = line.text
            return
        if self.pending_marker:
            line.marker = line.marker or "•"
            self.pending_marker = False

        text = line.text
        if self.pending_prefix:
            text = self.pending_prefix + text
            self.pending_prefix = ""

        # Later lines of a paragraph or list-item block continue it even when
        # set in bold: that is inline emphasis, not a heading.
        if (
            self.current is not None
            and self.current_line is not None
            and self.current.type in (BLOCK_PARAGRAPH, BLOCK_LIST_ITEM)
            and line.block_no == self.current_line.block_no
            and line.page == self.current_line.page
            and line.marker is None
        ):
            self.append(line, text)
            return

        if _is_heading_line(line, self.body):
            if len(line.cells) >= 2 and line.cells[1][0] - line.cells[0][1] > 4.0 * line.size and text == line.text:
                # A heading sharing its baseline with a sidebar's column label
                # ("What is the problem?   Additional information"): keep the heading.
                text = line.cells[0][2]
            if line.marker is not None and line.marker != "•":
                text = f"{line.marker} {text}"
            if self.current is not None and self.current.type == BLOCK_HEADING and self._continues_heading(line):
                self.append(line, text)
            else:
                self.start(BLOCK_HEADING, line, text)
            return

        if line.marker is not None:
            item_text = text if line.marker == "•" else f"{line.marker} {text}"
            self.start(BLOCK_LIST_ITEM, line, item_text)
            return

        if self.current is not None and self.current_line is not None and self.current_line.page == line.page:
            if self.current.type == BLOCK_LIST_ITEM and self._continues_item(line):
                self.append(line, text)
                return
            if self.current.type == BLOCK_PARAGRAPH and self._continues_paragraph(line):
                self.append(line, text)
                return
        elif self.current is not None and self.current.type == BLOCK_PARAGRAPH and self._continues_across_pages(line):
            self.append(line, text)
            return
        self.start(BLOCK_PARAGRAPH, line, text)

    def _gap(self, line: _Line) -> float:
        assert self.current_line is not None
        return line.y0 - self.current_line.y1

    def _continues_heading(self, line: _Line) -> bool:
        assert self.current is not None and self.current_line is not None
        if line.page != self.current_line.page:
            return False
        same_block = line.block_no == self.current_line.block_no and line.size == self.current_size
        if same_block:
            return True
        if _DANGLING_END.search(self.current.text) and self._gap(line) <= 4.0 * max(line.size, self.current_size):
            # "A closer look at" + "Fruit and vegetables": rank by the group name.
            self.current_size = line.size
            return True
        same_level = self.levels.get(line.size, MAX_HEADING_LEVEL) == self.levels.get(self.current_size, MAX_HEADING_LEVEL)
        return same_level and self._gap(line) <= 1.2 * max(line.size, self.current_size) and not _TERMINAL.search(self.current.text)

    def _continues_item(self, line: _Line) -> bool:
        assert self.current_line is not None
        if self._gap(line) > 2.0 * line.height:
            return False
        if line.block_no == self.current_line.block_no:
            return True
        return line.x0 > self.item_x0 + 3 or (line.leading_space and line.x0 >= self.item_x0 - 1)

    def _continues_paragraph(self, line: _Line) -> bool:
        assert self.current is not None and self.current_line is not None
        if line.block_no == self.current_line.block_no and line.size == self.current_size:
            return True
        if self._gap(line) > 2.5 * line.height:
            return False
        return not _TERMINAL.search(self.current.text) and (
            line.text[:1].islower() or self.current.text.endswith("-")
        )

    def _continues_across_pages(self, line: _Line) -> bool:
        assert self.current is not None
        return not _TERMINAL.search(self.current.text) and (
            line.text[:1].islower() or self.current.text.endswith("-")
        ) and line.marker is None


def parse_pdf(path: Path | str, document_id: str = "", hints: ParseHints | None = None, report: ParseReport | None = None) -> list[Block]:
    """Parse a PDF into blocks using its text layer and ruled tables."""
    hints = hints or DOCUMENT_HINTS.get(document_id, ParseHints())
    report = report or ParseReport(document_id=document_id, kind="pdf")
    doc = pymupdf.open(str(path))
    report.pages_total = doc.page_count

    pages: list[_PageData] = []
    for index in range(doc.page_count):
        number = index + 1
        if number in hints.skip_pages:
            report.pages_skipped.append(number)
            continue
        data = _collect_page(doc, doc[index], number, report)
        if data is not None:
            pages.append(data)
    report.pages_parsed = len(pages)

    body = _body_size(pages)
    _remove_noise(pages, body)
    for page in pages:
        _detect_unruled_tables(page, body, report)
    # Explicit page titles from the hints win over the PDF outline.
    outline_titles = {**_outline_titles(doc, hints.skip_pages), **hints.page_titles}
    levels = _heading_levels(pages, body, offset=1 if outline_titles else 0)

    assembler = _PdfAssembler(body, levels, outline_titles, hints, report)
    for page in pages:
        assembler.start_page(page.number)
        tables = sorted(page.tables, key=lambda item: item[0])
        table_index = 0
        for line in page.lines:
            while table_index < len(tables) and tables[table_index][0] <= line.y0:
                assembler.add_table(tables[table_index][1])
                table_index += 1
            assembler.add_line(line)
        while table_index < len(tables):
            assembler.add_table(tables[table_index][1])
            table_index += 1
    assembler.flush()
    doc.close()
    _finish_report(report, assembler.blocks)
    return assembler.blocks


# --------------------------------------------------------------------------
# Corpus entry points
# --------------------------------------------------------------------------


def _finish_report(report: ParseReport, blocks: Sequence[Block]) -> None:
    report.blocks_by_type = dict(Counter(block.type for block in blocks))


def raw_file_for(document: CorpusDocument, raw_dir: Path = RAW_DATA_DIR) -> Path | None:
    """Locate ``data/raw/<document_id>/source.<pdf|html>`` regardless of the registry kind."""
    folder = Path(raw_dir) / document.document_id
    for suffix in ("pdf", "html"):
        candidate = folder / f"source.{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_document(document: CorpusDocument, path: Path | str) -> ParsedDocument:
    """Parse one registry document from its raw file."""
    path = Path(path)
    hints = DOCUMENT_HINTS.get(document.document_id, ParseHints())
    kind = "pdf" if path.suffix.lower() == ".pdf" else "html"
    report = ParseReport(document_id=document.document_id, kind=kind)
    if kind == "pdf":
        blocks = parse_pdf(path, document.document_id, hints, report)
    else:
        blocks = parse_html(path.read_bytes(), document.document_id, hints, report)
    logger.info(
        "%s: %d blocks (%s), %d tables kept, %d rejected, pages parsed %d/%d",
        document.document_id, len(blocks),
        ", ".join(f"{k}={v}" for k, v in sorted(report.blocks_by_type.items())),
        report.tables_kept, report.tables_rejected, report.pages_parsed, report.pages_total,
    )
    if report.image_only_pages:
        logger.warning("%s: image-only pages without a text layer: %s", document.document_id, report.image_only_pages)
    return ParsedDocument(document.document_id, blocks, report)


def parse_corpus(documents: Sequence[CorpusDocument] = CORPUS, raw_dir: Path = RAW_DATA_DIR) -> dict[str, ParsedDocument]:
    """Parse every registry document that has a raw file. Missing files raise."""
    parsed: dict[str, ParsedDocument] = {}
    missing: list[str] = []
    for document in documents:
        path = raw_file_for(document, raw_dir)
        if path is None:
            missing.append(document.document_id)
            continue
        parsed[document.document_id] = parse_document(document, path)
    if missing:
        raise FileNotFoundError("No raw file for: " + ", ".join(missing) + ". Run scripts/fetch_corpus.py first.")
    return parsed


def write_parsed(parsed: dict[str, ParsedDocument], out_dir: Path = PARSED_DATA_DIR) -> tuple[Path, Path]:
    """Write ``blocks.jsonl`` and ``parse_report.json`` for inspection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = out_dir / "blocks.jsonl"
    report_path = out_dir / "parse_report.json"
    with blocks_path.open("w", encoding="utf-8") as handle:
        for document_id, item in parsed.items():
            for ordinal, block in enumerate(item.blocks):
                row = {"document_id": document_id, "ordinal": ordinal, **block.to_dict()}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(
        json.dumps({document_id: item.report.to_dict() for document_id, item in parsed.items()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return blocks_path, report_path


def iter_blocks(parsed: dict[str, ParsedDocument]) -> Iterator[tuple[str, Block]]:
    for document_id, item in parsed.items():
        for block in item.blocks:
            yield document_id, block
