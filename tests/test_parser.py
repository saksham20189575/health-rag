"""Offline tests for the corpus parser: an inline HTML page and a PDF generated with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from src.constants import CORPUS
from src.ingestion.parser import (
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    ParseReport,
    build_table,
    format_table_row,
    parse_corpus,
    parse_document,
    parse_html,
    parse_pdf,
    write_parsed,
)

DOCS = {document.document_id: document for document in CORPUS}

HTML_PAGE = """
<html><head><title>Cold Food Storage Chart | FoodSafety.gov</title></head>
<body>
<nav class="usa-nav"><a href="/">Home</a> <a href="/recalls">Recalls</a></nav>
<div class="usa-banner">An official website of the United States government</div>
<div class="cookie-banner">We use cookies. <button>Accept</button></div>
<main>
  <h1>Cold Food Storage Chart</h1>
  <p>Follow the guidelines below for storing food in the refrigerator and freezer.</p>
  <p>Looking for a specific item? Use the search box.</p>
  <h2>Storage times</h2>
  <table>
    <thead><tr><th>Food</th><th>Type</th><th>Refrigerator (40 °F or below)</th><th>Freezer (0 °F or below)</th></tr></thead>
    <tbody>
      <tr><td rowspan="2">Poultry</td><td>Chicken or turkey, whole</td><td>1 to 2 days</td><td>1 year</td></tr>
      <tr><td>Chicken or turkey, pieces</td><td>1 to 2 days</td><td>9 months</td></tr>
      <tr><td>Eggs</td><td>Raw eggs in shell</td><td>3 to 5 weeks</td><td>Do not freeze</td></tr>
    </tbody>
  </table>
  <h2>Tips</h2>
  <h3>Thawing</h3>
  <ul><li>Thaw in the refrigerator, not on the counter.</li><li>Cook thawed food promptly.</li></ul>
  <ol><li>Check the temperature.</li><li>Label leftovers with the date.</li></ol>
  <section class="feedback"><h2>Was this page helpful?</h2><p>Tell us what you think.</p></section>
</main>
<footer>Last reviewed 2024 · Privacy policy</footer>
</body></html>
"""


def _blocks_of(blocks, block_type):
    return [block for block in blocks if block.type == block_type]


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def test_html_strips_boilerplate_and_keeps_structure():
    blocks = parse_html(HTML_PAGE, "cold-food-storage")
    text = "\n".join(block.text for block in blocks)

    for noise in ("Home", "Recalls", "official website", "We use cookies", "Was this page helpful", "Privacy policy"):
        assert noise not in text
    assert "Looking for a specific item" not in text  # document hint

    headings = _blocks_of(blocks, BLOCK_HEADING)
    assert [(h.level, h.text) for h in headings] == [
        (1, "Cold Food Storage Chart"), (2, "Storage times"), (2, "Tips"), (3, "Thawing"),
    ]
    intro = _blocks_of(blocks, BLOCK_PARAGRAPH)[0]
    assert intro.heading_path == ("Cold Food Storage Chart",)

    items = _blocks_of(blocks, BLOCK_LIST_ITEM)
    assert [item.text for item in items] == [
        "Thaw in the refrigerator, not on the counter.",
        "Cook thawed food promptly.",
        "1. Check the temperature.",
        "2. Label leftovers with the date.",
    ]
    assert items[0].heading_path == ("Cold Food Storage Chart", "Tips", "Thawing")


def test_html_table_rows_carry_headers_and_rowspans():
    blocks = parse_html(HTML_PAGE, "cold-food-storage")
    (table,) = _blocks_of(blocks, BLOCK_TABLE)

    assert table.header == ["Food", "Type", "Refrigerator (40 °F or below)", "Freezer (0 °F or below)"]
    assert table.rows[0] == ["Poultry", "Chicken or turkey, whole", "1 to 2 days", "1 year"]
    assert table.rows[1][0] == "Poultry", "rowspan value must be repeated on the second row"
    assert table.heading_path == ("Cold Food Storage Chart", "Storage times")

    row = format_table_row(table.header, table.rows[1])
    assert row == (
        "Food: Poultry | Type: Chicken or turkey, pieces | "
        "Refrigerator (40 °F or below): 1 to 2 days | Freezer (0 °F or below): 9 months"
    )
    assert "| Poultry | Chicken or turkey, whole | 1 to 2 days | 1 year |" in table.text


def test_html_uses_document_title_when_body_has_no_h1():
    page = "<html><head><title>Chilling</title></head><body><main><p>Keep it cold.</p></main></body></html>"
    blocks = parse_html(page, "fsa-chill")
    assert blocks[0].type == BLOCK_HEADING and blocks[0].text == "Chilling"
    assert blocks[1].heading_path == ("Chilling",)


# --------------------------------------------------------------------------
# Table validation
# --------------------------------------------------------------------------


def test_build_table_handles_captions_stacked_headers_and_groups():
    grid = [
        ["Cold Storage Chart", "", ""],
        ["Product", "Refrigerator", "Freezer"],
        ["", "40 °F", "0 °F"],
        ["Hot dogs", "", ""],
        ["Opened package", "1 week", "1 to 2 months"],
        ["Unopened package", "2 weeks", "1 to 2 months"],
    ]
    caption, header, rows = build_table(grid)
    assert caption == "Cold Storage Chart"
    assert header == ["Product", "Refrigerator – 40 °F", "Freezer – 0 °F"]
    assert rows == [
        ["Hot dogs – Opened package", "1 week", "1 to 2 months"],
        ["Hot dogs – Unopened package", "2 weeks", "1 to 2 months"],
    ]


def test_build_table_rejects_page_layouts_and_graphics():
    prose = "Wash your hands before handling food and often during food preparation and after the toilet"
    layout = [["Core information", "Why?"], [prose, "Microorganisms are carried on hands."]]
    assert build_table(layout) is None

    graphic = [["3.0g", "1.3g", "34g"], ["LOW", "LOW", "HIGH"], ["4%", "7%", "38%"]]
    assert build_table(graphic) is None

    assert build_table([["only one row", "x"]]) is None


def test_format_table_row_skips_empty_cells():
    assert format_table_row(["Food", "Type", "Freezer"], ["Eggs", "", "Do not freeze"]) == "Food: Eggs | Freezer: Do not freeze"


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

BODY = 10.0
FOOTER = "Safe Food Handbook · Example Agency"


def _footer(page: pymupdf.Page, number: int) -> None:
    page.insert_text((72, 800), FOOTER, fontsize=8, fontname="helv")
    page.insert_text((300, 815), str(number), fontsize=8, fontname="helv")


def _lines(page: pymupdf.Page, x: float, y: float, lines: list[str], size: float = BODY, bold: bool = False) -> float:
    for line in lines:
        page.insert_text((x, y), line, fontsize=size, fontname="hebo" if bold else "helv")
        y += size * 1.3
    return y


def _draw_table(page: pymupdf.Page, x: float, y: float, rows: list[list[str]], widths: list[float], height: float = 20) -> None:
    for r, row in enumerate(rows):
        cx = x
        for cell, width in zip(row, widths):
            rect = pymupdf.Rect(cx, y + r * height, cx + width, y + (r + 1) * height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_text((cx + 4, y + r * height + 14), cell, fontsize=BODY, fontname="helv")
            cx += width


@pytest.fixture(scope="module")
def handbook_pdf(tmp_path_factory) -> Path:
    doc = pymupdf.open()

    page = doc.new_page()
    _lines(page, 72, 80, ["Safe Food Handbook"], size=20, bold=True)
    y = _lines(page, 72, 130, ["Storing Food"], size=14, bold=True)
    y = _lines(page, 72, y + 6, [
        "Perishable food should go into the refrigerator within two hours of purchase or",
        "cooking. Keep raw meat and poultry sealed on the bottom shelf so that their juices",
        "cannot drip onto ready-to-eat food.",
    ])
    y = _lines(page, 72, y + 6, [
        "• Keep the refrigerator at 40 °F or below.",
        "• Freeze leftovers you will not eat within 3 to 4 days.",
    ])
    _footer(page, 1)

    page = doc.new_page()
    y = _lines(page, 72, 80, ["Cooking"], size=14, bold=True)
    y = _lines(page, 72, y + 6, [
        "Use a food thermometer to check that food has reached a safe minimum internal",
        "temperature before serving it to anyone.",
    ])
    _lines(page, 72, y + 4, ["Safe Minimum Temperatures"], bold=True)
    _draw_table(page, 72, y + 12, [["Food", "°F"], ["Poultry", "165"], ["Ground meat", "160"], ["Leftovers", "165"]], [180, 100])
    _footer(page, 2)

    page = doc.new_page()
    y = _lines(page, 72, 80, ["References"], size=14, bold=True)
    _lines(page, 72, y + 6, ["1. World Health Organization. Food safety fact sheet. Geneva; 2020."])
    _footer(page, 3)

    path = tmp_path_factory.mktemp("pdf") / "handbook.pdf"
    doc.save(path)
    return path


def test_pdf_headings_paragraphs_lists_and_footers(handbook_pdf: Path):
    report = ParseReport(document_id="handbook", kind="pdf")
    blocks = parse_pdf(handbook_pdf, "handbook", report=report)
    text = "\n".join(block.text for block in blocks)

    assert report.pages_total == 3 and report.pages_parsed == 3
    assert report.rotated_pages == [] and report.image_only_pages == []
    assert FOOTER not in text, "running footer must be removed"
    assert not any(block.text in {"1", "2", "3"} for block in blocks), "page numbers must be removed"

    headings = [(h.level, h.text) for h in _blocks_of(blocks, BLOCK_HEADING)]
    assert headings[:3] == [(1, "Safe Food Handbook"), (2, "Storing Food"), (2, "Cooking")]

    paragraph = _blocks_of(blocks, BLOCK_PARAGRAPH)[0]
    assert paragraph.heading_path == ("Safe Food Handbook", "Storing Food")
    assert paragraph.text.startswith("Perishable food should go into the refrigerator")
    assert paragraph.text.endswith("ready-to-eat food.")
    assert paragraph.page == 1

    items = [item.text for item in _blocks_of(blocks, BLOCK_LIST_ITEM)]
    assert items == [
        "Keep the refrigerator at 40 °F or below.",
        "Freeze leftovers you will not eat within 3 to 4 days.",
    ]


def test_pdf_ruled_table_and_skipped_references(handbook_pdf: Path):
    report = ParseReport(document_id="handbook", kind="pdf")
    blocks = parse_pdf(handbook_pdf, "handbook", report=report)

    (table,) = _blocks_of(blocks, BLOCK_TABLE)
    assert table.header == ["Food", "°F"]
    assert ["Poultry", "165"] in table.rows and ["Ground meat", "160"] in table.rows
    assert table.heading_path == ("Safe Food Handbook", "Cooking", "Safe Minimum Temperatures")
    assert table.page == 2
    assert report.tables_kept == 1
    # Text inside the table must not be repeated as prose.
    assert not any(block.text in {"Poultry", "165", "Food"} for block in blocks if block.type != BLOCK_TABLE)

    assert report.skipped_sections == ["References"]
    assert "fact sheet" not in "\n".join(block.text for block in blocks)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def test_parse_corpus_and_write_parsed(tmp_path: Path, handbook_pdf: Path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "cold-food-storage").mkdir(parents=True)
    (raw_dir / "cold-food-storage" / "source.html").write_text(HTML_PAGE, encoding="utf-8")
    (raw_dir / "eatwell-guide").mkdir()
    (raw_dir / "eatwell-guide" / "source.pdf").write_bytes(handbook_pdf.read_bytes())

    documents = [DOCS["cold-food-storage"], DOCS["eatwell-guide"]]
    parsed = parse_corpus(documents, raw_dir=raw_dir)
    assert set(parsed) == {"cold-food-storage", "eatwell-guide"}
    assert parsed["cold-food-storage"].report.kind == "html"
    assert parsed["eatwell-guide"].report.kind == "pdf"

    blocks_path, report_path = write_parsed(parsed, tmp_path / "parsed")
    lines = blocks_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == sum(len(item.blocks) for item in parsed.values())
    assert '"document_id": "cold-food-storage"' in lines[0] and '"heading_path": [' in lines[0]
    assert '"eatwell-guide"' in report_path.read_text(encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="who-five-keys"):
        parse_corpus([DOCS["who-five-keys"]], raw_dir=raw_dir)


def test_parse_document_detects_kind_from_suffix(tmp_path: Path):
    path = tmp_path / "source.html"
    path.write_text(HTML_PAGE, encoding="utf-8")
    parsed = parse_document(DOCS["cold-food-storage"], path)
    assert parsed.report.kind == "html"
    assert parsed.report.blocks_by_type["table"] == 1
