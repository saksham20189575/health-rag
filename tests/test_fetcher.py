"""Offline tests for the corpus fetcher. HTTP is served by httpx.MockTransport."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import httpx
import pytest

from src.constants import CORPUS, CorpusDocument
from src.ingestion.fetcher import (
    BROWSER_HEADERS,
    CorpusFetchError,
    Fetcher,
    detect_kind,
    find_english_pdf_link,
)

FIXTURES = Path(__file__).parent / "fixtures"
LANDING_HTML = (FIXTURES / "who_landing_page.html").read_text(encoding="utf-8")
STORAGE_HTML = (FIXTURES / "cold_food_storage.html").read_text(encoding="utf-8")
DENIED_HTML = (FIXTURES / "akamai_denied.html").read_text(encoding="utf-8")
SAMPLE_PDF = (FIXTURES / "sample.pdf").read_bytes()

IRIS_PDF_URL = (
    "https://iris.who.int/server/api/core/bitstreams/"
    "dadab0b0-98e4-41a3-b432-e984d79f15a3/content"
)

DOCS = {document.document_id: document for document in CORPUS}
PDF_DOC = DOCS["eatwell-guide"]
HTML_DOC = DOCS["cold-food-storage"]
LANDING_DOC = DOCS["who-five-keys"]

Handler = Callable[[httpx.Request], httpx.Response]


def pdf_response(content: bytes = SAMPLE_PDF, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=content, headers={"content-type": "application/pdf"})


def html_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=text, headers={"content-type": "text/html; charset=utf-8"})


def route(table: dict[str, Handler | httpx.Response]) -> Handler:
    """Build a MockTransport handler from a URL -> response/handler table."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = str(request.url)
        if key not in table:
            return httpx.Response(404, text=f"unrouted {key}")
        entry = table[key]
        return entry(request) if callable(entry) else entry

    return handler


def make_fetcher(tmp_path: Path, handler: Handler, today: str = "2026-09-27") -> Fetcher:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers=BROWSER_HEADERS,
        follow_redirects=True,
    )
    return Fetcher(
        client=client,
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "corpus_manifest.json",
        today=lambda: today,
        attempts=2,
        backoff_seconds=0.0,
        sleep=lambda _seconds: None,
    )


def read_manifest(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["document_id"]: row for row in payload["documents"]}


# -- pure helpers -----------------------------------------------------------


def test_find_english_pdf_link_prefers_download_button_and_skips_other_languages():
    link = find_english_pdf_link(LANDING_HTML, LANDING_DOC.source_url)
    assert link == IRIS_PDF_URL


def test_find_english_pdf_link_returns_none_without_pdf_links():
    assert find_english_pdf_link(STORAGE_HTML, HTML_DOC.source_url) is None


def test_find_english_pdf_link_resolves_relative_hrefs():
    html = '<a href="/docs/report_eng.pdf">English</a><a href="/docs/report_fre.pdf">French</a>'
    assert find_english_pdf_link(html, "https://example.org/pubs/item") == (
        "https://example.org/docs/report_eng.pdf"
    )


@pytest.mark.parametrize(
    ("content", "content_type", "expected"),
    [
        (SAMPLE_PDF, "application/pdf", "pdf"),
        (SAMPLE_PDF, "application/octet-stream", "pdf"),
        (STORAGE_HTML.encode(), "text/html; charset=utf-8", "html"),
        (DENIED_HTML.encode(), None, "html"),
        (b"\x00\x01\x02", "application/octet-stream", None),
        # undecoded compressed body served with an HTML header
        (b"\x1b\x9a\xe6Q\x8b\xf0\x8c\xf1", "text/html; charset=utf-8", None),
        # header claims PDF but the body is not one
        (DENIED_HTML.encode(), "application/pdf", "html"),
    ],
)
def test_detect_kind(content: bytes, content_type: str | None, expected: str | None):
    assert detect_kind(content, content_type) == expected


# -- fetching ---------------------------------------------------------------


def test_pdf_document_is_saved_with_provenance(tmp_path: Path):
    fetcher = make_fetcher(tmp_path, route({PDF_DOC.source_url: pdf_response()}))

    [row] = fetcher.fetch_all([PDF_DOC])

    saved = tmp_path / "raw" / PDF_DOC.document_id / "source.pdf"
    assert saved.read_bytes() == SAMPLE_PDF
    assert row.status == "ok"
    assert row.http_status == 200
    assert row.sha256 == hashlib.sha256(SAMPLE_PDF).hexdigest()
    assert row.retrieval_date == "2026-09-27"
    assert row.publisher == PDF_DOC.publisher
    assert row.year == PDF_DOC.year
    assert row.source_url == PDF_DOC.source_url
    assert row.downloaded_url == PDF_DOC.source_url
    assert row.file == str(saved.resolve())

    manifest = read_manifest(tmp_path / "corpus_manifest.json")
    for key in ("publisher", "year", "source_url", "retrieval_date", "sha256", "http_status", "status"):
        assert manifest[PDF_DOC.document_id][key] == getattr(row, key)


def test_html_document_is_saved_as_source_html(tmp_path: Path):
    fetcher = make_fetcher(tmp_path, route({HTML_DOC.source_url: html_response(STORAGE_HTML)}))

    [row] = fetcher.fetch_all([HTML_DOC])

    saved = tmp_path / "raw" / HTML_DOC.document_id / "source.html"
    assert saved.read_text(encoding="utf-8") == STORAGE_HTML
    assert row.status == "ok"
    assert row.content_type.startswith("text/html")


def test_browser_headers_are_sent(tmp_path: Path):
    seen: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return pdf_response()

    make_fetcher(tmp_path, route({PDF_DOC.source_url: capture})).fetch_all([PDF_DOC])

    assert seen["user-agent"].startswith("Mozilla/5.0")
    assert seen["sec-fetch-mode"] == "navigate"


def test_redirect_is_followed_and_final_url_recorded(tmp_path: Path):
    final_url = "https://cdn.example.org/eatwell.pdf"
    table = {
        PDF_DOC.source_url: httpx.Response(302, headers={"location": final_url}),
        final_url: pdf_response(),
    }

    [row] = make_fetcher(tmp_path, route(table)).fetch_all([PDF_DOC])

    assert row.status == "ok"
    assert row.source_url == PDF_DOC.source_url
    assert row.downloaded_url == final_url


def test_landing_page_follows_english_pdf_and_records_both_urls(tmp_path: Path):
    requested: list[str] = []

    def landing(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return html_response(LANDING_HTML)

    def asset(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return pdf_response()

    table = {LANDING_DOC.source_url: landing, IRIS_PDF_URL: asset}
    [row] = make_fetcher(tmp_path, route(table)).fetch_all([LANDING_DOC])

    assert requested == [LANDING_DOC.source_url, IRIS_PDF_URL]
    assert row.status == "ok"
    assert row.source_url == LANDING_DOC.source_url
    assert row.downloaded_url == IRIS_PDF_URL
    saved = tmp_path / "raw" / LANDING_DOC.document_id / "source.pdf"
    assert saved.read_bytes() == SAMPLE_PDF
    assert not (saved.parent / "source.html").exists()


def test_landing_page_without_english_pdf_is_a_failure(tmp_path: Path):
    table = {LANDING_DOC.source_url: html_response(STORAGE_HTML)}
    fetcher = make_fetcher(tmp_path, route(table))

    with pytest.raises(CorpusFetchError) as excinfo:
        fetcher.fetch_all([LANDING_DOC], strict=True)

    assert excinfo.value.document_ids == [LANDING_DOC.document_id]
    row = read_manifest(tmp_path / "corpus_manifest.json")[LANDING_DOC.document_id]
    assert row["status"] == "stale"
    assert "English PDF" in row["error"]


def test_pdf_expected_but_html_received_is_a_failure(tmp_path: Path):
    """An HTML block page served with 200 must not be stored as the PDF."""
    table = {PDF_DOC.source_url: html_response(DENIED_HTML)}

    [row] = make_fetcher(tmp_path, route(table)).fetch_all([PDF_DOC])

    assert row.status == "stale"
    assert row.file is None
    assert "expected pdf" in row.error
    assert not (tmp_path / "raw" / PDF_DOC.document_id).exists()


def test_compressed_body_with_html_header_is_not_stored_as_html(tmp_path: Path):
    garbage = httpx.Response(
        200, content=b"\x1b\x9a\xe6Q\x8b\xf0\x8c\xf1\x00", headers={"content-type": "text/html"}
    )

    [row] = make_fetcher(tmp_path, route({HTML_DOC.source_url: garbage})).fetch_all([HTML_DOC])

    assert row.status == "stale"
    assert "unrecognised content" in row.error
    assert not (tmp_path / "raw" / HTML_DOC.document_id / "source.html").exists()


def test_accept_encoding_is_left_to_the_client(tmp_path: Path):
    """The fetcher must not advertise codings httpx cannot decode."""
    assert "Accept-Encoding" not in BROWSER_HEADERS


def test_403_is_retried_with_alternate_headers(tmp_path: Path):
    agents: list[str] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers["user-agent"])
        if len(agents) == 1:
            return html_response(DENIED_HTML, status=403)
        return pdf_response()

    [row] = make_fetcher(tmp_path, route({PDF_DOC.source_url: flaky})).fetch_all([PDF_DOC])

    assert row.status == "ok"
    assert len(agents) == 2
    assert agents[0] != agents[1]
    assert "Safari" in agents[1]


def test_first_run_failure_stops_with_named_document(tmp_path: Path):
    table = {
        PDF_DOC.source_url: pdf_response(),
        HTML_DOC.source_url: html_response(DENIED_HTML, status=403),
    }
    fetcher = make_fetcher(tmp_path, route(table))

    with pytest.raises(CorpusFetchError) as excinfo:
        fetcher.fetch_all([PDF_DOC, HTML_DOC], strict=True)

    assert excinfo.value.document_ids == [HTML_DOC.document_id]
    assert HTML_DOC.document_id in str(excinfo.value)
    manifest = read_manifest(tmp_path / "corpus_manifest.json")
    assert manifest[PDF_DOC.document_id]["status"] == "ok"
    failed = manifest[HTML_DOC.document_id]
    assert failed["status"] == "stale"
    assert failed["http_status"] == 403
    assert failed["sha256"] is None and failed["file"] is None
    # provenance that does not depend on the download is still recorded
    assert failed["publisher"] == HTML_DOC.publisher
    assert failed["year"] == HTML_DOC.year
    assert failed["source_url"] == HTML_DOC.source_url


def test_unchanged_hash_does_not_rewrite_file_or_retrieval_date(tmp_path: Path):
    handler = route({PDF_DOC.source_url: pdf_response()})
    first = make_fetcher(tmp_path, handler, today="2026-09-01")
    [first_row] = first.fetch_all([PDF_DOC])

    saved = tmp_path / "raw" / PDF_DOC.document_id / "source.pdf"
    old_mtime = 1_600_000_000
    os.utime(saved, (old_mtime, old_mtime))

    second = make_fetcher(tmp_path, handler, today="2026-09-27")
    [second_row] = second.fetch_all([PDF_DOC])

    assert saved.stat().st_mtime == old_mtime, "unchanged file was rewritten"
    assert second_row.status == "ok"
    assert second_row.sha256 == first_row.sha256
    assert second_row.retrieval_date == "2026-09-01"


def test_html_differing_only_in_script_token_is_not_rewritten(tmp_path: Path):
    """foodsafety.gov embeds a per-response analytics token in a <script>."""
    tokenised = STORAGE_HTML.replace(
        "</head>", '<script>var BOOMR_token="abc123";</script></head>'
    )
    first = make_fetcher(tmp_path, route({HTML_DOC.source_url: html_response(tokenised)}))
    [first_row] = first.fetch_all([HTML_DOC])

    saved = tmp_path / "raw" / HTML_DOC.document_id / "source.html"
    old_mtime = 1_600_000_000
    os.utime(saved, (old_mtime, old_mtime))

    retokenised = tokenised.replace("abc123", "zzz999")
    assert hashlib.sha256(retokenised.encode()).hexdigest() != first_row.sha256
    second = make_fetcher(tmp_path, route({HTML_DOC.source_url: html_response(retokenised)}))
    [row] = second.fetch_all([HTML_DOC])

    assert saved.stat().st_mtime == old_mtime
    assert row.status == "ok"
    assert row.sha256 == first_row.sha256, "sha256 still describes the bytes on disk"
    assert row.content_fingerprint == first_row.content_fingerprint


def test_html_body_change_is_detected_despite_fingerprint(tmp_path: Path):
    first = make_fetcher(tmp_path, route({HTML_DOC.source_url: html_response(STORAGE_HTML)}))
    first.fetch_all([HTML_DOC])

    changed = STORAGE_HTML.replace("1 to 2 days", "1 to 3 days")
    [row] = make_fetcher(tmp_path, route({HTML_DOC.source_url: html_response(changed)})).fetch_all([HTML_DOC])

    saved = tmp_path / "raw" / HTML_DOC.document_id / "source.html"
    assert saved.read_text(encoding="utf-8") == changed
    assert row.sha256 == hashlib.sha256(changed.encode()).hexdigest()


def test_changed_content_rewrites_file_and_updates_retrieval_date(tmp_path: Path):
    first = make_fetcher(tmp_path, route({PDF_DOC.source_url: pdf_response()}), today="2026-09-01")
    first.fetch_all([PDF_DOC])

    new_pdf = SAMPLE_PDF + b"\n% revised edition\n"
    second = make_fetcher(tmp_path, route({PDF_DOC.source_url: pdf_response(new_pdf)}), today="2026-09-27")
    [row] = second.fetch_all([PDF_DOC])

    saved = tmp_path / "raw" / PDF_DOC.document_id / "source.pdf"
    assert saved.read_bytes() == new_pdf
    assert row.sha256 == hashlib.sha256(new_pdf).hexdigest()
    assert row.retrieval_date == "2026-09-27"


def test_failed_download_keeps_previous_file_and_marks_stale(tmp_path: Path):
    first = make_fetcher(tmp_path, route({PDF_DOC.source_url: pdf_response()}), today="2026-09-01")
    [first_row] = first.fetch_all([PDF_DOC])

    failing = route({PDF_DOC.source_url: httpx.Response(500, text="upstream error")})
    second = make_fetcher(tmp_path, failing, today="2026-09-27")
    [row] = second.fetch_all([PDF_DOC])

    saved = tmp_path / "raw" / PDF_DOC.document_id / "source.pdf"
    assert saved.read_bytes() == SAMPLE_PDF
    assert row.status == "stale"
    assert row.http_status == 500
    assert row.sha256 == first_row.sha256
    assert row.retrieval_date == "2026-09-01"
    assert row.file == first_row.file
    assert "HTTP 500" in row.error

    manifest = read_manifest(tmp_path / "corpus_manifest.json")
    assert manifest[PDF_DOC.document_id]["status"] == "stale"


def test_transport_error_marks_row_stale(tmp_path: Path):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    [row] = make_fetcher(tmp_path, route({PDF_DOC.source_url: boom})).fetch_all([PDF_DOC])

    assert row.status == "stale"
    assert row.http_status is None
    assert "ConnectError" in row.error


def test_stale_row_recovers_on_next_successful_run(tmp_path: Path):
    make_fetcher(tmp_path, route({PDF_DOC.source_url: httpx.Response(503)})).fetch_all([PDF_DOC])
    [row] = make_fetcher(tmp_path, route({PDF_DOC.source_url: pdf_response()})).fetch_all([PDF_DOC])

    assert row.status == "ok"
    assert row.error is None
    assert (tmp_path / "raw" / PDF_DOC.document_id / "source.pdf").exists()


def test_full_registry_round_trip_writes_seven_ok_rows(tmp_path: Path):
    table: dict[str, Handler | httpx.Response] = {}
    for document in CORPUS:
        if document.landing_page:
            table[document.source_url] = html_response(LANDING_HTML)
        elif document.kind == "pdf":
            table[document.source_url] = pdf_response()
        else:
            table[document.source_url] = html_response(STORAGE_HTML)
    table[IRIS_PDF_URL] = pdf_response()

    rows = make_fetcher(tmp_path, route(table)).fetch_all(strict=True)

    assert [row.document_id for row in rows] == [document.document_id for document in CORPUS]
    assert all(row.status == "ok" for row in rows)
    manifest = read_manifest(tmp_path / "corpus_manifest.json")
    assert len(manifest) == 7
    for document in CORPUS:
        row = manifest[document.document_id]
        assert row["publisher"] == document.publisher
        assert row["year"] == document.year
        assert row["source_url"] == document.source_url
        assert row["retrieval_date"] == "2026-09-27"
        expected_suffix = "source.pdf" if (document.kind == "pdf" or document.landing_page) else "source.html"
        assert row["file"].endswith(f"{document.document_id}/{expected_suffix}")
