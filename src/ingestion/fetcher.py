"""Download the seven corpus documents and record provenance.

For each registry row the fetcher downloads the source with browser headers,
writes the bytes to ``data/raw/<document_id>/source.<pdf|html>``, and writes a
row to ``data/corpus_manifest.json`` with publisher, year, source URL, the URL
actually downloaded, the UTC retrieval date, a sha256, and the HTTP status.

Two registry rows (``fao-who-healthy-diets``, ``who-five-keys``) point at WHO
publication landing pages. When the fetched HTML is such a page, the fetcher
follows the English PDF link and stores that PDF instead. ``source_url`` stays
the registry (citation) URL; ``downloaded_url`` records the asset.

Re-runs skip a document whose bytes hash to the same value. A failed download
keeps the previous file and marks the row ``stale`` so a later index build can
refuse to run on a partial corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.constants import (
    CORPUS,
    CORPUS_MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    CorpusDocument,
)

logger = logging.getLogger(__name__)

# Several hosts (Akamai in front of foodsafety.gov and fsis.usda.gov) return
# 403 unless the request looks like a browser navigation. A bare User-Agent is
# not enough; the Sec-Fetch-* headers matter too.
#
# Accept-Encoding is deliberately absent: httpx advertises only the codings it
# can decode. Advertising ``br`` without a brotli decoder stores compressed
# bytes as the "HTML" source.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Second profile used on a retry after a 403.
ALTERNATE_BROWSER_HEADERS: dict[str, str] = {
    **BROWSER_HEADERS,
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
}

RETRYABLE_STATUSES = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
REQUEST_TIMEOUT_SECONDS = 60.0

STATUS_OK = "ok"
STATUS_STALE = "stale"

PDF_MAGIC = b"%PDF"

# Script, noscript, and style blocks carry per-response analytics tokens on
# some hosts (Akamai mPulse on foodsafety.gov). They are excluded from the
# change-detection fingerprint so an unchanged page is not re-saved.
_VOLATILE_HTML_BLOCK = re.compile(rb"<(script|noscript|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)

# Links that point at a PDF asset: a .pdf path, a DSpace bitstream "content"
# endpoint (iris.who.int), or a legacy IRIS bitstream handle.
_PDF_HREF = re.compile(
    r"\.pdf(?:[?#]|$)|/bitstreams?/[^/?#]+/content(?:[?#]|$)|/iris/bitstream/",
    re.IGNORECASE,
)
# WHO language suffixes on non-English editions, e.g. 9789241594639_jpn.pdf,
# 9789241594639-per.pdf. English editions carry no suffix or "_eng".
_NON_ENGLISH_SUFFIX = re.compile(
    r"[_-](ara|chi|zho|fre|fra|rus|spa|por|jpn|per|fas|ger|deu|ita|kor|vie|tha|hin|tur)"
    r"(?=[._?#-]|$)",
    re.IGNORECASE,
)


class CorpusFetchError(RuntimeError):
    """Raised when at least one document could not be fetched fresh."""

    def __init__(self, document_ids: Sequence[str]) -> None:
        self.document_ids = list(document_ids)
        super().__init__(
            "Corpus fetch did not complete for: " + ", ".join(self.document_ids)
        )


class FetchFailure(Exception):
    """A single document download failed. Carries the last HTTP status seen."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


@dataclass
class ManifestRow:
    """Provenance for one corpus document."""

    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    downloaded_url: str | None
    retrieval_date: str | None
    sha256: str | None
    http_status: int | None
    status: str
    file: str | None
    content_type: str | None = None
    # Hash used to decide whether the document changed. Equals ``sha256`` for
    # PDFs; for HTML it ignores script/noscript/style blocks.
    content_fingerprint: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ManifestRow":
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})  # type: ignore[arg-type]


@dataclass(frozen=True)
class DownloadedAsset:
    content: bytes
    final_url: str
    http_status: int
    content_type: str | None
    kind: str  # "pdf" or "html"


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def content_fingerprint(content: bytes, kind: str) -> str:
    """Hash for change detection. HTML is hashed without volatile blocks."""
    if kind == "html":
        return sha256_hex(_VOLATILE_HTML_BLOCK.sub(b"", content))
    return sha256_hex(content)


def detect_kind(content: bytes, content_type: str | None) -> str | None:
    """Return ``pdf``, ``html``, or ``None`` from the body, informed by the header.

    The body is authoritative: a PDF must start with ``%PDF`` and HTML must
    contain markup. The Content-Type header alone is not trusted, because a
    block page or an undecoded compressed body can arrive with a valid header.
    """
    head = content[:2048].lstrip()
    if head.startswith(PDF_MAGIC):
        return "pdf"
    lowered_head = head.lower()
    if b"<html" in lowered_head or b"<!doctype html" in lowered_head:
        return "html"
    if "html" in (content_type or "").lower() and lowered_head.startswith(b"<"):
        return "html"
    return None


def find_english_pdf_link(html: str, base_url: str) -> str | None:
    """Find the English PDF (or full-text) link on a publication landing page.

    Candidates are anchors whose href points at a PDF asset. Non-English
    editions, recognised by a WHO language suffix, are skipped. Among the rest,
    an anchor labelled "Download" or "English" wins; otherwise the first match.
    """
    soup = BeautifulSoup(html, "html.parser")
    best_href: str | None = None
    best_score = -1
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or not _PDF_HREF.search(href):
            continue
        path = href.split("?", 1)[0].split("#", 1)[0]
        if _NON_ENGLISH_SUFFIX.search(path):
            continue
        label = " ".join(
            part
            for part in (
                anchor.get_text(" ", strip=True),
                str(anchor.get("title") or ""),
                str(anchor.get("aria-label") or ""),
            )
            if part
        ).lower()
        score = 0
        if "download" in label:
            score += 2
        if "english" in label or "full text" in label:
            score += 1
        if score > best_score:
            best_href, best_score = href, score
    return urljoin(base_url, best_href) if best_href else None


def load_manifest(path: Path) -> dict[str, ManifestRow]:
    """Read a previous manifest. Missing or unreadable files yield an empty map."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable manifest %s: %s", path, exc)
        return {}
    rows = payload.get("documents", []) if isinstance(payload, dict) else payload
    manifest: dict[str, ManifestRow] = {}
    for row in rows:
        try:
            parsed = ManifestRow.from_dict(row)
        except TypeError as exc:
            logger.warning("Skipping malformed manifest row %r: %s", row, exc)
            continue
        manifest[parsed.document_id] = parsed
    return manifest


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _default_client() -> httpx.Client:
    common = dict(headers=BROWSER_HEADERS, follow_redirects=True, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        return httpx.Client(http2=True, **common)
    except ImportError:
        logger.warning("h2 is not installed; falling back to HTTP/1.1. Some hosts may return 403.")
        return httpx.Client(**common)


class Fetcher:
    """Download registry documents into ``raw_dir`` and maintain the manifest."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        raw_dir: Path = RAW_DATA_DIR,
        manifest_path: Path = CORPUS_MANIFEST_PATH,
        today: Callable[[], str] = utc_today,
        attempts: int = 3,
        backoff_seconds: float = 1.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self._today = today
        self._attempts = max(1, attempts)
        self._backoff = backoff_seconds
        self._sleep = sleep

    # -- public API ---------------------------------------------------------

    def fetch_all(
        self,
        documents: Sequence[CorpusDocument] = CORPUS,
        strict: bool = False,
    ) -> list[ManifestRow]:
        """Fetch every document, write the manifest, and return its rows.

        With ``strict=True`` a :class:`CorpusFetchError` naming every non-``ok``
        document is raised after the manifest has been written.
        """
        previous = load_manifest(self.manifest_path)
        rows: list[ManifestRow] = []
        try:
            for document in documents:
                rows.append(self.fetch_document(document, previous.get(document.document_id)))
        finally:
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None
        write_manifest(self.manifest_path, rows)
        failed = [row.document_id for row in rows if row.status != STATUS_OK]
        if failed:
            logger.error("Documents not fetched fresh: %s", ", ".join(failed))
            if strict:
                raise CorpusFetchError(failed)
        return rows

    def fetch_document(
        self, document: CorpusDocument, previous: ManifestRow | None = None
    ) -> ManifestRow:
        """Fetch one document and return its manifest row without writing the manifest."""
        try:
            asset = self._download(document)
        except FetchFailure as exc:
            return self._stale_row(document, previous, str(exc), exc.http_status)

        digest = sha256_hex(asset.content)
        fingerprint = content_fingerprint(asset.content, asset.kind)
        target = self.raw_dir / document.document_id / f"source.{asset.kind}"

        if (
            previous is not None
            and (previous.content_fingerprint or previous.sha256) == fingerprint
            and previous.file is not None
            and self._resolve(previous.file) == target.resolve()
            and target.exists()
        ):
            logger.info(
                "%s: unchanged (fingerprint %s), keeping existing file",
                document.document_id, fingerprint[:12],
            )
            return replace(
                previous,
                document_name=document.document_name,
                publisher=document.publisher,
                year=document.year,
                source_url=document.source_url,
                downloaded_url=asset.final_url,
                http_status=asset.http_status,
                content_type=asset.content_type,
                content_fingerprint=fingerprint,
                status=STATUS_OK,
                error=None,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        for sibling in target.parent.glob("source.*"):
            if sibling != target:
                sibling.unlink()
        target.write_bytes(asset.content)
        logger.info(
            "%s: saved %d bytes to %s (HTTP %d)",
            document.document_id, len(asset.content), target, asset.http_status,
        )
        return ManifestRow(
            document_id=document.document_id,
            document_name=document.document_name,
            publisher=document.publisher,
            year=document.year,
            source_url=document.source_url,
            downloaded_url=asset.final_url,
            retrieval_date=self._today(),
            sha256=digest,
            http_status=asset.http_status,
            status=STATUS_OK,
            file=self._display_path(target),
            content_type=asset.content_type,
            content_fingerprint=fingerprint,
        )

    # -- internals ----------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = _default_client()
        return self._client

    def _download(self, document: CorpusDocument) -> DownloadedAsset:
        response = self._get(document.source_url)
        content_type = response.headers.get("content-type")
        kind = detect_kind(response.content, content_type)

        if document.landing_page and kind == "html":
            pdf_url = find_english_pdf_link(response.text, str(response.url))
            if pdf_url is None:
                raise FetchFailure(
                    "landing page has no English PDF link", response.status_code
                )
            logger.info("%s: landing page, following %s", document.document_id, pdf_url)
            response = self._get(pdf_url)
            content_type = response.headers.get("content-type")
            kind = detect_kind(response.content, content_type)

        expected = "pdf" if document.kind == "pdf" or document.landing_page else "html"
        if kind != expected:
            raise FetchFailure(
                f"expected {expected} but received {kind or 'unrecognised content'} "
                f"(content-type {content_type or 'unknown'})",
                response.status_code,
            )
        if not response.content:
            raise FetchFailure("empty response body", response.status_code)
        return DownloadedAsset(
            content=response.content,
            final_url=str(response.url),
            http_status=response.status_code,
            content_type=content_type,
            kind=kind,
        )

    def _get(self, url: str) -> httpx.Response:
        last_status: int | None = None
        last_error = "request failed"
        for attempt in range(1, self._attempts + 1):
            headers = ALTERNATE_BROWSER_HEADERS if attempt > 1 and last_status == 403 else None
            try:
                response = self.client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("GET %s attempt %d failed: %s", url, attempt, last_error)
            else:
                last_status = response.status_code
                if 200 <= response.status_code < 300:
                    return response
                last_error = f"HTTP {response.status_code}"
                logger.warning("GET %s attempt %d returned %d", url, attempt, response.status_code)
                if response.status_code not in RETRYABLE_STATUSES:
                    break
            if attempt < self._attempts and self._backoff > 0:
                self._sleep(self._backoff * attempt)
        raise FetchFailure(f"{last_error} for {url}", last_status)

    def _stale_row(
        self,
        document: CorpusDocument,
        previous: ManifestRow | None,
        error: str,
        http_status: int | None,
    ) -> ManifestRow:
        if previous is not None and previous.file and self._resolve(previous.file).exists():
            logger.warning(
                "%s: download failed (%s); keeping previous file from %s",
                document.document_id, error, previous.retrieval_date,
            )
            return replace(
                previous,
                document_name=document.document_name,
                publisher=document.publisher,
                year=document.year,
                source_url=document.source_url,
                http_status=http_status,
                status=STATUS_STALE,
                error=error,
            )
        logger.error("%s: download failed (%s) and no previous file exists", document.document_id, error)
        return ManifestRow(
            document_id=document.document_id,
            document_name=document.document_name,
            publisher=document.publisher,
            year=document.year,
            source_url=document.source_url,
            downloaded_url=None,
            retrieval_date=None,
            sha256=None,
            http_status=http_status,
            status=STATUS_STALE,
            file=None,
            error=error,
        )

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path.resolve())

    def _resolve(self, stored: str) -> Path:
        path = Path(stored)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


def fetch_corpus(strict: bool = True, **kwargs: object) -> list[ManifestRow]:
    """Convenience wrapper: fetch the whole registry with a default client."""
    return Fetcher(**kwargs).fetch_all(strict=strict)  # type: ignore[arg-type]
