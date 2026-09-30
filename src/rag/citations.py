"""Citation renderer (implementation plan, phase 6; architecture §5.5, §6).

Takes the generator's payload and the retrieved groups. For every claim it
attaches document name, publisher, year, and ``source_url`` read from that
document's chunk records, never from the model. Sections follow registry
order so the same question always lays out the same way. A generated
document that was not in the retrieval scope has no chunk record to cite
and is left out; the validator then has nothing to keep for it.

The renderer never merges two documents into one sentence. Each section is
that document's claims and nothing else.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.constants import CORPUS
from src.rag.models import Citation, Claim, GeneratedPayload, Section
from src.rag.retriever import DocumentHits

logger = logging.getLogger(__name__)

_REGISTRY_POSITION: dict[str, int] = {
    document.document_id: index for index, document in enumerate(CORPUS)
}


def citation_for(group: DocumentHits) -> Citation:
    """Name, publisher, year, and URL from the first chunk record in ``group``."""
    if not group.hits:
        raise ValueError(f"no chunk record to cite for {group.document_id}")
    hit = group.hits[0]
    return Citation(
        document_name=hit.document_name,
        publisher=hit.publisher,
        year=hit.year,
        source_url=hit.source_url,
    )


def order_sections(sections: Sequence[Section]) -> list[Section]:
    """Registry order. A section whose id is not in the registry sorts last."""
    return sorted(
        sections,
        key=lambda section: _REGISTRY_POSITION.get(section.document_id, len(_REGISTRY_POSITION)),
    )


def render_sections(payload: GeneratedPayload, groups: Sequence[DocumentHits]) -> list[Section]:
    """One section per generated document that was retrieved and has claims.

    Claims keep the chunk ids the model gave; the validator checks them. If
    the model repeats a ``document_id``, its claims are merged into one
    section. A document with no claims is omitted: the model said it does not
    answer the question.
    """
    by_id: dict[str, DocumentHits] = {group.document_id: group for group in groups}
    sections: dict[str, Section] = {}
    for generated in payload.documents:
        group = by_id.get(generated.document_id)
        if group is None:
            logger.info("dropping generated document outside the retrieval scope: %s", generated.document_id)
            continue
        if not generated.claims:
            continue
        citation = citation_for(group)
        claims = [
            Claim(text=claim.text.strip(), chunk_ids=list(claim.chunk_ids), citation=citation)
            for claim in generated.claims
        ]
        section = sections.get(generated.document_id)
        if section is None:
            sections[generated.document_id] = Section(
                document_id=group.document_id,
                document_name=citation.document_name,
                publisher=citation.publisher,
                year=citation.year,
                source_url=citation.source_url,
                claims=claims,
            )
        else:
            section.claims.extend(claims)
    return order_sections(list(sections.values()))


def citation_line(citation: Citation) -> str:
    """``— Name, Publisher, Year`` as architecture §5.5 shows it."""
    return f"— {citation.document_name}, {citation.publisher}, {citation.year}"


def render_text(sections: Sequence[Section]) -> str:
    """Plain-text answer: one block per section, the citation under each claim.

    Layout from architecture §6. There is no sentence that attributes a
    combined statement to "the guidelines"; each block is one document.
    """
    blocks: list[str] = []
    for section in sections:
        lines = [f"According to {section.document_name} — {section.publisher}, {section.year}"]
        for claim in section.claims:
            lines.append(f"  {claim.text}")
            lines.append(f"  {citation_line(claim.citation)}")
            lines.append(f"    {claim.citation.source_url}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
