"""Print neighbours and the grouped result for one question.

Uses the index at ``data/index``. The first block is the top candidates
before the similarity threshold and the per-document cap, so a miss still
shows how close the nearest chunk was. The second block is what retrieval
keeps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import RetrievalResult, RetrievedChunk, Retriever  # noqa: E402

PREVIEW_CHARS = 80
CANDIDATES_SHOWN = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to embed and search")
    parser.add_argument(
        "--document-id",
        default=None,
        help="Restrict search to one registry document_id",
    )
    args = parser.parse_args()

    try:
        result = Retriever().retrieve(args.question, document_id=args.document_id)
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    _print_candidates(result)
    print()
    _print_groups(result)


def _print_candidates(result: RetrievalResult) -> None:
    shown = result.candidates[:CANDIDATES_SHOWN]
    print(f"Candidates before threshold: {result.candidate_count} (showing {len(shown)})")
    if not shown:
        print("  (none)")
        return
    for hit in shown:
        print(f"  {_line(hit)}")


def _print_groups(result: RetrievalResult) -> None:
    kept = sum(len(group.hits) for group in result.groups)
    print(f"Searched: {', '.join(result.searched)}")
    print(f"Grouped:  {kept} hits in {len(result.groups)} documents")
    if not result.groups:
        print("  (none)")
        return
    for group in result.groups:
        print(f"\n{group.document_id} ({len(group.hits)})")
        for hit in group.hits:
            print(f"  {_line(hit)}")


def _line(hit: RetrievedChunk) -> str:
    text = " ".join(hit.text.split())
    if len(text) > PREVIEW_CHARS:
        text = text[: PREVIEW_CHARS - 3] + "..."
    return (
        f"{hit.similarity:.3f}  {hit.chunk_id:<26} {hit.block_type:<10} "
        f"{hit.section_heading}  | {text}"
    )


if __name__ == "__main__":
    main()
