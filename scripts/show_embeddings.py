"""Print a few chunk embeddings stored in the local Chroma index.

Opens ``data/index`` and reads the ``guidance`` collection. Vectors were
written by the index build; this script does not load the embedding model.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chromadb.errors import NotFoundError  # noqa: E402

from src.constants import CHROMA_COLLECTION_NAME, INDEX_DIR  # noqa: E402
from src.ingestion.indexer import chroma_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=3,
        help="How many stored chunks to print (default: 3)",
    )
    parser.add_argument(
        "--values",
        type=int,
        default=8,
        help="How many leading and trailing vector values to print (default: 8)",
    )
    args = parser.parse_args()
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        raise SystemExit(2)
    if args.values < 1:
        print("--values must be at least 1", file=sys.stderr)
        raise SystemExit(2)

    client = chroma_client(INDEX_DIR)
    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=None,
        )
    except NotFoundError:
        print(f"No collection {CHROMA_COLLECTION_NAME!r} in {INDEX_DIR}", file=sys.stderr)
        raise SystemExit(1)

    total = collection.count()
    if total == 0:
        print(f"Collection {CHROMA_COLLECTION_NAME!r} is empty")
        return

    limit = min(args.count, total)
    stored = collection.get(
        limit=limit,
        include=["embeddings", "documents", "metadatas"],
    )

    print(f"Index:      {INDEX_DIR}")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Stored:     {total} chunks")
    print(f"Showing:    {limit}\n")

    ids = stored["ids"]
    embeddings = stored["embeddings"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]
    for index, chunk_id in enumerate(ids):
        vector = [float(value) for value in embeddings[index]]
        metadata = metadatas[index] or {}
        text = (documents[index] or "").replace("\n", " ")
        preview = text if len(text) <= 160 else text[:157] + "..."
        norm = math.sqrt(sum(value * value for value in vector))
        head = _format_slice(vector[: args.values])
        tail = _format_slice(vector[-args.values :])

        print(f"--- {chunk_id} ---")
        print(f"document:  {metadata.get('document_name', '')} ({metadata.get('document_id', '')})")
        print(f"section:   {metadata.get('section_heading', '')}")
        print(f"block:     {metadata.get('block_type', '')}")
        print(f"text:      {preview}")
        print(f"dimension: {len(vector)}")
        print(f"l2 norm:   {norm:.6f}")
        print(f"first {args.values}: {head}")
        print(f"last {args.values}:  {tail}")
        print()


def _format_slice(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


if __name__ == "__main__":
    main()
