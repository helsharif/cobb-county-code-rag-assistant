"""Debug checks for deterministic context expansion on known Cobb County cases."""

from __future__ import annotations

import argparse
import logging

from src.config import DOCLING_COLLECTION_NAME, ORIGINAL_COLLECTION_NAME
from src.context_expansion import expand_retrieved_docs
from src.retriever import search_documents


CHECKS = [
    {
        "name": "Temporary standpipe",
        "question": "When is a temporary standpipe required during multi-story building construction?",
        "expected": "temporary standpipe",
    },
    {
        "name": "Underground fire main burial depth",
        "question": "What is the minimum burial depth for underground fire main piping in Cobb County?",
        "expected": "42",
    },
    {
        "name": "Private hydrant clear space",
        "question": "What is the minimum required clear space around a private fire hydrant in Cobb County?",
        "expected": "clear",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug retrieval context expansion.")
    parser.add_argument("--collection-name", default=DOCLING_COLLECTION_NAME)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    failures = 0
    for check in CHECKS:
        docs, sources = search_documents(check["question"], k=args.k, collection_name=args.collection_name)
        expanded_docs, expanded_sources = expand_retrieved_docs(docs, sources, collection_name=args.collection_name)
        expanded_text = "\n\n".join(doc.page_content for doc in expanded_docs)
        found = check["expected"].lower() in expanded_text.lower()
        status = "PASS" if found else "CHECK"
        print(f"\n[{status}] {check['name']}")
        print(f"Question: {check['question']}")
        print(f"Original chunks: {len(docs)} | Expanded blocks: {len(expanded_docs)}")
        for index, source in enumerate(expanded_sources, start=1):
            print(f"  Expanded {index}: {source.source}, page={source.page}, score={source.score:.2f}")
        if not found:
            failures += 1
            print(f"Expected marker not found: {check['expected']!r}")
        if args.verbose:
            print(expanded_text[:4000])

    if args.collection_name == ORIGINAL_COLLECTION_NAME:
        print("\nChecked Option 1 / PyPDF collection.")
    else:
        print("\nChecked Docling-oriented collection.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
