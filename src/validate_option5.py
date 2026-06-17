"""Smoke validation for Option 5 RAG-Anything + LightRAG retrieval."""

from __future__ import annotations

import argparse
from urllib.parse import urlparse

from src.config import get_settings
from src.rag_anything_lightrag_option5 import option5_storage_exists, search_option5
from src.tools import _allowed_domains, _is_allowed_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the isolated Option 5 KG/index.")
    parser.add_argument(
        "--query",
        default="What fire inspection or building permit requirements are mentioned?",
        help="Sample query to run against Option 5.",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"Option 5 storage: {settings.rag_anything_storage_dir}")
    print(f"LightRAG working dir: {settings.lightrag_working_dir}")
    print(f"Storage folder exists: {settings.rag_anything_storage_dir.exists()}")
    print(f"Index load check: {option5_storage_exists(settings)}")
    if not option5_storage_exists(settings):
        raise SystemExit(
            "Option 5 index is not available. Build it with "
            "`python -m src.ingestion --pipeline rag_anything_lightrag_option5`."
        )

    docs, sources = search_option5(args.query, settings=settings)
    print(f"Query returned docs: {len(docs)}")
    print(f"Query returned sources: {len(sources)}")
    if not docs or not sources:
        raise SystemExit("Option 5 query returned no local sources.")
    for source in sources[:3]:
        print(f"Local source: {source.source} | score={source.score:.2f}")

    allowed_domains = _allowed_domains(settings.allowed_web_domains)
    rejected_example = "https://example.com/not-official"
    accepted_examples = [f"https://{domain}/" for domain in allowed_domains]
    rejected = not _is_allowed_url(rejected_example, allowed_domains)
    accepted = all(_is_allowed_url(url, allowed_domains) for url in accepted_examples)
    print(f"Allowed web domains: {', '.join(allowed_domains)}")
    print(f"Rejects non-approved domain {urlparse(rejected_example).netloc}: {rejected}")
    print(f"Accepts configured approved domains: {accepted}")
    if not rejected or not accepted:
        raise SystemExit("Web-domain allowlist validation failed.")

    print("Option 5 smoke validation passed.")


if __name__ == "__main__":
    main()
