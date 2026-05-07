"""Deterministic post-retrieval neighbor context expansion utilities.

This module intentionally does NOT perform:
- PyPDF fallback for Docling collections
- cross-parser expansion
- full-page/page-level expansion

Expansion behavior:
- retrieve top-k chunks
- for each retrieved chunk, add same-document neighbor chunks:
  chunk_index - 1, chunk_index, chunk_index + 1
- preserve raw retrieval rank globally
- sort only within each retrieved group by chunk_index
- apply context budget after retrieval-priority ordering
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import COLLECTION_SLUGS, Settings, get_settings
from src.retriever import RetrievedSource


logger = logging.getLogger(__name__)


TECHNICAL_QUERY_TERMS = (
    "minimum",
    "maximum",
    "required",
    "requirement",
    "requirements",
    "shall",
    "must",
    "may",
    "allowed",
    "prohibited",
    "not permitted",
    "permit",
    "inspection",
    "deadline",
    "fee",
    "fine",
    "penalty",
    "distance",
    "clearance",
    "clear space",
    "height",
    "depth",
    "width",
    "length",
    "area",
    "square feet",
    "feet",
    "ft",
    "inch",
    "inches",
    "psi",
    "gpm",
    "cfm",
    "degrees",
    "days",
    "hours",
    "section",
    "code section",
)


def expand_retrieved_docs(
    retrieved_docs: list[Document],
    retrieved_sources: list[RetrievedSource],
    collection_name: str,
    mode: str | None = None,
    neighbor_window: int | None = None,
    max_expanded_docs: int | None = None,
    max_chars: int | None = None,
    max_chars_per_doc: int = 6000,
    settings: Settings | None = None,
) -> tuple[list[Document], list[RetrievedSource]]:
    """Expand retrieved chunks with deterministic same-document neighbors.

    Important:
    - No page expansion.
    - No PyPDF fallback for Docling collections.
    - No cross-parser expansion.
    - Global order follows raw retrieval rank.
    - Within each retrieved result, neighbors are ordered by chunk_index.
    """

    settings = settings or get_settings()

    if not settings.context_expansion_enabled:
        logger.debug("Context expansion disabled by settings.")
        return retrieved_docs, retrieved_sources

    mode = (mode or settings.context_expansion_mode or "neighbors").lower()
    if mode == "off":
        logger.debug("Context expansion mode is off.")
        return retrieved_docs, retrieved_sources

    # Force neighbors-only behavior. Do not allow "auto" or "page" to trigger page expansion.
    if mode not in {"neighbors", "auto"}:
        logger.warning(
            "Unsupported context expansion mode '%s'. Forcing deterministic neighbor expansion only.",
            mode,
        )

    # User requested only immediate -1/+1 neighbor expansion.
    effective_neighbor_window = 1

    max_expanded_docs = (
        max_expanded_docs
        if max_expanded_docs is not None
        else settings.context_max_expanded_docs
    )
    max_chars = max_chars if max_chars is not None else settings.context_max_chars

    slug = collection_slug(collection_name)

    # Only load same-backend chunk records. Do not load page records.
    # Do not load pypdf_chroma fallback records for Docling collections.
    chunk_records = load_chunk_records(str(settings.context_store_dir), slug)

    expanded_pairs: list[tuple[Document, RetrievedSource]] = []
    seen: set[str] = set()

    logger.debug(
        "Raw retrieval results before neighbor expansion: %s",
        [
            _debug_doc_label(doc, src, prefix=f"rank={rank}")
            for rank, (doc, src) in enumerate(
                zip(retrieved_docs, retrieved_sources),
                start=1,
            )
        ],
    )

    for retrieved_rank, (doc, source) in enumerate(
        zip(retrieved_docs, retrieved_sources),
        start=1,
    ):
        group = _expand_one_doc_neighbors_only(
            doc=doc,
            source=source,
            chunk_records=chunk_records,
            collection_slug_value=slug,
            neighbor_window=effective_neighbor_window,
            retrieved_rank=retrieved_rank,
        )

        # Preserve global raw retrieval rank:
        # append each expanded group in retrieval order.
        for candidate_doc, candidate_source in group:
            candidate_metadata = dict(candidate_doc.metadata or {})
            candidate_metadata["retrieval_rank"] = retrieved_rank
            candidate_metadata.setdefault("collection_slug", slug)

            candidate_doc = Document(
                page_content=candidate_doc.page_content,
                metadata=candidate_metadata,
            )

            identity = _expanded_identity(candidate_doc, candidate_source, slug)
            text_hash = hashlib.sha1(
                candidate_doc.page_content.encode("utf-8", errors="ignore")
            ).hexdigest()

            if identity in seen or text_hash in seen:
                logger.debug("Skipping duplicate expanded chunk: %s", identity)
                continue

            seen.add(identity)
            seen.add(text_hash)
            expanded_pairs.append((candidate_doc, candidate_source))

    final_docs: list[Document] = []
    final_sources: list[RetrievedSource] = []
    total_chars = 0

    # Apply context budget after retrieval-priority ordering.
    for doc, source in expanded_pairs:
        if len(final_docs) >= max_expanded_docs:
            logger.debug(
                "Stopping expansion because max_expanded_docs=%s was reached.",
                max_expanded_docs,
            )
            break

        remaining = max_chars - total_chars
        if remaining <= 0:
            logger.debug("Stopping expansion because max_chars=%s was reached.", max_chars)
            break

        text = doc.page_content or ""
        text = text[: min(len(text), remaining, max_chars_per_doc)]

        if not text.strip():
            continue

        if text != doc.page_content:
            metadata = dict(doc.metadata or {})
            metadata["truncated_by_context_budget"] = True
            doc = Document(page_content=text, metadata=metadata)
            source = RetrievedSource(
                source=source.source,
                page=source.page,
                score=source.score,
                snippet=text[:350].replace("\n", " ").strip(),
            )

        final_docs.append(doc)
        final_sources.append(source)
        total_chars += len(text)

    if not final_docs:
        logger.warning(
            "Context expansion produced no final docs for collection=%s. Returning original retrieved docs.",
            slug,
        )
        return retrieved_docs, retrieved_sources

    logger.info(
        "Neighbor context expansion for %s expanded %s retrieved chunks to %s context blocks "
        "(%s chars, neighbors=-1/+1 only).",
        slug,
        len(retrieved_docs),
        len(final_docs),
        total_chars,
    )

    logger.debug(
        "Expanded neighbor chunks after budget: %s",
        [
            _debug_doc_label(doc, src, prefix=f"final_pos={idx}")
            for idx, (doc, src) in enumerate(zip(final_docs, final_sources), start=1)
        ],
    )

    return final_docs, final_sources


def _expand_one_doc_neighbors_only(
    doc: Document,
    source: RetrievedSource,
    chunk_records: list[dict[str, Any]],
    collection_slug_value: str,
    neighbor_window: int,
    retrieved_rank: int,
) -> list[tuple[Document, RetrievedSource]]:
    """Return previous/current/next chunks for the retrieved chunk.

    If chunk_index or chunk records are unavailable, return only the original chunk.
    No page fallback is allowed.
    """

    metadata = dict(doc.metadata or {})
    source_name = _source_name(doc, source)
    doc_id = _doc_id(doc, source)
    chunk_index = _as_int(metadata.get("chunk_index"), None)

    if chunk_index is None:
        logger.info(
            "Neighbor expansion skipped for rank=%s source=%s because chunk_index is missing. "
            "Using original chunk only.",
            retrieved_rank,
            source_name,
        )
        metadata["expansion_type"] = "original_chunk_no_chunk_index"
        metadata["neighbor_expansion_skipped"] = "missing_chunk_index"
        return [(Document(page_content=doc.page_content, metadata=metadata), source)]

    if not chunk_records:
        logger.info(
            "Neighbor expansion skipped for rank=%s source=%s chunk_index=%s because chunk sidecar "
            "records are missing. Using original chunk only.",
            retrieved_rank,
            source_name,
            chunk_index,
        )
        metadata["expansion_type"] = "original_chunk_no_sidecar"
        metadata["neighbor_expansion_skipped"] = "missing_chunk_sidecar"
        return [(Document(page_content=doc.page_content, metadata=metadata), source)]

    window = max(0, min(neighbor_window, 1))
    wanted_indices = set(range(chunk_index - window, chunk_index + window + 1))

    candidates = [
        record
        for record in chunk_records
        if _record_matches_doc(record, doc_id, source_name)
        and _record_matches_backend_or_parser(record, metadata, collection_slug_value)
        and _as_int(record.get("chunk_index"), -10**9) in wanted_indices
    ]

    if not candidates:
        logger.info(
            "No same-document neighbor records found for rank=%s source=%s chunk_index=%s. "
            "Using original chunk only.",
            retrieved_rank,
            source_name,
            chunk_index,
        )
        metadata["expansion_type"] = "original_chunk_no_neighbors_found"
        metadata["neighbor_expansion_skipped"] = "no_matching_neighbor_records"
        return [(Document(page_content=doc.page_content, metadata=metadata), source)]

    # Sort only within this retrieved group.
    candidates.sort(key=lambda record: _as_int(record.get("chunk_index"), 10**9) or 10**9)

    found_indices = {
        _as_int(record.get("chunk_index"), None)
        for record in candidates
    }
    missing_indices = sorted(idx for idx in wanted_indices if idx not in found_indices)

    if missing_indices:
        logger.debug(
            "Neighbor expansion for rank=%s source=%s chunk_index=%s missing neighbor indices: %s",
            retrieved_rank,
            source_name,
            chunk_index,
            missing_indices,
        )

    expanded: list[tuple[Document, RetrievedSource]] = []

    for record in candidates:
        text = str(record.get("text") or "").strip()
        if not text:
            continue

        record_chunk_index = _as_int(record.get("chunk_index"), None)

        expanded_metadata = dict(metadata)
        expanded_metadata.update(
            {
                "doc_id": record.get("doc_id") or expanded_metadata.get("doc_id"),
                "source": record.get("source") or expanded_metadata.get("source"),
                "source_path": record.get("source_path") or expanded_metadata.get("source_path"),
                "file_name": record.get("file_name") or expanded_metadata.get("file_name"),
                "page_start": record.get("page_start") or expanded_metadata.get("page_start"),
                "page_end": record.get("page_end") or expanded_metadata.get("page_end"),
                "page": record.get("page") or expanded_metadata.get("page"),
                "chunk_index": record_chunk_index,
                "chunk_id": record.get("chunk_id") or expanded_metadata.get("chunk_id"),
                "section": record.get("section") or expanded_metadata.get("section"),
                "parser": record.get("parser") or expanded_metadata.get("parser"),
                "backend": record.get("backend") or expanded_metadata.get("backend"),
                "collection_slug": record.get("collection_slug") or collection_slug_value,
                "expansion_type": (
                    "retrieved_chunk"
                    if record_chunk_index == chunk_index
                    else "neighbor_chunk"
                ),
                "anchor_chunk_index": chunk_index,
            }
        )

        expanded_source = RetrievedSource(
            source=str(expanded_metadata.get("source") or source.source),
            page=_as_int(
                expanded_metadata.get("page_start")
                or expanded_metadata.get("page"),
                source.page,
            ),
            score=source.score,
            snippet=text[:350].replace("\n", " ").strip(),
        )

        expanded.append(
            (
                Document(page_content=text, metadata=expanded_metadata),
                expanded_source,
            )
        )

    if not expanded:
        logger.info(
            "Neighbor expansion produced empty text for rank=%s source=%s chunk_index=%s. "
            "Using original chunk only.",
            retrieved_rank,
            source_name,
            chunk_index,
        )
        metadata["expansion_type"] = "original_chunk_empty_neighbors"
        metadata["neighbor_expansion_skipped"] = "empty_neighbor_text"
        return [(Document(page_content=doc.page_content, metadata=metadata), source)]

    logger.debug(
        "Neighbor expansion group for rank=%s source=%s anchor_chunk=%s -> chunks=%s",
        retrieved_rank,
        source_name,
        chunk_index,
        [
            candidate.metadata.get("chunk_index")
            for candidate, _ in expanded
        ],
    )

    return expanded


def is_technical_fact_query(query: str) -> bool:
    """Return whether a query asks for exact technical or numeric requirements."""

    normalized = query.lower()
    return any(term in normalized for term in TECHNICAL_QUERY_TERMS)


def collection_slug(collection_name: str) -> str:
    """Map a collection name to the sidecar slug used by ingestion."""

    return COLLECTION_SLUGS.get(
        collection_name,
        collection_name.replace("cobb_code_docs_", ""),
    )


@lru_cache(maxsize=16)
def load_chunk_records(context_store_dir: str, slug: str) -> list[dict[str, Any]]:
    """Load chunk sidecar records for the selected collection/backend only."""

    return _load_jsonl(Path(context_store_dir) / f"{slug}_chunks.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.info("Context expansion sidecar not found: %s.", path)
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL row in %s.", path)

    return records


def _record_matches_doc(record: dict[str, Any], doc_id: str, source_name: str) -> bool:
    """Match records by doc_id first, then normalized source path/name."""

    record_doc_id = str(record.get("doc_id") or "")
    if record_doc_id and record_doc_id == doc_id:
        return True

    record_source = str(
        record.get("source")
        or record.get("source_path")
        or record.get("file_name")
        or ""
    )

    return bool(record_source) and _normalize_source(record_source) == _normalize_source(source_name)


def _record_matches_backend_or_parser(
    record: dict[str, Any],
    metadata: dict[str, Any],
    collection_slug_value: str,
) -> bool:
    """Ensure neighbor chunks come from the same backend/parser/collection when metadata exists.

    This prevents Docling retrieval from pulling PyPDF-derived sidecar records.
    """

    record_collection = str(record.get("collection_slug") or record.get("collection") or "")
    if record_collection and record_collection != collection_slug_value:
        return False

    record_backend = str(record.get("backend") or "").lower()
    doc_backend = str(metadata.get("backend") or "").lower()

    if record_backend and doc_backend and record_backend != doc_backend:
        return False

    record_parser = str(record.get("parser") or "").lower()
    doc_parser = str(metadata.get("parser") or "").lower()

    if record_parser and doc_parser and record_parser != doc_parser:
        return False

    return True


def _doc_id(doc: Document, source: RetrievedSource) -> str:
    value = str(doc.metadata.get("doc_id") or "")
    if value:
        return value

    source_name = _source_name(doc, source).replace("\\", "/").lower().strip()
    digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", source_name).strip("-")[:80] or "document"

    return f"{slug}-{digest}"


def _source_name(doc: Document, source: RetrievedSource) -> str:
    return str(
        doc.metadata.get("source")
        or doc.metadata.get("source_path")
        or doc.metadata.get("file_name")
        or source.source
    )


def _normalize_source(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _expanded_identity(
    doc: Document,
    source: RetrievedSource,
    collection_slug_value: str,
) -> str:
    metadata = doc.metadata or {}

    return "::".join(
        str(part)
        for part in (
            collection_slug_value,
            metadata.get("backend") or "na",
            metadata.get("parser") or "na",
            metadata.get("doc_id") or source.source,
            metadata.get("source") or source.source,
            metadata.get("chunk_index") or "na",
        )
    )


def _debug_doc_label(
    doc: Document,
    source: RetrievedSource,
    prefix: str = "",
) -> str:
    metadata = doc.metadata or {}

    parts = [
        prefix,
        f"type={metadata.get('expansion_type', 'retrieved')}",
        f"source={metadata.get('source') or source.source}",
        f"page={metadata.get('page_start') or metadata.get('page') or source.page}",
        f"chunk={metadata.get('chunk_index')}",
        f"rank={metadata.get('retrieval_rank')}",
        f"backend={metadata.get('backend')}",
        f"parser={metadata.get('parser')}",
        f"chars={len(doc.page_content or '')}",
    ]

    return " | ".join(part for part in parts if part)


def _as_int(value: Any, default: int | None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default