"""Deterministic post-retrieval context expansion utilities."""

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
    "clearance",
    "clear space",
    "distance",
    "height",
    "depth",
    "burial",
    "standpipe",
    "hydrant",
    "fdc",
    "piv",
    "knox",
    "inspection",
    "inches",
    "inch",
    "feet",
    "ft",
)

CHECKLIST_SOURCE_TERMS = (
    "pre-construction",
    "pre construction",
    "checklist",
    "guide",
    "hydrant",
    "cobb_county_fire",
    "cobb county fire",
    "fire marshal",
)

SUSPICIOUS_ENDINGS = (
    "hydrants",
    "shall be:",
    "including but not limited to:",
    "in such a manner th",
    "shall include:",
    "as follows:",
    "the following:",
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
    """Expand retrieved chunks to page/range or neighboring context before evidence checks."""

    settings = settings or get_settings()
    if not settings.context_expansion_enabled:
        return retrieved_docs, retrieved_sources

    mode = (mode or settings.context_expansion_mode or "auto").lower()
    if mode == "off":
        return retrieved_docs, retrieved_sources

    neighbor_window = neighbor_window if neighbor_window is not None else settings.context_neighbor_window
    max_expanded_docs = max_expanded_docs if max_expanded_docs is not None else settings.context_max_expanded_docs
    max_chars = max_chars if max_chars is not None else settings.context_max_chars
    slug = collection_slug(collection_name)
    page_records = load_page_records(str(settings.context_store_dir), slug)
    chunk_records = load_chunk_records(str(settings.context_store_dir), slug)

    expanded: list[tuple[Document, RetrievedSource]] = []
    seen: set[str] = set()
    total_chars = 0
    original_count = len(retrieved_docs)

    for doc, source in zip(retrieved_docs, retrieved_sources):
        candidates = _expand_one_doc(
            doc,
            source,
            mode=mode,
            page_records=page_records,
            chunk_records=chunk_records,
            neighbor_window=neighbor_window,
            max_chars_per_doc=max_chars_per_doc,
        )
        for candidate_doc, candidate_source in candidates:
            identity = _expanded_identity(candidate_doc, candidate_source)
            text_hash = hashlib.sha1(candidate_doc.page_content.encode("utf-8", errors="ignore")).hexdigest()
            if identity in seen or text_hash in seen:
                continue
            seen.add(identity)
            seen.add(text_hash)
            expanded.append((candidate_doc, candidate_source))

    expanded.sort(key=lambda item: _sort_key(item[0], item[1]))
    final_docs: list[Document] = []
    final_sources: list[RetrievedSource] = []
    for doc, source in expanded:
        if len(final_docs) >= max_expanded_docs:
            break
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        text = doc.page_content[: min(len(doc.page_content), remaining, max_chars_per_doc)]
        if not text.strip():
            continue
        if text != doc.page_content:
            doc = Document(page_content=text, metadata=dict(doc.metadata))
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
        return retrieved_docs, retrieved_sources

    logger.info(
        "Context expansion for %s expanded %s retrieved chunks to %s context blocks (%s chars, mode=%s).",
        slug,
        original_count,
        len(final_docs),
        total_chars,
        mode,
    )
    logger.debug("Original retrieved chunks: %s", [_debug_doc_label(doc, src) for doc, src in zip(retrieved_docs, retrieved_sources)])
    logger.debug("Expanded chunks: %s", [_debug_doc_label(doc, src) for doc, src in zip(final_docs, final_sources)])
    return final_docs, final_sources


def is_technical_fact_query(query: str) -> bool:
    """Return whether a query asks for exact technical or numeric requirements."""

    normalized = query.lower()
    return any(term in normalized for term in TECHNICAL_QUERY_TERMS)


def collection_slug(collection_name: str) -> str:
    """Map a collection name to the sidecar slug used by ingestion."""

    return COLLECTION_SLUGS.get(collection_name, collection_name.replace("cobb_code_docs_", ""))


@lru_cache(maxsize=16)
def load_page_records(context_store_dir: str, slug: str) -> list[dict[str, Any]]:
    return _load_jsonl(Path(context_store_dir) / f"{slug}_pages.jsonl")


@lru_cache(maxsize=16)
def load_chunk_records(context_store_dir: str, slug: str) -> list[dict[str, Any]]:
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


def _expand_one_doc(
    doc: Document,
    source: RetrievedSource,
    mode: str,
    page_records: list[dict[str, Any]],
    chunk_records: list[dict[str, Any]],
    neighbor_window: int,
    max_chars_per_doc: int,
) -> list[tuple[Document, RetrievedSource]]:
    metadata = dict(doc.metadata or {})
    expansion_mode = _choose_expansion_mode(doc, source, mode)
    if expansion_mode == "page":
        page_doc = _page_expansion(doc, source, page_records, max_chars_per_doc=max_chars_per_doc)
        if page_doc:
            return [page_doc]

    neighbor_docs = _neighbor_expansion(doc, source, chunk_records, neighbor_window)
    if neighbor_docs:
        return neighbor_docs

    if expansion_mode != "page":
        page_doc = _page_expansion(doc, source, page_records, max_chars_per_doc=max_chars_per_doc)
        if page_doc:
            return [page_doc]

    metadata["expansion_type"] = "original_chunk"
    return [(Document(page_content=doc.page_content, metadata=metadata), source)]


def _choose_expansion_mode(doc: Document, source: RetrievedSource, mode: str) -> str:
    if mode in {"page", "neighbors"}:
        return mode
    text = doc.page_content
    source_text = " ".join(
        str(value)
        for value in (
            source.source,
            doc.metadata.get("source"),
            doc.metadata.get("source_path"),
            doc.metadata.get("file_name"),
            doc.metadata.get("section"),
        )
        if value
    ).lower()
    if any(term in source_text for term in CHECKLIST_SOURCE_TERMS):
        return "page"
    if _looks_truncated(text):
        return "page"
    page_span = _page_span(doc.metadata)
    if page_span and page_span[1] - page_span[0] <= 3 and len(text) < 5000:
        return "page"
    return "neighbors"


def _page_expansion(
    doc: Document,
    source: RetrievedSource,
    page_records: list[dict[str, Any]],
    max_chars_per_doc: int,
) -> tuple[Document, RetrievedSource] | None:
    if not page_records:
        return None
    doc_id = _doc_id(doc, source)
    source_name = _source_name(doc, source)
    page = _page_number(doc, source)
    candidates = [
        record
        for record in page_records
        if _record_matches_doc(record, doc_id, source_name)
        and _record_matches_page(record, page)
    ]
    if not candidates and page is None:
        candidates = [record for record in page_records if _record_matches_doc(record, doc_id, source_name)]
    if not candidates:
        return None

    candidates.sort(key=lambda record: (_as_int(record.get("page_start"), 10**9), len(str(record.get("text") or ""))))
    record = candidates[0]
    text = str(record.get("text") or "").strip()
    if not text:
        return None
    text = _focus_text_around_anchor(text, doc.page_content, max_chars_per_doc)
    metadata = dict(doc.metadata or {})
    metadata.update(
        {
            "doc_id": record.get("doc_id") or metadata.get("doc_id"),
            "source": record.get("source") or metadata.get("source"),
            "source_path": record.get("source_path") or metadata.get("source_path"),
            "file_name": record.get("file_name") or metadata.get("file_name"),
            "page_start": record.get("page_start") or metadata.get("page_start"),
            "page_end": record.get("page_end") or metadata.get("page_end"),
            "section": record.get("section") or metadata.get("section"),
            "expansion_type": "page_context",
        }
    )
    page_label = _as_int(metadata.get("page_start"), source.page)
    expanded_source = RetrievedSource(
        source=str(metadata.get("source") or source.source),
        page=page_label,
        score=source.score,
        snippet=text[:350].replace("\n", " ").strip(),
    )
    return Document(page_content=text, metadata=metadata), expanded_source


def _neighbor_expansion(
    doc: Document,
    source: RetrievedSource,
    chunk_records: list[dict[str, Any]],
    neighbor_window: int,
) -> list[tuple[Document, RetrievedSource]]:
    if not chunk_records:
        return []
    doc_id = _doc_id(doc, source)
    source_name = _source_name(doc, source)
    chunk_index = _as_int(doc.metadata.get("chunk_index"), None)
    if chunk_index is None:
        return []
    start = chunk_index - max(neighbor_window, 0)
    end = chunk_index + max(neighbor_window, 0)
    candidates = [
        record
        for record in chunk_records
        if _record_matches_doc(record, doc_id, source_name)
        and start <= _as_int(record.get("chunk_index"), -10**9) <= end
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda record: (_as_int(record.get("page_start"), 10**9), _as_int(record.get("chunk_index"), 10**9)))
    expanded: list[tuple[Document, RetrievedSource]] = []
    for record in candidates:
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        metadata = dict(doc.metadata or {})
        metadata.update(
            {
                "doc_id": record.get("doc_id") or metadata.get("doc_id"),
                "source": record.get("source") or metadata.get("source"),
                "source_path": record.get("source_path") or metadata.get("source_path"),
                "file_name": record.get("file_name") or metadata.get("file_name"),
                "page_start": record.get("page_start") or metadata.get("page_start"),
                "page_end": record.get("page_end") or metadata.get("page_end"),
                "chunk_index": record.get("chunk_index"),
                "section": record.get("section") or metadata.get("section"),
                "expansion_type": "neighbor_chunk",
            }
        )
        expanded.append(
            (
                Document(page_content=text, metadata=metadata),
                RetrievedSource(
                    source=str(metadata.get("source") or source.source),
                    page=_as_int(metadata.get("page_start"), source.page),
                    score=source.score,
                    snippet=text[:350].replace("\n", " ").strip(),
                ),
            )
        )
    return expanded


def _record_matches_doc(record: dict[str, Any], doc_id: str, source_name: str) -> bool:
    record_doc_id = str(record.get("doc_id") or "")
    record_source = str(record.get("source") or record.get("source_path") or "")
    return bool(record_doc_id and record_doc_id == doc_id) or bool(record_source and record_source == source_name)


def _record_matches_page(record: dict[str, Any], page: int | None) -> bool:
    if page is None:
        return True
    page_start = _as_int(record.get("page_start") or record.get("page"), None)
    page_end = _as_int(record.get("page_end") or page_start, page_start)
    if page_start is None:
        return False
    return page_start <= page <= (page_end or page_start)


def _doc_id(doc: Document, source: RetrievedSource) -> str:
    value = str(doc.metadata.get("doc_id") or "")
    if value:
        return value
    source_name = _source_name(doc, source).replace("\\", "/").lower().strip()
    digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", source_name).strip("-")[:80] or "document"
    return f"{slug}-{digest}"


def _source_name(doc: Document, source: RetrievedSource) -> str:
    return str(doc.metadata.get("source") or doc.metadata.get("source_path") or doc.metadata.get("file_name") or source.source)


def _page_number(doc: Document, source: RetrievedSource) -> int | None:
    page_start = _as_int(doc.metadata.get("page_start"), None)
    if page_start is not None:
        return page_start
    page = _as_int(doc.metadata.get("page"), None)
    if page is not None:
        return page + 1 if page == 0 or page < 10000 else page
    return source.page


def _page_span(metadata: dict[str, Any]) -> tuple[int, int] | None:
    page_start = _as_int(metadata.get("page_start"), None)
    page_end = _as_int(metadata.get("page_end"), page_start)
    if page_start is None:
        return None
    return page_start, page_end or page_start


def _looks_truncated(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    if any(stripped.endswith(ending) for ending in SUSPICIOUS_ENDINGS):
        return True
    tail = stripped[-90:]
    if re.search(r"(shall be|must include|including but not limited to|visibility and access requirements)\s*:?\s*$", tail):
        return True
    if tail.endswith((" th", " a", " an", " the", " of", " to", " for", " and", " or")):
        return True
    bullet_count = len(re.findall(r"(?m)^\s*[-*•]\s+", text))
    heading_like = bool(re.search(r"(?m)^\s*#{1,4}\s+\S+", text) or re.search(r"(?m)^[A-Z][A-Za-z /&-]{8,}:\s*$", text))
    return heading_like and bullet_count == 0 and len(text) < 1200


def _focus_text_around_anchor(text: str, anchor: str, max_chars: int) -> str:
    """Return a window from text centered near the retrieved chunk, favoring look-ahead."""

    if len(text) <= max_chars:
        return text
    normalized_text = _normalize_for_find(text)
    normalized_anchor = _normalize_for_find(anchor[:1200])
    position = normalized_text.find(normalized_anchor[:300]) if normalized_anchor else -1
    if position < 0:
        distinctive = _distinctive_anchor(anchor)
        position = normalized_text.find(distinctive) if distinctive else -1
    if position < 0:
        return text[:max_chars]

    start = max(position - int(max_chars * 0.25), 0)
    end = min(start + max_chars, len(text))
    if end - start < max_chars:
        start = max(end - max_chars, 0)
    return text[start:end]


def _normalize_for_find(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _distinctive_anchor(text: str) -> str:
    words = re.findall(r"[a-z0-9]{4,}", text.lower())
    if not words:
        return ""
    return " ".join(words[:8])


def _expanded_identity(doc: Document, source: RetrievedSource) -> str:
    metadata = doc.metadata or {}
    return "::".join(
        str(part)
        for part in (
            metadata.get("expansion_type") or "context",
            metadata.get("doc_id") or source.source,
            metadata.get("page_start") or metadata.get("page") or source.page or "na",
            metadata.get("page_end") or "na",
            metadata.get("chunk_index") or "na",
        )
    )


def _sort_key(doc: Document, source: RetrievedSource) -> tuple[str, int, int]:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("source") or source.source),
        _as_int(metadata.get("page_start") or source.page, 10**9) or 10**9,
        _as_int(metadata.get("chunk_index"), 10**9) or 10**9,
    )


def _debug_doc_label(doc: Document, source: RetrievedSource) -> str:
    metadata = doc.metadata or {}
    return (
        f"{metadata.get('expansion_type', 'retrieved')} | {metadata.get('source') or source.source} | "
        f"page={metadata.get('page_start') or source.page} | chunk={metadata.get('chunk_index')} | "
        f"chars={len(doc.page_content)}"
    )


def _as_int(value: Any, default: int | None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default
