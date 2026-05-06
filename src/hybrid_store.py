"""Local Chroma + BM25 hybrid retrieval utilities."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from src.config import DOCLING_COLLECTION_NAME, Settings, get_chat_model, get_settings
from src.retriever import RetrievedSource, get_vectorstore, vectorstore_exists


logger = logging.getLogger(__name__)


def bm25_index_path(settings: Settings | None = None) -> Path:
    """Return the persisted local BM25 corpus path."""

    settings = settings or get_settings()
    return settings.bm25_index_dir / settings.bm25_index_file


def save_bm25_corpus(documents: list[Document], settings: Settings | None = None) -> Path:
    """Persist chunks and metadata used by local BM25 retrieval."""

    settings = settings or get_settings()
    path = bm25_index_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
        for doc in documents
        if doc.page_content.strip()
    ]
    with path.open("w", encoding="utf-8") as file:
        json.dump({"documents": rows}, file, ensure_ascii=False)
    logger.info("Saved %s chunks to BM25 corpus at %s.", len(rows), path)
    return path


def bm25_index_exists(settings: Settings | None = None) -> bool:
    """Return whether the local BM25 corpus exists."""

    return bm25_index_path(settings).exists()


def search_chroma_bm25_hybrid(
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
) -> tuple[list[Document], list[RetrievedSource]]:
    """Fuse Docling Chroma vector retrieval with local BM25 keyword retrieval."""

    settings = settings or get_settings()
    top_k = k or settings.retriever_k
    vector_ranked = _search_docling_chroma(query, top_k, settings)
    bm25_ranked = _search_bm25(query, max(top_k * 4, 20), settings)
    return _fuse_ranked_results([vector_ranked, bm25_ranked], top_k)


def search_chroma_bm25_with_query_expansion(
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
) -> tuple[list[Document], list[RetrievedSource]]:
    """Expand the query, run local hybrid retrieval for each query, and fuse results."""

    settings = settings or get_settings()
    top_k = k or settings.retriever_k
    expanded_queries = expand_query(query)
    ranked_sets: list[list[tuple[str, Document, RetrievedSource]]] = []
    for expanded_query in expanded_queries:
        docs, sources = search_chroma_bm25_hybrid(
            expanded_query,
            k=max(top_k * 2, top_k),
            settings=settings,
        )
        ranked_sets.append(
            [(_document_identity(doc, source), doc, source) for doc, source in zip(docs, sources)]
        )
    return _fuse_ranked_results(ranked_sets, top_k)


def expand_query(query: str) -> list[str]:
    """Return the original query plus four LLM-expanded technical and step-back variants."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You expand Cobb County, Georgia building, fire, permit, inspection, and code questions for retrieval. "
                "Return only valid JSON with key queries containing exactly four strings. "
                "Generate diverse retrieval queries that cover technical terms, likely code vocabulary, synonyms, "
                "document headings, and at least one step-back conceptual query. Do not answer the question.",
            ),
            (
                "human",
                "Original question:\n{query}\n\n"
                "Return four additional search queries as JSON:\n"
                '{{"queries": ["...", "...", "...", "..."]}}',
            ),
        ]
    )
    try:
        response = (prompt | get_chat_model(temperature=0.0)).invoke({"query": query})
        content = getattr(response, "content", str(response)).strip()
        parsed = _parse_query_expansion_json(content)
        expansions = [str(item).strip() for item in parsed.get("queries", []) if str(item).strip()]
    except Exception as exc:
        logger.warning("Query expansion failed; using deterministic fallback queries: %s", exc)
        expansions = _fallback_expanded_queries(query)

    queries = [query]
    for expanded in expansions:
        if expanded.lower() != query.lower() and expanded not in queries:
            queries.append(expanded)
        if len(queries) == 5:
            break
    for fallback in _fallback_expanded_queries(query):
        if len(queries) == 5:
            break
        if fallback.lower() != query.lower() and fallback not in queries:
            queries.append(fallback)
    logger.info("Query expansion generated %s total queries.", len(queries))
    return queries[:5]


def _search_docling_chroma(
    query: str,
    k: int,
    settings: Settings,
) -> list[tuple[str, Document, RetrievedSource]]:
    if not vectorstore_exists(settings.vectorstore_dir, DOCLING_COLLECTION_NAME):
        return []
    vectorstore = get_vectorstore(settings, DOCLING_COLLECTION_NAME)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=max(k * 2, k))
    ranked: list[tuple[str, Document, RetrievedSource]] = []
    for doc, score in results:
        source = _source_from_document(doc, float(score))
        ranked.append((_document_identity(doc, source), doc, source))
    return ranked


def _search_bm25(query: str, k: int, settings: Settings) -> list[tuple[str, Document, RetrievedSource]]:
    documents = _load_bm25_documents(settings)
    if not documents:
        return []
    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    if len(scores) == 0:
        return []
    max_score = float(max(scores)) or 1.0
    ranked_indices = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)[:k]
    ranked: list[tuple[str, Document, RetrievedSource]] = []
    for index in ranked_indices:
        doc = documents[index]
        normalized_score = float(scores[index]) / max_score if max_score else 0.0
        source = _source_from_document(doc, normalized_score)
        ranked.append((_document_identity(doc, source), doc, source))
    return ranked


def _load_bm25_documents(settings: Settings) -> list[Document]:
    path = bm25_index_path(settings)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload.get("documents") or []
    return [
        Document(page_content=str(row.get("page_content") or ""), metadata=dict(row.get("metadata") or {}))
        for row in rows
        if str(row.get("page_content") or "").strip()
    ]


def _fuse_ranked_results(
    ranked_sets: list[list[tuple[str, Document, RetrievedSource]]],
    top_k: int,
    rrf_k: int = 60,
) -> tuple[list[Document], list[RetrievedSource]]:
    scores: dict[str, float] = {}
    best_docs: dict[str, Document] = {}
    best_sources: dict[str, RetrievedSource] = {}
    for ranked in ranked_sets:
        seen: set[str] = set()
        for rank, (doc_id, doc, source) in enumerate(ranked, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in best_sources or source.score > best_sources[doc_id].score:
                best_docs[doc_id] = doc
                best_sources[doc_id] = source
    if not scores:
        return [], []
    max_score = max(scores.values()) or 1.0
    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
    docs: list[Document] = []
    sources: list[RetrievedSource] = []
    for doc_id in ranked_ids:
        docs.append(best_docs[doc_id])
        source = best_sources[doc_id]
        sources.append(
            RetrievedSource(
                source=source.source,
                page=source.page,
                score=float(scores[doc_id] / max_score),
                snippet=source.snippet,
            )
        )
    return docs, sources


def _source_from_document(doc: Document, score: float) -> RetrievedSource:
    metadata = doc.metadata or {}
    page = metadata.get("page_start")
    if not isinstance(page, int):
        raw_page = metadata.get("page")
        page = int(raw_page) + 1 if isinstance(raw_page, int) else None
    return RetrievedSource(
        source=metadata.get("source") or metadata.get("file_name") or "local document",
        page=page,
        score=score,
        snippet=doc.page_content[:350].replace("\n", " ").strip(),
    )


def _document_identity(doc: Document, source: RetrievedSource) -> str:
    metadata = doc.metadata or {}
    return "::".join(
        str(part)
        for part in (
            metadata.get("source") or source.source,
            metadata.get("page_start") or metadata.get("page") or source.page or "na",
            metadata.get("chunk_index") or "na",
        )
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _parse_query_expansion_json(content: str) -> dict[str, Any]:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        content = match.group(0)
    return json.loads(content)


def _fallback_expanded_queries(query: str) -> list[str]:
    return [
        f"{query} Cobb County ordinance section requirement",
        f"{query} fire marshal building permit inspection code",
        f"What code concept or permit requirement governs: {query}",
        f"Cobb County Georgia building fire code technical standard {query}",
    ]
