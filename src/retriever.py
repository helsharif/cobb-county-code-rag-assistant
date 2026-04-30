"""Retriever utilities for the local Chroma vector database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import Settings, get_embeddings, get_settings


@dataclass
class RetrievedSource:
    source: str
    page: int | None
    score: float
    snippet: str


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    """Connect to the persisted Chroma vectorstore."""

    settings = settings or get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.vectorstore_dir),
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def vectorstore_exists(path: Path | None = None) -> bool:
    """Return whether a persisted vector index appears to exist."""

    settings = get_settings()
    vectorstore_path = path or settings.vectorstore_dir
    return vectorstore_path.exists() and any(vectorstore_path.iterdir())


def search_documents(query: str, k: int | None = None) -> tuple[list[Document], list[RetrievedSource]]:
    """Retrieve documents and normalized relevance metadata."""

    settings = get_settings()
    if not vectorstore_exists(settings.vectorstore_dir):
        return [], []

    vectorstore = get_vectorstore(settings)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k or settings.retriever_k)

    docs: list[Document] = []
    sources: list[RetrievedSource] = []
    for doc, score in results:
        docs.append(doc)
        page = doc.metadata.get("page")
        sources.append(
            RetrievedSource(
                source=doc.metadata.get("source") or doc.metadata.get("file_name") or "local document",
                page=int(page) + 1 if isinstance(page, int) else None,
                score=float(score),
                snippet=doc.page_content[:350].replace("\n", " ").strip(),
            )
        )
    return docs, sources


def has_sufficient_retrieval(sources: list[RetrievedSource]) -> bool:
    """Use a simple threshold to decide if local retrieval is strong enough."""

    settings = get_settings()
    if not sources:
        return False
    best_score = max(source.score for source in sources)
    return best_score >= settings.min_relevance_score
