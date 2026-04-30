"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Central configuration for ingestion, retrieval, and generation."""

    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    vectorstore_dir: Path = ROOT_DIR / "vectorstore"
    collection_name: str = "cobb_county_codes"

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    openai_api_key: str | None = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "3500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "500"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "5"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.30"))

    langsmith_tracing: str | None = os.getenv("LANGSMITH_TRACING")
    langsmith_project: str | None = os.getenv("LANGSMITH_PROJECT")
    langsmith_endpoint: str | None = os.getenv("LANGSMITH_ENDPOINT")
    langsmith_api_key: str | None = os.getenv("LANGSMITH_API_KEY")
    chroma_api_key: str | None = os.getenv("CHROMA_API_KEY")


def get_settings() -> Settings:
    """Return settings and mirror nonstandard env names expected by SDKs."""

    settings = Settings()
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.gemini_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", settings.langsmith_tracing)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    tiktoken_cache_dir = settings.root_dir / ".tiktoken_cache"
    tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(tiktoken_cache_dir))
    return settings


def get_embeddings(settings: Settings | None = None):
    """Create the configured embedding model."""

    settings = settings or get_settings()
    if settings.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini.")
        return GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model)

    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise ValueError("OPEN_API_KEY is required when EMBEDDING_PROVIDER=openai.")
    return OpenAIEmbeddings(model=settings.openai_embedding_model)


def get_chat_model(settings: Settings | None = None, temperature: float = 0.0):
    """Create the configured chat model."""

    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
        return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=temperature)

    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError("OPEN_API_KEY is required when LLM_PROVIDER=openai.")
    return ChatOpenAI(model=settings.openai_model, temperature=temperature)
