"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
load_dotenv(ROOT_DIR / ".env")

ORIGINAL_COLLECTION_NAME = "cobb_code_docs_original"
DOCLING_COLLECTION_NAME = "cobb_code_docs_docling"
DOCLING_CHROMA_BM25_COLLECTION_NAME = "docling_chroma_bm25_hybrid"
DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME = "docling_chroma_bm25_expansion"
OPTION_1_LABEL = "Option 1: PyPDF + Chromadb"
OPTION_2_LABEL = "Option 2: Docling + Chromadb"
OPTION_3_LABEL = "Option 3: Docling + Chroma + BM25 Hybrid Search"
OPTION_4_LABEL = "Option 4: Docling + Chroma + Query Expansion + BM25 Hybrid Search"
COLLECTION_OPTIONS = {
    OPTION_1_LABEL: ORIGINAL_COLLECTION_NAME,
    OPTION_2_LABEL: DOCLING_COLLECTION_NAME,
    OPTION_3_LABEL: DOCLING_CHROMA_BM25_COLLECTION_NAME,
    OPTION_4_LABEL: DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME,
}
COLLECTION_SLUGS = {
    ORIGINAL_COLLECTION_NAME: "pypdf_chroma",
    DOCLING_COLLECTION_NAME: "docling_chroma",
    DOCLING_CHROMA_BM25_COLLECTION_NAME: "docling_chroma_bm25_hybrid",
    DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME: "docling_chroma_bm25_expansion",
}
LEGACY_COLLECTION_LABELS = {
    "Original": OPTION_1_LABEL,
    "Docling": OPTION_2_LABEL,
}


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_optional(name: str) -> str | None:
    return os.getenv(name) or None


def _env_lower(name: str, default: str) -> str:
    return os.getenv(name, default).lower()


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _openai_api_key() -> str | None:
    return os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")


def _gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@dataclass(frozen=True)
class Settings:
    """Central configuration for ingestion, retrieval, and generation."""

    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    vectorstore_dir: Path = ROOT_DIR / "vectorstore"
    bm25_index_dir: Path = ROOT_DIR / "bm25_index"
    collection_name: str = field(default_factory=lambda: _env_str("CHROMA_COLLECTION_NAME", ORIGINAL_COLLECTION_NAME))

    llm_provider: str = field(default_factory=lambda: _env_lower("LLM_PROVIDER", "openai"))
    embedding_provider: str = field(default_factory=lambda: _env_lower("EMBEDDING_PROVIDER", "openai"))

    openai_api_key: str | None = field(default_factory=_openai_api_key)
    gemini_api_key: str | None = field(default_factory=_gemini_api_key)

    openai_model: str = field(default_factory=lambda: _env_str("OPENAI_MODEL", "gpt-4.1-mini"))
    gemini_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-1.5-flash"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: _env_str("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    )

    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", "3500"))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", "500"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", "32"))
    embedding_batch_delay_seconds: float = field(
        default_factory=lambda: _env_float("EMBEDDING_BATCH_DELAY_SECONDS", "1.0")
    )
    embedding_max_retries: int = field(default_factory=lambda: _env_int("EMBEDDING_MAX_RETRIES", "8"))
    retriever_k: int = field(default_factory=lambda: _env_int("RETRIEVER_K", "10"))
    min_relevance_score: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE_SCORE", "0.30"))
    eval_judge_model: str = field(default_factory=lambda: _env_str("EVAL_JUDGE_MODEL", "gpt-5.1"))
    eval_judge_delay_seconds: float = field(default_factory=lambda: _env_float("EVAL_JUDGE_DELAY_SECONDS", "1.0"))
    eval_judge_max_retries: int = field(default_factory=lambda: _env_int("EVAL_JUDGE_MAX_RETRIES", "8"))
    docling_accelerator_device: str = field(default_factory=lambda: _env_lower("DOCLING_ACCELERATOR_DEVICE", "auto"))
    docling_num_threads: int = field(default_factory=lambda: _env_int("DOCLING_NUM_THREADS", "4"))
    docling_do_ocr: bool = field(default_factory=lambda: _env_bool("DOCLING_DO_OCR"))
    docling_batch_size: int = field(default_factory=lambda: _env_int("DOCLING_BATCH_SIZE", "1"))
    docling_max_pages: int = field(default_factory=lambda: _env_int("DOCLING_MAX_PAGES", "250"))
    docling_page_chunk_size: int = field(default_factory=lambda: _env_int("DOCLING_PAGE_CHUNK_SIZE", "30"))
    docling_page_overlap: int = field(default_factory=lambda: _env_int("DOCLING_PAGE_OVERLAP", "5"))
    bm25_index_file: str = field(default_factory=lambda: _env_str("BM25_INDEX_FILE", "docling_bm25_chunks.json"))

    langsmith_tracing: str | None = field(default_factory=lambda: _env_optional("LANGSMITH_TRACING"))
    langsmith_project: str | None = field(default_factory=lambda: _env_optional("LANGSMITH_PROJECT"))
    langsmith_endpoint: str | None = field(default_factory=lambda: _env_optional("LANGSMITH_ENDPOINT"))
    langsmith_api_key: str | None = field(default_factory=lambda: _env_optional("LANGSMITH_API_KEY"))
    chroma_api_key: str | None = field(default_factory=lambda: _env_optional("CHROMA_API_KEY"))
    serpapi_api_key: str | None = field(default_factory=lambda: _env_optional("SERPAPI_API_KEY"))


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
    if settings.serpapi_api_key:
        os.environ.setdefault("SERPAPI_API_KEY", settings.serpapi_api_key)
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
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
