"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv(ROOT_DIR / ".env")

ORIGINAL_COLLECTION_NAME = "cobb_code_docs_original"
DOCLING_COLLECTION_NAME = "cobb_code_docs_docling"
DOCLING_CHROMA_BM25_COLLECTION_NAME = "docling_chroma_bm25_hybrid"
DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME = "docling_chroma_bm25_expansion"
RAG_ANYTHING_LIGHTRAG_COLLECTION_NAME = "rag_anything_lightrag_option5"
OPTION_1_LABEL = "Option 1: PyPDF + Chromadb"
OPTION_2_LABEL = "Option 2: Docling + Chromadb"
OPTION_3_LABEL = "Option 3: Docling + Chroma + BM25 Hybrid Search"
OPTION_4_LABEL = "Option 4: Docling + Chroma + Query Expansion + BM25 Hybrid Search"
OPTION_5_LABEL = "Option 5: RAG-Anything + LightRAG KG Search"
COLLECTION_OPTIONS = {
    OPTION_1_LABEL: ORIGINAL_COLLECTION_NAME,
    OPTION_2_LABEL: DOCLING_COLLECTION_NAME,
    OPTION_3_LABEL: DOCLING_CHROMA_BM25_COLLECTION_NAME,
    OPTION_4_LABEL: DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME,
    OPTION_5_LABEL: RAG_ANYTHING_LIGHTRAG_COLLECTION_NAME,
}
COLLECTION_SLUGS = {
    ORIGINAL_COLLECTION_NAME: "pypdf_chroma",
    DOCLING_COLLECTION_NAME: "docling_chroma",
    DOCLING_CHROMA_BM25_COLLECTION_NAME: "docling_chroma_bm25_hybrid",
    DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME: "docling_chroma_bm25_expansion",
    RAG_ANYTHING_LIGHTRAG_COLLECTION_NAME: "rag_anything_lightrag_option5",
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


def _env_path(name: str, default: str) -> Path:
    raw_path = Path(os.getenv(name, default))
    return raw_path if raw_path.is_absolute() else ROOT_DIR / raw_path


def _option5_storage_dir() -> Path:
    return _env_path("RAG_ANYTHING_OPTION5_STORAGE_DIR", "vectorstore/rag_anything_lightrag_option5")


def _option5_child_dir(name: str, default_child: str) -> Path:
    storage_dir = _option5_storage_dir()
    raw_value = os.getenv(name)
    if not raw_value:
        return storage_dir / default_child
    configured_path = Path(raw_value)
    resolved_path = configured_path if configured_path.is_absolute() else ROOT_DIR / configured_path
    try:
        resolved_path.relative_to(storage_dir)
        return resolved_path
    except ValueError:
        if _env_bool("RAG_ANYTHING_OPTION5_ALLOW_EXTERNAL_DIRS"):
            return resolved_path
        return storage_dir / default_child


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
    context_store_dir: Path = ROOT_DIR / "context_store"
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
    context_expansion_enabled: bool = field(default_factory=lambda: _env_bool("CONTEXT_EXPANSION_ENABLED", "true"))
    context_expansion_mode: str = field(default_factory=lambda: _env_lower("CONTEXT_EXPANSION_MODE", "neighbors"))
    context_neighbor_window: int = field(default_factory=lambda: _env_int("CONTEXT_NEIGHBOR_WINDOW", "1"))
    context_max_expanded_docs: int = field(default_factory=lambda: _env_int("CONTEXT_MAX_EXPANDED_DOCS", "8"))
    context_max_chars: int = field(default_factory=lambda: _env_int("CONTEXT_MAX_CHARS", "18000"))
    reranker_enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", "true"))
    reranker_model: str = field(default_factory=lambda: _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"))
    reranker_top_n: int = field(default_factory=lambda: _env_int("RERANKER_TOP_N", "8"))
    reranker_batch_size: int = field(default_factory=lambda: _env_int("RERANKER_BATCH_SIZE", "16"))
    reranker_min_score: float = field(default_factory=lambda: _env_float("RERANKER_MIN_SCORE", "0.0"))
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
    option5_enabled: bool = field(default_factory=lambda: _env_bool("OPTION5_ENABLED", "true"))
    rag_anything_storage_dir: Path = field(default_factory=_option5_storage_dir)
    lightrag_working_dir: Path = field(default_factory=lambda: _option5_child_dir("LIGHTRAG_WORKING_DIR", "lightrag"))
    rag_anything_output_dir: Path = field(
        default_factory=lambda: _option5_child_dir("RAG_ANYTHING_OPTION5_OUTPUT_DIR", "processed")
    )
    rag_anything_parser: str = field(default_factory=lambda: _env_lower("RAG_ANYTHING_PARSER", "docling"))
    rag_anything_parse_method: str = field(default_factory=lambda: _env_lower("RAG_ANYTHING_PARSE_METHOD", "auto"))
    rag_anything_file_extensions: str = field(
        default_factory=lambda: _env_str("RAG_ANYTHING_FILE_EXTENSIONS", ".pdf,.html,.htm,.md,.txt,.docx,.xlsx")
    )
    rag_anything_enable_image_processing: bool = field(
        default_factory=lambda: _env_bool("RAG_ANYTHING_ENABLE_IMAGE_PROCESSING", "false")
    )
    rag_anything_enable_table_processing: bool = field(
        default_factory=lambda: _env_bool("RAG_ANYTHING_ENABLE_TABLE_PROCESSING", "true")
    )
    rag_anything_enable_equation_processing: bool = field(
        default_factory=lambda: _env_bool("RAG_ANYTHING_ENABLE_EQUATION_PROCESSING", "false")
    )
    rag_anything_section_relations: bool = field(
        default_factory=lambda: _env_bool("RAG_ANYTHING_SECTION_RELATIONS", "true")
    )
    rag_anything_max_concurrent_files: int = field(
        default_factory=lambda: _env_int("RAG_ANYTHING_MAX_CONCURRENT_FILES", "3")
    )
    rag_anything_docling_tables: bool = field(
        default_factory=lambda: _env_bool("RAG_ANYTHING_DOCLING_TABLES", "true")
    )
    rag_anything_retry_without_tables: bool = field(
        default_factory=lambda: _env_bool("RAG_ANYTHING_RETRY_WITHOUT_TABLES", "true")
    )
    rag_anything_docling_table_mode: str = field(
        default_factory=lambda: _env_lower("RAG_ANYTHING_DOCLING_TABLE_MODE", "fast")
    )
    enable_pre_extraction_chunking: bool = field(
        default_factory=lambda: _env_bool("ENABLE_PRE_EXTRACTION_CHUNKING", "true")
    )
    target_extraction_chunk_tokens: int = field(
        default_factory=lambda: _env_int("TARGET_EXTRACTION_CHUNK_TOKENS", "4000")
    )
    max_extraction_chunk_tokens: int = field(
        default_factory=lambda: _env_int("MAX_EXTRACTION_CHUNK_TOKENS", "7000")
    )
    save_oversized_item_audit: bool = field(
        default_factory=lambda: _env_bool("SAVE_OVERSIZED_ITEM_AUDIT", "true")
    )
    oversized_item_audit_dir: Path = field(
        default_factory=lambda: _option5_child_dir("OVERSIZED_ITEM_AUDIT_DIR", "audit/oversized_items")
    )
    lightrag_mode: str = field(default_factory=lambda: _env_lower("LIGHTRAG_MODE", "mix"))
    lightrag_top_k: int = field(default_factory=lambda: _env_int("LIGHTRAG_TOP_K", "20"))
    lightrag_chunk_top_k: int = field(default_factory=lambda: _env_int("LIGHTRAG_CHUNK_TOP_K", "30"))
    lightrag_max_entity_tokens: int = field(default_factory=lambda: _env_int("LIGHTRAG_MAX_ENTITY_TOKENS", "6000"))
    lightrag_max_relation_tokens: int = field(default_factory=lambda: _env_int("LIGHTRAG_MAX_RELATION_TOKENS", "8000"))
    lightrag_max_total_tokens: int = field(default_factory=lambda: _env_int("LIGHTRAG_MAX_TOTAL_TOKENS", "30000"))
    option5_context_max_chars: int = field(default_factory=lambda: _env_int("OPTION5_CONTEXT_MAX_CHARS", "60000"))
    option5_query_rewrite_enabled: bool = field(
        default_factory=lambda: _env_bool("OPTION5_QUERY_REWRITE_ENABLED", "true")
    )
    option5_query_rewrite_alternates: int = field(
        default_factory=lambda: _env_int("OPTION5_QUERY_REWRITE_ALTERNATES", "3")
    )
    lightrag_rerank_enabled: bool = field(default_factory=lambda: _env_bool("LIGHTRAG_RERANK_ENABLED", "true"))
    lightrag_rerank_model: str = field(
        default_factory=lambda: _env_str("LIGHTRAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    )
    lightrag_min_rerank_score: float = field(default_factory=lambda: _env_float("LIGHTRAG_MIN_RERANK_SCORE", "0.0"))
    lightrag_embedding_model: str = field(
        default_factory=lambda: _env_str("LIGHTRAG_EMBEDDING_MODEL", _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    )
    lightrag_llm_model: str = field(
        default_factory=lambda: _env_str("LIGHTRAG_LLM_MODEL", _env_str("OPENAI_MODEL", "gpt-4.1-mini"))
    )
    option5_web_fallback_enabled: bool = field(
        default_factory=lambda: _env_bool("OPTION5_WEB_FALLBACK_ENABLED", "true")
    )
    allowed_web_domains: str = field(
        default_factory=lambda: _env_str(
            "ALLOWED_WEB_DOMAINS",
            "cobbcounty.gov,www.cobbcounty.org,cobbcountyga.gov,dca.georgia.gov,oci.georgia.gov,rules.sos.ga.gov",
        )
    )

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
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
