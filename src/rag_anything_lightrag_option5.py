"""Option 5: RAG-Anything + LightRAG knowledge graph integration.

This module keeps the experimental RAG-Anything/LightRAG storage isolated from
the existing Chroma and BM25 assets. RAG-Anything is the primary ingestion
surface; LightRAG supplies the persistent graph/vector stores and mixed-mode
querying underneath it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Any

from langchain_core.documents import Document
from lightrag.utils import compute_mdhash_id

from src.config import Settings, get_settings
from src.retriever import RetrievedSource


logger = logging.getLogger(__name__)

MANIFEST_FILE = "option5_manifest.json"
SIDE_CAR_FILE = "option5_sources.jsonl"
OVERSIZED_MANIFEST_FILE = "oversized_items_manifest.jsonl"

_OPTION5_SEARCH_LOOP: asyncio.AbstractEventLoop | None = None
_OPTION5_SEARCH_THREAD: threading.Thread | None = None
_OPTION5_SEARCH_LOOP_LOCK = threading.Lock()
_RERANK_MODEL: Any | None = None
_RERANK_MODEL_NAME: str | None = None
_RERANK_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class Option5DocumentRecord:
    source: str
    source_path: str
    sha256: str
    size_bytes: int
    modified_at_utc: str
    processed_at_utc: str
    status: str
    error: str = ""


@dataclass
class Option5ChunkingStats:
    documents_processed: int = 0
    content_items_processed: int = 0
    oversized_items_detected: int = 0
    items_split: int = 0
    truncation_events_prevented: int = 0
    chunks_created: int = 0
    by_content_type: dict[str, int] | None = None

    def count_type(self, content_type: str) -> None:
        if self.by_content_type is None:
            self.by_content_type = {}
        normalized = content_type or "other"
        self.by_content_type[normalized] = self.by_content_type.get(normalized, 0) + 1


@dataclass(frozen=True)
class MarkdownSection:
    index: int
    level: int
    title: str
    line_number: int
    parent_index: int | None


@dataclass(frozen=True)
class MarkdownTable:
    index: int
    start_line: int
    end_line: int
    section_index: int | None
    preview: str


def option5_storage_exists(settings: Settings | None = None) -> bool:
    """Return whether the isolated Option 5 folder appears to contain an index."""

    settings = settings or get_settings()
    working_dir = settings.lightrag_working_dir
    manifest = option5_manifest_path(settings)
    if not settings.option5_enabled:
        return False
    return working_dir.exists() and any(working_dir.iterdir()) and manifest.exists()


def option5_manifest_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.rag_anything_storage_dir / MANIFEST_FILE


def option5_sidecar_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.rag_anything_storage_dir / SIDE_CAR_FILE


def build_option5_index(rebuild: bool = False, settings: Settings | None = None) -> int:
    """Build or update the isolated RAG-Anything + LightRAG Option 5 index."""

    settings = settings or get_settings()
    if not settings.option5_enabled:
        logger.info("Option 5 ingestion skipped because OPTION5_ENABLED=false.")
        return 0
    if rebuild:
        raise ValueError(
            "Option 5 rebuild is intentionally non-destructive. Delete "
            f"{settings.rag_anything_storage_dir} manually if you want a clean rebuild."
        )
    return asyncio.run(_build_option5_index_async(settings))


async def _build_option5_index_async(settings: Settings) -> int:
    settings.rag_anything_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
    settings.rag_anything_output_dir.mkdir(parents=True, exist_ok=True)
    if settings.save_oversized_item_audit:
        settings.oversized_item_audit_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_records(settings)
    documents = _discover_option5_documents(settings)
    if not documents:
        logger.warning("No Option 5 source documents found under %s.", settings.data_dir)
        _write_manifest(settings, records=list(existing.values()), indexed_count=0)
        return 0

    rag = await _create_rag_anything(settings)
    stats = getattr(rag, "_option5_chunking_stats", Option5ChunkingStats())
    records: dict[str, Option5DocumentRecord] = dict(existing)
    pending_paths: list[Path] = []
    for path in documents:
        record = _record_for_path(path, settings, status="pending")
        previous = existing.get(record.source_path)
        if previous and previous.sha256 == record.sha256 and previous.status == "processed":
            logger.info("Option 5 skipping unchanged document: %s.", record.source)
            records[record.source_path] = previous
            continue
        pending_paths.append(path)

    if pending_paths:
        max_workers = max(1, settings.rag_anything_max_concurrent_files)
        logger.info("Option 5 processing %s changed documents with max_concurrent_files=%s.", len(pending_paths), max_workers)
        semaphore = asyncio.Semaphore(max_workers)

        async def process_one(path: Path) -> tuple[str, Option5DocumentRecord, bool]:
            async with semaphore:
                record = _record_for_path(path, settings, status="pending")
                try:
                    logger.info("Option 5 processing with RAG-Anything: %s.", record.source)
                    await _process_option5_document(rag, path, settings)
                    if settings.rag_anything_section_relations:
                        await _inject_section_relationships(rag, path, settings)
                    processed = Option5DocumentRecord(
                        **{
                            **asdict(record),
                            "processed_at_utc": _utc_now(),
                            "status": "processed",
                            "error": "",
                        }
                    )
                    return record.source_path, processed, True
                except Exception as exc:
                    logger.exception("Option 5 failed to process %s: %s", path, exc)
                    failed = Option5DocumentRecord(
                        **{
                            **asdict(record),
                            "processed_at_utc": _utc_now(),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    return record.source_path, failed, False

        results = await asyncio.gather(*(process_one(path) for path in pending_paths))
    else:
        results = []

    processed_count = 0
    for source_path, record, processed in results:
        records[source_path] = record
        if processed:
            processed_count += 1
            stats.documents_processed += 1
    await _finalize_rag(rag)
    _write_source_sidecar(settings, records.values())
    _write_manifest(settings, records=list(records.values()), indexed_count=processed_count, chunking_stats=stats)
    _log_chunking_summary(stats)
    return processed_count


async def _process_option5_document(rag: Any, path: Path, settings: Settings) -> None:
    """Process one document, retrying Docling table failures without table structure."""

    parser_kwargs = _option5_parser_kwargs(settings, tables=settings.rag_anything_docling_tables)
    try:
        await rag.process_document_complete(
            file_path=str(path),
            output_dir=str(settings.rag_anything_output_dir),
            parse_method=settings.rag_anything_parse_method,
            **parser_kwargs,
        )
    except Exception:
        if (
            settings.rag_anything_parser == "docling"
            and settings.rag_anything_docling_tables
            and settings.rag_anything_retry_without_tables
        ):
            logger.exception(
                "Option 5 Docling parse failed for %s with table structure enabled; "
                "retrying this document with tables=False.",
                path.name,
            )
            retry_kwargs = _option5_parser_kwargs(settings, tables=False)
            await rag.process_document_complete(
                file_path=str(path),
                output_dir=str(settings.rag_anything_output_dir),
                parse_method=settings.rag_anything_parse_method,
                **retry_kwargs,
            )
            return
        raise


def _option5_parser_kwargs(settings: Settings, tables: bool) -> dict[str, Any]:
    if settings.rag_anything_parser != "docling":
        return {}
    return {
        "tables": tables,
        "table_mode": settings.rag_anything_docling_table_mode,
        "ocr": settings.docling_do_ocr,
    }


def search_option5(
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
) -> tuple[list[Document], list[RetrievedSource]]:
    """Return LightRAG mixed-mode context as local document evidence."""

    settings = settings or get_settings()
    if not option5_storage_exists(settings):
        return [], []
    return _run_option5_search(_search_option5_async(query, k=k, settings=settings))


def _run_option5_search(coro: Any) -> tuple[list[Document], list[RetrievedSource]]:
    """Run LightRAG search on a stable loop so its shared locks stay loop-local."""

    loop = _get_option5_search_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise


def _get_option5_search_loop() -> asyncio.AbstractEventLoop:
    global _OPTION5_SEARCH_LOOP, _OPTION5_SEARCH_THREAD

    with _OPTION5_SEARCH_LOOP_LOCK:
        if _OPTION5_SEARCH_LOOP and _OPTION5_SEARCH_LOOP.is_running():
            return _OPTION5_SEARCH_LOOP

        loop_ready = threading.Event()
        created_loop: dict[str, asyncio.AbstractEventLoop] = {}

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop["loop"] = loop
            loop_ready.set()
            loop.run_forever()

        _OPTION5_SEARCH_THREAD = threading.Thread(
            target=run_loop,
            name="option5-lightrag-search-loop",
            daemon=True,
        )
        _OPTION5_SEARCH_THREAD.start()
        loop_ready.wait()
        _OPTION5_SEARCH_LOOP = created_loop["loop"]
        return _OPTION5_SEARCH_LOOP


async def _search_option5_async(
    query: str,
    k: int | None,
    settings: Settings,
) -> tuple[list[Document], list[RetrievedSource]]:
    rag = await _create_rag_anything(settings)
    top_k = k or settings.lightrag_top_k
    try:
        result = await rag.aquery(
            query,
            mode=settings.lightrag_mode,
            only_need_context=True,
            top_k=top_k,
            chunk_top_k=settings.lightrag_chunk_top_k,
            max_entity_tokens=settings.lightrag_max_entity_tokens,
            max_relation_tokens=settings.lightrag_max_relation_tokens,
            max_total_tokens=settings.lightrag_max_total_tokens,
            enable_rerank=settings.lightrag_rerank_enabled,
        )
    except TypeError:
        # Older RAG-Anything versions accept only question/mode and pass fewer
        # LightRAG QueryParam knobs through their wrapper.
        result = await rag.aquery(query, mode=settings.lightrag_mode)
    finally:
        await _finalize_rag(rag)

    context = _coerce_context_text(result)
    if not context.strip():
        return [], []
    source_name = "Option 5 local RAG-Anything/LightRAG context"
    doc = Document(
        page_content=context,
        metadata={
            "source": source_name,
            "source_path": str(settings.lightrag_working_dir),
            "parser_type": "rag_anything",
            "backend": "lightrag",
            "collection_slug": "rag_anything_lightrag_option5",
            "retrieval_mode": settings.lightrag_mode,
            "source_type": "local",
        },
    )
    source = RetrievedSource(
        source=source_name,
        page=None,
        score=1.0,
        snippet=context[:350].replace("\n", " ").strip(),
    )
    return [doc], [source]


async def _create_rag_anything(settings: Settings):
    try:
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig
    except ImportError as exc:
        raise RuntimeError(
            "Option 5 requires RAG-Anything and LightRAG. Install with "
            "`pip install \"raganything[all]\" lightrag-hku`."
        ) from exc

    if not settings.openai_api_key:
        raise ValueError("Option 5 currently requires OPENAI_API_KEY or OPEN_API_KEY for LightRAG functions.")

    embedding_dim = _embedding_dimension(settings.lightrag_embedding_model)

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # RAG-Anything may pass multimodal-only kwargs such as image_data into
        # this LightRAG text completion hook. LightRAG's OpenAI wrapper forwards
        # unknown kwargs to AsyncCompletions.create(), which rejects image_data.
        kwargs.pop("image_data", None)
        kwargs.pop("image_caption", None)
        kwargs.pop("modalities", None)
        return openai_complete_if_cache(
            settings.lightrag_llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=settings.openai_api_key,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=settings.max_extraction_chunk_tokens if settings.enable_pre_extraction_chunking else 8192,
        func=partial(
            openai_embed.func,
            model=settings.lightrag_embedding_model,
            api_key=settings.openai_api_key,
        ),
    )
    lightrag = LightRAG(
        working_dir=str(settings.lightrag_working_dir),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        chunk_token_size=settings.target_extraction_chunk_tokens,
        chunk_overlap_token_size=min(400, max(0, settings.target_extraction_chunk_tokens // 10)),
        rerank_model_func=_build_rerank_func(settings),
        min_rerank_score=settings.lightrag_min_rerank_score,
    )
    await lightrag.initialize_storages()
    try:
        await initialize_pipeline_status()
    except TypeError:
        initialize_pipeline_status()

    config = RAGAnythingConfig(
        working_dir=str(settings.rag_anything_storage_dir),
        parser=settings.rag_anything_parser,
        parse_method=settings.rag_anything_parse_method,
        enable_image_processing=settings.rag_anything_enable_image_processing,
        enable_table_processing=settings.rag_anything_enable_table_processing,
        enable_equation_processing=settings.rag_anything_enable_equation_processing,
        max_concurrent_files=settings.rag_anything_max_concurrent_files,
    )
    rag = RAGAnything(config=config, lightrag=lightrag)
    ensure_result = await rag._ensure_lightrag_initialized()
    if not ensure_result or not ensure_result.get("success"):
        raise RuntimeError(f"RAG-Anything initialization failed: {ensure_result}")
    _install_pre_extraction_chunking(rag, settings)
    return rag


async def _finalize_rag(rag: Any) -> None:
    lightrag = getattr(rag, "lightrag", None) or getattr(rag, "lightrag_instance", None)
    finalize = getattr(lightrag, "finalize_storages", None)
    if finalize:
        await finalize()


def _build_rerank_func(settings: Settings):
    if not settings.lightrag_rerank_enabled:
        return None

    async def rerank(query: str, documents: list[str], top_n: int | None = None, **_: Any) -> list[dict[str, float]]:
        if not documents:
            return []
        model = await asyncio.to_thread(_get_rerank_model, settings.lightrag_rerank_model)
        pairs = [(query, document) for document in documents]
        scores = await asyncio.to_thread(model.predict, pairs)
        ranked = sorted(
            (
                {"index": index, "relevance_score": float(score)}
                for index, score in enumerate(scores)
            ),
            key=lambda item: item["relevance_score"],
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked

    return rerank


def _get_rerank_model(model_name: str):
    global _RERANK_MODEL, _RERANK_MODEL_NAME

    with _RERANK_MODEL_LOCK:
        if _RERANK_MODEL is not None and _RERANK_MODEL_NAME == model_name:
            return _RERANK_MODEL
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "LightRAG reranking requires sentence-transformers. "
                "Install project requirements or run `pip install sentence-transformers`."
            ) from exc

        logger.info("Loading LightRAG reranker model: %s", model_name)
        _RERANK_MODEL = CrossEncoder(model_name)
        _RERANK_MODEL_NAME = model_name
        return _RERANK_MODEL


def _install_pre_extraction_chunking(rag: Any, settings: Settings) -> None:
    """Patch RAG-Anything's multimodal conversion before LightRAG extraction."""

    if not settings.enable_pre_extraction_chunking:
        logger.info("Option 5 pre-extraction chunking disabled.")
        return
    stats = Option5ChunkingStats()
    setattr(rag, "_option5_chunking_stats", stats)
    for processor in getattr(rag, "modal_processors", {}).values():
        setattr(processor, "_option5_settings", settings)
        setattr(processor, "_option5_chunking_stats", stats)
        setattr(
            processor,
            "_convert_to_lightrag_chunks_type_aware",
            MethodType(_patched_convert_to_lightrag_chunks_type_aware, processor),
        )
        setattr(
            processor,
            "_store_multimodal_main_entities",
            MethodType(_patched_store_multimodal_main_entities, processor),
        )
        setattr(
            processor,
            "_batch_add_belongs_to_relations_type_aware",
            MethodType(_patched_batch_add_belongs_to_relations_type_aware, processor),
        )
    logger.info(
        "Option 5 pre-extraction chunking enabled: target=%s tokens, hard_max=%s tokens, audit=%s.",
        settings.target_extraction_chunk_tokens,
        settings.max_extraction_chunk_tokens,
        settings.oversized_item_audit_dir,
    )


def _patched_convert_to_lightrag_chunks_type_aware(self, multimodal_data_list, file_path: str, doc_id: str):
    chunks: dict[str, Any] = {}
    settings: Settings = getattr(self, "_option5_settings")
    stats: Option5ChunkingStats = getattr(self, "_option5_chunking_stats")
    file_ref = self._get_file_reference(file_path)
    for data in multimodal_data_list:
        description = data["description"]
        entity_info = data["entity_info"]
        content_type = _normalize_content_type(data["content_type"])
        original_item = data["original_item"]
        item_info = data.get("item_info") or {}
        original_item_id = _original_item_id(original_item, content_type)
        parent_multimodal_item_id = entity_info.get("entity_name") or original_item_id
        formatted = self._apply_chunk_template(content_type, original_item, description)
        tokens = _count_tokens(self.lightrag.tokenizer, formatted)
        stats.content_items_processed += 1
        stats.count_type(content_type)
        split_texts = _split_content_for_extraction(
            text=formatted,
            content_type=content_type,
            tokenizer=self.lightrag.tokenizer,
            target_tokens=settings.target_extraction_chunk_tokens,
            max_tokens=settings.max_extraction_chunk_tokens,
            metadata=_base_split_metadata(
                source_file=file_ref,
                content_type=content_type,
                original_item_id=original_item_id,
                parent_document_id=doc_id,
                parent_multimodal_item_id=parent_multimodal_item_id,
                item_info=item_info,
                original_item=original_item,
            ),
        )
        if tokens > settings.max_extraction_chunk_tokens:
            stats.oversized_items_detected += 1
            stats.items_split += 1
            stats.truncation_events_prevented += 1
            _audit_oversized_item(
                settings=settings,
                source_file=file_ref,
                content_type=content_type,
                original_item_id=original_item_id,
                original_text=formatted,
                original_tokens=tokens,
                split_texts=split_texts,
                metadata=data,
            )
            logger.info(
                "Option 5 split oversized item source=%s type=%s item=%s tokens=%s threshold=%s chunks=%s.",
                file_ref,
                content_type,
                original_item_id,
                tokens,
                settings.max_extraction_chunk_tokens,
                len(split_texts),
            )
        else:
            logger.debug(
                "Option 5 item within extraction limit source=%s type=%s item=%s tokens=%s threshold=%s.",
                file_ref,
                content_type,
                original_item_id,
                tokens,
                settings.max_extraction_chunk_tokens,
            )
        chunk_ids: list[str] = []
        total_parts = len(split_texts)
        for part_index, part_text in enumerate(split_texts, start=1):
            part_metadata = _base_split_metadata(
                source_file=file_ref,
                content_type=content_type,
                original_item_id=original_item_id,
                parent_document_id=doc_id,
                parent_multimodal_item_id=parent_multimodal_item_id,
                item_info=item_info,
                original_item=original_item,
            )
            part_metadata.update(
                {
                    "chunk_part": part_index,
                    "chunk_part_total": total_parts,
                    "estimated_tokens": _count_tokens(self.lightrag.tokenizer, part_text),
                }
            )
            chunk_content = _format_split_chunk(part_text, part_metadata)
            chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")
            chunk_ids.append(chunk_id)
            chunks[chunk_id] = {
                "content": chunk_content,
                "tokens": _count_tokens(self.lightrag.tokenizer, chunk_content),
                "full_doc_id": doc_id,
                "chunk_order_index": int(data["chunk_order_index"]) * 1000 + part_index - 1,
                "file_path": file_ref,
                "llm_cache_list": [],
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": content_type,
                "page_idx": item_info.get("page_idx", 0),
                "option5_split_metadata": part_metadata,
            }
            logger.debug("Option 5 split chunk metadata: %s", part_metadata)
            stats.chunks_created += 1
        data["_option5_chunk_ids"] = chunk_ids
        data["_option5_original_item_id"] = original_item_id
        data["_option5_original_tokens"] = tokens
    logger.info("Option 5 converted %s multimodal items into %s extraction-safe chunks.", len(multimodal_data_list), len(chunks))
    return chunks


async def _patched_store_multimodal_main_entities(self, multimodal_data_list, lightrag_chunks, file_path: str, doc_id: str = None):
    if not multimodal_data_list:
        return
    file_ref = self._get_file_reference(file_path)
    entities_to_store: dict[str, Any] = {}
    doc_entity_name = f"Document: {Path(file_ref).name}"
    try:
        await self.lightrag.chunk_entity_relation_graph.upsert_node(
            doc_entity_name,
            {
                "entity_id": doc_entity_name,
                "entity_type": "source_document",
                "description": f"Source document for Option 5 ingestion: {Path(file_ref).name}",
                "source_id": doc_id or file_ref,
                "file_path": file_ref,
                "created_at": int(datetime.now(timezone.utc).timestamp()),
            },
        )
    except Exception as exc:
        logger.warning("Could not upsert Option 5 source document node: %s", exc)
    for data in multimodal_data_list:
        entity_info = data["entity_info"]
        entity_name = entity_info["entity_name"]
        source_id = (data.get("_option5_chunk_ids") or [""])[0]
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        entities_to_store[entity_id] = {
            "entity_name": entity_name,
            "entity_type": entity_info.get("entity_type", data.get("content_type", "other")),
            "content": entity_info.get("summary", data.get("description", "")),
            "source_id": source_id,
            "file_path": file_ref,
            "parent_document_id": doc_id,
            "original_item_id": data.get("_option5_original_item_id", ""),
        }
        try:
            await self.lightrag.chunk_entity_relation_graph.upsert_node(
                entity_name,
                {
                    "entity_id": entity_name,
                    "entity_type": entities_to_store[entity_id]["entity_type"],
                    "description": entities_to_store[entity_id]["content"],
                    "source_id": source_id,
                    "file_path": file_ref,
                    "parent_document_id": doc_id,
                    "original_item_id": data.get("_option5_original_item_id", ""),
                    "created_at": int(datetime.now(timezone.utc).timestamp()),
                },
            )
            await self.lightrag.chunk_entity_relation_graph.upsert_edge(
                entity_name,
                doc_entity_name,
                {
                    "description": f"{entity_name} belongs to source document {Path(file_ref).name}",
                    "keywords": "belongs_to,source_document,parent_document",
                    "source_id": source_id or doc_id or file_ref,
                    "weight": 10.0,
                    "file_path": file_ref,
                },
            )
        except Exception as exc:
            logger.warning("Could not create Option 5 parent document relation: %s", exc)
    if entities_to_store:
        await self.lightrag.entities_vdb.upsert(entities_to_store)
        await self.lightrag.entities_vdb.index_done_callback()
        if doc_id and self.lightrag.full_entities:
            await self._store_multimodal_entities_to_full_entities(entities_to_store, doc_id)


async def _patched_batch_add_belongs_to_relations_type_aware(self, chunk_results, multimodal_data_list):
    chunk_to_modal_entity: dict[str, str] = {}
    chunk_to_file_path: dict[str, str] = {}
    chunk_to_original_item: dict[str, str] = {}
    for data in multimodal_data_list:
        for chunk_id in data.get("_option5_chunk_ids") or []:
            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")
            chunk_to_original_item[chunk_id] = data.get("_option5_original_item_id", "")
    enhanced = []
    belongs_to_count = 0
    for maybe_nodes, maybe_edges in chunk_results:
        chunk_id = None
        for nodes_dict in maybe_nodes.values():
            if nodes_dict:
                chunk_id = nodes_dict[0].get("source_id")
                break
        if chunk_id and chunk_id in chunk_to_modal_entity:
            modal_entity_name = chunk_to_modal_entity[chunk_id]
            file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")
            original_item_id = chunk_to_original_item.get(chunk_id, "")
            for entity_name in maybe_nodes.keys():
                if entity_name != modal_entity_name:
                    edge_key = (entity_name, modal_entity_name)
                    maybe_edges.setdefault(edge_key, []).append(
                        {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to original content item {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in,original_content_item",
                            "source_id": chunk_id,
                            "weight": 10.0,
                            "file_path": file_path,
                            "original_item_id": original_item_id,
                        }
                    )
                    belongs_to_count += 1
        enhanced.append((maybe_nodes, maybe_edges))
    logger.info("Added %s belongs_to relations for split-aware multimodal entities.", belongs_to_count)
    return enhanced


async def _inject_section_relationships(rag: Any, source_path: Path, settings: Settings) -> None:
    """Add deterministic document/section/table relations from Docling Markdown."""

    markdown_path = _find_processed_markdown(source_path, settings)
    if not markdown_path:
        logger.info("Option 5 section relation injection skipped; no Markdown output found for %s.", source_path.name)
        return

    text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    sections = _parse_markdown_sections(text)
    tables = _parse_markdown_tables(text, sections)
    if not sections and not tables:
        logger.info("Option 5 found no heading/table structure in %s.", markdown_path)
        return

    lightrag = getattr(rag, "lightrag", None) or getattr(rag, "lightrag_instance", None)
    graph = getattr(lightrag, "chunk_entity_relation_graph", None)
    if graph is None:
        logger.info("Option 5 section relation injection skipped; LightRAG graph is unavailable.")
        return

    try:
        file_ref = str(source_path.relative_to(settings.root_dir))
    except ValueError:
        file_ref = str(source_path)
    doc_node = _graph_node_name("Document", source_path.stem, file_ref)
    source_id = compute_mdhash_id(file_ref, prefix="section-map-")
    created_at = int(datetime.now(timezone.utc).timestamp())
    await graph.upsert_node(
        doc_node,
        {
            "entity_id": doc_node,
            "entity_type": "source_document",
            "description": f"Source document parsed by Option 5: {source_path.name}",
            "source_id": source_id,
            "file_path": file_ref,
            "created_at": created_at,
        },
    )

    section_nodes: dict[int, str] = {}
    entities_to_store: dict[str, Any] = {}
    previous_section_node: str | None = None
    for section in sections:
        node_name = _graph_node_name("Section", f"{section.index}. {section.title}", file_ref)
        section_nodes[section.index] = node_name
        description = f"Heading level {section.level} section in {source_path.name}: {section.title}"
        await graph.upsert_node(
            node_name,
            {
                "entity_id": node_name,
                "entity_type": "document_section",
                "description": description,
                "source_id": source_id,
                "file_path": file_ref,
                "section_title": section.title,
                "heading_level": section.level,
                "line_number": section.line_number,
                "created_at": created_at,
            },
        )
        entities_to_store[compute_mdhash_id(node_name, prefix="ent-")] = {
            "entity_name": node_name,
            "entity_type": "document_section",
            "content": description,
            "source_id": source_id,
            "file_path": file_ref,
        }
        parent_node = section_nodes.get(section.parent_index) if section.parent_index is not None else doc_node
        await graph.upsert_edge(
            parent_node,
            node_name,
            {
                "description": f"{parent_node} contains section {section.title}",
                "keywords": "contains,section_hierarchy,parent_section,document_structure",
                "source_id": source_id,
                "weight": 8.0,
                "file_path": file_ref,
            },
        )
        if previous_section_node:
            await graph.upsert_edge(
                previous_section_node,
                node_name,
                {
                    "description": f"{node_name} follows {previous_section_node} in document order",
                    "keywords": "next_section,document_order,follows,precedes",
                    "source_id": source_id,
                    "weight": 4.0,
                    "file_path": file_ref,
                },
            )
        previous_section_node = node_name

    for table in tables:
        table_node = _graph_node_name("Table", f"{table.index}. lines {table.start_line}-{table.end_line}", file_ref)
        parent_node = section_nodes.get(table.section_index) or doc_node
        description = f"Table in {source_path.name}, lines {table.start_line}-{table.end_line}:\n{table.preview}"
        await graph.upsert_node(
            table_node,
            {
                "entity_id": table_node,
                "entity_type": "document_table",
                "description": description,
                "source_id": source_id,
                "file_path": file_ref,
                "start_line": table.start_line,
                "end_line": table.end_line,
                "created_at": created_at,
            },
        )
        await graph.upsert_edge(
            parent_node,
            table_node,
            {
                "description": f"{parent_node} contains table {table.index}",
                "keywords": "contains_table,table_context,section_table,document_structure",
                "source_id": source_id,
                "weight": 9.0,
                "file_path": file_ref,
            },
        )
        entities_to_store[compute_mdhash_id(table_node, prefix="ent-")] = {
            "entity_name": table_node,
            "entity_type": "document_table",
            "content": description,
            "source_id": source_id,
            "file_path": file_ref,
        }

    entities_vdb = getattr(lightrag, "entities_vdb", None)
    if entities_to_store and entities_vdb is not None:
        await entities_vdb.upsert(entities_to_store)
        await entities_vdb.index_done_callback()
    logger.info(
        "Option 5 injected document structure relations for %s: sections=%s, tables=%s.",
        source_path.name,
        len(sections),
        len(tables),
    )


def _find_processed_markdown(source_path: Path, settings: Settings) -> Path | None:
    if source_path.suffix.lower() in {".md", ".txt"}:
        return source_path
    if not settings.rag_anything_output_dir.exists():
        return None
    candidates = [
        path
        for path in settings.rag_anything_output_dir.rglob("*.md")
        if path.stem.lower() == source_path.stem.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _parse_markdown_sections(text: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    stack: list[MarkdownSection] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = _clean_markdown_title(match.group(2))
        while stack and stack[-1].level >= level:
            stack.pop()
        parent_index = stack[-1].index if stack else None
        section = MarkdownSection(
            index=len(sections) + 1,
            level=level,
            title=title,
            line_number=line_number,
            parent_index=parent_index,
        )
        sections.append(section)
        stack.append(section)
    return sections


def _parse_markdown_tables(text: str, sections: list[MarkdownSection]) -> list[MarkdownTable]:
    lines = text.splitlines()
    tables: list[MarkdownTable] = []
    section_by_line = sorted(sections, key=lambda item: item.line_number)
    index = 0
    while index < len(lines):
        if not _looks_like_table_line(lines[index]):
            index += 1
            continue
        start = index
        block: list[str] = []
        while index < len(lines) and _looks_like_table_line(lines[index]):
            block.append(lines[index])
            index += 1
        if len(block) >= 2 and any(set(line.strip()) <= {"|", "-", ":", " "} for line in block[:3]):
            start_line = start + 1
            section_index = _nearest_section_index(start_line, section_by_line)
            preview = "\n".join(block[:8])
            tables.append(
                MarkdownTable(
                    index=len(tables) + 1,
                    start_line=start_line,
                    end_line=index,
                    section_index=section_index,
                    preview=preview,
                )
            )
    return tables


def _looks_like_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _nearest_section_index(line_number: int, sections: list[MarkdownSection]) -> int | None:
    current: int | None = None
    for section in sections:
        if section.line_number > line_number:
            break
        current = section.index
    return current


def _clean_markdown_title(title: str) -> str:
    title = re.sub(r"`([^`]+)`", r"\1", title)
    title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
    return re.sub(r"\s+", " ", title).strip(" #")


def _graph_node_name(kind: str, label: str, source_ref: str) -> str:
    clean_label = re.sub(r"\s+", " ", label).strip()
    if len(clean_label) > 96:
        clean_label = clean_label[:93].rstrip() + "..."
    digest = hashlib.sha1(f"{source_ref}|{kind}|{label}".encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{kind}: {clean_label} [{digest}]"


def _discover_option5_documents(settings: Settings) -> list[Path]:
    blocked_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
    extensions = {
        item.strip().lower()
        for item in settings.rag_anything_file_extensions.split(",")
        if item.strip() and item.strip().lower() not in blocked_extensions
    }
    return sorted(
        path
        for path in settings.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _normalize_content_type(content_type: str) -> str:
    normalized = str(content_type or "other").strip().lower()
    if normalized in {"img", "picture"}:
        return "image"
    if normalized in {"figure", "diagram", "chart"}:
        return normalized
    if "ocr" in normalized or "scan" in normalized:
        return "ocr"
    if normalized not in {"text", "table", "image", "figure", "diagram", "ocr", "equation", "generic"}:
        return "other" if normalized == "unknown" else normalized
    return normalized


def _original_item_id(original_item: Any, content_type: str) -> str:
    if isinstance(original_item, dict):
        for key in ("id", "item_id", "_content_list_index", "img_path", "table_id"):
            value = original_item.get(key)
            if value not in {None, ""}:
                return f"{content_type}-{value}"
    return compute_mdhash_id(str(original_item), prefix=f"{content_type}-")


def _count_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def _decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    try:
        return tokenizer.decode(tokens)
    except Exception:
        return ""


def _base_split_metadata(
    source_file: str,
    content_type: str,
    original_item_id: str,
    parent_document_id: str,
    parent_multimodal_item_id: str,
    item_info: dict[str, Any],
    original_item: Any,
) -> dict[str, Any]:
    title = ""
    page_number = item_info.get("page_idx")
    if isinstance(original_item, dict):
        caption = (
            original_item.get("table_caption")
            or original_item.get("image_caption")
            or original_item.get("img_caption")
            or original_item.get("caption")
            or ""
        )
        if isinstance(caption, list):
            title = " ".join(str(item) for item in caption if str(item).strip())
        else:
            title = str(caption or "")
        page_number = original_item.get("page_number", original_item.get("page_idx", page_number))
    return {
        "source_file": source_file,
        "content_type": content_type,
        "original_item_id": original_item_id,
        "chunk_part": 1,
        "chunk_part_total": 1,
        "page_number": page_number,
        "caption_title": title,
        "parent_document_id": parent_document_id,
        "parent_multimodal_item_id": parent_multimodal_item_id,
    }


def _format_split_chunk(text: str, metadata: dict[str, Any]) -> str:
    header = [
        "[Option 5 split chunk metadata]",
        f"source_file: {metadata.get('source_file', '')}",
        f"content_type: {metadata.get('content_type', '')}",
        f"original_item_id: {metadata.get('original_item_id', '')}",
        f"chunk_part: {metadata.get('chunk_part', '')}/{metadata.get('chunk_part_total', '')}",
        f"page_number: {metadata.get('page_number', '')}",
        f"caption_title: {metadata.get('caption_title', '')}",
        f"parent_document_id: {metadata.get('parent_document_id', '')}",
        f"parent_multimodal_item_id: {metadata.get('parent_multimodal_item_id', '')}",
        "",
    ]
    return "\n".join(header) + text


def _split_content_for_extraction(
    text: str,
    content_type: str,
    tokenizer: Any,
    target_tokens: int,
    max_tokens: int,
    metadata: dict[str, Any],
) -> list[str]:
    if _count_tokens(tokenizer, text) <= max_tokens:
        return [text]
    if content_type == "table":
        parts = _split_table_text(text, tokenizer, target_tokens, max_tokens)
    else:
        parts = _split_logical_text(text, tokenizer, target_tokens, max_tokens)
    enforced: list[str] = []
    for part in parts:
        enforced.extend(_enforce_hard_token_limit(part, tokenizer, max_tokens))
    return [part for part in enforced if part.strip()] or _enforce_hard_token_limit(text, tokenizer, max_tokens)


def _split_table_text(text: str, tokenizer: Any, target_tokens: int, max_tokens: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return _split_logical_text(text, tokenizer, target_tokens, max_tokens)
    header_lines = _table_header_lines(lines)
    body_lines = lines[len(header_lines) :] if header_lines else lines
    prefix = "\n".join(header_lines).strip()
    parts: list[str] = []
    current: list[str] = []
    for line in body_lines:
        candidate_lines = ([prefix] if prefix else []) + current + [line]
        candidate = "\n".join(candidate_lines)
        if current and _count_tokens(tokenizer, candidate) > target_tokens:
            parts.append("\n".join(([prefix] if prefix else []) + current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(([prefix] if prefix else []) + current))
    return parts


def _table_header_lines(lines: list[str]) -> list[str]:
    header: list[str] = []
    for line in lines[:8]:
        header.append(line)
        lowered = line.lower()
        if "|" in line or "\t" in line or "column" in lowered or "header" in lowered:
            if len(header) >= 1:
                break
        if len(header) >= 3:
            break
    return header[: max(1, min(len(header), 3))]


def _split_logical_text(text: str, tokenizer: Any, target_tokens: int, max_tokens: int) -> list[str]:
    blocks = _logical_blocks(text)
    parts: list[str] = []
    current: list[str] = []
    for block in blocks:
        candidate = "\n\n".join(current + [block])
        if current and _count_tokens(tokenizer, candidate) > target_tokens:
            parts.append("\n\n".join(current))
            current = [block]
        elif _count_tokens(tokenizer, block) > max_tokens:
            if current:
                parts.append("\n\n".join(current))
                current = []
            parts.extend(_split_sentences(block, tokenizer, target_tokens))
        else:
            current.append(block)
    if current:
        parts.append("\n\n".join(current))
    return parts


def _logical_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [text.strip()]


def _split_sentences(text: str, tokenizer: Any, target_tokens: int) -> list[str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    if len(sentences) <= 1:
        return _split_by_tokens(text, tokenizer, target_tokens)
    parts: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and _count_tokens(tokenizer, candidate) > target_tokens:
            parts.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return parts


def _enforce_hard_token_limit(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    if _count_tokens(tokenizer, text) <= max_tokens:
        return [text]
    return _split_by_tokens(text, tokenizer, max_tokens)


def _split_by_tokens(text: str, tokenizer: Any, token_limit: int) -> list[str]:
    try:
        tokens = tokenizer.encode(text)
    except Exception:
        char_limit = max(1000, token_limit * 4)
        return [text[start : start + char_limit] for start in range(0, len(text), char_limit)]
    parts: list[str] = []
    for start in range(0, len(tokens), token_limit):
        decoded = _decode_tokens(tokenizer, tokens[start : start + token_limit]).strip()
        if decoded:
            parts.append(decoded)
    return parts


def _audit_oversized_item(
    settings: Settings,
    source_file: str,
    content_type: str,
    original_item_id: str,
    original_text: str,
    original_tokens: int,
    split_texts: list[str],
    metadata: dict[str, Any],
) -> None:
    if not settings.save_oversized_item_audit:
        return
    settings.oversized_item_audit_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", original_item_id)[:120]
    digest = hashlib.sha1(original_text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    base_name = f"{safe_id}_{digest}"
    full_path = settings.oversized_item_audit_dir / f"{base_name}.txt"
    full_path.write_text(original_text, encoding="utf-8")
    split_dir = settings.oversized_item_audit_dir / f"{base_name}_parts"
    split_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[str] = []
    for index, part in enumerate(split_texts, start=1):
        part_path = split_dir / f"part_{index:03d}.txt"
        part_path.write_text(part, encoding="utf-8")
        part_paths.append(str(part_path))
    record = {
        "source_file": source_file,
        "content_type": content_type,
        "original_item_id": original_item_id,
        "original_estimated_tokens": original_tokens,
        "threshold_tokens": settings.max_extraction_chunk_tokens,
        "target_tokens": settings.target_extraction_chunk_tokens,
        "would_have_been_truncated": True,
        "chunks_created": len(split_texts),
        "full_text_path": str(full_path),
        "split_part_paths": part_paths,
        "metadata_keys": sorted(str(key) for key in metadata.keys()),
        "created_at_utc": _utc_now(),
    }
    manifest_path = settings.oversized_item_audit_dir / OVERSIZED_MANIFEST_FILE
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Option 5 oversized item audit saved: %s", full_path)


def _log_chunking_summary(stats: Option5ChunkingStats) -> None:
    logger.info(
        "Option 5 pre-extraction chunking summary: documents=%s, content_items=%s, "
        "oversized=%s, split=%s, truncation_events_prevented=%s, chunks_created=%s, by_type=%s.",
        stats.documents_processed,
        stats.content_items_processed,
        stats.oversized_items_detected,
        stats.items_split,
        stats.truncation_events_prevented,
        stats.chunks_created,
        stats.by_content_type or {},
    )


def _load_existing_records(settings: Settings) -> dict[str, Option5DocumentRecord]:
    path = option5_sidecar_path(settings)
    if not path.exists():
        return {}
    records: dict[str, Option5DocumentRecord] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = Option5DocumentRecord(**json.loads(line))
                records[record.source_path] = record
            except Exception:
                logger.warning("Skipping invalid Option 5 source record in %s.", path)
    return records


def _write_source_sidecar(settings: Settings, records: Any) -> None:
    path = option5_sidecar_path(settings)
    with path.open("w", encoding="utf-8") as file:
        for record in sorted(records, key=lambda item: item.source_path):
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _write_manifest(
    settings: Settings,
    records: list[Option5DocumentRecord],
    indexed_count: int,
    chunking_stats: Option5ChunkingStats | None = None,
) -> None:
    processed = [record for record in records if record.status == "processed"]
    failed = [record for record in records if record.status == "failed"]
    payload = {
        "pipeline": "rag_anything_lightrag_option5",
        "storage_dir": str(settings.rag_anything_storage_dir),
        "lightrag_working_dir": str(settings.lightrag_working_dir),
        "processed_output_dir": str(settings.rag_anything_output_dir),
        "parser": settings.rag_anything_parser,
        "parse_method": settings.rag_anything_parse_method,
        "file_extensions": settings.rag_anything_file_extensions,
        "image_processing": settings.rag_anything_enable_image_processing,
        "table_processing": settings.rag_anything_enable_table_processing,
        "equation_processing": settings.rag_anything_enable_equation_processing,
        "section_relations": settings.rag_anything_section_relations,
        "max_concurrent_files": settings.rag_anything_max_concurrent_files,
        "retrieval_mode": settings.lightrag_mode,
        "embedding_model": settings.lightrag_embedding_model,
        "llm_model": settings.lightrag_llm_model,
        "pre_extraction_chunking": {
            "enabled": settings.enable_pre_extraction_chunking,
            "target_tokens": settings.target_extraction_chunk_tokens,
            "max_tokens": settings.max_extraction_chunk_tokens,
            "save_audit": settings.save_oversized_item_audit,
            "audit_dir": str(settings.oversized_item_audit_dir),
        },
        "chunking_summary": asdict(chunking_stats) if chunking_stats else {},
        "indexed_this_run": indexed_count,
        "processed_documents": len(processed),
        "failed_documents": len(failed),
        "updated_at_utc": _utc_now(),
    }
    with option5_manifest_path(settings).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _record_for_path(path: Path, settings: Settings, status: str) -> Option5DocumentRecord:
    stat = path.stat()
    relative = str(path.relative_to(settings.root_dir))
    return Option5DocumentRecord(
        source=path.name,
        source_path=relative,
        sha256=_sha256(path),
        size_bytes=stat.st_size,
        modified_at_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        processed_at_utc="",
        status=status,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_context_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("context", "response", "answer", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def _embedding_dimension(model: str) -> int:
    normalized = model.lower()
    if "3-large" in normalized:
        return 3072
    if "ada-002" in normalized:
        return 1536
    return 1536


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
