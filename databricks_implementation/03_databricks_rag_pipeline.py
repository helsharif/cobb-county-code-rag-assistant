# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Cobb County Building & Fire Code RAG Pipeline
# MAGIC
# MAGIC Intended to more closely follow the originally developed RAG app.

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Packages

# COMMAND ----------

# Install packages needed for Databricks AI Search and optional local-style reranking.
# If sentence-transformers causes issues on your trial compute, you can disable reranking later.

%pip install databricks-ai-search sentence-transformers
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

# Configuration for the Databricks version of the local hybrid-search RAG pipeline.

CHUNKS_TABLE = "cobb_rag.fire_code.document_chunks"
INDEX_NAME = "cobb_rag.fire_code.document_chunks_hybrid_index"

RETRIEVER_K = 15

# Match local app context expansion behavior:
# retrieved chunk plus chunk_index - 1 and chunk_index + 1.
CONTEXT_NEIGHBOR_WINDOW = 1
CONTEXT_MAX_EXPANDED_DOCS = 8
CONTEXT_MAX_CHARS = 18000
MAX_CHARS_PER_CONTEXT_BLOCK = 6000

# Match local app reranker settings.
# Set to False if sentence-transformers is too slow or unstable.
RERANKER_ENABLED = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_TOP_N = 8
RERANKER_BATCH_SIZE = 16

# COMMAND ----------

import os

os.environ["HF_TOKEN"] = dbutils.secrets.get(
    scope="cobb-county-rag",
    key="HF_TOKEN"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Connect to Databricks AI Search

# COMMAND ----------

# Connect to the Databricks AI Search hybrid index.
# This replaces the local app's Chroma + BM25 retrieval layer.

from databricks.ai_search.client import AISearchClient

client = AISearchClient(disable_notice=True)
index = client.get_index(index_name=INDEX_NAME)

print(f"Connected to index: {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Helper Functions

# COMMAND ----------

# Convert Databricks AI Search responses into plain Python dictionaries.

def parse_search_results(results):
    columns = [column["name"] for column in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]

    parsed = []

    for rank, row in enumerate(rows, start=1):
        item = dict(zip(columns, row))
        item["retrieval_rank"] = rank

        score = (
            item.get("score")
            or item.get("_score")
            or item.get("similarity_score")
            or item.get("relevance_score")
            or 0.0
        )

        try:
            item["score"] = float(score)
        except Exception:
            item["score"] = 0.0

        parsed.append(item)

    return parsed

# COMMAND ----------

# Run Databricks hybrid search.
# query_type="HYBRID" combines semantic vector search with keyword search.

def search_hybrid(query, k=RETRIEVER_K):
    results = index.similarity_search(
        query_text=query,
        columns=[
            "id",
            "text",
            "doc_id",
            "source_folder",
            "source_path",
            "source_file",
            "page_start",
            "page_end",
            "chunk_index",
            "parser",
        ],
        num_results=k,
        query_type="HYBRID",
    )

    return parse_search_results(results)

# COMMAND ----------

# Neighbor Context Expansion
# Expand each retrieved chunk with same-document neighbors:
# chunk_index - 1, chunk_index, chunk_index + 1.
#
# This mirrors the local app:
# - same source_path only
# - no page-level fallback
# - no cross-parser expansion
# - preserve raw retrieval rank globally
# - sort only within each retrieved group by chunk_index
# - dedupe repeated chunks
# - apply context budget after retrieval-priority ordering

import hashlib
from pyspark.sql import functions as F

def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def text_hash(text):
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def expand_with_neighbor_chunks(hits):
    expanded_pairs = []
    seen = set()

    for hit in hits:
        source_path = hit.get("source_path")
        chunk_index = safe_int(hit.get("chunk_index"))

        if source_path is None or chunk_index is None:
            fallback = dict(hit)
            fallback["expansion_type"] = "original_chunk_no_chunk_index"
            expanded_pairs.append(fallback)
            continue

        min_chunk = chunk_index - CONTEXT_NEIGHBOR_WINDOW
        max_chunk = chunk_index + CONTEXT_NEIGHBOR_WINDOW

        neighbor_rows = (
            spark.table(CHUNKS_TABLE)
            .where(F.col("source_path") == source_path)
            .where(F.col("chunk_index").between(min_chunk, max_chunk))
            .orderBy("chunk_index")
            .collect()
        )

        if not neighbor_rows:
            fallback = dict(hit)
            fallback["expansion_type"] = "original_chunk_no_neighbors_found"
            expanded_pairs.append(fallback)
            continue

        for row in neighbor_rows:
            item = row.asDict()
            item["retrieval_rank"] = hit["retrieval_rank"]
            item["score"] = hit.get("score", 0.0)
            item["anchor_chunk_index"] = chunk_index

            row_chunk_index = safe_int(item.get("chunk_index"))

            item["expansion_type"] = (
                "retrieved_chunk"
                if row_chunk_index == chunk_index
                else "neighbor_chunk"
            )

            identity = "::".join([
                str(item.get("doc_id")),
                str(item.get("source_path")),
                str(item.get("chunk_index")),
            ])

            hash_identity = text_hash(item.get("text"))

            if identity in seen or hash_identity in seen:
                continue

            seen.add(identity)
            seen.add(hash_identity)
            expanded_pairs.append(item)

    final_context = []
    total_chars = 0

    for item in expanded_pairs:
        if len(final_context) >= CONTEXT_MAX_EXPANDED_DOCS:
            break

        remaining = CONTEXT_MAX_CHARS - total_chars

        if remaining <= 0:
            break

        text = item.get("text") or ""
        trimmed_text = text[: min(len(text), remaining, MAX_CHARS_PER_CONTEXT_BLOCK)]

        if not trimmed_text.strip():
            continue

        output_item = dict(item)
        output_item["text"] = trimmed_text

        if trimmed_text != text:
            output_item["truncated_by_context_budget"] = True

        final_context.append(output_item)
        total_chars += len(trimmed_text)

    return final_context

# COMMAND ----------

# Optional CrossEncoder reranker.
# This mirrors the local app's reranking step after context expansion.

import math
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def sigmoid(score):
    try:
        return 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


def passage_text(item):
    title = item.get("source_file") or item.get("source_path") or ""
    folder = item.get("source_folder") or ""
    text = item.get("text") or ""

    return f"{title}\n{folder}\n{text}".strip()


def rerank_context(query, context_items):
    if not RERANKER_ENABLED or len(context_items) <= 1:
        return context_items

    try:
        model = get_cross_encoder()
        pairs = [(query, passage_text(item)) for item in context_items]
        scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
    except Exception as exc:
        print(f"Reranker failed; preserving retrieval order: {exc}")
        return context_items

    scored = []

    for index_position, (item, raw_score) in enumerate(zip(context_items, scores), start=1):
        rerank_score = float(raw_score)

        reranked_item = dict(item)
        reranked_item["cross_encoder_model"] = RERANKER_MODEL
        reranked_item["cross_encoder_score"] = rerank_score
        reranked_item["pre_rerank_position"] = index_position
        reranked_item["pre_rerank_score"] = item.get("score", 0.0)
        reranked_item["score"] = max(float(item.get("score", 0.0)), sigmoid(rerank_score))

        scored.append((rerank_score, reranked_item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [item for _, item in scored[:RERANKER_TOP_N]]

# COMMAND ----------

# Format context in the same style as the local app's answer prompt.

def format_local_context(context_items):
    blocks = []
    remaining_chars = CONTEXT_MAX_CHARS

    for index_position, item in enumerate(context_items, start=1):
        if remaining_chars <= 0:
            break

        source = item.get("source_file") or item.get("source_path") or "local document"
        page = item.get("page_start")
        score = float(item.get("score", 0.0))
        expansion_type = item.get("expansion_type")
        chunk_index = item.get("chunk_index")

        page_text = f", page {int(page)}" if page is not None else ""
        expansion_text = f" | {expansion_type}" if expansion_type else ""
        chunk_text = f" | chunk={int(chunk_index)}" if chunk_index is not None else ""

        text = item.get("text") or ""
        text = text[:remaining_chars]

        block = (
            f"[Local {index_position}] {source}{page_text} "
            f"| relevance={score:.2f}{expansion_text}{chunk_text}\n"
            f"{text}"
        )

        blocks.append(block)
        remaining_chars -= len(text)

    return "\n\n".join(blocks)

# COMMAND ----------

# Create source labels like the local app displays under an answer.

def source_labels(context_items, max_sources=8):
    labels = []
    seen = set()

    for item in context_items:
        source = item.get("source_file") or item.get("source_path") or "local document"
        page = item.get("page_start")
        score = float(item.get("score", 0.0))

        page_text = f", page {int(page)}" if page is not None else ""
        label = f"{source}{page_text} (score {score:.2f})"

        dedupe_key = (source, page)

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        labels.append(label)

        if len(labels) >= max_sources:
            break

    return labels
    return "\n\n".join(blocks)

# COMMAND ----------

# MAGIC %md
# MAGIC # Full Retrieval Pipeline

# COMMAND ----------

# Full Databricks version of the local app's retrieval pipeline:
# AI Search Hybrid -> neighbor expansion -> CrossEncoder rerank -> formatted evidence.

def retrieve_for_rag(query, k=RETRIEVER_K):
    raw_hits = search_hybrid(query, k=k)
    expanded_context = expand_with_neighbor_chunks(raw_hits)
    final_context_items = rerank_context(query, expanded_context)
    formatted_context = format_local_context(final_context_items)
    sources = source_labels(final_context_items)

    return {
        "query": query,
        "raw_hits": raw_hits,
        "expanded_context": expanded_context,
        "final_context_items": final_context_items,
        "formatted_context": formatted_context,
        "sources": sources,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC # Test One Query

# COMMAND ----------

result = retrieve_for_rag("When is a fire inspection required?")

print(result["formatted_context"][:6000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Final Context As Table

# COMMAND ----------

# Display final context blocks after expansion and reranking.

display(
    spark.createDataFrame(result["final_context_items"])
    .select(
        "retrieval_rank",
        "source_file",
        "page_start",
        "chunk_index",
        "expansion_type",
        "score",
        "text",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Several Test Queries

# COMMAND ----------

test_queries = [
    "When is a fire inspection required?",
    "What is required for tenant plan review?",
    "What are the requirements for consumer fireworks retail sales facilities?",
    "What does NFPA 241 require during construction?",
    "When is a certificate of occupancy required?",
    "What are the fire access road requirements?",
]

for query in test_queries:
    print("\n" + "#" * 100)
    print(f"QUERY: {query}")
    print("#" * 100)

    result = retrieve_for_rag(query)
    print(result["formatted_context"][:4000])

# COMMAND ----------

# MAGIC %md
# MAGIC # RAG Answering Component

# COMMAND ----------

CHAT_MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
# CHAT_MODEL_ENDPOINT = "databricks-gpt-oss-20b" # faster/cheaper testing

# COMMAND ----------

# Databricks exposes its model endpoints through an OpenAI-compatible API, 
# meaning the request/response shape looks like OpenAI’s Chat Completions API. 
# The openai Python package is just a convenient client library for making that compatible request.

%pip install openai
dbutils.library.restartPython()

# After restart, rerun your earlier retrieval/config cells, then proceed

# COMMAND ----------

from openai import OpenAI

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

llm_client = OpenAI(
    api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
    base_url=f"https://{workspace_url}/serving-endpoints",
)

response = llm_client.chat.completions.create(
    model=CHAT_MODEL_ENDPOINT,
    messages=[
        {"role": "user", "content": "Reply with only: working"}
    ],
    temperature=0,
    max_tokens=20,
)

print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Answer Prompt

# COMMAND ----------

ANSWER_SYSTEM_PROMPT = """
You answer Cobb County, Georgia building, fire, permit, inspection, and code questions.

Use ONLY the supplied local document evidence.
Do not use outside knowledge.
Do not infer from source titles, nearby topics, or general code knowledge.
Do not cite a requirement unless the exact requirement appears in the supplied evidence.

Rules:
1. Every factual claim must be directly supported by the supplied evidence.
2. If the evidence supports only part of the answer, answer only that part.
3. If the evidence does not contain the requested detail, say: "The retrieved context does not state that."
4. Include source names and page numbers inline when available.
5. For numeric, procedural, timing, fee, deadline, distance, height, or inspection requirements, quote or closely paraphrase the supporting phrase.
6. If the question is broad and the evidence shows multiple specific cases, do not collapse them into one general rule. List the specific cases shown in the evidence.
7. Keep the answer concise and practical.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def answer_question(question):
    retrieval = retrieve_for_rag(question)
    local_context = retrieval["formatted_context"]

    user_prompt = f"""
Question:
{question}

Local document evidence:
{local_context}

Grounded answer:
"""

    response = llm_client.chat.completions.create(
        model=CHAT_MODEL_ENDPOINT,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=800,
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": retrieval["sources"],
        "context": local_context,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Answering

# COMMAND ----------

qa = answer_question("When is a fire inspection required?")

print(qa["answer"])

print("\nSOURCES:")
for source in qa["sources"]:
    print("-", source)