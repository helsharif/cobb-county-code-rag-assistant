"""
Cobb County RAG App on Databricks Apps.

Pipeline:
1. User asks a question in Streamlit.
2. Databricks AI Search runs hybrid retrieval over the indexed chunk table.
3. The app expands each retrieved chunk with same-document neighbor chunks.
4. A local CrossEncoder reranks the expanded context blocks.
5. A Databricks-hosted chat model writes a grounded answer using only retrieved evidence.
6. The UI displays the answer, source labels, and retrieved context.
"""

import hashlib
import math
import os

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.service.sql import StatementParameterListItem
from databricks.ai_search.client import AISearchClient


# Avoid noisy tokenizer multiprocessing warnings in hosted app logs.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# HF_TOKEN is injected by the Databricks App secret resource.
# Hugging Face / sentence-transformers reads this automatically.
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


# Values supplied by app.yaml / Databricks App resources.
CHUNKS_TABLE = os.getenv("CHUNKS_TABLE", "cobb_rag.fire_code.document_chunks")
INDEX_NAME = os.getenv("AI_SEARCH_INDEX_NAME", "cobb_rag.fire_code.document_chunks_hybrid_index")
CHAT_MODEL_ENDPOINT = os.getenv("CHAT_MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")


# Retrieval/context settings mirror the local app's hybrid mode.
RETRIEVER_K = 15
CONTEXT_NEIGHBOR_WINDOW = 1
CONTEXT_MAX_EXPANDED_DOCS = 8
CONTEXT_MAX_CHARS = 18000
MAX_CHARS_PER_CONTEXT_BLOCK = 6000


# CrossEncoder reranker settings.
RERANKER_ENABLED = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_TOP_N = 8
RERANKER_BATCH_SIZE = 16


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
7. When citing numbered checklist items, call them "items" unless the evidence explicitly labels them as sections.
8. Keep the answer concise and practical.
"""


def databricks_host():
    """Return the workspace host with https:// included."""
    host = os.getenv("DATABRICKS_HOST")

    if not host:
        raise RuntimeError("DATABRICKS_HOST is not set in the app environment.")

    if not host.startswith("http"):
        host = f"https://{host}"

    return host.rstrip("/")


def databricks_client_id():
    """Return the Databricks App service principal client id."""
    value = os.getenv("DATABRICKS_CLIENT_ID")

    if not value:
        raise RuntimeError("DATABRICKS_CLIENT_ID is not set in the app environment.")

    return value


def databricks_client_secret():
    """Return the Databricks App service principal client secret."""
    value = os.getenv("DATABRICKS_CLIENT_SECRET")

    if not value:
        raise RuntimeError("DATABRICKS_CLIENT_SECRET is not set in the app environment.")

    return value


@st.cache_resource
def get_workspace_client():
    """Create one Databricks SDK client using app service-principal credentials."""
    return WorkspaceClient(
        host=databricks_host(),
        client_id=databricks_client_id(),
        client_secret=databricks_client_secret(),
    )


@st.cache_resource
def get_search_index():
    """Connect to the Databricks AI Search hybrid index using app service-principal credentials."""
    client = AISearchClient(
        workspace_url=databricks_host(),
        service_principal_client_id=databricks_client_id(),
        service_principal_client_secret=databricks_client_secret(),
        disable_notice=True,
    )

    return client.get_index(index_name=INDEX_NAME)


@st.cache_resource
def get_cross_encoder():
    """Load and cache the CrossEncoder reranker once per app process."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def safe_int(value, default=None):
    """Convert Databricks/Spark numeric values to int safely."""
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    """Convert scores to float without crashing the app."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def text_hash(text):
    """Hash text so duplicate chunks can be removed after expansion."""
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def parse_search_results(results):
    """Normalize AI Search response rows into dictionaries."""
    columns = [column["name"] for column in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]

    parsed = []

    for rank, row in enumerate(rows, start=1):
        item = dict(zip(columns, row))
        item["retrieval_rank"] = rank

        # Score names can differ across SDK/API versions.
        score = (
            item.get("score")
            or item.get("_score")
            or item.get("similarity_score")
            or item.get("relevance_score")
            or 0.0
        )
        item["score"] = safe_float(score)

        parsed.append(item)

    return parsed


def search_hybrid(query, k=RETRIEVER_K):
    """Run Databricks hybrid search over the indexed chunks."""
    index = get_search_index()

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


def fetch_neighbor_rows(source_path, min_chunk, max_chunk):
    """Fetch same-document neighboring chunks from the Delta table."""
    if not WAREHOUSE_ID:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not set. Check the sql_warehouse app resource.")

    workspace = get_workspace_client()

    statement = f"""
        SELECT
            id,
            text,
            doc_id,
            source_folder,
            source_path,
            source_file,
            page_start,
            page_end,
            chunk_index,
            parser
        FROM {CHUNKS_TABLE}
        WHERE source_path = :source_path
          AND chunk_index BETWEEN :min_chunk AND :max_chunk
        ORDER BY chunk_index
    """

    response = workspace.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        parameters=[
            StatementParameterListItem(name="source_path", value=str(source_path), type="STRING"),
            StatementParameterListItem(name="min_chunk", value=str(min_chunk), type="INT"),
            StatementParameterListItem(name="max_chunk", value=str(max_chunk), type="INT"),
        ],
        wait_timeout="30s",
    )

    if not response.result or not response.result.data_array:
        return []

    columns = [column.name for column in response.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in response.result.data_array]


def expand_with_neighbor_chunks(hits):
    """Expand each retrieved chunk with chunk_index -1, current chunk, and +1."""
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
        neighbor_rows = fetch_neighbor_rows(source_path, min_chunk, max_chunk)

        if not neighbor_rows:
            fallback = dict(hit)
            fallback["expansion_type"] = "original_chunk_no_neighbors_found"
            expanded_pairs.append(fallback)
            continue

        for row in neighbor_rows:
            item = dict(row)
            item["retrieval_rank"] = hit["retrieval_rank"]
            item["score"] = hit.get("score", 0.0)
            item["anchor_chunk_index"] = chunk_index

            row_chunk_index = safe_int(item.get("chunk_index"))
            item["expansion_type"] = (
                "retrieved_chunk" if row_chunk_index == chunk_index else "neighbor_chunk"
            )

            identity = "::".join(
                [
                    str(item.get("doc_id")),
                    str(item.get("source_path")),
                    str(item.get("chunk_index")),
                ]
            )
            hash_identity = text_hash(item.get("text"))

            if identity in seen or hash_identity in seen:
                continue

            seen.add(identity)
            seen.add(hash_identity)
            expanded_pairs.append(item)

    final_context = []
    total_chars = 0

    # Apply the context budget after preserving retrieval-rank order.
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


def sigmoid(score):
    """Map raw CrossEncoder score into a 0-1 confidence-style value."""
    try:
        return 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


def passage_text(item):
    """Build the reranker passage from source metadata plus chunk text."""
    title = item.get("source_file") or item.get("source_path") or ""
    folder = item.get("source_folder") or ""
    text = item.get("text") or ""

    return f"{title}\n{folder}\n{text}".strip()


def rerank_context(query, context_items):
    """Rerank expanded context blocks with the CrossEncoder model."""
    if not RERANKER_ENABLED or len(context_items) <= 1:
        return context_items

    try:
        model = get_cross_encoder()
        pairs = [(query, passage_text(item)) for item in context_items]
        scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
    except Exception as exc:
        st.warning(f"Reranker failed; preserving retrieval order: {exc}")
        return context_items

    scored = []

    for pos, (item, raw_score) in enumerate(zip(context_items, scores), start=1):
        rerank_score = float(raw_score)

        reranked = dict(item)
        reranked["cross_encoder_model"] = RERANKER_MODEL
        reranked["cross_encoder_score"] = rerank_score
        reranked["pre_rerank_position"] = pos
        reranked["pre_rerank_score"] = item.get("score", 0.0)
        reranked["score"] = max(safe_float(item.get("score")), sigmoid(rerank_score))

        scored.append((rerank_score, reranked))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:RERANKER_TOP_N]]


def format_local_context(context_items):
    """Format final context blocks for the answer-generation prompt."""
    blocks = []
    remaining_chars = CONTEXT_MAX_CHARS

    for index_position, item in enumerate(context_items, start=1):
        if remaining_chars <= 0:
            break

        source = item.get("source_file") or item.get("source_path") or "local document"
        page_value = safe_int(item.get("page_start"))
        chunk_value = safe_int(item.get("chunk_index"))
        score = safe_float(item.get("score"))
        expansion_type = item.get("expansion_type")

        page_text = f", page {page_value}" if page_value is not None else ""
        expansion_text = f" | {expansion_type}" if expansion_type else ""
        chunk_text = f" | chunk={chunk_value}" if chunk_value is not None else ""

        text = (item.get("text") or "")[:remaining_chars]

        block = (
            f"[Local {index_position}] {source}{page_text} "
            f"| relevance={score:.2f}{expansion_text}{chunk_text}\n"
            f"{text}"
        )

        blocks.append(block)
        remaining_chars -= len(text)

    return "\n\n".join(blocks)


def source_labels(context_items, max_sources=8):
    """Build compact source labels for display under the final answer."""
    labels = []
    seen = set()

    for item in context_items:
        source = item.get("source_file") or item.get("source_path") or "local document"
        page_value = safe_int(item.get("page_start"))
        score = safe_float(item.get("score"))

        page_text = f", page {page_value}" if page_value is not None else ""
        label = f"{source}{page_text} (score {score:.2f})"
        dedupe_key = (source, page_value)

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        labels.append(label)

        if len(labels) >= max_sources:
            break

    return labels


def retrieve_for_rag(query):
    """Run hybrid search, neighbor expansion, reranking, and context formatting."""
    raw_hits = search_hybrid(query)
    expanded_context = expand_with_neighbor_chunks(raw_hits)
    final_context = rerank_context(query, expanded_context)
    formatted_context = format_local_context(final_context)
    sources = source_labels(final_context)

    return {
        "raw_hits": raw_hits,
        "expanded_context": expanded_context,
        "final_context_items": final_context,
        "formatted_context": formatted_context,
        "sources": sources,
    }


def extract_model_text(response):
    """Extract answer text from the Databricks model serving response."""
    if response.choices:
        choice = response.choices[0]

        if choice.message and choice.message.content:
            content = choice.message.content

            # Some models can return structured content blocks.
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content")
                        if text:
                            parts.append(str(text))
                    else:
                        parts.append(str(block))
                return "\n".join(parts).strip()

            return str(content)

        if choice.text:
            return str(choice.text)

    return "The model returned an empty response."


def answer_question(question):
    """Retrieve evidence and ask the Databricks-hosted chat model for a grounded answer."""
    retrieval = retrieve_for_rag(question)
    local_context = retrieval["formatted_context"]

    user_prompt = f"""
Question:
{question}

Local document evidence:
{local_context}

Grounded answer:
"""

    workspace = get_workspace_client()
    response = workspace.serving_endpoints.query(
        name=CHAT_MODEL_ENDPOINT,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=ANSWER_SYSTEM_PROMPT),
            ChatMessage(role=ChatMessageRole.USER, content=user_prompt),
        ],
        temperature=0,
        max_tokens=800,
    )

    return {
        "answer": extract_model_text(response),
        "sources": retrieval["sources"],
        "context_items": retrieval["final_context_items"],
    }


# Streamlit UI
st.set_page_config(page_title="Cobb County RAG App", page_icon="CC", layout="wide")

st.title("Cobb County RAG App")
st.caption("Databricks AI Search Hybrid + CrossEncoder reranking + Databricks-hosted LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a Cobb County fire, building, inspection, or code question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching local documents, reranking evidence, and drafting a grounded answer..."):
            try:
                result = answer_question(question)
                st.markdown(result["answer"])

                if result["sources"]:
                    st.markdown("**Sources**")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")

                with st.expander("Retrieved context"):
                    context_df = pd.DataFrame(result["context_items"])
                    display_columns = [
                        "retrieval_rank",
                        "source_file",
                        "page_start",
                        "chunk_index",
                        "expansion_type",
                        "score",
                        "cross_encoder_score",
                        "pre_rerank_position",
                        "text",
                    ]
                    existing_columns = [col for col in display_columns if col in context_df.columns]

                    if existing_columns:
                        st.dataframe(context_df[existing_columns], use_container_width=True)
                    else:
                        st.write("No retrieved context available.")

                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )

            except Exception as exc:
                error_message = f"Something went wrong: `{exc}`"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )