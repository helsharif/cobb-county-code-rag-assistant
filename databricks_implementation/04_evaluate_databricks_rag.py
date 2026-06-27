# Databricks notebook source
# MAGIC %md
# MAGIC # Deprecation Note
# MAGIC The notebook used the deprecated Databricks VectorSearchClient (and an earlier version of app.py) which has since been replaced with AISearchClient. Use notebook 05_evaluate_databricks_rag_updated instead

# COMMAND ----------

# MAGIC %md
# MAGIC # Install

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch databricks-sdk sentence-transformers torch pandas numpy mlflow tabulate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Config

# COMMAND ----------

import hashlib
import json
import math
import time
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import mlflow

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F


TESTSET_PATH = "/Volumes/cobb_rag/fire_code/eval_testset/cobb_county_testset.csv"

CHUNKS_TABLE = "cobb_rag.fire_code.document_chunks"
INDEX_NAME = "cobb_rag.fire_code.document_chunks_hybrid_index"

ANSWER_MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
JUDGE_MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

RETRIEVER_K = 15
CONTEXT_NEIGHBOR_WINDOW = 1
CONTEXT_MAX_EXPANDED_DOCS = 8
CONTEXT_MAX_CHARS = 18000
MAX_CHARS_PER_CONTEXT_BLOCK = 6000

RERANKER_ENABLED = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_TOP_N = 8
RERANKER_BATCH_SIZE = 16

EVAL_OUTPUT_TABLE = "cobb_rag.fire_code.databricks_rag_eval_results"
EVAL_SUMMARY_TABLE = "cobb_rag.fire_code.databricks_rag_eval_summary"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Test Set

# COMMAND ----------

def normalize_testset_dataframe(df):
    df = df.copy()

    rename_map = {}
    if "user_input" in df.columns and "question" not in df.columns:
        rename_map["user_input"] = "question"
    if "reference" in df.columns and "ground_truth" not in df.columns:
        rename_map["reference"] = "ground_truth"
    if "answer" in df.columns and "ground_truth" not in df.columns:
        rename_map["answer"] = "ground_truth"

    df = df.rename(columns=rename_map)

    if "question" not in df.columns:
        raise ValueError("CSV must include a question column.")
    if "ground_truth" not in df.columns:
        raise ValueError("CSV must include ground_truth, reference, or answer column.")

    df["question"] = df["question"].astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["ground_truth"] != "")]

    if len(df) != 50:
        raise ValueError(f"Expected exactly 50 populated rows; found {len(df)}.")

    return df[["question", "ground_truth"]].reset_index(drop=True)


testset = normalize_testset_dataframe(pd.read_csv(TESTSET_PATH))
display(testset.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # RAG Pipeline Helpers

# COMMAND ----------

workspace = WorkspaceClient()
vector_client = VectorSearchClient(disable_notice=True)
index = vector_client.get_index(index_name=INDEX_NAME)


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def text_hash(text):
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def parse_search_results(results):
    columns = [column["name"] for column in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]

    parsed = []
    for rank, row in enumerate(rows, start=1):
        item = dict(zip(columns, row))
        item["retrieval_rank"] = rank
        item["score"] = safe_float(
            item.get("score")
            or item.get("_score")
            or item.get("similarity_score")
            or item.get("relevance_score")
            or 0.0
        )
        parsed.append(item)

    return parsed


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


def expand_with_neighbor_chunks(hits):
    expanded = []
    seen = set()

    for hit in hits:
        source_path = hit.get("source_path")
        chunk_index = safe_int(hit.get("chunk_index"))

        if source_path is None or chunk_index is None:
            fallback = dict(hit)
            fallback["expansion_type"] = "original_chunk_no_chunk_index"
            expanded.append(fallback)
            continue

        neighbor_rows = (
            spark.table(CHUNKS_TABLE)
            .where(F.col("source_path") == source_path)
            .where(
                F.col("chunk_index").between(
                    chunk_index - CONTEXT_NEIGHBOR_WINDOW,
                    chunk_index + CONTEXT_NEIGHBOR_WINDOW,
                )
            )
            .orderBy("chunk_index")
            .collect()
        )

        if not neighbor_rows:
            fallback = dict(hit)
            fallback["expansion_type"] = "original_chunk_no_neighbors_found"
            expanded.append(fallback)
            continue

        for row in neighbor_rows:
            item = row.asDict()
            item["retrieval_rank"] = hit["retrieval_rank"]
            item["score"] = hit.get("score", 0.0)
            item["anchor_chunk_index"] = chunk_index

            row_chunk_index = safe_int(item.get("chunk_index"))
            item["expansion_type"] = (
                "retrieved_chunk" if row_chunk_index == chunk_index else "neighbor_chunk"
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
            expanded.append(item)

    final_context = []
    total_chars = 0

    for item in expanded:
        if len(final_context) >= CONTEXT_MAX_EXPANDED_DOCS:
            break

        remaining = CONTEXT_MAX_CHARS - total_chars
        if remaining <= 0:
            break

        text = item.get("text") or ""
        trimmed = text[: min(len(text), remaining, MAX_CHARS_PER_CONTEXT_BLOCK)]

        if not trimmed.strip():
            continue

        output_item = dict(item)
        output_item["text"] = trimmed
        final_context.append(output_item)
        total_chars += len(trimmed)

    return final_context

# COMMAND ----------

# MAGIC %md
# MAGIC # Reranker

# COMMAND ----------

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

    model = get_cross_encoder()
    pairs = [(query, passage_text(item)) for item in context_items]
    scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)

    scored = []
    for pos, (item, raw_score) in enumerate(zip(context_items, scores), start=1):
        rerank_score = float(raw_score)
        reranked = dict(item)
        reranked["cross_encoder_score"] = rerank_score
        reranked["pre_rerank_position"] = pos
        reranked["score"] = max(safe_float(item.get("score")), sigmoid(rerank_score))
        scored.append((rerank_score, reranked))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:RERANKER_TOP_N]]

# COMMAND ----------

# MAGIC %md
# MAGIC # Answer Generation

# COMMAND ----------

ANSWER_SYSTEM_PROMPT = """
You answer Cobb County, Georgia code, zoning, building, fire, permit, and inspection questions.

Use ONLY the supplied local document evidence.
Do not use outside knowledge.
Do not infer from source titles, nearby topics, or general code knowledge.
Do not cite a requirement unless the exact requirement appears in the supplied evidence.

Rules:
1. Every factual claim must be directly supported by the supplied evidence.
2. If the evidence supports only part of the answer, answer only that part.
3. If the evidence does not contain the requested detail, say: "The retrieved context does not state that."
4. Include source names and page numbers inline when available.
5. For numeric, procedural, timing, fee, deadline, distance, height, parking, zoning, or inspection requirements, quote or closely paraphrase the supporting phrase.
6. If the question is broad and the evidence shows multiple specific cases, begin by saying the retrieved context does not state one universal rule. Then list only the specific cases shown in the evidence.
7. When citing numbered checklist items, call them "items" unless the evidence explicitly labels them as sections.
8. Keep the answer concise and practical.
"""


def format_local_context(context_items):
    blocks = []
    remaining_chars = CONTEXT_MAX_CHARS

    for i, item in enumerate(context_items, start=1):
        source = item.get("source_file") or item.get("source_path") or "local document"
        page = safe_int(item.get("page_start"))
        score = safe_float(item.get("score"))
        chunk = safe_int(item.get("chunk_index"))
        expansion = item.get("expansion_type")

        page_text = f", page {page}" if page is not None else ""
        chunk_text = f" | chunk={chunk}" if chunk is not None else ""
        expansion_text = f" | {expansion}" if expansion else ""

        text = (item.get("text") or "")[:remaining_chars]
        blocks.append(
            f"[Local {i}] {source}{page_text} | relevance={score:.2f}{expansion_text}{chunk_text}\n{text}"
        )
        remaining_chars -= len(text)

        if remaining_chars <= 0:
            break

    return "\n\n".join(blocks)


def source_labels(context_items, max_sources=8):
    labels = []
    seen = set()

    for item in context_items:
        source = item.get("source_file") or item.get("source_path") or "local document"
        page = safe_int(item.get("page_start"))
        score = safe_float(item.get("score"))

        key = (source, page)
        if key in seen:
            continue

        seen.add(key)
        page_text = f", page {page}" if page is not None else ""
        labels.append(f"{source}{page_text} (score {score:.2f})")

        if len(labels) >= max_sources:
            break

    return labels


def extract_model_text(response):
    choice = response.choices[0]
    if choice.message and choice.message.content:
        return str(choice.message.content)
    if choice.text:
        return str(choice.text)
    return ""


def call_chat_model(endpoint, system_prompt, user_prompt, max_tokens=800):
    response = workspace.serving_endpoints.query(
        name=endpoint,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER, content=user_prompt),
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return extract_model_text(response)


def run_rag(question):
    started = time.perf_counter()

    raw_hits = search_hybrid(question)
    expanded = expand_with_neighbor_chunks(raw_hits)
    final_context = rerank_context(question, expanded)
    formatted_context = format_local_context(final_context)

    user_prompt = f"""
Question:
{question}

Local document evidence:
{formatted_context}

Grounded answer:
"""

    answer = call_chat_model(
        ANSWER_MODEL_ENDPOINT,
        ANSWER_SYSTEM_PROMPT,
        user_prompt,
        max_tokens=800,
    )

    latency = round(time.perf_counter() - started, 3)

    return {
        "answer": answer,
        "contexts": [item.get("text", "") for item in final_context],
        "sources": source_labels(final_context),
        "context_items": final_context,
        "latency_seconds": latency,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC # LLM Judge Prompts

# COMMAND ----------

ALLOWED_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]

SCORE_SCALE = """
Use only this five-point score scale:
0.00 = no meaningful support, irrelevant, or unusable.
0.25 = minimal support with major missing or incorrect facts.
0.50 = partially correct with important gaps or mixed evidence.
0.75 = mostly correct with minor omissions, noise, or wording issues.
1.00 = fully correct, well-supported, and technically precise.
Do not output scores outside 0.00, 0.25, 0.50, 0.75, or 1.00.
"""


def quantize_score(value):
    try:
        numeric = min(max(float(value), 0.0), 1.0)
    except Exception:
        return 0.0
    return min(ALLOWED_SCORES, key=lambda allowed: abs(allowed - numeric))


def parse_judge_json(text):
    cleaned = text.strip()

    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "```")
        parts = cleaned.split("```")
        cleaned = max(parts, key=len).strip()

    try:
        payload = json.loads(cleaned)
    except Exception:
        return {
            "reasoning": f"Judge returned unparsable response: {text[:500]}",
            "score": 0.0,
        }

    return {
        "reasoning": str(payload.get("reasoning", "")),
        "score": quantize_score(payload.get("score", 0.0)),
    }


def judge_metric(metric_name, system_prompt, user_prompt):
    full_system_prompt = f"""
{system_prompt}

Return ONLY valid JSON with this shape:
{{"reasoning": "short explanation", "score": 0.0}}

The score must be one of: 0.0, 0.25, 0.5, 0.75, 1.0.
"""

    text = call_chat_model(
        JUDGE_MODEL_ENDPOINT,
        full_system_prompt,
        user_prompt,
        max_tokens=700,
    )

    grade = parse_judge_json(text)
    return {
        f"{metric_name}_score": grade["score"],
        f"{metric_name}_reasoning": grade["reasoning"],
    }


def contexts_to_text(contexts, sources=None, limit=30000):
    if not contexts:
        return "No retrieved context."

    blocks = []
    sources = sources or []

    for i, context in enumerate(contexts, start=1):
        source = sources[i - 1] if i - 1 < len(sources) else ""
        header = f"[Local {i}] {source}".strip()
        blocks.append(f"{header}\n{context}")

    return "\n\n".join(blocks)[:limit]

# COMMAND ----------

# MAGIC %md
# MAGIC # RAGAS-Style Metrics

# COMMAND ----------

def evaluate_one(question, answer, contexts, ground_truth, sources):
    context_text = contexts_to_text(contexts, sources)

    faithfulness = judge_metric(
        "faithfulness",
        f"""
You grade faithfulness/groundedness for a technical Cobb County RAG app.
Use ONLY the QUESTION, ANSWER, and CONTEXT. Do not use outside knowledge or the reference answer.

Task:
1. Extract factual claims from the ANSWER.
2. Decide whether each claim is supported, unsupported, or contradicted by CONTEXT.
3. Be extremely strict for numbers, dates, dimensions, fees, code sections, exceptions, parking ratios, zoning rules, and inspection procedures.
4. A numerical value is supported only if the same value appears in CONTEXT and refers to the same subject.
5. If the answer abstains and CONTEXT lacks the exact requested fact, score 1.00.
6. If the answer abstains but CONTEXT clearly contains the exact requested fact, score 0.00.

{SCORE_SCALE}
""",
        f"""
QUESTION:
{question}

CONTEXT:
{context_text}

ANSWER:
{answer}
""",
    )

    answer_relevancy = judge_metric(
        "answer_relevancy",
        f"""
You grade answer relevancy for a technical Cobb County RAG app.
Evaluate how well ANSWER addresses the user's intent in QUESTION.
Be strict with incorrect technical details, but flexible with semantic phrasing.

{SCORE_SCALE}
""",
        f"""
QUESTION:
{question}

ANSWER:
{answer}
""",
    )

    context_precision = judge_metric(
        "context_precision",
        f"""
You grade context precision, also called signal-to-noise, for a technical Cobb County RAG app.
Evaluate retrieved CONTEXTS against QUESTION.
Estimate the ratio of relevant chunks or relevant information to total retrieved chunks or information.
Reward cases where needed evidence is present even if some unrelated text is included.
Be strict about whether passages actually support the requested technical facts.

{SCORE_SCALE}
""",
        f"""
QUESTION:
{question}

CONTEXTS:
{context_text}
""",
    )

    context_recall = judge_metric(
        "context_recall",
        f"""
You grade context recall/coverage for a technical Cobb County RAG app.
Compare CONTEXTS against the REFERENCE ANSWER.
Identify key facts required to satisfy the reference answer, then score required facts found in context divided by total required facts.
Be extremely strict with numbers, dates, dimensions, fees, code sections, exceptions, parking ratios, zoning rules, and procedural requirements.

{SCORE_SCALE}
""",
        f"""
QUESTION:
{question}

REFERENCE ANSWER:
{ground_truth}

CONTEXTS:
{context_text}
""",
    )

    return {
        **faithfulness,
        **answer_relevancy,
        **context_precision,
        **context_recall,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Full 50 Question Evaluation

# COMMAND ----------

run_id = datetime.now(UTC).strftime("databricks-rag-eval-%Y%m%d-%H%M%S")
rows = []

mlflow.set_experiment("/Users/helsharifmle01@gmail.com/cobb-county-rag-eval")

with mlflow.start_run(run_name=run_id):
    mlflow.log_param("answer_model", ANSWER_MODEL_ENDPOINT)
    mlflow.log_param("judge_model", JUDGE_MODEL_ENDPOINT)
    mlflow.log_param("index_name", INDEX_NAME)
    mlflow.log_param("retriever_k", RETRIEVER_K)
    mlflow.log_param("reranker_model", RERANKER_MODEL)

    for i, row in testset.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]

        print(f"[{i + 1}/50] {question}")

        try:
            rag_result = run_rag(question)
            judge_result = evaluate_one(
                question=question,
                answer=rag_result["answer"],
                contexts=rag_result["contexts"],
                ground_truth=ground_truth,
                sources=rag_result["sources"],
            )

            record = {
                "run_id": run_id,
                "row_index": int(i),
                "question": question,
                "ground_truth": ground_truth,
                "answer": rag_result["answer"],
                "contexts": json.dumps(rag_result["contexts"]),
                "sources": json.dumps(rag_result["sources"]),
                "latency_seconds": rag_result["latency_seconds"],
                "error": None,
                **judge_result,
            }

        except Exception as exc:
            record = {
                "run_id": run_id,
                "row_index": int(i),
                "question": question,
                "ground_truth": ground_truth,
                "answer": "",
                "contexts": "[]",
                "sources": "[]",
                "latency_seconds": None,
                "error": f"{exc.__class__.__name__}: {exc}",
                "faithfulness_score": 0.0,
                "faithfulness_reasoning": "Evaluation failed.",
                "answer_relevancy_score": 0.0,
                "answer_relevancy_reasoning": "Evaluation failed.",
                "context_precision_score": 0.0,
                "context_precision_reasoning": "Evaluation failed.",
                "context_recall_score": 0.0,
                "context_recall_reasoning": "Evaluation failed.",
            }

        rows.append(record)

        # Keep judge/model endpoint pressure gentle.
        time.sleep(1)

    results_df = pd.DataFrame(rows)

    metric_cols = [
        "faithfulness_score",
        "answer_relevancy_score",
        "context_precision_score",
        "context_recall_score",
    ]

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "question_count": len(results_df),
        "faithfulness": float(results_df["faithfulness_score"].mean()),
        "answer_relevance": float(results_df["answer_relevancy_score"].mean()),
        "context_precision": float(results_df["context_precision_score"].mean()),
        "context_recall": float(results_df["context_recall_score"].mean()),
        "latency_average": float(results_df["latency_seconds"].dropna().mean()),
        "latency_p50": float(np.percentile(results_df["latency_seconds"].dropna(), 50)),
        "latency_p99": float(np.percentile(results_df["latency_seconds"].dropna(), 99)),
    }

    for key, value in summary.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)

    mlflow.log_dict(summary, "summary.json")
    mlflow.log_table(results_df, "eval_rows.json")

summary

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Results to Delta Table

# COMMAND ----------

results_df["error"] = results_df["error"].fillna("").astype(str) # force error column to strings (avoid issue where it's None)
spark_results_df = spark.createDataFrame(results_df)
spark_summary_df = spark.createDataFrame(pd.DataFrame([summary]))

(
    spark_results_df.write
    .format("delta")
    .mode("append")
    .saveAsTable(EVAL_OUTPUT_TABLE)
)

(
    spark_summary_df.write
    .format("delta")
    .mode("append")
    .saveAsTable(EVAL_SUMMARY_TABLE)
)

display(spark_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary Display

# COMMAND ----------

# Display the overall Databricks RAG metrics in the same style as the local app summary.

summary_display = pd.DataFrame([
    {
        "Setting": "Databricks AI Search Hybrid + CrossEncoder",
        "Faithfulness": round(summary["faithfulness"], 3),
        "Answer relevance": round(summary["answer_relevance"], 3),
        "Context precision": round(summary["context_precision"], 3),
        "Context recall": round(summary["context_recall"], 3),
        "Avg latency": round(summary["latency_average"], 1),
        "P50 latency": round(summary["latency_p50"], 1),
        "P99 latency": round(summary["latency_p99"], 1),
        "Questions": summary["question_count"],
        "Run ID": summary["run_id"],
    }
])

display(summary_display)

# COMMAND ----------

print(summary_display.to_markdown(index=False))