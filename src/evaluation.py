"""LangSmith evaluation helpers for the Cobb County RAG app."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from pydantic import BaseModel, Field

from src.agent import NO_ANSWER, CobbCountyRAGAgent
from src.config import (
    COLLECTION_SLUGS,
    DOCLING_CHROMA_BM25_COLLECTION_NAME,
    DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME,
    DOCLING_COLLECTION_NAME,
    OPTION_1_LABEL,
    OPTION_2_LABEL,
    OPTION_3_LABEL,
    OPTION_4_LABEL,
    ORIGINAL_COLLECTION_NAME,
    ROOT_DIR,
    get_settings,
)


logger = logging.getLogger(__name__)

TESTSET_PATH = ROOT_DIR / "eval_testset" / "cobb_county_testset.csv"
EVAL_RESULTS_DIR = ROOT_DIR / "eval_results"
EVAL_RESULTS_LOG_PATH = EVAL_RESULTS_DIR / "eval_results_log.csv"
EVAL_RESULTS_LOG_COLUMNS = [
    "assessment_date",
    "setting",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "latency_average",
    "latency_p50",
    "latency_p99",
]
EVAL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LANGSMITH_DATASET_PREFIX = "cobb-county-rag-eval-testset"

CONFIG_LABELS = {
    ORIGINAL_COLLECTION_NAME: OPTION_1_LABEL,
    DOCLING_COLLECTION_NAME: OPTION_2_LABEL,
    DOCLING_CHROMA_BM25_COLLECTION_NAME: OPTION_3_LABEL,
    DOCLING_CHROMA_BM25_EXPANSION_COLLECTION_NAME: OPTION_4_LABEL,
}

ProgressCallback = Callable[[dict[str, Any]], None]


class ScoredGrade(BaseModel):
    """Five-point evaluator grade returned by the LangSmith judge model."""

    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step analysis explaining the evidence, partial credit, and final score. "
            "Be strict with technical building/fire-code facts such as numbers, dates, dimensions, "
            "fees, fire ratings, code sections, and procedural requirements. "
            "The final score must be one of: 0.0, 0.25, 0.5, 0.75, or 1.0."
        ),
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Five-point score. Must be one of: 0.0, 0.25, 0.5, 0.75, or 1.0.",
    )


@dataclass(frozen=True)
class EvaluationResultFile:
    collection_name: str
    label: str
    path: Path


def eval_result_file(collection_name: str) -> EvaluationResultFile:
    """Return the cache path for one retrieval backend."""

    label = CONFIG_LABELS.get(collection_name, collection_name)
    slug = COLLECTION_SLUGS.get(collection_name, collection_name.replace("cobb_code_docs_", ""))
    return EvaluationResultFile(
        collection_name=collection_name,
        label=label,
        path=EVAL_RESULTS_DIR / f"{slug}_results.json",
    )


def load_eval_results(collection_name: str) -> dict[str, Any] | None:
    """Load persisted LangSmith evaluation results for the selected vector store."""

    result_file = eval_result_file(collection_name)
    if not result_file.path.exists():
        return None
    with result_file.path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    try:
        current_testset = ensure_testset()
        if payload.get("question_count") != len(current_testset):
            return None
        saved_hash = payload.get("testset_sha256")
        if saved_hash and saved_hash != _testset_hash(current_testset):
            return None
    except Exception as exc:
        logger.warning("Could not validate cached evaluation results: %s", exc)
    return payload


def save_eval_results(collection_name: str, payload: dict[str, Any]) -> Path:
    """Persist LangSmith results so Streamlit can display them after restarts."""

    result_file = eval_result_file(collection_name)
    result_file.path.parent.mkdir(parents=True, exist_ok=True)
    with result_file.path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    try:
        append_eval_results_log(payload)
    except OSError as exc:
        logger.warning("Evaluation results were saved, but the persistent log could not be updated: %s", exc)
    return result_file.path


def append_eval_results_log(payload: dict[str, Any]) -> Path:
    """Append one evaluation summary row to the persistent metric history log."""

    EVAL_RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = _eval_log_row(payload)
    if EVAL_RESULTS_LOG_PATH.exists():
        dataframe = pd.read_csv(EVAL_RESULTS_LOG_PATH)
        missing_columns = [column for column in EVAL_RESULTS_LOG_COLUMNS if column not in dataframe.columns]
        if missing_columns:
            dataframe = dataframe.reindex(columns=EVAL_RESULTS_LOG_COLUMNS)
    else:
        dataframe = pd.DataFrame(columns=EVAL_RESULTS_LOG_COLUMNS)

    duplicate = (
        (dataframe.get("assessment_date") == row["assessment_date"])
        & (dataframe.get("setting") == row["setting"])
    )
    if not dataframe.empty and duplicate.any():
        return EVAL_RESULTS_LOG_PATH

    updated = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(EVAL_RESULTS_LOG_PATH, index=False)
    return EVAL_RESULTS_LOG_PATH


def ensure_testset() -> pd.DataFrame:
    """Load the fixed 50-question evaluation test set from eval_testset/."""

    if not TESTSET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation test set not found at {TESTSET_PATH}. "
            "Create eval_testset/cobb_county_testset.csv with question and ground_truth columns."
        )
    dataframe = pd.read_csv(TESTSET_PATH)
    return _normalize_testset_dataframe(dataframe)


def ensure_langsmith_dataset(client: Client, testset: pd.DataFrame) -> tuple[str, str]:
    """Create or reuse a LangSmith dataset based on the fixed CSV contents."""

    dataset_hash = _testset_hash(testset)
    dataset_name = f"{LANGSMITH_DATASET_PREFIX}-{dataset_hash}"
    examples = [
        {
            "inputs": {"question": row["question"]},
            "outputs": {"answer": row["ground_truth"]},
            "metadata": {"source": str(TESTSET_PATH.relative_to(ROOT_DIR)), "row_index": int(index)},
        }
        for index, row in testset.iterrows()
    ]

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        existing_count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        if existing_count == len(examples):
            return dataset_name, str(dataset.id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        dataset_name = f"{dataset_name}-{timestamp}"

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Fixed 50-question Cobb County building and fire code RAG evaluation set "
            "loaded from eval_testset/cobb_county_testset.csv."
        ),
        metadata={"testset_sha256": dataset_hash, "row_count": len(examples)},
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset_name, str(dataset.id)


def run_langsmith_evaluation(
    collection_name: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the selected RAG backend as a LangSmith experiment and cache the returned scores."""

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPEN_API_KEY is required to run LangSmith LLM-as-judge evaluation.")
    if not settings.langsmith_api_key:
        raise ValueError("LANGSMITH_API_KEY is required to create and retrieve LangSmith evaluation scores.")

    _emit_progress(progress_callback, phase="loading_testset", message="Loading fixed 50-question CSV test set.")
    testset = ensure_testset()

    client = Client()
    dataset_name, dataset_id = ensure_langsmith_dataset(client, testset)
    _emit_progress(
        progress_callback,
        phase="dataset_ready",
        message=f"Using LangSmith dataset {dataset_name}.",
        current=0,
        total=len(testset),
    )

    answer_counter = {"count": 0}
    execution_times: list[dict[str, Any]] = []
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    label = CONFIG_LABELS.get(collection_name, collection_name)
    slug = COLLECTION_SLUGS.get(collection_name, collection_name.replace("cobb_code_docs_", ""))

    @traceable(name=f"cobb_county_rag_{slug}_target")
    def target(inputs: dict) -> dict:
        question = str(inputs["question"])
        answer_counter["count"] += 1
        _emit_progress(
            progress_callback,
            phase="answering_questions",
            message=f"Running selected RAG pipeline on question {answer_counter['count']} of {len(testset)}.",
            current=answer_counter["count"] - 1,
            total=len(testset),
            question=question,
        )
        started_query = time.perf_counter()
        agent = CobbCountyRAGAgent(collection_name=collection_name)
        error_message = None
        try:
            result = _answer_with_backoff(agent, question, settings)
            answer = result.answer
            contexts = list(result.contexts or [])
            sources = result.sources
            used_local = result.used_local
            used_web = result.used_web
        except Exception as exc:
            error_message = f"{exc.__class__.__name__}: {exc}"
            logger.exception("Evaluation target failed for question: %s", question)
            answer = NO_ANSWER
            contexts = []
            sources = []
            used_local = False
            used_web = False
        execution_time = round(time.perf_counter() - started_query, 3)
        execution_times.append({"question": question, "execution_time": execution_time})
        _emit_progress(
            progress_callback,
            phase="answering_questions",
            message=(
                f"Completed question {answer_counter['count']} of {len(testset)} "
                f"in {execution_time:.1f} seconds."
            ),
            current=answer_counter["count"],
            total=len(testset),
            question=question,
            execution_time=execution_time,
        )
        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "used_local": used_local,
            "used_web": used_web,
            "execution_time": execution_time,
            "error": error_message,
        }

    evaluators = _build_langsmith_evaluators()
    experiment_prefix = f"cobb-county-rag-{slug}-langsmith"
    _emit_progress(
        progress_callback,
        phase="langsmith_evaluation",
        message="Running LangSmith experiment and LLM-as-judge evaluators.",
        current=0,
        total=len(testset),
    )

    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        description=f"Cobb County RAG evaluation for {label} vector store.",
        metadata={
            "collection_name": collection_name,
            "config_label": label,
            "testset_path": str(TESTSET_PATH.relative_to(ROOT_DIR)),
            "testset_sha256": _testset_hash(testset),
        },
        max_concurrency=1,
        blocking=True,
        upload_results=True,
    )

    _emit_progress(
        progress_callback,
        phase="retrieving_scores",
        message="Retrieving LangSmith scores for display.",
        current=len(testset),
        total=len(testset),
    )
    scores_df = experiment_results.to_pandas()
    metrics = _extract_metric_means(scores_df)
    execution_time_records = _extract_execution_time_records(scores_df, execution_times)
    latency_metrics = _extract_latency_metrics(execution_time_records)
    metrics.update(latency_metrics)
    payload = {
        "collection_name": collection_name,
        "config_label": label,
        "config_slug": slug,
        "evaluation_backend": "LangSmith",
        "started_at_utc": started_at_utc,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "testset_path": str(TESTSET_PATH.relative_to(ROOT_DIR)),
        "testset_sha256": _testset_hash(testset),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "experiment_name": experiment_results.experiment_name,
        "experiment_id": str(experiment_results.experiment_id),
        "experiment_url": experiment_results.url,
        "question_count": len(testset),
        "metrics": metrics,
        "execution_times": execution_time_records,
        "rows": _json_safe_records(scores_df),
    }
    save_eval_results(collection_name, payload)
    _emit_progress(
        progress_callback,
        phase="saving_results",
        message="LangSmith evaluation scores retrieved and saved locally for the dashboard.",
        current=len(testset),
        total=len(testset),
    )
    return payload


def _build_langsmith_evaluators() -> list[Callable]:
    settings = get_settings()
    grader = ChatOpenAI(model=settings.eval_judge_model, temperature=0)
    score_scale = (
        "Use only this five-point score scale: "
        "0.00 = no meaningful support, irrelevant, or unusable; "
        "0.25 = minimal support with major missing or incorrect facts; "
        "0.50 = partially correct with important gaps or mixed evidence; "
        "0.75 = mostly correct with minor omissions, noise, or wording issues; "
        "1.00 = fully correct, well-supported, and technically precise. "
        "Do not output scores outside 0.00, 0.25, 0.50, 0.75, or 1.00."
    )

    faithfulness_llm = grader.with_structured_output(ScoredGrade, method="json_schema", strict=True)
    answer_relevance_llm = grader.with_structured_output(ScoredGrade, method="json_schema", strict=True)
    context_precision_llm = grader.with_structured_output(ScoredGrade, method="json_schema", strict=True)
    context_recall_llm = grader.with_structured_output(ScoredGrade, method="json_schema", strict=True)

    def faithfulness(inputs: dict, outputs: dict) -> dict:
        facts = _contexts_to_text(outputs)
        grade = _invoke_judge_with_backoff(
            faithfulness_llm,
            [
                {
                    "role": "system",
                    "content": (
                        "You grade faithfulness/groundedness for a technical Cobb County building and fire code RAG app. "
                        "Use ONLY the ANSWER and CONTEXT provided to you. Do not use outside knowledge, memory, "
                        "or the known correct answer. "
                        "Think step-by-step in the reasoning field before assigning the score. "
                        "Task: "
                        "1. Extract each factual claim from the ANSWER. "
                        "2. For each claim, decide whether it is supported, unsupported, or contradicted. "
                        "A supported claim is directly stated in the CONTEXT or is a faithful paraphrase. "
                        "An unsupported claim is not found in the CONTEXT. A contradicted claim conflicts with the CONTEXT. "
                        "3. Be extremely strict for numerical values, dimensions, dates, fire ratings, fees, code sections, "
                        "exceptions, permit requirements, and inspection procedures. "
                        "4. A numerical value is supported only if the same value appears in the CONTEXT and refers to the same subject. "
                        "5. Do not give credit for values that are plausible, common, or from outside code knowledge. "
                        "6. If the ANSWER states a specific numerical or code requirement that does not appear in the CONTEXT, "
                        "mark that claim unsupported or contradicted. "
                        "7. If the ANSWER appropriately says it could not find a reliable answer because the CONTEXT lacks "
                        "the needed fact, treat that as faithful. "
                        "Compute raw_score = supported_claims / total_claims, then map to the nearest allowed score. "
                        f"{score_scale} "
                        "Do not penalize heavily for minor conversational filler that is not technical. "
                        "Score 0.00 when no meaningful factual claims are supported. "
                        "Score 1.00 only when all technical claims are supported and precise."
                    ),
                },
                {"role": "user", "content": f"CONTEXT:\n{facts}\n\nANSWER:\n{outputs.get('answer', '')}"},
            ],
            "faithfulness",
            settings,
        )
        return _grade_to_feedback("faithfulness", grade)

    def answer_relevancy(inputs: dict, outputs: dict) -> dict:
        grade = _invoke_judge_with_backoff(
            answer_relevance_llm,
            [
                {
                    "role": "system",
                    "content": (
                        "You grade answer relevancy for a technical Cobb County building and fire code RAG app. "
                        "Think step-by-step in the reasoning field before assigning the score. "
                        "Evaluate how well the ANSWER addresses the user's intent in the QUESTION. "
                        "Score 1.0 for a perfect, concise answer that directly satisfies the question. "
                        "Score 0.75 for answers that are helpful but slightly verbose, incomplete, "
                        "or miss a minor constraint. Use lower buckets for larger gaps. "
                        f"{score_scale} "
                        "Score 0.0 only if the answer is completely irrelevant, "
                        "hallucinated, or fails to address the question. Be strict with incorrect technical details, "
                        "but flexible with semantic phrasing."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{inputs.get('question', '')}\n\nANSWER:\n{outputs.get('answer', '')}",
                },
            ],
            "answer_relevancy",
            settings,
        )
        return _grade_to_feedback("answer_relevancy", grade)

    def context_precision(inputs: dict, outputs: dict) -> dict:
        facts = _contexts_to_text(outputs)
        grade = _invoke_judge_with_backoff(
            context_precision_llm,
            [
                {
                    "role": "system",
                    "content": (
                        "You grade context precision, also called signal-to-noise, for a technical Cobb County building "
                        "and fire code RAG app. Think step-by-step in the reasoning field before assigning the score. "
                        "Evaluate the retrieved CONTEXTS against the QUESTION. Estimate the ratio of relevant chunks "
                        "or relevant information to total retrieved chunks or information. Reward cases where the "
                        "needed evidence is present even if some unrelated text is also included. Do not require every "
                        f"chunk to be perfect. {score_scale} Be strict about whether retrieved passages actually support technical "
                        "numbers, dates, code sections, fees, dimensions, fire ratings, and procedures requested."
                    ),
                },
                {"role": "user", "content": f"QUESTION:\n{inputs.get('question', '')}\n\nCONTEXTS:\n{facts}"},
            ],
            "context_precision",
            settings,
        )
        return _grade_to_feedback("context_precision", grade)

    def context_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        facts = _contexts_to_text(outputs)
        grade = _invoke_judge_with_backoff(
            context_recall_llm,
            [
                {
                    "role": "system",
                    "content": (
                        "You grade context recall/coverage for a technical Cobb County building and fire code RAG app. "
                        "Think step-by-step in the reasoning field before assigning the score. "
                        "Compare the CONTEXTS against the REFERENCE ANSWER. Identify the key facts required to satisfy "
                        "the reference answer, then score as required facts found in context divided by total required facts. "
                        f"{score_scale} "
                        "Be extremely strict with numerical values, dates, dimensions, fire ratings, fees, code sections, "
                        "exceptions, and procedural requirements. Flexible semantic phrasing is acceptable only when "
                        "the technical meaning is preserved."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{inputs.get('question', '')}\n\n"
                        f"REFERENCE ANSWER:\n{reference_outputs.get('answer', '')}\n\n"
                        f"CONTEXTS:\n{facts}"
                    ),
                },
            ],
            "context_recall",
            settings,
        )
        return _grade_to_feedback("context_recall", grade)

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _answer_with_backoff(agent: CobbCountyRAGAgent, question: str, settings):
    """Run the RAG target with retry/backoff for transient rate limits."""

    delay = max(settings.eval_judge_delay_seconds, 0.0)
    max_retries = max(settings.eval_judge_max_retries, 0)
    for attempt in range(max_retries + 1):
        try:
            if delay and attempt:
                time.sleep(delay)
            return agent.answer(question)
        except Exception as exc:
            if not _is_retryable_rate_limit(exc) or attempt >= max_retries:
                raise
            wait_seconds = min(max(delay * (2**attempt), 1.0), 60.0)
            logger.warning(
                "Rate limit while answering evaluation question; retrying in %.1fs "
                "(attempt %s/%s).",
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)


def _invoke_judge_with_backoff(llm, messages: list[dict[str, str]], metric_name: str, settings) -> dict:
    """Call an evaluator LLM with retry/backoff for rate limits only."""

    delay = max(settings.eval_judge_delay_seconds, 0.0)
    max_retries = max(settings.eval_judge_max_retries, 0)
    if delay:
        time.sleep(delay)
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if not _is_retryable_rate_limit(exc) or attempt >= max_retries:
                raise
            wait_seconds = min(max(delay * (2**attempt), 1.0), 60.0)
            logger.warning(
                "LangSmith evaluator %s hit a rate limit; retrying in %.1f seconds (%s/%s): %s",
                metric_name,
                wait_seconds,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Evaluator {metric_name} failed unexpectedly.")


def _is_retryable_rate_limit(exc: Exception) -> bool:
    """Return True for transient 429/rate-limit errors, not exhausted quota."""

    message = str(exc).lower()
    if "insufficient_quota" in message or "exceeded your current quota" in message:
        return False
    return (
        "ratelimit" in exc.__class__.__name__.lower()
        or "rate limit" in message
        or "429" in message
        or "too many requests" in message
    )


def _grade_to_feedback(key: str, grade: ScoredGrade | dict) -> dict:
    if isinstance(grade, BaseModel):
        raw_score = getattr(grade, "score", 0.0)
        comment = getattr(grade, "reasoning", "")
    else:
        raw_score = grade.get("score", 0.0)
        comment = grade.get("reasoning") or grade.get("explanation", "")
    score = _quantize_score(raw_score)
    return {"key": key, "score": score, "comment": comment}


def _quantize_score(value: Any) -> float:
    """Snap evaluator scores to the five-point scale used by the dashboard."""

    allowed_scores = (0.0, 0.25, 0.5, 0.75, 1.0)
    try:
        numeric_score = min(max(float(value), 0.0), 1.0)
    except Exception:
        return 0.0
    return min(allowed_scores, key=lambda allowed: abs(allowed - numeric_score))


def _contexts_to_text(outputs: dict, limit: int = 30000) -> str:
    contexts = outputs.get("contexts") or []
    if not contexts:
        return "No retrieved context."
    sources = outputs.get("sources") or []
    blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        source_label = ""
        if index - 1 < len(sources):
            source_label = str(sources[index - 1] or "").strip()
        header = f"[Local {index}]"
        if source_label:
            header = f"{header} {source_label}"
        blocks.append(f"{header}\n{context}")
    return "\n\n".join(blocks)[:limit]


def _testset_hash(testset: pd.DataFrame) -> str:
    normalized = testset[["question", "ground_truth"]].to_csv(index=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _normalize_testset_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize required question and ground-truth columns."""

    dataframe = dataframe.copy()
    rename_map = {}
    if "user_input" in dataframe.columns and "question" not in dataframe.columns:
        rename_map["user_input"] = "question"
    if "reference" in dataframe.columns and "ground_truth" not in dataframe.columns:
        rename_map["reference"] = "ground_truth"
    if "answer" in dataframe.columns and "ground_truth" not in dataframe.columns:
        rename_map["answer"] = "ground_truth"
    dataframe = dataframe.rename(columns=rename_map)

    if "question" not in dataframe.columns:
        raise ValueError("Evaluation CSV must include a question column.")
    if "ground_truth" not in dataframe.columns:
        raise ValueError("Evaluation CSV must include a ground_truth, reference, or answer column.")

    dataframe["question"] = dataframe["question"].astype(str).str.strip()
    dataframe["ground_truth"] = dataframe["ground_truth"].apply(_coerce_ground_truth)
    dataframe = dataframe[(dataframe["question"] != "") & (dataframe["ground_truth"] != "")]
    if len(dataframe) != 50:
        raise ValueError(f"Evaluation CSV must contain exactly 50 populated rows; found {len(dataframe)}.")
    return dataframe[["question", "ground_truth"]].reset_index(drop=True)


def _coerce_ground_truth(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if pd.isna(value):
        return ""
    return str(value).strip()


def _extract_metric_means(scores_df: pd.DataFrame) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for metric in EVAL_METRICS:
        column = f"feedback.{metric}"
        if column in scores_df.columns:
            metrics[metric] = _safe_float(scores_df[column].mean())
        elif metric in scores_df.columns:
            metrics[metric] = _safe_float(scores_df[metric].mean())
        else:
            metrics[metric] = None
    return metrics


def _extract_latency_metrics(execution_times: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        float(item["execution_time"])
        for item in execution_times
        if item.get("execution_time") is not None
    ]
    if not values:
        return {
            "average_latency": None,
            "p50_latency": None,
            "p99_latency": None,
        }
    array = np.array(values, dtype=float)
    return {
        "average_latency": round(float(np.mean(array)), 1),
        "p50_latency": round(float(np.percentile(array, 50)), 1),
        "p99_latency": round(float(np.percentile(array, 99)), 1),
    }


def _extract_execution_time_records(
    scores_df: pd.DataFrame,
    fallback_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if "execution_time" not in scores_df.columns:
        return fallback_records

    records: list[dict[str, Any]] = []
    for _, row in scores_df.iterrows():
        execution_time = _safe_float(row.get("execution_time"))
        if execution_time is None:
            continue
        question = row.get("inputs.question")
        records.append(
            {
                "question": "" if question is None or pd.isna(question) else str(question),
                "execution_time": round(execution_time, 3),
            }
        )
    return records or fallback_records


def _eval_log_row(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    return {
        "assessment_date": _format_assessment_date(str(payload.get("timestamp_utc") or "")),
        "setting": payload.get("config_slug") or COLLECTION_SLUGS.get(payload.get("collection_name", ""), ""),
        "faithfulness": _safe_float(metrics.get("faithfulness")),
        "answer_relevance": _safe_float(metrics.get("answer_relevancy")),
        "context_precision": _safe_float(metrics.get("context_precision")),
        "context_recall": _safe_float(metrics.get("context_recall")),
        "latency_average": _round_latency(metrics.get("average_latency")),
        "latency_p50": _round_latency(metrics.get("p50_latency")),
        "latency_p99": _round_latency(metrics.get("p99_latency")),
    }


def _format_assessment_date(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _round_latency(value: Any) -> float | None:
    latency = _safe_float(value)
    if latency is None:
        return None
    return round(latency, 1)


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _json_safe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    records = dataframe.to_dict(orient="records")
    safe_records: list[dict[str, Any]] = []
    for record in records:
        safe_records.append({key: _json_safe_value(value) for key, value in record.items()})
    return safe_records


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload: Any,
) -> None:
    if progress_callback:
        progress_callback(payload)
