"""LangSmith evaluation helpers for the Cobb County RAG app."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict

from src.agent import CobbCountyRAGAgent
from src.config import (
    DOCLING_COLLECTION_NAME,
    ORIGINAL_COLLECTION_NAME,
    ROOT_DIR,
    get_settings,
)
from src.retriever import search_documents


logger = logging.getLogger(__name__)

TESTSET_PATH = ROOT_DIR / "eval_testset" / "cobb_county_testset.csv"
EVAL_RESULTS_DIR = ROOT_DIR / "eval_results"
EVAL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LANGSMITH_DATASET_PREFIX = "cobb-county-rag-eval-testset"

CONFIG_LABELS = {
    ORIGINAL_COLLECTION_NAME: "Original",
    DOCLING_COLLECTION_NAME: "Docling",
}

ProgressCallback = Callable[[dict[str, Any]], None]


class BinaryGrade(TypedDict):
    explanation: Annotated[str, ..., "Briefly explain the score."]
    score: Annotated[bool, ..., "True if the criterion is satisfied; otherwise false."]


@dataclass(frozen=True)
class EvaluationResultFile:
    collection_name: str
    label: str
    path: Path


def eval_result_file(collection_name: str) -> EvaluationResultFile:
    """Return the cache path for one retrieval backend."""

    label = CONFIG_LABELS.get(collection_name, collection_name)
    slug = label.lower().replace(" ", "_")
    return EvaluationResultFile(
        collection_name=collection_name,
        label=label,
        path=EVAL_RESULTS_DIR / f"eval_results_{slug}.json",
    )


def load_eval_results(collection_name: str) -> dict[str, Any] | None:
    """Load persisted LangSmith evaluation results for the selected vector store."""

    result_file = eval_result_file(collection_name)
    if not result_file.path.exists():
        return None
    with result_file.path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_eval_results(collection_name: str, payload: dict[str, Any]) -> Path:
    """Persist LangSmith results so Streamlit can display them after restarts."""

    result_file = eval_result_file(collection_name)
    result_file.path.parent.mkdir(parents=True, exist_ok=True)
    with result_file.path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return result_file.path


def ensure_testset() -> pd.DataFrame:
    """Load the fixed 20-question evaluation test set from eval_testset/."""

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
            "Fixed 20-question Cobb County building and fire code RAG evaluation set "
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

    _emit_progress(progress_callback, phase="loading_testset", message="Loading fixed 20-question CSV test set.")
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
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    label = CONFIG_LABELS.get(collection_name, collection_name)

    @traceable(name=f"cobb_county_rag_{label.lower()}_target")
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
        agent = CobbCountyRAGAgent(collection_name=collection_name)
        result = agent.answer(question)
        docs, sources = search_documents(question, collection_name=collection_name)
        contexts = [doc.page_content for doc in docs]
        _emit_progress(
            progress_callback,
            phase="answering_questions",
            message=f"Completed question {answer_counter['count']} of {len(testset)}.",
            current=answer_counter["count"],
            total=len(testset),
            question=question,
        )
        return {
            "answer": result.answer,
            "contexts": contexts,
            "sources": [source.source for source in sources],
            "used_local": result.used_local,
            "used_web": result.used_web,
        }

    evaluators = _build_langsmith_evaluators()
    experiment_prefix = f"cobb-county-rag-{label.lower()}-langsmith"
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
    payload = {
        "collection_name": collection_name,
        "config_label": label,
        "evaluation_backend": "LangSmith",
        "started_at_utc": started_at_utc,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "testset_path": str(TESTSET_PATH.relative_to(ROOT_DIR)),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "experiment_name": experiment_results.experiment_name,
        "experiment_id": str(experiment_results.experiment_id),
        "experiment_url": experiment_results.url,
        "question_count": len(testset),
        "metrics": metrics,
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
    get_settings()
    grader = ChatOpenAI(model="gpt-4o", temperature=0)

    faithfulness_llm = grader.with_structured_output(BinaryGrade, method="json_schema", strict=True)
    answer_relevance_llm = grader.with_structured_output(BinaryGrade, method="json_schema", strict=True)
    context_precision_llm = grader.with_structured_output(BinaryGrade, method="json_schema", strict=True)
    context_recall_llm = grader.with_structured_output(BinaryGrade, method="json_schema", strict=True)

    def faithfulness(inputs: dict, outputs: dict) -> dict:
        facts = _contexts_to_text(outputs)
        grade = faithfulness_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You grade whether an answer is faithful to retrieved context. "
                        "Return true only if the answer is supported by the facts and does not add unsupported claims."
                    ),
                },
                {"role": "user", "content": f"FACTS:\n{facts}\n\nANSWER:\n{outputs.get('answer', '')}"},
            ]
        )
        return _grade_to_feedback("faithfulness", grade)

    def answer_relevancy(inputs: dict, outputs: dict) -> dict:
        grade = answer_relevance_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You grade whether an answer directly addresses the user's question. "
                        "Return true if the answer is relevant and helpful, even if conservative."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{inputs.get('question', '')}\n\nANSWER:\n{outputs.get('answer', '')}",
                },
            ]
        )
        return _grade_to_feedback("answer_relevancy", grade)

    def context_precision(inputs: dict, outputs: dict) -> dict:
        facts = _contexts_to_text(outputs)
        grade = context_precision_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You grade whether the retrieved contexts are relevant to the question. "
                        "Return true if most retrieved context is useful for answering the question."
                    ),
                },
                {"role": "user", "content": f"QUESTION:\n{inputs.get('question', '')}\n\nCONTEXTS:\n{facts}"},
            ]
        )
        return _grade_to_feedback("context_precision", grade)

    def context_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        facts = _contexts_to_text(outputs)
        grade = context_recall_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You grade whether retrieved contexts contain the information needed to support "
                        "the reference answer. Return true if the contexts cover the important facts."
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
            ]
        )
        return _grade_to_feedback("context_recall", grade)

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _grade_to_feedback(key: str, grade: dict) -> dict:
    score = 1.0 if grade.get("score") else 0.0
    return {"key": key, "score": score, "comment": grade.get("explanation", "")}


def _contexts_to_text(outputs: dict, limit: int = 8000) -> str:
    contexts = outputs.get("contexts") or []
    if not contexts:
        return "No retrieved context."
    return "\n\n".join(str(context) for context in contexts)[:limit]


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
    if len(dataframe) != 20:
        raise ValueError(f"Evaluation CSV must contain exactly 20 populated rows; found {len(dataframe)}.")
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
