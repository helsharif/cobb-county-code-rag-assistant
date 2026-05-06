"""Background runner for LangSmith evaluation jobs launched from Streamlit."""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import COLLECTION_SLUGS, ROOT_DIR
from src.evaluation import run_langsmith_evaluation


EVAL_STATUS_DIR = ROOT_DIR / "eval_status"
def main() -> int:
    parser = argparse.ArgumentParser(description="Run LangSmith evaluation for one Chroma collection.")
    parser.add_argument("--collection-name", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    collection_name = args.collection_name
    status_path = _status_path(collection_name)
    started_at_utc = _now_utc()
    _write_status(
        status_path,
        {
            "status": "running",
            "phase": "starting",
            "message": "Starting LangSmith evaluation.",
            "current": 0,
            "total": 50,
            "started_at_utc": started_at_utc,
            "updated_at_utc": started_at_utc,
        },
    )

    def progress_callback(update: dict[str, Any]) -> None:
        _write_status(
            status_path,
            {
                "status": "running",
                "started_at_utc": started_at_utc,
                "updated_at_utc": _now_utc(),
                **update,
            },
        )

    try:
        results = run_langsmith_evaluation(collection_name, progress_callback=progress_callback)
        _write_status(
            status_path,
            {
                "status": "complete",
                "started_at_utc": results.get("started_at_utc"),
                "finished_at_utc": _now_utc(),
                "phase": "complete",
                "message": "Evaluation complete. Refresh the Settings & Eval tab to load the saved results.",
                "current": results.get("question_count"),
                "total": results.get("question_count"),
                "result_file": str(_result_path(collection_name).relative_to(ROOT_DIR)),
                "experiment_url": results.get("experiment_url"),
            },
        )
        return 0
    except Exception as exc:
        _write_status(
            status_path,
            {
                "status": "error",
                "finished_at_utc": _now_utc(),
                "phase": "error",
                "message": "Evaluation failed. See error details below.",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            },
        )
        return 1


def _status_path(collection_name: str) -> Path:
    slug = _slug(collection_name)
    return EVAL_STATUS_DIR / f"{slug}_status.json"


def _result_path(collection_name: str) -> Path:
    slug = _slug(collection_name)
    return ROOT_DIR / "eval_results" / f"{slug}_results.json"


def _slug(collection_name: str) -> str:
    return COLLECTION_SLUGS.get(collection_name, collection_name.replace("cobb_code_docs_", ""))


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    temp_path.replace(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
