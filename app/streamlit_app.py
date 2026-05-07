"""Streamlit frontend for the Cobb County Agentic RAG app."""

from __future__ import annotations

import logging
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    COLLECTION_OPTIONS,
    COLLECTION_SLUGS,
    LEGACY_COLLECTION_LABELS,
    OPTION_1_LABEL,
    OPTION_2_LABEL,
    OPTION_3_LABEL,
    OPTION_4_LABEL,
    ORIGINAL_COLLECTION_NAME,
    get_settings,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


st.set_page_config(page_title="Cobb County Codes RAG", page_icon="CC", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stForm"] {
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-radius: 0.7rem;
        padding: 1rem 1rem 0.85rem;
        background: #ffffff;
        box-shadow: 0 0.5rem 1.4rem rgba(17, 24, 39, 0.05);
    }
    div[data-testid="stTextInput"] input {
        font-size: 1.02rem;
        line-height: 1.45;
        min-height: 3.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Cobb County Building & Fire Codes RAG")
st.caption("Grounded answers from local PDFs first, with web fallback when retrieval is weak.")

RETRIEVAL_OPTIONS = COLLECTION_OPTIONS
PAGE_OPTIONS = ["Ask", "Settings & Eval", "About the App"]
EVAL_RESULTS_DIR = ROOT_DIR / "eval_results"
EVAL_STATUS_DIR = ROOT_DIR / "eval_status"
EVAL_AUTO_REFRESH_SECONDS = 20
NO_ANSWER = "I could not find a reliable answer in the available documents or web sources."


def get_query_param(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def init_persistent_state() -> None:
    query_page = get_query_param("page", "Ask")
    if query_page not in PAGE_OPTIONS:
        query_page = "Ask"
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = query_page

    query_backend = get_query_param("backend", OPTION_1_LABEL)
    query_backend = LEGACY_COLLECTION_LABELS.get(query_backend, query_backend)
    if query_backend not in RETRIEVAL_OPTIONS:
        query_backend = OPTION_1_LABEL
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = RETRIEVAL_OPTIONS[query_backend]


def sync_query_state(page: str | None = None, backend_label: str | None = None) -> None:
    if page and st.query_params.get("page") != page:
        st.query_params["page"] = page
    if backend_label and st.query_params.get("backend") != backend_label:
        st.query_params["backend"] = backend_label


def on_backend_change() -> None:
    selected_label = st.session_state.get("backend_label", get_selected_mode_label())
    selected_label = LEGACY_COLLECTION_LABELS.get(selected_label, selected_label)
    if selected_label not in RETRIEVAL_OPTIONS:
        selected_label = OPTION_1_LABEL
    st.session_state.collection_name = RETRIEVAL_OPTIONS[selected_label]
    sync_query_state(backend_label=selected_label)


def on_page_change() -> None:
    selected_page = st.session_state.get("selected_page", "Ask")
    if selected_page not in PAGE_OPTIONS:
        selected_page = "Ask"
    sync_query_state(page=selected_page, backend_label=get_selected_mode_label())


init_persistent_state()


def get_selected_collection() -> str:
    return st.session_state.get("collection_name", ORIGINAL_COLLECTION_NAME)


def get_selected_mode_label() -> str:
    selected = get_selected_collection()
    for label, collection_name in RETRIEVAL_OPTIONS.items():
        if collection_name == selected:
            return label
    return OPTION_1_LABEL


def get_agent(collection_name: str):
    from src.agent import CobbCountyRAGAgent

    return CobbCountyRAGAgent(collection_name=collection_name)


@st.cache_data(ttl=60, show_spinner=False)
def cached_vectorstore_exists(collection_name: str) -> bool:
    from src.retriever import vectorstore_exists

    return vectorstore_exists(collection_name=collection_name)


def render_chat_tab() -> None:
    collection_name = get_selected_collection()
    mode_label = get_selected_mode_label()
    st.caption(f"Retrieval backend: {mode_label}")

    if not cached_vectorstore_exists(collection_name):
        st.warning(
            f"No local vector index found for `{collection_name}`. Build it with "
            "`python -m src.ingestion --rebuild --pipeline all`, then restart or refresh the app."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("ask_question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question",
            placeholder="Ask about Cobb County building or fire code requirements",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    if submitted and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Retrieving local code documents..."):
                result = get_agent(collection_name).answer(question)
            source_mode = (
                "local documents and web search"
                if result.used_local and result.used_web
                else "local documents"
                if result.used_local
                else "web search"
                if result.used_web
                else "no reliable source"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer or NO_ANSWER,
                    "sources": result.sources,
                    "source_mode": source_mode,
                    "route_reason": getattr(result, "route_reason", ""),
                    "route_needs_web": getattr(result, "route_needs_web", False),
                    "web_search_attempted": getattr(result, "web_search_attempted", False),
                    "web_search_error": getattr(result, "web_search_error", ""),
                    "web_query": getattr(result, "web_query", ""),
                    "retrieval_mode": mode_label,
                }
            )
        except Exception as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Unable to answer right now: {exc}",
                    "sources": [],
                    "source_mode": "error",
                }
            )

    for exchange in _latest_first_exchanges(st.session_state.messages):
        for message in exchange:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("source_mode"):
                    st.caption(f"Answer source: {message['source_mode']}")
                if message["role"] == "assistant" and message.get("retrieval_mode"):
                    st.caption(f"Retrieval backend: {message['retrieval_mode']}")
                if message["role"] == "assistant" and message.get("route_reason"):
                    st.caption(f"Routing note: {message['route_reason']}")
                if message["role"] == "assistant":
                    web_status = _web_status_caption(message)
                    if web_status:
                        st.caption(web_status)
                    if message.get("web_search_error"):
                        st.caption(f"Web search error: {message['web_search_error']}")
                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(source)


def _latest_first_exchanges(messages: list[dict]) -> list[list[dict]]:
    exchanges: list[list[dict]] = []
    current: list[dict] = []
    for message in messages:
        if message["role"] == "user" and current:
            exchanges.append(current)
            current = []
        current.append(message)
    if current:
        exchanges.append(current)
    return list(reversed(exchanges))


def _web_status_caption(message: dict) -> str:
    if not (message.get("route_needs_web") or message.get("web_search_attempted") or message.get("web_query")):
        return ""
    requested = "requested" if message.get("route_needs_web") else "not requested"
    attempted = "attempted" if message.get("web_search_attempted") else "not attempted"
    query = message.get("web_query", "")
    if query:
        return f"Web search: {attempted}; router: {requested}; query: {query}"
    return f"Web search: {attempted}; router: {requested}"


def render_settings_eval_tab() -> None:
    st.subheader("Settings & Eval")
    st.write(
        "Switch retrieval configurations and review persisted LangSmith evaluation metrics for each RAG pipeline."
    )
    current_collection = get_selected_collection()
    labels = list(RETRIEVAL_OPTIONS)
    current_label = get_selected_mode_label()
    if st.session_state.get("backend_label") != current_label:
        st.session_state.backend_label = current_label
    selected_label = st.radio(
        "Retrieval configuration",
        options=labels,
        index=labels.index(current_label) if current_label in labels else 0,
        horizontal=False,
        key="backend_label",
        on_change=on_backend_change,
    )
    selected_collection = RETRIEVAL_OPTIONS[selected_label]

    if selected_label == OPTION_1_LABEL:
        st.info(
            "Option 1: PyPDF + Chromadb uses the first PDF extraction pipeline. "
            "This preserves the app's original retrieval behavior."
        )
    elif selected_label == OPTION_2_LABEL:
        st.info(
            "Option 2: Docling + Chromadb uses layout-aware PDF parsing before chunking and embedding. "
            "It may improve retrieval from layout-heavy PDFs, tables, headings, sections, and regulatory documents."
        )
    elif selected_label == OPTION_3_LABEL:
        st.info(
            "Option 3: Docling + Chroma + BM25 Hybrid Search uses the Docling Chroma collection for vector "
            "retrieval plus a local BM25 keyword corpus, then fuses rankings before answer generation."
        )
    else:
        st.info(
            "Option 4: Docling + Chroma + Query Expansion + BM25 Hybrid Search reuses the Docling Chroma "
            "and BM25 corpus, expands the original question into five retrieval queries, then fuses and deduplicates "
            "results before answer generation. Latency metrics include the extra expansion LLM call."
        )

    st.code(selected_collection, language="text")
    if selected_collection != current_collection:
        st.caption("New questions will use the selected backend. Existing chat messages are left unchanged.")

    settings = get_settings()
    expansion_state = "enabled" if settings.context_expansion_enabled else "disabled"
    st.caption(
        f"Context expansion is {expansion_state} "
        f"(mode={settings.context_expansion_mode}, neighbor_window={settings.context_neighbor_window}, "
        f"max_blocks={settings.context_max_expanded_docs}, max_chars={settings.context_max_chars})."
    )

    st.divider()
    st.subheader("LangSmith Metrics")
    st.write(
        "Metrics are loaded from saved local result files first. Running or re-running evaluation uses the fixed "
        "`eval_testset/cobb_county_testset.csv` file, creates or reuses a LangSmith dataset, and retrieves the "
        "LangSmith experiment scores for this dashboard."
    )
    st.caption(
        "Quality metrics use a five-point evaluator scale: 0.00, 0.25, 0.50, 0.75, and 1.00. "
        "The judge gives partial credit for mostly correct answers while staying strict on technical code facts."
    )

    status = load_eval_status(selected_collection)
    results = load_eval_results(selected_collection)
    if results:
        _render_eval_results(results)
        button_label = "Re-run Evaluation"
    else:
        st.warning("No saved evaluation metrics found for this vector store.")
        button_label = "Run Evaluation Metrics"

    if status:
        _render_eval_status(status)

    disabled = bool(status and status.get("status") == "running")
    if st.button(button_label, type="primary", disabled=disabled):
        try:
            start_evaluation_process(selected_collection)
            st.success("LangSmith evaluation started in the background. This dashboard will update automatically.")
            st.rerun()
        except Exception as exc:
            write_eval_status(
                selected_collection,
                {
                    "status": "error",
                    "phase": "launch_error",
                    "message": "Evaluation process could not be launched.",
                    "error": str(exc),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            st.error(f"Could not start evaluation: {exc}")


def load_eval_results(collection_name: str) -> dict | None:
    result_path = eval_result_path(collection_name)
    if not result_path.exists():
        return None
    try:
        with result_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def eval_result_path(collection_name: str) -> Path:
    slug = get_collection_slug(collection_name)
    return EVAL_RESULTS_DIR / f"{slug}_results.json"


def eval_status_path(collection_name: str) -> Path:
    slug = get_collection_slug(collection_name)
    return EVAL_STATUS_DIR / f"{slug}_status.json"


def load_eval_status(collection_name: str) -> dict | None:
    status_path = eval_status_path(collection_name)
    if not status_path.exists():
        return None
    try:
        with status_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def write_eval_status(collection_name: str, payload: dict) -> None:
    status_path = eval_status_path(collection_name)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = status_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    temp_path.replace(status_path)


def start_evaluation_process(collection_name: str) -> None:
    EVAL_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_eval_status(
        collection_name,
        {
            "status": "running",
            "phase": "launching",
            "message": "Launching background LangSmith evaluator.",
            "current": 0,
            "total": 50,
            "started_at_utc": now,
            "updated_at_utc": now,
        },
    )
    executable = sys.executable
    if sys.platform.startswith("win"):
        pythonw_path = Path(sys.executable).with_name("pythonw.exe")
        if pythonw_path.exists():
            executable = str(pythonw_path)
    command = [
        executable,
        "-m",
        "src.evaluation_runner",
        "--collection-name",
        collection_name,
    ]
    kwargs = {
        "cwd": str(ROOT_DIR),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        kwargs.pop("start_new_session", None)
    subprocess.Popen(command, **kwargs)


def get_collection_display_label(collection_name: str) -> str:
    for label, option_collection in RETRIEVAL_OPTIONS.items():
        if option_collection == collection_name:
            return label
    return collection_name.replace("cobb_code_docs_", "")


def get_collection_slug(collection_name: str) -> str:
    return COLLECTION_SLUGS.get(collection_name, collection_name.replace("cobb_code_docs_", ""))


def _render_eval_status(status: dict) -> None:
    status_state = status.get("status")
    phase = str(status.get("phase") or status_state or "unknown").replace("_", " ").title()
    message = status.get("message", "")
    current = status.get("current")
    total = status.get("total")
    started_at = status.get("started_at_utc")
    updated_at = status.get("updated_at_utc")

    if status_state == "running":
        st.info(f"Evaluation running: {phase}. {message}")
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            progress_value = min(max(current / total, 0.0), 1.0)
            st.progress(progress_value, text=f"{current} of {total} questions processed")
        if started_at:
            st.caption(f"Started: {started_at} UTC | Elapsed: {_elapsed_since(started_at)}")
        if updated_at:
            st.caption(f"Last status update: {updated_at} UTC")
        if status.get("question"):
            st.caption(f"Current question: {status['question']}")
        st.caption(
            f"This dashboard polls every {EVAL_AUTO_REFRESH_SECONDS} seconds while evaluation is running. "
            "When the status changes to complete, saved LangSmith metrics will load automatically."
        )
        if st.button("Refresh now", type="secondary"):
            st.rerun()
        _enable_eval_auto_refresh(EVAL_AUTO_REFRESH_SECONDS)
    elif status_state == "error":
        st.error(f"Last evaluation failed: {status.get('error', 'Unknown error')}")
    elif status_state == "complete":
        st.success(
            f"Last evaluation completed at {status.get('finished_at_utc', 'unknown')} UTC. "
            "Saved metrics are displayed above when available."
        )
        if status.get("experiment_url"):
            st.markdown(f"[Open LangSmith experiment]({status['experiment_url']})")


def _enable_eval_auto_refresh(seconds: int) -> None:
    milliseconds = max(seconds, 1) * 1000
    components.html(
        f"""
        <script>
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, {milliseconds});
        </script>
        """,
        height=0,
        width=0,
    )


def _elapsed_since(timestamp: str) -> str:
    try:
        started = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - started.astimezone(timezone.utc)
        minutes, seconds = divmod(max(int(elapsed.total_seconds()), 0), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    except Exception:
        return "unknown"


def _render_eval_results(results: dict) -> None:
    timestamp = results.get("timestamp_utc", "unknown")
    question_count = results.get("question_count", "unknown")
    metrics = results.get("metrics", {})
    backend = results.get("evaluation_backend", "evaluation")
    st.caption(f"Last {backend} run: {timestamp} UTC | Questions: {question_count}")
    if results.get("dataset_name"):
        st.caption(f"LangSmith dataset: {results['dataset_name']}")
    if results.get("experiment_url"):
        st.markdown(f"[Open LangSmith experiment]({results['experiment_url']})")

    _render_metric_glossary()
    _persist_displayed_eval_log(results)

    columns = st.columns(5)
    metric_specs = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevance", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall"),
    ]
    for column, (label, key) in zip(columns, metric_specs):
        value = metrics.get(key)
        with column:
            _render_metric_card(label, value)
    with columns[-1]:
        _render_latency_card(metrics)

    rows = results.get("rows", [])
    if rows:
        with st.expander("Evaluation details"):
            st.dataframe(rows, use_container_width=True)


def _persist_displayed_eval_log(results: dict) -> None:
    try:
        from src.evaluation import append_eval_results_log

        append_eval_results_log(results)
    except Exception as exc:
        st.caption(f"Evaluation log could not be updated: {exc}")


def _render_metric_glossary() -> None:
    with st.expander("Metric definitions", expanded=False):
        st.markdown(
            """
            - **Faithfulness:** Did the model invent facts? The judge extracts factual claims from the answer and scores the share supported by retrieved context.
            - **Answer relevance:** Did the model answer the right question? The judge scores how directly and usefully the answer addresses the user's intent.
            - **Context precision:** How much useful signal was in the retrieved context? The judge scores the ratio of relevant chunks or information to total retrieved context.
            - **Context recall:** Did retrieval find the facts needed by the reference answer? The judge scores required facts found in context divided by required facts in the golden answer.
            - **Latency:** How long did the selected RAG configuration take per question? The card reports average, median (P50), and 99th percentile (P99) execution time in seconds.

            Quality metrics use five score buckets: **0.00** no meaningful support, **0.25** minimal support,
            **0.50** partially correct, **0.75** mostly correct, and **1.00** fully correct. The evaluator is strict
            with numerical values, dates, dimensions, fire ratings, fees, code sections, and procedural requirements,
            but flexible with equivalent wording.
            """
        )


def _render_metric_card(label: str, value: object) -> None:
    score = _coerce_score(value)
    if score is None:
        color = "#6b7280"
        background = "#f3f4f6"
        text = "N/A"
        band = "No score"
    else:
        color, background, band = _score_style(score)
        text = f"{score:.2f}"
    st.markdown(
        f"""
        <div style="
            border-left: 0.42rem solid {color};
            background: {background};
            border-radius: 0.45rem;
            padding: 0.8rem 0.9rem;
            min-height: 6.2rem;
            border-top: 1px solid rgba(17, 24, 39, 0.08);
            border-right: 1px solid rgba(17, 24, 39, 0.08);
            border-bottom: 1px solid rgba(17, 24, 39, 0.08);
            box-sizing: border-box;
        ">
            <div style="font-size: 0.86rem; color: #374151; font-weight: 650;">{label}</div>
            <div style="font-size: 1.75rem; color: #111827; font-weight: 750; line-height: 1.25;">{text}</div>
            <div style="font-size: 0.78rem; color: {color}; font-weight: 700;">{band}</div>
            <div style="font-size: 0.78rem; line-height: 1.2; visibility: hidden;">&nbsp;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_latency_card(metrics: dict) -> None:
    avg = _coerce_latency(metrics.get("average_latency"))
    p50 = _coerce_latency(metrics.get("p50_latency"))
    p99 = _coerce_latency(metrics.get("p99_latency"))
    if avg is None or p50 is None or p99 is None:
        color = "#6b7280"
        background = "#f3f4f6"
        text = "N/A"
        band = "No score"
    else:
        color, background, band = _latency_style(p99)
        text = f"{avg:.1f} | {p50:.1f} | {p99:.1f}"
    st.markdown(
        f"""
        <div style="
            border-left: 0.42rem solid {color};
            background: {background};
            border-radius: 0.45rem;
            padding: 0.8rem 0.9rem;
            min-height: 6.2rem;
            border-top: 1px solid rgba(17, 24, 39, 0.08);
            border-right: 1px solid rgba(17, 24, 39, 0.08);
            border-bottom: 1px solid rgba(17, 24, 39, 0.08);
        ">
            <div style="font-size: 0.86rem; color: #374151; font-weight: 650;">Latency (secs)</div>
            <div style="font-size: 1.35rem; color: #111827; font-weight: 750; line-height: 1.35; white-space: nowrap;">{text}</div>
            <div style="font-size: 0.78rem; color: #374151; font-weight: 650;">Avg | P50 | P99</div>
            <div style="font-size: 0.78rem; color: {color}; font-weight: 700;">{band}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _coerce_score(value: object) -> float | None:
    try:
        if value is None:
            return None
        score = float(value)
        if score != score:
            return None
        return max(0.0, min(score, 1.0))
    except Exception:
        return None


def _coerce_latency(value: object) -> float | None:
    try:
        if value is None:
            return None
        latency = float(value)
        if latency != latency:
            return None
        return max(0.0, latency)
    except Exception:
        return None


def _score_style(score: float) -> tuple[str, str, str]:
    if score < 0.60:
        return "#b91c1c", "#fef2f2", "Needs attention"
    if score < 0.80:
        return "#92400e", "#fffbeb", "Moderate"
    return "#166534", "#f0fdf4", "Strong"


def _latency_style(p99_latency: float) -> tuple[str, str, str]:
    if p99_latency > 15.0:
        return "#b91c1c", "#fef2f2", "Needs attention"
    if p99_latency > 8.0:
        return "#92400e", "#fffbeb", "Moderate"
    return "#166534", "#f0fdf4", "Strong"


def render_about_tab() -> None:
    settings = get_settings()
    runtime_model = settings.openai_model if settings.llm_provider == "openai" else settings.gemini_model
    evaluator_model = settings.eval_judge_model
    embedding_model = (
        settings.openai_embedding_model
        if settings.embedding_provider == "openai"
        else settings.gemini_embedding_model
    )

    st.subheader("What This App Does")
    st.write(
        "This app answers questions about Cobb County, Georgia building and fire code materials. "
        "It uses a lightweight LLM router to decide whether a question may need current web verification, "
        "then searches indexed local documents, runs a strict evidence gate, and uses web search when the router "
        "or retrieval-quality checks say it is needed."
    )
    st.write(
        "After initial retrieval, the app deterministically expands small retrieved chunks into full page/range context "
        "or neighboring chunks before the adequacy gate. This helps checklist items, table rows, and bullet values remain "
        "visible when the first retrieved chunk stops just before the answer."
    )

    st.image(
        str(ROOT_DIR / "assets" / "Rag Flow Chart.png"),
        caption="Agentic RAG architecture: local document retrieval first, web fallback when evidence is weak or current-code verification is needed.",
        use_container_width=True,
    )

    st.subheader("Under the Hood")
    ingest_col, query_col = st.columns(2)
    with ingest_col:
        st.markdown("**Index build**")
        st.write(
            "PDFs under `data/` are indexed into four optional retrieval backends: Option 1 uses the PyPDF-based "
            "pipeline, Option 2 uses Docling-enhanced parsing that first exports layout-aware Markdown, and Option 3 "
            "uses the Docling Chroma collection with a local BM25 keyword corpus. Option 4 reuses those local indexes "
            "and adds LLM query expansion at retrieval time."
        )
        st.graphviz_chart(
            """
            digraph {
                graph [rankdir=TB, bgcolor="transparent", pad="0.2"];
                node [shape=box, style="rounded,filled", color="#94a3b8", fillcolor="#f8fafc", fontname="Arial", fontsize=10];
                edge [color="#64748b", arrowsize=0.65];
                pdf [label="Local PDFs"];
                load [label="PyPDF or Docling\\nPDF parsing"];
                structure [label="Docling mode:\\nTOC/bookmark ranges\\nwith overlap fallback"];
                split [label="Chunk text\\nwith overlap"];
                embed [label="Create embeddings"];
                store [label="Persist in Chroma plus BM25 corpus\\nOption 1, Option 2, or Option 3"];
                expansion [label="Option 4 reuses local hybrid indexes\\nwith query expansion"];
                pdf -> load -> structure -> split -> embed -> store;
                store -> expansion;
            }
            """,
            use_container_width=True,
        )
    with query_col:
        st.markdown("**Question answering**")
        st.write(
            "A lightweight LLM router first classifies whether the question may need local retrieval, web search, or both. "
            "The selected retrieval backend controls which Chroma collection or local hybrid strategy is searched before the agent evaluates evidence quality."
        )
        st.graphviz_chart(
            """
            digraph {
                graph [rankdir=TB, bgcolor="transparent", pad="0.2"];
                node [shape=box, style="rounded,filled", color="#94a3b8", fillcolor="#f8fafc", fontname="Arial", fontsize=10];
                edge [color="#64748b", arrowsize=0.65];
                q [label="Question"];
                router [label="LLM router"];
                retrieve [label="Retrieve local chunks"];
                expand [label="Option 4:\\nexpand into 5 queries"];
                expand_context [label="Deterministic\\ncontext expansion"];
                judge [label="Strict JSON\\nevidence gate"];
                cite [label="Answer with citations"];
                abstain [label="Conservative\\nabstention"];
                fallback [label="Search web if needed"];
                select [label="Selected backend\\nOption 1, 2, 3, or 4"];
                q -> router -> select -> expand -> retrieve -> expand_context -> judge -> cite;
                router -> fallback;
                judge -> fallback -> cite;
                judge -> abstain;
            }
            """,
            use_container_width=True,
        )

    st.subheader("Why This Is Agentic RAG")
    col1, col2, col3 = st.columns(3)
    col1.metric("Router", "LLM signal")
    col2.metric("Retrieval", "Local first")
    col3.metric("Fallback", "SerpAPI web")

    st.table(
        [
            {"Layer": "Frontend", "What it does": "Provides a simple chat UI and displays sources.", "Tech": "Streamlit"},
            {"Layer": "Router", "What it does": "Classifies whether the query may need local docs, web search, or both.", "Tech": "LangChain + LLM"},
            {"Layer": "Retriever", "What it does": "Finds relevant chunks from the selected local index.", "Tech": "Chroma, local BM25 fusion, or query expansion"},
            {"Layer": "Context expansion", "What it does": "Expands small hits to page/range context or neighboring chunks before adequacy checks.", "Tech": "JSONL sidecars + deterministic rules"},
            {"Layer": "Agent logic", "What it does": "Combines router signal, retrieval scores, expanded context, and a strict JSON evidence gate.", "Tech": "LangChain"},
            {"Layer": "Generation", "What it does": "Synthesizes a short answer from retrieved evidence only.", "Tech": f"{settings.llm_provider}: {runtime_model}"},
            {"Layer": "Deployment", "What it does": "Runs locally, in Docker, or on Streamlit Community Cloud.", "Tech": "Docker + Streamlit"},
        ]
    )

    st.caption(
        f"Current model configuration: runtime LLM `{runtime_model}`, evaluation judge `{evaluator_model}`, "
        f"and embedding model `{embedding_model}`. Changing the runtime or judge model does not require rebuilding "
        "the indexes as long as the embedding model and source documents stay the same."
    )

    st.subheader("Settings and Evaluation")
    st.write(
        "The Settings & Eval tab lets users choose between four retrieval modes and inspect persisted LangSmith metrics. "
        "Option 1: PyPDF + Chromadb uses the first PDF text extraction pipeline. Option 2: Docling + Chromadb uses "
        "layout-aware parsing before content is embedded into Chroma. Option 3: Docling + Chroma + BM25 Hybrid Search "
        "uses Docling chunks with local BM25 keyword retrieval and Chroma vector retrieval, which can help compare "
        "keyword-heavy and semantic retrieval behavior. Option 4 adds LLM query expansion before local hybrid retrieval, which can "
        "improve recall for underspecified or vocabulary-sensitive code questions at the cost of extra latency."
    )
    st.write(
        "Evaluation metrics are measured against an independent 50-question golden dataset in "
        "`eval_testset/cobb_county_testset.csv`. The ground truths were generated with Claude 4.6 Sonnet, "
        "separate from the RAG agent's runtime LLM, to reduce self-evaluation bias and test retrieval quality "
        f"against dense, code-focused reference answers. LangSmith evaluator scoring currently uses `{evaluator_model}`."
    )
    st.write(
        "The four quality metrics use a five-point scale: `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`. "
        "This gives partial credit for mostly correct answers while reducing evaluator jitter. The judge is strict "
        "with technical details such as dimensions, dates, fire ratings, fees, code sections, and required procedures, "
        "but flexible with equivalent wording."
    )
    st.write(
        "Docling improves document parsing, while deterministic context expansion improves what the gate sees after retrieval. "
        "Retrieved chunks may be expanded to full page/range context for checklist and guide PDFs, or to neighboring chunks "
        "for long ordinance documents. The answer model still answers only when the expanded context explicitly supports the requested fact."
    )

    st.subheader("Guardrails")
    st.write(
        "The app is intentionally conservative: it keeps answers to two or three paragraphs, shows sources when available, "
        "and says it could not find a reliable answer when the evidence is not strong enough. Before generation, a JSON "
        "adequacy checker verifies that the supplied context contains the exact fact needed to answer. For numeric, code, "
        "inspection, permit, and procedural questions, the app requires the exact value or requirement to appear in context."
    )
    st.info(
        "Portfolio demonstration only. This is not legal, engineering, or permitting advice. "
        "Always verify requirements with Cobb County and the relevant authority having jurisdiction."
    )

    with st.expander("Example questions"):
        st.markdown(
            """
            - What permits are required for residential construction in Cobb County?
            - What are fire sprinkler requirements for commercial buildings?
            - When is a fire inspection required?
            """
        )


selected_page = st.radio(
    "Navigation",
    PAGE_OPTIONS,
    index=PAGE_OPTIONS.index(
        st.session_state.get("selected_page", "Ask")
        if st.session_state.get("selected_page", "Ask") in PAGE_OPTIONS
        else "Ask"
    ),
    horizontal=True,
    label_visibility="collapsed",
    key="selected_page",
    on_change=on_page_change,
)

if selected_page == "Ask":
    render_chat_tab()
elif selected_page == "Settings & Eval":
    render_settings_eval_tab()
else:
    render_about_tab()
