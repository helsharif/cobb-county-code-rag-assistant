"""Streamlit frontend for the Cobb County Agentic RAG app."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import CobbCountyRAGAgent, NO_ANSWER  # noqa: E402
from src.retriever import vectorstore_exists  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


st.set_page_config(page_title="Cobb County Codes RAG", page_icon="CC", layout="centered")
st.title("Cobb County Building & Fire Codes RAG")
st.caption("Grounded answers from local PDFs first, with web fallback when retrieval is weak.")


def get_agent() -> CobbCountyRAGAgent:
    return CobbCountyRAGAgent()


def render_chat_tab() -> None:
    if not vectorstore_exists():
        st.warning(
            "No local vector index found. Build it first with "
            "`python -m src.ingestion --rebuild`, then restart the app."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    question = st.chat_input("Ask about Cobb County building or fire code requirements")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Retrieving local code documents..."):
                result = get_agent().answer(question)
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


def render_about_tab() -> None:
    st.subheader("What This App Does")
    st.write(
        "This app answers questions about Cobb County, Georgia building and fire code materials. "
        "It searches local PDFs first, checks whether those results look strong enough, and only uses web search "
        "when the local evidence is weak or incomplete."
    )

    st.image(
        str(ROOT_DIR / "assets" / "Rag Flow Chart.png"),
        caption="Agentic RAG architecture: local document retrieval first, web fallback when evidence is weak or current-code verification is needed.",
        width="stretch",
    )

    st.subheader("Under the Hood")
    ingest_col, query_col = st.columns(2)
    with ingest_col:
        st.markdown("**Index build**")
        st.write(
            "PDFs under `data/` are loaded page by page, split into overlapping chunks, embedded, "
            "and stored in `vectorstore/` with source and page metadata."
        )
        st.graphviz_chart(
            """
            digraph {
                graph [rankdir=TB, bgcolor="transparent", pad="0.2"];
                node [shape=box, style="rounded,filled", color="#94a3b8", fillcolor="#f8fafc", fontname="Arial", fontsize=10];
                edge [color="#64748b", arrowsize=0.65];
                pdf [label="Local PDFs"];
                load [label="Load pages"];
                split [label="Chunk text\\nwith overlap"];
                embed [label="Create embeddings"];
                store [label="Persist in Chroma"];
                pdf -> load -> split -> embed -> store;
            }
            """,
            width="stretch",
        )
    with query_col:
        st.markdown("**Question answering**")
        st.write(
            "Each question is embedded and matched against Chroma. The agent evaluates the retrieved chunks, "
            "uses web search only when needed, then writes a short grounded response."
        )
        st.graphviz_chart(
            """
            digraph {
                graph [rankdir=TB, bgcolor="transparent", pad="0.2"];
                node [shape=box, style="rounded,filled", color="#94a3b8", fillcolor="#f8fafc", fontname="Arial", fontsize=10];
                edge [color="#64748b", arrowsize=0.65];
                q [label="Question"];
                retrieve [label="Retrieve local chunks"];
                judge [label="Judge evidence"];
                cite [label="Answer with citations"];
                fallback [label="Search web if needed"];
                q -> retrieve -> judge -> cite;
                judge -> fallback -> cite;
            }
            """,
            width="stretch",
        )

    st.subheader("Why This Is Agentic RAG")
    col1, col2, col3 = st.columns(3)
    col1.metric("First move", "Local retrieval")
    col2.metric("Fallback", "Web search")
    col3.metric("Response", "Grounded + concise")

    st.table(
        [
            {"Layer": "Frontend", "What it does": "Provides a simple chat UI and displays sources.", "Tech": "Streamlit"},
            {"Layer": "Retriever", "What it does": "Finds relevant PDF chunks from the local index.", "Tech": "Chroma"},
            {"Layer": "Agent logic", "What it does": "Decides whether local evidence is enough or web search is needed.", "Tech": "LangChain"},
            {"Layer": "Generation", "What it does": "Synthesizes a short answer from retrieved evidence only.", "Tech": "OpenAI or Gemini"},
            {"Layer": "Deployment", "What it does": "Runs locally, in Docker, or on Streamlit Community Cloud.", "Tech": "Docker + Streamlit"},
        ]
    )

    st.subheader("Guardrails")
    st.write(
        "The app is intentionally conservative: it keeps answers to two or three paragraphs, shows sources when available, "
        "and says it could not find a reliable answer when the evidence is not strong enough."
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


chat_tab, about_tab = st.tabs(["Ask", "About the App"])

with chat_tab:
    render_chat_tab()

with about_tab:
    render_about_tab()
