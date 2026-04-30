"""LangChain tools for local retrieval and fallback web search."""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from src.retriever import search_documents


def format_local_documents(query: str) -> str:
    docs, sources = search_documents(query)
    if not docs:
        return "No local document results found. The vector index may need to be built."

    blocks: list[str] = []
    for index, (doc, source) in enumerate(zip(docs, sources), start=1):
        page_text = f", page {source.page}" if source.page else ""
        blocks.append(
            f"[Local {index}] {source.source}{page_text} | relevance={source.score:.2f}\n"
            f"{doc.page_content[:1200]}"
        )
    return "\n\n".join(blocks)


@tool
def retrieve_cobb_county_documents(query: str) -> str:
    """Search the local Cobb County building and fire code PDF vector database."""

    return format_local_documents(query)


def web_search(query: str) -> str:
    """Run a lightweight web search and return concise source-bearing results."""

    search = DuckDuckGoSearchResults(output_format="list", num_results=5)
    results = search.invoke(query)
    if not results:
        return "No web search results found."

    formatted: list[str] = []
    for index, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        formatted.append(f"[Web {index}] {title}\n{link}\n{snippet}")
    return "\n\n".join(formatted)


@tool
def search_web_for_cobb_codes(query: str) -> str:
    """Search the web for Cobb County, Georgia building and fire code information."""

    return web_search(f"Cobb County Georgia building fire code {query}")


TOOLS = [retrieve_cobb_county_documents, search_web_for_cobb_codes]
