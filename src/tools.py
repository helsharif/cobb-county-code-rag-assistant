"""LangChain tools for local retrieval and fallback web search."""

from __future__ import annotations

from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from src.config import get_settings
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
    """Run SerpAPI Google Search and return official-domain source-bearing results."""

    settings = get_settings()
    if not settings.serpapi_api_key:
        return "No web search results found. SERPAPI_API_KEY is not configured."
    allowed_domains = _allowed_domains(settings.allowed_web_domains)
    scoped_query = _domain_scoped_query(query, allowed_domains)

    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": scoped_query,
            "api_key": settings.serpapi_api_key,
            "num": 5,
            "safe": "active",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("organic_results", [])
    if not results:
        return "No web search results found."

    formatted: list[str] = []
    for item in results:
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        if not _is_allowed_url(link, allowed_domains):
            continue
        snippet = item.get("snippet", "")
        index = len(formatted) + 1
        formatted.append(f"[Web {index}] {title}\n{link}\n{snippet}")
    if not formatted:
        return "No web search results found from approved official domains."
    return "\n\n".join(formatted)


@tool
def search_web_for_cobb_codes(query: str) -> str:
    """Search the web for Cobb County, Georgia building and fire code information."""

    return web_search(f"Cobb County Georgia building fire code {query}")


TOOLS = [retrieve_cobb_county_documents, search_web_for_cobb_codes]


def _allowed_domains(raw_domains: str) -> list[str]:
    domains = [domain.strip().lower() for domain in raw_domains.split(",") if domain.strip()]
    return domains or ["cobbcounty.gov", "dca.georgia.gov"]


def _domain_scoped_query(query: str, allowed_domains: list[str]) -> str:
    if "site:" in query.lower():
        return query
    site_clause = " OR ".join(f"site:{domain}" for domain in allowed_domains)
    return f"({site_clause}) {query}"


def _is_allowed_url(url: str, allowed_domains: list[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)
