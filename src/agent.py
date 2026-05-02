"""Agentic RAG orchestration for concise, grounded answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from src.config import ORIGINAL_COLLECTION_NAME, get_chat_model
from src.retriever import RetrievedSource, has_sufficient_retrieval, search_documents
from src.tools import web_search


logger = logging.getLogger(__name__)

NO_ANSWER = "I could not find a reliable answer in the available documents or web sources."


@dataclass
class AgentResult:
    answer: str
    sources: list[str]
    used_local: bool
    used_web: bool
    route_reason: str = ""
    route_needs_web: bool = False
    web_search_attempted: bool = False
    web_search_error: str = ""
    web_query: str = ""


@dataclass
class QueryRoute:
    needs_local: bool
    needs_web: bool
    reason: str


class CobbCountyRAGAgent:
    """Retrieve first, search the web if needed, then synthesize with citations."""

    def __init__(self, collection_name: str = ORIGINAL_COLLECTION_NAME) -> None:
        self.collection_name = collection_name
        self.llm = get_chat_model(temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You primarily answer questions about Cobb County, Georgia building and fire codes. "
                    "Use only the supplied local document excerpts, web search results, and runtime context. "
                    "If the evidence is missing, conflicting, or not authoritative enough, answer exactly: "
                    f"{NO_ANSWER} "
                    "Keep the answer to 2-3 short paragraphs maximum. Include source names or URLs inline when available. "
                    "For simple current-date questions, use the runtime date. Do not provide legal, engineering, or permitting advice.",
                ),
                (
                    "human",
                    "Runtime context:\nCurrent date: {current_date}\n\nQuestion:\n{question}\n\nLocal document evidence:\n{local_context}\n\n"
                    "Web evidence:\n{web_context}\n\nGrounded answer:",
                ),
            ]
        )
        self.adequacy_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Decide whether the supplied evidence is enough to answer the user question. "
                    "Reply with only YES or NO. Reply NO if the evidence is off-topic, too vague, incomplete, "
                    "or does not establish the current effective requirements requested by the question.",
                ),
                ("human", "Question:\n{question}\n\nEvidence:\n{evidence}\n\nEnough evidence?"),
            ]
        )
        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a lightweight routing classifier for a Cobb County building and fire code RAG app. "
                    "Return only valid JSON with keys needs_local, needs_web, and reason. "
                    "needs_local should usually be true for Cobb County, Georgia building, fire, permit, zoning, inspection, code, fee, or ordinance questions. "
                    "needs_web should be true when the question asks for current, latest, recent, adopted, effective-date, dated, fee schedule, news, update, or web-published information. "
                    "needs_web should also be true for simple current-date questions. "
                    "Keep reason under 20 words.",
                ),
                ("human", "Question:\n{question}\n\nJSON route:"),
            ]
        )

    def answer(self, question: str, force_web: bool = False) -> AgentResult:
        route = self._route_query(question)
        docs, local_sources = search_documents(question, collection_name=self.collection_name)
        local_context = self._format_local_context(docs, local_sources)
        local_is_sufficient = has_sufficient_retrieval(local_sources)

        use_web = force_web or route.needs_web or self._should_use_web(question, local_context, local_sources)
        web_context = ""
        web_query = ""
        web_search_error = ""
        if use_web:
            logger.info("Using fallback web search. Router reason: %s", route.reason)
            web_query = self._web_query(question)
            try:
                web_context = web_search(web_query)
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)
                web_search_error = str(exc)
                web_context = ""

        has_web_results = self._has_web_results(web_context)

        if not local_is_sufficient and not has_web_results and not self._is_current_date_question(question):
            return AgentResult(
                answer=NO_ANSWER,
                sources=[],
                used_local=False,
                used_web=False,
                route_reason=route.reason,
                route_needs_web=route.needs_web,
                web_search_attempted=use_web,
                web_search_error=web_search_error,
                web_query=web_query,
            )

        chain = self.prompt | self.llm
        response = chain.invoke(
            {
                "question": question,
                "current_date": date.today().strftime("%A, %B %d, %Y"),
                "local_context": local_context if local_is_sufficient else "No sufficient local evidence available.",
                "web_context": web_context if has_web_results else "No web evidence used.",
            }
        )
        answer = getattr(response, "content", str(response)).strip()
        if not answer:
            answer = NO_ANSWER

        sources = self._source_labels(local_sources) if local_is_sufficient else []
        if has_web_results:
            sources.extend(self._web_source_labels(web_context))
        if self._is_current_date_question(question):
            sources.append("Runtime context: current system date")

        return AgentResult(
            answer=answer,
            sources=sources[:8],
            used_local=local_is_sufficient,
            used_web=has_web_results,
            route_reason=route.reason,
            route_needs_web=route.needs_web,
            web_search_attempted=use_web,
            web_search_error=web_search_error,
            web_query=web_query,
        )

    @staticmethod
    def _format_local_context(docs, sources: list[RetrievedSource]) -> str:
        blocks: list[str] = []
        for index, (doc, source) in enumerate(zip(docs, sources), start=1):
            page_text = f", page {source.page}" if source.page else ""
            blocks.append(
                f"[Local {index}] {source.source}{page_text} | relevance={source.score:.2f}\n"
                f"{doc.page_content[:1600]}"
            )
        return "\n\n".join(blocks)

    def _should_use_web(self, question: str, local_context: str, sources: list[RetrievedSource]) -> bool:
        if self._needs_current_web_verification(question):
            return True
        if not has_sufficient_retrieval(sources):
            return True
        try:
            response = (self.adequacy_prompt | self.llm).invoke(
                {"question": question, "evidence": local_context[:6000]}
            )
            decision = getattr(response, "content", str(response)).strip().upper()
            return not decision.startswith("YES")
        except Exception as exc:
            logger.warning("Retrieval adequacy check failed; using score threshold only: %s", exc)
            return False

    def _route_query(self, question: str) -> QueryRoute:
        fallback = QueryRoute(
            needs_local=True,
            needs_web=self._needs_current_web_verification(question) or self._is_current_date_question(question),
            reason="Keyword fallback route.",
        )
        try:
            response = (self.router_prompt | self.llm).invoke({"question": question})
            content = getattr(response, "content", str(response)).strip()
            data = self._parse_route_json(content)
            needs_local = bool(data.get("needs_local", fallback.needs_local))
            needs_web = bool(data.get("needs_web", fallback.needs_web)) or fallback.needs_web
            reason = str(data.get("reason") or fallback.reason).strip()
            return QueryRoute(needs_local=needs_local, needs_web=needs_web, reason=reason[:180])
        except Exception as exc:
            logger.warning("LLM router failed; using keyword fallback: %s", exc)
            return fallback

    @staticmethod
    def _parse_route_json(content: str) -> dict:
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            content = match.group(0)
        return json.loads(content)

    @staticmethod
    def _is_current_date_question(question: str) -> bool:
        normalized = question.lower()
        return "today" in normalized and "date" in normalized

    @staticmethod
    def _needs_current_web_verification(question: str) -> bool:
        normalized = question.lower()
        current_terms = (
            "currently",
            "current ",
            "latest",
            "newest",
            "recent",
            "as of",
            "today",
            "now",
            "in effect",
            "effective",
            "effective date",
            "became effective",
            "take effect",
            "took effect",
            "when did",
            "adopted",
            "fee schedule",
            "fees calculated",
            "permit fee",
        )
        code_terms = (
            "code",
            "codes",
            "building",
            "fire",
            "permit",
            "inspection",
            "sprinkler",
            "construction",
            "ordinance",
            "fee",
            "fees",
            "schedule",
        )
        has_explicit_date = bool(
            re.search(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}\b",
                normalized,
            )
            or re.search(r"\b20\d{2}\b", normalized)
        )
        return any(term in normalized for term in current_terms) and any(
            term in normalized for term in code_terms
        ) or has_explicit_date and any(term in normalized for term in code_terms)

    def _web_query(self, question: str) -> str:
        if self._is_current_date_question(question):
            return question
        if self._needs_current_web_verification(question):
            normalized = question.lower()
            if "fee" in normalized or "fees" in normalized or "schedule" in normalized:
                return f'"Cobb County" "building permit" "fee schedule" "July 1, 2024" site:cobbcounty.gov'
            return f'"Cobb County" "current" "construction codes" site:cobbcounty.gov OR site:dca.georgia.gov'
        return f"site:cobbcounty.gov Cobb County Georgia building fire code {question}"

    @staticmethod
    def _source_labels(sources: list[RetrievedSource]) -> list[str]:
        labels: list[str] = []
        for source in sources:
            page_text = f", page {source.page}" if source.page else ""
            labels.append(f"{source.source}{page_text} (score {source.score:.2f})")
        return labels

    @staticmethod
    def _web_source_labels(web_context: str) -> list[str]:
        labels: list[str] = []
        for line in web_context.splitlines():
            if line.startswith("http"):
                labels.append(line.strip())
        return labels

    @staticmethod
    def _has_web_results(web_context: str) -> bool:
        return bool(web_context.strip()) and "No web search results found" not in web_context
