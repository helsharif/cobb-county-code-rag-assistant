"""Agentic RAG orchestration for concise, grounded answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from src.config import ORIGINAL_COLLECTION_NAME, get_chat_model
from src.context_expansion import expand_retrieved_docs
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
    contexts: list[str] | None = None
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

    def __init__(self, collection_name: str = ORIGINAL_COLLECTION_NAME, llm=None) -> None:
        self.collection_name = collection_name
        self.llm = llm or get_chat_model(temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer questions about Cobb County, Georgia building and fire code documents.\n\n"
                    "Use ONLY the supplied context in this request:\n"
                    "- local document excerpts\n"
                    "- web results\n"
                    "- runtime context\n\n"
                    "Do NOT use:\n"
                    "- memory\n"
                    "- prior conversation turns\n"
                    "- general code knowledge\n"
                    "- outside assumptions\n"
                    "- likely values\n"
                    "- common construction practice\n"
                    "- NFPA/IBC knowledge unless explicitly quoted in the supplied context\n"
                    "- facts from documents that are not included in the supplied context\n\n"
                    "CRITICAL RULE:\n"
                    "If a fact is not visible in the supplied context for this request, it does not exist for purposes "
                    "of your answer.\n\n"
                    "Before answering, silently perform this evidence check:\n"
                    "1. Identify the exact fact the user is asking for.\n"
                    "2. Search the supplied context for an exact supporting phrase.\n"
                    f"3. If the exact supporting phrase is absent, answer exactly:\n\"{NO_ANSWER}\"\n\n"
                    "Faithfulness rules:\n"
                    "1. Every factual claim must be directly supported by the supplied context.\n"
                    "2. Do not state thresholds, dates, dimensions, distances, heights, fire ratings, fees, code sections, "
                    "exceptions, inspection requirements, permit requirements, procedural steps, responsible offices, "
                    "deadlines, forms, penalties, or approval conditions unless they appear explicitly in the supplied context.\n"
                    "3. For numeric, dimensional, code, inspection, permit, or procedural questions:\n"
                    "   - The exact value or requirement must appear in the supplied context.\n"
                    "   - The answer must quote or closely paraphrase the supporting phrase.\n"
                    "   - Cite the source name, local source ID, or URL.\n"
                    f"   - If the exact value is not present, answer exactly:\n     \"{NO_ANSWER}\"\n"
                    "4. Do not infer from related topics, similar codes, nearby sections, source titles, document names, "
                    "or common practice.\n"
                    "5. Do not combine facts from separate excerpts unless the relationship is explicitly stated in the context.\n"
                    "6. Do not guess section numbers. Cite section numbers only when clearly shown in the context.\n\n"
                    "Special rule for missing evidence:\n"
                    "If the user asks for a numeric or procedural requirement and the supplied context does not contain "
                    f"the exact requested value or procedure, do not explain, qualify, or speculate. Output only:\n\"{NO_ANSWER}\"\n\n"
                    "Partial-answer rule:\n"
                    "If the context directly supports part of the answer but not the requested detail, state only the "
                    "supported part and say:\n"
                    "\"The retrieved context does not state [missing detail].\"\n\n"
                    "Do not say:\n"
                    "- \"no other details were found\"\n"
                    "- \"no additional requirements exist\"\n"
                    "- \"the minimum is...\"\n"
                    "unless the supplied context explicitly supports that statement.\n\n"
                    "Answer relevancy:\n"
                    "- Answer only the user's specific question.\n"
                    "- Avoid generic background, meta-commentary, and unnecessary caveats.\n"
                    "- Do not provide legal, engineering, design, or permitting advice.\n"
                    "- For simple current-date questions, use the runtime date.\n\n"
                    "Output format:\n"
                    "1. Keep the answer to 2-3 short paragraphs maximum.\n"
                    "2. Include source names, local source IDs, or URLs inline when available.\n"
                    "3. When giving a technical requirement, include a short supporting phrase from the context.\n"
                    "4. If refusing due to insufficient evidence, output only the exact abstention sentence and nothing else.",
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
                    "You are an evidence sufficiency checker for a Cobb County building and fire code RAG system.\n\n"
                    "Use ONLY the supplied evidence. Do not use memory, prior conversation, outside knowledge, "
                    "common code knowledge, NFPA/IBC assumptions, or likely values.\n\n"
                    "Your job is NOT to answer the question. Your job is only to decide whether the supplied evidence "
                    "contains the exact fact needed to answer.\n\n"
                    "For questions asking for numbers, dimensions, distances, heights, fire ratings, dates, fees, "
                    "code sections, inspection requirements, permit requirements, or procedures:\n"
                    "- Set answerable=true only if the exact value or exact requirement appears in the supplied evidence.\n"
                    "- Set answerable=false if the evidence is merely related, vague, incomplete, off-topic, or missing the exact requested value.\n"
                    "- Set answerable=false if the answer would require inference from related topics, similar codes, source titles, "
                    "document names, or common practice.\n\n"
                    "If the evidence does not contain a direct quote that answers the question, set answerable=false.\n\n"
                    "Return only valid JSON:\n"
                    "{{\n"
                    '  "answerable": true/false,\n'
                    '  "required_fact": "the exact fact needed to answer the question",\n'
                    '  "supporting_quote": "exact quote from the evidence that answers the question, or empty string",\n'
                    '  "source_id": "source name/local ID/URL if shown, or empty string"\n'
                    "}}",
                ),
                (
                    "human",
                    "Question:\n{question}\n\nEvidence:\n{evidence}\n\nIs the exact answer present in the evidence?",
                ),
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
        original_docs = docs
        original_sources = local_sources
        docs, local_sources = expand_retrieved_docs(
            docs,
            local_sources,
            collection_name=self.collection_name,
        )
        local_context = self._format_local_context(docs, local_sources)
        local_is_sufficient = has_sufficient_retrieval(local_sources)
        logger.info(
            "Retrieved %s chunks and expanded to %s context blocks for collection %s.",
            len(original_docs),
            len(docs),
            self.collection_name,
        )
        logger.debug("Original retrieved chunks: %s", self._debug_sources(original_sources))
        logger.debug("Expanded retrieved chunks: %s", self._debug_sources(local_sources))

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
                contexts=[doc.page_content for doc in docs],
                route_reason=route.reason,
                route_needs_web=route.needs_web,
                web_search_attempted=use_web,
                web_search_error=web_search_error,
                web_query=web_query,
            )

        final_evidence = "\n\n".join(
            evidence
            for evidence in (
                local_context if local_is_sufficient else "",
                web_context if has_web_results else "",
            )
            if evidence.strip()
        )
        if not self._is_current_date_question(question) and not self._evidence_is_answerable(
            question,
            final_evidence,
        ):
            return AgentResult(
                answer=NO_ANSWER,
                sources=[],
                used_local=False,
                used_web=False,
                contexts=[doc.page_content for doc in docs] if local_is_sufficient else [],
                route_reason=route.reason,
                route_needs_web=route.needs_web,
                web_search_attempted=use_web,
                web_search_error=web_search_error,
                web_query=web_query,
            )

        logger.debug("Adequacy gate input context: %s", final_evidence)
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
        logger.debug("Final answer: %s", answer)

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
            contexts=[doc.page_content for doc in docs] if local_is_sufficient else [],
            route_reason=route.reason,
            route_needs_web=route.needs_web,
            web_search_attempted=use_web,
            web_search_error=web_search_error,
            web_query=web_query,
        )

    @staticmethod
    def _format_local_context(docs, sources: list[RetrievedSource]) -> str:
        blocks: list[str] = []
        remaining_chars = 18000
        for index, (doc, source) in enumerate(zip(docs, sources), start=1):
            if remaining_chars <= 0:
                break
            page_text = f", page {source.page}" if source.page else ""
            metadata = doc.metadata or {}
            expansion_text = f" | expansion={metadata.get('expansion_type')}" if metadata.get("expansion_type") else ""
            section_text = f" | section={metadata.get('section')}" if metadata.get("section") else ""
            content = doc.page_content[:remaining_chars]
            blocks.append(
                f"[Local {index}] {source.source}{page_text} | relevance={source.score:.2f}{expansion_text}{section_text}\n"
                f"{content}"
            )
            remaining_chars -= len(content)
        return "\n\n".join(blocks)

    def _should_use_web(self, question: str, local_context: str, sources: list[RetrievedSource]) -> bool:
        if self._needs_current_web_verification(question):
            return True
        if not has_sufficient_retrieval(sources):
            return True
        return not self._evidence_is_answerable(question, local_context)

    def _evidence_is_answerable(self, question: str, evidence: str) -> bool:
        if not evidence.strip():
            return False
        from src.config import get_settings

        settings = get_settings()
        try:
            response = (self.adequacy_prompt | self.llm).invoke(
                {"question": question, "evidence": evidence[: settings.context_max_chars]}
            )
            content = getattr(response, "content", str(response)).strip()
            data = self._parse_route_json(content)
            answerable = bool(data.get("answerable", False))
            supporting_quote = str(data.get("supporting_quote") or "").strip()
            logger.info(
                "Strict adequacy decision answerable=%s required_fact=%s source_id=%s.",
                answerable,
                data.get("required_fact", ""),
                data.get("source_id", ""),
            )
            if answerable and not supporting_quote:
                logger.info("Strict adequacy gate rejected answerable=true without a supporting quote.")
                return False
            return answerable
        except Exception as exc:
            logger.warning("Strict evidence adequacy check failed; rejecting evidence: %s", exc)
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

    @staticmethod
    def _debug_sources(sources: list[RetrievedSource]) -> list[str]:
        return [
            f"{source.source} page={source.page} score={source.score:.2f} snippet={source.snippet[:120]}"
            for source in sources
        ]
