"""Agentic RAG orchestration for concise, grounded answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from src.config import get_chat_model
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


class CobbCountyRAGAgent:
    """Retrieve first, search the web if needed, then synthesize with citations."""

    def __init__(self) -> None:
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

    def answer(self, question: str) -> AgentResult:
        docs, local_sources = search_documents(question)
        local_context = self._format_local_context(docs, local_sources)
        local_is_sufficient = has_sufficient_retrieval(local_sources)

        use_web = self._should_use_web(question, local_context, local_sources)
        web_context = ""
        if use_web:
            logger.info("Local retrieval was weak; using fallback web search.")
            try:
                web_context = web_search(self._web_query(question))
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)
                web_context = ""

        if not local_is_sufficient and not web_context and not self._is_current_date_question(question):
            return AgentResult(answer=NO_ANSWER, sources=[], used_local=False, used_web=False)

        chain = self.prompt | self.llm
        response = chain.invoke(
            {
                "question": question,
                "current_date": date.today().strftime("%A, %B %d, %Y"),
                "local_context": local_context if local_is_sufficient else "No sufficient local evidence available.",
                "web_context": web_context or "No web evidence used.",
            }
        )
        answer = getattr(response, "content", str(response)).strip()
        if not answer:
            answer = NO_ANSWER

        sources = self._source_labels(local_sources) if local_is_sufficient else []
        if web_context:
            sources.extend(self._web_source_labels(web_context))
        if self._is_current_date_question(question):
            sources.append("Runtime context: current system date")

        return AgentResult(
            answer=answer,
            sources=sources[:8],
            used_local=local_is_sufficient,
            used_web=bool(web_context),
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
            "effective date",
            "take effect",
            "took effect",
            "when did",
            "adopted",
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
        )
        return any(term in normalized for term in current_terms) and any(
            term in normalized for term in code_terms
        )

    def _web_query(self, question: str) -> str:
        if self._is_current_date_question(question):
            return question
        return f"Cobb County Georgia building fire code {question}"

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
