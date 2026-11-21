"""RAG pipeline: adaptive retrieval -> Groq generation -> verified citations."""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from backend.app.core import prompts
from backend.app.schemas.chat import ChatMessage, ChatResponse, Source
from backend.app.services.retrieval_service import RetrievalService, merge_hits
from backend.app.services.vector_store_service import SearchHit

logger = logging.getLogger(__name__)

_HISTORY_MESSAGE_CHARS = 800
_MIN_QUOTE_CHARS = 15
# Only short questions ("Does it have to be unused?") are treated as possible follow-ups whose
# search is widened with the previous question. Longer questions are complete on their own.
FOLLOW_UP_MAX_WORDS = 6


# --------------------------------------------------------------------------- errors
class LLMError(Exception):
    pass
class LLMConfigError(LLMError):
    pass
class LLMAuthError(LLMError):
    pass
class LLMRateLimitError(LLMError):
    pass
class LLMUnavailableError(LLMError):
    pass
class LLMResponseError(LLMError):
    pass

class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

class GroqClient:
    """Groq Chat Completions client using its OpenAI-compatible HTTP API."""
    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.groq.com/openai/v1",
                 timeout_seconds: int = 90) -> None:
        if not api_key:
            raise LLMConfigError("GROQ_API_KEY is not set. Add it to your .env file.")
        self._api_key, self._model = api_key, model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout_seconds

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {"model": self._model, "temperature": 0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}]}
        try:
            response = httpx.post(self._url,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload, timeout=self._timeout)
            if response.status_code == 429:
                raise LLMRateLimitError("Groq rate limit or quota reached. Wait and try again.")
            if response.status_code in (401, 403):
                raise LLMAuthError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.")
            if response.status_code >= 500:
                raise LLMUnavailableError("Groq is temporarily unavailable. Please try again shortly.")
            if response.status_code >= 400:
                raise LLMError(f"Groq request failed ({response.status_code}): {response.text[:500]}")
            content = response.json()["choices"][0]["message"]["content"]
            if not content:
                raise LLMResponseError("Groq returned an empty response. Please retry.")
            return content
        except LLMError:
            raise
        except (httpx.HTTPError, TimeoutError, ConnectionError) as exc:
            logger.warning("Network error calling Groq: %s", exc)
            raise LLMUnavailableError("Could not reach Groq. Check your internet connection and try again.") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.exception("Unexpected Groq response format")
            raise LLMResponseError("Groq returned an unexpected response format. Please retry.") from exc


# --------------------------------------------------------------------------- model output
class _Citation(BaseModel):
    source: str
    quote: str = ""


class _ModelAnswer(BaseModel):
    found: bool
    answer: str = ""
    citations: list[_Citation] = Field(default_factory=list)


def parse_model_output(raw: str) -> _ModelAnswer:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return _ModelAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Unusable model output: %r", raw[:300])
        raise LLMResponseError("Groq returned an answer in an unexpected format. Please retry.") from exc


# --------------------------------------------------------------------------- prompt building
def _sanitize(text: str) -> str:
    """Stop document text from closing our delimiter tags."""
    return re.sub(r"</?\s*(source|document_context|conversation_history)[^>]*>", "[tag removed]", text, flags=re.I)


def select_relevant(hits: list[SearchHit], max_distance: float, limit: int | None = None) -> list[SearchHit]:
    """Evidence check: keep only chunks that are close enough to the question.

    This is a heuristic on embedding distance. It cannot prove a chunk answers the question.
    """
    selected = [h for h in sorted(hits, key=lambda h: h.distance) if h.distance <= max_distance]
    return selected[:limit] if limit is not None else selected


def log_candidates(question: str, hits: list[SearchHit], max_distance: float) -> None:
    """Log every retrieved candidate (rank, distance, file, page, chunk, excerpt) for tuning."""
    logger.info("Retrieval for %r: %d candidates (limit %.3f)", question[:80], len(hits), max_distance)
    for rank, hit in enumerate(sorted(hits, key=lambda h: h.distance), start=1):
        excerpt = re.sub(r"\s+", " ", hit.text)[:100]
        verdict = "KEEP" if hit.distance <= max_distance else "drop"
        logger.info(
            "  #%d dist=%.4f %s | %s p.%d chunk=%d | %s",
            rank, hit.distance, verdict, hit.filename, hit.page, hit.chunk_index, excerpt,
        )


def build_context(hits: list[SearchHit]) -> tuple[str, dict[str, SearchHit]]:
    blocks: list[str] = []
    by_id: dict[str, SearchHit] = {}
    for number, hit in enumerate(hits, start=1):
        source_id = f"S{number}"
        by_id[source_id] = hit
        filename = hit.filename.replace('"', "'")
        blocks.append(f'<source id="{source_id}" doc_id="{hit.doc_id}" chunk_id="{hit.chunk_id}" file="{filename}" page="{hit.page}" section="{_sanitize(hit.section)}">\n{_sanitize(hit.text)}\n</source>')
    return "\n\n".join(blocks), by_id
def build_history_block(history: list[ChatMessage], max_messages: int) -> str:
    recent = history[-max_messages:] if max_messages > 0 else []
    if not recent:
        return ""
    lines = []
    for message in recent:
        speaker = "User" if message.role == "user" else "Assistant"
        lines.append(f"{speaker}: {_sanitize(message.content[:_HISTORY_MESSAGE_CHARS])}")
    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>\n\n"


# --------------------------------------------------------------------------- excerpts
def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def verbatim_quote(quote: str, chunk_text: str) -> str | None:
    """Return the quote as it is written in the chunk, or None if it is not really there."""
    words = quote.split()
    if len(quote.strip()) < _MIN_QUOTE_CHARS or not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, chunk_text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", match.group(0)) if match else None


def best_excerpt(chunk_text: str, reference: str, max_chars: int) -> str:
    """Pick the sentences of the chunk that share the most words with `reference`."""
    flat = re.sub(r"\s+", " ", chunk_text).strip()
    if len(flat) <= max_chars:
        return flat
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    terms = {w for w in re.findall(r"[a-z0-9]+", reference.lower()) if len(w) > 3}
    scores = [len(terms & set(re.findall(r"[a-z0-9]+", s.lower()))) for s in sentences]
    start = max(range(len(sentences)), key=lambda i: scores[i])
    picked: list[str] = []
    for sentence in sentences[start:]:
        if picked and len(" ".join(picked + [sentence])) > max_chars:
            break
        picked.append(sentence)
    return _truncate(" ".join(picked), max_chars)


# --------------------------------------------------------------------------- service
class RagService:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient | None,
        *,
        max_distance: float,
        top_k: int,
        max_history_messages: int,
        excerpt_chars: int,
        max_context_chunks: int = 20,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._max_distance = max_distance
        self._top_k = top_k
        self._max_history = max_history_messages
        self._excerpt_chars = excerpt_chars
        self._max_context_chunks = max_context_chunks

    def _retrieve(self, question: str, history: list[ChatMessage]) -> list[SearchHit]:
        """Search with the question; for short (follow-up style) questions also with the previous one."""
        candidate_count = max(self._max_context_chunks * 3, self._top_k * 4)
        result_lists = [self._retrieval.retrieve(question, top_k=candidate_count)]
        previous = [m.content for m in history if m.role == "user"]
        if previous and len(question.split()) <= FOLLOW_UP_MAX_WORDS:
            combined = f"{previous[-1][:_HISTORY_MESSAGE_CHARS]} {question}"
            result_lists.append(self._retrieval.retrieve(combined, top_k=candidate_count))
        return merge_hits(*result_lists)

    def answer(self, question: str, history: list[ChatMessage] | None = None) -> ChatResponse:
        history = history or []
        question = question.strip()

        if not self._retrieval.has_documents():
            return ChatResponse(answer=prompts.NO_DOCUMENTS_MESSAGE, found=False, sources=[])

        hits = self._retrieve(question, history)
        log_candidates(question, hits, self._max_distance)
        relevant = select_relevant(hits, self._max_distance, self._max_context_chunks)
        if not relevant:
            best = min((h.distance for h in hits), default=None)
            logger.info("Evidence too weak (best distance=%s, threshold=%s)", best, self._max_distance)
            return ChatResponse(
                answer=prompts.NOT_FOUND_MESSAGE,
                found=False,
                sources=[],
                closest_distance=round(best, 4) if best is not None else None,
                distance_limit=self._max_distance,
            )

        if self._llm is None:
            raise LLMConfigError("GROQ_API_KEY is not set. Add it to your .env file and restart the backend.")

        context, by_id = build_context(relevant)
        user_prompt = prompts.USER_PROMPT_TEMPLATE.format(
            history_block=build_history_block(history, self._max_history),
            context=context,
            question=_sanitize(question),
        )
        parsed = parse_model_output(self._llm.generate(prompts.SYSTEM_PROMPT, user_prompt))

        if not parsed.found:
            return ChatResponse(answer=parsed.answer.strip() or prompts.NOT_FOUND_MESSAGE, found=False, sources=[])

        sources = self._verify_citations(parsed, by_id, question)
        if not sources:
            logger.warning("Model answered without any valid citation; refusing the answer")
            return ChatResponse(answer=prompts.UNVERIFIED_ANSWER_MESSAGE, found=False, sources=[])
        return ChatResponse(answer=parsed.answer.strip(), found=True, sources=sources)

    def _verify_citations(self, parsed: _ModelAnswer, by_id: dict[str, SearchHit], question: str) -> list[Source]:
        """Build sources only from real retrieved chunks; ignore ids the model made up."""
        sources: list[Source] = []
        seen: set[str] = set()
        for citation in parsed.citations:
            hit = by_id.get(citation.source.strip().upper())
            if hit is None or hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            quote = verbatim_quote(citation.quote, hit.text)
            if quote:
                excerpt = _truncate(quote, self._excerpt_chars)
            else:
                excerpt = best_excerpt(hit.text, f"{question} {parsed.answer}", self._excerpt_chars)
            sources.append(Source(filename=hit.filename, doc_id=hit.doc_id, chunk_id=hit.chunk_id, page=hit.page, section=hit.section, excerpt=excerpt, distance=round(hit.distance, 4)))
        return sources
