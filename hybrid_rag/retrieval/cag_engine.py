"""
retrieval/cag_engine.py — Cache-Augmented Generation engine.

preload(): fetches all chunks for a doc and stores the combined text in the cache.
answer():  uses preloaded context to answer queries via Groq without re-fetching.
"""
from __future__ import annotations

import time
from typing import List, Optional

import structlog

from hybrid_rag.config import settings
from hybrid_rag.storage.cache_manager import cache_manager

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

_ANSWER_PROMPT = """\
You are an expert research assistant.
Use the context provided below to write a thorough, insightful answer.
You are allowed to synthesize, infer, and reason across the context — draw connections,
identify relationships, and form summaries even when they are expressed implicitly.
Do NOT refuse to answer or say information is missing if the context contains relevant facts.
Cite the source document and page number inline where possible.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


_SYSTEM_PROMPT = (
    "You are an expert research assistant. Synthesize clear, complete answers by "
    "reasoning across all provided context — including implicit relationships and "
    "inferences. Never refuse to answer if the context contains relevant facts."
)


def _llm_call(prompt: str) -> str:
    client = _groq_client()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"CAG LLM call failed: {last_exc}")


def _truncate_to_token_limit(text: str, limit: int) -> str:
    """Rough truncation by word count (~1.3 words/token for English)."""
    words = text.split()
    max_words = int(limit * 0.75)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n[...context truncated...]"


class CAGEngine:
    async def preload(self, doc_id: str) -> None:
        """
        Fetch all chunks for this doc from Qdrant, concatenate, and store in cache.
        This simulates KV-cache preloading: the context is always ready.
        """
        from hybrid_rag.storage.qdrant_client import qdrant_client

        try:
            payloads = await qdrant_client.get_all_chunks_for_doc(doc_id)
            if not payloads:
                logger.warning("cag_preload_no_chunks", doc_id=doc_id)
                return

            # Sort by page + position for coherent ordering
            payloads.sort(key=lambda p: (p.get("page") or 0, p.get("position") or 0))
            full_text = "\n\n".join(p.get("text", "") for p in payloads if p.get("text"))
            cache_manager.set_doc_context(doc_id, full_text)
            logger.info("cag_preloaded", doc_id=doc_id, chunks=len(payloads))
        except Exception as exc:
            logger.error("cag_preload_failed", doc_id=doc_id, error=str(exc))
            raise

    async def answer(self, query: str, doc_ids: Optional[List[str]] = None) -> str:
        """
        Answer a query using preloaded doc contexts.

        1. Check exact match cache.
        2. Build context from preloaded doc texts.
        3. Truncate to CAG_CONTEXT_LIMIT tokens.
        4. Call Groq.
        5. Cache and return.
        """
        # Step 1: exact cache hit
        cached = await cache_manager.get(query)
        if cached:
            logger.info("cag_exact_cache_hit")
            return cached

        # Step 2: build context
        context_parts: List[str] = []
        effective_doc_ids = doc_ids or []

        if effective_doc_ids:
            for doc_id in effective_doc_ids:
                ctx = cache_manager.get_doc_context(doc_id)
                if ctx:
                    context_parts.append(f"[Document: {doc_id}]\n{ctx}")
                    cache_manager.increment_access(doc_id)

        if not context_parts:
            # Fallback: check all preloaded contexts
            for key in cache_manager._l1.keys():
                if key.startswith("ctx:"):
                    ctx = cache_manager._l1.get(key)
                    if ctx:
                        context_parts.append(ctx)

        if not context_parts:
            return "No preloaded document context available. Please ingest documents first."

        combined_context = "\n\n---\n\n".join(context_parts)

        # Step 3: truncate
        combined_context = _truncate_to_token_limit(
            combined_context, settings.cag_context_limit
        )

        # Step 4: call LLM
        prompt = _ANSWER_PROMPT.format(context=combined_context, question=query)
        answer_text = _llm_call(prompt)

        # Step 5: cache
        primary_doc = effective_doc_ids[0] if effective_doc_ids else None
        await cache_manager.set(query, answer_text, doc_id=primary_doc)

        logger.info("cag_answered", query_len=len(query))
        return answer_text

    async def invalidate(self, doc_id: str) -> None:
        await cache_manager.invalidate_doc(doc_id)


# Singleton
cag_engine = CAGEngine()
