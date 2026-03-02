"""
retrieval/reranker.py — Cross-encoder reranking for retrieved chunks.

Replaces the rough RRF score ordering with a proper relevance score from a
small cross-encoder model (ms-marco-MiniLM-L-6-v2, ~85 MB).

  Input : query + List[RetrievalResult] (up to retrieval_top_k = 10)
  Output: top rerank_top_k results sorted by cross-encoder score

Why:
  - Sends fewer tokens to the LLM → faster Groq response
  - Higher precision context → better answers
  - Cross-encoder sees (query, chunk) together → much better than embedding cosine sim alone
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

import structlog

from hybrid_rag.config import settings
from hybrid_rag.storage.schema import RetrievalResult

logger = structlog.get_logger(__name__)

_cross_encoder: Optional[object] = None  # lazy-loaded


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("cross_encoder_loaded")
        except Exception as exc:
            logger.warning("cross_encoder_unavailable", error=str(exc))
            _cross_encoder = False  # sentinel: don't retry
    return _cross_encoder if _cross_encoder is not False else None


def rerank_sync(
    query: str,
    results: List[RetrievalResult],
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """
    Synchronous rerank. Falls back to score-sorted top_k if cross-encoder is unavailable.
    """
    effective_top_k = top_k if top_k is not None else settings.rerank_top_k
    if not results:
        return results

    model = _get_cross_encoder()
    if model is None:
        # Fallback: just return top_k by existing score
        logger.debug("rerank_fallback_score_sort")
        return sorted(results, key=lambda r: r.score, reverse=True)[:effective_top_k]

    try:
        pairs = [(query, r.text[:512]) for r in results]  # cap chunk len for speed
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: float(x[0]), reverse=True)
        top = [r for _, r in ranked[:effective_top_k]]
        logger.debug("rerank_done", input=len(results), output=len(top))
        return top
    except Exception as exc:
        logger.warning("rerank_failed_fallback", error=str(exc))
        return results[:effective_top_k]


async def rerank(
    query: str,
    results: List[RetrievalResult],
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """Async wrapper — runs cross-encoder in thread pool so event loop isn't blocked."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank_sync, query, results, top_k)
