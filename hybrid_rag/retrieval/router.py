"""
retrieval/router.py — Smart Query Router.

Decides which retrieval path to use:
  CAG        — cache-first fast path
  KAG        — multi-step reasoning pipeline
  KAG_SIMPLE — single-step hybrid retrieval (default)
"""
from __future__ import annotations

from typing import Dict, List

import structlog

logger = structlog.get_logger(__name__)

_COMPLEX_WORDS = [
    "if", "when", "before", "after", "since", "because",
    "compare", "difference", "how many", "calculate", "between",
    "why", "explain", "relationship", "impact", "effect", "cause",
]

_GLOBAL_WORDS = [
    "overview", "summarize", "summarise", "summary", "overall",
    "main theme", "main themes", "main topic", "main topics",
    "key theme", "key topics", "key ideas", "key concepts",
    "across", "throughout", "entire document", "whole document",
    "what is this about", "what are the key", "what are the main",
    "high-level", "broad", "generally", "in general",
    "landscape", "big picture", "at a high level",
]


async def route(query: str, session_ctx: Dict | None = None) -> str:
    """
    Determine the retrieval strategy for the given query.

    Returns one of: "CAG" | "KAG" | "KAG_SIMPLE"
    """
    from hybrid_rag.storage.cache_manager import cache_manager

    ctx = session_ctx or {}
    q_lower = query.lower()   # defined once, used by both GLOBAL and KAG checks

    # ── CAG signals ───────────────────────────────────────────────────────────

    # Route to CAG only when EVERY selected doc is preloaded in the context
    # store. If even one doc is missing, CAG would silently answer based only
    # on the preloaded subset — instead fall through to KAG/GLOBAL so all
    # docs are searched properly.
    cached_docs: List[str] = ctx.get("cached_docs", [])
    if cached_docs:
        preloaded = set(cache_manager._doc_contexts.keys())
        all_covered = all(doc in preloaded for doc in cached_docs)
        if all_covered:
            logger.info("router_decision", decision="CAG", reason="preloaded_doc_context",
                        preloaded_docs=list(preloaded & set(cached_docs)))
            return "CAG"
        elif preloaded & set(cached_docs):
            # Some docs are preloaded but not all — warn and fall through
            logger.info("router_decision_partial_cache",
                        preloaded=list(preloaded & set(cached_docs)),
                        missing=list(set(cached_docs) - preloaded),
                        note="falling_through_to_retrieval")

    if await cache_manager.has_exact(query):
        logger.info("router_decision", query_len=len(query), decision="CAG", reason="exact_cache_hit")
        return "CAG"

    # Very short query — likely a simple factual lookup
    words = query.split()
    if len(words) < 8 and "?" not in query:
        logger.info("router_decision", decision="CAG", reason="short_factual")
        return "CAG"

    # ── GLOBAL signals (broad/thematic — use community summaries) ────────────

    if any(w in q_lower for w in _GLOBAL_WORDS):
        logger.info("router_decision", decision="GLOBAL", reason="thematic_keywords")
        return "GLOBAL"

    # ── KAG complex signals ───────────────────────────────────────────────────

    if any(w in q_lower for w in _COMPLEX_WORDS):
        logger.info("router_decision", decision="KAG", reason="complex_keywords")
        return "KAG"

    if query.count(" and ") >= 2:
        logger.info("router_decision", decision="KAG", reason="multi_and_clause")
        return "KAG"

    if len(words) > 25:
        logger.info("router_decision", decision="KAG", reason="long_query")
        return "KAG"

    # ── Default ───────────────────────────────────────────────────────────────
    logger.info("router_decision", decision="KAG_SIMPLE", reason="default")
    return "KAG_SIMPLE"
