"""
retrieval/router.py — LLM Agent Query Router.

CAG  is decided by cache/preload state (no LLM needed — it is purely about
     whether data is already in memory).

GLOBAL / KAG / KAG_SIMPLE are decided by a Groq LLM agent that reads the
query and reasons about which retrieval strategy best fits it.
The agent receives only the definitions of each path and the query — no
keyword lists, no hardcoded rules, no examples.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List

import structlog

from hybrid_rag.config import settings

logger = structlog.get_logger(__name__)

# ── Router agent prompt ───────────────────────────────────────────────────────
# Describes what each path IS and what kind of query belongs to it.
# The LLM reasons about the query intent and returns a single JSON decision.

_ROUTER_PROMPT = """\
You are a query routing agent for a document intelligence system.
Your job is to read a user query and decide which retrieval strategy to use.

Available strategies:

GLOBAL
  Purpose: Answer questions about the overall document — broad themes, main topics,
           general summaries, high-level overviews, what the document is about as a whole.
  Use when: The query asks about the document at a macro level, across all content,
            or wants a synthesized view of the entire material.

KAG
  Purpose: Answer complex, multi-step, or relational questions that require reasoning
           across multiple pieces of information, understanding cause and effect,
           comparisons, chains of logic, or deep explanations.
  Use when: The query requires connecting multiple concepts, tracing relationships,
            understanding why something happens, or comparing different things.

KAG_SIMPLE
  Purpose: Answer specific, focused, factual questions about a particular concept,
           definition, value, person, date, or piece of information in the document.
  Use when: The query has a clear single target and can likely be answered from
            one or two relevant passages.

Instructions:
- Read the query carefully.
- Reason about what kind of answer it needs, not what words it contains.
- Return ONLY valid JSON with a single key "route".
- Do not explain your reasoning. Do not add any other text.

Query: {query}

Response format:
{{"route": "GLOBAL" | "KAG" | "KAG_SIMPLE"}}
"""

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


def _call_router_agent(query: str) -> str:
    """
    Call the Groq LLM router agent.
    Returns one of: "GLOBAL" | "KAG" | "KAG_SIMPLE"
    Falls back to KAG_SIMPLE on any error or invalid response.
    """
    client = _groq_client()
    prompt = _ROUTER_PROMPT.format(query=query)
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,        # deterministic — routing must be consistent
                max_tokens=32,          # only needs {"route": "KAG_SIMPLE"} — 10 tokens max
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            decision = data.get("route", "KAG_SIMPLE").upper().strip()

            if decision not in {"GLOBAL", "KAG", "KAG_SIMPLE"}:
                logger.warning("router_agent_invalid_decision", raw=raw, fallback="KAG_SIMPLE")
                return "KAG_SIMPLE"

            logger.info("router_agent_decision", query=query[:80], decision=decision)
            return decision

        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning("router_agent_retry", attempt=attempt, error=str(exc), wait=wait)
            time.sleep(wait)

    logger.error("router_agent_failed", error=str(last_exc), fallback="KAG_SIMPLE")
    return "KAG_SIMPLE"


# ── Public API ────────────────────────────────────────────────────────────────

async def route(query: str, session_ctx: Dict | None = None) -> str:
    """
    Determine the retrieval strategy for the given query.
    Returns one of: "CAG" | "KAG" | "KAG_SIMPLE" | "GLOBAL"
    """
    from hybrid_rag.storage.cache_manager import cache_manager

    ctx = session_ctx or {}

    # ── CAG signals — purely about memory/cache state, no LLM needed ─────────

    cached_docs: List[str] = ctx.get("cached_docs", [])
    if cached_docs:
        preloaded = set(cache_manager._doc_contexts.keys())
        all_covered = all(doc in preloaded for doc in cached_docs)
        if all_covered:
            logger.info("router_decision", decision="CAG", reason="all_docs_preloaded",
                        preloaded_docs=list(preloaded & set(cached_docs)))
            return "CAG"
        elif preloaded & set(cached_docs):
            logger.info("router_decision_partial_cache",
                        preloaded=list(preloaded & set(cached_docs)),
                        missing=list(set(cached_docs) - preloaded),
                        note="falling_through_to_retrieval")

    # ── LLM agent routing for GLOBAL / KAG / KAG_SIMPLE ─────────────────────
    # Run in a thread executor so the blocking Groq HTTP call doesn't stall
    # the async event loop.
    decision = await asyncio.get_event_loop().run_in_executor(
        None, _call_router_agent, query
    )
    logger.info("router_decision", decision=decision, reason="llm_agent")
    return decision
