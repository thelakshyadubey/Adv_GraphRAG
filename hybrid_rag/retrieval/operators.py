"""
retrieval/operators.py — The 6 retrieval operators used by the KAG planner.

Each operator returns List[RetrievalResult].

  GraphExactOperator    — Neo4j name search + subgraph traversal
  VectorSearchOperator  — Qdrant dense vector search on chunks
  HybridOperator        — Graph + Vector in parallel → RRF merge → mutual expand
  MultiHopOperator      — Neo4j BFS multi-hop reasoning
  SummaryOperator       — Qdrant search over doc/section/entity summaries
  LLMReasonOperator     — Direct Groq LLM call with assembled context
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog

from hybrid_rag.config import settings
from hybrid_rag.ingestion import embedder
from hybrid_rag.storage.schema import Entity, RetrievalResult

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


def _llm_call(prompt: str, max_tokens: int = 1024) -> str:
    client = _groq_client()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"LLM call failed: {last_exc}")


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseOperator:
    async def run(self, sub_query: str, context: Dict[str, Any] = None) -> List[RetrievalResult]:
        raise NotImplementedError


# ── 1. Graph Exact ────────────────────────────────────────────────────────────

class GraphExactOperator(BaseOperator):
    """Search Neo4j by entity name, then traverse the subgraph."""

    async def run(
        self, sub_query: str, context: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        from hybrid_rag.storage.neo4j_client import neo4j_client
        from hybrid_rag.storage.qdrant_client import qdrant_client
        from qdrant_client.http import models as qm  # type: ignore

        ctx = context or {}
        doc_ids: Optional[List[str]] = ctx.get("doc_ids") or (
            [ctx["doc_id"]] if ctx.get("doc_id") else None
        )

        try:
            entities: List[Entity] = await neo4j_client.search_by_name(sub_query, fuzzy=True)
            if not entities:
                return []

            results: List[RetrievalResult] = []
            for entity in entities[:5]:
                chunk_ids = await neo4j_client.get_chunks_for_node(entity.node_id)
                for cid in chunk_ids[:3]:
                    # Fetch chunk text from Qdrant filtered by chunk_id AND doc scope
                    must = [qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=cid))]
                    if doc_ids:
                        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)))
                    flt = qm.Filter(must=must)
                    records, _ = await qdrant_client._client.scroll(
                        collection_name="chunks",
                        scroll_filter=flt,
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for rec in records:
                        p = rec.payload or {}
                        results.append(RetrievalResult(
                            id=cid,
                            text=p.get("text", ""),
                            score=0.9,
                            source_doc=p.get("doc_id"),
                            source_page=p.get("page"),
                            node_ids=[entity.node_id],
                        ))

            logger.info("graph_exact_results", count=len(results))
            return results
        except Exception as exc:
            logger.error("graph_exact_failed", error=str(exc))
            return []


# ── 2. Vector Search ──────────────────────────────────────────────────────────

class VectorSearchOperator(BaseOperator):
    """Dense vector search over the chunks collection in Qdrant."""

    async def run(
        self,
        sub_query: str,
        context: Dict[str, Any] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        from hybrid_rag.storage.qdrant_client import qdrant_client

        ctx = context or {}
        try:
            q_vec = embedder.embed(sub_query)
            doc_ids: Optional[List[str]] = ctx.get("doc_ids") or (
                [ctx["doc_id"]] if ctx.get("doc_id") else None
            )
            results = await qdrant_client.search_chunks(
                q_vec, doc_ids=doc_ids, limit=limit
            )
            logger.info("vector_search_results", count=len(results))
            return results
        except Exception as exc:
            logger.error("vector_search_failed", error=str(exc))
            return []


# ── 3. Hybrid ──────────────────────────────────────────────────────────────────

class HybridOperator(BaseOperator):
    """
    Run GraphExact + VectorSearch in parallel, merge with RRF,
    then expand via mutual index.
    """

    def __init__(self) -> None:
        self._graph = GraphExactOperator()
        self._vector = VectorSearchOperator()

    async def run(
        self, sub_query: str, context: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        from hybrid_rag.ingestion.mutual_index import expand
        from hybrid_rag.retrieval.rrf_merger import rrf_merge

        try:
            graph_res, vector_res = await asyncio.gather(
                self._graph.run(sub_query, context),
                self._vector.run(sub_query, context),
                return_exceptions=True,
            )
            graph_list = graph_res if isinstance(graph_res, list) else []
            vector_list = vector_res if isinstance(vector_res, list) else []

            merged = rrf_merge([graph_list, vector_list])

            # Expand via mutual index
            original_ids = [r.id for r in merged[:10]]
            expanded_ids = await expand(original_ids)

            # Fetch expanded chunks from Qdrant (best effort)
            from hybrid_rag.storage.qdrant_client import qdrant_client
            from qdrant_client.http import models as qm  # type: ignore
            ctx = context or {}
            doc_ids_ctx: Optional[List[str]] = ctx.get("doc_ids") or (
                [ctx["doc_id"]] if ctx.get("doc_id") else None
            )
            extra: List[RetrievalResult] = []
            for cid in expanded_ids[:5]:
                must = [qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=cid))]
                if doc_ids_ctx:
                    must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids_ctx)))
                flt = qm.Filter(must=must)
                records, _ = await qdrant_client._client.scroll(
                    collection_name="chunks", scroll_filter=flt,
                    limit=1, with_payload=True, with_vectors=False,
                )
                for rec in records:
                    p = rec.payload or {}
                    extra.append(RetrievalResult(
                        id=cid, text=p.get("text", ""), score=0.5,
                        source_doc=p.get("doc_id"), source_page=p.get("page"),
                        node_ids=p.get("node_ids", []),
                    ))

            final = rrf_merge([merged, extra]) if extra else merged
            logger.info("hybrid_results", count=len(final))
            return final
        except Exception as exc:
            logger.error("hybrid_failed", error=str(exc))
            return []


# ── 4. Multi-hop ──────────────────────────────────────────────────────────────

class MultiHopOperator(BaseOperator):
    """BFS multi-hop traversal starting from any named entity in the query."""

    async def run(
        self, sub_query: str, context: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        from hybrid_rag.storage.neo4j_client import neo4j_client
        from hybrid_rag.storage.qdrant_client import qdrant_client

        try:
            seed_entities = await neo4j_client.search_by_name(sub_query, fuzzy=True)
            if not seed_entities:
                return []

            all_hops = []
            for e in seed_entities[:2]:
                hops = await neo4j_client.multi_hop(e.node_id, depth=3)
                all_hops.extend(hops)

            # Collect chunk texts reachable via hops
            q_vec = embedder.embed(sub_query)
            results: List[RetrievalResult] = []
            hop_chunk_ids: set[str] = set()
            for hop in all_hops[:20]:
                for cid in (hop.get("chunk_ids") or []):
                    hop_chunk_ids.add(cid)

            from qdrant_client.http import models as qm  # type: ignore
            ctx = context or {}
            doc_ids: Optional[List[str]] = ctx.get("doc_ids") or (
                [ctx["doc_id"]] if ctx.get("doc_id") else None
            )
            for cid in list(hop_chunk_ids)[:10]:
                must = [qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=cid))]
                if doc_ids:
                    must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)))
                flt = qm.Filter(must=must)
                records, _ = await qdrant_client._client.scroll(
                    collection_name="chunks", scroll_filter=flt,
                    limit=1, with_payload=True, with_vectors=False,
                )
                for rec in records:
                    p = rec.payload or {}
                    results.append(RetrievalResult(
                        id=cid, text=p.get("text", ""), score=0.75,
                        source_doc=p.get("doc_id"), source_page=p.get("page"),
                        node_ids=p.get("node_ids", []),
                    ))

            logger.info("multi_hop_results", count=len(results))
            return results
        except Exception as exc:
            logger.error("multi_hop_failed", error=str(exc))
            return []


# ── 5. Summary ────────────────────────────────────────────────────────────────

class SummaryOperator(BaseOperator):
    """Search Qdrant summaries collection for relevant doc/section/entity summaries."""

    async def run(
        self, sub_query: str, context: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        from hybrid_rag.storage.qdrant_client import qdrant_client

        try:
            q_vec = embedder.embed(sub_query)
            results = await qdrant_client.search_summaries(q_vec, limit=3)
            logger.info("summary_results", count=len(results))
            return results
        except Exception as exc:
            logger.error("summary_search_failed", error=str(exc))
            return []

# ── 6. Community Search ───────────────────────────────────────────────────

class CommunitySearchOperator(BaseOperator):
    """
    Search the pre-built GraphRAG community summaries stored in Qdrant.

    Best used for broad/global/thematic queries such as:
      "What are the main themes?"
      "Summarize the key relationships."
      "Give me an overview of this document."

    Returns community-level summaries (result_type="community") that describe
    clusters of related entities, enabling coherent answers to high-level questions.
    """

    async def run(
        self, sub_query: str, context: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        from hybrid_rag.storage.qdrant_client import qdrant_client

        ctx = context or {}
        doc_ids: Optional[List[str]] = ctx.get("doc_ids") or (
            [ctx["doc_id"]] if ctx.get("doc_id") else None
        )
        try:
            q_vec = embedder.embed(sub_query)
            results = await qdrant_client.search_communities(q_vec, doc_ids=doc_ids, limit=5)
            logger.info("community_search_results", count=len(results))
            return results
        except Exception as exc:
            logger.error("community_search_failed", error=str(exc))
            return []

# ── 6. LLM Reason ─────────────────────────────────────────────────────────────

class LLMReasonOperator(BaseOperator):
    """
    Use the LLM to reason over already-retrieved context from previous steps.
    The context is passed via context["prior_results"].
    """

    async def run(
        self, sub_query: str, context: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        ctx = context or {}
        prior: List[RetrievalResult] = ctx.get("prior_results", [])

        context_text = "\n\n".join(r.text for r in prior[:8])
        prompt = (
            f"Context:\n\n{context_text}\n\n"
            f"Question: {sub_query}\n\n"
            "Give a direct, specific answer using the facts in the context above. "
            "State the answer as a plain fact — do NOT say 'the context says', "
            "'the context mentions', or 'the context does not explicitly'. "
            "If the context contains relevant facts, use them confidently."
        )
        try:
            answer = _llm_call(prompt, max_tokens=1024)
            doc_ids = list({r.source_doc for r in prior if r.source_doc})
            return [RetrievalResult(
                id="llm_reason_result",
                text=answer,
                score=1.0,
                source_doc=doc_ids[0] if doc_ids else None,
            )]
        except Exception as exc:
            logger.error("llm_reason_failed", error=str(exc))
            return []


# ── Operator registry ─────────────────────────────────────────────────────────

REGISTRY: Dict[str, BaseOperator] = {
    "GRAPH_EXACT": GraphExactOperator(),
    "VECTOR_SEARCH": VectorSearchOperator(),
    "HYBRID": HybridOperator(),
    "MULTI_HOP": MultiHopOperator(),
    "SUMMARY": SummaryOperator(),
    "LLM_REASON": LLMReasonOperator(),
    "COMMUNITY_SEARCH": CommunitySearchOperator(),
}


def get_operator(name: str) -> BaseOperator:
    op = REGISTRY.get(name.upper())
    if op is None:
        logger.warning("unknown_operator_fallback", name=name)
        return REGISTRY["HYBRID"]
    return op
