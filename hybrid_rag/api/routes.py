"""
api/routes.py — All HTTP route handlers.

Endpoints:
  POST   /ingest            — ingest a document
  POST   /query             — query with smart routing
  POST   /cache/preload     — force CAG preload for a doc
  DELETE /cache/{doc_id}    — invalidate cache for a doc
  GET    /health            — check all service connections
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import os
import pathlib
import tempfile

import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from hybrid_rag.storage.schema import (
    CachePreloadRequest,
    HealthResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── /ingest ───────────────────────────────────────────────────────────────────

@router.post("/ingest", summary="Ingest a document")
async def ingest_document(req: IngestRequest) -> Dict[str, Any]:
    """
    Parse, chunk, embed, and index a document.
    Entities and relations are extracted and stored in Neo4j.
    Chunks and summaries are stored in Qdrant.
    """
    from hybrid_rag.ingestion.pipeline import ingest

    try:
        stats = await ingest(req.file_path, req.doc_id)
        return {"status": "success", **stats}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("ingest_api_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── /ingest/upload ────────────────────────────────────────────────────────────

@router.post("/ingest/upload", summary="Ingest a document via file upload")
async def ingest_document_upload(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
) -> dict:
    """
    Accept a multipart file upload, persist to a temp file, run the full
    ingestion pipeline, then delete the temp file.
    """
    from hybrid_rag.ingestion.pipeline import ingest

    suffix = pathlib.Path(file.filename or "upload").suffix or ".bin"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await file.read()
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(content)
        stats = await ingest(tmp_path, doc_id)
        return {"status": "success", **stats}
    except Exception as exc:
        logger.error("ingest_upload_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── /query ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Query the system")
async def query(req: QueryRequest) -> QueryResponse:
    """
    Route the query to CAG / KAG / KAG_SIMPLE and return the answer.
    """
    from hybrid_rag.retrieval import router as query_router
    from hybrid_rag.retrieval.cag_engine import cag_engine
    from hybrid_rag.retrieval.kag_planner import plan
    from hybrid_rag.retrieval.operators import get_operator
    from hybrid_rag.retrieval.context_builder import build as build_context
    from hybrid_rag.storage.schema import RetrievalResult

    start_ms = time.time() * 1000
    session_ctx: Dict[str, Any] = {}
    if req.doc_ids:
        session_ctx["cached_docs"] = req.doc_ids

    try:
        path = await query_router.route(req.query, session_ctx)
        answer = ""
        sources: List[Dict[str, Any]] = []

        # ── CAG path ──────────────────────────────────────────────────────────
        if path == "CAG":
            answer = await cag_engine.answer(req.query, req.doc_ids)
            sources = [{"doc_id": d} for d in (req.doc_ids or [])]
        # ── GLOBAL — community-first path for broad/thematic queries ───────────
        elif path == "GLOBAL":
            from hybrid_rag.retrieval.operators import CommunitySearchOperator
            from hybrid_rag.retrieval.rrf_merger import rrf_merge
            import asyncio as _asyncio

            comm_op = CommunitySearchOperator()
            vec_op = get_operator("VECTOR_SEARCH")
            comm_res, vec_res = await _asyncio.gather(
                comm_op.run(req.query, {"doc_ids": req.doc_ids,
                                        "doc_id": (req.doc_ids or [None])[0]}),
                vec_op.run(req.query, {"doc_id": (req.doc_ids or [None])[0]}),
                return_exceptions=True,
            )
            comm_list = comm_res if isinstance(comm_res, list) else []
            vec_list  = vec_res  if isinstance(vec_res,  list) else []
            all_results = rrf_merge([comm_list, vec_list])
            context = build_context(all_results, query=req.query)
            answer = await _llm_answer(context)
            sources = _extract_sources(all_results)
        # ── KAG_SIMPLE — single HYBRID step ───────────────────────────────────
        elif path == "KAG_SIMPLE":
            op = get_operator("HYBRID")
            results = await op.run(req.query, {"doc_id": (req.doc_ids or [None])[0]})
            context = build_context(results, query=req.query)
            answer = await _llm_answer(context)
            sources = _extract_sources(results)

        # ── KAG — multi-step planning ─────────────────────────────────────────
        else:
            query_plan = plan(req.query)
            step_results: Dict[int, List[RetrievalResult]] = {}
            reasoning_steps: List[str] = []
            all_results: List[RetrievalResult] = []

            for step in query_plan.steps:
                # Gather results from dependency steps
                prior: List[RetrievalResult] = []
                for dep_id in step.depends_on:
                    prior.extend(step_results.get(dep_id, []))

                op = get_operator(step.operator.value)
                ctx = {"prior_results": prior, "doc_id": (req.doc_ids or [None])[0]}
                step_res = await op.run(step.sub_query, context=ctx)
                step_results[step.step_id] = step_res
                all_results.extend(step_res)
                reasoning_steps.append(f"Step {step.step_id} [{step.operator}]: {step.sub_query}")

            # Final answer via LLM
            context = build_context(
                all_results,
                reasoning_steps=reasoning_steps,
                query=req.query,
            )
            answer = await _llm_answer(context)
            sources = _extract_sources(all_results)

        latency = time.time() * 1000 - start_ms
        return QueryResponse(
            answer=answer,
            sources=sources,
            path_used=path,
            latency_ms=round(latency, 2),
        )

    except Exception as exc:
        logger.error("query_api_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── /cache/preload ────────────────────────────────────────────────────────────

@router.post("/cache/preload", summary="Force CAG preload for a document")
async def cache_preload(req: CachePreloadRequest) -> Dict[str, str]:
    from hybrid_rag.retrieval.cag_engine import cag_engine
    try:
        await cag_engine.preload(req.doc_id)
        return {"status": "preloaded", "doc_id": req.doc_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── DELETE /cache/{doc_id} ────────────────────────────────────────────────────

@router.delete("/cache/{doc_id}", summary="Invalidate cache for a document")
async def cache_invalidate(doc_id: str) -> Dict[str, str]:
    from hybrid_rag.retrieval.cag_engine import cag_engine
    await cag_engine.invalidate(doc_id)
    return {"status": "invalidated", "doc_id": doc_id}


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health() -> HealthResponse:
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client
    from hybrid_rag.storage.cache_manager import cache_manager
    from hybrid_rag.config import settings

    results = await asyncio.gather(
        _check_neo4j(neo4j_client),
        _check_qdrant(qdrant_client),
        _check_redis(cache_manager),
        _check_groq(settings.groq_api_key),
        return_exceptions=True,
    )

    statuses = ["ok" if r is True else f"error: {type(r).__name__}: {r}" for r in results]

    overall = "healthy" if all(s == "ok" for s in statuses) else "degraded"
    return HealthResponse(
        status=overall,
        neo4j=statuses[0],
        qdrant=statuses[1],
        redis=statuses[2],
        groq=statuses[3],
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _llm_answer(context: str) -> str:
    """Async-friendly wrapper for the Groq LLM call."""
    import asyncio
    from hybrid_rag.config import settings
    from groq import Groq  # type: ignore

    _SYSTEM_PROMPT = (
        "You are an expert research analyst who gives direct, precise answers. "
        "Rules you must always follow:\n"
        "- Lead every answer with the direct answer itself — never with a preamble about what the context contains.\n"
        "- NEVER say: 'the context does not explicitly', 'although the context', 'based on the context', "
        "'the context mentions', 'the context says', 'not explicitly stated', 'while not explicitly'.\n"
        "- State retrieved facts as plain facts, not as attributed quotes from a document.\n"
        "- Synthesize and reason across all provided information, including implicit clues.\n"
        "- If context is genuinely irrelevant, say only: 'Insufficient context to answer.'"
    )

    def _sync():
        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)


def _extract_sources(results: List) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    sources: List[Dict[str, Any]] = []
    for r in results:
        key = f"{r.source_doc}:{r.source_page}"
        if key not in seen and r.source_doc:
            seen.add(key)
            sources.append({"doc_id": r.source_doc, "page": r.source_page})
    return sources


async def _check_neo4j(client) -> bool:
    try:
        await client.run_cypher("RETURN 1")
        return True
    except Exception as exc:
        return exc


async def _check_qdrant(client) -> bool:
    try:
        if client._client is None:
            raise RuntimeError("Client not initialised — startup connection failed")
        await client._client.get_collections()
        return True
    except Exception as exc:
        return exc


async def _check_redis(cm) -> bool:
    try:
        if cm._redis:
            await cm._redis.ping()
        return True
    except Exception as exc:
        return exc


async def _check_groq(api_key: str) -> bool:
    try:
        from groq import Groq  # type: ignore
        Groq(api_key=api_key)  # just instantiate — actual call is unnecessary
        return True
    except Exception as exc:
        return exc
