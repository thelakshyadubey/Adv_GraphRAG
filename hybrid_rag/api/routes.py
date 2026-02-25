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
        file_size_kb = round(len(content) / 1024, 1)
        logger.info(
            "ingest_upload_received",
            filename=file.filename,
            doc_id=doc_id,
            size_kb=file_size_kb,
        )
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(content)
        stats = await ingest(tmp_path, doc_id)
        logger.info(
            "ingest_upload_complete",
            doc_id=doc_id,
            chunks=stats.get("chunks"),
            entities=stats.get("entities"),
            summaries=stats.get("summaries"),
            communities=stats.get("communities"),
        )
        return {"status": "success", **stats}
    except Exception as exc:
        logger.error("ingest_upload_error", doc_id=doc_id, filename=file.filename, error=str(exc))
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

    logger.info(
        "query_received",
        query=req.query[:120],
        doc_ids=req.doc_ids or "all",
    )

    try:
        path = await query_router.route(req.query, session_ctx)
        logger.info("query_routed", path=path, query=req.query[:80])
        answer = ""
        sources: List[Dict[str, Any]] = []

        # ── CAG path ──────────────────────────────────────────────────────────
        if path == "CAG":
            logger.debug("cag_path", doc_ids=req.doc_ids)
            answer = await cag_engine.answer(req.query, req.doc_ids)
            sources = [{"doc_id": d} for d in (req.doc_ids or [])]
        # ── GLOBAL — community-first path for broad/thematic queries ───────────
        elif path == "GLOBAL":
            from hybrid_rag.retrieval.operators import CommunitySearchOperator
            from hybrid_rag.retrieval.rrf_merger import rrf_merge
            import asyncio as _asyncio

            logger.debug("global_path_start", query=req.query[:80], doc_count=len(req.doc_ids or []))
            comm_op = CommunitySearchOperator()
            vec_op = get_operator("VECTOR_SEARCH")

            # Fetch communities per-doc so every selected doc is represented.
            # A single top-K search skews toward whichever doc has the highest
            # semantic score, leaving other docs unrepresented.
            per_doc_ids = req.doc_ids if req.doc_ids else [None]
            comm_tasks = [
                comm_op.run(req.query, {"doc_ids": [did], "doc_id": did})
                for did in per_doc_ids
            ]
            # Vector search must cover ALL selected docs, not just the first
            vec_task = vec_op.run(req.query, {
                "doc_ids": req.doc_ids or None,
                "doc_id": (req.doc_ids or [None])[0],
            })

            all_tasks = await _asyncio.gather(*comm_tasks, vec_task, return_exceptions=True)
            comm_lists = [r for r in all_tasks[:-1] if isinstance(r, list)]
            vec_list   = all_tasks[-1] if isinstance(all_tasks[-1], list) else []

            # Flatten per-doc community results then merge with vector results
            comm_list = [item for sublist in comm_lists for item in sublist]
            all_results = rrf_merge([comm_list, vec_list])
            logger.debug(
                "global_path_results",
                docs_queried=len(per_doc_ids),
                community_hits=len(comm_list),
                vector_hits=len(vec_list),
                merged_total=len(all_results),
            )
            context = build_context(all_results, query=req.query)
            answer = await _llm_answer(context)
            sources = _extract_sources(all_results)
        # ── KAG_SIMPLE — single HYBRID step ───────────────────────────────────
        elif path == "KAG_SIMPLE":
            logger.debug("kag_simple_path", query=req.query[:80])
            op = get_operator("HYBRID")
            results = await op.run(req.query, {
                "doc_id": (req.doc_ids or [None])[0],
                "doc_ids": req.doc_ids or None,
            })
            logger.debug("kag_simple_results", hits=len(results))
            context = build_context(results, query=req.query)
            answer = await _llm_answer(context)
            sources = _extract_sources(results)

        # ── KAG — multi-step planning ─────────────────────────────────────────
        else:
            query_plan = plan(req.query)
            logger.info(
                "kag_plan_created",
                steps=len(query_plan.steps),
                operators=[s.operator.value for s in query_plan.steps],
            )
            step_results: Dict[int, List[RetrievalResult]] = {}
            reasoning_steps: List[str] = []
            all_results: List[RetrievalResult] = []

            # Group steps into execution waves: steps with no unresolved
            # dependencies can run in parallel; steps that depend on earlier
            # steps wait until their dependencies are done.
            remaining = list(query_plan.steps)
            while remaining:
                # Find all steps whose dependencies are already resolved
                ready = [s for s in remaining if all(d in step_results for d in s.depends_on)]
                if not ready:
                    # Circular or unresolvable — fall back to sequential
                    ready = [remaining[0]]

                async def _run_step(step):
                    prior: List[RetrievalResult] = []
                    for dep_id in step.depends_on:
                        prior.extend(step_results.get(dep_id, []))
                    logger.debug(
                        "kag_step_start",
                        step_id=step.step_id,
                        operator=step.operator.value,
                        sub_query=step.sub_query[:80],
                        depends_on=step.depends_on,
                    )
                    op = get_operator(step.operator.value)
                    ctx = {
                        "prior_results": prior,
                        "doc_id": (req.doc_ids or [None])[0],
                        "doc_ids": req.doc_ids or None,
                    }
                    res = await op.run(step.sub_query, context=ctx)
                    logger.debug(
                        "kag_step_done",
                        step_id=step.step_id,
                        operator=step.operator.value,
                        results=len(res),
                    )
                    return step, res

                wave_results = await asyncio.gather(*[_run_step(s) for s in ready])
                for step, step_res in wave_results:
                    step_results[step.step_id] = step_res
                    all_results.extend(step_res)
                    reasoning_steps.append(f"Step {step.step_id} [{step.operator}]: {step.sub_query}")
                    remaining.remove(step)

            # Final answer via LLM
            context = build_context(
                all_results,
                reasoning_steps=reasoning_steps,
                query=req.query,
            )
            answer = await _llm_answer(context)
            sources = _extract_sources(all_results)

        latency = time.time() * 1000 - start_ms
        logger.info(
            "query_complete",
            path=path,
            latency_ms=round(latency, 1),
            sources=len(sources),
            answer_chars=len(answer),
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            path_used=path,
            latency_ms=round(latency, 2),
        )

    except Exception as exc:
        logger.error("query_api_error", query=req.query[:80], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── /cache/preload ────────────────────────────────────────────────────────────

@router.post("/cache/preload", summary="Force CAG preload for a document")
async def cache_preload(req: CachePreloadRequest) -> Dict[str, str]:
    from hybrid_rag.retrieval.cag_engine import cag_engine
    logger.info("cache_preload_start", doc_id=req.doc_id)
    try:
        await cag_engine.preload(req.doc_id)
        logger.info("cache_preload_done", doc_id=req.doc_id)
        return {"status": "preloaded", "doc_id": req.doc_id}
    except Exception as exc:
        logger.error("cache_preload_error", doc_id=req.doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── DELETE /cache/{doc_id} ──────────────────────────────────────────────────────

@router.delete("/cache/{doc_id}", summary="Invalidate cache for a document")
async def cache_invalidate(doc_id: str) -> Dict[str, str]:
    from hybrid_rag.retrieval.cag_engine import cag_engine
    logger.info("cache_invalidate", doc_id=doc_id)
    await cag_engine.invalidate(doc_id)
    logger.info("cache_invalidate_done", doc_id=doc_id)
    return {"status": "invalidated", "doc_id": doc_id}


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health() -> HealthResponse:
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client
    from hybrid_rag.storage.cache_manager import cache_manager
    from hybrid_rag.config import settings

    logger.debug("health_check_start")
    results = await asyncio.gather(
        _check_neo4j(neo4j_client),
        _check_qdrant(qdrant_client),
        _check_redis(cache_manager),
        _check_groq(settings.groq_api_key),
        return_exceptions=True,
    )

    statuses = ["ok" if r is True else f"error: {type(r).__name__}: {r}" for r in results]

    overall = "healthy" if all(s == "ok" for s in statuses) else "degraded"
    logger.info(
        "health_check_complete",
        status=overall,
        neo4j=statuses[0],
        qdrant=statuses[1],
        redis=statuses[2],
        groq=statuses[3],
    )
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
