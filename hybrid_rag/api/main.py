"""
api/main.py — FastAPI application entry-point.

Startup: connect to Neo4j, Qdrant, Redis.
Shutdown: gracefully close all connections.
"""
from __future__ import annotations

# ── Logging must be configured before any other local imports ─────────────────
from hybrid_rag.logging_config import setup_logging
setup_logging(log_level="DEBUG")
# ─────────────────────────────────────────────────────────────────────────────

# ── Set HuggingFace token before any HF/sentence-transformers imports ─────────
import os as _os
from hybrid_rag.config import settings as _settings
if _settings.hf_token:
    _os.environ.setdefault("HF_TOKEN", _settings.hf_token)
    _os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _settings.hf_token)
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from hybrid_rag.api.routes import router

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown using the modern lifespan API."""
    import asyncio as _asyncio
    # ── Startup ───────────────────────────────────────────────────────────────
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client
    from hybrid_rag.storage.cache_manager import cache_manager

    logger.info("startup_begin")

    # Warm up embedding model and cross-encoder BEFORE taking the first request.
    # Both load from local disk/HuggingFace cache — takes 5-35s on first run.
    # Running them in executors here means zero cold-start delay on queries.
    async def _warmup_models():
        loop = _asyncio.get_event_loop()
        try:
            from hybrid_rag.ingestion.embedder import _get_model
            await loop.run_in_executor(None, _get_model)
            logger.info("embedding_model_ready")
        except Exception as exc:
            logger.warning("embedding_model_warmup_failed", error=str(exc))
        try:
            from hybrid_rag.retrieval.reranker import _get_cross_encoder
            await loop.run_in_executor(None, _get_cross_encoder)
            logger.info("cross_encoder_ready")
        except Exception as exc:
            logger.warning("cross_encoder_warmup_failed", error=str(exc))

    # Start model warmup and storage connections concurrently
    warmup_task = _asyncio.create_task(_warmup_models())

    try:
        await neo4j_client.connect()
    except Exception as exc:
        logger.warning("neo4j_startup_failed", error=str(exc))

    try:
        await qdrant_client.connect()
    except Exception as exc:
        logger.warning("qdrant_startup_failed", error=str(exc))

    try:
        await cache_manager.connect()
    except Exception as exc:
        logger.warning("cache_startup_failed", error=str(exc))

    await warmup_task   # ensure models are loaded before accepting requests
    logger.info("startup_complete")

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client
    from hybrid_rag.storage.cache_manager import cache_manager

    await neo4j_client.close()
    await qdrant_client.close()
    await cache_manager.close()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid RAG - GraphRAG + KAG + CAG",
        description=(
            "Production-grade hybrid document Q&A system combining "
            "Graph-based RAG, Knowledge-Augmented Generation, and "
            "Cache-Augmented Generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="")

    # ── Serve static frontend at /ui/ ─────────────────────────────────────────
    if _FRONTEND_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    # Root → redirect to frontend
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui/index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hybrid_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
