"""
api/main.py — FastAPI application entry-point.

Startup: connect to Neo4j, Qdrant, Redis.
Shutdown: gracefully close all connections.
"""
from __future__ import annotations

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
    # ── Startup ───────────────────────────────────────────────────────────────
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client
    from hybrid_rag.storage.cache_manager import cache_manager

    logger.info("startup_begin")
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
