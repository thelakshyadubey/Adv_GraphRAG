"""
tests/test_smoke.py — Smoke tests covering all layers of the hybrid RAG system.

Run with:  pytest hybrid_rag/tests/ -v

Most tests mock external services (Neo4j, Qdrant, Redis, Groq) so they can
run without live connections.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import textwrap
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helper: set required env vars before importing settings
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

def test_settings_load():
    from hybrid_rag.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    assert s.groq_api_key == "test_key"
    assert s.chunk_size == 512
    assert s.rrf_k == 60


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_chunk():
    from hybrid_rag.storage.schema import Chunk, ChunkType
    c = Chunk(chunk_id="c1", doc_id="d1", text="hello world", chunk_type=ChunkType.TEXT)
    assert c.chunk_id == "c1"
    assert c.node_ids == []


def test_schema_entity():
    from hybrid_rag.storage.schema import Entity
    e = Entity(node_id="n1", entity_type="Person", name="Alice")
    assert e.properties == {}


def test_schema_query_plan():
    from hybrid_rag.storage.schema import OperatorType, PlanStep, QueryPlan
    plan = QueryPlan(
        original_query="test",
        steps=[PlanStep(step_id=1, operator=OperatorType.HYBRID, sub_query="test")],
    )
    assert plan.steps[0].operator == OperatorType.HYBRID


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parser (file-based)
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_txt():
    from hybrid_rag.ingestion.parser import parse
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello from a text file.\nSecond line.")
        path = f.name
    try:
        result = parse(path)
        assert "Hello" in result.text
        assert result.metadata["file_ext"] == ".txt"
    finally:
        os.unlink(path)


def test_parse_csv():
    from hybrid_rag.ingestion.parser import parse
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("name,age\nAlice,30\nBob,25\n")
        path = f.name
    try:
        result = parse(path)
        assert "Alice" in result.text
        assert result.metadata["row_count"] == 2
    finally:
        os.unlink(path)


def test_parse_missing_file():
    from hybrid_rag.ingestion.parser import parse
    with pytest.raises(FileNotFoundError):
        parse("/nonexistent/path/file.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chunker
# ─────────────────────────────────────────────────────────────────────────────

def test_fixed_chunk_basic():
    from hybrid_rag.ingestion.chunker import fixed_chunk
    text = "This is sentence one. " * 100
    chunks = fixed_chunk(text, size=50, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_table():
    from hybrid_rag.ingestion.chunker import chunk_table
    table = "col1 | col2 | col3\nA | B | C\nD | E | F"
    result = chunk_table(table)
    assert len(result) >= 1
    assert "[TABLE]" in result[0]


def test_auto_chunk_routes_table():
    from hybrid_rag.ingestion.chunker import auto_chunk
    result = auto_chunk("A | B\n1 | 2", content_type="table")
    assert "[TABLE]" in result[0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Embedder
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_embed_shape():
    """Uses a mock to avoid downloading the model in CI."""
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((3, 384))

    with patch("hybrid_rag.ingestion.embedder._get_model", return_value=mock_model):
        from hybrid_rag.ingestion import embedder
        embedder._get_model.cache_clear() if hasattr(embedder._get_model, "cache_clear") else None
        with patch.object(embedder, "_get_model", return_value=mock_model):
            result = embedder.batch_embed(["a", "b", "c"])
            assert len(result) == 3
            assert len(result[0]) == 384


# ─────────────────────────────────────────────────────────────────────────────
# 6. Extractor
# ─────────────────────────────────────────────────────────────────────────────

def test_extractor_parses_valid_json():
    from hybrid_rag.ingestion.extractor import extract

    mock_response = json.dumps({
        "entities": [
            {"name": "Alice", "type": "Person", "properties": {}},
            {"name": "Acme Corp", "type": "Company", "properties": {}},
        ],
        "relations": [
            {"source": "Alice", "target": "Acme Corp", "type": "WORKS_AT", "properties": {}}
        ]
    })

    mock_choice = MagicMock()
    mock_choice.message.content = mock_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = mock_completion

    with patch("hybrid_rag.ingestion.extractor._groq_client", return_value=mock_groq):
        entities, relations = extract("Alice works at Acme Corp.")

    assert len(entities) == 2
    assert entities[0].name == "Alice"
    assert len(relations) == 1
    assert relations[0].rel_type == "WORKS_AT"


def test_extractor_handles_bad_json():
    from hybrid_rag.ingestion.extractor import extract

    mock_choice = MagicMock()
    mock_choice.message.content = "this is not json"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = mock_completion

    with patch("hybrid_rag.ingestion.extractor._groq_client", return_value=mock_groq):
        entities, relations = extract("some text")

    # Should return empty lists gracefully
    assert entities == []
    assert relations == []


# ─────────────────────────────────────────────────────────────────────────────
# 7. RRF Merger
# ─────────────────────────────────────────────────────────────────────────────

def test_rrf_merge_basic():
    from hybrid_rag.retrieval.rrf_merger import rrf_merge
    from hybrid_rag.storage.schema import RetrievalResult

    list1 = [RetrievalResult(id="a", text="A", score=0.9), RetrievalResult(id="b", text="B", score=0.8)]
    list2 = [RetrievalResult(id="b", text="B", score=0.7), RetrievalResult(id="c", text="C", score=0.6)]

    merged = rrf_merge([list1, list2])
    ids = [r.id for r in merged]
    assert "b" in ids
    assert "a" in ids
    assert "c" in ids
    # 'b' should rank first (appears in both lists)
    assert ids[0] == "b"


def test_rrf_merge_empty():
    from hybrid_rag.retrieval.rrf_merger import rrf_merge
    assert rrf_merge([]) == []
    assert rrf_merge([[]]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 8. KAG Planner
# ─────────────────────────────────────────────────────────────────────────────

def test_planner_returns_valid_plan():
    from hybrid_rag.retrieval.kag_planner import plan

    mock_json = json.dumps({"steps": [
        {"step_id": 1, "operator": "HYBRID", "sub_query": "test query", "depends_on": []}
    ]})
    mock_choice = MagicMock()
    mock_choice.message.content = mock_json
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = mock_completion

    with patch("hybrid_rag.retrieval.kag_planner._groq_client", return_value=mock_groq):
        result = plan("test query")

    assert result.original_query == "test query"
    assert len(result.steps) == 1
    assert result.steps[0].operator.value == "HYBRID"


def test_planner_fallback_on_bad_json():
    from hybrid_rag.retrieval.kag_planner import plan

    mock_choice = MagicMock()
    mock_choice.message.content = "not json"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = mock_completion

    with patch("hybrid_rag.retrieval.kag_planner._groq_client", return_value=mock_groq):
        result = plan("fallback test")

    assert len(result.steps) == 1
    assert result.steps[0].operator.value == "HYBRID"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Router
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_exact_cache_returns_cag():
    with patch("hybrid_rag.storage.cache_manager.cache_manager") as mock_cm:
        mock_cm.has_exact = AsyncMock(return_value=True)
        from hybrid_rag.retrieval import router as qr
        with patch.object(qr, "cache_manager", mock_cm):
            # Direct import patch
            import hybrid_rag.retrieval.router as router_module
            original = router_module.cache_manager
            router_module.cache_manager = mock_cm
            try:
                decision = await router_module.route("cached query")
                assert decision == "CAG"
            finally:
                router_module.cache_manager = original


@pytest.mark.asyncio
async def test_router_complex_returns_kag():
    with patch("hybrid_rag.storage.cache_manager.cache_manager") as mock_cm:
        mock_cm.has_exact = AsyncMock(return_value=False)
        import hybrid_rag.retrieval.router as router_module
        original = router_module.cache_manager
        router_module.cache_manager = mock_cm
        try:
            decision = await router_module.route("compare the difference between A and B because of X")
            assert decision == "KAG"
        finally:
            router_module.cache_manager = original


# ─────────────────────────────────────────────────────────────────────────────
# 10. Cache Manager
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_manager_l1():
    from hybrid_rag.storage.cache_manager import CacheManager
    cm = CacheManager()
    cm._redis = None  # disable Redis

    await cm.set("hello world", "the answer", doc_id="doc1")
    cached = await cm.get("hello world")
    assert cached == "the answer"


@pytest.mark.asyncio
async def test_cache_manager_has_exact():
    from hybrid_rag.storage.cache_manager import CacheManager
    cm = CacheManager()
    cm._redis = None

    assert not await cm.has_exact("unknown query")
    await cm.set("known query", "answer")
    assert await cm.has_exact("known query")


@pytest.mark.asyncio
async def test_cache_manager_invalidate():
    from hybrid_rag.storage.cache_manager import CacheManager
    cm = CacheManager()
    cm._redis = None
    cm.set_doc_context("doc99", "some context text")
    assert cm.get_doc_context("doc99") is not None
    await cm.invalidate_doc("doc99")
    assert cm.get_doc_context("doc99") is None


# ─────────────────────────────────────────────────────────────────────────────
# 11. Context Builder
# ─────────────────────────────────────────────────────────────────────────────

def test_context_builder_structure():
    from hybrid_rag.retrieval.context_builder import build
    from hybrid_rag.storage.schema import RetrievalResult

    results = [
        RetrievalResult(id="c1", text="chunk text", score=0.9, source_doc="doc1", source_page=3),
    ]
    ctx = build(results, reasoning_steps=["step1"], query="what is X?")
    assert "RETRIEVED CONTEXT" in ctx
    assert "REASONING STEPS" in ctx
    assert "QUESTION: what is X?" in ctx
    assert "doc1" in ctx
    assert "page 3" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# 12. FastAPI routes (no live services)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def test_client():
    """Create a test client with all external calls mocked."""
    from fastapi.testclient import TestClient
    from hybrid_rag.api.main import app

    # Patch startup to avoid connecting to real services
    with patch("hybrid_rag.api.main.neo4j_client") as mock_neo4j, \
         patch("hybrid_rag.api.main.qdrant_client") as mock_qdrant, \
         patch("hybrid_rag.api.main.cache_manager") as mock_cache:

        mock_neo4j.connect = AsyncMock()
        mock_qdrant.connect = AsyncMock()
        mock_cache.connect = AsyncMock()

        with TestClient(app) as client:
            yield client


def test_health_endpoint(test_client):
    with patch("hybrid_rag.api.routes._check_neo4j", new_callable=lambda: lambda: AsyncMock(return_value=True)), \
         patch("hybrid_rag.api.routes._check_qdrant", new_callable=lambda: lambda: AsyncMock(return_value=True)), \
         patch("hybrid_rag.api.routes._check_redis", new_callable=lambda: lambda: AsyncMock(return_value=True)), \
         patch("hybrid_rag.api.routes._check_groq", new_callable=lambda: lambda: AsyncMock(return_value=True)):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
