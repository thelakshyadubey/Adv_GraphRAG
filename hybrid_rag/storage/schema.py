"""
schema.py — All Pydantic models used across the system.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CODE = "code"


class SummaryType(str, Enum):
    DOC = "doc"
    SECTION = "section"
    ENTITY = "entity"
    COMMUNITY = "community"


class OperatorType(str, Enum):
    GRAPH_EXACT = "GRAPH_EXACT"
    VECTOR_SEARCH = "VECTOR_SEARCH"
    HYBRID = "HYBRID"
    MULTI_HOP = "MULTI_HOP"
    SUMMARY = "SUMMARY"
    LLM_REASON = "LLM_REASON"
    COMMUNITY_SEARCH = "COMMUNITY_SEARCH"


# ── Core document models ───────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    embedding: Optional[List[float]] = None
    chunk_type: ChunkType = ChunkType.TEXT
    section: Optional[str] = None
    page: Optional[int] = None
    node_ids: List[str] = Field(default_factory=list)
    position: int = 0                # index within the document


class Entity(BaseModel):
    node_id: str
    entity_type: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    chunk_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None


class Relation(BaseModel):
    rel_id: str
    source_id: str
    target_id: str
    rel_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    summary_id: str
    summary_type: SummaryType
    parent_id: str                   # doc_id, section name, or node_id
    text: str
    embedding: Optional[List[float]] = None
    # Community-specific fields (populated by community_builder)
    entity_names: List[str] = Field(default_factory=list)
    community_id: Optional[int] = None


# ── Query planning models ──────────────────────────────────────────────────────

class PlanStep(BaseModel):
    step_id: int
    operator: OperatorType
    sub_query: str
    depends_on: List[int] = Field(default_factory=list)


class QueryPlan(BaseModel):
    original_query: str
    steps: List[PlanStep]


# ── Retrieval result ───────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    id: str
    text: str
    score: float = 0.0
    source_doc: Optional[str] = None
    source_page: Optional[int] = None
    node_ids: List[str] = Field(default_factory=list)
    result_type: str = "chunk"    # "chunk" | "community" | "summary"


# ── API request / response models ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    file_path: str
    doc_id: str


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    path_used: str = ""
    latency_ms: float = 0.0


class CachePreloadRequest(BaseModel):
    doc_id: str


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    qdrant: str
    redis: str
    groq: str
