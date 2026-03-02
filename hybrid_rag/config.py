"""
config.py — All settings loaded from environment variables via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known embedding model → expected output dimension.
# Used by the startup validator to catch mismatched config before anything
# is stored in Qdrant (a mismatch would cause silent upsert failures).
_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
}

# Resolve .env relative to this file so it works regardless of CWD
_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None          # required for Qdrant Cloud

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Groq ───────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_llm_model: str = "llama-3.3-70b-versatile"  # answering — quality matters
    groq_fast_model: str = "llama-3.1-8b-instant"     # planning/routing — speed matters
    # Groq does not offer embeddings — we use sentence-transformers locally
    groq_embed_model: str = "all-MiniLM-L6-v2"   # local ST model

    # ── Embedding ──────────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2 → 384  |  nomic-ai/nomic-embed-text-v1.5 → 768
    embedding_dim: int = 384

    # ── HuggingFace ────────────────────────────────────────────────────────────
    hf_token: Optional[str] = None          # set HF_TOKEN in .env to avoid rate limits

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_threshold: float = 0.75
    embed_batch_size: int = 32          # batch size for SentenceTransformer.encode()

    # ── Retrieval ──────────────────────────────────────────────────────────────
    rrf_k: int = 60
    retrieval_top_k: int = 10           # max chunks fetched from Qdrant / Neo4j
    rerank_top_k: int = 4               # chunks kept after cross-encoder reranking → LLM context
    community_search_limit: int = 3     # top-N community summaries per doc in GLOBAL path
    plan_cache_size: int = 256          # max KAG query plans cached in RAM

    # ── Qdrant collection names ────────────────────────────────────────────────
    qdrant_chunks_collection: str = "chunks"
    qdrant_summaries_collection: str = "summaries"
    qdrant_entities_collection: str = "entities"

    # ── Neo4j index / label names ─────────────────────────────────────────────
    neo4j_entity_label: str = "Entity"
    neo4j_fulltext_index: str = "entity_name_ft"
    neo4j_node_id_index: str = "entity_node_id"

    # ── Cache ──────────────────────────────────────────────────────────────────
    cache_ttl: int = 3600               # Redis L2 TTL in seconds
    l1_cache_max: int = 1000            # max entries in L1 RAM LRU cache
    semantic_cache_threshold: float = 0.92  # cosine similarity above which queries are treated as equivalent

    # ── CAG ────────────────────────────────────────────────────────────────────
    cag_context_limit: int = 6000       # approx token budget for context window
    cag_min_access_count: int = 5

    # ── Startup validator ─────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _check_embedding_dim(self) -> "Settings":
        """Catch model/dim mismatch before anything is written to Qdrant."""
        expected = _MODEL_DIMS.get(self.groq_embed_model)
        if expected is not None and self.embedding_dim != expected:
            raise ValueError(
                f"embedding_dim={self.embedding_dim} does not match "
                f"groq_embed_model='{self.groq_embed_model}' "
                f"(expected {expected}). "
                f"Update EMBEDDING_DIM in your .env to {expected}."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
