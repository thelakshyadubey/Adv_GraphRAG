"""
config.py — All settings loaded from environment variables via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

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
    groq_llm_model: str = "llama-3.3-70b-versatile"
    # Groq does not offer embeddings — we use sentence-transformers locally
    groq_embed_model: str = "all-MiniLM-L6-v2"   # local ST model

    # ── Embedding ──────────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2 → 384  |  nomic-ai/nomic-embed-text-v1.5 → 768
    embedding_dim: int = 384

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_threshold: float = 0.75

    # ── Retrieval ──────────────────────────────────────────────────────────────
    rrf_k: int = 60

    # ── CAG ────────────────────────────────────────────────────────────────────
    cag_context_limit: int = 6000        # approx token budget for context window
    cag_min_access_count: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
