"""
ingestion/embedder.py — Local sentence-transformer batch embedding.

Groq does not expose an embeddings API.  We use sentence-transformers locally:
  - all-MiniLM-L6-v2            → dim 384  (fast, good quality)
  - nomic-ai/nomic-embed-text-v1.5 → dim 768  (best quality)

The model is chosen by settings.groq_embed_model.
"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import List

import numpy as np
import structlog

# Suppress HuggingFace symlink warning on Windows — caching still works fine
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*symlinks.*", category=UserWarning)

from hybrid_rag.config import settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once and cache it."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("loading_embedding_model", model=settings.groq_embed_model)
        return SentenceTransformer(settings.groq_embed_model)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embeddings: "
            "pip install sentence-transformers"
        )


def embed(text: str) -> List[float]:
    """Embed a single string."""
    model = _get_model()
    vec = model.encode(text, show_progress_bar=False)
    return vec.tolist()


def batch_embed(texts: List[str], batch_size: int | None = None) -> List[List[float]]:
    """
    Embed a list of strings in batches.

    Returns
    -------
    List of float vectors, one per input text.
    """
    if not texts:
        return []
    if batch_size is None:
        batch_size = settings.embed_batch_size
    try:
        model = _get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        logger.info("batch_embed_done", count=len(texts))
        return embeddings.tolist()
    except Exception as exc:
        logger.error("batch_embed_failed", error=str(exc))
        raise
