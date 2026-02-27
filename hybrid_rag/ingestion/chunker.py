"""
ingestion/chunker.py — Semantic + fixed-size + table-aware chunking.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import structlog

from hybrid_rag.config import settings

logger = structlog.get_logger(__name__)

# Sentence splitter (handles common abbreviations)
_SENT_RE = re.compile(r'(?<=[.!?])\s+')

# Rough chars-per-token ratio used for size estimation when tiktoken is absent
_CHARS_PER_TOKEN = 4


def _token_count(text: str) -> int:
    """Estimate token count. Uses tiktoken if available, otherwise uses char ratio."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // _CHARS_PER_TOKEN)


def _sentences(text: str) -> List[str]:
    sents = _SENT_RE.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


# ── Public API ────────────────────────────────────────────────────────────────

def semantic_chunk(
    text: str,
    threshold: float | None = None,
) -> List[str]:
    """
    Split text into variable-size chunks by detecting similarity drops between
    consecutive sentences.  Falls back to fixed_chunk if sentence-transformers
    embeddings fail.

    Parameters
    ----------
    text      : raw text string
    threshold : cosine similarity threshold below which a new chunk starts
    """
    if threshold is None:
        threshold = settings.semantic_threshold

    sents = _sentences(text)
    if len(sents) <= 2:
        return [text] if text.strip() else []

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore

        model = SentenceTransformer(settings.groq_embed_model)
        embeddings = model.encode(sents, batch_size=settings.embed_batch_size, show_progress_bar=False)

        chunks: List[str] = []
        current: List[str] = [sents[0]]

        for i in range(1, len(sents)):
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            # Cosine similarity
            sim = float(
                np.dot(prev_emb, curr_emb)
                / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb) + 1e-9)
            )

            candidate = " ".join(current + [sents[i]])
            too_long = _token_count(candidate) > settings.chunk_size
            too_short = _token_count(" ".join(current)) < 100

            if (sim < threshold and not too_short) or too_long:
                chunks.append(" ".join(current))
                current = [sents[i]]
            else:
                current.append(sents[i])

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]

    except Exception as exc:
        logger.warning("semantic_chunk_fallback", error=str(exc))
        return fixed_chunk(text)


def fixed_chunk(
    text: str,
    size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """
    Token-based sliding window chunker.  Preserves sentence boundaries where
    possible by operating on sentences rather than raw tokens.
    """
    if size is None:
        size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    sents = _sentences(text)
    if not sents:
        return []

    chunks: List[str] = []
    current_sents: List[str] = []
    current_tokens = 0

    for sent in sents:
        sent_tokens = _token_count(sent)
        if current_tokens + sent_tokens > size and current_sents:
            chunks.append(" ".join(current_sents))
            # Keep overlap sentences
            while current_sents and current_tokens > overlap:
                removed = current_sents.pop(0)
                current_tokens -= _token_count(removed)

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        chunks.append(" ".join(current_sents))

    return [c for c in chunks if c.strip()]


def chunk_table(table_text: str) -> List[str]:
    """
    Preserve the entire table as a single chunk, prefixed with a [TABLE] marker.
    Large tables are split row-wise while always keeping the header.
    """
    lines = table_text.strip().splitlines()
    if not lines:
        return []

    header = lines[0]
    rows = lines[1:]

    # If the whole table fits in one chunk, return it
    if _token_count(table_text) <= settings.chunk_size:
        return [f"[TABLE]\n{table_text}\n[/TABLE]"]

    # Otherwise split row groups, but always include the header
    chunks: List[str] = []
    current_rows: List[str] = []
    current_tokens = _token_count(header)

    for row in rows:
        row_tokens = _token_count(row)
        if current_tokens + row_tokens > settings.chunk_size and current_rows:
            chunk_text = f"[TABLE]\n{header}\n" + "\n".join(current_rows) + "\n[/TABLE]"
            chunks.append(chunk_text)
            current_rows = []
            current_tokens = _token_count(header)
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        chunk_text = f"[TABLE]\n{header}\n" + "\n".join(current_rows) + "\n[/TABLE]"
        chunks.append(chunk_text)

    return chunks


def auto_chunk(text: str, content_type: str = "text") -> List[str]:
    """
    Choose the right strategy based on content type and size.

    Parameters
    ----------
    text         : raw text
    content_type : 'table', 'text', 'code', etc.
    """
    if content_type == "table":
        return chunk_table(text)
    if _token_count(text) > 2000:
        return semantic_chunk(text)
    return fixed_chunk(text)
