"""
ingestion/summarizer.py — LLM-based doc, section, and entity summarization.

All summaries are embedded and stored in Qdrant's 'summaries' collection.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Dict, List

import structlog

from hybrid_rag.config import settings
from hybrid_rag.ingestion import embedder
from hybrid_rag.storage.schema import Chunk, Entity, Summary, SummaryType

logger = structlog.get_logger(__name__)

_SUMMARY_PROMPT = "Summarize the following in 3-5 sentences:\n\n{text}"
_ENTITY_SUMMARY_PROMPT = (
    "Based on these excerpts about {entity_name}, write a 2-3 sentence summary:\n\n{chunks_text}"
)
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


def _llm_summarize(prompt: str) -> str:
    client = _groq_client()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFF_BASE ** attempt)
    logger.error("summarize_failed", error=str(last_exc))
    return ""


def _make_summary(summary_type: SummaryType, parent_id: str, text: str) -> Summary:
    summary_id = str(uuid.uuid4())
    emb = embedder.embed(text)
    return Summary(
        summary_id=summary_id,
        summary_type=summary_type,
        parent_id=parent_id,
        text=text,
        embedding=emb,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_doc_summary(chunks: List[Chunk]) -> Summary:
    """Generate a single document-level summary from all chunks."""
    combined = " ".join(c.text for c in chunks)[:8000]  # cap input
    prompt = _SUMMARY_PROMPT.format(text=combined)
    text = _llm_summarize(prompt)
    doc_id = chunks[0].doc_id if chunks else "unknown"
    logger.info("doc_summary_generated", doc_id=doc_id)
    return _make_summary(SummaryType.DOC, doc_id, text or "No summary available.")


def generate_section_summaries(chunks_by_section: Dict[str, List[Chunk]]) -> List[Summary]:
    """Generate per-section summaries."""
    summaries: List[Summary] = []
    for section_name, chunks in chunks_by_section.items():
        if not chunks:
            continue
        combined = " ".join(c.text for c in chunks)[:4000]
        prompt = _SUMMARY_PROMPT.format(text=combined)
        text = _llm_summarize(prompt)
        summaries.append(_make_summary(SummaryType.SECTION, section_name, text or combined[:300]))
    logger.info("section_summaries_generated", count=len(summaries))
    return summaries


def generate_entity_summary(entity: Entity, chunks: List[Chunk]) -> Summary:
    """Generate a summary for a single entity from its associated chunks."""
    chunks_text = "\n---\n".join(c.text for c in chunks[:5])[:4000]
    prompt = _ENTITY_SUMMARY_PROMPT.format(
        entity_name=entity.name, chunks_text=chunks_text
    )
    text = _llm_summarize(prompt)
    return _make_summary(SummaryType.ENTITY, entity.node_id, text or entity.name)


def generate_all(
    chunks: List[Chunk],
    entities: List[Entity],
) -> List[Summary]:
    """
    Convenience function called by the pipeline.
    Generates doc summary + section summaries + entity summaries.
    """
    summaries: List[Summary] = []

    # Doc-level
    if chunks:
        summaries.append(generate_doc_summary(chunks))

    # Section-level
    sections: Dict[str, List[Chunk]] = defaultdict(list)
    for c in chunks:
        key = c.section or "default"
        sections[key].append(c)
    summaries.extend(generate_section_summaries(dict(sections)))

    # Entity-level: build a lookup of chunk by chunk_id
    chunk_map = {c.chunk_id: c for c in chunks}
    for entity in entities[:20]:  # cap to avoid runaway LLM calls
        entity_chunks = [chunk_map[cid] for cid in entity.chunk_ids if cid in chunk_map]
        if entity_chunks:
            summaries.append(generate_entity_summary(entity, entity_chunks))

    logger.info("all_summaries_generated", total=len(summaries))
    return summaries
