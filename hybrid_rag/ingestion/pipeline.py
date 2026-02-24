"""
ingestion/pipeline.py — Full document ingestion orchestrator.

Stages:
  1. Parse file → raw text + metadata
  2. Auto-chunk text
  3. Batch embed chunks
  4. Extract entities + relations per chunk (via Groq)
  5. Store entities/relations in Neo4j
  6. Build mutual index (Neo4j ↔ Qdrant chunk linkage)
  7. Generate and store summaries
  8. Auto-preload CAG context if doc is small enough
"""
from __future__ import annotations

import uuid
from typing import Dict, List

import structlog

from hybrid_rag.config import settings
from hybrid_rag.ingestion import chunker, embedder, extractor, mutual_index, summarizer
from hybrid_rag.ingestion.parser import parse
from hybrid_rag.storage.schema import Chunk, ChunkType, Entity, Relation

logger = structlog.get_logger(__name__)


async def ingest(file_path: str, doc_id: str) -> Dict:
    """
    Ingest a document end-to-end.

    Parameters
    ----------
    file_path : absolute path to the document file
    doc_id    : stable identifier for this document (caller-supplied)

    Returns
    -------
    dict with ingestion statistics
    """
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client

    logger.info("ingestion_started", doc_id=doc_id, file=file_path)

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    parsed = parse(file_path)
    logger.info("parsed", doc_id=doc_id, chars=len(parsed.text))

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    content_type = "table" if file_path.endswith(".csv") else "text"
    raw_chunks = chunker.auto_chunk(parsed.text, content_type)

    # For multi-page docs also chunk each page separately to preserve page metadata
    all_chunks: List[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunk_id = str(uuid.uuid4())
        # Try to assign page number from parsed.pages overlap
        page_num: int | None = None
        for p_idx, page_text in enumerate(parsed.pages):
            if text[:100] in page_text:
                page_num = p_idx + 1
                break

        ctype = ChunkType.TABLE if "[TABLE]" in text else ChunkType.TEXT
        all_chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            chunk_type=ctype,
            section=parsed.metadata.get("section"),
            page=page_num,
            position=idx,
        ))

    logger.info("chunks_created", doc_id=doc_id, count=len(all_chunks))

    if not all_chunks:
        logger.warning("no_chunks", doc_id=doc_id)
        return {"doc_id": doc_id, "chunks": 0, "entities": 0}

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    texts = [c.text for c in all_chunks]
    embeddings = embedder.batch_embed(texts)
    for i, chunk in enumerate(all_chunks):
        chunk.embedding = embeddings[i]

    # ── 4-5-6. Extract, store, link ───────────────────────────────────────────
    all_entities: List[Entity] = []
    all_relations: List[Relation] = []
    seen_entity_ids: set[str] = set()

    for chunk in all_chunks:
        try:
            entities, relations = extractor.extract(chunk.text)
        except Exception as exc:
            logger.warning("extraction_skipped", chunk_id=chunk.chunk_id, error=str(exc))
            entities, relations = [], []

        # Store entities + relations in Neo4j
        for entity in entities:
            try:
                await neo4j_client.create_entity(entity)
                if entity.node_id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity.node_id)
            except Exception as exc:
                logger.warning("entity_store_failed", error=str(exc))

        for rel in relations:
            try:
                await neo4j_client.create_relation(rel)
                all_relations.append(rel)
            except Exception as exc:
                logger.warning("relation_store_failed", error=str(exc))
                all_relations.append(rel)  # keep for community detection even if Neo4j is down

        # Mutual index: link chunk ↔ entities + upsert chunk to Qdrant
        try:
            await mutual_index.build(chunk, entities)
        except Exception as exc:
            logger.warning("mutual_index_failed", chunk_id=chunk.chunk_id, error=str(exc))

    # ── 7. Summaries ──────────────────────────────────────────────────────────
    summaries = summarizer.generate_all(all_chunks, all_entities)
    if summaries:
        try:
            await qdrant_client.batch_upsert("summaries", summaries)
        except Exception as exc:
            logger.warning("summaries_upsert_failed", error=str(exc))

    # ── 7b. Community detection + summarisation (GraphRAG) ────────────────────
    community_summaries: list = []
    if all_entities:
        try:
            from hybrid_rag.ingestion.community_builder import build_communities
            community_summaries = build_communities(doc_id, all_entities, all_relations)
            if community_summaries:
                await qdrant_client.batch_upsert("summaries", community_summaries)
                logger.info(
                    "community_summaries_stored",
                    doc_id=doc_id,
                    count=len(community_summaries),
                )
        except Exception as exc:
            logger.warning("community_build_failed", doc_id=doc_id, error=str(exc))

    # ── 8. Auto-CAG preload for small docs ────────────────────────────────────
    total_words = sum(len(c.text.split()) for c in all_chunks)
    if total_words < settings.cag_context_limit:
        try:
            from hybrid_rag.retrieval.cag_engine import cag_engine
            await cag_engine.preload(doc_id)
            logger.info("cag_preloaded", doc_id=doc_id)
        except Exception as exc:
            logger.warning("cag_preload_failed", doc_id=doc_id, error=str(exc))

    stats = {
        "doc_id": doc_id,
        "chunks": len(all_chunks),
        "entities": len(all_entities),
        "summaries": len(summaries),
        "communities": len(community_summaries),
        "words": total_words,
    }
    logger.info("ingestion_complete", **stats)
    return stats
