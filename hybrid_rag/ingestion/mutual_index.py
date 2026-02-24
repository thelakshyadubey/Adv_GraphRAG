"""
ingestion/mutual_index.py — Bidirectional chunk_id <-> node_id linking.

build():   writes each entity's chunk_id into Neo4j AND the chunk's node_ids into Qdrant.
expand():  from a set of chunk_ids → expands to include all related chunks via shared nodes.
"""
from __future__ import annotations

from typing import List

import structlog

from hybrid_rag.storage.schema import Chunk, Entity

logger = structlog.get_logger(__name__)


async def build(chunk: Chunk, entities: List[Entity]) -> None:
    """
    For each entity extracted from this chunk:
      - register chunk_id on the Neo4j node
      - append node_id to chunk.node_ids
    Then upsert the updated chunk into Qdrant.
    """
    from hybrid_rag.storage.neo4j_client import neo4j_client  # local import avoids circular
    from hybrid_rag.storage.qdrant_client import qdrant_client

    for entity in entities:
        try:
            await neo4j_client.add_chunk_to_node(entity.node_id, chunk.chunk_id)
            if entity.node_id not in chunk.node_ids:
                chunk.node_ids.append(entity.node_id)
        except Exception as exc:
            logger.warning(
                "mutual_index_build_entity_failed",
                node_id=entity.node_id,
                error=str(exc),
            )

    try:
        await qdrant_client.upsert_chunk(chunk)
    except Exception as exc:
        logger.error("mutual_index_upsert_chunk_failed", chunk_id=chunk.chunk_id, error=str(exc))
        raise


async def expand(chunk_ids: List[str]) -> List[str]:
    """
    Graph-driven chunk expansion:
      1. Collect all node_ids referenced by the given chunks (from Qdrant payload).
      2. For each node, collect all chunk_ids registered in Neo4j.
      3. Return the deduplicated additional chunk_ids (originals excluded).
    """
    from hybrid_rag.storage.neo4j_client import neo4j_client
    from hybrid_rag.storage.qdrant_client import qdrant_client

    if not chunk_ids:
        return []

    original_set = set(chunk_ids)
    all_node_ids: set[str] = set()

    # Step 1: get node_ids from Qdrant payloads
    for cid in chunk_ids:
        try:
            # scroll for exact chunk_id match
            from qdrant_client.http import models as qm  # type: ignore
            flt = qm.Filter(must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=cid))])
            records, _ = await qdrant_client._client.scroll(
                collection_name="chunks",
                scroll_filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                if rec.payload:
                    for nid in rec.payload.get("node_ids", []):
                        all_node_ids.add(nid)
        except Exception as exc:
            logger.warning("expand_get_node_ids_failed", chunk_id=cid, error=str(exc))

    # Step 2: get chunk_ids from those nodes in Neo4j
    expanded: set[str] = set()
    for node_id in all_node_ids:
        try:
            related_chunk_ids = await neo4j_client.get_chunks_for_node(node_id)
            for rc in related_chunk_ids:
                if rc not in original_set:
                    expanded.add(rc)
        except Exception as exc:
            logger.warning("expand_get_chunks_failed", node_id=node_id, error=str(exc))

    return list(expanded)
