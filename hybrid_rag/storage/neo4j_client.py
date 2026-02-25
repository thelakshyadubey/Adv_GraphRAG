"""
storage/neo4j_client.py — Neo4j async driver wrapper with CRUD + traversal.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from hybrid_rag.config import settings
from hybrid_rag.storage.schema import Entity, Relation

logger = structlog.get_logger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            # AuraDB drops idle TCP connections after ~30 min.
            # Set max_connection_lifetime well below that so the driver
            # proactively recycles connections before the server closes them,
            # avoiding the ConnectionResetError(10054) on first request after idle.
            max_connection_lifetime=180,      # recycle connections every 3 min
            max_connection_pool_size=10,
            connection_timeout=10,
            keep_alive=True,
        )
        await self._driver.verify_connectivity()
        await self.create_indexes()
        logger.info("neo4j_connected", uri=settings.neo4j_uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    # ── Index setup ───────────────────────────────────────────────────────────

    async def create_indexes(self) -> None:
        async with self._driver.session() as session:
            # Standard property index
            try:
                await session.run(
                    "CREATE INDEX entity_node_id IF NOT EXISTS FOR (e:Entity) ON (e.node_id)"
                )
            except Exception as exc:
                logger.warning("index_creation_skipped", index="entity_node_id", error=str(exc))

            # Fulltext index — drop and recreate if it somehow ended up in a bad state
            try:
                await session.run(
                    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]"
                )
                logger.info("fulltext_index_ensured", index="entity_name_ft")
            except Exception as exc:
                # Some Neo4j versions reject IF NOT EXISTS for fulltext — try without it
                logger.warning("fulltext_index_create_failed", error=str(exc))
                try:
                    await session.run(
                        "CREATE FULLTEXT INDEX entity_name_ft FOR (e:Entity) ON EACH [e.name]"
                    )
                    logger.info("fulltext_index_created_fallback", index="entity_name_ft")
                except Exception as exc2:
                    logger.warning("fulltext_index_already_exists_or_failed", error=str(exc2))

            # Wait for all indexes to come online (important for AuraDB / cloud)
            try:
                await session.run("CALL db.awaitIndexes(120)")
            except Exception:
                pass  # older Neo4j versions may not support this

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create_entity(self, entity: Entity) -> None:
        query = """
        MERGE (e:Entity {node_id: $node_id})
        SET   e.entity_type = $entity_type,
              e.name        = $name,
              e.properties  = $properties,
              e.chunk_ids   = $chunk_ids
        """
        try:
            async with self._driver.session() as session:
                await session.run(
                    query,
                    node_id=entity.node_id,
                    entity_type=entity.entity_type,
                    name=entity.name,
                    properties=str(entity.properties),
                    chunk_ids=entity.chunk_ids,
                )
        except Exception as exc:
            logger.error("create_entity_failed", node_id=entity.node_id, error=str(exc))
            raise

    async def create_entities(self, entities: List[Entity]) -> None:
        for entity in entities:
            await self.create_entity(entity)

    async def create_relation(self, rel: Relation) -> None:
        query = """
        MATCH (src:Entity {node_id: $source_id})
        MATCH (tgt:Entity {node_id: $target_id})
        MERGE (src)-[r:RELATES_TO {rel_id: $rel_id, rel_type: $rel_type}]->(tgt)
        SET   r.properties = $properties
        """
        try:
            async with self._driver.session() as session:
                await session.run(
                    query,
                    rel_id=rel.rel_id,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    rel_type=rel.rel_type,
                    properties=str(rel.properties),
                )
        except Exception as exc:
            logger.error("create_relation_failed", rel_id=rel.rel_id, error=str(exc))
            raise

    async def create_relations(self, relations: List[Relation]) -> None:
        for rel in relations:
            await self.create_relation(rel)

    async def add_chunk_to_node(self, node_id: str, chunk_id: str) -> None:
        query = """
        MATCH (e:Entity {node_id: $node_id})
        SET   e.chunk_ids = CASE
                WHEN $chunk_id IN coalesce(e.chunk_ids, []) THEN e.chunk_ids
                ELSE coalesce(e.chunk_ids, []) + [$chunk_id]
              END
        """
        try:
            async with self._driver.session() as session:
                await session.run(query, node_id=node_id, chunk_id=chunk_id)
        except Exception as exc:
            logger.error("add_chunk_to_node_failed", node_id=node_id, error=str(exc))
            raise

    async def get_entity(self, node_id: str) -> Optional[Entity]:
        query = "MATCH (e:Entity {node_id: $node_id}) RETURN e"
        try:
            async with self._driver.session() as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                if record is None:
                    return None
                node = record["e"]
                return Entity(
                    node_id=node["node_id"],
                    entity_type=node.get("entity_type", ""),
                    name=node.get("name", ""),
                    properties={},
                    chunk_ids=node.get("chunk_ids", []),
                )
        except Exception as exc:
            logger.error("get_entity_failed", node_id=node_id, error=str(exc))
            return None

    # ── Search ────────────────────────────────────────────────────────────────

    async def search_by_name(self, name: str, fuzzy: bool = True) -> List[Entity]:
        if fuzzy:
            query = """
            CALL db.index.fulltext.queryNodes('entity_name_ft', $name)
            YIELD node, score
            RETURN node
            ORDER BY score DESC
            LIMIT 10
            """
        else:
            # Substring match — used as fallback when fulltext index unavailable
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e AS node
            LIMIT 10
            """
        try:
            async with self._driver.session() as session:
                result = await session.run(query, name=name)
                records = await result.data()
                entities = []
                for r in records:
                    n = r["node"]
                    entities.append(Entity(
                        node_id=n["node_id"],
                        entity_type=n.get("entity_type", ""),
                        name=n.get("name", ""),
                        chunk_ids=n.get("chunk_ids", []),
                    ))
                return entities
        except Exception as exc:
            err_str = str(exc)
            # Fulltext index missing — fall back to case-insensitive CONTAINS search
            if fuzzy and "entity_name_ft" in err_str:
                logger.warning(
                    "fulltext_index_missing_fallback",
                    detail="entity_name_ft not found, using CONTAINS match",
                )
                return await self.search_by_name(name, fuzzy=False)
            logger.error("search_by_name_failed", name=name, error=err_str)
            return []

    # ── Traversal ─────────────────────────────────────────────────────────────

    async def traverse(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        query = """
        MATCH path = (start:Entity {node_id: $node_id})-[*1..$depth]-(related)
        RETURN path LIMIT 50
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(query, node_id=node_id, depth=depth)
                records = await result.data()
                return {"paths": records, "start": node_id}
        except Exception as exc:
            logger.error("traverse_failed", node_id=node_id, error=str(exc))
            return {}

    async def multi_hop(self, start_id: str, depth: int = 3) -> List[Dict[str, Any]]:
        query = """
        MATCH path = (start:Entity {node_id: $start_id})-[*1..$depth]->(end:Entity)
        RETURN DISTINCT end.node_id AS node_id,
                        end.name    AS name,
                        end.entity_type AS entity_type,
                        end.chunk_ids   AS chunk_ids,
                        length(path)    AS hop_count
        ORDER BY hop_count
        LIMIT 30
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(query, start_id=start_id, depth=depth)
                return await result.data()
        except Exception as exc:
            logger.error("multi_hop_failed", start_id=start_id, error=str(exc))
            return []

    async def get_chunks_for_node(self, node_id: str) -> List[str]:
        entity = await self.get_entity(node_id)
        if entity:
            return entity.chunk_ids
        return []

    async def run_cypher(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        params = params or {}
        try:
            async with self._driver.session() as session:
                result = await session.run(query, **params)
                return await result.data()
        except Exception as exc:
            logger.error("run_cypher_failed", error=str(exc))
            return []


# Singleton
neo4j_client = Neo4jClient()
