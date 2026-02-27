"""
storage/qdrant_client.py — Qdrant vector DB wrapper. Creates 3 collections on startup.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from hybrid_rag.config import settings
from hybrid_rag.storage.schema import Chunk, Entity, RetrievalResult, Summary

logger = structlog.get_logger(__name__)

# Collection names are configured in settings so they can be changed per
# deployment without touching code (e.g. staging vs production namespacing).
def _collections() -> list[str]:
    return [
        settings.qdrant_chunks_collection,
        settings.qdrant_entities_collection,
        settings.qdrant_summaries_collection,
    ]


class QdrantVectorClient:
    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        # Strip any accidental scheme prefix from the host value
        host = settings.qdrant_host.removeprefix("https://").removeprefix("http://")

        if settings.qdrant_api_key:
            # Cloud: build URL without port if port is 443 (standard HTTPS)
            if settings.qdrant_port == 443:
                url = f"https://{host}"
            else:
                url = f"https://{host}:{settings.qdrant_port}"
            self._client = AsyncQdrantClient(
                url=url,
                api_key=settings.qdrant_api_key,
            )
            logger.info("qdrant_connected_cloud", url=url)
        else:
            # Local: plain host + port, no TLS
            self._client = AsyncQdrantClient(
                host=host,
                port=settings.qdrant_port,
            )
            logger.info("qdrant_connected_local", host=host)

        await self._ensure_collections()

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    # ── Collection setup ──────────────────────────────────────────────────────

    async def _ensure_collections(self) -> None:
        existing = {c.name for c in (await self._client.get_collections()).collections}
        for name in _collections():
            if name not in existing:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=qdrant_models.VectorParams(
                        size=settings.embedding_dim,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logger.info("collection_created", name=name)
        # Ensure payload indexes exist (required by Qdrant Cloud for filtered scrolls)
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes needed for filtered queries."""
        _indexes = {
            settings.qdrant_chunks_collection:    [("doc_id", qdrant_models.PayloadSchemaType.KEYWORD),
                                                   ("chunk_id", qdrant_models.PayloadSchemaType.KEYWORD)],
            settings.qdrant_entities_collection:  [("node_id", qdrant_models.PayloadSchemaType.KEYWORD),
                                                   ("entity_type", qdrant_models.PayloadSchemaType.KEYWORD)],
            settings.qdrant_summaries_collection: [("parent_id", qdrant_models.PayloadSchemaType.KEYWORD),
                                                   ("summary_type", qdrant_models.PayloadSchemaType.KEYWORD)],
        }
        for collection, fields in _indexes.items():
            for field, schema in fields:
                try:
                    await self._client.create_payload_index(
                        collection_name=collection,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception:
                    pass  # index already exists — safe to ignore

    # ── Upsert helpers ────────────────────────────────────────────────────────

    async def upsert_chunk(self, chunk: Chunk) -> None:
        if chunk.embedding is None:
            logger.warning("upsert_chunk_no_embedding", chunk_id=chunk.chunk_id)
            return
        point = qdrant_models.PointStruct(
            id=self._to_qdrant_id(chunk.chunk_id),
            vector=chunk.embedding,
            payload={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type.value,
                "section": chunk.section,
                "page": chunk.page,
                "node_ids": chunk.node_ids,
                "position": chunk.position,
            },
        )
        await self._client.upsert(collection_name=settings.qdrant_chunks_collection, points=[point])

    async def upsert_entity(self, entity: Entity) -> None:
        if entity.embedding is None:
            return
        point = qdrant_models.PointStruct(
            id=self._to_qdrant_id(entity.node_id),
            vector=entity.embedding,
            payload={
                "node_id": entity.node_id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "chunk_ids": entity.chunk_ids,
            },
        )
        await self._client.upsert(collection_name=settings.qdrant_entities_collection, points=[point])

    async def upsert_summary(self, summary: Summary) -> None:
        if summary.embedding is None:
            return
        payload: dict = {
            "summary_id": summary.summary_id,
            "summary_type": summary.summary_type.value,
            "parent_id": summary.parent_id,
            "text": summary.text,
        }
        if summary.entity_names:
            payload["entity_names"] = summary.entity_names
        if summary.community_id is not None:
            payload["community_id"] = summary.community_id
        point = qdrant_models.PointStruct(
            id=self._to_qdrant_id(summary.summary_id),
            vector=summary.embedding,
            payload=payload,
        )
        await self._client.upsert(collection_name=settings.qdrant_summaries_collection, points=[point])

    async def batch_upsert(self, collection: str, items: List[Any]) -> None:
        """Batch upsert for Chunk, Entity, or Summary objects."""
        if not items:
            return
        points: List[qdrant_models.PointStruct] = []
        for item in items:
            if isinstance(item, Chunk):
                if item.embedding is None:
                    continue
                points.append(qdrant_models.PointStruct(
                    id=self._to_qdrant_id(item.chunk_id),
                    vector=item.embedding,
                    payload={
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "text": item.text,
                        "chunk_type": item.chunk_type.value,
                        "section": item.section,
                        "page": item.page,
                        "node_ids": item.node_ids,
                        "position": item.position,
                    },
                ))
            elif isinstance(item, Entity):
                if item.embedding is None:
                    continue
                points.append(qdrant_models.PointStruct(
                    id=self._to_qdrant_id(item.node_id),
                    vector=item.embedding,
                    payload={
                        "node_id": item.node_id,
                        "entity_type": item.entity_type,
                        "name": item.name,
                        "chunk_ids": item.chunk_ids,
                    },
                ))
            elif isinstance(item, Summary):
                if item.embedding is None:
                    continue
                payload: dict = {
                    "summary_id": item.summary_id,
                    "summary_type": item.summary_type.value,
                    "parent_id": item.parent_id,
                    "text": item.text,
                }
                if item.entity_names:
                    payload["entity_names"] = item.entity_names
                if item.community_id is not None:
                    payload["community_id"] = item.community_id
                points.append(qdrant_models.PointStruct(
                    id=self._to_qdrant_id(item.summary_id),
                    vector=item.embedding,
                    payload=payload,
                ))

        if points:
            await self._client.upsert(collection_name=collection, points=points)

    # ── Search ────────────────────────────────────────────────────────────────

    async def search_chunks(
        self,
        query_vec: List[float],
        filters: Optional[Dict[str, Any]] = None,
        doc_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        if doc_ids:
            must: list = [qdrant_models.FieldCondition(
                key="doc_id",
                match=qdrant_models.MatchAny(any=doc_ids),
            )]
            if filters:
                for k, v in filters.items():
                    must.append(qdrant_models.FieldCondition(
                        key=k, match=qdrant_models.MatchValue(value=v)
                    ))
            qdrant_filter = qdrant_models.Filter(must=must)
        else:
            qdrant_filter = self._build_filter(filters) if filters else None
        try:
            result = await self._client.query_points(
                collection_name=settings.qdrant_chunks_collection,
                query=query_vec,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
            return [self._hit_to_result(h) for h in result.points]
        except Exception as exc:
            logger.error("search_chunks_failed", error=str(exc))
            return []

    async def search_communities(
        self,
        query_vec: List[float],
        doc_ids: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[RetrievalResult]:
        """Search community summaries in Qdrant for global/thematic queries."""
        must_conditions = [
            qdrant_models.FieldCondition(
                key="summary_type",
                match=qdrant_models.MatchValue(value="community"),
            )
        ]
        if doc_ids:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="parent_id",
                    match=qdrant_models.MatchAny(any=doc_ids),
                )
            )
        qdrant_filter = qdrant_models.Filter(must=must_conditions)
        try:
            result = await self._client.query_points(
                collection_name=settings.qdrant_summaries_collection,
                query=query_vec,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
            results = []
            for h in result.points:
                p = h.payload or {}
                results.append(RetrievalResult(
                    id=p.get("summary_id", str(h.id)),
                    text=p.get("text", ""),
                    score=h.score,
                    source_doc=p.get("parent_id"),
                    result_type="community",
                    node_ids=p.get("entity_names", []),
                ))
            return results
        except Exception as exc:
            logger.error("search_communities_failed", error=str(exc))
            return []

    async def search_summaries(
        self,
        query_vec: List[float],
        summary_type: Optional[str] = None,
        limit: int = 3,
    ) -> List[RetrievalResult]:
        filters = {"summary_type": summary_type} if summary_type else None
        qdrant_filter = self._build_filter(filters) if filters else None
        try:
            result = await self._client.query_points(
                collection_name=settings.qdrant_summaries_collection,
                query=query_vec,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
            return [self._hit_to_result(h) for h in result.points]
        except Exception as exc:
            logger.error("search_summaries_failed", error=str(exc))
            return []

    async def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict]:
        """Fetch all chunks belonging to a document (for CAG context preload)."""
        try:
            qdrant_filter = qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="doc_id",
                    match=qdrant_models.MatchValue(value=doc_id),
                )]
            )
            records, _ = await self._client.scroll(
                collection_name=settings.qdrant_chunks_collection,
                scroll_filter=qdrant_filter,
                limit=5000,
                with_payload=True,
                with_vectors=False,
            )
            return [r.payload for r in records if r.payload]
        except Exception as exc:
            logger.error("get_all_chunks_failed", doc_id=doc_id, error=str(exc))
            return []

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_qdrant_id(uid: str) -> str:
        """Qdrant accepts UUID strings natively — return the ID as-is.

        All chunk_id / node_id / summary_id values are already uuid4 strings,
        so no conversion is needed.  The old approach (MD5 → integer) could
        produce values > 2^63 which Qdrant rejects with a 400 Bad Request.
        """
        return uid

    @staticmethod
    def _hit_to_result(hit: Any) -> RetrievalResult:
        p = hit.payload or {}
        return RetrievalResult(
            id=p.get("chunk_id") or p.get("summary_id") or p.get("node_id", str(hit.id)),
            text=p.get("text", ""),
            score=hit.score,
            source_doc=p.get("doc_id") or p.get("parent_id"),
            source_page=p.get("page"),
            node_ids=p.get("node_ids", []),
        )

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> qdrant_models.Filter:
        conditions = [
            qdrant_models.FieldCondition(
                key=k, match=qdrant_models.MatchValue(value=v)
            )
            for k, v in filters.items()
        ]
        return qdrant_models.Filter(must=conditions)


# Singleton
qdrant_client = QdrantVectorClient()
