"""
storage/cache_manager.py — Two-level cache: L1 in-memory LRU + L2 Redis.

CAG context store: preloaded full-document text keyed by doc_id, so answering
questions about a doc doesn't require re-fetching chunks from Qdrant.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional

import structlog

from hybrid_rag.config import settings

logger = structlog.get_logger(__name__)

_L1_MAX = 1000          # Max entries in L1 LRU cache
_L2_TTL = 3600          # Redis TTL in seconds


class LRUCache:
    """Simple thread-safe LRU cache using OrderedDict."""

    def __init__(self, maxsize: int = _L1_MAX) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def keys(self):
        return list(self._cache.keys())


class CacheManager:
    def __init__(self) -> None:
        self._l1 = LRUCache(maxsize=_L1_MAX)
        self._redis: Optional[Any] = None          # redis.asyncio.Redis instance
        self._doc_contexts: dict[str, str] = {}    # doc_id -> full context string
        self._access_counts: dict[str, int] = {}

    async def connect(self) -> None:
        """Optionally connect to Redis. Gracefully degrades to L1-only if unavailable."""
        try:
            import redis.asyncio as aioredis  # type: ignore
            self._redis = await aioredis.from_url(
                settings.redis_url, decode_responses=True, socket_connect_timeout=2
            )
            await self._redis.ping()
            logger.info("redis_connected", url=settings.redis_url)
        except Exception as exc:
            logger.warning("redis_unavailable_l1_only", error=str(exc))
            self._redis = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    # ── Query hash ────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:32]

    @staticmethod
    def _doc_key(doc_id: str, query_hash: str) -> str:
        return f"cag:{doc_id}:{query_hash}"

    # ── Exact match cache ─────────────────────────────────────────────────────

    async def has_exact(self, query: str) -> bool:
        h = self._hash(query)
        if self._l1.get(h) is not None:
            return True
        if self._redis:
            try:
                return await self._redis.exists(h) == 1
            except Exception:
                pass
        return False

    async def get(self, query: str) -> Optional[str]:
        h = self._hash(query)
        # L1 first
        val = self._l1.get(h)
        if val is not None:
            return val
        # L2
        if self._redis:
            try:
                val = await self._redis.get(h)
                if val:
                    self._l1.set(h, val)
                    return val
            except Exception:
                pass
        return None

    async def set(self, query: str, answer: str, doc_id: Optional[str] = None) -> None:
        h = self._hash(query)
        self._l1.set(h, answer)
        if self._redis:
            try:
                await self._redis.setex(h, _L2_TTL, answer)
                # Also store under doc-scoped key for easy invalidation
                if doc_id:
                    await self._redis.setex(self._doc_key(doc_id, h), _L2_TTL, answer)
            except Exception as exc:
                logger.warning("redis_set_failed", error=str(exc))

    # ── CAG document context ──────────────────────────────────────────────────

    def get_doc_context(self, doc_id: str) -> Optional[str]:
        ctx = self._doc_contexts.get(doc_id)
        if ctx:
            return ctx
        # Try L1 cache with special key
        return self._l1.get(f"ctx:{doc_id}")

    def set_doc_context(self, doc_id: str, text: str) -> None:
        self._doc_contexts[doc_id] = text
        self._l1.set(f"ctx:{doc_id}", text)
        logger.info("doc_context_cached", doc_id=doc_id, chars=len(text))

    async def invalidate_doc(self, doc_id: str) -> None:
        """Remove all cached answers and the preloaded context for a document."""
        # Clear from in-memory doc context store
        self._doc_contexts.pop(doc_id, None)
        self._l1.delete(f"ctx:{doc_id}")

        # Remove any L1 entries that look like doc-scoped keys
        for key in self._l1.keys():
            if doc_id in key:
                self._l1.delete(key)

        # Remove Redis keys matching the pattern
        if self._redis:
            try:
                pattern = f"cag:{doc_id}:*"
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as exc:
                logger.warning("redis_invalidate_failed", doc_id=doc_id, error=str(exc))

        logger.info("cache_invalidated", doc_id=doc_id)

    # ── Access counting (for CAG promotion) ───────────────────────────────────

    def increment_access(self, doc_id: str) -> int:
        count = self._access_counts.get(doc_id, 0) + 1
        self._access_counts[doc_id] = count
        return count

    def get_access_count(self, doc_id: str) -> int:
        return self._access_counts.get(doc_id, 0)


# Singleton
cache_manager = CacheManager()
