"""
storage/cache_manager.py — Two-level cache: L1 in-memory LRU + L2 Redis.

Exact cache  : SHA-256 hash of normalised query → answer.
Semantic cache: embedding of query stored alongside answer. On every new query
                the stored vectors are compared via cosine similarity. Queries
                above settings.semantic_cache_threshold are treated as the same
                question even if worded differently.

  "what is the revenue"  ≈  "what's the revenue"  → same cache hit (sim ≈ 0.97)
  "what is the revenue"  ≈  "tell me the costs"    → miss          (sim ≈ 0.41)

CAG context store: preloaded full-document text keyed by doc_id.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np
import structlog

from hybrid_rag.config import settings

logger = structlog.get_logger(__name__)

_L1_MAX = settings.l1_cache_max   # Max entries in L1 LRU cache
_L2_TTL = settings.cache_ttl      # Redis TTL in seconds


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


_REDIS_RETRY_INTERVAL = 60  # seconds between reconnect attempts


class CacheManager:
    def __init__(self) -> None:
        self._l1 = LRUCache(maxsize=_L1_MAX)
        self._redis: Optional[Any] = None          # redis.asyncio.Redis instance
        self._doc_contexts: dict[str, str] = {}    # doc_id -> full context string
        self._access_counts: dict[str, int] = {}
        # Semantic index: list of (unit_vector, query_hash) kept in insertion order.
        # Used to find near-duplicate queries without exact string match.
        self._sem_index: List[Tuple[np.ndarray, str]] = []
        self._last_redis_attempt: float = 0.0      # epoch timestamp of last connect attempt

    async def _try_connect_redis(self) -> None:
        """Attempt to connect to Redis. Called at startup and lazily retried every
        _REDIS_RETRY_INTERVAL seconds so the app self-heals without a restart."""
        now = time.monotonic()
        if now - self._last_redis_attempt < _REDIS_RETRY_INTERVAL:
            return  # too soon to retry
        self._last_redis_attempt = now
        try:
            import redis.asyncio as aioredis  # type: ignore
            client = await aioredis.from_url(
                settings.redis_url, decode_responses=True, socket_connect_timeout=3
            )
            await client.ping()
            self._redis = client
            logger.info("redis_connected", url=settings.redis_url)
        except Exception as exc:
            if self._redis is None:
                # Only log on first failure or after recovering → avoids log spam
                logger.warning("redis_unavailable_l1_only", error=str(exc))
            self._redis = None

    async def connect(self) -> None:
        """Called at app startup. Failures are non-fatal — L1 cache still works."""
        await self._try_connect_redis()

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

    # ── Semantic helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        """Embed a single string and return a unit vector."""
        from hybrid_rag.ingestion.embedder import embed as _embed_fn
        vec = np.array(_embed_fn(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def _sem_find(self, q_vec: np.ndarray) -> Optional[str]:
        """
        Walk the in-memory semantic index and return the cached query hash
        whose stored vector is most similar to q_vec, if similarity >=
        settings.semantic_cache_threshold.  Returns None on miss.
        """
        if not self._sem_index:
            return None
        threshold = settings.semantic_cache_threshold
        best_score = -1.0
        best_hash: Optional[str] = None
        for stored_vec, stored_hash in self._sem_index:
            sim = float(np.dot(q_vec, stored_vec))
            if sim > best_score:
                best_score = sim
                best_hash = stored_hash
        if best_score >= threshold:
            logger.info("semantic_cache_hit", similarity=round(best_score, 4))
            return best_hash
        return None

    def _sem_add(self, query: str, query_hash: str) -> None:
        """Embed the query and add it to the in-memory semantic index."""
        try:
            vec = self._embed(query)
            self._sem_index.append((vec, query_hash))
            # Cap the index at _L1_MAX entries (drop oldest)
            if len(self._sem_index) > _L1_MAX:
                self._sem_index.pop(0)
        except Exception as exc:
            logger.warning("sem_index_add_failed", error=str(exc))

    # ── Cache lookup (exact + semantic) ──────────────────────────────────────

    async def has_exact(self, query: str) -> bool:
        """
        Returns True if either:
          • the exact normalised query has been seen before, OR
          • a semantically equivalent query (cosine sim >= threshold) is cached.
        """
        if self._redis is None:
            await self._try_connect_redis()

        h = self._hash(query)

        # 1. Exact match — L1
        if self._l1.get(h) is not None:
            return True

        # 2. Exact match — L2 Redis
        if self._redis:
            try:
                if await self._redis.exists(h) == 1:
                    return True
            except Exception:
                self._redis = None  # will reconnect on next call

        # 3. Semantic match — in-memory index
        try:
            q_vec = self._embed(query)
            if self._sem_find(q_vec) is not None:
                return True
        except Exception:
            pass

        return False

    async def get(self, query: str) -> Optional[str]:
        """
        Return cached answer for query.  Checks exact hash first,
        then falls back to semantic similarity lookup.
        """
        if self._redis is None:
            await self._try_connect_redis()

        h = self._hash(query)

        # 1. Exact — L1
        val = self._l1.get(h)
        if val is not None:
            return val

        # 2. Exact — L2 Redis
        if self._redis:
            try:
                val = await self._redis.get(h)
                if val:
                    self._l1.set(h, val)        # promote to L1
                    return val
            except Exception:
                self._redis = None  # will reconnect on next call

        # 3. Semantic — in-memory index → resolve hash → fetch answer
        try:
            q_vec = self._embed(query)
            matched_hash = self._sem_find(q_vec)
            if matched_hash:
                # Fetch the answer stored under the matched hash
                val = self._l1.get(matched_hash)
                if val is not None:
                    return val
                if self._redis:
                    try:
                        val = await self._redis.get(matched_hash)
                        if val:
                            self._l1.set(h, val)    # cache under new hash too
                            return val
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("semantic_get_failed", error=str(exc))

        return None

    async def set(self, query: str, answer: str, doc_id: Optional[str] = None) -> None:
        if self._redis is None:
            await self._try_connect_redis()

        h = self._hash(query)
        self._l1.set(h, answer)

        # Add to semantic index so future paraphrases hit this answer
        self._sem_add(query, h)

        if self._redis:
            try:
                await self._redis.setex(h, _L2_TTL, answer)
                if doc_id:
                    await self._redis.setex(self._doc_key(doc_id, h), _L2_TTL, answer)
            except Exception as exc:
                logger.warning("redis_set_failed", error=str(exc))
                self._redis = None  # will reconnect on next call

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
