"""
retrieval/rrf_merger.py — Reciprocal Rank Fusion for combining result lists.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from hybrid_rag.config import settings
from hybrid_rag.storage.schema import RetrievalResult


def rrf_merge(
    result_lists: List[List[RetrievalResult]],
    k: int | None = None,
) -> List[RetrievalResult]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score(d) = Σ  1 / (k + rank(d, list_i))

    Parameters
    ----------
    result_lists : one or more ranked lists of RetrievalResult
    k            : ranking constant (default: settings.rrf_k = 60)

    Returns
    -------
    Single merged and re-ranked list of RetrievalResult.
    """
    if k is None:
        k = settings.rrf_k

    scores: dict[str, float] = defaultdict(float)
    items: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            scores[item.id] += 1.0 / (k + rank + 1)
            # Keep the item with the highest original score if seen multiple times
            if item.id not in items or item.score > items[item.id].score:
                items[item.id] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Attach merged RRF score to result
    merged: List[RetrievalResult] = []
    for uid, rrf_score in ranked:
        result = items[uid].model_copy()
        result.score = rrf_score
        merged.append(result)

    return merged
