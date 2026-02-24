"""
ingestion/community_builder.py — GraphRAG-style community detection and summarisation.

Algorithm (pure Python, no heavy graph library required):
  1. Register every extracted entity as a node.
  2. Union-Find: merge nodes connected by a relation into the same community.
  3. Skip singleton communities (single isolated entity).
  4. For each multi-entity community: call Groq to generate a thematic paragraph
     that describes key entities, their relationships, and the overarching theme.
  5. Embed the summary and return as List[Summary] with summary_type=COMMUNITY.

These summaries are stored in Qdrant's "summaries" collection at ingestion time
and enable global / thematic query answering at retrieval time — the core
innovation of the original Microsoft GraphRAG paper.

  "Local queries" → retrieve specific chunks (KAG / KAG_SIMPLE path)
  "Global queries" → retrieve community summaries first (GLOBAL path)
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set

import structlog

from hybrid_rag.config import settings
from hybrid_rag.ingestion import embedder
from hybrid_rag.storage.schema import Entity, Relation, Summary, SummaryType

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_MIN_COMMUNITY_SIZE = 2   # communities with only 1 entity are skipped


# ── Union-Find (disjoint-set) ─────────────────────────────────────────────────

class _UnionFind:
    """Path-compressed union-find for community detection."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])   # path compression
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        self._parent[self.find(a)] = self.find(b)

    def communities(self) -> Dict[str, Set[str]]:
        """Return {root_id: set_of_member_ids}."""
        groups: Dict[str, Set[str]] = defaultdict(set)
        for node in self._parent:
            groups[self.find(node)].add(node)
        return dict(groups)


# ── LLM summarisation ─────────────────────────────────────────────────────────

_COMMUNITY_PROMPT = """\
You are summarising a community of related entities detected in a knowledge graph.

Entities in this community:
{entity_descriptions}

Known relationships between them:
{relation_descriptions}

Write a concise paragraph (3-5 sentences) that:
1. Names the key entities and their types.
2. Describes how they are connected or related.
3. Identifies the overarching theme or topic this community represents.

Community Summary:"""

_SYSTEM_PROMPT = (
    "You are an expert knowledge-graph analyst. Summarise communities of related "
    "entities clearly and concisely, highlighting their connections and shared themes."
)


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


def _llm_summarise(entity_descriptions: str, relation_descriptions: str) -> str:
    prompt = _COMMUNITY_PROMPT.format(
        entity_descriptions=entity_descriptions,
        relation_descriptions=relation_descriptions,
    )
    client = _groq_client()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Community summarisation failed after {_MAX_RETRIES} retries: {last_exc}")


# ── Main public function ──────────────────────────────────────────────────────

def build_communities(
    doc_id: str,
    entities: List[Entity],
    relations: List[Relation],
) -> List[Summary]:
    """
    Detect entity communities from the extracted graph and generate a
    thematic LLM summary for each community.

    Parameters
    ----------
    doc_id    : document being ingested (used as parent_id on each Summary)
    entities  : all entities extracted from the document
    relations : all relations extracted from the document

    Returns
    -------
    List[Summary] with summary_type=COMMUNITY, embedding filled in,
    ready to be stored via qdrant_client.batch_upsert("summaries", ...).
    """
    if not entities:
        logger.info("community_build_skipped", doc_id=doc_id, reason="no_entities")
        return []

    entity_map: Dict[str, Entity] = {e.node_id: e for e in entities}

    # ── Step 1: build graph with Union-Find ───────────────────────────────────
    uf = _UnionFind()
    for e in entities:
        uf.find(e.node_id)                    # register all nodes
    for rel in relations:
        if rel.source_id in entity_map and rel.target_id in entity_map:
            uf.union(rel.source_id, rel.target_id)

    raw_communities = uf.communities()

    # ── Step 2: summarise each qualifying community ───────────────────────────
    summaries: List[Summary] = []
    community_idx = 0

    for _root, member_ids in raw_communities.items():
        member_entities = [entity_map[nid] for nid in member_ids if nid in entity_map]

        if len(member_entities) < _MIN_COMMUNITY_SIZE:
            continue   # skip singletons

        community_idx += 1

        # Entity description block
        entity_lines = "\n".join(
            f"  - {e.name} ({e.entity_type})" for e in member_entities
        )

        # Intra-community relations
        comm_ids: Set[str] = {e.node_id for e in member_entities}
        intra_rels = [
            r for r in relations
            if r.source_id in comm_ids and r.target_id in comm_ids
        ]
        if intra_rels:
            rel_lines = "\n".join(
                f"  - {entity_map[r.source_id].name if r.source_id in entity_map else r.source_id}"
                f" --[{r.rel_type}]--> "
                f"{entity_map[r.target_id].name if r.target_id in entity_map else r.target_id}"
                for r in intra_rels
            )
        else:
            rel_lines = "  (co-occur in the same document context)"

        try:
            text = _llm_summarise(entity_lines, rel_lines)
            entity_names = [e.name for e in member_entities]
            vec = embedder.embed(text)

            summary = Summary(
                summary_id=str(uuid.uuid4()),
                summary_type=SummaryType.COMMUNITY,
                parent_id=doc_id,
                text=text,
                embedding=vec,
                entity_names=entity_names,
                community_id=community_idx,
            )
            summaries.append(summary)

            logger.info(
                "community_summary_generated",
                doc_id=doc_id,
                community_id=community_idx,
                members=len(member_entities),
                entity_names=entity_names,
            )
        except Exception as exc:
            logger.warning(
                "community_summary_failed",
                doc_id=doc_id,
                community_id=community_idx,
                error=str(exc),
            )

    logger.info("communities_built", doc_id=doc_id, total_communities=len(summaries))
    return summaries
