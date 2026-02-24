"""
ingestion/extractor.py — Schema-free entity + relation extraction via Groq LLM.

Entity and relation types are inferred dynamically from the text — no fixed schema.
Entities are deduplicated and assigned stable deterministic UUIDs.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Dict, List, Tuple

import structlog

from hybrid_rag.config import settings
from hybrid_rag.storage.schema import Entity, Relation

logger = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """\
Extract all entities and relationships from the text below.
Infer entity types from context (e.g. Person, Company, Product, Event, Concept, Location, Date, Policy — or any other type that fits).
Normalize all names to title case.
Return ONLY valid JSON with no additional text:
{{
  "entities": [{{"name": "", "type": "", "properties": {{}}}}],
  "relations": [{{"source": "", "target": "", "type": "", "properties": {{}}}}]
}}
Text: {text}
"""

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


def _stable_id(*parts: str) -> str:
    """Deterministic UUID5 from concatenated string parts."""
    key = "|".join(p.strip().lower() for p in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _call_groq(prompt: str) -> str:
    """Call Groq with retry + exponential backoff. Returns raw content string."""
    client = _groq_client()
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning("groq_retry", attempt=attempt, wait=wait, error=str(exc))
            time.sleep(wait)

    raise RuntimeError(f"Groq call failed after {_MAX_RETRIES} attempts: {last_exc}")


def extract(text: str) -> Tuple[List[Entity], List[Relation]]:
    """
    Extract entities and relations from a text chunk.

    Returns
    -------
    (entities, relations) — deduplicated, with stable IDs.
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:6000])  # guard token budget

    raw: str = ""
    try:
        raw = _call_groq(prompt)
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Retry once with an explicit instruction to fix JSON
        try:
            fix_prompt = f"Fix this broken JSON and return ONLY valid JSON:\n{raw}"
            raw = _call_groq(fix_prompt)
            data = json.loads(raw)
        except Exception as exc:
            logger.error("extraction_json_parse_failed", error=str(exc), raw=raw[:200])
            return [], []
    except Exception as exc:
        logger.error("extraction_failed", error=str(exc))
        return [], []

    # ── Build entity objects ──────────────────────────────────────────────────

    raw_entities: List[Dict] = data.get("entities") or []
    seen_entity_keys: set[str] = set()
    entity_name_to_id: Dict[str, str] = {}
    entities: List[Entity] = []

    for e in raw_entities:
        name = str(e.get("name", "")).strip().title()
        etype = str(e.get("type", "Unknown")).strip().title()
        if not name:
            continue
        key = f"{name}|{etype}"
        if key in seen_entity_keys:
            continue
        seen_entity_keys.add(key)

        node_id = _stable_id(name, etype)
        entity_name_to_id[name.lower()] = node_id

        entities.append(Entity(
            node_id=node_id,
            entity_type=etype,
            name=name,
            properties=e.get("properties") or {},
            chunk_ids=[],
        ))

    # ── Build relation objects ────────────────────────────────────────────────

    raw_relations: List[Dict] = data.get("relations") or []
    relations: List[Relation] = []

    for r in raw_relations:
        src_name = str(r.get("source", "")).strip().title()
        tgt_name = str(r.get("target", "")).strip().title()
        rel_type = str(r.get("type", "RELATED_TO")).strip().upper().replace(" ", "_")

        src_id = entity_name_to_id.get(src_name.lower())
        tgt_id = entity_name_to_id.get(tgt_name.lower())

        if not src_id or not tgt_id:
            continue  # skip relations whose entities weren't extracted

        rel_id = _stable_id(src_id, tgt_id, rel_type)
        relations.append(Relation(
            rel_id=rel_id,
            source_id=src_id,
            target_id=tgt_id,
            rel_type=rel_type,
            properties=r.get("properties") or {},
        ))

    logger.info(
        "extraction_done",
        entities=len(entities),
        relations=len(relations),
    )
    return entities, relations
