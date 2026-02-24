"""
retrieval/kag_planner.py — LLM-based query decomposition into a QueryPlan.

The plan specifies which operators to run in which order, with dependency tracking.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import structlog

from hybrid_rag.config import settings
from hybrid_rag.storage.schema import OperatorType, PlanStep, QueryPlan

logger = structlog.get_logger(__name__)

PLANNER_PROMPT = """\
Decompose the following query into a minimal set of reasoning steps.
Available operators:
  GRAPH_EXACT   - search for named entities in a knowledge graph
  VECTOR_SEARCH - semantic vector similarity search over document chunks
  HYBRID        - combined graph + vector search (best general-purpose)
  MULTI_HOP     - multi-hop graph traversal for relationship chains
  SUMMARY       - search over document and entity summaries
  LLM_REASON    - use an LLM to reason over retrieved context (final step)

Rules:
- Use the fewest steps necessary.
- LLM_REASON should be the LAST step if included, and should have depends_on pointing to prior steps.
- For simple queries, a single HYBRID step is sufficient.
- Return ONLY valid JSON, no additional text.

Format:
{{
  "steps": [
    {{"step_id": 1, "operator": "OPERATOR_NAME", "sub_query": "...", "depends_on": []}},
    {{"step_id": 2, "operator": "LLM_REASON", "sub_query": "...", "depends_on": [1]}}
  ]
}}

Query: {query}
"""

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)


def _call_groq(prompt: str) -> str:
    client = _groq_client()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Groq planner call failed: {last_exc}")


def _safe_fallback(query: str) -> QueryPlan:
    """Return a safe single-step HYBRID fallback plan."""
    return QueryPlan(
        original_query=query,
        steps=[PlanStep(step_id=1, operator=OperatorType.HYBRID, sub_query=query, depends_on=[])],
    )


def _validate_no_circular(steps: List[PlanStep]) -> bool:
    step_ids = {s.step_id for s in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                return False
            if dep >= step.step_id:
                return False  # Only allow backward dependencies
    return True


def plan(query: str) -> QueryPlan:
    """
    Decompose query into a QueryPlan.
    Falls back to single HYBRID step on any parse/validation failure.
    """
    prompt = PLANNER_PROMPT.format(query=query)
    raw = ""
    try:
        raw = _call_groq(prompt)
        data = json.loads(raw)
        raw_steps: List[Dict] = data.get("steps", [])

        steps: List[PlanStep] = []
        for s in raw_steps:
            op_str = str(s.get("operator", "HYBRID")).upper()
            # Validate operator name
            try:
                op = OperatorType(op_str)
            except ValueError:
                op = OperatorType.HYBRID

            steps.append(PlanStep(
                step_id=int(s.get("step_id", len(steps) + 1)),
                operator=op,
                sub_query=str(s.get("sub_query", query)),
                depends_on=[int(d) for d in s.get("depends_on", [])],
            ))

        if not steps:
            raise ValueError("Empty steps list from planner")

        if not _validate_no_circular(steps):
            logger.warning("circular_dep_in_plan_fallback")
            return _safe_fallback(query)

        logger.info("plan_created", steps=len(steps))
        return QueryPlan(original_query=query, steps=steps)

    except Exception as exc:
        logger.warning("planner_fallback", error=str(exc), raw=raw[:200])
        return _safe_fallback(query)
