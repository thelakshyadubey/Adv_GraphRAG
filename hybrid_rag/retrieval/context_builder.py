"""
retrieval/context_builder.py — Assemble the final LLM context string from
retrieved results, graph paths, and reasoning steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from hybrid_rag.storage.schema import RetrievalResult


def build(
    results: List[RetrievalResult],
    graph_paths: Optional[List[Dict[str, Any]]] = None,
    reasoning_steps: Optional[List[str]] = None,
    query: str = "",
) -> str:
    """
    Assemble a structured context string for the final LLM call.

    Sections:
      RETRIEVED CONTEXT   — top text chunks with source attribution
      GRAPH RELATIONSHIPS — graph traversal paths (if any)
      REASONING STEPS     — intermediate reasoning (if any)
      QUESTION            — the original user query
    """
    sections: List[str] = []

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    if results:
        chunk_lines: List[str] = []
        seen_ids: set[str] = set()
        for r in results:
            if r.id in seen_ids:
                continue
            seen_ids.add(r.id)
            source_info = ""
            if r.source_doc:
                source_info = f"[Source: {r.source_doc}"
                if r.source_page:
                    source_info += f", page {r.source_page}"
                source_info += "]"
            chunk_lines.append(f"{source_info}\n{r.text.strip()}")
        sections.append("RETRIEVED CONTEXT:\n" + "\n\n".join(chunk_lines))

    # ── Graph relationships ───────────────────────────────────────────────────
    if graph_paths:
        path_lines: List[str] = []
        for path in graph_paths[:10]:
            if isinstance(path, dict):
                path_lines.append(str(path))
        if path_lines:
            sections.append("GRAPH RELATIONSHIPS:\n" + "\n".join(path_lines))

    # ── Reasoning steps ───────────────────────────────────────────────────────
    if reasoning_steps:
        steps_text = "\n".join(f"- {s}" for s in reasoning_steps)
        sections.append(f"REASONING STEPS:\n{steps_text}")

    # ── Question ──────────────────────────────────────────────────────────────
    if query:
        sections.append(f"QUESTION: {query}")

    sections.append(
        "Instructions: Using ONLY the context provided above, write a thorough and "
        "insightful answer. You are allowed to synthesize, infer, and reason across "
        "multiple pieces of context to form a cohesive answer — do not just quote "
        "verbatim. If the context contains relevant facts, use them to answer fully "
        "even if the exact wording of the question isn't present. "
        "Cite source document and page number inline when available (e.g. [Source: docname, page N]). "
        "Only say the information is unavailable if the context genuinely contains "
        "no facts relevant to the question."
    )

    return "\n\n".join(sections)
