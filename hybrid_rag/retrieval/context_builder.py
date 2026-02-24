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
    # Separate community summaries from regular chunks (treated differently in context)
    community_results = [r for r in results if r.result_type == "community"]
    chunk_results = [r for r in results if r.result_type != "community"]

    # Community overview section (put first — sets the thematic frame)
    if community_results:
        comm_lines: List[str] = []
        seen_ids: set[str] = set()
        for r in community_results:
            if r.id in seen_ids:
                continue
            seen_ids.add(r.id)
            source_info = f"[Community summary, doc: {r.source_doc}]" if r.source_doc else "[Community summary]"
            comm_lines.append(f"{source_info}\n{r.text.strip()}")
        sections.append("COMMUNITY OVERVIEW (high-level themes and entity clusters):\n" + "\n\n".join(comm_lines))

    if chunk_results:
        chunk_lines: List[str] = []
        seen_ids2: set[str] = set()
        for r in chunk_results:
            if r.id in seen_ids2:
                continue
            seen_ids2.add(r.id)
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
        "INSTRUCTIONS:\n"
        "Answer the QUESTION above using the retrieved context. Rules:\n"
        "1. Give a direct, specific, declarative answer — lead with the answer itself, not a preamble.\n"
        "2. Synthesize and reason across multiple context chunks; draw connections even when implicit.\n"
        "3. NEVER use these phrases (they are forbidden):\n"
        "   - 'the context does not explicitly'\n"
        "   - 'although the context'\n"
        "   - 'the context does not mention'\n"
        "   - 'based on the context'\n"
        "   - 'the context says'\n"
        "   - 'not explicitly stated'\n"
        "   - 'while not explicitly'\n"
        "4. State facts directly as facts, not as 'the document states that…'\n"
        "5. Cite source inline as [doc: name, page N] when available.\n"
        "6. If and only if the context contains zero relevant facts, write: 'Insufficient context to answer.'"
    )

    return "\n\n".join(sections)
