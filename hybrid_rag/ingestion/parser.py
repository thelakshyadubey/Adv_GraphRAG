"""
ingestion/parser.py — Parse any supported document type to text + metadata.

Supported: .pdf  .docx  .txt  .md  .html  .csv
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ParsedDocument:
    text: str
    pages: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse(file_path: str) -> ParsedDocument:
    """Dispatch to the correct parser based on file extension."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".txt": _parse_text,
        ".md": _parse_text,
        ".html": _parse_html,
        ".htm": _parse_html,
        ".csv": _parse_csv,
    }

    parser_fn = parsers.get(ext, _parse_text)
    logger.info("parsing_file", file=str(path), ext=ext)

    try:
        result = parser_fn(str(path))
        result.metadata["file_name"] = path.name
        result.metadata["file_ext"] = ext
        result.metadata["file_size"] = path.stat().st_size
        return result
    except Exception as exc:
        logger.error("parse_failed", file=str(path), error=str(exc))
        raise


# ── PDF ───────────────────────────────────────────────────────────────────────

def _parse_pdf(file_path: str) -> ParsedDocument:
    """
    Fast PDF parser using PyMuPDF (fitz).
    Typically 10-50x faster than pdfplumber for large documents.
    """
    try:
        import fitz  # type: ignore  (PyMuPDF)
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF parsing: pip install pymupdf")

    pages: List[str] = []
    tables: List[str] = []

    doc = fitz.open(file_path)
    try:
        for page in doc:
            # "text" mode: fastest plain-text extraction (pure C, no layout analysis)
            page_text: str = page.get_text("text") or ""

            # Lightweight table detection via block geometry (no ML, very fast)
            for block in page.get_text("blocks"):
                # block = (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 = text, 1 = image — skip images
                if block[6] != 0:
                    continue
                blk_text = block[4].strip()
                lines = blk_text.splitlines()
                # Treat as a table if it has 2+ lines where most lines are multi-column
                if len(lines) >= 2 and sum(len(l.split()) >= 3 for l in lines) >= 2:
                    tables.append(blk_text)

            pages.append(page_text)
    finally:
        doc.close()

    full_text = "\n\n".join(pages)
    return ParsedDocument(
        text=full_text,
        pages=pages,
        tables=tables,
        metadata={"page_count": len(pages)},
    )


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _parse_docx(file_path: str) -> ParsedDocument:
    try:
        from docx import Document  # type: ignore
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")

    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    tables: List[str] = []

    for table in doc.tables:
        rows = []
        for row in table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        tables.append("\n".join(rows))

    full_text = "\n\n".join(paragraphs)
    if tables:
        full_text += "\n\n" + "\n\n[TABLE]\n".join(tables)

    return ParsedDocument(
        text=full_text,
        pages=[full_text],
        tables=tables,
        metadata={"paragraph_count": len(paragraphs)},
    )


# ── Plain text / Markdown ─────────────────────────────────────────────────────

def _parse_text(file_path: str) -> ParsedDocument:
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return ParsedDocument(text=text, pages=[text], metadata={})


# ── HTML ──────────────────────────────────────────────────────────────────────

def _parse_html(file_path: str) -> ParsedDocument:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    with open(file_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    soup = BeautifulSoup(raw, "html.parser")
    # Remove scripts and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)

    return ParsedDocument(text=text, pages=[text], metadata={})


# ── CSV ───────────────────────────────────────────────────────────────────────

def _parse_csv(file_path: str) -> ParsedDocument:
    rows: List[List[str]] = []
    with open(file_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        return ParsedDocument(text="", metadata={})

    header = rows[0]
    table_lines = [" | ".join(header)]
    for row in rows[1:]:
        table_lines.append(" | ".join(row))
    table_text = "\n".join(table_lines)

    return ParsedDocument(
        text=table_text,
        pages=[table_text],
        tables=[table_text],
        metadata={"row_count": len(rows) - 1, "column_count": len(header)},
    )
