"""
logging_config.py — Configure structlog + stdlib logging for the entire app.

Call setup_logging() once at startup (done in main.py).
All modules that do  `structlog.get_logger(__name__)`  will automatically
emit coloured, timestamped lines to stdout/stderr once this runs.
"""
from __future__ import annotations

import logging
import sys


def setup_logging(log_level: str = "DEBUG") -> None:
    """
    Wire structlog into Python's standard logging system so every
    `logger.info(...)` / `logger.warning(...)` call appears in the terminal.

    Output format  (human-readable, coloured):
        2026-02-24 12:34:56 [info     ] message   key=value  module=x
    """
    import structlog

    level = getattr(logging, log_level.upper(), logging.DEBUG)

    # ── 1. Configure Python stdlib logging (for third-party libs) ────────────
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "neo4j", "watchfiles",
                  "multipart", "asyncio", "sentence_transformers",
                  "huggingface_hub", "transformers", "filelock",
                  "torch", "tqdm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── 2. Configure structlog (PrintLoggerFactory for direct stdout) ─────────
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
