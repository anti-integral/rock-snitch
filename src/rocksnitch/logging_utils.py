"""Structured console logging helpers used across the package."""
from __future__ import annotations

import logging
import os
import sys
from logging import Logger

from rich.logging import RichHandler


_CONFIGURED = False


def get_logger(name: str = "rocksnitch", level: str | int | None = None) -> Logger:
    """Return a configured Logger.

    Level resolves in order: explicit arg > ROCKSNITCH_LOG_LEVEL env > INFO.
    Attaches a rich handler exactly once.
    """
    global _CONFIGURED
    lvl = level if level is not None else os.environ.get("ROCKSNITCH_LOG_LEVEL", "INFO")
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(lvl)
        _CONFIGURED = True
    logger.setLevel(lvl)
    return logger


def ensure_console_logging() -> None:
    """Force sys.stdout to flush lines (useful in subprocess contexts)."""
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
