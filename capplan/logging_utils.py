"""Logging setup shared by the CLI and the scheduled batch."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once. Later calls only adjust the level."""
    global _CONFIGURED
    resolved = level or os.environ.get("CAPPLAN_LOG_LEVEL", "INFO")
    if not _CONFIGURED:
        logging.basicConfig(
            level=resolved,
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(resolved)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
