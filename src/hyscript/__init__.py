"""Shared generation core plus separately consumed offline evaluation tools."""

import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["agent", "artifacts", "config", "evaluation", "llm", "search", "trends"]
__version__ = "0.1.0"
