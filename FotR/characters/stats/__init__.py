"""
stats/__init__.py
==================
Registry of all available FRODO Stats classes.

A value of None means the format is known but has no Stats
implementation (FRODO will print a warning and set db.stats = None).
A missing key is treated identically to None.

Adding a new Stats class
-------------------------
1. Create ``stats/<format>.py`` subclassing ``BaseStats``.
2. The import block below picks it up automatically on next restart.
"""

from .base import BaseStats   # always available

# ── Formats with no Stats implementation ──────────────────────────────────
STATS_REGISTRY: dict = {}

# ── Progressive imports ───────────────────────────────────────────────────────
try:
    from .coda import CODAStats
    STATS_REGISTRY['CODA'] = CODAStats
    STATS_REGISTRY['NUMPY'] = CODAStats # Use the same data_dict format
except ModuleNotFoundError:
    pass

__all__ = ['BaseStats', 'STATS_REGISTRY']