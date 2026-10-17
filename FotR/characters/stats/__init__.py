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

from ..._lazy import LazyRegistry
from .base import BaseStats   # always available, and cheap

#: Declared formats. Each class is imported the first time its format is
#: looked up, so using one format does not import the others. ``None`` means
#: "format known, no implementation"; a missing optional dependency resolves
#: to ``None`` too, exactly as the old try/except blocks did.
STATS_REGISTRY = LazyRegistry(__name__, {
    'CODA': '.coda:CODAStats',
    'NUMPY': '.coda:CODAStats',
    'CODA_SINGLE': '.coda_single:CODASingleStats',
})

__all__ = ['BaseStats', 'STATS_REGISTRY']
