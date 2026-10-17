"""
rings/__init__.py
=================
Registry of the GANDALF case-generation rings, mirroring the FRODO
reader/sets/residuals/stats registries.

* ``'coda'`` — one shared mesh for every case (:class:`CODARing`).
* ``'coda_single'`` — one mesh per case (:class:`CODASingleRing`).
"""

from ..._lazy import LazyRegistry
from .base import BaseRing   # always available

#: Each ring class is imported the first time its name is looked up.
RING_REGISTRY = LazyRegistry(__name__, {
    'coda': '.coda:CODARing',
    'coda_single': '.coda_single:CODASingleRing',
})


def __getattr__(name):
    """Expose CODARing / CODASingleRing without importing them eagerly."""
    if name in ('CODARing', 'CODASingleRing'):
        obj = RING_REGISTRY['coda' if name == 'CODARing' else 'coda_single']
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['BaseRing', 'CODARing', 'CODASingleRing', 'RING_REGISTRY']
