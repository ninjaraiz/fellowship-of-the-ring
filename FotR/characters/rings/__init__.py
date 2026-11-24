"""
rings/__init__.py
=================
Registry of the GANDALF case-generation rings, mirroring the FRODO
reader/sets/residuals/stats registries.

* ``'coda'`` — one shared mesh for every case (:class:`CODARing`).
* ``'coda_single'`` — one mesh per case (:class:`CODASingleRing`).
* ``'horses3d'`` — HORSES3D, one shared mesh and a ``p`` ladder
  (:class:`Horses3DRing`).
"""

from ..._lazy import LazyRegistry
from .base import BaseRing   # always available

#: Each ring class is imported the first time its name is looked up.
RING_REGISTRY = LazyRegistry(__name__, {
    'coda': '.coda:CODARing',
    'coda_single': '.coda_single:CODASingleRing',
    'horses3d': '.horses3d:Horses3DRing',
})


_CLASS_TO_RING = {
    'CODARing': 'coda',
    'CODASingleRing': 'coda_single',
    'Horses3DRing': 'horses3d',
}


def __getattr__(name):
    """Expose the ring classes without importing them eagerly."""
    if name in _CLASS_TO_RING:
        obj = RING_REGISTRY[_CLASS_TO_RING[name]]
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['BaseRing', 'CODARing', 'CODASingleRing', 'Horses3DRing',
           'RING_REGISTRY']
