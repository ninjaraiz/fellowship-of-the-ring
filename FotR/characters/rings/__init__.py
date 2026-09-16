"""
rings/__init__.py
=================
Registry of the GANDALF case-generation rings, mirroring the FRODO
reader/sets/residuals/stats registries.

* ``'coda'`` — one shared mesh for every case (:class:`CODARing`).
* ``'coda_single'`` — one mesh per case (:class:`CODASingleRing`).
"""

from .base import BaseRing
from .coda import CODARing
from .coda_single import CODASingleRing

RING_REGISTRY: dict = {
    'coda': CODARing,
    'coda_single': CODASingleRing,
}

__all__ = ['BaseRing', 'CODARing', 'CODASingleRing', 'RING_REGISTRY']
