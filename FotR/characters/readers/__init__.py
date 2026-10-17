"""
readers/__init__.py
===================
Registry of all available FRODO reader classes.

Each reader is imported inside a try/except block so the package works
progressively: a format becomes available as soon as its module file is
created, without touching this file.

Adding a new reader
-------------------
1. Create ``readers/<format>.py`` subclassing ``BaseReader``.
2. The import block below picks it up automatically on next restart.
   No other changes needed anywhere.

READER_REGISTRY
---------------
Maps a format string ('CODA', 'Airfoil', …) to the concrete reader class.
A missing key means the format is unknown; FRODO raises ValueError.
A value of None is not used here – every registered entry must be a class.
"""

from ..._lazy import LazyRegistry
from .base import BaseReader   # always available, and cheap

#: Declared formats. Each class is imported the first time its format is
#: looked up, so using one format does not import the others. ``None`` means
#: "format known, no implementation"; a missing optional dependency resolves
#: to ``None`` too, exactly as the old try/except blocks did.
READER_REGISTRY = LazyRegistry(__name__, {
    'CODA': '.coda:CODAReader',
    'CODA_SINGLE': '.coda_single:CODASingleReader',
    'NUMPY': '.numpy:NUMPYReader',
    'NUMPYFILE': '.numpy_file:NUMPYFILEReader',
    'PYLOM': '.pylom:PYLOMReader',
    'HORSES3D': '.horses3d:Horses3DReader',
})

__all__ = ['BaseReader', 'READER_REGISTRY']
