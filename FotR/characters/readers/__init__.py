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

from .base import BaseReader   # always available

# ── Progressive imports ───────────────────────────────────────────────────────
# Only ModuleNotFoundError is caught so that real import errors inside the
# individual modules (syntax errors, missing dependencies, …) still propagate.

READER_REGISTRY: dict = {}

try:
    from .coda import CODAReader
    READER_REGISTRY['CODA'] = CODAReader
except ModuleNotFoundError:
    pass

try:
    from .airfoil import AIRFOILReader
    READER_REGISTRY['Airfoil'] = AIRFOILReader
except ModuleNotFoundError:
    pass

try:
    from .numpy_file import NUMPYFILEReader
    READER_REGISTRY['NUMPYFILE'] = NUMPYFILEReader
except ModuleNotFoundError:
    pass

try:
    from .numpy import NUMPYReader
    READER_REGISTRY['NUMPY'] = NUMPYReader
except ModuleNotFoundError:
    pass

try:
    from .pylom import PYLOMReader
    READER_REGISTRY['PYLOM'] = PYLOMReader
except ModuleNotFoundError:
    pass

__all__ = ['BaseReader', 'READER_REGISTRY']