"""
sets/__init__.py
================
Registry of all available FRODO Sets classes.

A value of None means the format is known but has no Sets implementation
(FRODO will print a warning and set db.sets = None).
A missing key is treated identically to None.

Adding a new Sets class
-----------------------
1. Create ``sets/<format>.py`` subclassing ``BaseSets``.
2. The import block below picks it up automatically on next restart.
"""

from ..._lazy import LazyRegistry
from .base import BaseSets   # always available, and cheap

# CODA_SINGLE keeps its own Sets: one mesh per case is a different data
# contract. CODASets is shared by CODA and NUMPY, which use the same
# data_dict layout.
#: Declared formats. Each class is imported the first time its format is
#: looked up, so using one format does not import the others. ``None`` means
#: "format known, no implementation"; a missing optional dependency resolves
#: to ``None`` too, exactly as the old try/except blocks did.
SETS_REGISTRY = LazyRegistry(__name__, {
    'Airfoil': None,
    'CODA': '.coda:CODASets',
    'NUMPY': '.coda:CODASets',
    'CODA_SINGLE': '.coda_single:CODASingleSets',
    'NUMPYFILE': '.numpy_file:NUMPYFILESets',
    'PYLOM': '.pylom:PYLOMSets',
})

__all__ = ['BaseSets', 'SETS_REGISTRY']
