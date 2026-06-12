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

from .base import BaseSets   # always available

# ── Formats with no Sets implementation ──────────────────────────────────────
# Explicit None entries make the registry self-documenting.
SETS_REGISTRY: dict = {
    'Airfoil': None,   # Airfoil format has no Sets class
}

# ── Progressive imports ───────────────────────────────────────────────────────
try:
    from .coda import CODASets
    SETS_REGISTRY['CODA'] = CODASets
except ModuleNotFoundError:
    pass

try:
    from .numpy_file import NUMPYFILESets
    SETS_REGISTRY['NUMPYFILE'] = NUMPYFILESets
except ModuleNotFoundError:
    pass

try:
    from .pylom import PYLOMSets
    SETS_REGISTRY['PYLOM'] = PYLOMSets
except ModuleNotFoundError:
    pass

__all__ = ['BaseSets', 'SETS_REGISTRY']