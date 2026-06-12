"""
residuals/__init__.py
=====================
Registry of all available FRODO Residuals classes.

A value of None means the format is known but has no Residuals
implementation (FRODO will print a warning and set db.residuals = None).

Adding a new Residuals class
----------------------------
1. Create ``residuals/<format>.py`` subclassing ``BaseResiduals``.
2. The import block below picks it up automatically on next restart.
"""

from .base import BaseResiduals   # always available

# ── Formats with no Residuals implementation ──────────────────────────────────
RESIDUALS_REGISTRY: dict = {
    'Airfoil':   None,
    'NUMPYFILE': None,
    'PYLOM':     None,
}

# ── Progressive imports ───────────────────────────────────────────────────────
try:
    from .coda import CODAResiduals
    RESIDUALS_REGISTRY['CODA'] = CODAResiduals
except ModuleNotFoundError:
    pass

__all__ = ['BaseResiduals', 'RESIDUALS_REGISTRY']