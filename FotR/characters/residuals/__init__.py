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

from ..._lazy import LazyRegistry
from .base import BaseResiduals   # always available, and cheap

# CODA_SINGLE reuses the CODA residuals: monitor files and metadata
# layout are identical, only the mesh differs per case.
#: Declared formats. Each class is imported the first time its format is
#: looked up, so using one format does not import the others. ``None`` means
#: "format known, no implementation"; a missing optional dependency resolves
#: to ``None`` too, exactly as the old try/except blocks did.
RESIDUALS_REGISTRY = LazyRegistry(__name__, {
    'Airfoil': None,
    'NUMPYFILE': None,
    'PYLOM': None,
    'CODA': '.coda:CODAResiduals',
    'CODA_SINGLE': '.coda:CODAResiduals',
    'HORSES3D': '.horses3d:Horses3DResiduals',
})

__all__ = ['BaseResiduals', 'RESIDUALS_REGISTRY']
