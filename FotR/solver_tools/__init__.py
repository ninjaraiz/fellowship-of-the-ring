"""Tools tied to one solver: XFoil, Gmsh, HORSES3D binary files and CODA
run logs (``coda_log``, and ``coda_monitor``, the live web monitor).

Loaded lazily, like the rest of the package (``FotR/_lazy.py``):
``FotR.solver_tools.horses3d`` has to import with numpy alone, e.g. from
ParaView's ``pvpython`` on a cluster, and ``coda_log``/``coda_monitor``
with the standard library alone, so this file must not import its
siblings (``xfoil_tools`` pulls pandas) until one of their names is used.
"""

from typing import TYPE_CHECKING

#: Public name → module that defines it.
_EXPORTS = {
    'xfoil_parser': '.xfoil_tools',
    'XFoilCase':    '.xfoil_tools',
    'GmshCase':     '.gmsh_tools',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Import the module that defines *name* on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    obj = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from .xfoil_tools import xfoil_parser, XFoilCase
    from .gmsh_tools import GmshCase
