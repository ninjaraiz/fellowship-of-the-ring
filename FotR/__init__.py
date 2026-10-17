"""The Fellowship of the Ring: CFD database tooling.

Importing this package is deliberately cheap. Each character is loaded the
first time you name it, so ``from FotR import FRODO`` does not drag in
LEGOLAS, GIMLI or ARAGORN, and none of them import ``torch``, ``seaborn``
or ``pyvista`` until a method that needs those is actually called. See
``FotR/_lazy.py`` for the mechanism and its limits.

Nothing changes at the call site::

    from FotR import FRODO, SAM          # works as before
    import FotR; FotR.GANDALF(...)       # works as before
"""

from typing import TYPE_CHECKING

#: Public name → module that defines it.
_EXPORTS = {
    'FRODO':          '.characters.frodo',
    'SAM':            '.characters.sam',
    'GANDALF':        '.characters.gandalf',
    'LEGOLAS':        '.characters.legolas',
    'ARAGORN':        '.characters.aragorn',
    'RunResult':      '.characters.aragorn',
    'GIMLI':          '.characters.gimli',
    'EarendilsLight': '.earendils_light',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Import the module that defines *name* on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    obj = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = obj          # subsequent lookups skip this function
    return obj


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:   # so that editors and type checkers still resolve them
    from .characters.frodo import FRODO
    from .characters.sam import SAM
    from .characters.gandalf import GANDALF
    from .characters.legolas import LEGOLAS
    from .characters.aragorn import ARAGORN, RunResult
    from .characters.gimli import GIMLI
    from .earendils_light import EarendilsLight
