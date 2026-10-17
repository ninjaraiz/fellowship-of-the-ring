"""Deferred imports, so that using one character does not pay for all of them.

Importing ``FotR`` used to cost around three seconds on a warm cache. Almost
none of it was our code: every module pulled ``torch``, ``seaborn``,
``scikit-learn``, ``pyvista`` and ``matplotlib`` at module level, whether or
not the caller ever touched them. Opening a CODA database paid for ``torch``;
listing a folder paid for ``seaborn``, which in turn drags in ``ipywidgets``
and ``IPython``.

:func:`lazy_module` returns a stand-in that imports the real module the first
time one of its attributes is used, so the cost falls on whoever actually
needs it. Call sites do not change::

    torch = lazy_module('torch')        # at the top of the module
    ...
    def f(x):
        return torch.as_tensor(x)       # torch is imported here, not before

Why not a module-level ``__getattr__`` (PEP 562)
------------------------------------------------
Because it does not work for this. PEP 562's ``__getattr__`` is consulted
only for attribute access *on the module object* (``mod.torch``). A bare
global name looked up inside a function body goes locals → module globals →
builtins and never reaches it, so the function raises ``NameError``. The
stand-in below works precisely because the name *is* present in the module
globals; it is the attribute access on it that triggers the import.

Limits
------
* ``from heavy import Name`` cannot be deferred this way — the name is bound
  at import time. Move those imports inside the function that needs them.
* Annotations that mention a deferred module (``x: torch.Tensor``) are
  evaluated when the ``def`` runs, which would defeat the whole thing. Add
  ``from __future__ import annotations`` to such modules so annotations stay
  strings.
* ``isinstance(x, torch.Tensor)`` is fine: the attribute access resolves to
  the real class before ``isinstance`` sees it.
"""

import importlib
import types
from collections.abc import MutableMapping

__all__ = ['lazy_module', 'is_loaded', 'LazyRegistry']


class _LazyModule(types.ModuleType):
    """A module stand-in that imports its target on first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._lazy_target = name
        self._lazy_real = None

    def _lazy_load(self):
        real = self._lazy_real
        if real is None:
            real = importlib.import_module(self._lazy_target)
            self._lazy_real = real
        return real

    def __getattr__(self, attr):
        # Guard against recursion while __init__ is still running, and keep
        # dunder probes from triggering a multi-second import.
        if attr.startswith('_lazy_') or attr in ('__all__', '__path__'):
            raise AttributeError(attr)
        return getattr(self._lazy_load(), attr)

    def __dir__(self):
        return dir(self._lazy_load())

    def __repr__(self) -> str:
        state = 'loaded' if self._lazy_real is not None else 'not loaded yet'
        return f"<lazy module {self._lazy_target!r} ({state})>"


def lazy_module(name: str) -> types.ModuleType:
    """Return a stand-in for *name*, imported on first attribute access.

    Parameters
    ----------
    name : str
        Dotted module path, exactly as you would write after ``import``.
        Submodules work: ``lazy_module('matplotlib.pyplot')``.

    Returns
    -------
    types.ModuleType
        An object that behaves like the module once anything is read from it.

    Examples
    --------
    ::

        torch = lazy_module('torch')
        plt = lazy_module('matplotlib.pyplot')
    """
    return _LazyModule(name)


def is_loaded(mod) -> bool:
    """Whether a stand-in has already imported its target.

    Useful in tests and when measuring import cost; a real module always
    counts as loaded.
    """
    if isinstance(mod, _LazyModule):
        return mod._lazy_real is not None
    return True


class LazyRegistry(MutableMapping):
    """A format → class registry that imports each class on first lookup.

    The four FRODO registries used to be plain dicts filled by a column of
    ``try: from .coda import CODASets ... except ModuleNotFoundError: pass``
    blocks at import time. That meant opening a CODA database also imported
    the HORSES3D, NUMPY, NUMPYFILE and PYLOM readers, and with them pyvista,
    plotly, meshio and pyLOM.

    This keeps the same observable behaviour while importing nothing until a
    format is actually asked for:

    * a value of ``None`` means "format known, not implemented" — the lookup
      returns ``None`` and FRODO warns, exactly as before;
    * a class whose module raises ``ModuleNotFoundError`` (an optional
      dependency is missing) is reported as ``None`` as well, which is what
      the old ``except ModuleNotFoundError: pass`` produced;
    * any other import error propagates, as it did before.

    Parameters
    ----------
    package : str
        Package the relative module paths are resolved against, normally
        ``__name__`` of the registry's ``__init__.py``.
    specs : dict
        ``{format: '.module:ClassName'}``, or ``{format: None}`` for a known
        format with no implementation.

    Notes
    -----
    Iterating yields every *declared* format, including ones whose optional
    dependency turns out to be missing; the old eager registries simply left
    those out. Once a lookup has proved a format unavailable, it is dropped
    from the iteration, so error messages settle on the truth after the first
    probe. Use :meth:`available` to force the check for all of them.
    """

    def __init__(self, package: str, specs: dict) -> None:
        self._package = package
        self._specs = dict(specs)
        self._cache: dict = {}
        self._missing: set = set()

    # ── resolution ──────────────────────────────────────────────────────
    def _resolve(self, key):
        if key in self._cache:
            return self._cache[key]
        spec = self._specs[key]
        if spec is None:
            obj = None
        else:
            mod_path, _, cls_name = spec.partition(':')
            try:
                module = importlib.import_module(mod_path, self._package)
                obj = getattr(module, cls_name)
            except ModuleNotFoundError:
                obj = None
                self._missing.add(key)
        self._cache[key] = obj
        return obj

    def available(self) -> list:
        """Formats that really resolve to a class, importing to find out."""
        return [k for k in self._specs if self._resolve(k) is not None]

    # ── mapping protocol ────────────────────────────────────────────────
    def __getitem__(self, key):
        if key not in self._specs:
            raise KeyError(key)
        return self._resolve(key)

    def __setitem__(self, key, value):
        self._specs[key] = value
        self._cache[key] = value
        self._missing.discard(key)

    def __delitem__(self, key):
        del self._specs[key]
        self._cache.pop(key, None)
        self._missing.discard(key)

    def __iter__(self):
        return (k for k in self._specs if k not in self._missing)

    def __len__(self):
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"LazyRegistry({sorted(self._specs)})"
