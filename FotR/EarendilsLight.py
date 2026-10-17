"""Eärendil's Light: the one question ``help()`` cannot answer.

Galadriel's phial was for dark places, when all other lights go out. Most
places are not dark: for almost anything in this package the built-in
``help()`` is better than anything written here could be — it is free, it
is never out of date, and it works on any object. Use it.

This module exists for the two corners where it is not enough.

Delegation
----------
A :class:`~FotR.characters.frodo.FRODO` forwards most calls to its
format quartet, and two of them — ``extract_inputs`` and
``extract_outputs`` — are re-declared on FRODO as ``(*args, **kwargs)``.
``help(db.extract_inputs)`` therefore shows an empty signature and
FRODO's generic prose, while the signature you must actually call lives
on the reader your format resolved to. Nothing static can tell you which
one that is, because it depends on the object in your hands::

    db.some_light('extract_inputs')

answers it: which class you really got, where it is defined, and its real
signature.

Finding a name
--------------
``help()`` needs the object; often you only remember the name. The index
below is built by parsing the package with :mod:`ast` — **nothing is
imported** — so asking where ``compute_param`` lives costs no torch, no
pyvista and no seaborn::

    FRODO.some_light('compute_param')

Only the module holding the match is imported, and only to read its
signature.

What this is not
----------------
Not a documentation browser. The full reference — parameters, returns,
raises, worked examples in two languages — is generated into
``doc/frodo_sam/`` and ``doc/gandalf/`` and read in a browser. This is a
prompt for the two questions you ask mid-session, with your hands on a
live object.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import os
import textwrap

__all__ = ['EarendilsLight']

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE = __package__ or 'FotR'

#: Directories never indexed: tests are not API, caches are not source.
_SKIP_DIRS = {'__pycache__', 'tests'}


# ── Name index (ast only, no imports) ───────────────────────────────────


@functools.lru_cache(maxsize=1)
def _name_index() -> dict:
    """Map every class/function name in the package to where it is defined.

    Returns
    -------
    dict
        ``{name: [(module, qualname, lineno), ...]}``, built by parsing the
        source with :mod:`ast`. Nothing is imported, so building the index
        costs no optional dependency; on this package it is roughly 150 ms
        and is cached for the life of the process.
    """
    index: dict = {}
    for dirpath, dirnames, filenames in os.walk(_HERE):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in sorted(filenames):
            if not filename.endswith('.py'):
                continue
            path = os.path.join(dirpath, filename)
            try:
                with open(path, encoding='utf-8') as handle:
                    tree = ast.parse(handle.read())
            except (OSError, SyntaxError):
                continue
            rel = os.path.relpath(path, _HERE)[:-3].replace(os.sep, '.')
            module = _PACKAGE if rel == '__init__' else f'{_PACKAGE}.{rel}'
            module = module.removesuffix('.__init__')
            _collect(tree, module, '', index)
    return index


def _collect(node, module: str, prefix: str, index: dict) -> None:
    """Record the classes and functions directly under *node*, recursively."""
    for child in node.body:
        if not isinstance(child, (ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
            continue
        qualname = f'{prefix}{child.name}'
        index.setdefault(child.name, []).append(
            (module, qualname, child.lineno)
        )
        if isinstance(child, ast.ClassDef):
            _collect(child, module, qualname + '.', index)


def _load(module: str, qualname: str):
    """Import *module* and walk *qualname*, or return None if unreachable."""
    try:
        obj = importlib.import_module(module)
    except Exception:
        return None
    for part in qualname.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ── Rendering (plain text; no rich, so importing this costs nothing) ─────


def _parameters(obj):
    """The callable's parameters without ``self``/``cls``, or None."""
    try:
        params = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return None
    if params and params[0].name in ('self', 'cls'):
        params = params[1:]
    return params


def _signature(obj) -> str:
    """Render a callable's signature as you would have to call it.

    ``self`` is dropped: these are looked up on the class, but the reader
    is holding an instance and will not be passing it.
    """
    params = _parameters(obj)
    if params is None:
        return ''
    try:
        sig = inspect.signature(obj).replace(parameters=params)
    except (TypeError, ValueError):
        return ''
    return f'{obj.__name__}{sig}'


def _where(obj) -> str:
    """``path:lineno`` for *obj*, relative to the package, or ''."""
    try:
        path = inspect.getsourcefile(obj)
        line = inspect.getsourcelines(obj)[1]
    except (TypeError, OSError):
        return ''
    if path is None:
        return ''
    try:
        path = os.path.join(_PACKAGE, os.path.relpath(path, _HERE))
    except ValueError:
        pass
    return f'{path}:{line}'


def _doc(obj, width: int = 78) -> str:
    """The object's docstring, indented, or a note that there is none."""
    text = inspect.getdoc(obj)
    if not text:
        return '    (no docstring)'
    return textwrap.indent(text, '    ')


def _is_opaque(obj) -> bool:
    """Whether a callable only accepts ``*args, **kwargs`` — i.e. forwards.

    ``self`` does not count, or no method would ever look opaque.
    """
    params = _parameters(obj)
    if not params:
        return False
    kinds = {p.kind for p in params}
    return kinds <= {inspect.Parameter.VAR_POSITIONAL,
                     inspect.Parameter.VAR_KEYWORD}


# ── The descriptor behind ``some_light`` ────────────────────────────────


class _Shortcut:
    """Bind :meth:`EarendilsLight.help` to whatever it is reached through.

    Accessed on the class it passes the class; accessed on an instance it
    passes the instance, which is what lets the delegation chain be walked
    for a live object. Both spellings keep working::

        FRODO.some_light('merge_datasets')   # class: static lookup
        db.some_light('extract_inputs')      # instance: resolves the quartet
    """

    def __init__(self, light: 'EarendilsLight') -> None:
        self._light = light
        self._name = 'some_light'

    def __set_name__(self, owner, name: str) -> None:
        self._name = name

    def __get__(self, obj, objtype=None):
        light, owner = self._light, obj if obj is not None else objtype

        def some_light(target=None, *, show=True):
            return light.help(target, owner=owner, show=show)

        some_light.__name__ = self._name
        some_light.__doc__ = EarendilsLight.help.__doc__
        return some_light


# ── The light itself ────────────────────────────────────────────────────


class EarendilsLight:
    """A prompt for the two questions ``help()`` cannot answer.

    Parameters
    ----------
    root : str or module or type, optional
        Only used to label the output. Kept for the historical
        ``EarendilsLight(__name__)`` spelling.

    Examples
    --------
    ::

        class FRODO:
            light = EarendilsLight(__name__)
            some_light = light.shortcut()

    See Also
    --------
    help : the built-in, which is the right tool nearly all of the time.
    """

    def __init__(self, root=None) -> None:
        """Store the label used in the overview header.

        Parameters
        ----------
        root : str or module or type, optional
            Only used for display. Kept so that the historical
            ``EarendilsLight(__name__)`` spelling keeps working.
        """
        self.root = root

    def shortcut(self) -> _Shortcut:
        """Return the descriptor to assign to a class's ``some_light``."""
        return _Shortcut(self)

    # ── public API ──────────────────────────────────────────────────────

    def help(self, target: str = None, *, owner=None, show: bool = True) -> str:
        """Explain *target*, resolving delegation when *owner* is live.

        Parameters
        ----------
        target : str, optional
            Name to look up, e.g. ``'extract_inputs'`` or
            ``'CODAReader.extract_inputs'``. With ``None`` a short
            reminder of what this does is printed instead.
        owner : object, optional
            Supplied automatically by ``some_light``. When it is an
            instance rather than a class, its delegation chain is walked
            first, so the answer is about the object you actually hold.
        show : bool, optional
            Print the result. Default ``True``. The text is returned
            either way.

        Returns
        -------
        str
            The rendered text.
        """
        if target is None:
            text = self._overview(owner)
        else:
            name = target.split('.')[-1]
            blocks = self._live(owner, name) or self._static(target, name)
            text = '\n'.join(blocks) if blocks else self._missing(target)
        if show:
            print(text)
        return text

    # ── live resolution: the reason this module exists ──────────────────

    def _live(self, owner, name: str) -> list:
        """Resolve *name* through a live object's delegation chain."""
        if owner is None or isinstance(owner, type):
            return []
        chain = getattr(type(owner), '_LIGHT_DELEGATES', ())
        if not chain:
            return []

        own = getattr(type(owner), name, None)
        out = []
        for slot in chain:
            sub = getattr(owner, slot, None)
            if sub is None:
                continue
            fn = getattr(type(sub), name, None)
            if fn is None:
                continue
            out.append(self._render_live(owner, name, slot, type(sub), fn, own))
            break                     # first in the chain wins, as __getattr__ does
        return out

    def _render_live(self, owner, name, slot, cls, fn, own) -> str:
        """Render one resolved delegation, saying where it was reached.

        Parameters
        ----------
        owner : object
            The live instance the lookup started from.
        name : str
            Attribute that was looked up.
        slot : str
            Entry of ``_LIGHT_DELEGATES`` that provided it.
        cls : type
            Class of the sub-object that actually defines it.
        fn : callable
            The implementation, unbound.
        own : callable or None
            The owner's own version, if it has one. When that version
            merely forwards, a line says so.

        Returns
        -------
        str
            Header, signature and docstring.
        """
        label = type(owner).__name__
        lines = [f'{label}.{name}  →  {cls.__name__}.{name}',
                 f'    reached through .{slot}']
        fmt = getattr(owner, 'format', None)
        if fmt:
            lines[-1] += f"   (format {fmt!r})"
        where = _where(fn)
        if where:
            lines.append(f'    {where}')
        if own is not None and _is_opaque(own):
            lines.append(f'    {label}.{name} only forwards (*args, **kwargs);'
                         ' the signature below is the one to call.')
        sig = _signature(fn)
        lines += ['', f'  {sig}' if sig else '', '', _doc(fn)]
        return '\n'.join(l for l in lines if l is not None)

    # ── static lookup through the ast index ─────────────────────────────

    def _static(self, target: str, name: str) -> list:
        """Find *name* in the index and render every match."""
        matches = _name_index().get(name, [])
        if '.' in target:                       # 'CODAReader.extract_inputs'
            wanted = target.rsplit('.', 2)[0] if target.count('.') > 1 else target
            narrowed = [m for m in matches if m[1].endswith(target)
                        or m[1].split('.')[0] == target.split('.')[0]]
            matches = narrowed or matches
        if not matches:
            return []

        out = []
        if len(matches) > 1:
            listing = '\n'.join(f'    {q}   ({m})' for m, q, _ in matches)
            out.append(f'{len(matches)} matches for {name!r}:\n{listing}\n')
        for module, qualname, lineno in matches:
            obj = _load(module, qualname)
            if obj is None:
                out.append(f'{qualname}   ({module}:{lineno})\n'
                           '    could not be imported — an optional '
                           'dependency is probably missing.')
                continue
            sig = _signature(obj)
            head = f'{qualname}   {module}:{lineno}'
            out.append('\n'.join(x for x in
                                 [head, '', f'  {sig}' if sig else '', '',
                                  _doc(obj)] if x is not None))
        return out

    # ── fallbacks ───────────────────────────────────────────────────────

    def _missing(self, target: str) -> str:
        """Explain that *target* is nowhere, and where to look instead."""
        return (f"Nothing called {target!r} in {_PACKAGE}.\n"
                "    The index holds classes and functions only, and skips "
                "tests.\n"
                "    For prose and worked examples, open doc/frodo_sam/ or "
                "doc/gandalf/.")

    def _overview(self, owner) -> str:
        """The reminder printed when no target is given.

        Mentions the live resolution only when *owner* is an instance
        that actually declares a delegation chain, so the hint is never
        advice the reader cannot follow.
        """
        label = getattr(owner, '__name__', type(owner).__name__) if owner else _PACKAGE
        live = ''
        if owner is not None and not isinstance(owner, type) \
                and getattr(type(owner), '_LIGHT_DELEGATES', ()):
            live = (f"\n  .some_light('extract_inputs')   resolves through this "
                    f"object's\n                                  delegation chain")
        return (
            f"Eärendil's Light on {label}\n"
            f"  .some_light('name')            where it lives, and its real "
            f"signature{live}\n"
            f"\n"
            f"  help(obj) is better for anything else, and always current.\n"
            f"  Full reference with examples: doc/frodo_sam/, doc/gandalf/."
        )
