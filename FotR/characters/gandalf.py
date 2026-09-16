"""
gandalf.py
==========
GANDALF facade: case generation for CODA simulation databases.

The real logic lives in :mod:`FotR.characters.rings` (one ring per mesh
policy, same scheme as the FRODO format registries).  This class keeps
the historical ``GANDALF(root_dir, eq_type, num_stages, ...)`` API
working unchanged by delegating every attribute to the selected ring
(default ``'coda'``: one shared mesh, i.e. the original behaviour).
"""

import copy
from typing import Optional

from ..EarendilsLight import EarendilsLight
from .rings import RING_REGISTRY


class GANDALF:
    """
    GANDALF: Great Algorithm to N — case definition and SLURM launching.

    Parameters
    ----------
    root_dir : str
        Dataset root directory.
    eq_type : str
        ``'euler'`` or ``'rans'``.
    num_stages : int
        Number of solver stages (required).
    ring : str
        Case-generation ring: ``'coda'`` (single shared mesh, default)
        or ``'coda_single'`` (one mesh per case via ``mesh_paths``).
        Keyword-only; historical positional calls are unaffected.
    **kwargs
        Forwarded to the ring (see :class:`BaseRing`).

    All other attributes and methods are served by the ring itself
    (``define_cases``, ``generate_folders``, ``assign_jobs``,
    ``submit_cases``, ``recover_pending_jobs``, …), except ``Backpack``,
    which is only available as ``BaseRing.Backpack`` and is deliberately
    not delegated.
    Attribute writes (except ``_ring`` itself) are forwarded to the
    ring, so the facade and the ring never diverge.
    """

    #: Attributes never served from the ring (stay on BaseRing only).
    _NON_DELEGATED = frozenset({'Backpack'})

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Shortcut to Earendil's Light help system."""
        return cls.light.help(name)

    def __init__(
        self,
        root_dir: str,
        eq_type: str = "rans",
        num_stages: Optional[int] = None,
        **kwargs
    ) -> None:
        """Create the facade over the selected ring (default ``'coda'``)."""
        # Same positional order as the historical API; the ring is an
        # optional keyword so old calls keep working unchanged.
        ring = kwargs.pop('ring', 'coda')
        ring_cls = RING_REGISTRY.get(ring)
        if ring_cls is None:
            raise ValueError(
                f"Ring '{ring}' is not supported. "
                f"Available rings: {list(RING_REGISTRY)}"
            )
        object.__setattr__(self, '_ring', ring_cls(
            root_dir=root_dir, eq_type=eq_type,
            num_stages=num_stages, **kwargs
        ))

    # ── Transparent delegation ──────────────────────────────────────────

    def __getattr__(self, name):
        """Delegate everything (except dunders and Backpack) to the ring."""
        if name.startswith('__') or name in type(self)._NON_DELEGATED:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        try:
            ring = object.__getattribute__(self, '_ring')
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no active ring "
                f"(no attribute '{name}')"
            )
        try:
            return getattr(ring, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        """Forward writes to the ring, except the ring itself."""
        if name == '_ring' or '_ring' not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._ring, name, value)

    def __delattr__(self, name):
        """Forward deletions to the ring, except the ring itself."""
        if name == '_ring' or '_ring' not in self.__dict__:
            object.__delattr__(self, name)
        else:
            delattr(self._ring, name)

    def __dir__(self):
        """Expose the ring's attributes for tab-completion and help()."""
        names = set(super().__dir__())
        try:
            names.update(
                n for n in dir(object.__getattribute__(self, '_ring'))
                if n not in type(self)._NON_DELEGATED
            )
        except AttributeError:
            pass
        return sorted(names)

    def __str__(self) -> str:
        try:
            ring = object.__getattribute__(self, '_ring')
        except AttributeError:
            return "GANDALF(no active ring)"
        return f"GANDALF(ring={type(ring).__name__})"

    def __repr__(self) -> str:
        return self.__str__()

    # ── Copy / pickle support (never alias the ring) ────────────────────

    def __copy__(self):
        cls = type(self)
        new = cls.__new__(cls)
        object.__setattr__(new, '_ring', copy.copy(
            object.__getattribute__(self, '_ring')
        ))
        return new

    def __deepcopy__(self, memo):
        cls = type(self)
        new = cls.__new__(cls)
        object.__setattr__(new, '_ring', copy.deepcopy(
            object.__getattribute__(self, '_ring'), memo
        ))
        memo[id(self)] = new
        return new

    def __getstate__(self):
        return {'_ring': object.__getattribute__(self, '_ring')}

    def __setstate__(self, state):
        object.__setattr__(self, '_ring', state['_ring'])
