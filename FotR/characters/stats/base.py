"""
stats/base.py
=============
Abstract base class that every FRODO Stats class must implement.

Responsibility
---------------
A Stats class operates on an already-populated ``FRODO.data_dict`` (and,
when available, ``FRODO.df_state`` / ``FRODO.metadata``) to provide
higher-level statistical analyses on top of the raw simulation data:

* **Descriptive statistics** (``compute_stats``) — the one method every
  Stats class *must* implement.  It is the generic, always-available
  entry point that every format provides, regardless of its internal
  ``data_dict`` layout.
* **Format-specific statistical studies** — concrete classes are free to
  add any additional method that makes sense for their data layout (e.g.
  CODA's ``stage_difference_stats``, which compares field variables
  across two or more solver stages).

All methods receive the FRODO instance at construction (``self.db``) and
operate directly on ``self.db.data_dict``, ``self.db.metadata``,
``self.db.df_state``, etc. — exactly like ``BaseSets`` and
``BaseResiduals``.

How to implement a new Stats class
------------------------------------
::

    # stats/my_format.py
    from .base import BaseStats
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ..frodo import FRODO

    class MyFormatStats(BaseStats):

        def __init__(self, db: 'FRODO'):
            super().__init__(db)

        def compute_stats(self, **kwargs) -> dict:
            # compute descriptive statistics from db.data_dict
            ...

Then register it in ``stats/__init__.py``::

    from .my_format import MyFormatStats
    STATS_REGISTRY['MY_FORMAT'] = MyFormatStats
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checkers; avoids circular import at runtime.
    from ..frodo import FRODO


class BaseStats(ABC):
    """
    Abstract base class for all FRODO Stats classes.

    Subclasses must implement ``compute_stats``.  All other methods are
    optional and can be added as the format requires (e.g. CODA's
    ``stage_difference_stats``).

    Attributes
    ----------
    db : FRODO
        Reference to the parent FRODO instance.  Gives access to
        ``db.data_dict``, ``db.metadata``, ``db.sim_metadata``,
        ``db.df_state``, etc.

    Examples
    --------
    Accessing a concrete Stats class through FRODO::

        from FotR import FRODO

        db = FRODO(root_dir='/data/sim', format='CODA')
        db.extract_inputs(id_groups=(3,))
        db.extract_outputs(stage=0, id_groups=(3,))

        # db.stats is an instance of CODAStats, registered automatically
        # by FRODO._set_subclasses() from STATS_REGISTRY.
        result = db.stats.compute_stats(id_group='3', stage='0')
        print(result['table'])
    """

    def __init__(self, db: 'FRODO'):
        """
        Parameters
        ----------
        db : FRODO
            The parent FRODO instance that owns this Stats object.
        """
        self.db = db

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def compute_stats(self, *args, **kwargs) -> dict:
        """
        Compute descriptive statistics from the data stored in
        ``db.data_dict``.

        The exact signature is defined by each concrete Stats class (the
        arguments differ between formats — e.g. CODA needs ``id_group``
        and ``stage`` while a flat NUMPYFILE-style format might only need
        a variable name).

        Implementations should, at minimum, store their result on
        ``self.db`` (mirroring the side-effect pattern used by
        ``BaseSets.create_jset`` and
        ``BaseResiduals.get_all_final_residuals``) in addition to
        returning it, so that downstream code (e.g. LEGOLAS plotting
        helpers, or a follow-up call from a notebook) can access the
        latest statistics without having to keep the return value around.

        Parameters
        ----------
        *args, **kwargs
            Format-specific arguments forwarded from the caller.

        Returns
        -------
        dict
            Result dictionary. Keys and structure are format-specific but
            must be documented in detail in the concrete implementation's
            docstring.
        """

    # ── Optional hooks (override as needed) ──────────────────────────────────

    def summary(self) -> str:
        """Return a short description of the Stats class and its db."""
        keys = list(self.db.data_dict.keys()) if self.db.data_dict else []
        return (
            f"{type(self).__name__}  |  format: {self.db.format}\n"
            f"  data_dict keys: {keys}"
        )

    def __repr__(self) -> str:
        return self.summary()