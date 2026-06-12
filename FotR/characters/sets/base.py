"""
sets/base.py
============
Abstract base class that every FRODO Sets class must implement.

Responsibility
--------------
A Sets class operates on an already-populated ``FRODO.data_dict`` to
provide higher-level operations:

* **ML tensor assembly** (``create_jset``) – the one method every Sets
  class *must* implement.
* **Mesh operations** – interpolation, cropping, coordinate reordering.
* **I/O helpers** – saving tensors / datasets to HDF5, npy, PyVista, …
* **pyLOM integration** – building ``SMEAGOL.Dataset`` objects.

All methods receive the FRODO instance at construction (``self.db``) and
operate directly on ``self.db.data_dict``, ``self.db.metadata``, etc.

How to implement a new Sets class
----------------------------------
::

    # sets/my_format.py
    from .base import BaseSets
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ..frodo import FRODO

    class MyFormatSets(BaseSets):

        def __init__(self, db: 'FRODO'):
            super().__init__(db)

        def create_jset(self, sol='all', save_path=False, verbose=False):
            # assemble tensor from db.data_dict
            ...

Then register it in ``sets/__init__.py``::

    from .my_format import MyFormatSets
    SETS_REGISTRY['MY_FORMAT'] = MyFormatSets
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    # Imported only for type checkers; avoids circular import at runtime.
    from ..frodo import FRODO


class BaseSets(ABC):
    """
    Abstract base class for all FRODO Sets classes.

    Subclasses must implement ``create_jset``.  All other methods are
    optional and can be added as the format requires.

    Attributes
    ----------
    db : FRODO
        Reference to the parent FRODO instance.  Gives access to
        ``db.data_dict``, ``db.metadata``, ``db.sim_metadata``, etc.
    """

    def __init__(self, db: 'FRODO'):
        """
        Parameters
        ----------
        db : FRODO
            The parent FRODO instance that owns this Sets object.
        """
        self.db = db

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def create_jset(self, *args, **kwargs) -> dict:
        """
        Assemble mesh coordinates, flight conditions, auxiliary arrays and
        output variables into a single flat ML-ready joint tensor.

        The exact signature is defined by each concrete Sets class (the
        arguments differ between CODA, NUMPYFILE and PYLOM formats).

        Must set at least ``self.db.jset`` or ``self.db.dict_tensors`` and
        ``self.db.df_data`` as side-effects so that downstream code can
        access the assembled data.

        Parameters
        ----------
        *args, **kwargs
            Format-specific arguments forwarded from the caller.

        Returns
        -------
        dict
            Result dict from ``SAM.Gardener.create_final_tensor``, with
            keys: 'tensor', 'scaled', 'mins', 'maxs', 'info'.
        """

    # ── Optional hooks (override as needed) ──────────────────────────────────

    def add_aux(
        self,
        array_name: str,
        array,
        notes: str = None,
    ) -> None:
        """
        Store an auxiliary array in ``db.data_dict['aux']`` and record its
        description in ``db.sim_metadata['keys_aux']``.

        Default implementation handles the generic aux dict pattern shared
        by NUMPYFILE and PYLOM.  Concrete classes that use a different
        storage layout (e.g. CODA's per-group 'Aux' sub-dict) should
        override this method.

        Parameters
        ----------
        array_name : str
            Key used in ``data_dict['aux']``.
        array : array-like
            Array to store.
        notes : str or None
            Human-readable description.
        """
        import numpy as np

        db = self.db
        db.data_dict.setdefault("aux", {})
        db.sim_metadata.setdefault("info_aux", []).append(notes)
        db.sim_metadata.setdefault("keys_aux", {})[array_name] = notes
        db.data_dict["aux"][array_name] = np.asarray(array)

    def summary(self) -> str:
        """Return a short description of the Sets class and its db."""
        keys = list(self.db.data_dict.keys()) if self.db.data_dict else []
        return (
            f"{type(self).__name__}  |  format: {self.db.format}\n"
            f"  data_dict keys: {keys}"
        )

    def __repr__(self) -> str:
        return self.summary()