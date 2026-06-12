"""
residuals/base.py
=================
Abstract base class that every FRODO Residuals class must implement.

Responsibility
--------------
A Residuals class works on an already-parsed FRODO instance and provides:

* Access to solver residual files (convergence monitoring).
* Extraction of integral metrics (lift, drag, …) over the last N iterations.
* Visualisation helpers for residual maps and state plots.

All methods receive the FRODO instance at construction (``self.db``).

How to implement a new Residuals class
---------------------------------------
::

    # residuals/my_format.py
    from .base import BaseResiduals
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ..frodo import FRODO

    class MyFormatResiduals(BaseResiduals):

        def __init__(self, db: 'FRODO'):
            super().__init__(db)

        def get_all_final_residuals(self, **kwargs):
            ...

Then register it in ``residuals/__init__.py``::

    from .my_format import MyFormatResiduals
    RESIDUALS_REGISTRY['MY_FORMAT'] = MyFormatResiduals
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..frodo import FRODO


class BaseResiduals(ABC):
    """
    Abstract base class for all FRODO Residuals classes.

    Subclasses must implement ``get_all_final_residuals``.

    Attributes
    ----------
    db : FRODO
        Reference to the parent FRODO instance.
    """

    def __init__(self, db: 'FRODO'):
        """
        Parameters
        ----------
        db : FRODO
            The parent FRODO instance that owns this Residuals object.
        """
        self.db = db

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def get_all_final_residuals(
        self,
        stage='all',
        verbose: bool = False,
        only_finished: bool = True,
        load_in_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with the last residual values for every
        simulation found in ``db.sim_metadata``.

        Parameters
        ----------
        stage : list, tuple or 'all'
            Stages to include. Default 'all'.
        verbose : bool
            Print per-case information. Default False.
        only_finished : bool
            Skip simulations that have not completed all stages.
        load_in_metadata : bool
            If True, save the result to metadata/all_final_residuals.csv.

        Returns
        -------
        pd.DataFrame
            One row per simulation. Columns are residual names followed by
            design variable names.
        """

    # ── Optional hooks ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Short description of this Residuals object."""
        return (
            f"{type(self).__name__}  |  format: {self.db.format}\n"
            f"  root_dir: {self.db.root_dir}"
        )

    def __repr__(self) -> str:
        return self.summary()