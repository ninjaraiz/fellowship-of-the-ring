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
from typing import Optional, TYPE_CHECKING, Union

import os
import warnings

import numpy as np
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

    def __init__(self, db: 'FRODO') -> None:
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
        stage: Union[list, tuple, str] = 'all',
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

    # ── Shared helpers (same behaviour in every format) ───────────────────

    def _save_final_residuals(
        self,
        df_final: pd.DataFrame,
        load_in_metadata: bool = True,
        filename: str = 'all_final_residuals.csv',
    ) -> pd.DataFrame:
        """Warn-and-return-empty when there is no data, else persist to CSV.

        Shared tail of every ``get_all_final_residuals`` implementation.
        """
        if df_final is None or df_final.empty:
            warnings.warn(
                "No residual data found. Returning empty DataFrame.",
                UserWarning,
            )
            return pd.DataFrame()
        if load_in_metadata:
            os.makedirs(
                os.path.join(self.db.root_dir, 'metadata'), exist_ok=True
            )
            df_final.to_csv(
                os.path.join(self.db.root_dir, 'metadata', filename),
                index=False,
            )
        return df_final

    def _resolve_case_name(
        self,
        case_idx: Optional[int] = None,
        case_name: Optional[str] = None,
    ) -> str:
        """Resolve a case folder name from either selector.

        ``case_name`` takes precedence; otherwise the reader maps
        ``case_idx`` (``case_per_idx`` in CODA, ``_case_name_from_idx``
        in HORSES3D).
        """
        if case_name is not None:
            return case_name
        if case_idx is None:
            raise ValueError("Provide either case_name or case_idx.")
        reader = self.db.reader
        if hasattr(reader, 'case_per_idx'):
            return reader.case_per_idx(case_idx)
        return reader._case_name_from_idx(case_idx)

    @staticmethod
    def _resolve_p(solutions: dict, p: Union[int, str]) -> int:
        """Resolve one polynomial order from a case's solutions dict.

        ``'max'`` picks the highest available ``p``; anything else must
        name an existing entry.
        """
        p_use = max(solutions) if p == 'max' else int(p)
        if p_use not in solutions:
            raise KeyError(
                f"p={p_use} not available. Available: {sorted(solutions)}."
            )
        return p_use

    @staticmethod
    def _cycled_colors(n: int):
        """``tab10`` colors cycled so no curve is silently dropped."""
        import matplotlib.pyplot as plt

        colors = plt.get_cmap('tab10').colors
        return [colors[i % len(colors)] for i in range(n)]