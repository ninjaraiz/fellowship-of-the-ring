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

import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Literal, Optional, TYPE_CHECKING, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..sam import SAM

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

    # ── Shared helpers (same behaviour in every format) ───────────────────────

    @staticmethod
    def _normalise_cases_idx(
        cases_idx: Union[list, tuple, range, int, str],
        n_cases: int,
    ) -> list:
        """
        Normalise a case selector into a validated list of integer positions
        on an **already-extracted** case axis.

        This is the *local* case space: positions ``0..n_cases-1`` into a
        CADGroup's own arrays (``FlCc``, ``Vars``, ``idx_sort``, ``Aux``).
        It is deliberately different from
        ``CODAReader._normalise_cases_idx`` / ``._resolve_cases_idx``, which
        work on the *global* ``df_cases`` row-position space and understand
        named subsets. Every Sets method that slices an already-extracted
        group by case shares this contract.

        Parameters
        ----------
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Case selection. The string form is matched case-insensitively.
        n_cases : int
            Number of cases available on the axis being selected from.

        Returns
        -------
        list[int]

        Raises
        ------
        ValueError
            If ``cases_idx`` is a string other than ``'all'``, or has an
            unsupported type.
        IndexError
            If any requested index is out of range for ``n_cases``.

        Examples
        --------
        ::

            idx = BaseSets._normalise_cases_idx('all', n_cases=50)
            idx = BaseSets._normalise_cases_idx(range(0, 10), n_cases=50)
        """
        if isinstance(cases_idx, str):
            if cases_idx.lower() == 'all':
                cases_idx = list(range(n_cases))
            else:
                raise ValueError("Invalid string for cases_idx. Use 'all'.")
        elif isinstance(cases_idx, bool):
            raise ValueError(
                "cases_idx must be 'all', int, list[int], tuple[int] or range."
            )
        elif isinstance(cases_idx, int):
            cases_idx = [cases_idx]
        elif isinstance(cases_idx, range):
            cases_idx = list(cases_idx)
        elif isinstance(cases_idx, (list, tuple)):
            cases_idx = list(cases_idx)
        else:
            raise ValueError(
                "cases_idx must be 'all', int, list[int], tuple[int] or range."
            )

        if any(i >= n_cases or i < 0 for i in cases_idx):
            raise IndexError("cases_idx contains out-of-range values.")

        return cases_idx

    @staticmethod
    def _save_result(result: dict, save_path: str) -> None:
        """
        Persist a ``SAM.Gardener`` result dict to disk.

        Parameters
        ----------
        result : dict
            Must contain ``'tensor'``, ``'scaled'``, ``'mins'`` and
            ``'maxs'`` as torch tensors.
        save_path : str
            Destination path. Supported extensions: ``.h5``, ``.pt``,
            ``.npy``.

        Raises
        ------
        NameError
            If the extension is not supported.

        Examples
        --------
        ::

            BaseSets._save_result(result, '/output/jset.h5')
        """
        if save_path.endswith('.h5'):
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("tensor", data=result['tensor'].numpy())
                hf.create_dataset("scaled", data=result['scaled'].numpy())
                hf.create_dataset("mins",   data=result['mins'].numpy())
                hf.create_dataset("maxs",   data=result['maxs'].numpy())
        elif save_path.endswith('.pt'):
            torch.save(result, save_path)
        elif save_path.endswith('.npy'):
            np.save(save_path, result, allow_pickle=True)
        else:
            raise NameError(
                "save_path extension not supported. "
                "Use '.h5', '.pt' or '.npy'."
            )

    # =========================================================================
    # Visualisation
    # =========================================================================

    def plot_wall_integrals(
        self,
        cases_idx: Union[list, tuple],
        var_metrics: Union[str, list, tuple] = 'all',
        stage: Union[int, list, tuple, str] = 'all',
        x_axis: Literal['Iteration', 'Time'] = 'Iteration',
        stride: int = 1,
        band: bool = False,
        save_dir: Union[str, None] = None,
        figsize: tuple = (10, 6),
        cmap: str = 'tab10',
    ) -> None:
        """
        Plot aerodynamic wall-integral monitors vs iteration or time.

        Reads ``*_monitors_wall_boundary_integrals.dat`` files (e.g.
        ``CoefDrag``, ``CoefLift``, ``CoefMomentY``) with
        ``SAM.Backpack.get_df_from_csv`` — the same reading used by
        ``CODAResiduals.integrals_convergence_criteria`` — and draws one
        figure per variable, one curve per requested case.

        Cases are always explicit: ``cases_idx`` must be a non-empty
        list/tuple of global ``df_cases`` positions (no ``'all'``), so
        each figure stays readable.  Case selection is resolved through
        ``CODAReader._resolve_cases_idx``, the same central mechanism
        used everywhere else.

        Parameters
        ----------
        cases_idx : list[int] or tuple[int]
            Global case positions to plot. Required, must be non-empty.
        var_metrics : str, list[str] or 'all'
            Integral variables to plot. ``'all'`` (default) plots every
            numeric column except ``Iteration``/``Time``/``total_iter``.
        stage : int, list[int] or 'all'
            Stages whose monitor files are read. Default ``'all'``.
            Curves are labelled ``<folder> (stage <s>)``.
        x_axis : 'Iteration' or 'Time'
            Horizontal axis. Default ``'Iteration'``.
        stride : int
            Plot every ``stride``-th row (long monitor files). Must be
            ``>= 1``. Default 1.
        band : bool
            If True, overlay the mean ± std across the plotted cases
            (interpolated onto the shortest x axis). Default False.
        save_dir : str or None
            If provided, saves each figure as ``<var>.png``. Default
            None (show).
        figsize : tuple
            Figure size. Default ``(10, 6)``.
        cmap : str
            Matplotlib colormap for the per-case curves. Default
            ``'tab10'``.

        Raises
        ------
        ValueError
            If ``cases_idx`` is not a non-empty list/tuple, if ``stride``
            is ``< 1``, or if ``x_axis`` is unknown.
        UserWarning
            If a case has no integral files, or a requested variable is
            missing in a case (that case is skipped for that variable).

        Examples
        --------
        ::

            db.sets.plot_wall_integrals(
                cases_idx=[0, 1, 2],
                var_metrics=['CoefLift', 'CoefDrag'],
                save_dir='/output/plots/',
            )
        """
        if not isinstance(cases_idx, (list, tuple)) or not cases_idx:
            raise ValueError(
                "cases_idx must be a non-empty list or tuple of global "
                "case positions (no 'all': plot cases explicitly)."
            )
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be an int >= 1, got {stride!r}.")
        if x_axis not in ('Iteration', 'Time'):
            raise ValueError(
                f"x_axis must be 'Iteration' or 'Time', got {x_axis!r}."
            )

        positions = self.db.reader._resolve_cases_idx(
            list(cases_idx), None
        )
        df_cases = self.db.metadata.get('df_cases', pd.DataFrame())
        folders = df_cases.loc[positions, 'folder'].tolist()

        if isinstance(var_metrics, str) and var_metrics != 'all':
            var_metrics = [var_metrics]
        stages = (
            list(range(self.db.metadata['num_stages']))
            if stage == 'all'
            else [int(stage)] if isinstance(stage, int)
            else [int(s) for s in stage]
        )

        series: dict = {}
        for folder in folders:
            case_path = os.path.join(self.db.root_dir, 'outputs', folder)
            files = SAM.Backpack.pattern_pocket.find_files(
                path=case_path,
                endswith='_wall_boundary_integrals.dat',
                verbose=False,
            )
            for fname in files:
                match = re.search(r"output_(\d+)__", fname)
                if not match:
                    continue
                stage_no = int(match.group(1))
                if stage_no not in stages:
                    continue
                df = SAM.Backpack.get_df_from_csv(files_list=[
                    os.path.join(case_path, fname)
                ])
                if df.empty:
                    continue
                key = (folder, stage_no)
                series.setdefault(key, df)

        if not series:
            warnings.warn(
                "No wall-integral data found for the requested cases.",
                UserWarning,
            )
            return

        first = next(iter(series.values()))
        if var_metrics == 'all':
            var_list = [
                c for c in first.columns
                if c not in ('Iteration', 'Time', 'total_iter')
                and pd.api.types.is_numeric_dtype(first[c])
            ]
        else:
            var_list = list(var_metrics)
        if not var_list:
            warnings.warn("No variables to plot.", UserWarning)
            return

        cmap_obj = plt.get_cmap(cmap)
        for var in var_list:
            fig, ax = plt.subplots(figsize=figsize)
            curves = []
            for i, ((folder, stage_no), df) in enumerate(series.items()):
                if var not in df.columns:
                    warnings.warn(
                        f"Variable '{var}' missing in '{folder}' "
                        f"(stage {stage_no}); skipping that case.",
                        UserWarning,
                    )
                    continue
                sub = df.iloc[::stride]
                x = sub[x_axis].values
                y = sub[var].values
                color = cmap_obj(i % cmap_obj.N)
                ax.plot(
                    x, y, color=color, linewidth=1.2,
                    label=f'{folder} (stage {stage_no})',
                )
                curves.append((x, y))

            if band and len(curves) > 1:
                # Common x grid: shortest curve; longer ones interpolated.
                # Needs strictly increasing x on every curve.
                if not all(
                    np.all(np.diff(x) > 0) for x, _ in curves
                ):
                    warnings.warn(
                        f"band=True needs strictly increasing {x_axis}; "
                        f"skipping the band for '{var}'.",
                        UserWarning,
                    )
                else:
                    n_short = min(len(x) for x, _ in curves)
                    x_ref = curves[0][0][:n_short]
                    stack = np.column_stack([
                        y[:n_short] if len(x) == n_short
                        else np.interp(x_ref, x, y)
                        for x, y in curves
                    ])
                    mean = stack.mean(axis=1)
                    std = stack.std(axis=1)
                    ax.plot(x_ref, mean, color='k', linewidth=2.0,
                            label='mean')
                    ax.fill_between(x_ref, mean - std, mean + std,
                                    color='k', alpha=0.15, label='± std')

            ax.set(
                title=f'{var} vs {x_axis}',
                xlabel=x_axis,
                ylabel=var,
            )
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='best', fontsize='small')
            fig.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(save_dir, f"{var}.png"),
                    dpi=150, bbox_inches='tight',
                )
                plt.close(fig)
            else:
                plt.show()

    # ── Optional hooks (override as needed) ──────────────────────────────────

    def add_aux(
        self,
        array_name: str,
        array: np.ndarray,
        notes: Optional[str] = None,
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