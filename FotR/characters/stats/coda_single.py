"""
stats/coda_single.py
====================
Stats class for the CODA_SINGLE format: analysis across meshes.

Where ``CODASingleSets`` assembles and exports, this analyses. Its reason
to exist is the question CODA_SINGLE was built for: **is the solution
mesh-converged, and by how much?**

The workflow is two-step and deliberately so:

1. ``db.sets.reduce_cases(...)`` collapses each mesh's field to one
   number — the only way to compare meshes that share no point;
2. :meth:`CODASingleStats.richardson` turns that table into an observed
   order of convergence, a Richardson-extrapolated value and a Grid
   Convergence Index (Roache).

Representative mesh size
------------------------
GCI needs a representative cell size ``h`` per mesh. Only the *ratios*
``h_coarse / h_fine`` enter the formulas, so any quantity proportional to
``h`` works. This module uses ``h ∝ N**(-1/dim)`` with ``N`` the cell
count and ``dim`` the topological dimension (2 for a surface patch, 3 for
a volume), which makes the proportionality constant cancel exactly. A
column of explicit sizes can be given instead.
"""

import logging
import warnings
from typing import Literal, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BaseStats

if TYPE_CHECKING:
    from ..frodo import FRODO

log = logging.getLogger(__name__)

#: Safety factor of the Grid Convergence Index for three or more meshes.
GCI_SAFETY_FACTOR = 1.25


class CODASingleStats(BaseStats):
    """
    Stats class for CODA_SINGLE-format FRODO databases.

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance whose reader is a ``CODASingleReader``.

    Examples
    --------
    ::

        db = FRODO(root_dir='/data/gci/aoa_1', format='CODA_SINGLE')
        db.extract_inputs(id_groups=(3,))
        db.extract_outputs(stage=1, id_groups=(3,))

        table = db.stats.compute_stats(id_group='3', stage=1)
        gci   = db.stats.richardson(table, 'BoundaryValues_CoefPressure_mean')
    """

    def __init__(self, db: 'FRODO') -> None:
        """Attach to the parent FRODO database (see BaseStats)."""
        super().__init__(db)

    # ── BaseStats interface ───────────────────────────────────────────

    def compute_stats(
        self,
        id_group: Union[str, int],
        stage: Union[int, str],
        variables: Union[str, list, tuple, None] = None,
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        reductions: Union[list, tuple] = ('mean', 'rms', 'min', 'max', 'std'),
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Descriptive statistics of every variable, per case.

        One row per case and one column per ``(variable, reduction)``
        pair, so meshes of different sizes sit side by side in a single
        table. Built on ``CODASingleSets.reduce_cases`` rather than
        reimplementing the reduction, so the two cannot disagree.

        Parameters
        ----------
        id_group : str or int
            CADGroup identifier.
        stage : int or str
            Stage to analyse.
        variables : str, list[str] or None
            Variables to describe. ``None`` takes every non-bookkeeping
            variable of the stage.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to include. Default ``'all'``.
        reductions : tuple[str]
            Reductions to apply, from ``CODASingleSets.REDUCTIONS``.
        verbose : bool
            Log the resulting shape.

        Returns
        -------
        pd.DataFrame
            Columns: ``'case_idx'``, ``'case'``, ``'mesh_file'``,
            ``'n_points'``, ``'stages_available'``, the design variables,
            and ``'<var>_<reduction>'`` for every combination.

        Raises
        ------
        RuntimeError
            If the database has no CODA_SINGLE Sets attached.

        Side-effects
        ------------
        Stores the table in
        ``db.stats_results['CADGroup_<id>_stage_<stage>']``.

        Examples
        --------
        ::

            table = db.stats.compute_stats(id_group='3', stage=1)
            table[['case', 'n_points', 'BoundaryValues_CoefPressure_mean']]
        """
        sets = getattr(self.db, 'sets', None)
        if sets is None or not hasattr(sets, 'reduce_cases'):
            raise RuntimeError(
                "compute_stats needs the CODA_SINGLE Sets class "
                "(db.sets.reduce_cases); db.sets is "
                f"{type(sets).__name__}."
            )

        key, group = sets._resolve_group(id_group)
        stage_vars = group.get('Vars', {}).get(str(stage), {})
        if isinstance(variables, str):
            variables = [variables]
        names = (
            [v for v in stage_vars
             if v not in ('GlobalNumber', 'CADGroupID')]
            if variables is None else list(variables)
        )
        if not names:
            raise ValueError(
                f"{key} stage '{stage}': no variable to describe."
            )

        table = None
        for how in reductions:
            part = sets.reduce_cases(
                names, stage=stage, id_group=id_group, how=how,
                cases_idx=cases_idx,
            )
            renamed = part.rename(
                columns={name: f'{name}_{how}' for name in names}
            )
            if table is None:
                table = renamed
            else:
                table = table.join(
                    renamed[[f'{name}_{how}' for name in names]]
                )

        self.db.stats_results = getattr(self.db, 'stats_results', {})
        self.db.stats_results[f'{key}_stage_{stage}'] = table

        if verbose:
            log.info(
                "[CODASingleStats] %s stage %s: table %s",
                key, stage, table.shape,
            )
        return table

    # ── Mesh convergence ──────────────────────────────────────────────

    @staticmethod
    def representative_size(
        n_points: np.ndarray,
        dim: int = 2,
    ) -> np.ndarray:
        """
        Representative cell size ``h`` from a cell count.

        Uses ``h = N ** (-1/dim)``. Only the ratios ``h_coarse/h_fine``
        enter the GCI formulas, so the missing proportionality constant
        (the domain measure) cancels exactly and no area or volume is
        needed.

        Parameters
        ----------
        n_points : array-like
            Cell count per mesh.
        dim : int
            Topological dimension: 2 for a surface patch, 3 for a volume.

        Returns
        -------
        np.ndarray

        Examples
        --------
        ::

            h = CODASingleStats.representative_size([6499, 2165, 1082])
        """
        n = np.asarray(n_points, dtype=np.float64)
        if np.any(n <= 0):
            raise ValueError("n_points must be strictly positive.")
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2 or 3.")
        return n ** (-1.0 / dim)

    def richardson(
        self,
        table: pd.DataFrame,
        value_col: str,
        size_col: str = 'n_points',
        dim: int = 2,
        h_col: Optional[str] = None,
        safety_factor: float = GCI_SAFETY_FACTOR,
        max_iter: int = 200,
        tol: float = 1e-10,
    ) -> pd.DataFrame:
        """
        Observed order of convergence, Richardson extrapolation and GCI.

        Implements the three-mesh procedure (Roache) on every consecutive
        triplet of the family, ordered **fine to coarse**. For a triplet
        ``(1, 2, 3)`` with sizes ``h1 < h2 < h3`` and values ``f1, f2,
        f3``::

            r21 = h2/h1                  r32 = h3/h2
            e21 = f2 - f1                e32 = f3 - f2
            s   = sign(e32 / e21)
            p   = |ln|e32/e21| + ln((r21^p - s)/(r32^p - s))| / ln(r21)

        solved by fixed-point iteration, then::

            f_ext  = (r21^p * f1 - f2) / (r21^p - 1)
            GCI21  = Fs * |(f1 - f2)/f1| / (r21^p - 1)

        Parameters
        ----------
        table : pd.DataFrame
            One row per mesh, as returned by
            ``CODASingleSets.reduce_cases`` or :meth:`compute_stats`.
        value_col : str
            Column holding the quantity of interest.
        size_col : str
            Column holding the cell count. Default ``'n_points'``.
        dim : int
            Topological dimension for :meth:`representative_size`.
            Default 2 (surface patch).
        h_col : str or None
            Column of explicit representative sizes. Overrides
            ``size_col``/``dim`` when given.
        safety_factor : float
            ``Fs`` of the GCI. Default 1.25.
        max_iter, tol : int, float
            Fixed-point iteration controls for ``p``.

        Returns
        -------
        pd.DataFrame
            One row per consecutive triplet, fine to coarse. Columns:
            ``'fine'``, ``'medium'``, ``'coarse'`` (case names), ``'h1'``,
            ``'h2'``, ``'h3'``, ``'r21'``, ``'r32'``, ``'f1'``, ``'f2'``,
            ``'f3'``, ``'p'`` (observed order), ``'f_extrapolated'``,
            ``'e_approx_21'`` (relative), ``'e_extrapolated_21'``,
            ``'GCI_21'``, ``'GCI_32'``, ``'asymptotic_ratio'`` (≈1 when
            the triplet is in the asymptotic range) and ``'converged'``
            (whether ``p`` converged).

        Raises
        ------
        ValueError
            If fewer than three usable meshes remain, or the columns are
            missing.

        Notes
        -----
        ``p`` is undefined when two meshes give the same value
        (``e21 == 0``) and meaningless when the differences flip sign
        (oscillatory convergence); both cases yield ``NaN`` with
        ``converged=False`` rather than a number that looks authoritative.

        Examples
        --------
        ::

            table = db.sets.reduce_cases('cp', stage=1, id_group='3')
            gci   = db.stats.richardson(table, 'cp')
            gci[['fine', 'medium', 'coarse', 'p', 'GCI_21']]
        """
        for col in (value_col, h_col or size_col):
            if col not in table.columns:
                raise ValueError(
                    f"Column '{col}' not in table. Available: "
                    f"{list(table.columns)}."
                )

        work = table.dropna(subset=[value_col]).copy()
        if h_col is not None:
            work['_h'] = np.asarray(work[h_col], dtype=np.float64)
        else:
            work['_h'] = self.representative_size(
                work[size_col].to_numpy(), dim=dim
            )

        # Fine to coarse: the finest mesh has the smallest h.
        work = work.sort_values('_h', kind='stable').reset_index(drop=True)
        if len(work) < 3:
            raise ValueError(
                f"Richardson needs at least 3 meshes with a value for "
                f"'{value_col}'; got {len(work)}."
            )

        label_col = 'case' if 'case' in work.columns else work.columns[0]
        rows = []
        for i in range(len(work) - 2):
            h1, h2, h3 = work.loc[i:i + 2, '_h'].to_numpy()
            f1, f2, f3 = work.loc[i:i + 2, value_col].to_numpy()
            r21, r32 = h2 / h1, h3 / h2
            e21, e32 = f2 - f1, f3 - f2

            p, converged = self._observed_order(
                e21, e32, r21, r32, max_iter=max_iter, tol=tol,
            )

            if np.isfinite(p) and abs(r21 ** p - 1.0) > 1e-14:
                f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1.0)
                gci21 = (
                    safety_factor * abs(e21 / f1) / (r21 ** p - 1.0)
                    if f1 != 0 else np.nan
                )
                gci32 = (
                    safety_factor * abs(e32 / f2) / (r32 ** p - 1.0)
                    if f2 != 0 and abs(r32 ** p - 1.0) > 1e-14 else np.nan
                )
                e_ext = (
                    abs((f_ext - f1) / f_ext) if f_ext != 0 else np.nan
                )
                asymptotic = (
                    gci32 / (r21 ** p * gci21)
                    if np.isfinite(gci21) and gci21 != 0
                    and np.isfinite(gci32) else np.nan
                )
            else:
                f_ext = gci21 = gci32 = e_ext = asymptotic = np.nan

            rows.append({
                'fine':              work.loc[i, label_col],
                'medium':            work.loc[i + 1, label_col],
                'coarse':            work.loc[i + 2, label_col],
                'h1': h1, 'h2': h2, 'h3': h3,
                'r21': r21, 'r32': r32,
                'f1': f1, 'f2': f2, 'f3': f3,
                'p':                 p,
                'f_extrapolated':    f_ext,
                'e_approx_21':       abs(e21 / f1) if f1 != 0 else np.nan,
                'e_extrapolated_21': e_ext,
                'GCI_21':            gci21,
                'GCI_32':            gci32,
                'asymptotic_ratio':  asymptotic,
                'converged':         converged,
            })

        return pd.DataFrame.from_records(rows)

    @staticmethod
    def _observed_order(
        e21: float,
        e32: float,
        r21: float,
        r32: float,
        max_iter: int = 200,
        tol: float = 1e-10,
        p_max: float = 20.0,
    ) -> tuple:
        """
        Solve the observed order ``p`` by fixed-point iteration.

        Returns ``(nan, False)`` when the problem is ill-posed: identical
        values on two meshes, non-refining ratios, oscillatory
        differences that make the logarithm undefined, an iterate that
        runs past ``p_max``, or no convergence within ``max_iter``.
        Reporting ``NaN`` is deliberate — an unconverged ``p`` silently
        rounded to a number is how a convergence study ends up claiming an
        order it never had.

        ``p_max`` also keeps ``r ** p`` from overflowing: with refinement
        ratios close to 1 the iteration can shoot off, and an observed
        order above ~20 is not a physical result anyway.
        """
        if not np.isfinite([e21, e32, r21, r32]).all():
            return np.nan, False
        if e21 == 0 or r21 <= 1.0 or r32 <= 1.0:
            return np.nan, False

        ratio = e32 / e21
        if not np.isfinite(ratio) or ratio == 0:
            return np.nan, False

        s = float(np.sign(ratio))
        p = abs(np.log(abs(ratio)) / np.log(r21))
        if p > p_max:
            return np.nan, False
        for _ in range(max_iter):
            denom = r32 ** p - s
            numer = r21 ** p - s
            if denom == 0 or numer / denom <= 0:
                return np.nan, False
            q = np.log(numer / denom)
            p_new = abs((np.log(abs(ratio)) + q) / np.log(r21))
            if not np.isfinite(p_new) or p_new > p_max:
                return np.nan, False
            if abs(p_new - p) < tol:
                return float(p_new), True
            p = p_new
        return float(p), False
