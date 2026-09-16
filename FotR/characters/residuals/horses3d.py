"""
residuals/horses3d.py
======================
Residuals class for the HORSES3D CFD solver format.

Provides the HORSES3D-side counterpart of ``CODAResiduals``:

* Parsing of HORSES3D ``.residuals`` monitor files (ASCII) into pandas
  DataFrames.
* Aggregation of final residuals across every case (and, within a case,
  a selectable polynomial order ``p``).
* Convergence-state tagging.
* Visualisation helpers: per-case residual plots, and a design-space
  scatter of final residuals across all cases (mirroring
  ``CODAResiduals.plot_all_final_residuals``).

Unlike CODA — where a simulation goes through a fixed sequence of
solver *stages*, each with its own residual monitor file — HORSES3D
exposes exactly one ``.residuals`` file per **solution** (i.e. per
``(case, p)`` pair). Every method here therefore takes an explicit
(or defaulted) ``p`` selector in addition to the case selector, but
never introduces a second, independent case-discovery mechanism: case
identity, ``case_idx``, subsets and the location of every ``.residuals``
file are all read directly from ``self.db.reader.sim_metadata``, exactly
as populated by ``Horses3DReader.parse_simulation_dirs``.
"""

import os
import re
import logging
import warnings
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import BaseResiduals

if TYPE_CHECKING:
    from ..frodo import FRODO

log = logging.getLogger(__name__)


class Horses3DResiduals(BaseResiduals):
    """
    Residuals class for HORSES3D-format FRODO databases.

    All methods work on ``self.db.reader.sim_metadata`` (populated by
    ``Horses3DReader.parse_simulation_dirs``) and read ``.residuals``
    files directly from the paths recorded there — no independent case
    discovery is performed here.

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance whose reader is a ``Horses3DReader``.

    Quick-reference
    ---------------
    ::

        db.residuals.get_df_residuals_from_case(case_idx=0, p='max')
        db.residuals.get_all_final_residuals(p='max', only_finished=True)
        db.residuals.update_converged_state(threshold=1e-6)
        db.residuals.plot_residuals_from_case(case_idx=0, p='max')
        db.residuals.plot_all_final_residuals(p='max')
    """

    def __init__(self, db: 'FRODO') -> None:
        """Attach to the parent FRODO database (see BaseResiduals)."""
        super().__init__(db)

    # =========================================================================
    # BaseResiduals interface
    # =========================================================================

    def get_all_final_residuals(
        self,
        p: Union[int, str] = 'max',
        subset: Union[str, None] = None,
        cases_idx: Union[list, tuple, int, str] = 'all',
        only_finished: bool = True,
        load_in_metadata: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with the last residual values for every
        selected case, at a given (or the highest available)
        polynomial order ``p``.

        For each selected case, the representative solution is chosen
        via ``p`` (``'max'`` picks the highest ``p`` available for that
        specific case — cases are allowed to have different ``p`` sets,
        so this is resolved independently per case). If that solution's
        ``.residuals`` file was not found on disk during discovery, the
        case is skipped under ``only_finished=True`` (or filled with NaN
        residual columns otherwise).

        Parameters
        ----------
        p : int or 'max'
            Polynomial order to read the final residuals from. ``'max'``
            (default) uses the highest ``p`` available per case.
        subset : str or None
            Name of a subset registered via ``Horses3DReader.define_subset``.
            Mutually exclusive with an explicit ``cases_idx``, exactly as
            enforced by ``Horses3DReader._resolve_cases_idx``.
        cases_idx : list, tuple, int or 'all'
            Subset of physical ``case_idx`` to include. Default ``'all'``.
        only_finished : bool
            If True, skip cases whose selected solution has no
            ``.residuals`` file on disk. Default True.
        load_in_metadata : bool
            If True, save the result to
            ``<root_dir>/metadata/all_final_residuals.csv``. Default True.
        verbose : bool
            Print information about skipped cases. Default False.

        Returns
        -------
        pd.DataFrame
            One row per included case. Columns: ``'case'``, ``'case_idx'``,
            ``'p'`` (the resolved polynomial order actually read), every
            design-variable column known to the reader (when available),
            followed by every residual column found in the ``.residuals``
            files (the header-derived column names — see
            :meth:`get_df_residuals_from_case`).

        Examples
        --------
        ::

            df = db.residuals.get_all_final_residuals(
                only_finished=False, load_in_metadata=False,
            )
            print(df.head())
        """
        reader = self.db.reader
        resolved_idx = reader._resolve_cases_idx(subset, cases_idx)
        design_vars  = reader.metadata.get('design_vars') or []

        rows: list = []
        for case_idx in resolved_idx:
            case_name = reader._case_name_from_idx(case_idx)
            sols = reader.sim_metadata[case_name]['solutions']

            if not sols:
                if verbose:
                    log.debug("Skipping '%s': no solutions found.", case_name)
                continue

            try:
                p_use = self._resolve_p(sols, p)
            except KeyError:
                if verbose:
                    log.debug(
                        "Skipping '%s': p=%s not available (has %s).",
                        case_name, p, sorted(sols),
                    )
                continue

            sol = sols[p_use]
            residuals_file = sol.get('residuals_file')

            if residuals_file is None:
                if only_finished:
                    if verbose:
                        log.debug(
                            "Skipping '%s' (p=%s): no .residuals file found.",
                            case_name, p_use,
                        )
                    continue
                df_last = pd.DataFrame()
            else:
                full_path = os.path.join(
                    reader.sim_metadata[case_name]['path'], residuals_file
                )
                try:
                    df_hist = self._read_residuals_file(full_path)
                    df_last = df_hist.tail(1).reset_index(drop=True)
                except Exception as exc:
                    if verbose:
                        log.debug(
                            "Could not read residuals for '%s' (p=%s): %s",
                            case_name, p_use, exc,
                        )
                    if only_finished:
                        continue
                    df_last = pd.DataFrame()

            row = {'case': case_name, 'case_idx': case_idx, 'p': p_use}
            row.update(reader.sim_metadata[case_name].get('design_vars', {}))
            if not df_last.empty:
                for k, v in df_last.iloc[0].to_dict().items():
                    if k in row:
                        warnings.warn(
                            f"Residual column '{k}' collides with case "
                            f"identity; keeping the identity value.",
                            UserWarning,
                        )
                        continue
                    row[k] = v
            rows.append(row)

        if not rows:
            return self._save_final_residuals(
                pd.DataFrame(), load_in_metadata=False
            )

        df_final = pd.DataFrame.from_records(rows)

        return self._save_final_residuals(
            df_final, load_in_metadata=load_in_metadata
        )

    # =========================================================================
    # Case-level residual extraction
    # =========================================================================

    def get_df_residuals_from_case(
        self,
        case_idx: Union[int, None] = None,
        case_name: Union[str, None] = None,
        p: Union[int, str] = 'max',
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Return the full iteration history stored in a single case's
        ``.residuals`` file, for a given (or the highest available)
        polynomial order.

        Parameters
        ----------
        case_idx : int or None
            Physical ``case_idx`` (row of ``db.df_state``). Used when
            ``case_name`` is None.
        case_name : str or None
            Case folder name, as used to key ``sim_metadata``. Takes
            precedence over ``case_idx`` when both are given.
        p : int or 'max'
            Polynomial order to read. ``'max'`` (default) uses the
            highest ``p`` available for this case.
        verbose : bool
            Print the resolved file path and column names.

        Returns
        -------
        pd.DataFrame
            Columns exactly as found in the ``.residuals`` file's header
            (e.g. ``'Iteration'``, ``'Time'``, ``'Total_elapsed_Time(s)'``,
            ``'Solver_elapsed_Time(s)'``, one column per solved equation,
            ``'Max-Residual'``), plus ``'p'`` (constant, the resolved
            polynomial order) and ``'total_iterations'`` (a simple
            ``0..N-1`` row counter, mirroring the convention used by
            ``CODAResiduals.get_df_residuals_from_case``).

        Raises
        ------
        ValueError
            If neither ``case_name`` nor ``case_idx`` is provided.
        KeyError
            If the case, the requested ``p``, or its ``.residuals`` file
            cannot be resolved.

        Examples
        --------
        By case_idx::

            df = db.residuals.get_df_residuals_from_case(case_idx=0)

        By case name and an explicit polynomial order::

            df = db.residuals.get_df_residuals_from_case(
                case_name='case_001', p=2,
            )
        """
        reader = self.db.reader

        case_name = self._resolve_case_name(
            case_idx=case_idx, case_name=case_name
        )

        sols = reader.sim_metadata[case_name]['solutions']
        if not sols:
            raise KeyError(f"Case '{case_name}' has no solutions.")

        try:
            p_use = self._resolve_p(sols, p)
        except KeyError as exc:
            raise KeyError(
                f"p={p} not available for case '{case_name}'. "
                f"Available: {sorted(sols)}."
            ) from exc

        residuals_file = sols[p_use].get('residuals_file')
        if residuals_file is None:
            raise KeyError(
                f"No '.residuals' file found for case '{case_name}', p={p_use}."
            )

        full_path = os.path.join(
            reader.sim_metadata[case_name]['path'], residuals_file
        )
        df = self._read_residuals_file(full_path)
        df['p'] = p_use
        df['total_iterations'] = np.arange(len(df))

        if verbose:
            log.debug("[Horses3DResiduals] Read '%s'", full_path)
            log.debug("[Horses3DResiduals] Columns: %s", list(df.columns))

        return df

    # =========================================================================
    # Convergence state
    # =========================================================================

    def update_converged_state(
        self,
        threshold: float = 1e-4,
        p: Union[int, str] = 'max',
        residual_columns: Union[list, tuple, None] = None,
        exclude_residuals: Union[list, tuple] = (),
    ) -> None:
        """
        Tag each row in ``db.df_state`` as converged / not-converged
        based on the final residual values of the selected polynomial
        order.

        A case is considered converged if **all** selected residual
        columns are below ``threshold`` in the last recorded iteration
        of its ``.residuals`` file.

        Parameters
        ----------
        threshold : float
            Convergence threshold. Default ``1e-4``.
        p : int or 'max'
            Polynomial order whose final residuals are checked. Default
            ``'max'``.
        residual_columns : list, tuple or None
            Explicit residual column names to check. ``None`` (default)
            auto-detects every equation-residual column returned by
            :meth:`get_all_final_residuals` — i.e. every column except
            the identity columns (``'case'``, ``'case_idx'``, ``'p'``,
            design variables) and the timing columns
            (``'Iteration'``, ``'Time'``, ``'Total_elapsed_Time(s)'``,
            ``'Solver_elapsed_Time(s)'``).
        exclude_residuals : list or tuple
            Column names to exclude from the auto-detected set. Default
            ``()``.

        Side-effects
        ------------
        Adds or overwrites column ``'Converged'`` (0/1) in
        ``db.df_state``, matched by the ``'case'`` column (the case
        folder name — always available and unambiguous for HORSES3D,
        unlike CODA where the match has to go through design-variable
        values).

        Examples
        --------
        ::

            db.residuals.update_converged_state(threshold=1e-6)
            converged_cases = db.df_state[db.df_state['Converged'] == 1]
        """
        df_res = self.get_all_final_residuals(
            p=p, only_finished=True, load_in_metadata=False, verbose=False,
        )
        if df_res.empty:
            warnings.warn(
                "No residual data available; cannot update converged state.",
                UserWarning,
            )
            return

        design_vars = self.db.reader.metadata.get('design_vars') or []
        identity_cols = {'case', 'case_idx', 'p', *design_vars}
        timing_cols = {
            'Iteration', 'Time', 'Total_elapsed_Time(s)',
            'Solver_elapsed_Time(s)',
        }

        if residual_columns is None:
            residual_columns = [
                c for c in df_res.columns
                if c not in identity_cols
                and c not in timing_cols
                and c not in exclude_residuals
            ]

        missing = [c for c in residual_columns if c not in df_res.columns]
        if missing:
            raise KeyError(f"Residual column(s) {missing} not found.")

        converged_mask = (df_res[residual_columns] < threshold).all(axis=1)
        converged_cases = set(df_res.loc[converged_mask, 'case'])

        self.db.df_state['Converged'] = (
            self.db.df_state['case'].isin(converged_cases).astype(int)
        )

    # =========================================================================
    # Visualisation
    # =========================================================================

    def plot_residuals_from_case(
        self,
        case_idx: Optional[int] = None,
        case_name: Optional[str] = None,
        p: Union[int, str] = 'max',
        columns: Optional[Union[list, tuple]] = None,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Plot residual convergence history for a single case, with a
        wall-clock-time panel below (mirroring
        ``CODAResiduals.plot_residuals_from_case``'s CFL sub-panel,
        adapted to HORSES3D's ``'Solver elapsed Time (s)'`` column).

        Parameters
        ----------
        case_idx : int or None
            Physical ``case_idx``. Used when ``case_name`` is None.
        case_name : str or None
            Case folder name. Takes precedence over ``case_idx``.
        p : int or 'max'
            Polynomial order to plot. Default ``'max'``.
        columns : list, tuple or None
            Residual column(s) to plot. ``None`` (default) auto-detects
            every column except ``'Iteration'``, ``'Time'``, the two
            elapsed-time columns and ``'p'`` / ``'total_iterations'``.
        save_dir : str or None
            If provided, saves the figure here instead of displaying it.
        verbose : bool
            Print intermediate information.
        **kwargs
            Extra keyword arguments forwarded to ``plt.subplots`` (e.g.
            ``figsize``).

        Examples
        --------
        ::

            db.residuals.plot_residuals_from_case(case_idx=0, p='max')
        """
        case_name = self._resolve_case_name(
            case_idx=case_idx, case_name=case_name
        )

        df_res = self.get_df_residuals_from_case(
            case_name=case_name, p=p, verbose=verbose,
        )
        p_used = df_res['p'].iloc[0]

        excluded = {
            'Iteration', 'Time', 'Total_elapsed_Time(s)',
            'Solver_elapsed_Time(s)', 'p', 'total_iterations',
        }
        if columns is None:
            columns = [c for c in df_res.columns if c not in excluded]

        _, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        colors = self._cycled_colors(len(columns))
        for ycol, color in zip(columns, colors):
            df_res.plot(
                x='total_iterations', y=ycol, s=3,
                kind='scatter', ax=ax, label=ycol,
                color=color, grid=True, logy=True,
            )

        ax.set(
            title=f"Case {case_name} (p={p_used})",
            ylabel='Residual',
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)

        if 'Solver_elapsed_Time(s)' in df_res.columns:
            divider = make_axes_locatable(ax)
            ax_time = divider.append_axes(
                "bottom", size="35%", pad=0.1, sharex=ax
            )
            ax_time.scatter(
                df_res['total_iterations'], df_res['Solver_elapsed_Time(s)'],
                color='black', s=1.5,
            )
            ax_time.set(ylabel="Solver time (s)", xlabel="Iterations")
            ax_time.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(
                save_dir, f"{case_name}_p{p_used}_residuals.png"
            )
            plt.savefig(path, bbox_inches='tight')
            log.info("Figure saved to %s", path)
        else:
            plt.show()

    def plot_all_final_residuals(
        self,
        p: Union[int, str] = 'max',
        save_dir: Union[str, None] = None,
        only_finished: bool = False,
        residual_columns: Union[list, tuple, None] = None,
        activate_idx: bool = True,
        ncols: int = 2,
        lim_converged: float = 1e-5,
        **kwargs,
    ) -> None:
        """
        Plot scatter maps of final residuals across every case in the
        design-variable space, mirroring
        ``CODAResiduals.plot_all_final_residuals``.

        Requires at least one design variable to be known
        (``db.reader.metadata['design_vars']``), either from
        ``cases_metadata.json`` or inferred from the case-folder names
        by ``Horses3DReader``. If none are available, this raises rather
        than fabricating an axis to plot against.

        Parameters
        ----------
        p : int or 'max'
            Polynomial order whose final residuals are plotted. Default
            ``'max'``.
        save_dir : str or None
            If provided, saves the figure here. Default None (show).
        only_finished : bool
            Only include cases whose selected solution has a
            ``.residuals`` file. Default False.
        residual_columns : list, tuple or None
            Residual columns to plot, one subplot each. ``None``
            (default) auto-detects every equation-residual column (same
            logic as :meth:`update_converged_state`).
        activate_idx : bool
            Annotate each scatter point with its ``case_idx``. Default
            True.
        ncols : int
            Number of subplot columns. Default 2.
        lim_converged : float
            Residual threshold used to distinguish converged (star
            marker) from non-converged (circle marker) cases. Default
            ``1e-5``.
        **kwargs
            ``cmap`` (colormap name, default ``'summer'``).

        Raises
        ------
        ValueError
            If no design variables are known, or if more than two of
            them actually vary across the selected cases (only 1-D and
            2-D design spaces are supported, mirroring CODA).

        Examples
        --------
        ::

            db.residuals.plot_all_final_residuals(
                only_finished=True, lim_converged=1e-6,
            )
        """
        df_finals = self.get_all_final_residuals(
            p=p, only_finished=only_finished, load_in_metadata=False,
        )
        if df_finals.empty:
            warnings.warn("No data to plot.", UserWarning)
            return

        design_vars = self.db.reader.metadata.get('design_vars') or []
        if not design_vars:
            raise ValueError(
                "No design_vars known for this HORSES3D dataset "
                "(neither from cases_metadata.json nor inferred from "
                "case-folder names); cannot plot a design-space scatter."
            )

        identity_cols = {'case', 'case_idx', 'p', *design_vars}
        timing_cols = {
            'Iteration', 'Time', 'Total_elapsed_Time(s)',
            'Solver_elapsed_Time(s)',
        }
        if residual_columns is None:
            residual_columns = [
                c for c in df_finals.columns
                if c not in identity_cols and c not in timing_cols
            ]

        if not residual_columns:
            warnings.warn("No residual columns to plot.", UserWarning)
            return

        dvf = [v for v in design_vars if df_finals[v].nunique() > 1]
        if len(dvf) < 1:
            dvf = design_vars[:1]
        if len(dvf) > 2:
            raise ValueError(
                "plot_all_final_residuals only supports 1 or 2 varying "
                f"design variables (found {len(dvf)}: {dvf})."
            )

        nrows = int(np.ceil(len(residual_columns) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 5 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()

        converged_mask = (df_finals[residual_columns] < lim_converged).all(axis=1)
        cmap_name = kwargs.get('cmap', 'summer')

        for i, col in enumerate(residual_columns):
            if len(dvf) == 1:
                x = df_finals[dvf[0]]
                sc_nc = axes[i].scatter(
                    x[~converged_mask], df_finals[col][~converged_mask],
                    c='tab:red', s=60, edgecolor='k', label='Non-converged',
                )
                axes[i].scatter(
                    x[converged_mask], df_finals[col][converged_mask],
                    c='tab:green', s=60, marker='*', linewidth=1.5,
                    label='Converged',
                )
                axes[i].set_yscale('log')
                axes[i].set(title=col, xlabel=dvf[0], ylabel=col)
                if activate_idx:
                    for xi, yi, ci in zip(x, df_finals[col], df_finals['case_idx']):
                        axes[i].annotate(
                            f"{ci}", (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha='center', fontsize=8,
                        )
            else:
                x, y = df_finals[dvf[0]], df_finals[dvf[1]]
                sc_nc = axes[i].scatter(
                    x[~converged_mask], y[~converged_mask],
                    c=df_finals[col][~converged_mask], cmap=cmap_name,
                    norm=mcolors.LogNorm(), s=60, edgecolor='k',
                    label='Non-converged',
                )
                axes[i].scatter(
                    x[converged_mask], y[converged_mask],
                    c=df_finals[col][converged_mask], cmap=cmap_name,
                    norm=mcolors.LogNorm(), s=60, marker='*', linewidth=1.5,
                    label='Converged',
                )
                axes[i].set(title=col, xlabel=dvf[0], ylabel=dvf[1])
                fig.colorbar(sc_nc, ax=axes[i]).ax.set_title(col)
                if activate_idx:
                    for xi, yi, ci in zip(x, y, df_finals['case_idx']):
                        axes[i].annotate(
                            f"{ci}", (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha='center', fontsize=8,
                        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', frameon=False, ncols=2)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "residuals_all_cases.png"),
                dpi=150, bbox_inches='tight',
            )
        else:
            plt.show()

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _read_residuals_file(path: str) -> pd.DataFrame:
        """
        Parse a HORSES3D ``.residuals`` ASCII file into a DataFrame.

        The file's own header line (the line starting with
        ``'#Iteration'``) is used to derive column names whenever
        possible — the set of solved-equation columns (``continuity``,
        ``x-momentum``, …) is never assumed to be fixed. The two known
        multi-word column labels (``'Total elapsed Time (s)'`` and
        ``'Solver elapsed Time (s)'``) are merged into single
        underscore-joined tokens (``'Total_elapsed_Time(s)'`` /
        ``'Solver_elapsed_Time(s)'``) before splitting on whitespace, so
        that a plain whitespace-delimited read still lines up one column
        name per data column.

        If the number of header tokens obtained this way does not match
        the number of numeric tokens in the first data row (e.g. an
        unanticipated multi-word column name), this falls back to
        generic ``col_0, col_1, ...`` names and emits a warning, rather
        than silently misaligning columns.

        Parameters
        ----------
        path : str
            Absolute path to the ``.residuals`` file.

        Returns
        -------
        pd.DataFrame
            One row per iteration, columns named as described above.

        Raises
        ------
        ValueError
            If no header line (starting with ``'#Iteration'``) and no
            data rows can be found in the file.

        Examples
        --------
        ::

            df = Horses3DResiduals._read_residuals_file(
                '/data/case_001/RESULTS/solution_p2.residuals'
            )
            print(df.columns.tolist())
        """
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            stripped = line.lstrip('#').strip()
            if stripped.lower().startswith('iteration'):
                header_idx = i
                break

        data_lines = [
            ln for ln in lines[(header_idx + 1) if header_idx is not None else 0:]
            if ln.strip() and not ln.strip().startswith('#')
        ]
        if not data_lines:
            raise ValueError(f"No data rows found in '{path}'.")

        n_data_cols = len(data_lines[0].split())

        columns = None
        if header_idx is not None:
            header_text = lines[header_idx].lstrip('#').strip()
            header_text = re.sub(
                r'total elapsed time\s*\(s\)', 'Total_elapsed_Time(s)',
                header_text, flags=re.IGNORECASE,
            )
            header_text = re.sub(
                r'solver elapsed time\s*\(s\)', 'Solver_elapsed_Time(s)',
                header_text, flags=re.IGNORECASE,
            )
            candidate_columns = header_text.split()
            if len(candidate_columns) == n_data_cols:
                columns = candidate_columns
            else:
                warnings.warn(
                    f"Header column count ({len(candidate_columns)}) does "
                    f"not match data column count ({n_data_cols}) in "
                    f"'{path}'. Falling back to generic column names.",
                    UserWarning,
                )

        if columns is None:
            columns = [f"col_{i}" for i in range(n_data_cols)]

        rows = []
        for ln in data_lines:
            try:
                rows.append([float(tok) for tok in ln.split()])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric token in '{path}': {exc}"
                ) from exc
        df = pd.DataFrame(rows, columns=columns)
        if 'Iteration' in df.columns:
            try:
                df['Iteration'] = df['Iteration'].astype(np.int64)
            except (ValueError, TypeError):
                pass

        return df