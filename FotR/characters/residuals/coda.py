"""
residuals/coda.py
=================
Residuals class for the CODA CFD solver format.

Provides:

* Parsing of CODA monitor files (``.dat``) and residual log files
  (``-out.txt``) into pandas DataFrames.
* Aggregation of final residuals across all simulations.
* Integral-metric convergence analysis (lift, drag, …).
* Visualisation helpers: residual scatter plots, state maps, convergence
  surfaces.
"""

import os
import re
import socket
import getpass
import logging
import warnings
import subprocess
from typing import Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..._lazy import lazy_module

# Deferred heavy imports; see ``FotR/_lazy.py``.
# ``make_axes_locatable`` is imported by the methods that need it.
plt     = lazy_module('matplotlib.pyplot')
mcolors = lazy_module('matplotlib.colors')
go      = lazy_module('plotly.graph_objects')

from ..sam import SAM
from .base import BaseResiduals

if TYPE_CHECKING:
    from ..frodo import FRODO

log = logging.getLogger(__name__)


class CODAResiduals(BaseResiduals):
    """
    Residuals class for CODA-format FRODO databases.

    All methods work on ``self.db.sim_metadata`` (populated by
    ``CODAReader.parse_simulation_dirs``) and read monitor files directly
    from the simulation output directories.

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance.
    """

    def __init__(self, db: 'FRODO') -> None:
        """Attach to the parent FRODO database (see BaseResiduals)."""
        super().__init__(db)

    # =========================================================================
    # BaseResiduals interface
    # =========================================================================

    def get_all_final_residuals(
        self,
        stage: Union[list, tuple, str] = 'all',
        verbose: bool = False,
        only_finished: bool = True,
        load_in_metadata: bool = True,
        subset: Union[str, None] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with the last residual values for every
        simulation found in ``db.sim_metadata``.

        For each simulation that passes the ``only_finished`` filter,
        ``get_df_residuals_from_case`` is called and the last row (final
        residual state) is extracted.

        Parameters
        ----------
        stage : list, tuple or 'all'
            Stages to include when reading residual monitor files.
            Default ``'all'``.
        verbose : bool
            Print information about skipped simulations. Default False.
        only_finished : bool
            If True, skip simulations whose number of completed stages is
            less than ``db.metadata['num_stages']``. Default True.
        load_in_metadata : bool
            If True, save the result to
            ``<root_dir>/metadata/all_final_residuals.csv``. Default True.
        subset : str or None
            Name of a subset previously defined via
            ``db.define_subset(...)`` (see ``CODAReader.define_subset``).
            When given, only folders whose case belongs to that subset are
            included; every other folder is skipped exactly as if it had
            failed the ``only_finished`` filter. Case selection is
            resolved through ``CODAReader._resolve_cases_idx`` — the same
            central mechanism used by ``extract_inputs`` /
            ``extract_outputs`` — so no subset-specific logic is
            duplicated here. Default None (no subset filtering, i.e. the
            previous behaviour).

        Returns
        -------
        pd.DataFrame
            One row per included simulation.  Columns are residual names
            followed by design-variable names.

        Examples
        --------
        ::

            df = db.residuals.get_all_final_residuals(
                only_finished=False, load_in_metadata=False
            )
            print(df.head())

        Restrict to a previously defined subset::

            db.define_subset(name='mach_07', cases_idx=[0, 1, 2])
            df = db.residuals.get_all_final_residuals(subset='mach_07')
        """
        folder_fmt = self.db.metadata.get('folder_fmt', '')
        pattern    = SAM.Backpack.pattern_pocket.FilenamePattern.from_template(
            folder_fmt, numeric=True
        ).compiled

        allowed_folders = None
        if subset is not None:
            resolved_idx    = self.db.reader._resolve_cases_idx('all', subset)
            allowed_folders = set(
                self.db.metadata['df_cases'].loc[resolved_idx, 'folder']
            )

        df_all = []
        df_ref = None

        for folder in self.db.sim_metadata:
            if not re.match(pattern, folder):
                continue
            if allowed_folders is not None and folder not in allowed_folders:
                continue

            design_vars = self.db.metadata['design_vars']
            matched = self.db.metadata['df_cases'][design_vars][
                self.db.metadata['df_cases']['folder'] == folder
            ]
            # A folder with no df_cases row has no design-variable values,
            # so its row would be narrower than the rest and the final
            # np.vstack would die with an opaque shape error. Skip it with
            # a message that says which folder and why.
            if len(matched) != 1:
                warnings.warn(
                    f"Folder '{folder}' matches {len(matched)} rows of "
                    "df_cases (expected exactly 1); it has no usable "
                    "design-variable values and is left out of the final "
                    "residuals. Regenerate cases_metadata.json so that its "
                    "'folder' column matches the directories on disk.",
                    UserWarning,
                )
                continue
            params_float = np.atleast_1d(
                matched.values.squeeze()
            ).astype(np.float64).tolist()
            stages_done = len(self.db.sim_metadata[folder]['stages'])

            if only_finished and stages_done < self.db.metadata['num_stages']:
                if verbose:
                    log.debug(
                        "Skipping '%s': %s/%s stages.",
                        folder, stages_done,
                        self.db.metadata['num_stages'],
                    )
                continue

            try:
                df_one = self.get_df_residuals_from_case(
                    case_name=folder, stage=stage
                )
            except (FileNotFoundError, ValueError) as exc:
                # Pending case: monitor files not on disk (or unreadable).
                # Behaves like "no residual data" below instead of aborting
                # the whole aggregation.
                if verbose:
                    log.debug(
                        "No readable residual data for '%s': %s", folder, exc
                    )
                df_one = None

            if df_one is None or df_one.empty:
                if verbose:
                    log.debug("No residual data for '%s'.", folder)
                if df_ref is None:
                    continue
                res = np.full((1, len(df_ref.columns)), np.nan)
            else:
                if df_ref is None:
                    df_ref = df_one
                res = df_one.tail(1).values.reshape(1, -1)
            fila = np.concatenate(
                (res, np.atleast_2d(np.asarray(params_float, dtype=np.float64))),
                axis=1, dtype=np.float64,
            )
            if df_all and fila.shape[1] != df_all[0].shape[1]:
                warnings.warn(
                    f"Folder '{folder}' produced {fila.shape[1]} columns but "
                    f"the previous cases produced {df_all[0].shape[1]}; "
                    "skipping it instead of building a misaligned table.",
                    UserWarning,
                )
                continue
            df_all.append(fila)

        if not df_all or df_ref is None:
            return self._save_final_residuals(
                pd.DataFrame(), load_in_metadata=False
            )

        names = list(df_ref.columns) + self.db.metadata['design_vars']
        df_final = pd.DataFrame(np.vstack(df_all), columns=names)

        return self._save_final_residuals(
            df_final, load_in_metadata=load_in_metadata
        )

    # =========================================================================
    # Case-level residual extraction
    # =========================================================================

    def get_df_residuals_from_case(
        self,
        case_name: Optional[str] = None,
        case_idx: Optional[int] = None,
        stage: Union[list, tuple, str] = 'all',
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with absolute, normalised and scaled residuals
        for all requested stages of a single simulation.

        Reads three CODA monitor files per stage:

        * ``output_<s>__monitors_TimeIntegration.dat``   – raw residuals.
        * ``output_<s>__monitors_stage<s>InitialResidual.dat`` – r0 values.
        * ``output_<s>__monitors_CFLRamp.dat``            – reference scales.

        Normalised residual: ``r / r0``.
        Scaled residual:     ``r / S``  (S from SERReference columns in CFL file).

        Parameters
        ----------
        case_name : str or None
            Folder name key in ``db.sim_metadata``.
        case_idx : int or None
            Row index in ``db.df_state``.  Used only when ``case_name`` is
            None; resolved via ``db.reader.case_per_idx``.
        stage : list, tuple or 'all'
            Stages to read. Default ``'all'``.
        verbose : bool
            Print the number of iterations and variable list per stage.

        Returns
        -------
        pd.DataFrame
            Columns include original residual columns plus ``<name>_norm``,
            ``<name>_scaled``, ``'stage'``, and ``'total_iterations'``.

        Raises
        ------
        ValueError
            If neither ``case_name`` nor ``case_idx`` is provided.

        Examples
        --------
        By folder name::

            df = db.residuals.get_df_residuals_from_case(case_name='aoa_3.0_mach_0.75')

        By df_state index::

            df = db.residuals.get_df_residuals_from_case(case_idx=12)
        """
        if case_name is None:
            if case_idx is None:
                raise ValueError("Provide either case_name or case_idx.")
            resolved  = self.db.reader.case_per_idx(case_idx)
            case_path = self.db.sim_metadata[resolved]['path']
            stages    = (
                list(self.db.sim_metadata[resolved]['stages'].keys())
                if stage == 'all' else stage
            )
        else:
            case_path = self.db.sim_metadata[case_name]['path']
            stages    = (
                list(self.db.sim_metadata[case_name]['stages'].keys())
                if stage == 'all' else stage
            )

        dfs_stage = []
        for s in stages:
            fp_time = os.path.join(
                case_path, f"output_{s}__monitors_TimeIntegration.dat"
            )
            fp_init = os.path.join(
                case_path,
                f"output_{s}__monitors_stage{s}InitialResidual.dat",
            )
            fp_cfl  = os.path.join(
                case_path, f"output_{s}__monitors_CFLRamp.dat"
            )

            df_abs  = SAM.Backpack.get_df_from_csv([fp_time])
            df_init = SAM.Backpack.get_df_from_csv([fp_init])
            df_cfl  = SAM.Backpack.get_df_from_csv([fp_cfl])

            r0_vals  = df_init.iloc[0].to_dict()
            ref_cols = [c for c in df_cfl.columns if c.startswith("SERReference")]
            ref_vals = df_cfl[ref_cols].copy()
            ref_vals.columns = [
                re.sub(r"^SERReference", "", c) for c in ref_cols
            ]

            df_stage = df_abs.copy()
            for col in df_abs.columns:
                if "Residual" in col:
                    r0 = r0_vals.get(col, None)
                    S  = ref_vals.get(col, None)
                    if r0 is not None:
                        df_stage[f"{col}_norm"]   = df_abs[col] / r0
                    if S is not None:
                        df_stage[f"{col}_scaled"] = df_abs[col] / S

            df_stage["stage"] = s
            dfs_stage.append(df_stage)

            if verbose:
                res_cols = [c for c in df_stage.columns if 'Residual' in c]
                log.debug(
                    "[INFO] Stage %s: %s iterations  |  vars: %s",
                    s, len(df_stage), res_cols,
                )

        df_all                    = pd.concat(dfs_stage, ignore_index=True)
        df_all['total_iterations'] = np.arange(len(df_all))
        return df_all

    # =========================================================================
    # Text-file residual parsing (CODA -out.txt)
    # =========================================================================

    @staticmethod
    def get_df_residuals_from_txt(
        case_path: str,
        verbose: bool = True,
        txt_from_end: int = 1,
    ) -> Optional[pd.DataFrame]:
        """
        Parse a CODA ``-out.txt`` residual log file into a pandas DataFrame.

        Extracts iteration number, CFL, density residual, momentum residual
        and energy residual via a regular-expression search on each line.

        Parameters
        ----------
        case_path : str
            Simulation output folder path (not the file itself).
        verbose : bool
            If True, prints warnings when no file is found and the name of
            each file being read. Default True.
        txt_from_end : int
            Index from the end of the sorted file list to read.  ``1`` means
            the last file (most recent run). Default 1.

        Returns
        -------
        pd.DataFrame or None
            Columns: ``['iters', 'cfl', 'rho_res', 'mom_res', 'energ_res']``.
            Returns None if no matching file is found in ``case_path``.

        Examples
        --------
        ::

            df = CODAResiduals.get_df_residuals_from_txt(
                '/data/outputs/aoa_3.0_mach_0.75', verbose=False
            )
            if df is not None:
                plt.semilogy(df['iters'], df['rho_res'])
        """
        files = SAM.Backpack.pattern_pocket.find_files(
            path=case_path, endswith="-out.txt", verbose=False
        )
        if not files:
            if verbose:
                warnings.warn(
                    f"No -out.txt file found in {case_path}.",
                    UserWarning,
                )
            return None

        if not 1 <= txt_from_end <= len(files):
            raise IndexError(
                f"txt_from_end={txt_from_end} out of range for "
                f"{len(files)} file(s) in {case_path}."
            )
        files = [files[-txt_from_end]] if isinstance(files, list) else [files]

        regex = re.compile(
            r"Iteration (\d+):\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
            r"\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
        )

        list_df = []
        for file in files:
            if verbose:
                log.debug("Reading %s", file)
            with open(file, 'r', encoding='utf-8', errors='replace') as fh:
                content = fh.read()

            rows = [
                (cnt, float(m.group(2)), float(m.group(3)),
                 float(m.group(4)), float(m.group(5)))
                for cnt, m in enumerate(
                    m for m in (regex.search(ln) for ln in content.splitlines())
                    if m
                )
            ]
            if rows:
                iters, cfls, rhos, moms, energs = zip(*rows)
            else:
                iters = cfls = rhos = moms = energs = []

            list_df.append(pd.DataFrame({
                "iters":     list(iters),
                "cfl":       list(cfls),
                "rho_res":   list(rhos),
                "mom_res":   list(moms),
                "energ_res": list(energs),
            }))

        return pd.concat(list_df, axis=0, ignore_index=True)

    # =========================================================================
    # Convergence state helpers
    # =========================================================================

    def update_converged_state(
        self,
        threshold: float = 1e-4,
        exclude_residuals: Union[list, tuple] = ('MomentumYResidual',),
    ) -> None:
        """
        Tag each row in ``db.df_state`` as converged / not-converged based
        on final normalised residual values.

        A simulation is considered converged if **all** normalised residual
        columns (excluding those in ``exclude_residuals``) are below
        ``threshold`` in the last recorded iteration.

        Parameters
        ----------
        threshold : float
            Convergence threshold. Default 1e-4.
        exclude_residuals : list or tuple
            Residual column name substrings to exclude from the check.
            Default ``('MomentumYResidual',)``.

        Side-effects
        ------------
        Adds or overwrites column ``'Converged'`` (0/1) in ``db.df_state``.

        Examples
        --------
        ::

            db.residuals.update_converged_state(threshold=1e-5)
            converged_cases = db.df_state[db.df_state['Converged'] == 1]
        """
        df_res = self.get_all_final_residuals(
            verbose=False, only_finished=True, load_in_metadata=False
        )
        cols = [
            c for c in df_res.columns
            if c.endswith('norm')
            and all(ex not in c for ex in exclude_residuals)
        ]
        df_conv = df_res[
            (df_res[cols] < threshold).all(axis=1)
        ][self.db.metadata['design_vars']]

        dv = self.db.metadata['design_vars']
        self.db.df_state["key"] = list(
            zip(*[self.db.df_state[p] for p in dv])
        )
        df_conv["key"] = list(zip(*[df_conv[p] for p in dv]))

        self.db.df_state["Converged"] = (
            self.db.df_state["key"].isin(df_conv["key"]).astype(int)
        )
        self.db.df_state.drop(columns=["key"], inplace=True)

    def get_df_metrics(
        self,
        var_metrics: Union[str, list, tuple],
        iter_var: int = 1000,
        save: bool = False,
        subset: Union[str, None] = None,
    ) -> pd.DataFrame:
        """
        Build a DataFrame with mean and variance of integral metrics
        (e.g. lift, drag) over the last ``iter_var`` iterations for every
        case and stage.

        For each case, the method:
        1. Merges final residuals per stage into ``df_post``.
        2. Reads ``*_monitors_wall_boundary_integrals.dat`` files and
           computes mean and variance over the last ``iter_var`` rows.

        Parameters
        ----------
        var_metrics : str or list[str]
            Names of the integral variables to track.  They must exist as
            column names in the ``_wall_boundary_integrals.dat`` files.
            Example: ``['CoefLift', 'CoefDrag']``.
        iter_var : int
            Number of trailing iterations used to compute statistics.
            Default 1000.
        save : bool
            If True, saves ``df_post`` to
            ``<root_dir>/metadata/df_post.csv``. Default False.
        subset : str or None
            Name of a subset previously defined via
            ``db.define_subset(...)``. When given, only the cases
            belonging to that subset are processed and returned: their
            metrics are computed exactly as before, while every other
            case is simply left out of the result instead of being kept
            with all-NaN metric columns. Selection is resolved through
            ``CODAReader._resolve_cases_idx``, the same central mechanism
            used everywhere else, so ``df_post`` still represents metrics
            for the real, processed cases and no independent subset logic
            is introduced here. Default None (process every case, i.e.
            the previous behaviour).

        Returns
        -------
        pd.DataFrame
            ``db.df_state`` (optionally restricted to ``subset``)
            augmented with columns ``<var>_mean_stage<s>`` and
            ``<var>_var_stage<s>`` for each variable and stage
            combination.

        Examples
        --------
        ::

            df = db.residuals.get_df_metrics(
                var_metrics=['CoefLift', 'CoefDrag'],
                iter_var=500,
                save=True,
            )
            print(df[['AoA', 'Mach', 'CoefLift_mean_stage0']].head())

        Restrict to a previously defined subset::

            db.define_subset(name='mach_07', cases_idx=[0, 1, 2])
            df = db.residuals.get_df_metrics(
                var_metrics=['CoefLift'], subset='mach_07',
            )
        """
        db = self.db
        if isinstance(var_metrics, str):
            var_metrics = [var_metrics]

        allowed_case_idx = None
        if subset is not None:
            resolved_idx     = db.reader._resolve_cases_idx('all', subset)
            allowed_case_idx = set(int(i) for i in resolved_idx)

        df_post     = db.df_state.copy()
        rename_dict = {
            col:col.lower() for col in df_post.columns
        }
        df_post  = df_post.rename(columns=rename_dict)
        
        design_vars = db.metadata['design_vars']
        n_stages    = db.metadata['num_stages']

        # Pre-allocate metric columns
        for stage in range(n_stages):
            for v in var_metrics:
                df_post[f"{v}_mean_stage{stage}"] = np.nan
                df_post[f"{v}_var_stage{stage}"]  = np.nan

        # Merge final residuals per stage
        for stage in range(n_stages):
            df_finals = self.get_all_final_residuals(
                verbose=False, stage=[stage],
                only_finished=False, load_in_metadata=False,
            ).copy()
            df_finals.columns = df_finals.columns.astype(str).str.lower()
            rename_dict = {
                col: f"{col}_stage{stage}"
                for col in df_finals.columns
                if col not in [dv.lower() for dv in design_vars]
            }
            
            df_finals  = df_finals.rename(columns=rename_dict)

            df_post    = df_post.merge(df_finals, on=[v.lower() for v in design_vars], how="left")

        # Fill integral metric columns.
        #
        # A row position of df_post is NOT a case_idx: df_post derives from
        # df_state, whose row order need not follow df_cases (CODA_SINGLE
        # discovers folders in lexicographic order, so 'f10' lands before
        # 'f2'). Using the position to resolve the case via case_per_idx()
        # while writing at that same position silently attributed each
        # case's metrics to a different row. The case identity is taken
        # from the row's own 'case_idx' whenever it is available.
        if 'case_idx' in df_post.columns:
            row_case_idx = df_post['case_idx'].tolist()
        else:
            row_case_idx = list(range(len(df_post)))

        for pos, case_idx in enumerate(row_case_idx):
            if pd.isna(case_idx):
                continue
            case_idx = int(case_idx)
            if allowed_case_idx is not None and case_idx not in allowed_case_idx:
                continue
            row_label   = df_post.index[pos]
            case_name   = db.reader.case_per_idx(case_idx)
            output_path = os.path.join(db.root_dir, 'outputs', case_name)

            if not os.path.exists(output_path):
                continue

            files_list = [
                f for f in os.listdir(output_path)
                if f.endswith("_monitors_wall_boundary_integrals.dat")
            ]
            if not files_list:
                continue

            for fname in files_list:
                match = re.search(r"output_(\d+)__", fname)
                if not match:
                    continue
                stage    = int(match.group(1))
                full_path = os.path.join(output_path, fname)
                df_int   = SAM.Backpack.get_df_from_csv(files_list=[full_path])

                if not all(v in df_int.columns for v in var_metrics):
                    continue

                df_tail = df_int[var_metrics].tail(iter_var)
                for v in var_metrics:
                    df_post.at[row_label, f"{v}_mean_stage{stage}"] = (
                        df_tail[v].mean()
                    )
                    df_post.at[row_label, f"{v}_var_stage{stage}"] = (
                        df_tail[v].var()
                    )

        if allowed_case_idx is not None:
            # 'case_idx' is carried through from df_cases via the merges
            # above; restrict the returned table to the requested subset
            # instead of keeping every other case with all-NaN metrics.
            case_idx_col = 'case_idx' if 'case_idx' in df_post.columns else None
            if case_idx_col is not None:
                df_post = df_post[
                    df_post[case_idx_col].astype(int).isin(allowed_case_idx)
                ]
            else:
                df_post = df_post[df_post.index.isin(allowed_case_idx)]

        df_post = df_post.sort_values(
            by=db.metadata['design_vars'][0].lower(), ignore_index=True
        ).reset_index(drop=True)

        if save:
            df_post.to_csv(
                os.path.join(db.root_dir, 'metadata', 'df_post.csv')
            )

        return df_post

    def integrals_convergence_criteria(
        self,
        iterations_back: int = 1000,
        only_finished: bool = False,
        only_converged: bool = False,
        residual_threshold: float = 1e-4,
        columns_to_remove: Union[list, tuple] = (
            'total_iter', 'Iteration', 'Time'
        ),
        mode: Literal['2D', '3D'] = '3D',
        plot: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyse convergence of integral variables based on the last
        ``iterations_back`` iterations, with optional 2-D or 3-D plots.

        For each simulation (filtered by ``only_finished`` /
        ``only_converged``), the mean and standard deviation of all
        integral variables over the last ``iterations_back`` rows are
        computed.

        Parameters
        ----------
        iterations_back : int
            Number of trailing iterations used for statistics. Default 1000.
        only_finished : bool
            Skip simulations that have not completed all stages. Default False.
        only_converged : bool
            Skip simulations that do not meet the residual convergence
            criterion (< 1e-4 in all normalised residuals except
            MomentumYResidual). Implies ``only_finished=True``.
        columns_to_remove : list or tuple
            Columns excluded from mean/std computation. Default
            ``('total_iter', 'Iteration', 'Time')``.
        mode : '2D' or '3D'
            Plot type when ``plot=True``. ``'2D'`` produces scatter plots;
            ``'3D'`` produces interactive Plotly surfaces.
        plot : bool
            Generate diagnostic plots. Default False.
        verbose : bool
            Print information about skipped simulations. Default False.
        **kwargs
            Forwarded to the plot functions (e.g. ``figsize``, ``width``,
            ``height``).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(result_mean, result_std)`` – one row per simulation,
            columns are design variables followed by integral variable names.

        Notes
        -----
        Each simulation folder found in ``db.sim_metadata`` is matched
        against ``db.metadata['df_cases']`` by an **exact** lookup on the
        ``'folder'`` column — the same column populated once (and
        precisely) by ``CODAReader.parse_simulation_dirs`` — rather than by
        re-parsing numeric values out of the folder name with a regular
        expression.

        This matters because folder names are frequently *rounded* for
        readability (e.g. ``'aoa_3.00_mach_0.750'`` for an underlying AoA
        of ``3.0042``), while ``df_cases`` stores the precise design
        variable values. Re-parsing the rounded folder name and comparing
        it against the precise ``df_cases`` values with a tight-tolerance
        ``np.isclose`` (the previous behaviour) could silently fail to
        match, causing valid simulations — including fully converged ones
        — to be skipped even when ``only_converged=False``. Looking the
        case up directly via the ``'folder'`` column removes this source
        of mismatch entirely, and also makes the design-variable values
        reported in ``result_mean`` / ``result_std`` exact rather than the
        rounded folder-name values.

        Examples
        --------
        ::

            mean_df, std_df = db.residuals.integrals_convergence_criteria(
                iterations_back=500,
                only_finished=True,
                plot=True,
                mode='2D',
            )
        """
        if only_converged and not only_finished:
            warnings.warn(
                "only_converged requires only_finished=True. "
                "Enabling it automatically.",
                UserWarning,
            )
            only_finished = True

        all_means: list = []
        all_std:   list = []

        df_res = self.get_all_final_residuals(
            verbose=False, only_finished=only_finished,
            load_in_metadata=False,
        )
        cols = [
            c for c in df_res.columns
            if c.endswith('norm') and 'MomentumYResidual' not in c
        ]
        df_filtered = (
            df_res[(df_res[cols] < residual_threshold).all(axis=1)][
                self.db.metadata['design_vars']
            ]
            if only_converged
            else df_res[self.db.metadata['design_vars']]
        )

        folder_fmt       = self.db.metadata.get('folder_fmt', '')
        pattern          = SAM.Backpack.pattern_pocket.FilenamePattern.from_template(folder_fmt, numeric=True).compiled
        df_integrals_ref = None
        df_cases         = self.db.metadata.get('df_cases', pd.DataFrame())
        design_vars      = self.db.metadata['design_vars']

        if 'folder' not in df_cases.columns:
            raise KeyError(
                "df_cases has no 'folder' column. Run "
                "db.reader.parse_simulation_dirs() before calling "
                "integrals_convergence_criteria()."
            )

        for folder_name, dic in self.db.sim_metadata.items():
            if not re.match(pattern, folder_name):
                continue

            stages_done = len(self.db.sim_metadata[folder_name]['stages'])

            if only_finished and stages_done < self.db.metadata['num_stages']:
                if verbose:
                    log.debug("Skipping '%s': not finished.", folder_name)
                continue

            # ── Resolve exact design-variable values for this folder ───────
            # Looked up directly from df_cases via the precise 'folder'
            # column instead of being re-parsed (and rounded) from the
            # folder name itself — see the "Notes" section in the
            # docstring for why this matters.
            case_row = df_cases.loc[df_cases['folder'] == folder_name]
            if case_row.empty:
                if verbose:
                    log.debug(
                        "Skipping '%s': no matching entry in df_cases.",
                        folder_name,
                    )
                continue
            valores = case_row[design_vars].iloc[0].astype(float).tolist()

            mask = np.ones(len(df_filtered), dtype=bool)
            for val, var in zip(valores, design_vars):
                mask &= np.isclose(df_filtered[var].values, val)

            if not mask.any():
                if verbose:
                    log.debug(
                        "Skipping '%s': does not meet convergence criteria.",
                        folder_name,
                    )
                continue

            df_int = SAM.Backpack.get_df_from_csv(
                files_list=SAM.Backpack.pattern_pocket.find_files(
                    path=dic['path'],
                    endswith='_wall_boundary_integrals.dat',
                    verbose=False,
                )
            )
            df_integrals_ref = df_int
            last = df_int.tail(iterations_back).drop(
                list(columns_to_remove), axis=1, errors='ignore'
            )
            all_means.append(valores + list(last.mean().values))
            all_std.append(valores   + list(last.std().values))

        if df_integrals_ref is None:
            warnings.warn("No integral data found.", UserWarning)
            return pd.DataFrame(), pd.DataFrame()

        cols_clean = list(
            df_integrals_ref.drop(
                list(columns_to_remove), axis=1, errors='ignore'
            ).columns
        )
        columns     = self.db.metadata['design_vars'] + cols_clean
        result_mean = pd.DataFrame(np.array(all_means), columns=columns)
        result_std  = pd.DataFrame(np.array(all_std),   columns=columns)

        if plot and len(self.db.metadata['design_vars']) == 2:
            self._plot_convergence(
                result_mean, result_std, columns, mode, **kwargs
            )

        return result_mean, result_std

    # =========================================================================
    # Restart helpers
    # =========================================================================

    def restart_stalled_cases(
        self,
        cases: Union[str, int, list, tuple],
        cpus: int,
        node: str,
        stage: Optional[int] = None,
        cluster: str = 'flexo',
        file_sh: str = 'run.sh',
        script: Optional[str] = None,
        dry_run: bool = True,
        submit: bool = False,
    ) -> pd.DataFrame:
        """
        Prepare (and optionally resubmit) cases whose last stage stalled, so
        that the stage is computed again on its own.

        Automates the manual restart of a case that hit the iteration limit
        without meeting the residual criterion. For every case:

        1. ``output_<stage>.h5`` becomes ``<ci_name>.h5`` (``ci_name`` is
           read from the run script, default ``'ci'``) and every other
           ``output_<stage>*`` file is deleted, so neither the run script
           nor ``CODAReader.parse_simulation_dirs`` consider the stage done.
        2. In the run script: ``custom_ci = True`` and
           ``stages = ['stage<stage>']``. Nothing else in the script is
           touched; see Notes for where the stage then starts from.
        3. In ``file_sh``: ``#SBATCH --nodelist=<node>``,
           ``--ntasks=<cpus>`` and ``--tasks-per-node=<cpus>``.
        4. With ``submit=True``, ``sbatch <file_sh>`` from the case folder.

        Each case is fully validated before anything is written; a case
        that fails a check is reported as ``'skipped'`` and left untouched.

        Parameters
        ----------
        cases : str, int, list or tuple
            Case folder names (keys of ``db.sim_metadata``) or case indices
            (resolved with ``case_per_idx``), or a list mixing both.
        cpus : int
            MPI processes for the restarted job (``--ntasks`` and
            ``--tasks-per-node``).
        node : str
            Node for ``#SBATCH --nodelist``, e.g. ``'n002'``.
        stage : int or None
            Stage to restart. Default ``metadata['num_stages'] - 1`` (the
            last one).
        cluster : str
            Slurm cluster the cases run on, compared with the
            ``ClusterName`` of the cluster this method runs on. Default
            ``'flexo'``. See Notes.
        file_sh : str
            Slurm script inside each case folder. Default ``'run.sh'``.
        script : str or None
            Python run script inside each case folder. Default: the ``.py``
            file launched by the ``mpiexec``/``mpirun``/``srun`` line of
            ``file_sh``.
        dry_run : bool
            If True (default), only report what would change.
        submit : bool
            If True, submit each restarted case with ``sbatch``. Requires
            ``dry_run=False``. Default False.

        Returns
        -------
        pd.DataFrame
            One row per case with columns ``folder``, ``status``
            (``'planned'``, ``'restarted'``, ``'submitted'``, ``'skipped'``
            or ``'failed'``), ``reason``, ``changes`` (list of str) and
            ``job_id``.

        Raises
        ------
        RuntimeError
            If ``dry_run=False`` and this is not ``cluster`` (nothing is
            modified), or if ``squeue``/``sinfo`` fail on ``cluster``.
        ValueError
            If ``submit=True`` with ``dry_run=True``, if ``cpus`` < 1, or if
            ``node`` does not exist on ``cluster`` or has fewer than
            ``cpus`` CPUs.

        Side-effects
        ------------
        With ``dry_run=False``, per restarted case: rewrites the run script
        and ``file_sh``, renames ``output_<stage>.h5`` to ``<ci_name>.h5``
        (overwriting a previous one), **deletes** every other
        ``output_<stage>*`` file (Slurm ``-out.txt``/``-err.txt`` logs are
        kept) and drops ``<stage>`` from
        ``db.sim_metadata[folder]['stages']``. ``db.df_state`` is not
        rebuilt; call ``parse_simulation_dirs()`` for that.

        Notes
        -----
        **Cluster check.** A case must never be modified while one of its
        jobs is pending or running, and that is checked with ``squeue``.
        From a different cluster ``squeue`` does not fail: it answers with
        *that* cluster's queue, so the check would always pass. The method
        therefore compares ``cluster`` with the current ``ClusterName``
        (``scontrol show config``; the short hostname if ``scontrol`` is
        missing). Off-cluster, only ``dry_run=True`` is allowed and it
        warns that the job check was not done.

        **Where the restarted stage starts.** That is decided by the run
        script, not by this method. With ``run_sst_v5.py`` as generated for
        the TIFON datasets, ``if not isNotFirstStage:`` re-initialises the
        first stage of ``stages`` to free stream even when ``custom_ci`` is
        set, so the stage restarts from **free stream** and
        ``<ci_name>.h5`` only provides the mesh (``ImportMeshHDF5``). Its
        relative residual criterion is then measured against the
        free-stream residual and is met almost at once, so in practice the
        Cauchy criterion on the integral forces decides when it stops. This
        reproduces the manual restart procedure. To really continue from
        the saved state, the script needs
        ``if not isNotFirstStage and not custom_ci:`` and
        ``not custom_ci`` instead of ``~custom_ci`` (``~`` on a bool is -1
        or -2, always true).

        Examples
        --------
        Review the changes, then apply and submit (on the right cluster)::

            plan = db.residuals.restart_stalled_cases(
                ['aoa_0.3347_mach_0.8863', 'aoa_0.7253_mach_0.8678'],
                cpus=40, node='n002',
            )
            print(plan[['folder', 'status', 'reason']])

            done = db.residuals.restart_stalled_cases(
                plan.loc[plan['status'] == 'planned', 'folder'].tolist(),
                cpus=40, node='n002', dry_run=False, submit=True,
            )
        """
        if submit and dry_run:
            raise ValueError("submit=True requires dry_run=False.")
        cpus = int(cpus)
        if cpus < 1:
            raise ValueError(f"cpus must be >= 1 (got {cpus}).")
        if stage is None:
            stage = int(self.db.metadata['num_stages']) - 1
        stage = int(stage)

        if isinstance(cases, (str, int, np.integer)):
            cases = [cases]
        folders: list = []
        for case in cases:
            if isinstance(case, (int, np.integer)):
                folder = self._resolve_case_name(case_idx=int(case))
            else:
                folder = self._resolve_case_name(case_name=case)
            if folder not in folders:
                folders.append(folder)

        # ── Cluster check: before any Slurm query or file change ─────────────
        current = _current_slurm_cluster()
        if current is None or current.lower() != cluster.lower():
            msg = (
                f"This is cluster '{current}', but the cases run on "
                f"'{cluster}': its Slurm queue cannot be queried from here."
            )
            if not dry_run:
                raise RuntimeError(
                    f"{msg} Nothing was modified; run this method on "
                    f"'{cluster}' to restart the cases."
                )
            warnings.warn(
                f"{msg} Dry run only: whether a case still has a pending or "
                "running job was NOT checked, nor were 'node' and 'cpus'.",
                UserWarning,
            )
            active = None
        else:
            node_cpus = _slurm_node_cpus()
            if node not in node_cpus:
                raise ValueError(
                    f"Node '{node}' does not exist on cluster '{cluster}'. "
                    f"Available: {sorted(node_cpus)}."
                )
            if cpus > node_cpus[node]:
                raise ValueError(
                    f"Node '{node}' has {node_cpus[node]} CPUs; "
                    f"cpus={cpus} would never be scheduled."
                )
            active = _slurm_active_workdirs()

        rows = []
        for folder in folders:
            row = {'folder': folder, 'status': 'skipped', 'reason': '',
                   'changes': [], 'job_id': None}
            try:
                plan = self._prepare_restart(
                    folder, stage, cpus, node, file_sh, script, active
                )
            except (ValueError, FileNotFoundError) as exc:
                row['reason'] = str(exc)
                log.info("Skipping '%s': %s", folder, exc)
                rows.append(row)
                continue

            row['changes'] = plan['changes']
            if dry_run:
                row['status'] = 'planned'
                rows.append(row)
                continue

            try:
                self._apply_restart(plan, stage)
            except OSError as exc:
                row['status'] = 'failed'
                row['reason'] = f"while applying the restart: {exc}"
                log.warning("Restart of '%s' failed: %s", folder, exc)
                rows.append(row)
                continue
            row['status'] = 'restarted'
            log.info("Restarted '%s' (stage %s).", folder, stage)

            if submit:
                try:
                    row['job_id'] = _sbatch(plan['case_dir'], file_sh)
                    row['status'] = 'submitted'
                    log.info("Submitted '%s': job %s.", folder, row['job_id'])
                except (OSError, subprocess.SubprocessError) as exc:
                    stderr = (getattr(exc, 'stderr', '') or '').strip()
                    row['reason'] = f"sbatch failed: {exc} {stderr}".strip()
                    log.warning("sbatch failed for '%s': %s", folder, exc)
            rows.append(row)

        return pd.DataFrame(
            rows, columns=['folder', 'status', 'reason', 'changes', 'job_id']
        )

    def _prepare_restart(
        self,
        folder: str,
        stage: int,
        cpus: int,
        node: str,
        file_sh: str,
        script: Optional[str],
        active: Optional[dict],
    ) -> dict:
        """Validate one case and build, in memory, everything the restart
        writes. Raises ``ValueError``/``FileNotFoundError`` with the reason
        the case must be skipped; nothing on disk is touched."""
        sim = self.db.sim_metadata.get(folder)
        if sim is None:
            raise ValueError(
                "not in db.sim_metadata; run parse_simulation_dirs() first."
            )
        case_dir = sim['path']

        h5_path = os.path.join(case_dir, f'output_{stage}.h5')
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(
                f"output_{stage}.h5 not found: stage {stage} has no state "
                "to restart from."
            )
        if active is not None:
            jobs = active.get(os.path.realpath(case_dir))
            if jobs:
                raise ValueError(
                    "Slurm job(s) still pending or running in this folder: "
                    + ", ".join(f"{j} ({st})" for j, st in jobs)
                )

        sh_path = os.path.join(case_dir, file_sh)
        if not os.path.isfile(sh_path):
            raise FileNotFoundError(f"{file_sh} not found.")
        with open(sh_path) as fh:
            sh_text = fh.read()

        script_name = script or _script_from_sh(sh_text, file_sh)
        py_path = os.path.join(case_dir, script_name)
        if not os.path.isfile(py_path):
            raise FileNotFoundError(f"{script_name} not found.")
        with open(py_path) as fh:
            py_text = fh.read()

        new_py, py_changes, ci_name = _restart_script_text(py_text, stage)

        new_sh, sh_changes = sh_text, []
        for option, value in (('nodelist', node),
                              ('ntasks', cpus),
                              ('tasks-per-node', cpus)):
            new_sh, change = _set_sbatch_option(new_sh, option, value)
            if change:
                sh_changes.append(change)

        ci_path = os.path.join(case_dir, f'{ci_name}.h5')
        to_delete = sorted(
            f for f in os.listdir(case_dir)
            if _output_stage(f) == stage and f != f'output_{stage}.h5'
        )
        changes = (
            [f"{script_name}: {c}" for c in py_changes]
            + [f"{file_sh}: {c}" for c in sh_changes]
            + [f"output_{stage}.h5 -> {ci_name}.h5"
               + (" (overwrites the existing one)"
                  if os.path.exists(ci_path) else "")]
            + ([f"delete: {', '.join(to_delete)}"] if to_delete else [])
        )
        return {
            'folder': folder, 'case_dir': case_dir,
            'py_path': py_path, 'new_py': new_py,
            'py_changed': bool(py_changes),
            'sh_path': sh_path, 'new_sh': new_sh,
            'sh_changed': bool(sh_changes),
            'h5_path': h5_path, 'ci_path': ci_path, 'to_delete': to_delete,
            'changes': changes,
        }

    def _apply_restart(self, plan: dict, stage: int) -> None:
        """Write what ``_prepare_restart`` built and move/delete the stage
        outputs. Scripts are replaced atomically (temp file + rename)."""
        for changed, path, text in (
            (plan['py_changed'], plan['py_path'], plan['new_py']),
            (plan['sh_changed'], plan['sh_path'], plan['new_sh']),
        ):
            if changed:
                tmp = path + '.restart_tmp'
                with open(tmp, 'w') as fh:
                    fh.write(text)
                os.chmod(tmp, os.stat(path).st_mode)
                os.replace(tmp, path)

        os.replace(plan['h5_path'], plan['ci_path'])
        for fname in plan['to_delete']:
            os.remove(os.path.join(plan['case_dir'], fname))
        self.db.sim_metadata[plan['folder']]['stages'].pop(stage, None)

    # =========================================================================
    # Live monitor
    # =========================================================================

    def monitor_cases(
        self,
        cases=None,
        subset: Optional[str] = None,
        query: Optional[str] = None,
        port: int = 8765,
        interval: float = 5.0,
        stale_after: float = 900.0,
        idle_timeout: float = 1800.0,
        background: bool = True,
        title: Optional[str] = None,
    ):
        """
        Open a live web monitor that follows a list of cases while they run.

        Unlike ``plot_state_calculation``, which draws every case once, the
        page refreshes itself every ``interval`` seconds and shows one case
        at a time (with a table of all of them a key away): status, stage,
        iteration against the limit, time per iteration, and each stop
        criterion against its threshold. Its plot stacks the residual
        reductions, every integral (``CoefLift``, ``CoefDrag``…) with its
        Cauchy window, and the CFL.

        Everything comes from the Slurm ``-out.txt`` log, the only file
        CODA writes while it runs (the ``.dat`` monitors appear at the end
        of each stage), so cases on another cluster sharing the file
        system can be followed too. See ``FotR.solver_tools.coda_log``.

        Give the cases with exactly one of ``cases``, ``subset`` or
        ``query``.

        Parameters
        ----------
        cases : int, str, list, pd.DataFrame or 'all'
            Case indices (``case_idx``, resolved with ``case_per_idx``),
            folder names (keys of ``db.sim_metadata``) or a list mixing
            both; a DataFrame with a ``folder`` (or ``case_idx``) column,
            such as a filtered ``df_cases``; or ``'all'``.
        subset : str or None
            Name of a subset defined with ``define_subset``.
        query : str or None
            ``pandas.DataFrame.query`` expression on
            ``metadata['df_cases']``, e.g. ``'mach > 0.78 and aoa < 2'``.
        port : int
            First port to try on 127.0.0.1; the next free one is used if
            it is taken. Default 8765.
        interval : float
            Seconds between refreshes. Default 5.
        stale_after : float
            Seconds without log writes after which a running case is shown
            as quiet. Default 900.
        idle_timeout : float
            Seconds without the page open after which the server stops by
            itself; 0 never stops. Default 1800.
        background : bool
            If True (default), run the server in a separate process and
            return at once: it does not block the notebook and outlives
            it. If False, serve from this process until Ctrl-C.
        title : str or None
            Page title. Default: the folder name of ``db.root_dir``.

        Returns
        -------
        MonitorHandle or None
            With ``background=True``: ``url``, ``pid``, ``alive()`` and
            ``stop()``. None otherwise.

        Raises
        ------
        ValueError
            If not exactly one of ``cases``, ``subset`` and ``query`` is
            given, if a folder is not in ``db.sim_metadata``, or if no
            case is left.

        Notes
        -----
        The server listens on 127.0.0.1 of the machine running Python.
        From VS Code with Remote-SSH the port is forwarded automatically
        (check the *Ports* panel); for a window of its own, open the URL
        with ``brave --app=<url>`` (or chrome/chromium) on your computer.
        Plotly is loaded from cdnjs, so that browser needs internet.

        Statuses: ``pending`` (no log yet), ``queued`` (the last log
        ended and the ``output_<stage>.h5`` of its last stage became
        ``ci.h5``, as ``restart_stalled_cases`` leaves it), ``starting``,
        ``running``, ``quiet``, ``finished``, ``limit`` (ended at the
        iteration limit), ``diverged`` (NaN), ``cancelled`` and ``error``
        (non-empty ``-err.txt``).

        The Cauchy check is ``max |x_i - x_k|`` over the current value and
        the ``number of samples`` before it. It was deduced from the logs,
        and reproduces CODA's own ``Cauchy convergence achieved`` message
        at every iteration of the rans5 logs.

        Examples
        --------
        By case index, folder name, or a query on ``df_cases``::

            mon = db.residuals.monitor_cases([219, 228, 243])
            mon = db.residuals.monitor_cases(['aoa_4.3363_mach_0.7247'])
            mon = db.residuals.monitor_cases(query='mach > 0.78')
            mon.url      # 'http://127.0.0.1:8765/'
            mon.stop()
        """
        from ...solver_tools import coda_monitor   # deferred: stdlib only

        folders = self._monitor_folders(cases, subset, query)
        entries = [
            {'path': self.db.sim_metadata[f]['path'], 'label': f,
             'info': self._monitor_info(f)}
            for f in folders
        ]
        if title is None:
            title = os.path.basename(os.path.normpath(self.db.root_dir))
        options = dict(title=title, port=port, interval=interval,
                       stale_after=stale_after, idle_timeout=idle_timeout)
        if background:
            handle = coda_monitor.launch(entries, **options)
            log.info("CODA monitor at %s (pid %s).", handle.url, handle.pid)
            return handle
        server = coda_monitor.MonitorServer(entries, **options)
        print(f"CODA monitor: {server.url}  (Ctrl-C to stop)", flush=True)
        server.serve_forever()
        return None

    def _monitor_folders(self, cases, subset, query) -> list:
        """Folder names selected by ``monitor_cases``, in the order given."""
        given = [x is not None for x in (cases, subset, query)]
        if sum(given) != 1:
            raise ValueError(
                "Give exactly one of 'cases', 'subset' or 'query'."
            )
        df_cases = self.db.metadata.get('df_cases')

        if subset is not None or query is not None:
            if df_cases is None or 'folder' not in df_cases.columns:
                raise ValueError(
                    "metadata['df_cases'] has no 'folder' column; run "
                    "parse_simulation_dirs() first."
                )
            if subset is not None:
                rows = df_cases.iloc[self.db.reader.get_subset(subset)]
            else:
                rows = df_cases.query(query)
            items = list(rows['folder'])
        elif isinstance(cases, str) and cases.lower() == 'all':
            items = [f for f in (df_cases['folder'] if df_cases is not None
                                 and 'folder' in df_cases.columns else [])
                     if isinstance(f, str)]
            items += [f for f in self.db.sim_metadata if f not in items]
        elif isinstance(cases, pd.DataFrame):
            if 'folder' in cases.columns:
                items = list(cases['folder'])
            elif 'case_idx' in cases.columns:
                items = [int(i) for i in cases['case_idx']]
            else:
                raise ValueError(
                    "A DataFrame of cases needs a 'folder' or a 'case_idx' "
                    "column."
                )
        elif isinstance(cases, (str, int, np.integer)):
            items = [cases]
        else:
            items = list(cases)

        folders, skipped = [], []
        for item in items:
            if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                folder = self._resolve_case_name(case_idx=int(item))
            elif isinstance(item, str):
                folder = item
            else:
                skipped.append(item)
                continue
            if folder not in folders:
                folders.append(folder)
        if skipped:
            warnings.warn(
                f"{len(skipped)} case(s) without a folder were skipped "
                "(not generated yet?).",
                UserWarning,
            )
        unknown = [f for f in folders if f not in self.db.sim_metadata]
        if unknown:
            raise ValueError(
                f"Not in db.sim_metadata: {unknown}. Folder names must match "
                "outputs/; run parse_simulation_dirs() if they are new."
            )
        if not folders:
            raise ValueError("No case selected.")
        return folders

    def _monitor_info(self, folder: str) -> dict:
        """``case_idx`` and design variables of a case, for the page."""
        info = {}
        df_cases = self.db.metadata.get('df_cases')
        if df_cases is not None and 'folder' in df_cases.columns:
            match = df_cases.index[df_cases['folder'] == folder]
            if len(match):
                row = df_cases.loc[match[0]]
                info['case_idx'] = int(row['case_idx'] if 'case_idx' in row
                                       else match[0])
        sim = self.db.sim_metadata[folder]
        for var in self.db.metadata.get('design_vars', []):
            if var in sim:
                info[var] = float(sim[var])
        return info

    # =========================================================================
    # Visualisation
    # =========================================================================

    def plot_residuals_from_case(
        self,
        case_name: Optional[str] = None,
        case_idx: Optional[int] = None,
        stage: Union[list, tuple, str] = 'all',
        mode: Literal['absolute', 'norm', 'scaled'] = 'scaled',
        save_dir: Union[str, None] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Plot scaled (or normalised / absolute) residuals for a single case,
        with a CFL panel below.

        Parameters
        ----------
        case_name : str or None
            Folder name in ``db.sim_metadata``.
        case_idx : int or None
            ``df_state`` row index (used when ``case_name`` is None).
        stage : list, tuple or 'all'
            Stages to include. Default ``'all'``.
        mode : 'absolute', 'norm' or 'scaled'
            Which residual column suffix to plot. Default ``'scaled'``.
        save_dir : str or None
            If provided, saves the figure to this directory instead of
            displaying it.
        verbose : bool
            Print intermediate information. Default False.
        **kwargs
            Extra keyword arguments forwarded to ``plt.subplots``
            (e.g. ``figsize``).

        Examples
        --------
        ::

            db.residuals.plot_residuals_from_case(case_idx=5, mode='norm')
        """
        case_name = self._resolve_case_name(
            case_idx=case_idx, case_name=case_name
        )

        stages = (
            list(self.db.sim_metadata[case_name]['stages'].keys())
            if stage == 'all' else stage
        )
        df_res = self.get_df_residuals_from_case(
            case_name=case_name, stage=stages, verbose=verbose
        )

        _, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        columns = [
            c for c in df_res.columns
            if 'Residual' in c
            and 'MomentumYResidual' not in c
            and mode in c
        ]
        colors = self._cycled_colors(len(columns))
        for ycol, color in zip(columns, colors):
            df_res.plot(
                x='total_iterations', y=ycol, s=3,
                kind='scatter', ax=ax,
                label=ycol.replace(f"Residual_{mode}", ''),
                color=color, grid=True, logy=True,
            )

        ax.set(
            title=f'Case {case_name}',
            ylim=(1e-8, 1e2),
            ylabel=f'Residual ({mode})',
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)

        from mpl_toolkits.axes_grid1 import make_axes_locatable   # deferred
        divider = make_axes_locatable(ax)
        ax_cfl  = divider.append_axes("bottom", size="35%", pad=0.1, sharex=ax)
        ax_cfl.scatter(
            df_res['total_iterations'], df_res['CFL'],
            color='black', s=1.5,
        )
        ax_cfl.set(ylabel="CFL", xlabel="Iterations")
        ax_cfl.set_yscale('log')
        ax_cfl.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{case_name}_residuals.png")
            plt.savefig(path, bbox_inches='tight')
            log.info("Figure saved to %s", path)
        else:
            plt.show()

    def plot_all_final_residuals(
        self,
        save_dir: Union[str, None] = None,
        mode: Literal['absolute', 'norm', 'scaled'] = 'scaled',
        stage: Union[tuple, list, str] = 'all',
        only_finished: bool = False,
        print_non_converged: bool = False,
        activate_idx: bool = True,
        ncols: int = 2,
        lim_converged: float = 1e-5,
        xscale: Literal['auto', 'linear', 'log'] = 'auto',
        **kwargs,
    ) -> None:
        """
        Plot scatter maps of final residuals across all simulations in the
        design-variable space.

        Converged cases (all plotted residuals below ``lim_converged``) are
        shown with star markers; non-converged cases with circles.

        With a single varying design variable the plot collapses to one
        semilogy axes per stage: cases sorted along the variable, one
        curve per residual column, converged points as ``*`` and a
        dashed ``lim_converged`` line.  Each stage is drawn in its own
        figure (``residuals_all_cases_stage<s>.png``) so stages are
        never mixed; a single stage keeps the plain
        ``residuals_all_cases.png`` name.

        Parameters
        ----------
        save_dir : str or None
            If provided, saves the figure here. Default None (show).
        mode : 'absolute', 'norm' or 'scaled'
            Residual column suffix to plot. Default ``'scaled'``.
        stage : list, tuple or 'all'
            Stages to include. Default ``'all'``.
        only_finished : bool
            Only include completed simulations. Default False.
        print_non_converged : bool
            Print and save non-converged case details to
            ``metadata/non_converged_cases.csv``. Default False.
        activate_idx : bool
            Annotate each scatter point with its ``df_state`` row index.
            Default True.
        ncols : int
            Number of subplot columns. Default 2.
        xscale : 'auto', 'linear' or 'log'
            Horizontal scale of the single-design-variable view.
            ``'auto'`` (default) uses a logarithmic axis when the
            variable is positive and spans more than a decade, which is
            what a mesh-refinement family looks like; a linear axis
            would pile every fine mesh on the left edge.
        lim_converged : float
            Residual threshold for convergence classification. Default 1e-5.
        **kwargs
            Extra keyword arguments: ``cmap`` (colormap name, default
            ``'summer'``).

        Examples
        --------
        ::

            db.residuals.plot_all_final_residuals(
                mode='norm',
                only_finished=True,
                lim_converged=1e-4,
                save_dir='/output/plots/',
            )
        """
        df_finals = self.get_all_final_residuals(
            verbose=kwargs.get('verbose', False), stage=stage,
            only_finished=only_finished, load_in_metadata=False,
        )
        if df_finals.empty:
            warnings.warn("No data to plot.", UserWarning)
            return

        columns = [
            c for c in df_finals.columns
            if 'Residual' in c
            and 'MomentumYResidual' not in c
            and 'TurbulentSANuTilde' not in c
            and mode in c
        ]
        if not columns:
            warnings.warn(
                f"No residual columns found for mode '{mode}'.",
                UserWarning,
            )
            return

        converged_mask = (
            df_finals[columns].lt(lim_converged)
        ).all(axis=1).to_numpy()
        norm = mcolors.LogNorm(vmin=lim_converged, vmax=1e0)
        cmap_name = kwargs.get('cmap', 'summer')

        dvf = [
            v for v in self.db.metadata['design_vars']
            if df_finals[v].nunique() > 1
        ]

        if not dvf:
            warnings.warn(
                "Design variables are constant; nothing to scatter.",
                UserWarning,
            )
            return

        if len(dvf) == 1:
            if 'stage' in df_finals.columns:
                stages_present = sorted(df_finals['stage'].unique().tolist())
            else:
                stages_present = ['all']
            for stage_here in stages_present:
                if stage_here == 'all':
                    df_stage = df_finals
                    mask_stage = np.ones(len(df_finals), dtype=bool)
                    suffix = ''
                else:
                    mask_stage = (
                        df_finals['stage'].to_numpy() == stage_here
                    )
                    df_stage = df_finals.loc[mask_stage]
                    suffix = (
                        '' if len(stages_present) == 1
                        else f'_stage{stage_here}'
                    )
                self._plot_final_residuals_1d(
                    df_finals=df_stage,
                    columns=columns,
                    dvf=dvf,
                    converged_mask=np.asarray(converged_mask)[mask_stage],
                    stage=stage_here,
                    mode=mode,
                    lim_converged=lim_converged,
                    activate_idx=activate_idx,
                    save_dir=save_dir,
                    filename_suffix=suffix,
                    xscale=xscale,
                    **kwargs,
                )
            return

        nrows = int(np.ceil(len(columns) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 5 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()

        if len(dvf) == 2:
            for i, col in enumerate(columns):
                x, y, c = (df_finals[dvf[0]], df_finals[dvf[1]],
                            df_finals[col])
                sc_nc = axes[i].scatter(
                    x[~converged_mask], y[~converged_mask],
                    c=c[~converged_mask], cmap=cmap_name, norm=norm,
                    s=60, edgecolor='k', label='Non-converged',
                )
                axes[i].scatter(
                    x[converged_mask], y[converged_mask],
                    c=c[converged_mask], cmap=cmap_name, norm=norm,
                    s=60, marker='*', linewidth=1.5, label='Converged',
                )
                if activate_idx:
                    for p in df_finals[dvf].values:
                        matches = np.where(
                            (self.db.df_state.iloc[:, 0] == p[0]) &
                            (self.db.df_state.iloc[:, 1] == p[1])
                        )[0]
                        if matches.size > 0:
                            axes[i].annotate(
                                f"{matches[0]}", (p[0], p[1]),
                                textcoords="offset points",
                                xytext=(0, 7), ha='center', fontsize=8,
                            )
                axes[i].set(title=col, xlabel=dvf[0], ylabel=dvf[1])
                fig.colorbar(sc_nc, ax=axes[i]).ax.set_title(
                    f'Residual stage {stage}'
                )

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',
                       frameon=False, ncols=2)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "residuals_all_cases.png"),
                dpi=150, bbox_inches='tight',
            )
        else:
            plt.show()

        if print_non_converged:
            log.info("Non-converged cases:")
            df_finals[columns][~converged_mask].to_csv(
                os.path.join(
                    self.db.root_dir, 'metadata', 'non_converged_cases.csv'
                ),
                index=False,
            )
            for i, row in df_finals[~converged_mask].iterrows():
                res_str = " ".join(f"{v:.2E}" for v in row[3:].values)
                log.info(
                    "  Case %s: %s  Residuals: %s",
                    i,
                    ", ".join(
                        f"{dvf[j]}={row[dvf[j]]:.4f}"
                        for j in range(len(dvf))
                    ),
                    res_str,
                )

    def plot_state_calculation(
        self,
        num_stages: int = 1,
        txt_from_end: int = 1,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        Plot one residual panel per simulation that has exactly
        ``num_stages`` completed stages, reading from the ``-out.txt`` log.

        Simulations are arranged in a two-column grid.  Each panel shows
        density, momentum and energy residuals on a log scale together with
        a CFL sub-panel below.

        Parameters
        ----------
        num_stages : int
            Number of completed stages required for a simulation to be
            included. Default 1.
        txt_from_end : int
            Which ``-out.txt`` file to read (from the end of the sorted
            list). Default 1 (last file).
        figsize : tuple or None
            Custom figure size. If None, uses ``(15, 6 × n_rows)``.

        Examples
        --------
        ::

            db.residuals.plot_state_calculation(num_stages=2)
        """
        data_to_plot = []
        files_with_ci = []
        for name, case in self.db.sim_metadata.items():
            if len(case['stages']) == num_stages:
                # comprobar si ese caso tiene condicion inicial en un ci.h5
                files_with_ci.append(
                    os.path.exists(os.path.join(case['path'], 'ci.h5'))
                )
                df = CODAResiduals.get_df_residuals_from_txt(
                    case_path=case['path'],
                    verbose=False,
                    txt_from_end=txt_from_end,
                )
                if df is not None:
                    data_to_plot.append((name, df))
                else:
                    warnings.warn(
                        f"Case '{name}' has not started yet.",
                        UserWarning,
                    )

        if not data_to_plot:
            warnings.warn(
                "No data found for the specified criteria.",
                UserWarning,
            )
            return

        ncols = 2 if len(data_to_plot) > 1 else 1
        nrows = (len(data_to_plot) + 1) // 2
        if figsize is None:
            figsize = (15, 6 * nrows)

        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax_flat = ax.flatten()

        for i, (name, df_txt) in enumerate(data_to_plot):
            cur = ax_flat[i]
            for res, color in zip(
                df_txt.columns.tolist()[2:], ['blue', 'orange', 'green']
            ):
                cur.scatter(df_txt['iters'], df_txt[res],
                            color=color, label=res, s=1.5)
            cur.set_yscale('log')
            cur.set(title=name, ylabel="Residuals")
            cur.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)
            cur.legend(loc='upper right', fontsize='small', markerscale=4)
            # añadir en la leyenda una casilla que diga si tiene ci o no (valor de file_with_ci correspondiente)
            cur.text(0.05, 0.05, f"CI: {'Yes' if files_with_ci[i] else 'No'}",
                     transform=cur.transAxes, fontsize=10, verticalalignment='top',
                        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            cur.tick_params(labelbottom=False)

            from mpl_toolkits.axes_grid1 import make_axes_locatable   # deferred
            divider = make_axes_locatable(cur)
            ax_cfl  = divider.append_axes(
                "bottom", size="35%", pad=0.1, sharex=cur
            )
            if 'cfl' in df_txt.columns:
                ax_cfl.scatter(df_txt['iters'], df_txt['cfl'],
                               color='black', s=1.5)
            ax_cfl.set(ylabel="CFL", xlabel="Iterations")
            ax_cfl.set_yscale('log')
            ax_cfl.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)

        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].axis('off')

        plt.tight_layout()
        plt.show()

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _plot_final_residuals_1d(
        self,
        df_finals: pd.DataFrame,
        columns: list,
        dvf: list,
        converged_mask,
        stage,
        mode: str,
        lim_converged: float,
        activate_idx: bool,
        save_dir,
        filename_suffix: str = '',
        xscale: Literal['auto', 'linear', 'log'] = 'auto',
        **kwargs,
    ) -> None:
        """Single-axes semilogy view of final residuals vs one design var.

        Cases are sorted by the varying design variable; each residual
        column is one curve, converged points wear ``*`` (the rest
        ``o``), and a dashed line marks ``lim_converged``.  ``NaN``
        values (e.g. pending cases) are masked per curve instead of
        breaking it.  Case-index annotations are skipped above 40 rows
        to keep the plot readable.

        ``xscale='auto'`` switches the horizontal axis to logarithmic
        when the design variable is strictly positive and spans more
        than a decade -- the usual shape of a mesh-refinement family,
        where a linear axis crushes every fine mesh against the origin.
        """
        order = np.argsort(df_finals[dvf[0]].values)
        x_all = df_finals[dvf[0]].values[order]
        conv_all = np.asarray(converged_mask)[order]

        fig, ax = plt.subplots(
            figsize=kwargs.get('figsize', (10, 6)),
            constrained_layout=True,
        )
        # Colour encodes the residual, marker encodes the convergence
        # state. Labelling both marker artists per column listed every
        # residual twice ("<name> (converged)" and "<name>
        # (non-converged)"), which is what made the legend duplicate.
        colors = self._cycled_colors(len(columns))
        var_handles = []
        any_converged = any_non_converged = False

        for col, color in zip(columns, colors):
            y_all = df_finals[col].values[order].astype(float)
            finite = np.isfinite(y_all)
            if not finite.any():
                continue
            label = col.replace(f"Residual_{mode}", "")
            xf, yf, cf = x_all[finite], y_all[finite], conv_all[finite]
            if len(xf) > 1:
                ax.semilogy(xf, yf, '-', color=color, linewidth=1.5)
            ax.semilogy(
                xf[cf], yf[cf], '*', color=color, markersize=10,
                markeredgecolor='k', markeredgewidth=0.5,
            )
            ax.semilogy(
                xf[~cf], yf[~cf], 'o', color=color, markersize=6,
                markeredgecolor='k',
            )
            any_converged = any_converged or bool(cf.any())
            any_non_converged = any_non_converged or bool((~cf).any())
            # A proxy, so the residual still gets its legend entry even
            # when it has a single point and therefore no line.
            var_handles.append(
                plt.Line2D([], [], color=color, linewidth=1.5, label=label)
            )

        ax.axhline(
            lim_converged, color='k', linestyle='--', linewidth=1.0,
            label=f'converged < {lim_converged:.0e}',
        )
        ax.set(
            title=f'Final residuals vs {dvf[0]} (stage {stage})',
            xlabel=dvf[0],
            ylabel=f"Residual ({mode})",
        )
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

        positive = x_all[np.isfinite(x_all) & (x_all > 0)]
        use_log = (
            xscale == 'log'
            or (
                xscale == 'auto'
                and len(positive) == len(x_all)
                and len(positive) > 1
                and positive.max() / positive.min() >= 10.0
            )
        )
        if use_log:
            ax.set_xscale('log')

        if activate_idx:
            if len(df_finals) <= 40:
                # Match on the design variable that actually VARIES. Using
                # df_state's first column silently compared the varying
                # variable against a different (often constant) one, so no
                # point ever matched and the annotations never appeared —
                # while the caption still promised them.
                state = self.db.df_state
                key = dvf[0] if dvf[0] in state.columns else state.columns[0]
                labels_by_value = {}
                for pos, value in enumerate(state[key].to_numpy()):
                    case_id = (
                        state['case_idx'].iloc[pos]
                        if 'case_idx' in state.columns else pos
                    )
                    if pd.notna(case_id):
                        labels_by_value.setdefault(float(value), int(case_id))

                ticks, labels = [], []
                for value in np.unique(x_all):
                    case_id = labels_by_value.get(float(value))
                    if case_id is None:
                        continue
                    ticks.append(value)
                    labels.append(str(case_id))

                if ticks:
                    # A secondary axis rather than annotations at
                    # y = 1.0: matplotlib then reserves the space, so
                    # the indices can never end up written across the
                    # title, and the labels move with the axis when the
                    # scale is logarithmic.
                    top = ax.secondary_xaxis('top')
                    top.set_xticks(ticks)
                    top.set_xticklabels(
                        labels, fontsize=7, rotation=90, color='0.35',
                    )
                    top.set_xlabel(
                        'case_idx', fontsize=7.5, color='0.35',
                    )
                    top.tick_params(axis='x', length=2, color='0.6')
            else:
                log.info(
                    "Skipping case annotations (%s rows).",
                    len(df_finals),
                )

        # One entry per residual (colour) + one per state actually
        # present (marker) + the threshold line.
        state_handles = []
        if any_converged:
            state_handles.append(plt.Line2D(
                [], [], color='0.3', marker='*', linestyle='none',
                markersize=10, markeredgecolor='k', markeredgewidth=0.5,
                label='converged',
            ))
        if any_non_converged:
            state_handles.append(plt.Line2D(
                [], [], color='0.3', marker='o', linestyle='none',
                markersize=6, markeredgecolor='k', label='not converged',
            ))
        threshold_handles, _ = ax.get_legend_handles_labels()

        handles = var_handles + state_handles + threshold_handles
        if handles:
            # 'outside' keeps the legend out of the data area: placed at
            # 'lower center' it sat on top of the curves.
            fig.legend(
                handles, [h.get_label() for h in handles],
                loc='outside lower center', frameon=False,
                ncols=min(3, len(handles)), fontsize='small',
            )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(
                    save_dir, f"residuals_all_cases{filename_suffix}.png"
                ),
                dpi=150, bbox_inches='tight',
            )
        else:
            plt.show()

    def _plot_convergence(
        self,
        result_mean: pd.DataFrame,
        result_std:  pd.DataFrame,
        columns:     list,
        mode:        str,
        **kwargs,
    ) -> None:
        """
        Internal dispatcher to 2-D scatter or 3-D Plotly surface plots.

        Parameters
        ----------
        result_mean : pd.DataFrame
        result_std  : pd.DataFrame
        columns     : list[str]
        mode        : '2D' or '3D'
        **kwargs    : figsize, width, height
        """
        param1, param2 = self.db.metadata['design_vars']

        if mode == '2D':
            fig, axes = plt.subplots(
                2, len(columns[2:]),
                figsize=kwargs.get('figsize', (5 * len(columns[2:]), 8)),
            )
            axes = np.atleast_2d(axes).T if len(columns[2:]) == 1 else axes
            for i, col in enumerate(columns[2:]):
                sc1 = axes[0, i].scatter(
                    result_mean[param1], result_mean[param2],
                    c=result_mean[col], cmap='viridis', s=100, edgecolors='k',
                )
                plt.colorbar(sc1, ax=axes[0, i], label=f'Mean {col}')
                axes[0, i].set(xlabel=param1, ylabel=param2,
                               title=f'Mean {col}')
                axes[0, i].grid(True, linestyle='--', linewidth=0.5)

                sc2 = axes[1, i].scatter(
                    result_std[param1], result_std[param2],
                    c=result_std[col], cmap='plasma', s=100, edgecolors='k',
                    norm=mcolors.LogNorm(),
                )
                plt.colorbar(sc2, ax=axes[1, i], label=f'Std {col}')
                axes[1, i].set(xlabel=param1, ylabel=param2,
                               title=f'Std {col}')
                axes[1, i].grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()

        elif mode == '3D':
            from scipy.interpolate import griddata
            for col in columns[2:]:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=result_mean[param1], y=result_mean[param2],
                    z=result_mean[col], mode='markers',
                    name=f'Mean {col}',
                    marker=dict(size=5, color='blue'), opacity=0.8,
                ))
                g1 = np.linspace(result_mean[param1].min(),
                                  result_mean[param1].max(), 50)
                g2 = np.linspace(result_mean[param2].min(),
                                  result_mean[param2].max(), 50)
                G1, G2 = np.meshgrid(g1, g2)
                for df_r, cs, name in [
                    (result_mean, 'Blues',  f'Mean {col}'),
                    (result_std,  'Reds',   f'Std {col}'),
                ]:
                    Z = griddata(
                        (df_r[param1], df_r[param2]),
                        df_r[col], (G1, G2), method='cubic',
                    )
                    fig.add_trace(go.Surface(
                        x=G1, y=G2, z=Z, colorscale=cs,
                        opacity=0.5, showscale=False, name=name,
                    ))
                    if df_r is result_std:
                        fig.add_trace(go.Scatter3d(
                            x=result_std[param1], y=result_std[param2],
                            z=result_std[col], mode='markers',
                            name=f'Std {col}',
                            marker=dict(size=5, color='red',
                                        symbol='diamond'),
                            opacity=0.8,
                        ))
                fig.update_layout(
                    title=f'Integral variable: {col}',
                    scene=dict(xaxis_title=param1,
                               yaxis_title=param2,
                               zaxis_title=col),
                    margin=dict(l=0, r=0, b=0, t=50),
                    width=kwargs.get('width', 1200),
                    height=kwargs.get('height', 800),
                )
                fig.show()


# =============================================================================
# Restart helpers (module level: pure text logic and thin Slurm wrappers)
# =============================================================================

def _output_stage(fname: str) -> Optional[int]:
    """Stage number of a CODA ``output_<n>...`` file, or None.

    Same rule as ``CODAReader.parse_simulation_dirs``, so ``output_1``
    never matches ``output_10``.
    """
    if not fname.startswith('output_'):
        return None
    parts = fname.split('_')
    if len(parts) < 2:
        return None
    raw = os.path.splitext(parts[1])[0]
    return int(raw) if raw.isdigit() else None


def _patch_line(
    text: str, pattern: str, new_line: str, label: str
) -> Tuple[str, Optional[str]]:
    """Replace the single unindented line matching ``pattern`` by
    ``new_line``.

    Only top-level lines count (an indented one never matches), and a
    trailing ``# comment`` is allowed in the match and dropped.

    Returns ``(new_text, change)``, ``change`` being None when nothing
    changed. Raises ``ValueError`` unless exactly one line matches.
    """
    found = list(re.finditer(
        rf'^{pattern}[ \t]*(?:#.*)?$', text, re.MULTILINE
    ))
    if len(found) != 1:
        raise ValueError(
            f"{label}: expected exactly one matching line in the run "
            f"script, found {len(found)}."
        )
    match = found[0]
    old = match.group(0)
    if old == new_line:
        return text, None
    return (text[:match.start()] + new_line + text[match.end():],
            f"{old.strip()} -> {new_line}")


def _restart_script_text(text: str, stage: int) -> Tuple[str, list, str]:
    """Return ``(new_text, changes, ci_name)`` for a CODA run script that
    must run only ``stage``, reading its mesh from ``<ci_name>.h5``.

    Only ``custom_ci`` and ``stages`` are changed; where the stage starts
    from is left to the script (see ``restart_stalled_cases``, Notes).
    Raises ``ValueError`` if a line to patch is missing or ambiguous.
    """
    m = re.search(r"^ci_name\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    ci_name = m.group(1) if m else 'ci'

    changes = []
    for pattern, new_line, label in (
        (r'custom_ci\s*=\s*(?:True|False)', 'custom_ci = True', 'custom_ci'),
        (r'stages\s*=[^\n]*?', f"stages = ['stage{stage}']", 'stages'),
    ):
        text, change = _patch_line(text, pattern, new_line, label)
        if change:
            changes.append(change)
    return text, changes, ci_name


def _set_sbatch_option(
    text: str, option: str, value
) -> Tuple[str, Optional[str]]:
    """Set the value of the single active ``#SBATCH --<option>=`` line.

    Commented-out lines (``##SBATCH``) never match, and ``--ntasks=`` does
    not match ``--ntasks-per-node=``. Returns ``(new_text, change)``;
    raises ``ValueError`` unless exactly one line sets the option.
    """
    found = list(re.finditer(
        rf'^([ \t]*#SBATCH[ \t]+--{re.escape(option)}=)(\S*)',
        text, re.MULTILINE,
    ))
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one '#SBATCH --{option}=' line, "
            f"found {len(found)}."
        )
    match = found[0]
    old = match.group(2)
    if old == str(value):
        return text, None
    return (text[:match.start(2)] + str(value) + text[match.end(2):],
            f"--{option}: {old} -> {value}")


def _script_from_sh(text: str, file_sh: str) -> str:
    """Name of the ``.py`` launched by the active mpiexec/mpirun/srun line
    of a Slurm script. Raises ``ValueError`` unless there is exactly one."""
    found = re.findall(
        r'^[ \t]*(?:mpiexec|mpirun|srun)\b[^\n#]*?\bpython3?[ \t]+(\S+\.py)\b',
        text, re.MULTILINE,
    )
    if len(set(found)) != 1:
        raise ValueError(
            f"could not tell the run script from {file_sh} "
            f"(found {found or 'none'}); pass script=... explicitly."
        )
    return found[0]


def _current_slurm_cluster() -> Optional[str]:
    """``ClusterName`` of the Slurm cluster this process runs on.

    Falls back to the short hostname when ``scontrol`` is unavailable and
    returns None when neither gives a name. Works on compute nodes too,
    where the hostname is the node's, not the cluster's.
    """
    try:
        out = subprocess.run(
            ['scontrol', 'show', 'config'],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        m = re.search(r'^\s*ClusterName\s*=\s*(\S+)', out, re.MULTILINE)
        if m:
            return m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    host = socket.gethostname().split('.')[0]
    return host or None


def _slurm_active_workdirs() -> dict:
    """Map ``realpath(WorkDir) -> [(job_id, state), ...]`` for every job of
    the current user known to ``squeue`` (pending, running, ...).

    Raises ``RuntimeError`` if ``squeue`` cannot be run: a restart must
    never proceed without knowing whether the case is still running.
    """
    try:
        out = subprocess.run(
            ['squeue', '-u', getpass.getuser(), '-h', '-o', '%i|%T|%Z'],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"squeue failed ({exc}); cannot tell whether the cases are "
            "still running, so nothing is restarted."
        ) from exc
    active: dict = {}
    for line in out.splitlines():
        parts = line.strip().split('|', 2)
        if len(parts) == 3 and parts[2]:
            active.setdefault(os.path.realpath(parts[2]), []).append(
                (parts[0], parts[1])
            )
    return active


def _slurm_node_cpus() -> dict:
    """Map node name -> CPUs, from ``sinfo``. Raises ``RuntimeError`` if
    ``sinfo`` cannot be run."""
    try:
        out = subprocess.run(
            ['sinfo', '-h', '-N', '-o', '%N %c'],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"sinfo failed ({exc}); cannot validate 'node' and 'cpus'."
        ) from exc
    nodes: dict = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            nodes[parts[0]] = max(nodes.get(parts[0], 0), int(parts[1]))
    return nodes


def _sbatch(case_dir: str, file_sh: str) -> str:
    """Submit ``file_sh`` from ``case_dir`` (Slurm resolves the relative
    ``--output``/``--error`` paths there) and return the job id."""
    result = subprocess.run(
        ['sbatch', file_sh], cwd=case_dir,
        capture_output=True, text=True, check=True, timeout=120,
    )
    m = re.search(r'(\d+)', result.stdout)
    return m.group(1) if m else result.stdout.strip()
