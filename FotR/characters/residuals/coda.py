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
import warnings
from typing import Literal, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go

from ..sam import SAM
from .base import BaseResiduals

if TYPE_CHECKING:
    from ..frodo import FRODO


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

    def __init__(self, db: 'FRODO'):
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
        """
        folder_fmt = self.db.metadata.get('folder_fmt', '')
        pattern    = SAM.Backpack.pattern_pocket.FilenamePattern.from_template(
            folder_fmt, numeric=True
        ).compiled

        df_all  = []
        df_one  = None

        for folder in self.db.sim_metadata:
            if not re.match(pattern, folder):
                continue

            params_float = (
                self.db.metadata['df_cases'][
                    self.db.metadata['design_vars']
                ][
                    self.db.metadata['df_cases']['folder'] == folder
                ].values.squeeze().tolist()
            )
            stages_done = len(self.db.sim_metadata[folder]['stages'])

            if only_finished and stages_done < self.db.metadata['num_stages']:
                if verbose:
                    print(
                        f"Skipping '{folder}': "
                        f"{stages_done}/{self.db.metadata['num_stages']} stages."
                    )
                continue

            df_one = self.get_df_residuals_from_case(
                case_name=folder, stage=stage
            )

            if df_one is None or df_one.empty:
                if verbose:
                    print(f"No residual data for '{folder}'.")
                res = np.full((1, 26), np.nan)
            else:
                res = df_one.tail(1).values.reshape(1, -1)
            # print(res, res.shape, np.asarray(params_float))
            fila = np.concatenate(
                (res, np.atleast_2d(np.asarray(params_float, dtype=np.float64))),
                axis=1, dtype=np.float64,
            )
            df_all.append(fila)

        if not df_all or df_one is None:
            warnings.warn(
                "No residual data found. Returning empty DataFrame.",
                UserWarning,
            )
            return pd.DataFrame()

        names    = list(df_one.columns) + self.db.metadata['design_vars']
        df_final = pd.DataFrame(np.vstack(df_all), columns=names)

        if load_in_metadata:
            os.makedirs(
                os.path.join(self.db.root_dir, 'metadata'), exist_ok=True
            )
            df_final.to_csv(
                os.path.join(
                    self.db.root_dir, 'metadata', 'all_final_residuals.csv'
                ),
                index=False,
            )

        return df_final

    # =========================================================================
    # Case-level residual extraction
    # =========================================================================

    def get_df_residuals_from_case(
        self,
        case_name: str = None,
        case_idx: Union[int, None] = None,
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
                print(
                    f"[INFO] Stage {s}: {len(df_stage)} iterations  |  "
                    f"vars: {res_cols}"
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
    ) -> pd.DataFrame:
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
        files = SAM.Backpack.pattern_pocket.find_files(case_path, endswith="-out.txt", verbose=False)
        if not files:
            if verbose:
                print(f"WARNING: No -out.txt file found in {case_path}.")
            return None

        files = [files[-txt_from_end]] if isinstance(files, list) else [files]

        regex = re.compile(
            r"Iteration (\d+):\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
            r"\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
        )

        list_df = []
        for file in files:
            if verbose:
                print(f"Reading {file}")
            with open(file, 'r') as fh:
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

        Returns
        -------
        pd.DataFrame
            ``db.df_state`` augmented with columns
            ``<var>_mean_stage<s>`` and ``<var>_var_stage<s>`` for each
            variable and stage combination.

        Examples
        --------
        ::

            df = db.residuals.get_df_metrics(
                var_metrics=['CoefLift', 'CoefDrag'],
                iter_var=500,
                save=True,
            )
            print(df[['AoA', 'Mach', 'CoefLift_mean_stage0']].head())
        """
        db = self.db
        if isinstance(var_metrics, str):
            var_metrics = [var_metrics]

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
            print(df_finals.columns.to_list()[-3:], df_post.columns.to_list(), [v.lower() for v in design_vars])
            
            df_post    = df_post.merge(df_finals, on=[v.lower() for v in design_vars], how="left")

        # Fill integral metric columns
        for irow in range(len(db.df_state)):
            case_name   = db.reader.case_per_idx(irow)
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
                    df_post.loc[irow, f"{v}_mean_stage{stage}"] = df_tail[v].mean()
                    df_post.loc[irow, f"{v}_var_stage{stage}"]  = df_tail[v].var()

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
    ) -> tuple:
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
            print(
                "WARNING: only_converged requires only_finished=True. "
                "Enabling it automatically."
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
                    print(f"Skipping '{folder_name}': not finished.")
                continue

            # ── Resolve exact design-variable values for this folder ───────
            # Looked up directly from df_cases via the precise 'folder'
            # column instead of being re-parsed (and rounded) from the
            # folder name itself — see the "Notes" section in the
            # docstring for why this matters.
            case_row = df_cases.loc[df_cases['folder'] == folder_name]
            if case_row.empty:
                if verbose:
                    print(
                        f"Skipping '{folder_name}': "
                        "no matching entry found in df_cases."
                    )
                continue
            valores = case_row[design_vars].iloc[0].astype(float).tolist()

            mask = np.ones(len(df_filtered), dtype=bool)
            for val, var in zip(valores, design_vars):
                mask &= np.isclose(df_filtered[var].values, val)

            if not mask.any():
                if verbose:
                    print(
                        f"Skipping '{folder_name}': "
                        "does not meet convergence criteria."
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
    # Visualisation
    # =========================================================================

    def plot_residuals_from_case(
        self,
        case_name: str = None,
        case_idx: Union[int, None] = None,
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
        if case_name is None:
            if case_idx is None:
                raise ValueError("Provide either case_name or case_idx.")
            case_name = self.db.reader.case_per_idx(case_idx)

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
        colors = cm.tab10.colors[:len(columns)]
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
            print(f"Figure saved to {path}")
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
        **kwargs,
    ) -> None:
        """
        Plot scatter maps of final residuals across all simulations in the
        design-variable space.

        Converged cases (all plotted residuals below ``lim_converged``) are
        shown with star markers; non-converged cases with circles.

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
            print("No data to plot.")
            return

        columns = [
            c for c in df_finals.columns
            if 'Residual' in c
            and 'MomentumYResidual' not in c
            and 'TurbulentSANuTilde' not in c
            and mode in c
        ]
        nrows = int(np.ceil(len(columns) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 5 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()

        converged_mask = (df_finals[columns].lt(lim_converged)).all(axis=1)
        norm           = mcolors.LogNorm(vmin=lim_converged, vmax=1e0)
        cmap_name      = kwargs.get('cmap', 'summer')

        dvf = [
            v for v in self.db.metadata['design_vars']
            if df_finals[v].nunique() > 1
        ]

        if len(dvf) == 1:
            for i, col in enumerate(columns):
                x = df_finals[dvf[0]]
                y = df_finals[col]
                c = df_finals["total_iterations"]
                sc_nc = axes[i].scatter(
                    x[~converged_mask], y[~converged_mask],
                    c=c[~converged_mask], cmap=cmap_name, norm=None,
                    s=60, edgecolor='k', label='Non-converged',
                )
                axes[i].scatter(
                    x[converged_mask], y[converged_mask],
                    c=c[converged_mask], cmap=cmap_name, norm=None,
                    s=60, marker='*', linewidth=1.5, label='Converged',
                )
                if activate_idx:
                    for p in df_finals[dvf].values:
                        matches = np.where(
                            self.db.df_state.iloc[:, 0] == p[0]
                        )[0]
                        if matches.size > 0:
                            axes[i].annotate(
                                f"{matches[0]}", (p[0], p[1]),
                                textcoords="offset points",
                                xytext=(0, 7), ha='center', fontsize=8,
                            )
                axes[i].set(title=col, xlabel=dvf[0], ylabel=col)
                fig.colorbar(sc_nc, ax=axes[i]).ax.set_title(
                    f'"Total iterations" {stage}'
                )

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',
                       frameon=False, ncols=2)
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
            print("Non-converged cases:")
            df_finals[columns][~converged_mask].to_csv(
                os.path.join(
                    self.db.root_dir, 'metadata', 'non_converged_cases.csv'
                ),
                index=False,
            )
            for i, row in df_finals[~converged_mask].iterrows():
                res_str = " ".join(f"{v:.2E}" for v in row[3:].values)
                print(
                    f"  Case {i}: "
                    + ", ".join(
                        f"{dvf[j]}={row[dvf[j]]:.4f}"
                        for j in range(len(dvf))
                    )
                    + f"  Residuals: {res_str}"
                )

    def plot_state_calculation(
        self,
        num_stages: int = 1,
        txt_from_end: int = 1,
        figsize: tuple = None,
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
        for name, case in self.db.sim_metadata.items():
            if len(case['stages']) == num_stages:
                df = CODAResiduals.get_df_residuals_from_txt(
                    case_path=case['path'],
                    verbose=False,
                    txt_from_end=txt_from_end,
                )
                if df is not None:
                    data_to_plot.append((name, df))
                else:
                    print(f"\tWARNING: Case '{name}' has not started yet.\n")

        if not data_to_plot:
            print("No data found for the specified criteria.")
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
            cur.tick_params(labelbottom=False)

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