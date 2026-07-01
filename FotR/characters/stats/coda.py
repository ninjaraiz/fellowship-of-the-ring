"""
stats/coda.py
=============
Stats class for the CODA CFD solver format.

Provides higher-level statistical analyses on top of an already-populated
FRODO instance whose reader is ``CODAReader``.  Unlike ``CODASets`` (which
assembles ML-ready tensors) or ``CODAResiduals`` (which deals with solver
convergence monitors), ``CODAStats`` is concerned with *descriptive and
comparative statistics of the field variables themselves*, e.g.:

* Per-variable descriptive statistics (mean, std, percentiles, ...) for a
  single solver stage (:meth:`CODAStats.compute_stats`).
* Statistical comparison of field variables between two or more solver
  stages — e.g. how much did the pressure field change between the
  coarse and the fine-tuned stage?
  (:meth:`CODAStats.stage_difference_stats`).

All methods operate on ``self.db.data_dict``, which is assumed to be
structured exactly as described in ``CODASets``::

    data_dict = {
        'CADGroup_<id>': {
            'Coord':  np.ndarray (n_points, n_dim),
            'FlCc':   np.ndarray (n_cases,  n_dvars),
            'Vars': {
                '<stage>': {
                    '<var_name>': np.ndarray (n_points, n_cases)
                                  or (n_dim, n_points, n_cases) for vectors,
                    ...
                },
                ...
            },
        },
        ...
    }

Typical workflow
-----------------

::

    db = FRODO(root_dir='/data/sim', format='CODA')
    db.extract_inputs(id_groups=(3,))
    db.extract_outputs(stage=0, id_groups=(3,))
    db.extract_outputs(stage=1, id_groups=(3,))

    # Descriptive stats of a single stage:
    stats0 = db.stats.compute_stats(id_group='3', stage=0)
    print(stats0['table'])

    # Statistical comparison between two stages:
    diff = db.stats.stage_difference_stats(
        id_group='3', stages=(0, 1), paired_test='wilcoxon',
    )
    print(diff['0_vs_1']['global'])
"""

import os
import warnings
from itertools import combinations
from typing import Literal, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns

from .base import BaseStats

if TYPE_CHECKING:
    from ..frodo import FRODO


class CODAStats(BaseStats):
    """
    Stats class for CODA-format FRODO databases.

    All methods read from ``self.db.data_dict``, which is assumed to use
    the per-CADGroup layout populated by ``CODAReader.extract_inputs`` /
    ``extract_outputs`` (see the module docstring above for the exact
    structure).

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance whose ``data_dict`` is already populated.

    Quick-reference
    ---------------
    ::

        db.stats.compute_stats(id_group='3', stage='0')
        db.stats.stage_difference_stats(id_group='3', stages=(0, 1))
    """

    def __init__(self, db: 'FRODO'):
        super().__init__(db)

    # =========================================================================
    # BaseStats interface
    # =========================================================================

    def compute_stats(
        self,
        id_group: str,
        stage: Union[int, str],
        variables: Union[str, list, tuple, None] = None,
        cases_idx: Union[list, tuple, int, str] = 'all',
        vector_handling: Literal['magnitude', 'components'] = 'magnitude',
        percentiles: Union[list, tuple] = (5, 25, 50, 75, 95),
        per_case: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Compute descriptive statistics for every field variable of a
        CADGroup at a single solver stage.

        For every selected variable, the mean, standard deviation, minimum,
        maximum and the requested percentiles are computed either:

        * over **all** selected points and cases pooled together
          (``per_case=False``, the default) — one row of statistics per
          variable; or
        * **per case** (``per_case=True``) — one row of statistics per
          ``(variable, case_idx)`` pair, useful to spot outlier cases
          (e.g. a case that diverged or has a much wider pressure range
          than the rest of the design space).

        Vector fields (stored as ``(n_dim, n_points, n_cases)``) are first
        reduced to one or more scalar fields, according to
        ``vector_handling``, before any statistic is computed.

        Parameters
        ----------
        id_group : str
            CADGroup identifier string (e.g. ``'3'`` or ``'1_2'``).
        stage : int or str
            Stage key in ``data_dict['CADGroup_<id_group>']['Vars']``
            (e.g. ``0`` or ``'0'``).
        variables : str, list[str], tuple[str] or None
            Variable(s) to analyse.  ``None`` (default) analyses every
            variable present in the stage except ``'GlobalNumber'`` and
            ``'CADGroupID'``.
        cases_idx : 'all', int, list[int] or tuple[int]
            Subset of case indices (columns of the variable arrays) to
            include. Default ``'all'``.
        vector_handling : 'magnitude' or 'components'
            How to reduce a vector field (``ndim == 3``) to scalar data:

            * ``'magnitude'`` (default) — Euclidean norm across the
              component axis, producing one combined scalar field per
              vector variable (e.g. ``'Velocity'``).
            * ``'components'`` — each spatial component is treated as an
              independent scalar field, named ``'<var>_<i>'`` (e.g.
              ``'Velocity_0'``, ``'Velocity_1'``, ``'Velocity_2'``).
        percentiles : list[float] or tuple[float]
            Percentiles (0-100) to compute via ``numpy.percentile``.
            Default ``(5, 25, 50, 75, 95)``.
        per_case : bool
            If True, return one row of statistics per case instead of
            pooling every selected case together. Default False.
        verbose : bool
            Print the list of analysed variables and the resulting table
            shape. Default False.

        Returns
        -------
        dict
            ``{'table': pd.DataFrame, 'variables': list[str]}``.

            ``table`` has one row per variable (``per_case=False``) or one
            row per ``(variable, case_idx)`` pair (``per_case=True``),
            indexed accordingly.  Columns: ``'mean'``, ``'std'``,
            ``'min'``, ``'max'``, ``'n_points'``, ``'n_cases'`` (only when
            ``per_case=False``), and one column per requested percentile
            named ``'p<value>'`` (e.g. ``'p50'`` for the median).

        Side-effects
        ------------
        Stores the result dict in
        ``db.stats_results['CADGroup_<id_group>_stage_<stage>']``
        (a new dict is created on first use; later calls for other
        groups/stages are added alongside, not overwritten).

        Raises
        ------
        KeyError
            If ``id_group`` or ``stage`` is not found in ``data_dict``, or
            a requested variable does not exist.
        ValueError
            If no variable remains to analyse after filtering.

        Examples
        --------
        Pooled statistics for every variable at stage 0::

            db = FRODO(root_dir='/data/sim', format='CODA')
            db.extract_inputs(id_groups=(3,))
            db.extract_outputs(stage=0, id_groups=(3,))

            result = db.stats.compute_stats(id_group='3', stage=0, verbose=True)
            print(result['table'])
            #                mean       std  ...   p75   p95
            # variable
            # Pressure       0.41      0.08  ...  0.46  0.55
            # Velocity       3.20      0.31  ...  3.41  3.78

        Per-case statistics for a single variable::

            result = db.stats.compute_stats(
                id_group='3', stage=0, variables='Pressure', per_case=True,
            )
            print(result['table'].head())

        Treat a velocity vector field component-wise instead of by
        magnitude::

            result = db.stats.compute_stats(
                id_group='3', stage=0, variables='Velocity',
                vector_handling='components',
            )
            print(result['variables'])   # ['Velocity_0', 'Velocity_1', 'Velocity_2']
        """
        key_group = f'CADGroup_{id_group}'
        if key_group not in self.db.data_dict:
            raise KeyError(f"'{key_group}' not found in data_dict.")
        group = self.db.data_dict[key_group]

        if 'Vars' not in group or str(stage) not in group['Vars']:
            raise KeyError(
                f"Stage '{stage}' not found in '{key_group}'['Vars']."
            )
        stage_vars = group['Vars'][str(stage)]

        n_cases_total = group['FlCc'].shape[0]
        cases_idx = self._normalise_cases_idx(cases_idx, n_cases_total)

        var_list = self._select_variables(stage_vars, variables)
        fields = self._expand_vector_fields(stage_vars, var_list, vector_handling)

        if not fields:
            raise ValueError(
                f"No variables left to analyse in '{key_group}' stage "
                f"'{stage}' after filtering."
            )

        records = []
        for label, arr in fields.items():
            arr_sel = arr[:, cases_idx]

            if per_case:
                for j, case_i in enumerate(cases_idx):
                    records.append(
                        self._describe_array(
                            arr_sel[:, j], percentiles,
                            label=label, case_idx=case_i,
                        )
                    )
            else:
                records.append(
                    self._describe_array(
                        arr_sel.reshape(-1), percentiles,
                        label=label, n_cases=len(cases_idx),
                    )
                )

        index_cols = ['variable', 'case_idx'] if per_case else ['variable']
        table = pd.DataFrame.from_records(records).set_index(index_cols)

        result = {'table': table, 'variables': list(fields.keys())}

        self.db.stats_results = getattr(self.db, 'stats_results', {})
        self.db.stats_results[f'{key_group}_stage_{stage}'] = result

        if verbose:
            print(f"[CODAStats] Variables analysed: {list(fields.keys())}")
            print(f"[CODAStats] Table shape: {table.shape}")

        return result

    # =========================================================================
    # Stage comparison
    # =========================================================================

    def stage_difference_stats(
        self,
        id_group: str,
        stages: Union[list, tuple],
        pairs: Union[Literal['consecutive', 'all'], list] = 'consecutive',
        variables: Union[str, list, tuple, None] = None,
        cases_idx: Union[list, tuple, int, str] = 'all',
        diff_mode: Literal['absolute', 'relative'] = 'absolute',
        vector_handling: Literal['magnitude', 'components'] = 'magnitude',
        percentiles: Union[list, tuple] = (5, 25, 50, 75, 95),
        paired_test: Union[Literal['ttest', 'wilcoxon'], None] = None,
        relative_eps: float = 1e-12,
        store_raw_diff: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Statistically compare field variables across two or more CODA
        solver stages.

        For every pair of stages selected via ``stages`` and ``pairs``,
        and for every requested field variable, the point-wise difference
        between the two stages (``stage_b - stage_a``, optionally
        normalised by ``stage_a``) is computed and summarised both
        per-case and globally (pooling every selected case together).

        Parameters
        ----------
        id_group : str
            CADGroup identifier string (e.g. ``'3'`` or ``'1_2'``).
        stages : list or tuple
            Sequence of stage keys to compare (e.g. ``(0, 1)`` or
            ``(0, 1, 2)``). At least two stages are required.
        pairs : 'consecutive', 'all' or list[tuple]
            How to build the stage pairs to compare from ``stages``:

            * ``'consecutive'`` (default) — compares ``stages[i]``
              against ``stages[i + 1]`` for every ``i``. E.g.
              ``(0, 1, 2)`` produces the pairs ``(0, 1)`` and ``(1, 2)``.
            * ``'all'`` — compares every unordered combination of two
              stages. E.g. ``(0, 1, 2)`` produces ``(0, 1)``, ``(0, 2)``
              and ``(1, 2)``.
            * An explicit list of ``(stage_a, stage_b)`` tuples — used
              as-is, ignoring the default pairing logic. Every stage
              referenced must also appear in ``stages``.
        variables : str, list[str], tuple[str] or None
            Variable(s) to compare. ``None`` (default) compares every
            variable present in **both** stages of a pair, except
            ``'GlobalNumber'`` and ``'CADGroupID'``.
        cases_idx : 'all', int, list[int] or tuple[int]
            Subset of case indices to include. Default ``'all'``.
        diff_mode : 'absolute' or 'relative'
            * ``'absolute'`` (default) — ``diff = stage_b - stage_a``.
            * ``'relative'`` — ``diff = (stage_b - stage_a) / (|stage_a| + relative_eps)``,
              i.e. the fractional change with respect to ``stage_a``.
        vector_handling : 'magnitude' or 'components'
            How to reduce a vector field to scalar data before
            differencing. See :meth:`compute_stats` for the full
            description.
        percentiles : list[float] or tuple[float]
            Percentiles (0-100) of the difference distribution to
            compute. Default ``(5, 25, 50, 75, 95)``.
        paired_test : 'ttest', 'wilcoxon' or None
            If given, runs a paired statistical test (per case, across
            points) comparing the point values of ``stage_a`` against
            ``stage_b`` for every variable:

            * ``'ttest'`` — ``scipy.stats.ttest_rel`` (paired Student's
              t-test). Assumes approximately normal differences.
            * ``'wilcoxon'`` — ``scipy.stats.wilcoxon`` (paired
              non-parametric signed-rank test). Use when the differences
              are not expected to be normally distributed.

            Default ``None`` (no test is run; only descriptive statistics
            of the difference are computed).
        relative_eps : float
            Small constant added to ``|stage_a|`` to avoid division by
            zero when ``diff_mode='relative'``. Default ``1e-12``.
        store_raw_diff : bool
            If True, also store the raw point-wise difference arrays in
            the result, under ``'raw_diff'``. Useful for further custom
            analysis or plotting, at the cost of extra memory. Default
            False.
        verbose : bool
            Print progress information for every stage pair. Default
            False.

        Returns
        -------
        dict
            One entry per stage pair, keyed by ``'<stage_a>_vs_<stage_b>'``
            (e.g. ``'0_vs_1'``). Each entry is itself a dict with keys:

            * ``'per_case'`` : pd.DataFrame
                  One row per ``(variable, case_idx)`` pair. Columns:
                  ``'mean'``, ``'std'``, ``'min'``, ``'max'``,
                  ``'n_points'``, one column per requested percentile
                  (``'p<value>'``), and, if ``paired_test`` is set,
                  ``'test_stat'`` and ``'test_pvalue'``.
            * ``'global'`` : pd.DataFrame
                  One row per variable, pooling every selected case
                  together. Same descriptive columns as ``'per_case'``
                  (plus ``'n_cases'``), but **no** statistical-test
                  columns — see the Notes section below for why.
            * ``'raw_diff'`` : dict[str, np.ndarray] or None
                  Only populated when ``store_raw_diff=True``. Maps each
                  variable label to its raw
                  ``(n_points, n_selected_cases)`` difference array.

        Raises
        ------
        ValueError
            If fewer than two stages are given, if ``pairs`` is an
            unsupported string, or if an explicit pair references a
            stage not present in ``stages``.
        KeyError
            If ``id_group`` is not found in ``data_dict``, or a requested
            stage is missing from it.

        Side-effects
        ------------
        Stores the full result dict in ``db.stage_diff_results``
        (overwriting any previous call's result).

        Notes
        -----
        **Spatial autocorrelation caveat.** The paired tests are
        structurally valid (each point in ``stage_a`` is paired with the
        *same* point in ``stage_b``), but CFD surface and volume fields
        are spatially correlated: neighbouring mesh points are not
        independent samples. This violates the independence assumption
        behind the reported p-value and tends to make the test **overly
        confident** (artificially small p-values) for a given true effect
        size. The test is still useful as a *relative* indicator (e.g.
        comparing the test statistic across cases or variables), but the
        p-value should not be read as a calibrated probability. For this
        reason ``paired_test`` is only computed **per case** (where the
        number of points is fixed and comparisons stay apples-to-apples)
        and deliberately **not** computed on the pooled ``'global'``
        table, where pooling cases together would inflate the apparent
        sample size even further.

        Examples
        --------
        Compare two consecutive stages for every common variable::

            db = FRODO(root_dir='/data/sim', format='CODA')
            db.extract_inputs(id_groups=(3,))
            db.extract_outputs(stage=0, id_groups=(3,))
            db.extract_outputs(stage=1, id_groups=(3,))

            result = db.stats.stage_difference_stats(
                id_group='3', stages=(0, 1), verbose=True,
            )
            print(result['0_vs_1']['global'])

        Relative change with a paired Wilcoxon test, for a single
        variable, keeping the raw differences for later plotting::

            result = db.stats.stage_difference_stats(
                id_group='3', stages=(0, 1),
                variables='Pressure',
                diff_mode='relative',
                paired_test='wilcoxon',
                store_raw_diff=True,
            )
            diff_arr = result['0_vs_1']['raw_diff']['Pressure']
            print(result['0_vs_1']['per_case'].head())

        Compare every pair among three stages::

            result = db.stats.stage_difference_stats(
                id_group='3', stages=(0, 1, 2), pairs='all',
            )
            print(list(result.keys()))   # ['0_vs_1', '0_vs_2', '1_vs_2']
        """
        key_group = f'CADGroup_{id_group}'
        if key_group not in self.db.data_dict:
            raise KeyError(f"'{key_group}' not found in data_dict.")
        group = self.db.data_dict[key_group]

        if len(stages) < 2:
            raise ValueError("At least two stages are required.")

        stage_pairs = self._resolve_stage_pairs(stages, pairs)

        n_cases_total = group['FlCc'].shape[0]
        cases_idx = self._normalise_cases_idx(cases_idx, n_cases_total)

        test_fn = None
        if paired_test is not None:
            from scipy import stats as scipy_stats
            test_fn_map = {
                'ttest':    scipy_stats.ttest_rel,
                'wilcoxon': scipy_stats.wilcoxon,
            }
            if paired_test not in test_fn_map:
                raise ValueError(
                    f"paired_test '{paired_test}' not supported. "
                    f"Options: {list(test_fn_map)} or None."
                )
            test_fn = test_fn_map[paired_test]

        results: dict = {}

        for stage_a, stage_b in stage_pairs:
            pair_key = f'{stage_a}_vs_{stage_b}'

            if str(stage_a) not in group.get('Vars', {}):
                raise KeyError(f"Stage '{stage_a}' not found in '{key_group}'.")
            if str(stage_b) not in group.get('Vars', {}):
                raise KeyError(f"Stage '{stage_b}' not found in '{key_group}'.")

            vars_a = group['Vars'][str(stage_a)]
            vars_b = group['Vars'][str(stage_b)]

            common = {
                v: vars_a[v] for v in vars_a
                if v in vars_b and v not in ('GlobalNumber', 'CADGroupID')
            }
            var_list = self._select_variables(common, variables, excluded=())

            if not var_list:
                warnings.warn(
                    f"No common variables found between stages "
                    f"'{stage_a}' and '{stage_b}' for '{key_group}'. "
                    f"Skipping pair '{pair_key}'.",
                    UserWarning,
                )
                continue

            if verbose:
                print(f"[CODAStats] {pair_key}: variables = {var_list}")

            fields_a = self._expand_vector_fields(vars_a, var_list, vector_handling)
            fields_b = self._expand_vector_fields(vars_b, var_list, vector_handling)

            per_case_records = []
            global_records   = []
            raw_diff: dict   = {} if store_raw_diff else None

            for label in fields_a:
                arr_a = fields_a[label][:, cases_idx]
                arr_b = fields_b[label][:, cases_idx]

                if diff_mode == 'absolute':
                    diff = arr_b - arr_a
                elif diff_mode == 'relative':
                    diff = (arr_b - arr_a) / (np.abs(arr_a) + relative_eps)
                else:
                    raise ValueError(
                        f"diff_mode '{diff_mode}' not supported. "
                        "Options: 'absolute', 'relative'."
                    )

                if store_raw_diff:
                    raw_diff[label] = diff

                for j, case_i in enumerate(cases_idx):
                    record = self._describe_array(
                        diff[:, j], percentiles, label=label, case_idx=case_i,
                    )
                    if test_fn is not None:
                        try:
                            test_res = test_fn(arr_a[:, j], arr_b[:, j])
                            record['test_stat']   = float(test_res.statistic)
                            record['test_pvalue'] = float(test_res.pvalue)
                        except Exception as exc:
                            record['test_stat']   = np.nan
                            record['test_pvalue'] = np.nan
                            if verbose:
                                print(
                                    f"[CODAStats] {pair_key}/{label}/case "
                                    f"{case_i}: paired test failed ({exc})."
                                )
                    per_case_records.append(record)

                global_records.append(
                    self._describe_array(
                        diff.reshape(-1), percentiles, label=label,
                        n_cases=len(cases_idx),
                    )
                )

            per_case_df = (
                pd.DataFrame.from_records(per_case_records)
                .set_index(['variable', 'case_idx'])
            )
            global_df = (
                pd.DataFrame.from_records(global_records)
                .set_index('variable')
            )

            results[pair_key] = {
                'per_case': per_case_df,
                'global':   global_df,
                'raw_diff': raw_diff,
            }

        self.db.stage_diff_results = results

        return results

    # =========================================================================
    # Stage comparison — single-variable, plot-ready
    # =========================================================================

    def stage_difference_stats_variable(
        self,
        id_group: str,
        stages: Union[list, tuple],
        variable: str,
        pairs: Union[Literal['consecutive', 'all'], list] = 'consecutive',
        cases_idx: Union[list, tuple, int, str] = 'all',
        diff_mode: Literal['absolute', 'relative'] = 'absolute',
        vector_handling: Literal['magnitude', 'components'] = 'magnitude',
        relative_eps: float = 1e-12,
        coord_idx: int = 0,
        n_coord_bins: int = 10,
        flcc_vars: Union[list, tuple, None] = None,
        case_metric: Literal[
            'max_diff', 'min_diff', 'max_abs_diff', 'L_2_diff', 'L_inf_diff', 'mean_diff', 'std_diff'
        ] = 'L_2_diff',
        plots: Union[dict, None] = None,
        kwargs_plots: dict = {},
        annotate_case_idx: bool = True,
        figsize: tuple = (9, 5),
        save_dir: Union[str, None] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Compare a single field variable between two (or more) CODA solver
        stages and draw illustrative plots of the difference, both with
        respect to the **geometric position** of each point (``Coord``)
        and with respect to the **design / flight-condition space**
        (``FlCc``).

        For every stage pair selected via ``stages`` and ``pairs``, two
        tables are assembled (see ``Returns``) and, according to
        ``plots``, up to three figures are drawn per pair (per
        vector-component label, when ``vector_handling='components'``):

        * **``'boxplot'``** — distribution of the point-wise difference
          grouped by spatial position: points are binned along coordinate
          axis ``coord_idx`` into ``n_coord_bins`` quantile bins (so each
          bin holds roughly the same number of points, even on a
          non-uniform mesh), and a seaborn boxplot shows the difference
          distribution (pooling every selected case) for each bin. This
          highlights *where* on the geometry the two stages disagree the
          most.
        * **``'scatter'``** — one point per case, ``x``/``y`` given by the
          first one or two entries of ``flcc_vars`` (defaulting to
          ``db.metadata['design_vars'][:2]``), coloured by
          ``case_metric`` (e.g. the maximum absolute difference observed
          for that case). This highlights *which region of the design
          space* (e.g. high AoA, transonic Mach) is most affected.
        * **``'histogram'``** — a single overall histogram (with KDE) of
          the pooled point-wise difference, ignoring both position and
          design space. Off by default; mostly useful as a quick sanity
          check of the difference's overall shape (symmetric, skewed,
          bimodal, ...).

        Vector fields (``ndim == 3``) are first reduced to one or more
        scalar fields according to ``vector_handling`` — see
        :meth:`compute_stats` for the full description of
        ``'magnitude'`` vs. ``'components'``.

        Parameters
        ----------
        id_group : str
            CADGroup identifier string (e.g. ``'3'`` or ``'1_2'``).
        stages : list or tuple
            Sequence of stage keys to compare (e.g. ``(0, 1)``). At least
            two stages are required.
        variable : str
            Name of the single field variable to analyse (e.g.
            ``'Pressure'`` or ``'Velocity'``).
        pairs : 'consecutive', 'all' or list[tuple]
            How to build stage pairs from ``stages``. See
            :meth:`stage_difference_stats` for the full description.
        cases_idx : 'all', int, list[int] or tuple[int]
            Subset of case indices to include. Default ``'all'``.
        diff_mode : 'absolute' or 'relative'
            ``'absolute'`` (default): ``diff = stage_b - stage_a``.
            ``'relative'``: ``diff = (stage_b - stage_a) / (|stage_a| + relative_eps)``.
        vector_handling : 'magnitude' or 'components'
            Vector-field reduction strategy. See :meth:`compute_stats`.
        relative_eps : float
            Stability constant for ``diff_mode='relative'``. Default
            ``1e-12``.
        coord_idx : int
            Index of the ``Coord`` column used to bin points spatially
            for the boxplot (e.g. ``0`` for ``x``, ``2`` for ``z``).
            Default ``0``.
        n_coord_bins : int
            Number of quantile bins along ``coord_idx`` for the boxplot.
            Default ``10``.
        flcc_vars : list[str], tuple[str] or None
            Design-variable name(s) (must match
            ``db.metadata['design_vars']``) used as the scatter's ``x``
            (and ``y``, if two are given) axes. ``None`` (default) uses
            the first two entries of ``db.metadata['design_vars']`` (or
            just the first one if only a single design variable exists).
        case_metric : 'max_diff', 'min_diff', 'max_abs_diff', 'mean_diff' or 'std_diff'
            Per-case aggregate of the point-wise difference used to
            colour (or, with a single ``flcc_vars`` entry, to use as the
            ``y``-axis of) the scatter plot. Default ``'max_abs_diff'``.
        plots : dict or None
            Which figures to draw. Missing keys fall back to the
            defaults below, so a partial dict only overrides what is
            given::

                {'boxplot': False, 'scatter': False, 'histogram': True, 'violinplot': False}

        annotate_case_idx : bool
            If True, annotate each scatter point with its case index
            (mirrors the convention used elsewhere in FotR, e.g.
            ``CODAReader.plot_state``). Default True.
        figsize : tuple
            Figure size forwarded to ``plt.subplots``. Default ``(9, 5)``.
        save_dir : str or None
            If given, every figure is saved here (one file per pair,
            label and plot type) instead of being displayed inline.
            Default None (``plt.show()``).
        verbose : bool
            Print progress information for every stage pair and label.
            Default False.

        Returns
        -------
        dict
            One entry per stage pair, keyed by ``'<stage_a>_vs_<stage_b>'``.
            Each entry is a dict with keys:

            * ``'by_point'`` : pd.DataFrame
                  Long-format table, one row per ``(label, case_idx,
                  point_idx)``. Columns: ``'label'``, ``'case_idx'``,
                  ``'point_idx'``, ``f'coord_{coord_idx}'`` (raw
                  coordinate value), ``'coord_bin'`` (quantile-bin
                  midpoint, as a string — this is exactly what the
                  boxplot's ``x`` axis groups on), ``'diff'``.
            * ``'by_case'`` : pd.DataFrame
                  One row per ``(label, case_idx)``. Columns: ``'label'``,
                  ``'case_idx'``, every entry of
                  ``db.metadata['design_vars']``, ``'max_diff'``,
                  ``'min_diff'``, ``'max_abs_diff'``, ``'mean_diff'``,
                  ``'std_diff'`` — this is exactly what the scatter plots.

        Raises
        ------
        ValueError
            If fewer than two stages are given.
        KeyError
            If ``id_group`` is not found, a requested stage is missing,
            ``variable`` is not present in both stages of a pair, or
            ``flcc_vars`` references an unknown design variable.
        IndexError
            If ``coord_idx`` is out of bounds for ``Coord``.

        Side-effects
        ------------
        Stores the full result dict in ``db.stage_diff_variable_results``
        (overwriting any previous call's result), in addition to drawing
        the requested figures.

        Notes
        -----
        Building ``'by_point'`` materialises one row per selected
        ``(point, case)`` pair, so for a fine surface mesh and many
        selected cases this table — and therefore the memory used while
        drawing the boxplot — can get large. Restricting ``cases_idx`` is
        the simplest way to keep this in check; the boxplot itself only
        needs enough cases to populate every coordinate bin meaningfully,
        so a representative subset is usually sufficient.

        Examples
        --------
        Boxplot by chordwise position and scatter coloured by the worst
        per-case pressure change, for two consecutive stages::

            db = FRODO(root_dir='/data/sim', format='CODA')
            db.extract_inputs(id_groups=(3,))
            db.extract_outputs(stage=0, id_groups=(3,))
            db.extract_outputs(stage=1, id_groups=(3,))

            result = db.stats.stage_difference_stats_variable(
                id_group='3', stages=(0, 1), variable='Pressure',
                coord_idx=0, n_coord_bins=12,
            )
            print(result['0_vs_1']['by_case'].head())

        Relative change of a velocity vector field, treated
        component-wise, only the scatter plot, saved to disk::

            db.stats.stage_difference_stats_variable(
                id_group='3', stages=(0, 1), variable='Velocity',
                vector_handling='components', diff_mode='relative',
                plots={'boxplot': False, 'scatter': True},
                save_dir='/output/stage_diff_plots/',
            )

        Custom design-variable pair for the scatter, coloured by the
        mean (rather than max) difference::

            db.stats.stage_difference_stats_variable(
                id_group='3', stages=(0, 1), variable='Pressure',
                flcc_vars=['Mach', 'AoA'], case_metric='mean_diff',
            )
        """
        from scipy.stats import skew, kurtosis
        
        key_group = f'CADGroup_{id_group}'
        if key_group not in self.db.data_dict:
            raise KeyError(f"'{key_group}' not found in data_dict.")
        group = self.db.data_dict[key_group]

        if len(stages) < 2:
            raise ValueError("At least two stages are required.")

        plot_flags = {'boxplot': False, 'scatter': False, 'histogram': False, 'violinplot': False, 'histogram2D': False}

        for plot_type in plot_flags.keys():
            if plot_type not in kwargs_plots:
                kwargs_plots[plot_type] = {}
                
        if plots:
            plot_flags.update(plots)

        stage_pairs = self._resolve_stage_pairs(stages, pairs)

        n_cases_total = group['FlCc'].shape[0]
        cases_idx = self._normalise_cases_idx(cases_idx, n_cases_total)

        coords = group['Coord']
        if coord_idx < 0 or coord_idx >= coords.shape[1]:
            raise IndexError(
                f"coord_idx {coord_idx} out of bounds for Coord with "
                f"shape {coords.shape}."
            )
        coord_values   = coords[:, coord_idx]
        coord_bin_lbls = self._bin_coordinate(coord_values, n_coord_bins)

        flcc = group['FlCc']
        design_vars = self.db.metadata.get('design_vars', None) or [
            f'flcc_{k}' for k in range(flcc.shape[1])
        ]
        if flcc_vars is None:
            flcc_vars_use = (
                design_vars[:2] if len(design_vars) >= 2 else design_vars[:1]
            )
        else:
            flcc_vars_use = list(flcc_vars)
            missing = [v for v in flcc_vars_use if v not in design_vars]
            if missing:
                raise KeyError(
                    f"flcc_vars {missing} not found. "
                    f"Available design vars: {design_vars}."
                )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        results: dict = {}

        for stage_a, stage_b in stage_pairs:
            pair_key = f'{stage_a}_vs_{stage_b}'

            if str(stage_a) not in group.get('Vars', {}):
                raise KeyError(f"Stage '{stage_a}' not found in '{key_group}'.")
            if str(stage_b) not in group.get('Vars', {}):
                raise KeyError(f"Stage '{stage_b}' not found in '{key_group}'.")

            vars_a = group['Vars'][str(stage_a)]
            vars_b = group['Vars'][str(stage_b)]
            if variable not in vars_a or variable not in vars_b:
                raise KeyError(
                    f"Variable '{variable}' not found in both stages "
                    f"'{stage_a}' and '{stage_b}' of '{key_group}'."
                )

            fields_a = self._expand_vector_fields(vars_a, [variable], vector_handling)
            fields_b = self._expand_vector_fields(vars_b, [variable], vector_handling)

            if verbose:
                print(f"[CODAStats] {pair_key}: labels = {list(fields_a)}")

            by_point_frames = []
            by_case_frames  = []
            by_bin_stats_frames = []
            
            for label in fields_a:
                arr_a = fields_a[label][:, cases_idx]
                arr_b = fields_b[label][:, cases_idx]

                if diff_mode == 'absolute':
                    diff = arr_b - arr_a
                elif diff_mode == 'relative':
                    diff = (arr_b - arr_a) / (np.abs(arr_a) + relative_eps)
                else:
                    raise ValueError(
                        f"diff_mode '{diff_mode}' not supported. "
                        "Options: 'absolute', 'relative'."
                    )

                n_points, n_sel = diff.shape

                # ── by_point (boxplot / histogram source) ────────────────────
                df_point = pd.DataFrame({
                    'label':              label,
                    'case_idx':           np.repeat(cases_idx, n_points),
                    'point_idx':          np.tile(np.arange(n_points), n_sel),
                    f'coord_{coord_idx}': np.tile(coord_values, n_sel),
                    'coord_bin':          np.tile(coord_bin_lbls, n_sel),
                    'diff':               diff.T.reshape(-1),
                })
                by_point_frames.append(df_point)
                # ── by_bin_stats (boxplot statistics) ─────────────────────────────

                stats_records = []

                for bin_name, group_bin in df_point.groupby("coord_bin", sort=False):

                    values = group_bin["diff"].dropna().to_numpy()

                    if values.size == 0:
                        continue

                    q1 = np.percentile(values, 25)
                    median = np.percentile(values, 50)
                    q3 = np.percentile(values, 75)

                    iqr = q3 - q1

                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                    outliers = (values < lower) | (values > upper)

                    stats_records.append({

                        "label": label,

                        "coord_bin": bin_name,

                        "n_points": values.size,

                        "mean": np.mean(values),

                        "median": median,

                        "std": np.std(values, ddof=1),

                        "variance": np.var(values, ddof=1),

                        "min": np.min(values),

                        "q1": q1,

                        "q3": q3,

                        "max": np.max(values),

                        "iqr": iqr,

                        "mad": np.median(np.abs(values - median)),

                        "rms": np.sqrt(np.mean(values**2)),

                        "abs_mean": np.mean(np.abs(values)),

                        "abs_max": np.max(np.abs(values)),

                        "cv": (
                            np.std(values, ddof=1)
                            / max(abs(np.mean(values)), 1e-15)
                        ),

                        "skew": skew(values, bias=False),

                        "kurtosis": kurtosis(values, bias=False),

                        "lower_fence": lower,

                        "upper_fence": upper,

                        "n_outliers": int(outliers.sum()),

                        "outlier_percent": 100 * outliers.mean(),

                    })

                by_bin_stats_frames.append(
                    pd.DataFrame.from_records(stats_records)
                )

                # ── by_case (scatter source) ──────────────────────────────────
                case_records = []
                for j, case_i in enumerate(cases_idx):
                    col    = diff[:, j]
                    finite = col[np.isfinite(col)]
                    record = {
                        'label':        label,
                        'case_idx':     case_i,
                        'max_diff':     float(np.max(finite))                           if finite.size else np.nan,
                        'min_diff':     float(np.min(finite))                           if finite.size else np.nan,
                        'max_abs_diff': float(np.max(np.abs(finite)))                   if finite.size else np.nan,
                        'L_2_diff':     float(np.linalg.norm(finite, ord=2, axis=0))      if finite.size else np.nan,
                        'L_inf_diff':   float(np.linalg.norm(finite, ord=np.inf, axis=0)) if finite.size else np.nan,
                        'mean_diff':    float(np.mean(finite))                          if finite.size else np.nan,
                        'std_diff':     float(np.std(finite))                           if finite.size else np.nan,
                    }
                    for k, dv in enumerate(design_vars):
                        record[dv] = float(flcc[case_i, k])
                    case_records.append(record)
                df_case = pd.DataFrame.from_records(case_records)
                by_case_frames.append(df_case)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                # ── Plots ──────────────────────────────────────────────────────
                if plot_flags.get('boxplot'):
                    self._plot_diff_boxplot(
                        df_point = df_point,
                        label = label,
                        stage_a = stage_a,
                        stage_b = stage_b,
                        diff_mode = diff_mode,
                        coord_idx = coord_idx,
                        figsize = figsize,
                        save_dir = save_dir,
                        **kwargs_plots['boxplot']
                    )
                    
                if plot_flags.get('scatter'):
                    self._plot_diff_scatter(
                        df_case = df_case,
                        label = label,
                        stage_a = stage_a,
                        stage_b = stage_b,
                        flcc_vars_use = flcc_vars_use,
                        case_metric = case_metric,
                        annotate_case_idx = annotate_case_idx,
                        figsize = figsize,
                        save_dir = save_dir,
                        **kwargs_plots['scatter']
                    )
                    
                if plot_flags.get('histogram'):
                    if 'bins' not in kwargs_plots['histogram']:
                        kwargs_plots['histogram']['bins'] = n_coord_bins
                        
                    self._plot_diff_histogram(
                        df_point = df_point,
                        label = label,
                        stage_a = stage_a,
                        stage_b = stage_b,
                        diff_mode = diff_mode,
                        coord_idx = coord_idx,
                        figsize = figsize,
                        save_dir = save_dir,
                        **kwargs_plots['histogram']
                    )
                    
                if plot_flags.get('violinplot'):
                    self._plot_diff_violinplot(
                        df_point,
                        label,stage_a,stage_b,
                        diff_mode,
                        coord_idx, figsize, save_dir,
                    )
                if plot_flags.get('histogram2D'):
                    self._plot_diff_histogram_2D(
                        df_point=df_point,
                        label=label,
                        stage_a=stage_a,
                        stage_b=stage_b,
                        diff_mode=diff_mode,
                        coord_idx=coord_idx,
                        figsize=figsize,
                        save_dir=save_dir,
                        **kwargs_plots['histogram2D']
                    )
            # Concatenar dataframes con tipos de datos personalizados por columnas (int, float y str)
            results[pair_key] = {
                'by_point': pd.concat(
                    by_point_frames,
                    ignore_index=True,
                ).astype({
                    'label': 'str',
                    'case_idx': 'int32',
                    'point_idx': 'int32',
                    f'coord_{coord_idx}': 'float32',
                    'coord_bin': 'str',
                    'diff': 'float32',
                }),
                'by_case': pd.concat(
                    by_case_frames,
                    ignore_index=True,
                ).astype({
                    'label': 'str',
                    'case_idx': 'int32',
                    'max_diff': 'float32',
                    'min_diff': 'float32',
                    'max_abs_diff': 'float32',
                    'L_2_diff':     'float32',
                    'L_inf_diff':   'float32',
                    'mean_diff': 'float32',
                    'std_diff': 'float32',
                    **{dv: 'float32' for dv in design_vars},
                }),
                'by_bin_stats': pd.concat(
                    by_bin_stats_frames,
                    ignore_index=True,
                ).astype({
                    'label': 'str',
                    'coord_bin': 'str',
                    'n_points': 'int32',
                    'mean': 'float32',
                    'median': 'float32',
                    'std': 'float32',
                    'variance': 'float32',
                    'min': 'float32',
                    'q1': 'float32',
                    'q3': 'float32',
                    'max': 'float32',
                    'iqr': 'float32',
                    'mad': 'float32',
                    'rms': 'float32',
                    'abs_mean': 'float32',
                    'abs_max': 'float32',
                    'cv': 'float32',
                    'skew': 'float32',
                    'kurtosis': 'float32',
                    'lower_fence': 'float32',
                    'upper_fence': 'float32',
                    'n_outliers': 'int32',
                    'outlier_percent': 'float32',
                }),
            }
            
            if save_dir:
                np.savez_compressed(
                    os.path.join(save_dir, f"dict_results_db_{self.db.name}_{pair_key}.npy"), results[pair_key]
                )
            # results[pair_key] = {
            #     'by_point': pd.concat(
            #         by_point_frames,
            #         ignore_index=True,
            #     ).astype("float32"),

            #     'by_case': pd.concat(
            #         by_case_frames,
            #         ignore_index=True,
            #     ).astype("float32"),

            #     'by_bin_stats': pd.concat(
            #         by_bin_stats_frames,
            #         ignore_index=True,
            #     ).astype("float32"),
            # }

        self.db.stage_diff_variable_results = results

        return results

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _normalise_cases_idx(cases_idx, n_cases: int) -> list:
        """
        Normalise a ``cases_idx`` argument to a list of valid integer case
        indices.

        Mirrors ``CODAReader._normalise_cases_idx`` but takes the total
        number of cases directly (instead of a ``df_cases`` DataFrame),
        since ``CODAStats`` only needs to validate against the case
        dimension of the already-extracted field arrays.

        Parameters
        ----------
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Case selection to normalise.
        n_cases : int
            Total number of cases available (e.g.
            ``data_dict['CADGroup_<id>']['FlCc'].shape[0]``).

        Returns
        -------
        list[int]

        Raises
        ------
        ValueError
            If ``cases_idx`` is a string other than ``'all'``, or has an
            unsupported type.
        IndexError
            If any requested index is out of range.

        Examples
        --------
        ::

            idx = CODAStats._normalise_cases_idx('all', n_cases=50)
            idx = CODAStats._normalise_cases_idx([0, 1, 4], n_cases=50)
        """
        if isinstance(cases_idx, str):
            if cases_idx.lower() == 'all':
                cases_idx = list(range(n_cases))
            else:
                raise ValueError("Invalid string for cases_idx. Use 'all'.")
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
    def _select_variables(
        stage_vars: dict,
        variables: Union[str, list, tuple, None],
        excluded: tuple = ('GlobalNumber', 'CADGroupID'),
    ) -> list:
        """
        Resolve which variable names to analyse from a ``Vars[<stage>]``
        dict (or an equivalent dict restricted to the variables common to
        two stages).

        Parameters
        ----------
        stage_vars : dict
            Mapping from variable name to its array, e.g.
            ``data_dict['CADGroup_3']['Vars']['0']``.
        variables : str, list[str], tuple[str] or None
            Requested variable(s). ``None`` selects every key in
            ``stage_vars`` not present in ``excluded``.
        excluded : tuple[str]
            Variable names to always exclude when ``variables is None``.
            Default ``('GlobalNumber', 'CADGroupID')``.

        Returns
        -------
        list[str]

        Raises
        ------
        KeyError
            If an explicitly requested variable is not found in
            ``stage_vars``.

        Examples
        --------
        ::

            stage_vars = {'Pressure': arr1, 'Velocity': arr2, 'GlobalNumber': arr3}
            CODAStats._select_variables(stage_vars, None)
            # → ['Pressure', 'Velocity']
            CODAStats._select_variables(stage_vars, 'Pressure')
            # → ['Pressure']
        """
        available = [v for v in stage_vars if v not in excluded]
        if variables is None:
            return available

        if isinstance(variables, str):
            variables = [variables]

        missing = [v for v in variables if v not in stage_vars]
        if missing:
            raise KeyError(
                f"Variable(s) {missing} not found. Available: {available}."
            )
        return list(variables)

    @staticmethod
    def _expand_vector_fields(
        stage_vars: dict,
        var_list: list,
        vector_handling: Literal['magnitude', 'components'],
    ) -> dict:
        """
        Reduce every requested variable to one (or more) 2-D scalar
        arrays of shape ``(n_points, n_cases)``, expanding vector fields
        according to ``vector_handling``.

        Scalar variables (``ndim == 2``) are passed through unchanged.
        Vector variables (``ndim == 3``, shape
        ``(n_dim, n_points, n_cases)``) are reduced according to
        ``vector_handling``:

        * ``'magnitude'`` — the Euclidean norm across the component axis
          (``np.linalg.norm(arr, axis=0)``) is used as a single combined
          scalar field, keeping the original variable name.
        * ``'components'`` — each spatial component becomes an
          independent scalar field, named ``'<var>_<i>'``.

        Parameters
        ----------
        stage_vars : dict
            Mapping from variable name to its array.
        var_list : list[str]
            Variable names to process (already resolved, e.g. via
            :meth:`_select_variables`).
        vector_handling : 'magnitude' or 'components'
            Reduction strategy for vector fields.

        Returns
        -------
        dict[str, np.ndarray]
            Maps each scalar field label to its ``(n_points, n_cases)``
            array.

        Raises
        ------
        ValueError
            If ``vector_handling`` is not one of the supported options,
            or if a variable has an unsupported number of dimensions.

        Examples
        --------
        ::

            stage_vars = {
                'Pressure': np.random.rand(500, 10),         # scalar
                'Velocity': np.random.rand(3, 500, 10),       # vector
            }
            fields = CODAStats._expand_vector_fields(
                stage_vars, ['Pressure', 'Velocity'], 'magnitude',
            )
            print(list(fields))               # ['Pressure', 'Velocity']
            print(fields['Velocity'].shape)    # (500, 10)

            fields = CODAStats._expand_vector_fields(
                stage_vars, ['Velocity'], 'components',
            )
            print(list(fields))   # ['Velocity_0', 'Velocity_1', 'Velocity_2']
        """
        fields: dict = {}
        for var in var_list:
            arr = stage_vars[var]
            if arr.ndim == 2:
                fields[var] = arr
            elif arr.ndim == 3:
                if vector_handling == 'magnitude':
                    fields[var] = np.linalg.norm(arr, axis=0)
                elif vector_handling == 'components':
                    for i in range(arr.shape[0]):
                        fields[f'{var}_{i}'] = arr[i]
                else:
                    raise ValueError(
                        f"vector_handling '{vector_handling}' not "
                        "supported. Options: 'magnitude', 'components'."
                    )
            else:
                raise ValueError(
                    f"Variable '{var}' has unsupported ndim {arr.ndim}."
                )
        return fields

    @staticmethod
    def _describe_array(
        values: np.ndarray,
        percentiles: Union[list, tuple],
        label: str,
        case_idx: Union[int, None] = None,
        n_cases: Union[int, None] = None,
    ) -> dict:
        """
        Build a single descriptive-statistics record for a 1-D array of
        values (e.g. every point of one case, or every point of every
        pooled case for one variable).

        Non-finite values (``NaN`` / ``Inf``) are dropped before computing
        any statistic, so that a few invalid points (e.g. coming from a
        diverged solver iteration or a failed interpolation) do not
        silently propagate into every metric.

        Parameters
        ----------
        values : np.ndarray
            1-D (or reshape-to-1-D) array of values to summarise.
        percentiles : list[float] or tuple[float]
            Percentiles (0-100) to compute.
        label : str
            Variable (or vector-component) name, recorded under the
            ``'variable'`` key.
        case_idx : int or None
            If given, recorded under the ``'case_idx'`` key — used for
            per-case statistics. Default None.
        n_cases : int or None
            If given, recorded under the ``'n_cases'`` key — used for
            pooled / global statistics. Default None.

        Returns
        -------
        dict
            Keys: ``'variable'``, optionally ``'case_idx'``, ``'mean'``,
            ``'std'``, ``'min'``, ``'max'``, ``'n_points'``, optionally
            ``'n_cases'``, and one ``'p<value>'`` key per requested
            percentile (e.g. ``'p50'``).

        Examples
        --------
        ::

            record = CODAStats._describe_array(
                np.array([1.0, 2.0, np.nan, 3.0]),
                percentiles=(25, 50, 75),
                label='Pressure',
                case_idx=4,
            )
            print(record['mean'], record['n_points'])   # 2.0  3
        """
        values   = np.asarray(values, dtype=np.float64).reshape(-1)
        finite   = values[np.isfinite(values)]
        has_data = finite.size > 0

        record: dict = {'variable': label}
        if case_idx is not None:
            record['case_idx'] = case_idx

        record.update({
            'mean':     float(np.mean(finite)) if has_data else np.nan,
            'std':      float(np.std(finite))  if has_data else np.nan,
            'min':      float(np.min(finite))  if has_data else np.nan,
            'max':      float(np.max(finite))  if has_data else np.nan,
            'n_points': int(finite.size),
        })
        if n_cases is not None:
            record['n_cases'] = n_cases

        for p in percentiles:
            record[f"p{p:g}"] = (
                float(np.percentile(finite, p)) if has_data else np.nan
            )

        return record

    @staticmethod
    def _resolve_stage_pairs(
        stages: Union[list, tuple],
        pairs: Union[Literal['consecutive', 'all'], list],
    ) -> list:
        """
        Build the list of ``(stage_a, stage_b)`` pairs to compare from a
        sequence of stages and a pairing strategy.

        Parameters
        ----------
        stages : list or tuple
            Sequence of stage keys.
        pairs : 'consecutive', 'all' or list[tuple]
            Pairing strategy. See
            :meth:`CODAStats.stage_difference_stats` for the full
            description of each option.

        Returns
        -------
        list[tuple]
            ``(stage_a, stage_b)`` pairs, in the order they should be
            processed.

        Raises
        ------
        ValueError
            If ``pairs`` is an unsupported string, or an explicit pair
            references a stage not present in ``stages``.
        TypeError
            If ``pairs`` is neither a recognised string nor a list/tuple.

        Examples
        --------
        ::

            CODAStats._resolve_stage_pairs((0, 1, 2), 'consecutive')
            # → [(0, 1), (1, 2)]

            CODAStats._resolve_stage_pairs((0, 1, 2), 'all')
            # → [(0, 1), (0, 2), (1, 2)]

            CODAStats._resolve_stage_pairs((0, 1, 2), [(0, 2)])
            # → [(0, 2)]
        """
        stages = list(stages)

        if isinstance(pairs, str):
            if pairs == 'consecutive':
                return [
                    (stages[i], stages[i + 1])
                    for i in range(len(stages) - 1)
                ]
            elif pairs == 'all':
                return list(combinations(stages, 2))
            else:
                raise ValueError(
                    f"pairs '{pairs}' not supported. Options: "
                    "'consecutive', 'all', or an explicit list of "
                    "(stage_a, stage_b) tuples."
                )

        if isinstance(pairs, (list, tuple)):
            stage_set = set(stages)
            for stage_a, stage_b in pairs:
                if stage_a not in stage_set or stage_b not in stage_set:
                    raise ValueError(
                        f"Pair ({stage_a}, {stage_b}) references a stage "
                        f"not present in stages={stages}."
                    )
            return list(pairs)

        raise TypeError(
            "pairs must be 'consecutive', 'all', or a list of "
            "(stage_a, stage_b) tuples."
        )

    @staticmethod
    def _bin_coordinate(coord_values: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Bin a 1-D coordinate array into ``n_bins`` quantile-based bins and
        return, for every input point, the midpoint of the bin it falls
        into — formatted as a string, ready to use as a categorical
        boxplot ``x`` label.

        Quantile binning (``pd.qcut``) is used instead of fixed-width
        binning so that every bin holds roughly the same number of
        points even when the mesh is non-uniformly distributed along the
        chosen coordinate (e.g. denser near a leading edge or a wall).
        If ``coord_values`` has too many repeated values for ``n_bins``
        distinct quantile edges to exist, this falls back to fixed-width
        binning (``pd.cut``).

        Parameters
        ----------
        coord_values : np.ndarray, shape (n_points,)
            Coordinate values to bin (one component of ``Coord``).
        n_bins : int
            Requested number of bins.

        Returns
        -------
        np.ndarray[str], shape (n_points,)
            Bin-midpoint label for every input point, e.g. ``'0.123'``.

        Examples
        --------
        ::

            labels = CODAStats._bin_coordinate(coord[:, 0], n_bins=10)
            df = pd.DataFrame({'coord_bin': labels, 'diff': diff_values})
            order = sorted(df['coord_bin'].unique(), key=float)
            sns.boxplot(data=df, x='coord_bin', y='diff', order=order)
        """
        series = pd.Series(coord_values)
        try:
            binned = pd.qcut(series, q=n_bins, duplicates='drop')
        except ValueError:
            binned = pd.cut(series, bins=n_bins)

        mids = binned.apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
        return np.array([f"{m:.3g}" if pd.notna(m) else "nan" for m in mids])

    @staticmethod
    def _plot_diff_boxplot(
        df_point: pd.DataFrame,
        label: str,
        stage_a,
        stage_b,
        diff_mode: str,
        coord_idx: int,
        figsize: tuple,
        save_dir: Union[str, None],
        **kwargs_plot
    ) -> None:
        """
        Draw a boxplot of the point-wise difference (``df_point['diff']``)
        grouped by spatial coordinate bin (``df_point['coord_bin']``).

        Bins are ordered numerically along the ``x`` axis (by their
        midpoint value) regardless of their first-appearance order in
        ``df_point``, so the plot always reads left-to-right as
        increasing coordinate.

        Parameters
        ----------
        df_point : pd.DataFrame
            Must contain the ``'coord_bin'`` and ``'diff'`` columns, as
            produced by :meth:`stage_difference_stats_variable`.
        label : str
            Variable (or vector-component) name, used in the title.
        stage_a, stage_b : int or str
            Stage keys being compared, used in the title.
        diff_mode : str
            ``'absolute'`` or ``'relative'``, used to label the ``y``
            axis.
        coord_idx : int
            Coordinate axis index used for binning, used to label the
            ``x`` axis.
        figsize : tuple
            Figure size.
        save_dir : str or None
            If given, the figure is saved to
            ``<save_dir>/<label>_<stage_a>_vs_<stage_b>_boxplot.png``
            and closed; otherwise ``plt.show()`` is called.

        Examples
        --------
        ::

            CODAStats._plot_diff_boxplot(
                df_point, 'Pressure', 0, 1, 'absolute', 0, (9, 5), None,
            )
        """
        order = sorted(df_point['coord_bin'].unique(), key=float)

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            data=df_point,
            x='coord_bin',
            y='diff',
            order=order,
            ax=ax,
            flierprops = {"markersize": kwargs_plot.get("marker_size", 4)}, showfliers = kwargs_plot.get("showfliers", True))

        ax.set_xlabel(f"Coord[{coord_idx}] (bin midpoint)")
        ax.set_ylabel(f"{diff_mode.capitalize()} diff")
        ax.set_title(
            f"{label}: stage {stage_a} → {stage_b}  —  diff by coordinate bin"
        )
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        fig.tight_layout()

        if save_dir:
            fname = f"{label}_{stage_a}_vs_{stage_b}_boxplot.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def _plot_diff_scatter(
        df_case: pd.DataFrame,
        label: str,
        stage_a,
        stage_b,
        flcc_vars_use: list,
        case_metric: str,
        annotate_case_idx: bool,
        figsize: tuple,
        save_dir: Union[str, None],
        **kwargs_plot
    ) -> None:
        """
        Draw a scatter plot of ``case_metric`` across the design / flight
        condition space, one point per case.

        If ``flcc_vars_use`` has two (or more) entries, the first two are
        used as the ``x``/``y`` axes and ``case_metric`` colours each
        point (with a colourbar) — mirroring the convention used by
        ``LEGOLAS.Residuals.plot_vs_params``. If only one entry is given,
        a simple 2-D scatter of ``flcc_vars_use[0]`` vs. ``case_metric``
        is drawn instead.

        Parameters
        ----------
        df_case : pd.DataFrame
            Must contain ``'case_idx'``, every entry of
            ``flcc_vars_use`` and ``case_metric`` as columns, as produced
            by :meth:`stage_difference_stats_variable`.
        label : str
            Variable (or vector-component) name, used in the title.
        stage_a, stage_b : int or str
            Stage keys being compared, used in the title.
        flcc_vars_use : list[str]
            One or two design-variable column names to use as axes.
        case_metric : str
            Column of ``df_case`` to use as colour (or as the ``y`` axis,
            when ``flcc_vars_use`` has a single entry).
        annotate_case_idx : bool
            If True, annotate each point with its ``'case_idx'`` value.
        figsize : tuple
            Figure size.
        save_dir : str or None
            If given, the figure is saved to
            ``<save_dir>/<label>_<stage_a>_vs_<stage_b>_scatter.png``
            and closed; otherwise ``plt.show()`` is called.

        Examples
        --------
        ::

            CODAStats._plot_diff_scatter(
                df_case, 'Pressure', 0, 1, ['AoA', 'Mach'],
                'max_abs_diff', True, (9, 5), None,
            )
        """

        # ------------------------------------------------------------
        # Scatter normal (1 ó 2 variables) o PCA (>2 variables)
        # ------------------------------------------------------------

        projection = kwargs_plot.pop("projection", "auto")
        scale_before_pca = kwargs_plot.pop("scale_before_pca", True)
        pca_dim = kwargs_plot.pop("pca_dim", 2)

        norm = kwargs_plot.pop("norm", "linear")
        linthresh = kwargs_plot.pop("linthresh", 1e-3)

        if projection == "auto":
            projection = "normal" if len(flcc_vars_use) <= 2 else "pca2"
            
        if norm == "linear":
            color_norm = None
        elif norm == "log":
            color_norm = LogNorm()
        elif norm == "symlog":
            color_norm = SymLogNorm(
                linthresh=kwargs_plot.pop("linthresh", 1e-3)
            )
        else:
            raise ValueError(f"Unknown norm '{norm}'.")

        fig, ax = plt.subplots(figsize=figsize)

        # ============================================================
        # Caso 1: hasta dos variables
        # ============================================================

        if len(flcc_vars_use) <= 2:

            if len(flcc_vars_use) == 2:

                x_var, y_var = flcc_vars_use

                sc = ax.scatter(
                    df_case[x_var],
                    df_case[y_var],
                    c=df_case[case_metric],
                    cmap='viridis',
                    s=80,
                    edgecolor='k',
                    norm=color_norm,
                    **kwargs_plot,
                )

                fig.colorbar(sc, ax=ax, label=case_metric)

                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)

                xs = df_case[x_var].values
                ys = df_case[y_var].values

            else:

                x_var = flcc_vars_use[0]

                ax.scatter(
                    df_case[x_var],
                    df_case[case_metric],
                    s=80,
                    edgecolor='k',
                    **kwargs_plot,
                )

                ax.set_xlabel(x_var)
                ax.set_ylabel(case_metric)

                xs = df_case[x_var].values
                ys = df_case[case_metric].values


        # ============================================================
        # Caso 2: PCA automático
        # ============================================================

        else:
            
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            
            X = df_case[flcc_vars_use].to_numpy()

            scale_before_pca = kwargs_plot.pop("scale_before_pca", True)

            if scale_before_pca:
                X = StandardScaler().fit_transform(X)

            pca_dim = kwargs_plot.pop("pca_dim", 2)

            if pca_dim not in (2, 3):
                raise ValueError("'pca_dim' must be 2 or 3.")

            pca = PCA(n_components=pca_dim)

            Xred = pca.fit_transform(X)

            if pca_dim == 2:

                sc = ax.scatter(
                    Xred[:, 0],
                    Xred[:, 1],
                    c=df_case[case_metric],
                    cmap="viridis",
                    s=80,
                    edgecolor="k",
                    norm=color_norm,
                    **kwargs_plot,
                )

                fig.colorbar(sc, ax=ax, label=case_metric)

                ax.set_xlabel(
                    f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
                )

                ax.set_ylabel(
                    f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
                )

                xs = Xred[:, 0]
                ys = Xred[:, 1]

            else:

                ax.remove()

                ax = fig.add_subplot(111, projection="3d")

                sc = ax.scatter(
                    Xred[:, 0],
                    Xred[:, 1],
                    Xred[:, 2],
                    c=df_case[case_metric],
                    cmap="viridis",
                    s=80,
                    edgecolor="k",
                    norm=color_norm,
                    **kwargs_plot,
                )

                fig.colorbar(sc, ax=ax, label=case_metric)

                ax.set_xlabel(
                    f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
                )

                ax.set_ylabel(
                    f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
                )

                ax.set_zlabel(
                    f"PC3 ({100*pca.explained_variance_ratio_[2]:.1f}%)"
                )

                xs = Xred[:, 0]
                ys = Xred[:, 1]
                
        # if len(flcc_vars_use) >= 2:
        #     x_var, y_var = flcc_vars_use[:2]
        #     norm = kwargs_plot.pop("norm", "linear")

        #     if norm == "linear":
        #         color_norm = None

        #     elif norm == "log":
        #         color_norm = LogNorm()

        #     elif norm == "symlog":
        #         color_norm = SymLogNorm(
        #             linthresh=kwargs_plot.pop("linthresh",1e-3)
        #         )
        #     sc = ax.scatter(
        #         df_case[x_var], df_case[y_var], c=df_case[case_metric],
        #         cmap='viridis', s=80, edgecolor='k', norm=color_norm
        #     )
        #     fig.colorbar(sc, ax=ax, label=case_metric)
        #     ax.set_xlabel(x_var)
        #     ax.set_ylabel(y_var)
        #     xs, ys = df_case[x_var].values, df_case[y_var].values
            
        # else: # (cambiar para usar PCA)
        #     x_var = flcc_vars_use[0]
        #     ax.scatter(df_case[x_var], df_case[case_metric], s=80, edgecolor='k')
        #     ax.set_xlabel(x_var)
        #     ax.set_ylabel(case_metric)
        #     xs, ys = df_case[x_var].values, df_case[case_metric].values

        if annotate_case_idx:
            for x, y, ci in zip(xs, ys, df_case['case_idx']):
                ax.annotate(
                    str(int(ci)), (x, y), textcoords="offset points",
                    xytext=(0, 6), ha='center', fontsize=7,
                )

        ax.set_title(
            f"{label}: stage {stage_a} → {stage_b}  —  "
            f"{case_metric} vs design space"
        )
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()

        if save_dir:
            fname = f"{label}_{stage_a}_vs_{stage_b}_scatter.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def _plot_diff_violinplot(
        df_point: pd.DataFrame,
        label: str,
        stage_a,
        stage_b,
        diff_mode: str,
        coord_idx: int,
        figsize: tuple,
        save_dir: Union[str, None],
        **kwargs_plot
    ) -> None:
        """
        Draw a violin plot of the point-wise difference (``df_point['diff']``)
        grouped by spatial coordinate bin (``df_point['coord_bin']``).

        Bins are ordered numerically along the ``x`` axis (by their
        midpoint value) regardless of their first-appearance order in
        ``df_point``, so the plot always reads left-to-right as
        increasing coordinate.

        Parameters
        ----------
        df_point : pd.DataFrame
            Must contain the ``'coord_bin'`` and ``'diff'`` columns, as
            produced by :meth:`stage_difference_stats_variable`.
        label : str
            Variable (or vector-component) name, used in the title.
        stage_a, stage_b : int or str
            Stage keys being compared, used in the title.
        diff_mode : str
            ``'absolute'`` or ``'relative'``, used to label the ``y``
            axis.
        coord_idx : int
            Coordinate axis index used for binning, used to label the
            ``x`` axis.
        figsize : tuple
            Figure size.
        save_dir : str or None
            If given, the figure is saved to
            ``<save_dir>/<label>_<stage_a>_vs_<stage_b>_boxplot.png``
            and closed; otherwise ``plt.show()`` is called.

        Examples
        --------
        ::

            CODAStats._plot_diff_violinplot(
                df_point, 'Pressure', 0, 1, 'absolute', 0, (9, 5), None,
            )
        """
        order = sorted(df_point['coord_bin'].unique(), key=float)

        fig, ax = plt.subplots(figsize=figsize)
        sns.violinplot(data=df_point, x='coord_bin', y='diff', order=order, ax=ax, **kwargs_plot)
        ax.set_xlabel(f"Coord[{coord_idx}] (bin midpoint)")
        ax.set_ylabel(f"{diff_mode.capitalize()} diff")
        ax.set_title(
            f"{label}: stage {stage_a} → {stage_b}  —  diff by coordinate bin"
        )
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        fig.tight_layout()

        if save_dir:
            fname = f"{label}_{stage_a}_vs_{stage_b}_boxplot.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    @staticmethod
    def _plot_diff_histogram(
        df_point: pd.DataFrame,
        label: str,
        stage_a,
        stage_b,
        diff_mode: str,
        figsize: tuple,
        save_dir: Union[str, None],
        **kwargs_plot,
    ) -> None:
        """
        Draw a histogram (with KDE) of the pooled point-wise difference
        (``df_point['diff']``), ignoring both spatial position and
        design space.

        Parameters
        ----------
        df_point : pd.DataFrame
            Must contain the ``'diff'`` column, as produced by
            :meth:`stage_difference_stats_variable`.
        label : str
            Variable (or vector-component) name, used in the title.
        stage_a, stage_b : int or str
            Stage keys being compared, used in the title.
        diff_mode : str
            ``'absolute'`` or ``'relative'``, used to label the ``x``
            axis.
        figsize : tuple
            Figure size.
        save_dir : str or None
            If given, the figure is saved to
            ``<save_dir>/<label>_<stage_a>_vs_<stage_b>_histogram.png``
            and closed; otherwise ``plt.show()`` is called.

        Examples
        --------
        ::

            CODAStats._plot_diff_histogram(
                df_point, 'Pressure', 0, 1, 'absolute', (9, 5), None,
            )
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(df_point['diff'], kde=True, ax=ax, bins=kwargs_plot.get('bins', 100))
        ax.set_xlabel(f"{diff_mode.capitalize()} diff")
        
        ax.set_xscale(kwargs_plot.get('xscale', 'linear'))
    
        ax.set_yscale(kwargs_plot.get('yscale', 'linear'))
        
        ax.set_title(
            f"{label}: stage {stage_a} → {stage_b}  —  overall diff distribution"
        )
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()

        if save_dir:
            fname = f"{label}_{stage_a}_vs_{stage_b}_histogram.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            
    @staticmethod
    def _plot_diff_histogram_2D(
        df_point,
        label,
        stage_a,
        stage_b,
        diff_mode,
        figsize,
        save_dir,
        **kwargs_plot,
    ):

        fig, ax = plt.subplots(figsize=figsize)

        sns.histplot(
            data=df_point,
            x="coord_bin",
            y="diff",
            hue = "coord_bin",
            cbar=True,
            ax=ax,
            **kwargs_plot,
        )

        ax.set_title(
            f"{label}: stage {stage_a} → {stage_b}"
        )
        ax.set_xlabel(f"{diff_mode.capitalize()} diff")

        fig.tight_layout()

        if save_dir:
            fname = f"{label}_{stage_a}_vs_{stage_b}_histogram2D.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_heatmap():
        pass