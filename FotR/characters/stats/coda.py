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

import warnings
from itertools import combinations
from typing import Literal, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

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

    def stage_difference_stats_variable(
        id_group: str,
        stages: Union[list, tuple],
        variable: str,
        pairs: Union[Literal['consecutive', 'all'], list] = 'consecutive',
        cases_idx: Union[list, tuple, int, str] = 'all',
        diff_mode: Literal['absolute', 'relative'] = 'absolute',
        vector_handling: Literal['magnitude', 'components'] = 'magnitude',
        plots: dict = {'boxplot': True, 'histogram': True},
        verbose: bool = False,
    ):
        # Método para pintar la distribución del error/diferencia entre stages de cada variable respecto a la posición geométrica, a los demás casos o respecto a su posición global.
        
        
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