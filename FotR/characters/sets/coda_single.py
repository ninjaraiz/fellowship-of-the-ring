"""
sets/coda_single.py
===================
Sets class for the CODA_SINGLE format: one mesh per case.

Why this is not ``CODASets`` with a branch
-------------------------------------------
``CODASets`` is built on a single shared mesh: ``data_dict`` holds one
``Coord (n_points, n_dim)`` and every variable is a dense
``(n_points, n_cases)`` matrix whose case axis lines up with ``FlCc``.

CODA_SINGLE has one mesh per case, so ``Coord`` and ``Vars[stage][var]``
are **lists aligned with ``case_order``** with different ``n_points_i``.
There is no common point axis, therefore nothing can be a matrix indexed
by (point, case), and every dense operation of ``CODASets`` fails.

Crucially, **none of the things this format is for needs a common mesh**:
comparing meshes *is* the point of a convergence study, a pointwise ML
surrogate is happy with diverse sampling, per-mesh export is per-mesh by
definition, and variable algebra is intra-case. A common mesh is only
needed to reuse the dense machinery — so resampling is offered as an
explicit operation (:meth:`CODASingleSets.sample_on`), never as the
default. Interpolating to a reference mesh in a GCI study would destroy
the very signal being measured.

Layout consumed
---------------
``data_dict['CADGroup_<id>']`` as written by ``CODASingleReader``::

    'FlCc'             np.ndarray (n_cases, n_dvars)   # the only dense one
    'Coord'            list[(n_points_i, n_dim)]
    'NodeCoord'        list[(n_nodes_i,  n_dim)]
    'Conec'            list[(n_cells_i, max_nodes)]
    'eltype'           list[(n_cells_i,)]
    'idx_sort'         list[{stage: sorter}]
    'case_order'       list[str]                       # folder names
    'stages_available' list[list[int]]
    'Vars'[stage][var] list[(n_points_i,) | (n_points_i, n_dims) | None]

Entries may be ``None`` when a case lacks that stage or variable, which
is routine while a study is still running.
"""

import logging
import os
import warnings
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..sam import SAM
from .base import BaseSets

if TYPE_CHECKING:
    from ..frodo import FRODO

log = logging.getLogger(__name__)

#: Variables that are bookkeeping, not physics.
EXCLUDED_VARS = ('GlobalNumber', 'CADGroupID')


class CaseRef:
    """
    Lazy handle to one case of one CADGroup.

    Holds *the address and the logic*, not the mesh: the case folder, the
    CAD group ids, the ``.vtu`` flavour and the sort permutation are
    enough to rebuild any array on demand. If the reader already
    materialised the arrays (the default, eager path) they are returned
    straight from ``data_dict``; otherwise the ``.vtu`` is re-read through
    ``CODASingleReader.read_case_geometry`` / ``read_case_vars``, so the
    two paths cannot drift apart.

    Re-reading is cheap: a 78 MB surface ``.vtu`` parses in ~0.04 s.

    Parameters
    ----------
    db : FRODO
        Parent database.
    group_key : str
        ``'CADGroup_<id>'`` key in ``data_dict``.
    position : int
        Index into the group's ``case_order`` (its local case axis).
    cache : bool
        Keep materialised arrays in the handle. Default False.

    Attributes
    ----------
    name : str
        Case folder name.
    stages : list[int]
        Stages available for this case.

    Examples
    --------
    ::

        ref = db.sets.case_refs('3')[0]
        ref.name, ref.npts
        cp = ref.var('BoundaryValues_CoefPressure', stage=1)
    """

    def __init__(
        self,
        db: 'FRODO',
        group_key: str,
        position: int,
        cache: bool = False,
    ) -> None:
        self.db = db
        self.group_key = group_key
        self.position = position
        self.cache = cache
        self._cached: dict = {}

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def _group(self) -> dict:
        return self.db.data_dict[self.group_key]

    @property
    def name(self) -> str:
        """Case folder name."""
        return self._group['case_order'][self.position]

    @property
    def mesh_file(self) -> Optional[str]:
        """Basename of the mesh that generated this case, if known."""
        files = self._group.get('mesh_files') or []
        return files[self.position] if self.position < len(files) else None

    @property
    def stages(self) -> list:
        """Stages available on disk for this case."""
        available = self._group.get('stages_available') or []
        if self.position < len(available):
            return list(available[self.position])
        return sorted(self._group['idx_sort'][self.position])

    @property
    def npts(self) -> int:
        """Number of points (cells) of this case's group."""
        return int(self.coord().shape[0])

    def __repr__(self) -> str:
        return (
            f"<CaseRef {self.name!r} group={self.group_key!r} "
            f"stages={self.stages}>"
        )

    # ── Geometry ──────────────────────────────────────────────────────

    def _stored(self, key: str):
        """Return the eager array for this case, or None if unavailable."""
        values = self._group.get(key)
        if not isinstance(values, list) or self.position >= len(values):
            return None
        return values[self.position]

    def _reference_stage(self) -> int:
        stages = self.stages
        if not stages:
            raise RuntimeError(
                f"Case '{self.name}' has no stage available; its geometry "
                "cannot be read."
            )
        return stages[0]

    def _read_geometry(self) -> dict:
        if 'geometry' in self._cached:
            return self._cached['geometry']
        group = self._group
        geo = self.db.reader.read_case_geometry(
            self.name,
            self._reference_stage(),
            group.get('group_ids', ()),
            vtu_type=group.get('vtu_type', 'surface'),
            sort_fn=self.db.reader.resolve_sort_fn(
                group.get('method_to_sort', 'lexsort')
            ),
        )
        if self.cache:
            self._cached['geometry'] = geo
        return geo

    def _geometry_key(self, key: str) -> np.ndarray:
        stored = self._stored(key)
        if stored is not None:
            return stored
        return self._read_geometry()[key]

    def coord(self) -> np.ndarray:
        """Cell-centroid coordinates, ``(n_points, n_dim)``."""
        return self._geometry_key('Coord')

    def nodes(self) -> np.ndarray:
        """Node coordinates, ``(n_nodes, n_dim)``."""
        return self._geometry_key('NodeCoord')

    def conec(self) -> np.ndarray:
        """Connectivity into :meth:`nodes`, ``(n_cells, max_nodes)``."""
        return self._geometry_key('Conec')

    def eltype(self) -> np.ndarray:
        """VTK cell types, row-aligned with :meth:`coord`."""
        return self._geometry_key('eltype')

    def sorter(self, stage: int) -> np.ndarray:
        """Cell permutation used for this case at ``stage``."""
        sorters = self._group['idx_sort'][self.position]
        if stage in sorters:
            return np.asarray(sorters[stage])
        return self._read_geometry()['idx']

    # ── Variables ─────────────────────────────────────────────────────

    def varnames(self, stage: Union[int, str]) -> list:
        """Variable names available for this case at ``stage``."""
        stage_vars = self._group.get('Vars', {}).get(str(stage), {})
        return [
            name for name, values in stage_vars.items()
            if isinstance(values, list)
            and self.position < len(values)
            and values[self.position] is not None
        ]

    def var(
        self,
        name: str,
        stage: Union[int, str],
    ) -> Optional[np.ndarray]:
        """
        One variable of this case at ``stage``.

        Returns ``None`` when the case genuinely has no such value (no
        such stage on disk, or the variable is missing from its ``.vtu``),
        which callers must treat as "not available", not as zero.

        Parameters
        ----------
        name : str
            Variable name.
        stage : int or str
            Stage key.

        Returns
        -------
        np.ndarray or None
            ``(n_points,)`` for scalars, ``(n_points, n_dims)`` for
            vectors.

        Examples
        --------
        ::

            cp = ref.var('BoundaryValues_CoefPressure', stage=1)
        """
        stage_vars = self._group.get('Vars', {}).get(str(stage), {})
        values = stage_vars.get(name)
        if isinstance(values, list) and self.position < len(values):
            stored = values[self.position]
            if stored is not None:
                return stored

        if int(stage) not in self.stages:
            return None

        group = self._group
        read = self.db.reader.read_case_vars(
            self.name, int(stage), group.get('group_ids', ()),
            sorter=self.sorter(int(stage)),
            vtu_type=group.get('vtu_type', 'surface'),
            var_names=[name],
        )
        return read.get(name)


class CODASingleSets(BaseSets):
    """
    Sets class for CODA_SINGLE-format FRODO databases.

    Operates on one geometry per case. Methods split by what they need:

    * **no common mesh** — :meth:`compute_var` (variable algebra inside a
      case), :meth:`reduce_cases` (one scalar per case, the feeder for
      convergence studies), :meth:`create_jset` (a long, per-case-block
      tensor for pointwise ML);
    * **a common mesh, requested explicitly** — :meth:`sample_on`, which
      resamples every case onto shared points and hands back the dense
      ``(n_points, n_cases)`` layout that ``CODASets`` understands.

    ``plot_wall_integrals`` is inherited from :class:`BaseSets`: it reads
    the solver's integral monitors and never touches a mesh.

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

        db.sets.compute_var('cp2', 'BoundaryValues_CoefPressure**2', stage=1,
                            id_group='3')
        table = db.sets.reduce_cases('cp2', stage=1, id_group='3', how='mean')
    """

    def __init__(self, db: 'FRODO') -> None:
        """Attach to the parent FRODO database (see BaseSets)."""
        super().__init__(db)

    # ── Group / case resolution ───────────────────────────────────────

    def _resolve_group(self, id_group: Union[str, int]) -> tuple:
        """Return ``(key, group_dict)`` for ``id_group``."""
        key = str(id_group)
        if not key.startswith('CADGroup_'):
            key = f'CADGroup_{key}'
        if key not in self.db.data_dict:
            raise KeyError(
                f"'{key}' not found in data_dict. Run extract_inputs() for "
                f"this group first. Available: "
                f"{[k for k in self.db.data_dict if k.startswith('CADGroup_')]}."
            )
        group = self.db.data_dict[key]
        if 'case_order' not in group:
            raise KeyError(
                f"'{key}' has no 'case_order'; it does not look like a "
                "CODA_SINGLE group."
            )
        return key, group

    def n_cases(self, id_group: Union[str, int]) -> int:
        """Number of cases extracted into ``id_group``."""
        _key, group = self._resolve_group(id_group)
        return len(group['case_order'])

    def case_refs(
        self,
        id_group: Union[str, int],
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        cache: bool = False,
    ) -> list:
        """
        Lazy handles for the selected cases of a group.

        Parameters
        ----------
        id_group : str or int
            CADGroup identifier (``'3'`` or ``'CADGroup_3'``).
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Selection on the group's **local** case axis (the same space
            as ``case_order``), normalised by
            ``BaseSets._normalise_cases_idx``. Default ``'all'``.
        cache : bool
            Let each handle keep what it materialises. Default False.

        Returns
        -------
        list[CaseRef]

        Examples
        --------
        ::

            for ref in db.sets.case_refs('3'):
                print(ref.name, ref.npts, ref.mesh_file)
        """
        key, group = self._resolve_group(id_group)
        positions = self._normalise_cases_idx(
            cases_idx, len(group['case_order'])
        )
        return [CaseRef(self.db, key, pos, cache=cache) for pos in positions]

    # ── M5 · variable algebra inside each case ────────────────────────

    def compute_var(
        self,
        name: str,
        formula: str,
        stage: Union[int, str],
        id_group: Union[str, int],
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        externals: Optional[dict] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> list:
        """
        Derive a new variable per case from the existing ones.

        The formula is evaluated **once per case**, on that case's own
        arrays, by the restricted parser
        :func:`SAM.Backpack.eval_formula` — no ``eval``. Available names:
        every variable of that case at ``stage``, every design variable
        (as a scalar, from ``FlCc``), anything in ``externals``, plus
        ``pi``, ``np`` and the functions ``sin cos tan exp log log10 sqrt
        abs minimum maximum clip``.

        Because each case is evaluated on its own mesh, this works with no
        common point axis — it is the intra-case half of "compute
        something indirect and then look at the trend across
        simulations"; the trend half is :meth:`reduce_cases`.

        Parameters
        ----------
        name : str
            Name of the new variable in ``Vars[stage]``.
        formula : str
            Expression, e.g. ``'(p - pinf) / q'``.
        stage : int or str
            Stage the inputs are read from and the result is written to.
        id_group : str or int
            CADGroup identifier.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to compute. Others keep ``None``. Default ``'all'``.
        externals : dict or None
            Extra scalars/arrays available to the formula.
        overwrite : bool
            Allow replacing an existing variable. Default False.
        verbose : bool
            Log one line per case.

        Returns
        -------
        list
            The per-case values just stored (``None`` where not computed),
            aligned with ``case_order``.

        Raises
        ------
        ValueError
            If ``name`` already exists and ``overwrite=False``, or the
            formula references an unknown name.
        KeyError
            If the group or the stage is missing.

        Examples
        --------
        Dynamic pressure ratio, then its trend across meshes::

            db.sets.compute_var(
                'cp_sq', 'BoundaryValues_CoefPressure**2',
                stage=1, id_group='3',
            )
            db.sets.reduce_cases('cp_sq', stage=1, id_group='3', how='mean')

        With an external constant::

            db.sets.compute_var(
                'dp', '(p - pinf)', stage=1, id_group='3',
                externals={'pinf': 101325.0},
            )
        """
        key, group = self._resolve_group(id_group)
        stage_key = str(stage)

        vars_all = group.setdefault('Vars', {})
        if stage_key not in vars_all:
            raise KeyError(
                f"Stage '{stage_key}' not found in '{key}'['Vars']. "
                f"Available: {list(vars_all)}. Run extract_outputs() first."
            )
        stage_vars = vars_all[stage_key]

        if name in stage_vars and not overwrite:
            raise ValueError(
                f"Variable '{name}' already exists in '{key}' stage "
                f"'{stage_key}'. Pass overwrite=True to replace it."
            )

        n_cases = len(group['case_order'])
        positions = set(self._normalise_cases_idx(cases_idx, n_cases))

        design_vars = list(self.db.metadata.get('design_vars') or [])
        flcc = np.asarray(group['FlCc'])
        allowed = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs,
            'minimum': np.minimum, 'maximum': np.maximum, 'clip': np.clip,
            'np': np,
        }

        results: list = []
        for pos in range(n_cases):
            if pos not in positions:
                previous = stage_vars.get(name)
                results.append(
                    previous[pos]
                    if isinstance(previous, list) and pos < len(previous)
                    else None
                )
                continue

            env = {'pi': np.pi, 'np': np}
            for i, dvar in enumerate(design_vars):
                if i < flcc.shape[1] and pos < flcc.shape[0]:
                    env[dvar] = float(flcc[pos, i])
            for var_name, values in stage_vars.items():
                if (isinstance(values, list) and pos < len(values)
                        and values[pos] is not None):
                    env[var_name] = values[pos]
            if externals:
                env.update(externals)

            case_name = group['case_order'][pos]

            # A case with no field at all for this stage is an unfinished
            # simulation, not a broken formula: record None quietly
            # instead of warning about an expression that is fine.
            has_fields = any(
                name in env for name in stage_vars
                if name not in EXCLUDED_VARS
            )
            if not has_fields:
                log.debug(
                    "%s, case '%s': no field at stage %s; '%s' not computed.",
                    key, case_name, stage_key, name,
                )
                results.append(None)
                continue

            try:
                value = SAM.Backpack.eval_formula(formula, env, allowed)
            except Exception as exc:
                warnings.warn(
                    f"{key}, case '{case_name}': could not evaluate "
                    f"'{formula}' ({exc}); storing None.",
                    UserWarning,
                )
                results.append(None)
                continue

            value = np.asarray(value, dtype=np.float64)
            if value.ndim == 0:
                # A formula made only of design vars/constants is still a
                # field: broadcast it over the case's points so that the
                # result stays usable wherever a field is expected.
                npts = np.shape(group['Coord'][pos])[0]
                value = np.full(npts, float(value), dtype=np.float64)

            if verbose:
                log.info(
                    "[CODASingleSets] %s / %s: %s = %s -> %s",
                    key, case_name, name, formula, value.shape,
                )
            results.append(value)

        stage_vars[name] = results
        return results

    # ── M1 · one value per case: the convergence/trend feeder ─────────

    #: Reductions accepted by :meth:`reduce_cases`.
    REDUCTIONS = {
        'mean':   np.nanmean,
        'median': np.nanmedian,
        'min':    np.nanmin,
        'max':    np.nanmax,
        'std':    np.nanstd,
        'sum':    np.nansum,
        'rms':    lambda a: float(np.sqrt(np.nanmean(np.square(a)))),
        'absmax': lambda a: float(np.nanmax(np.abs(a))),
    }

    def reduce_cases(
        self,
        variables: Union[str, list, tuple],
        stage: Union[int, str],
        id_group: Union[str, int],
        how: Union[str, Callable] = 'mean',
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        dropna: bool = False,
    ) -> pd.DataFrame:
        """
        Reduce each case to one number per variable.

        This is what makes meshes comparable without touching them: a
        field of ``n_points_i`` values collapses to a scalar, so the
        result is a tidy table indexed by case — the input a convergence
        study (GCI/Richardson) or a trend plot needs.

        Parameters
        ----------
        variables : str or list[str]
            Variable name(s) in ``Vars[stage]``. Bookkeeping variables
            (``GlobalNumber``, ``CADGroupID``) are rejected explicitly.
        stage : int or str
            Stage to read.
        id_group : str or int
            CADGroup identifier.
        how : str or callable
            Reduction: ``'mean'``, ``'median'``, ``'min'``, ``'max'``,
            ``'std'``, ``'sum'``, ``'rms'``, ``'absmax'``, or any callable
            taking a 1-D array and returning a float. Default ``'mean'``.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to include. Default ``'all'``.
        dropna : bool
            Drop cases whose value could not be computed (no such stage,
            variable absent). Default False — an unfinished case is
            information, not noise.

        Returns
        -------
        pd.DataFrame
            One row per case. Columns: ``'case_idx'`` (local position),
            ``'case'`` (folder), ``'mesh_file'``, ``'n_points'``,
            ``'stages_available'``, every design variable, and one column
            per requested variable.

        Raises
        ------
        KeyError
            If the group or stage is missing.
        ValueError
            If ``how`` is an unknown string, or a variable is one of the
            bookkeeping ones.

        Examples
        --------
        Trend of a coefficient across a mesh-refinement family::

            table = db.sets.reduce_cases(
                'BoundaryValues_CoefPressure', stage=1, id_group='3',
                how='rms',
            )
            table[['mesh', 'n_points', 'BoundaryValues_CoefPressure']]

        Several variables at once, with a custom reduction::

            db.sets.reduce_cases(
                ['cp', 'cf'], stage=1, id_group='3',
                how=lambda a: float(np.percentile(a, 95)),
            )
        """
        key, group = self._resolve_group(id_group)
        stage_key = str(stage)

        stage_vars = group.get('Vars', {}).get(stage_key)
        if stage_vars is None:
            raise KeyError(
                f"Stage '{stage_key}' not found in '{key}'['Vars']. "
                f"Available: {list(group.get('Vars', {}))}."
            )

        if isinstance(variables, str):
            variables = [variables]
        variables = list(variables)
        bad = [v for v in variables if v in EXCLUDED_VARS]
        if bad:
            raise ValueError(
                f"{bad} are bookkeeping variables, not physical fields; "
                "reducing them is meaningless."
            )
        missing = [v for v in variables if v not in stage_vars]
        if missing:
            raise KeyError(
                f"Variable(s) {missing} not found in '{key}' stage "
                f"'{stage_key}'. Available: "
                f"{[v for v in stage_vars if v not in EXCLUDED_VARS]}."
            )

        if isinstance(how, str):
            if how not in self.REDUCTIONS:
                raise ValueError(
                    f"how='{how}' not supported. "
                    f"Options: {list(self.REDUCTIONS)}, or a callable."
                )
            reduce_fn = self.REDUCTIONS[how]
        elif callable(how):
            reduce_fn = how
        else:
            raise ValueError("how must be a string or a callable.")

        design_vars = list(self.db.metadata.get('design_vars') or [])
        flcc = np.asarray(group['FlCc'])
        refs = self.case_refs(id_group, cases_idx)

        records = []
        for ref in refs:
            pos = ref.position
            row = {
                'case_idx':         pos,
                'case':             ref.name,
                'mesh_file':        ref.mesh_file,
                'stages_available': list(ref.stages),
            }
            try:
                row['n_points'] = int(np.shape(group['Coord'][pos])[0])
            except Exception:
                row['n_points'] = np.nan
            for i, dvar in enumerate(design_vars):
                row[dvar] = (
                    float(flcc[pos, i])
                    if pos < flcc.shape[0] and i < flcc.shape[1] else np.nan
                )

            for var_name in variables:
                values = stage_vars[var_name]
                value = (
                    values[pos]
                    if isinstance(values, list) and pos < len(values) else None
                )
                if value is None:
                    row[var_name] = np.nan
                    continue
                flat = np.asarray(value, dtype=np.float64).reshape(-1)
                finite = flat[np.isfinite(flat)]
                row[var_name] = (
                    float(reduce_fn(finite)) if finite.size else np.nan
                )
            records.append(row)

        table = pd.DataFrame.from_records(records)
        if dropna and variables:
            table = table.dropna(subset=variables).reset_index(drop=True)
        return table

    # ── M2 · long joint tensor for pointwise ML ───────────────────────

    def create_jset(
        self,
        stage: Union[int, str],
        id_group: Union[str, int],
        variables: Union[str, list, tuple, None] = None,
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        save_path: Union[str, None] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Assemble a **long** joint tensor: one row per (point, case).

        Layout per row, identical to ``CODASets.create_jset``::

            [coords | design_vars | field_variables]

        The difference is vertical, not horizontal: with one mesh per case
        the blocks have different heights, so the result cannot be
        reshaped back to ``(n_points, n_cases)``. ``'case_offsets'`` and
        ``'case_ids'`` are returned so any case can still be sliced out.

        Implementation reuses the existing machinery rather than
        duplicating it: ``SAM.Gardener.create_final_tensor`` is called
        once per case (with that case's own ``Coord`` and a single row of
        ``FlCc``) and the blocks are joined with
        ``SAM.Gardener.concatenate_sets``. ``'scaled'`` is then recomputed
        over the concatenated tensor, because ``concatenate_sets`` copies
        ``mins``/``maxs`` from the reference block but leaves each block
        scaled by its own.

        Parameters
        ----------
        stage : int or str
            Stage whose variables are assembled.
        id_group : str or int
            CADGroup identifier.
        variables : str, list[str] or None
            Variables to include, in order. ``None`` (default) takes every
            variable of the stage except ``GlobalNumber`` / ``CADGroupID``
            that is present in **every** selected case — a variable
            missing from one case would misalign the columns.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to include. Default ``'all'``.
        save_path : str or None
            If given, saves via ``BaseSets._save_result`` (``.h5``,
            ``.pt`` or ``.npy``).
        verbose : bool
            Log per-case progress.

        Returns
        -------
        dict
            ``'tensor'``, ``'scaled'``, ``'mins'``, ``'maxs'``, ``'info'``
            (as in CODA) plus ``'case_offsets'`` (``n_cases + 1`` row
            boundaries), ``'case_ids'`` (local case index per row) and
            ``'case_order'``.

        Side-effects
        ------------
        Sets ``db.jset`` and ``db.df_data`` (with a ``case`` column).

        Raises
        ------
        ValueError
            If no case or no variable survives the selection.

        Examples
        --------
        ::

            jset = db.sets.create_jset(stage=1, id_group='3')
            jset['tensor'].shape[0] == sum(jset['case_offsets'][-1:])
            block = jset['tensor'][
                jset['case_offsets'][2]:jset['case_offsets'][3]
            ]
        """
        import torch

        key, group = self._resolve_group(id_group)
        stage_key = str(stage)
        stage_vars = group.get('Vars', {}).get(stage_key)
        if stage_vars is None:
            raise KeyError(
                f"Stage '{stage_key}' not found in '{key}'['Vars']. "
                f"Available: {list(group.get('Vars', {}))}."
            )

        refs = self.case_refs(id_group, cases_idx)
        if not refs:
            raise ValueError(f"{key}: no case selected.")

        var_names = self._common_variables(stage_vars, refs, variables)
        if not var_names:
            raise ValueError(
                f"{key} stage '{stage_key}': no variable is available in "
                "every selected case; pass 'variables' explicitly or "
                "restrict 'cases_idx'."
            )

        design_vars = list(self.db.metadata.get('design_vars') or [])
        flcc = np.asarray(group['FlCc'])

        blocks, offsets, case_ids, kept = [], [0], [], []
        for ref in refs:
            pos = ref.position
            coord = np.asarray(ref.coord(), dtype=np.float64)

            tensors_out = []
            for var_name in var_names:
                value = np.asarray(
                    stage_vars[var_name][pos], dtype=np.float64
                )
                # create_final_tensor wants (n_points, n_cases[, n_out]);
                # here n_cases is 1, so a scalar field becomes (n, 1) and a
                # vector field (n, 1, n_dims).
                if value.ndim == 1:
                    tensors_out.append(value[:, None])
                elif value.ndim == 2:
                    tensors_out.append(value[:, None, :])
                else:
                    raise ValueError(
                        f"Variable '{var_name}' of case '{ref.name}' has "
                        f"unsupported ndim {value.ndim}."
                    )

            block = SAM.Gardener.create_final_tensor(
                tensor_ptos=coord,
                tensor_flcc=flcc[pos:pos + 1, :],
                tensors_out=tensors_out,
                tensors_aux=[],
                sol='all',
            )
            blocks.append(block)
            offsets.append(offsets[-1] + coord.shape[0])
            case_ids.extend([pos] * coord.shape[0])
            kept.append(ref.name)
            if verbose:
                log.info(
                    "[CODASingleSets] %s / %s: block %s",
                    key, ref.name, tuple(block['tensor'].shape),
                )

        result = (
            SAM.Gardener.concatenate_sets(tuple(blocks), ref=0)
            if len(blocks) > 1 else dict(blocks[0])
        )

        # concatenate_sets copies mins/maxs from the reference block and
        # leaves each block scaled by its own; renormalise globally so
        # 'scaled' is comparable across cases.
        tensor = result['tensor']
        mins = tensor.min(dim=0, keepdim=True)[0]
        maxs = tensor.max(dim=0, keepdim=True)[0]
        denom = maxs - mins
        denom[denom == 0] = 1e-8
        result['scaled'] = (tensor - mins) / denom
        result['mins'] = mins.squeeze()
        result['maxs'] = maxs.squeeze()

        result['case_offsets'] = np.asarray(offsets, dtype=np.int64)
        result['case_ids'] = np.asarray(case_ids, dtype=np.int64)
        result['case_order'] = kept

        if save_path:
            self._save_result(result, save_path)
            if verbose:
                log.info("[CODASingleSets] jset saved to %s", save_path)

        self.db.jset = result

        coord_cols = self._coord_columns(refs[0].coord().shape[1])
        columns = coord_cols + design_vars + list(var_names)
        n_cols = int(tensor.shape[1])
        if len(columns) != n_cols:
            warnings.warn(
                f"Column names ({len(columns)}) do not match tensor width "
                f"({n_cols}); storing an unnamed DataFrame.",
                UserWarning,
            )
            df = pd.DataFrame(tensor.numpy())
        else:
            df = pd.DataFrame(tensor.numpy(), columns=columns)
        df.insert(0, 'case', [kept[i] for i in
                              np.searchsorted(result['case_offsets'][1:],
                                              np.arange(len(df)),
                                              side='right')])
        self.db.df_data = df

        return result

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _coord_columns(n_dim: int) -> list:
        """Column names for an ``n_dim`` coordinate array."""
        if n_dim == 2:
            return ['x', 'z']
        if n_dim == 3:
            return ['x', 'y', 'z']
        return [f'coord_{i}' for i in range(n_dim)]

    @staticmethod
    def _common_variables(stage_vars: dict, refs: list, variables) -> list:
        """
        Variables usable across every selected case.

        A variable present in some cases but not others would silently
        shift the columns of the long tensor, so it is excluded (with a
        warning) instead of being padded.
        """
        if isinstance(variables, str):
            variables = [variables]
        candidates = (
            [v for v in stage_vars if v not in EXCLUDED_VARS]
            if variables is None else list(variables)
        )

        usable, dropped = [], []
        for name in candidates:
            values = stage_vars.get(name)
            if not isinstance(values, list):
                dropped.append(name)
                continue
            if all(
                ref.position < len(values) and values[ref.position] is not None
                for ref in refs
            ):
                usable.append(name)
            else:
                dropped.append(name)

        if dropped:
            warnings.warn(
                f"Variable(s) {dropped} are missing in at least one selected "
                "case and were left out of the joint tensor.",
                UserWarning,
            )
        return usable

    # ── M4 · round-trip artefacts ─────────────────────────────────────

    def save_to_npy(
        self,
        filepath: str,
        stage: Union[int, str],
        id_group: Union[str, int],
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        variables: Union[str, list, tuple, None] = None,
        verbose: bool = False,
    ) -> str:
        """
        Save a group as a ``.npy`` that ``FRODO(format='NUMPY')`` can read.

        One **top-level key per case** (``'CADGroup_3__f10'``, …), each a
        self-contained CADGroup with a single case. This is what makes the
        round-trip work without touching ``NUMPYReader``: that reader
        rejects ragged arrays but already treats every top-level key as an
        independent case space, so n meshes become n one-case groups
        instead of one impossible ragged group.

        Shapes follow ``NUMPYReader``'s contract exactly: ``FlCc``
        ``(1, n_dvars)``, ``Coord`` ``(n_points, n_dim)``, ``idx_sort``
        ``(1, 1, n_points)`` (it validates ``ndim == 3``), and scalar
        variables transposed to ``(1, n_points)`` — the
        ``(n_cases, n_points)`` layout it accepts.

        The artefact is **self-contained**: geometry is written in, not
        referenced. A file of paths would break the moment the dataset
        moved.

        Parameters
        ----------
        filepath : str
            Destination. ``.npy`` is appended if absent.
        stage : int or str
            Stage to export.
        id_group : str or int
            CADGroup identifier.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to export. Default ``'all'``.
        variables : str, list[str] or None
            Variables to export. ``None`` exports every non-bookkeeping
            variable each case actually has.
        verbose : bool
            Log one line per case.

        Returns
        -------
        str
            The path written.

        Examples
        --------
        ::

            db.sets.save_to_npy('/out/gci.npy', stage=1, id_group='3')
            db2 = FRODO(root_dir='/out', format='NUMPY', file='gci.npy')
            db2.extract_inputs(id_groups='3__f10')
            db2.extract_outputs(stage=1, id_groups='3__f10')
        """
        key, group = self._resolve_group(id_group)
        stage_key = str(stage)
        stage_vars = group.get('Vars', {}).get(stage_key, {})
        refs = self.case_refs(id_group, cases_idx)
        flcc = np.asarray(group['FlCc'])

        if isinstance(variables, str):
            variables = [variables]

        out: dict = {}
        for ref in refs:
            pos = ref.position
            coord = np.asarray(ref.coord(), dtype=np.float64)
            npts = coord.shape[0]

            names = (
                [v for v in stage_vars if v not in EXCLUDED_VARS]
                if variables is None else list(variables)
            )

            case_vars: dict = {}
            for name in names:
                values = stage_vars.get(name)
                value = (
                    values[pos]
                    if isinstance(values, list) and pos < len(values) else None
                )
                if value is None:
                    continue
                value = np.asarray(value, dtype=np.float64)
                if value.ndim == 1:
                    # (n_cases=1, n_points): the transposed layout the
                    # NUMPY reader accepts.
                    case_vars[name] = value[None, :]
                elif value.ndim == 2:
                    # (n_dim, n_points, n_cases=1)
                    case_vars[name] = value.T[:, :, None]
                else:
                    raise ValueError(
                        f"Variable '{name}' of case '{ref.name}' has "
                        f"unsupported ndim {value.ndim}."
                    )

            sorter = np.asarray(
                ref.sorter(int(stage)) if int(stage) in ref.stages
                else np.arange(npts),
                dtype=np.int32,
            )

            out[f'{key}__{ref.name}'] = {
                'Coord':     coord,
                'NodeCoord': np.asarray(ref.nodes(), dtype=np.float64),
                'FlCc':      flcc[pos:pos + 1, :],
                'Conec':     np.asarray(ref.conec()),
                'eltype':    np.asarray(ref.eltype()),
                'cellOrder': np.arange(npts, dtype=np.float64),
                'idx_sort':  sorter.reshape(1, 1, -1),
                'Vars':      {stage_key: case_vars},
                'mesh_file': ref.mesh_file,
                'case':      ref.name,
            }
            if verbose:
                log.info(
                    "[CODASingleSets] %s__%s: %s points, %s variables",
                    key, ref.name, npts, len(case_vars),
                )

        if not filepath.endswith('.npy'):
            filepath += '.npy'
        np.save(filepath, out, allow_pickle=True)
        return filepath

    def save_to_h5(
        self,
        filepath: str,
        stage: Union[int, str, None] = None,
        id_group: Union[str, int, None] = None,
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        overwrite: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Save one or more groups to HDF5, one subgroup per case.

        HDF5 has no ragged datasets, so the per-case layout is explicit::

            <CADGroup_key>/cases/<folder>/
                Coord, NodeCoord, Conec, eltype, FlCc
                Vars/<stage>/<var>

        Unlike ``CODASets.save_to_h5`` this does not walk the whole
        ``data_dict`` blindly: it takes the groups it is given, so the
        CODA_SINGLE bookkeeping keys (``case_order``, ``mesh_files``,
        ``stages_available``) never reach ``create_dataset``.

        Parameters
        ----------
        filepath : str
            Destination. ``.h5`` is appended if absent — and existence is
            checked **after** that, unlike the CODA version.
        stage : int, str or None
            Stage to export. ``None`` exports every stage present.
        id_group : str, int or None
            Group to export. ``None`` exports every CODA_SINGLE group.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to export. Default ``'all'``.
        overwrite : bool
            Replace an existing file. Default True.
        verbose : bool
            Log each case written.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        FileExistsError
            If the file exists and ``overwrite=False``.

        Examples
        --------
        ::

            db.sets.save_to_h5('/out/gci.h5', stage=1, id_group='3')
        """
        import h5py

        if not filepath.endswith('.h5'):
            filepath += '.h5'
        if os.path.exists(filepath):
            if not overwrite:
                raise FileExistsError(f"File already exists: {filepath}")
            os.remove(filepath)

        if id_group is None:
            keys = [
                k for k, v in self.db.data_dict.items()
                if isinstance(v, dict) and 'case_order' in v
            ]
        else:
            keys = [self._resolve_group(id_group)[0]]

        with h5py.File(filepath, 'w') as fh:
            for key in keys:
                group = self.db.data_dict[key]
                grp = fh.create_group(key)
                grp.attrs['case_order'] = [
                    str(c) for c in group['case_order']
                ]
                cases_grp = grp.create_group('cases')
                flcc = np.asarray(group['FlCc'])

                for ref in self.case_refs(key, cases_idx):
                    pos = ref.position
                    cg = cases_grp.create_group(ref.name)
                    cg.attrs['mesh_file'] = str(ref.mesh_file)
                    cg.attrs['case_idx'] = pos
                    cg.create_dataset('Coord', data=ref.coord())
                    cg.create_dataset('NodeCoord', data=ref.nodes())
                    cg.create_dataset('Conec', data=ref.conec())
                    cg.create_dataset('eltype', data=ref.eltype())
                    cg.create_dataset('FlCc', data=flcc[pos:pos + 1, :])

                    vars_grp = cg.create_group('Vars')
                    stages = (
                        list(group.get('Vars', {}))
                        if stage is None else [str(stage)]
                    )
                    for stage_key in stages:
                        stage_vars = group.get('Vars', {}).get(stage_key, {})
                        sg = vars_grp.create_group(stage_key)
                        for name, values in stage_vars.items():
                            value = (
                                values[pos]
                                if isinstance(values, list)
                                and pos < len(values) else None
                            )
                            if value is None:
                                continue
                            sg.create_dataset(
                                name, data=np.asarray(value),
                                compression='gzip', compression_opts=4,
                            )
                    if verbose:
                        log.info(
                            "[CODASingleSets] %s/cases/%s written",
                            key, ref.name,
                        )
        return filepath

    # ── Bridge · resample onto a common point set ─────────────────────

    def sample_on(
        self,
        target: Union[str, int, np.ndarray],
        stage: Union[int, str],
        id_group: Union[str, int],
        variables: Union[str, list, tuple, None] = None,
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        method: Literal['idw', 'griddata'] = 'idw',
        k: int = 4,
        new_group_id: Union[str, None] = None,
    ) -> dict:
        """
        Resample every selected case onto one common set of points.

        This is the **explicit** bridge back to the dense
        ``(n_points, n_cases)`` layout that ``CODASets``, pyLOM and
        LEGOLAS understand. It is a method and not a default on purpose:
        in a mesh-convergence study, interpolating every case onto a
        reference mesh destroys exactly the difference being measured.
        Use it when the question is "how does this field vary with the
        design variables", not "how does it vary with the mesh".

        Parameters
        ----------
        target : str, int or np.ndarray
            Where to sample. ``'case:<folder>'`` or an ``int`` uses that
            case's own points (the reference-mesh choice); an
            ``(n_target, n_dim)`` array uses arbitrary probe points.
        stage : int or str
            Stage to read.
        id_group : str or int
            CADGroup identifier.
        variables : str, list[str] or None
            Variables to resample. ``None`` takes every variable present
            in all selected cases.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to include. Default ``'all'``.
        method : {'idw', 'griddata'}
            Interpolation. ``'idw'`` (default) reuses
            ``SAM.Weapons._interpolate_idw_tree``; ``'griddata'`` reuses
            ``SAM.Weapons._interpolate_griddata``.
        k : int
            Neighbours for IDW. Default 4.
        new_group_id : str or None
            If given, the dense result is also written to
            ``data_dict['CADGroup_<new_group_id>']`` in the CODA layout,
            so ``CODASets`` methods can be pointed at it.

        Returns
        -------
        dict
            ``'Coord'`` ``(n_target, n_dim)``, ``'FlCc'``
            ``(n_cases, n_dvars)``, ``'Vars'`` ``{stage: {var:
            (n_target, n_cases)}}``, ``'case_order'`` and ``'target'``
            (how the points were chosen).

        Raises
        ------
        ValueError
            If ``target`` cannot be resolved or ``method`` is unknown.

        Examples
        --------
        Onto the coarsest mesh of the family::

            dense = db.sets.sample_on('case:f10', stage=1, id_group='3')
            dense['Vars']['1']['BoundaryValues_CoefPressure'].shape

        Onto arbitrary probes, exposed as a dense CADGroup::

            probes = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
            db.sets.sample_on(probes, stage=1, id_group='3',
                              new_group_id='3_probes')
        """
        key, group = self._resolve_group(id_group)
        stage_key = str(stage)
        stage_vars = group.get('Vars', {}).get(stage_key)
        if stage_vars is None:
            raise KeyError(
                f"Stage '{stage_key}' not found in '{key}'['Vars']."
            )

        refs = self.case_refs(id_group, cases_idx)
        if not refs:
            raise ValueError(f"{key}: no case selected.")

        # ── Resolve the target point set ────────────────────────────
        if isinstance(target, np.ndarray):
            coord_dst = np.asarray(target, dtype=np.float64)
            target_label = f'array({coord_dst.shape[0]} points)'
        else:
            if isinstance(target, str) and target.startswith('case:'):
                wanted = target.split(':', 1)[1]
                order = group['case_order']
                if wanted not in order:
                    raise ValueError(
                        f"{key}: target case '{wanted}' is not in "
                        f"case_order {order}."
                    )
                pos = order.index(wanted)
            elif isinstance(target, (int, np.integer)):
                pos = int(target)
            else:
                raise ValueError(
                    "target must be 'case:<folder>', an int position, or "
                    f"an (n, ndim) array; got {target!r}."
                )
            coord_dst = np.asarray(
                CaseRef(self.db, key, pos).coord(), dtype=np.float64
            )
            target_label = f'case:{group["case_order"][pos]}'

        var_names = self._common_variables(stage_vars, refs, variables)
        if not var_names:
            raise ValueError(
                f"{key} stage '{stage_key}': no variable available in "
                "every selected case."
            )

        if method not in ('idw', 'griddata'):
            raise ValueError(
                f"method '{method}' not supported. Use 'idw' or 'griddata'."
            )

        flcc = np.asarray(group['FlCc'])
        n_target = coord_dst.shape[0]
        dense = {
            name: np.full((n_target, len(refs)), np.nan, dtype=np.float64)
            for name in var_names
        }

        from scipy.spatial import cKDTree
        for column, ref in enumerate(refs):
            pos = ref.position
            coord_src = np.asarray(ref.coord(), dtype=np.float64)
            # One stack per case so the KDTree is built once, not once
            # per variable.
            stack = np.column_stack([
                np.asarray(
                    stage_vars[name][pos], dtype=np.float64
                ).reshape(coord_src.shape[0], -1)
                for name in var_names
            ])
            widths = [
                np.asarray(
                    stage_vars[name][pos]
                ).reshape(coord_src.shape[0], -1).shape[1]
                for name in var_names
            ]

            if method == 'idw':
                tree = cKDTree(coord_src)
                out = SAM.Weapons._interpolate_idw_tree(
                    tree, coord_src, stack, coord_dst, k=k,
                )
            else:
                out = SAM.Weapons._interpolate_griddata(
                    coord_src, stack, coord_dst,
                )
            out = np.asarray(out).reshape(n_target, -1)

            col = 0
            for name, width in zip(var_names, widths):
                chunk = out[:, col:col + width]
                dense[name][:, column] = (
                    chunk[:, 0] if width == 1 else np.linalg.norm(chunk, axis=1)
                )
                col += width

        result = {
            'Coord':      coord_dst,
            'FlCc':       np.vstack([flcc[r.position:r.position + 1, :]
                                     for r in refs]),
            'Vars':       {stage_key: dense},
            'case_order': [r.name for r in refs],
            'target':     target_label,
        }

        if new_group_id is not None:
            new_key = f'CADGroup_{new_group_id}'
            if new_key in self.db.data_dict:
                warnings.warn(
                    f"Overwriting existing group '{new_key}'.", UserWarning
                )
            self.db.data_dict[new_key] = {
                'Coord':      coord_dst,
                'NodeCoord':  coord_dst,
                'FlCc':       result['FlCc'],
                'Conec':      None,
                'idx_sort':   np.tile(
                    np.arange(n_target, dtype=np.int32)[None, None, :],
                    (1, len(refs), 1),
                ),
                'eltype':     None,
                'cellOrder':  np.arange(n_target, dtype=np.float64),
                'pointOrder': np.arange(n_target, dtype=np.float64),
                'Vars':       {stage_key: dense},
                'sampled_from': key,
                'case_order':   result['case_order'],
            }

        return result

    # ── M3 · export with a mesh ───────────────────────────────────────

    def to_pylom(
        self,
        stage: Union[int, str],
        id_group: Union[str, int],
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        variables: Union[str, list, tuple, None] = None,
        save_path: Union[str, None] = None,
        nan_policy: Literal['fill', 'raise'] = 'fill',
        nan_fill_value: float = 0.0,
        verbose: bool = False,
    ) -> list:
        """
        Build one ``pyLOM.Dataset`` per case.

        ``pyLOM.Dataset`` holds a single ``xyz`` and stores fields as
        rectangular ``(ndim * n_points, n_cases)`` arrays, so several
        meshes in one Dataset is structurally impossible: n meshes are n
        Datasets, each with a single case. (To merge them into one, call
        :meth:`sample_on` first.)

        pyLOM is imported **inside the method**, not at module level, so
        that this class stays importable — and CODA_SINGLE stays in
        ``SETS_REGISTRY`` — when pyLOM is not installed.

        Parameters
        ----------
        stage : int or str
            Stage to export.
        id_group : str or int
            CADGroup identifier.
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Cases to export. Default ``'all'``.
        variables : str, list[str] or None
            Fields to export. ``None`` takes every non-bookkeeping
            variable the case has.
        save_path : str or None
            Directory to write ``<key>_<case>_stage_<stage>.h5`` into.
        nan_policy : {'fill', 'raise'}
            What to do with NaNs. Default ``'fill'``.
        nan_fill_value : float
            Replacement when filling. Default 0.0.
        verbose : bool
            Log each Dataset built.

        Returns
        -------
        list
            ``(case_name, Dataset)`` pairs, in ``case_order``.

        Raises
        ------
        ImportError
            If pyLOM is not installed.
        ValueError
            If ``nan_policy='raise'`` and a field has NaNs.

        Examples
        --------
        ::

            pairs = db.sets.to_pylom(stage=1, id_group='3',
                                     save_path='/out/pylom/')
            name, ds = pairs[0]
        """
        try:
            from pyLOM.dataset import Dataset
            from pyLOM.partition_table import PartitionTable
        except ImportError as exc:                      # pragma: no cover
            raise ImportError(
                "to_pylom needs pyLOM, which is not installed."
            ) from exc

        key, group = self._resolve_group(id_group)
        stage_key = str(stage)
        stage_vars = group.get('Vars', {}).get(stage_key, {})
        refs = self.case_refs(id_group, cases_idx)
        design_vars = list(self.db.metadata.get('design_vars') or [])
        flcc = np.asarray(group['FlCc'])

        if isinstance(variables, str):
            variables = [variables]

        def _sanitize(name, value):
            nan_mask = np.isnan(value)
            if not np.any(nan_mask):
                return np.ascontiguousarray(value)
            if nan_policy == 'raise':
                raise ValueError(
                    f"Field '{name}' has {int(nan_mask.sum())} NaN values."
                )
            warnings.warn(
                f"Field '{name}': replacing {int(nan_mask.sum())} NaN "
                f"values with {nan_fill_value}.",
                RuntimeWarning,
            )
            value = value.copy()
            value[nan_mask] = nan_fill_value
            return np.ascontiguousarray(value)

        out = []
        for ref in refs:
            pos = ref.position
            xyz = np.ascontiguousarray(
                np.asarray(ref.coord(), dtype=np.float64)
            )
            npoints = xyz.shape[0]
            ptable = PartitionTable.new(1, npoints, npoints)

            # Parametric variables: dense idim, as Dataset.X() expects
            # (create_NN_pylom sets them all to 0, which is wrong).
            param_dict = {
                dvar: {'idim': i,
                       'value': np.asarray([flcc[pos, i]], dtype=np.float64)}
                for i, dvar in enumerate(design_vars)
                if i < flcc.shape[1]
            }

            names = (
                [v for v in stage_vars if v not in EXCLUDED_VARS]
                if variables is None else list(variables)
            )
            field_dict = {}
            for name in names:
                values = stage_vars.get(name)
                value = (
                    values[pos]
                    if isinstance(values, list) and pos < len(values) else None
                )
                if value is None:
                    continue
                value = np.asarray(value, dtype=np.float64)
                if value.ndim == 1:
                    # Must be 2-D: pyLOM writes with value[istart:iend, :].
                    field_dict[name] = {
                        'ndim': 1, 'value': _sanitize(name, value[:, None]),
                    }
                elif value.ndim == 2:
                    # (n_points, n_dims) -> interleaved (n_dims*n_points, 1),
                    # point-major, which is what Dataset.get_dim() expects.
                    n_dims = value.shape[1]
                    field_dict[name] = {
                        'ndim': n_dims,
                        'value': _sanitize(
                            name, value.reshape(npoints * n_dims, 1)
                        ),
                    }
                else:
                    raise ValueError(
                        f"Field '{name}' of case '{ref.name}' has "
                        f"unsupported ndim {value.ndim}."
                    )

            dataset = Dataset(
                xyz=xyz, ptable=ptable,
                order=np.arange(npoints, dtype=np.int32),
                point=True, vars=param_dict, **field_dict,
            )
            out.append((ref.name, dataset))

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                target = os.path.join(
                    save_path, f"{key}_{ref.name}_stage_{stage_key}.h5"
                )
                dataset.save(target)
                if verbose:
                    log.info("[CODASingleSets] Dataset saved to %s", target)

        return out
