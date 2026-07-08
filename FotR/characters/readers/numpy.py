"""
readers/numpy.py
=================
Reader for pre-assembled numpy CADGroup datasets (the ``NUMPY`` format).

Unlike the ``NUMPYFILE`` format — which stored a flat, ad-hoc
``{'inputs': ..., 'outputs': ..., 'aux': ...}`` dictionary — the ``NUMPY``
format stores data that is *already shaped exactly like a CODA CADGroup*.
This reader's only job is to load that data and expose it through
``self.data_dict`` using the very same layout produced by
``CODAReader.extract_inputs`` / ``CODAReader.extract_outputs``, so that
every downstream consumer (``CODASets``, ``CODAStats``, ``CODAResiduals``,
``LEGOLAS``, …) works on a ``NUMPY``-backed :class:`FRODO` instance without
any special-casing.

Expected on-disk layout
-----------------------
Each ``.npy`` file passed via ``file`` must contain a Python dict (saved
with ``np.save(path, obj, allow_pickle=True)``) whose top-level keys are
CADGroup names (e.g. ``'CADGroup_3_completo'``) and whose values are
themselves dicts with the following keys::

    {
        'CADGroup_3_completo': {
            'Coord':          np.ndarray (n_points, n_dim),
            'NodeCoord':      np.ndarray (n_nodes,  n_dim),      # optional
            'FlCc':           np.ndarray (n_cases,  n_dvars),
            'Conec':          np.ndarray (n_cells,  max_nodes),  # optional
            'idx_sort':       np.ndarray (n_stages, n_cases, n_points),  # optional
            'idx_sort_nodes': np.ndarray (n_stages, n_cases, n_nodes),   # optional
            'eltype':         np.ndarray (n_points,),           # optional
            'cellOrder':      np.ndarray (n_points,),           # optional
            'pointOrder':     np.ndarray (n_nodes,),            # optional
            'Vars': {
                '0': {
                    'GlobalNumber':                 (n_points, n_cases),
                    'BoundaryValues_CoefPressure':   (n_points, n_cases),
                    'CADGroupID':                    (n_points, n_cases),
                    ...
                },
                '1': {...},
                ...
            },
        },
        # a single .npy file MAY contain more than one CADGroup key
        'CADGroup_5': {...},
    }

This is exactly the in-memory structure of
``FRODO.data_dict['CADGroup_<id>']`` for the CODA format (see
``sets/coda.py`` and ``frodo.py::merge_datasets`` — this is precisely
what a ``CODASets`` merge or a manual ``np.save(data_dict['CADGroup_X'],
...)`` would produce), so files produced by CODA pipelines can be
dropped in as-is.

Two-step workflow
-----------------
1. ``parse_simulation_dirs()`` — mirrors ``CODAReader.parse_simulation_dirs``
   as closely as the format allows: it loads every ``.npy`` file, inspects
   every CADGroup it contains, and builds ``self.sim_metadata`` (one entry
   per CADGroup, analogous to CODA's one-entry-per-simulation-folder) and
   ``self.df_state`` (one row per *case*, analogous to CODA's one-row-per-
   simulation, with the design-variable columns taken from ``FlCc`` and a
   ``'stage'`` column giving the number of stages available for that
   CADGroup).
2. ``extract_inputs`` / ``extract_outputs`` — copy (and optionally
   sub-select by case) the arrays already present in the loaded dict
   straight into ``self.data_dict[key]``, with ``key`` following CODA's
   ``'CADGroup_<id>'`` convention.  No geometric re-sorting or cell
   filtering is performed, since the data is assumed to already be in its
   final, analysis-ready ordering.
"""

import os
import json
import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd

from .base import BaseReader


class NUMPYReader(BaseReader):
    """
    Reader for the ``NUMPY`` format: pre-assembled CADGroup dictionaries
    stored as ``.npy`` files, exposed through the same ``data_dict``
    layout used by :class:`~FotR.characters.readers.coda.CODAReader`.

    Parameters
    ----------
    root_dir : str
        Directory containing the ``.npy`` file(s) and, optionally, a
        ``metadata/cases_metadata.json`` file (same convention as CODA)
        with keys ``'design_vars'`` and ``'num_stages'``.  When present,
        these override the automatic inference performed from the array
        shapes themselves.
    file : str, list[str] or tuple[str]
        Relative path(s), inside ``root_dir``, to the ``.npy`` file(s) to
        load.  A single string is accepted as well as a list/tuple of
        strings.  Every file is loaded eagerly at construction time.

    Attributes
    ----------
    files : list[str]
        Normalised list of relative file paths.
    npy_dict : dict[str, dict]
        Maps each loaded filename to its raw content: a dict whose keys
        are CADGroup names and whose values are the raw per-group dicts
        described in the module docstring.
    group_index : dict[str, str]
        Maps every discovered CADGroup name to the filename it was loaded
        from.  Populated by ``parse_simulation_dirs``.
    metadata : dict
        Keys: ``'eq_type'``, ``'design_vars'``, ``'num_stages'``. Loaded
        from ``metadata/cases_metadata.json`` when available, otherwise
        left as ``None`` / inferred lazily per-group in
        ``parse_simulation_dirs``.

    Raises
    ------
    FileNotFoundError
        If any of the requested files do not exist on disk.
    TypeError
        If ``file`` is not a string or list/tuple of strings.
    ValueError
        If ``file`` is None.

    Examples
    --------
    Construct via FRODO::

        from FotR.characters.frodo import FRODO

        db = FRODO(
            root_dir='/data/numpy_db',
            format='NUMPY',
            file='merged_groups.npy',
        )
        db.extract_inputs(id_groups='3_completo', cases_idx='all')
        db.extract_outputs(stage=0, id_groups='3_completo')
        print(db.data_dict['CADGroup_3_completo']['Coord'].shape)

    Construct the reader directly (for testing)::

        from FotR.characters.readers.numpy import NUMPYReader

        reader = NUMPYReader(root_dir='/data/numpy_db', file='groups.npy')
        reader.parse_simulation_dirs()
        print(reader.df_state.head())
    """

    def __init__(self, root_dir: str, file: Union[str, list, tuple], **kwargs):
        super().__init__(root_dir, **kwargs)

        # ── Normalise ``file`` argument ────────────────────────────────────
        if file is None:
            raise ValueError("'file' must not be None.")
        if isinstance(file, str):
            self.files = [file]
        elif isinstance(file, (list, tuple)):
            if not all(isinstance(f, str) for f in file):
                raise TypeError("Every element in 'file' must be a str path.")
            self.files = list(file)
        else:
            raise TypeError(
                "'file' must be a string or a list/tuple of strings."
            )

        # ── Verify files exist ────────────────────────────────────────────
        for f in self.files:
            full = os.path.join(root_dir, f)
            if not os.path.exists(full):
                raise FileNotFoundError(f"File not found: {full}")

        # ── Load all dictionaries eagerly ────────────────────────────────
        self.npy_dict: dict = {
            f: np.load(os.path.join(root_dir, f), allow_pickle=True).item()
            for f in self.files
        }

        self.group_index: dict = {}
        self.data_dict:   dict = {}

        # ── Optional metadata/cases_metadata.json (mirrors CODA) ──────────
        self.metadata = {'eq_type': None, 'design_vars': None, 'num_stages': None}
        meta_path = os.path.join(root_dir, 'metadata', 'cases_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as fh:
                    cm = json.load(fh)
                self.metadata = {
                    'eq_type':     cm.get('eq_type',     None),
                    'design_vars': cm.get('design_vars', None),
                    'num_stages':  cm.get('num_stages',  None),
                }
            except Exception as exc:
                warnings.warn(
                    f"Could not parse '{meta_path}': {exc}. "
                    "design_vars / num_stages will be inferred per group.",
                    UserWarning,
                )

    # =========================================================================
    # BaseReader interface
    # =========================================================================

    def parse_simulation_dirs(self) -> None:
        """
        Discover every CADGroup contained in the loaded ``.npy`` files and
        build ``self.sim_metadata`` and ``self.df_state``, mirroring as
        closely as possible the contract followed by
        ``CODAReader.parse_simulation_dirs``.

        For each CADGroup this inspects ``FlCc`` (to get ``n_cases`` and,
        when ``self.metadata['design_vars']`` is unavailable, to invent
        generic design-variable names ``'dv_0', 'dv_1', ...``) and
        ``Vars`` (to get the list of available stage keys, analogous to
        how CODA counts ``output_<stage>_*`` files per simulation folder).

        Populates
        ---------
        self.group_index : dict[str, str]
            Maps each discovered CADGroup name to the filename it was
            loaded from.
        self.sim_metadata : dict
            Keyed by CADGroup name (the NUMPY-format analogue of CODA's
            per-folder keys). Each value contains::

                {
                    'file':        'merged_groups.npy',
                    'n_points':    13862,
                    'n_nodes':     27724,          # 0 if NodeCoord absent
                    'n_cases':     180,
                    'design_vars': ['aoa', 'mach'],
                    'stages':      ['0', '1'],
                    'has_conec':   True,
                }

        self.df_state : pd.DataFrame
            One row per **case** (pooling every CADGroup found across
            every loaded file), with columns: every design variable
            (taken from ``FlCc``), ``'stage'`` (number of stages available
            for that CADGroup — constant within a group, since the tensor
            is already fully assembled), ``'group'`` (CADGroup name) and
            ``'case_idx'`` (row index within that group's ``FlCc``).

        Examples
        --------
        ::

            reader.parse_simulation_dirs()
            print(reader.sim_metadata['CADGroup_3_completo']['n_cases'])
            print(reader.df_state[['aoa', 'mach', 'stage', 'group']].head())
        """
        self.group_index  = {}
        self.sim_metadata  = {}
        state_rows: list = []

        for fname, content in self.npy_dict.items():
            if not isinstance(content, dict):
                warnings.warn(
                    f"'{fname}' does not contain a top-level dict; skipped.",
                    UserWarning,
                )
                continue

            for group_key, gd in content.items():
                if not isinstance(gd, dict) or 'FlCc' not in gd:
                    warnings.warn(
                        f"Entry '{group_key}' in '{fname}' does not look "
                        "like a CADGroup dict (missing 'FlCc'); skipped.",
                        UserWarning,
                    )
                    continue

                self.group_index[group_key] = fname

                flcc    = np.asarray(gd['FlCc'])
                n_cases = flcc.shape[0]
                n_dvars = flcc.shape[1] if flcc.ndim > 1 else 1

                design_vars = self.metadata.get('design_vars')
                if design_vars is None or len(design_vars) != n_dvars:
                    design_vars = [f'dv_{i}' for i in range(n_dvars)]

                stages = sorted(gd.get('Vars', {}).keys(), key=str)
                n_points = np.asarray(gd['Coord']).shape[0] if 'Coord' in gd else 0
                n_nodes  = (
                    np.asarray(gd['NodeCoord']).shape[0]
                    if gd.get('NodeCoord') is not None else 0
                )

                self.sim_metadata[group_key] = {
                    'file':        fname,
                    'n_points':    n_points,
                    'n_nodes':     n_nodes,
                    'n_cases':     n_cases,
                    'design_vars': design_vars,
                    'stages':      stages,
                    'has_conec':   gd.get('Conec') is not None,
                }

                flcc_2d = flcc.reshape(n_cases, n_dvars)
                for case_idx in range(n_cases):
                    row = {dv: flcc_2d[case_idx, i] for i, dv in enumerate(design_vars)}
                    row['stage']    = len(stages)
                    row['group']    = group_key
                    row['case_idx'] = case_idx
                    state_rows.append(row)

        print(
            f"{len(self.sim_metadata)} CADGroup(s) found across "
            f"{len(self.npy_dict)} file(s)."
        )

        if state_rows:
            self.df_state = pd.DataFrame.from_records(state_rows)
        else:
            self.df_state = pd.DataFrame()

    def extract_inputs(
        self,
        id_groups: Union[str, int, list, tuple],
        cases_idx: Union[list, tuple, int, str] = 'all',
        verbose: bool = False,
    ) -> None:
        """
        Copy geometry and flight-condition arrays for one or more CADGroups
        straight from the loaded ``.npy`` content into ``self.data_dict``,
        following the exact same key layout as
        ``CODAReader.extract_inputs``.

        No geometric sorting, cell filtering or connectivity remapping is
        performed here (unlike CODA, which has to derive this from raw
        ``.vtu`` files): the NUMPY format assumes the arrays are already
        in their final, analysis-ready ordering. The only transformation
        applied is an optional sub-selection of cases via ``cases_idx``,
        which is propagated consistently to ``FlCc``, ``idx_sort`` and
        ``idx_sort_nodes`` (their case axis) so that a later call to
        ``extract_outputs`` with the same ``cases_idx`` yields consistent
        data.

        Parameters
        ----------
        id_groups : str, int, list or tuple
            CADGroup identifier(s) to load. Each entry is matched against
            the loaded top-level keys either directly (e.g.
            ``'CADGroup_3_completo'``) or, if not found verbatim, by
            prefixing it with ``'CADGroup_'`` (e.g. passing ``'3_completo'``
            or ``3`` both resolve to ``'CADGroup_3_completo'`` /
            ``'CADGroup_3'``). A single str/int is treated as a
            one-element list.
        cases_idx : list, tuple, int or 'all'
            Subset of case indices (rows of that group's ``FlCc``) to
            keep. Default ``'all'``.
        verbose : bool
            Print per-group progress information.

        Populates
        ---------
        self.data_dict[key] with the same keys used by CODA:
        ``'Coord'``, ``'NodeCoord'``, ``'FlCc'``, ``'Conec'``,
        ``'idx_sort'``, ``'idx_sort_nodes'``, ``'eltype'``,
        ``'cellOrder'``, ``'pointOrder'``. Keys whose source array is
        absent from the raw dict are set to ``None`` (mirroring how CODA
        would leave them ``None`` if a stage/case failed to load).

        Side-effects
        ------------
        Stores the resolved ``cases_idx`` (as a list of ints, positions
        into the *original* ``FlCc``) in ``self._active_cases_idx[key]``
        so that ``extract_outputs`` can subset the corresponding
        ``Vars`` columns identically.

        Raises
        ------
        KeyError
            If a requested CADGroup cannot be resolved in any loaded file.
        IndexError
            If ``cases_idx`` contains out-of-range values.

        Examples
        --------
        Load every case of a single pre-merged group::

            reader.extract_inputs(id_groups='3_completo')
            print(reader.data_dict['CADGroup_3_completo']['Coord'].shape)

        Load two groups, restricting the first to its first 50 cases::

            reader.extract_inputs(id_groups=['3_completo', '5'])
        """
        if isinstance(id_groups, (str, int)):
            id_groups = [id_groups]

        self._active_cases_idx = getattr(self, '_active_cases_idx', {})

        for group_id in id_groups:
            key, gd = self._resolve_group(group_id)

            n_cases_total = np.asarray(gd['FlCc']).shape[0]
            local_cases_idx = self._normalise_cases_idx(cases_idx, n_cases_total)

            if verbose:
                print(
                    f"[NUMPYReader] extract_inputs — group '{key}': "
                    f"{len(local_cases_idx)}/{n_cases_total} case(s)."
                )

            flcc = np.asarray(gd['FlCc'])
            flcc = flcc.reshape(n_cases_total, -1)[local_cases_idx]

            idx_sort = gd.get('idx_sort')
            if idx_sort is not None:
                idx_sort = np.asarray(idx_sort)[:, local_cases_idx, :]

            idx_sort_nodes = gd.get('idx_sort_nodes')
            if idx_sort_nodes is not None:
                idx_sort_nodes = np.asarray(idx_sort_nodes)[:, local_cases_idx, :]

            def _copy(name):
                val = gd.get(name)
                return np.asarray(val).copy() if val is not None else None

            self.data_dict.setdefault(key, {}).update({
                'Coord':          _copy('Coord'),
                'NodeCoord':      _copy('NodeCoord'),
                'FlCc':           flcc,
                'Conec':          _copy('Conec'),
                'idx_sort':       idx_sort,
                'idx_sort_nodes': idx_sort_nodes,
                'eltype':         _copy('eltype'),
                'cellOrder':      _copy('cellOrder'),
                'pointOrder':     _copy('pointOrder'),
            })

            self._active_cases_idx[key] = local_cases_idx

    def extract_outputs(
        self,
        stage: Union[int, str],
        id_groups: Union[str, int, list, tuple],
        var_name_excluded: Union[list, tuple, None] = None,
        verbose: bool = False,
    ) -> None:
        """
        Copy field variables for the given ``stage`` and one or more
        CADGroups from the loaded ``.npy`` content into
        ``self.data_dict[key]['Vars'][str(stage)]``, matching CODA's
        layout exactly.

        Requires ``extract_inputs`` to have been called first for every
        requested group (so that the case subset — stored in
        ``self._active_cases_idx`` — is known and applied consistently).

        Parameters
        ----------
        stage : int or str
            Stage key to read from the source ``Vars`` dict (e.g. ``0``
            or ``'0'``).
        id_groups : str, int, list or tuple
            Same resolution rules as in ``extract_inputs``.
        var_name_excluded : list, tuple or None
            Variable names to skip during extraction (e.g.
            ``['GlobalNumber', 'CADGroupID']``). Default ``None`` (keep
            every variable found for that stage).
        verbose : bool
            Print the list of copied variables per group.

        Populates
        ---------
        self.data_dict[key]['Vars'][str(stage)][var_name] with arrays of
        shape ``(n_points, n_cases)`` (scalar) restricted to the case
        subset selected in ``extract_inputs``, exactly like
        ``CODAReader.extract_outputs``.

        Raises
        ------
        RuntimeError
            If ``extract_inputs`` was not called first for a requested
            group.
        KeyError
            If the requested CADGroup or stage cannot be resolved.

        Examples
        --------
        ::

            reader.extract_inputs(id_groups='3_completo')
            reader.extract_outputs(
                stage=0,
                id_groups='3_completo',
                var_name_excluded=['GlobalNumber', 'CADGroupID'],
            )
            print(
                reader.data_dict['CADGroup_3_completo']['Vars']['0']
                .keys()
            )
        """
        if isinstance(id_groups, (str, int)):
            id_groups = [id_groups]

        active = getattr(self, '_active_cases_idx', {})

        for group_id in id_groups:
            key, gd = self._resolve_group(group_id)

            if key not in active:
                raise RuntimeError(
                    f"No case selection found for '{key}'. "
                    "Run extract_inputs() for this group first."
                )
            local_cases_idx = active[key]

            stage_vars = gd.get('Vars', {}).get(str(stage))
            if stage_vars is None:
                raise KeyError(
                    f"Stage '{stage}' not found in group '{key}'. "
                    f"Available stages: {list(gd.get('Vars', {}).keys())}."
                )

            self.data_dict.setdefault(key, {})
            self.data_dict[key].setdefault('Vars', {})
            self.data_dict[key]['Vars'].setdefault(str(stage), {})

            copied = []
            for var_name, arr in stage_vars.items():
                if var_name_excluded and var_name in var_name_excluded:
                    continue

                arr = np.asarray(arr)
                if arr.ndim == 2:
                    self.data_dict[key]['Vars'][str(stage)][var_name] = (
                        arr[:, local_cases_idx].astype(np.float64)
                    )
                elif arr.ndim == 3:
                    # (n_dim, n_points, n_cases) vector field
                    self.data_dict[key]['Vars'][str(stage)][var_name] = (
                        arr[:, :, local_cases_idx].astype(np.float64)
                    )
                else:
                    raise ValueError(
                        f"Variable '{var_name}' in group '{key}' has "
                        f"unsupported ndim {arr.ndim}."
                    )
                copied.append(var_name)

            if verbose:
                print(
                    f"[NUMPYReader] extract_outputs — group '{key}', "
                    f"stage '{stage}': {copied}"
                )

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _resolve_group(self, group_id: Union[str, int]) -> tuple:
        """
        Resolve a user-supplied group identifier into its canonical
        ``'CADGroup_<id>'`` key and the raw dict loaded for it.

        Resolution order
        -----------------
        1. If ``group_id`` (as a string) is already a key present in
           ``self.group_index``, use it verbatim.
        2. Otherwise, try ``f'CADGroup_{group_id}'``.

        Parameters
        ----------
        group_id : str or int
            Identifier as passed by the caller to ``extract_inputs`` /
            ``extract_outputs``.

        Returns
        -------
        tuple[str, dict]
            ``(key, raw_group_dict)``.

        Raises
        ------
        KeyError
            If neither resolution attempt matches a loaded CADGroup.

        Examples
        --------
        ::

            key, gd = reader._resolve_group('3_completo')
            # key → 'CADGroup_3_completo'

            key, gd = reader._resolve_group('CADGroup_5')
            # key → 'CADGroup_5'  (used verbatim)
        """
        candidate = str(group_id)
        if candidate in self.group_index:
            key = candidate
        else:
            prefixed = f'CADGroup_{group_id}'
            if prefixed in self.group_index:
                key = prefixed
            else:
                raise KeyError(
                    f"CADGroup '{group_id}' not found. "
                    f"Available groups: {list(self.group_index)}. "
                    "Run parse_simulation_dirs() first if you haven't."
                )

        fname = self.group_index[key]
        return key, self.npy_dict[fname][key]

    @staticmethod
    def _normalise_cases_idx(cases_idx, n_cases: int) -> list:
        """
        Normalise a ``cases_idx`` argument to a sorted list of valid
        integer case indices, mirroring
        ``CODAReader._normalise_cases_idx`` but taking the total case
        count directly (there is no ``df_cases`` DataFrame to consult in
        this format — the count comes straight from the group's own
        ``FlCc`` array).

        Parameters
        ----------
        cases_idx : 'all', int, range, list[int] or tuple[int]
            Case selection to normalise.
        n_cases : int
            Total number of cases available for the group being
            processed (i.e. ``FlCc.shape[0]``).

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

            idx = NUMPYReader._normalise_cases_idx('all', n_cases=180)
            idx = NUMPYReader._normalise_cases_idx([0, 1, 4], n_cases=180)
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