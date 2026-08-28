"""
frodo.py – Framework for Reusable Organized Data Output
========================================================
Lightweight coordinator for CFD simulation data management.

FRODO delegates all format-specific logic to three independent subpackages:

  readers/    → parse simulation folders, extract inputs / outputs.
  sets/       → ML tensor assembly, mesh operations, I/O helpers.
  residuals/  → convergence monitoring and integral metrics.

Adding a new format only requires:
  1. A reader in readers/<format>.py  (subclass of BaseReader).
  2. Optionally sets in sets/<format>.py (subclass of BaseSets).
  3. Optionally residuals in residuals/<format>.py (subclass of BaseResiduals).
  4. Registering each class in the corresponding subpackage __init__.py.
  FRODO itself never changes.
"""

import time
import os
import copy
import json
from typing import Literal, Union

import numpy as np
import pandas as pd

from ..EarendilsLight import EarendilsLight
from .sam import SAM
from .readers   import READER_REGISTRY
from .sets      import SETS_REGISTRY
from .residuals import RESIDUALS_REGISTRY
from .stats     import STATS_REGISTRY

class FRODO:
    """
    Framework for Reusable Organized Data Output
    ─────────────────────────────────────────────
    One tool to rule them all, one tool to find them.

    FRODO manages, organises and archives CFD simulation data across
    multiple formats, forging results into a unified structure ready for
    analysis, plotting, or machine learning.

    Supported formats
    -----------------
    Determined at runtime from READER_REGISTRY. Currently:
    'CODA', 'Airfoil', 'NUMPYFILE', 'PYLOM'.
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Shortcut to Eärendil's Light help system."""
        return cls.light.help(name)

    def __str__(self):
        return (
            f"{self.name}; root_dir: {self.root_dir}; format: {self.format}"
        )

    def __getattr__(self, name):
        """
        Dynamic delegation: if FRODO does not have *name*, search in
        self.sets, self.reader and self.residuals (in that order).

        This lets callers write ``db.add_aux(...)`` when the method lives on
        ``db.sets``, without explicitly exposing it on FRODO.
        """
        for sub in ('sets', 'reader', 'residuals', 'stats'):
            try:
                obj = object.__getattribute__(self, sub)
            except AttributeError:
                obj = None
            if obj is not None and hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __init__(
        self,
        root_dir: str,
        format: Literal['CODA', 'Airfoil', 'NUMPYFILE', 'PYLOM'],
        initial_parse: bool = True,
        **kwargs,
    ):
        self.format          = format
        self.root_dir        = os.path.abspath(root_dir)
        self.sim_metadata    = {}
        self.data_dict       = {}
        self.kwargs          = kwargs
        self.update_df_state = kwargs.pop("update_df_state", False)
        self.name            = kwargs.pop("name", "FRODO Database")

        self._set_subclasses()

        if initial_parse:
            t0 = time.perf_counter()
            self._parse()
            print(f"Parse took: {time.perf_counter() - t0:.4f} s")

    # ── Internal wiring ───────────────────────────────────────────────────────

    def _set_subclasses(self):
        """
        Instantiate reader, sets and residuals from the format registries.

        Registry values of None mean "format known but submodule not yet
        implemented"; a warning is printed and the attribute is set to None.
        A missing key raises ValueError.
        """
        # ── READER ──────────────────────────────────────────────────────────
        reader_cls = READER_REGISTRY.get(self.format)
        if reader_cls is None:
            raise ValueError(
                f"Format '{self.format}' is not supported. "
                f"Available formats: {list(READER_REGISTRY)}"
            )
        self.reader = reader_cls(root_dir=self.root_dir, **self.kwargs)

        # ── SETS ────────────────────────────────────────────────────────────
        sets_cls = SETS_REGISTRY.get(self.format)
        if sets_cls is not None:
            self.sets = sets_cls(db=self)
        else:
            self.sets = None
            print(
                "\n\tWARNING: No Sets class for this format. "
                "Sets methods will not be available.\n"
            )

        # ── RESIDUALS ───────────────────────────────────────────────────────
        residuals_cls = RESIDUALS_REGISTRY.get(self.format)
        if residuals_cls is not None:
            self.residuals = residuals_cls(db=self)
        else:
            self.residuals = None
            print(
                "\n\tWARNING: No Residuals class for this format. "
                "Residuals methods will not be available.\n"
            )

        # ── STATS ───────────────────────────────────────────────────────────
        
        stats_cls = STATS_REGISTRY.get(self.format)
        if stats_cls is not None:
            self.stats = stats_cls(db=self)
        else:
            self.stats = None
            print(
                "\n\tWARNING: No Stats class for this format. "
                "Stats methods will not be available.\n"
            )
            
    def _parse(self):
        self.reader.parse_simulation_dirs()
        self.sim_metadata = self.reader.sim_metadata
        self.df_state     = self.reader.df_state

    def _sync_reader(self):
        """Sync attributes computed by the reader back into FRODO."""
        for attr in ('sim_metadata', 'df_state', 'data_dict'):
            if hasattr(self.reader, attr):
                setattr(self, attr, getattr(self.reader, attr))

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_inputs(self, *args, **kwargs):
        self.reader.extract_inputs(*args, **kwargs)
        self._sync_reader()

    def extract_outputs(self, *args, **kwargs):
        self.reader.extract_outputs(*args, **kwargs)
        self._sync_reader()

    def summary_data(self):
        """Print a rich-tree summary of data_dict."""
        if hasattr(self, 'data_dict'):
            SAM.DictVisualizer.rich_tree(self.data_dict)
        else:
            raise KeyError(
                "'data_dict' not found. Run extract_inputs() first."
            )

    def copy(self) -> 'FRODO':
        """Return a deep copy of this FRODO instance."""
        new                 = FRODO.__new__(FRODO)
        new.format          = self.format
        new.root_dir        = self.root_dir
        new.sim_metadata    = copy.deepcopy(self.sim_metadata)
        new.data_dict       = copy.deepcopy(self.data_dict)
        new.kwargs          = copy.deepcopy(self.kwargs)
        new.update_df_state = self.update_df_state
        new.name            = self.name + "_copy"
        new._set_subclasses()
        return new

    # ── merge_datasets — internal helpers ───────────────────────────────────
    #
    # These are pure, format-agnostic helpers that give merge_datasets() a
    # single, unambiguous way to answer two questions:
    #
    #   1. "Are two extracted cases (rows of FlCc) actually the same
    #      physical case?"                       → _case_identity_key
    #   2. "Which row of some other case-indexed DataFrame (df_state,
    #      df_post, ...) corresponds to a given extracted case (row of
    #      FlCc)?"                                → _match_rows_by_identity
    #
    # Both answer these questions using the case's design-variable values
    # (i.e. the physical identity of the case) rather than its position in
    # any particular array — positions are only ever guaranteed to be
    # consistent *within* a single already-extracted FlCc, never across
    # df_cases / df_state / df_post of a source database, since those can
    # legitimately contain cases that were never extracted.

    @staticmethod
    def _case_identity_key(row: np.ndarray, decimals: int) -> tuple:
        """
        Build a hashable identity key for a single case from its
        design-variable values.

        Parameters
        ----------
        row : np.ndarray, shape (n_dvars,)
            Design-variable values for one case (one row of ``FlCc`` or of
            a design-variable-indexed DataFrame).
        decimals : int
            Number of decimal places used to round the values before
            hashing.  This defines the numerical precision at which two
            cases are considered "the same physical case" — see the
            ``dedup_decimals`` parameter of :meth:`merge_datasets`.

        Returns
        -------
        tuple[float, ...]
            Rounded, hashable identity key.

        Examples
        --------
        ::

            key = FRODO._case_identity_key(np.array([3.00000001, 0.75]), 6)
            # (3.0, 0.75)
        """
        return tuple(np.round(np.asarray(row, dtype=np.float64), decimals=decimals))

    @staticmethod
    def _check_no_duplicate_cases(
        flcc_by_source: list,
        decimals: int,
        source_labels: list,
    ) -> None:
        """
        Verify that no two sources contain the same physical case.

        Every row of every source's ``FlCc`` is turned into an identity
        key via :meth:`_case_identity_key`; if the same key is produced by
        two different sources (or twice within the same source), a
        ``ValueError`` is raised immediately, naming both offending
        sources and row positions.

        This function deliberately does **not** deduplicate: the design
        philosophy of :meth:`merge_datasets` is "different cases →
        concatenate, duplicate cases → error", never "duplicate cases →
        silently keep one".

        Parameters
        ----------
        flcc_by_source : list[np.ndarray]
            One ``FlCc`` array (shape ``(n_cases, n_dvars)``) per source,
            in the same order as ``source_labels``.
        decimals : int
            Rounding precision used to build identity keys — see
            ``dedup_decimals`` in :meth:`merge_datasets`.
        source_labels : list[str]
            Human-readable label per source (e.g. ``'source 0'``), used
            only for the error message.

        Raises
        ------
        ValueError
            On the first duplicate found.

        Examples
        --------
        ::

            FRODO._check_no_duplicate_cases(
                [flcc_a, flcc_b], decimals=8,
                source_labels=['source 0', 'source 1'],
            )
        """
        seen: dict = {}
        for i, flcc in enumerate(flcc_by_source):
            for j in range(flcc.shape[0]):
                key = FRODO._case_identity_key(flcc[j], decimals)
                if key in seen:
                    prev_i, prev_j = seen[key]
                    raise ValueError(
                        f"Duplicate case detected: {source_labels[prev_i]} "
                        f"(row {prev_j}) and {source_labels[i]} (row {j}) "
                        f"both resolve to design-variable values {key} "
                        f"(rounded to {decimals} decimals). "
                        "merge_datasets() assumes sources contain disjoint "
                        "sets of cases and does not deduplicate "
                        "automatically; remove the overlap in the source "
                        "databases (or adjust 'dedup_decimals' if this is "
                        "a false positive caused by floating-point noise) "
                        "before merging."
                    )
                seen[key] = (i, j)

    @staticmethod
    def _match_rows_by_identity(
        flcc: np.ndarray,
        design_vars: list,
        reference_df: pd.DataFrame,
        decimals: int,
        context: str,
    ) -> pd.DataFrame:
        """
        Reorder (and filter) a case-indexed DataFrame so that its rows
        correspond, one-to-one and in order, to the extracted cases in
        ``flcc``.

        For every row of ``flcc`` this looks up the single matching row of
        ``reference_df`` — matched on ``design_vars`` values, rounded to
        ``decimals`` places — and returns those rows stitched back
        together in ``flcc``'s row order. This is the mechanism that keeps
        ``df_state`` / ``df_post`` synchronised with the cases that were
        *actually extracted* (``FlCc``) instead of accidentally including
        every case merely *defined* in the source database.

        Parameters
        ----------
        flcc : np.ndarray, shape (n_cases, n_dvars)
            Extracted-case design-variable array (one source's ``FlCc``,
            typically for a single CADGroup).
        design_vars : list[str]
            Column names in ``reference_df`` holding the design-variable
            values, in the same order as the columns of ``flcc``.
        reference_df : pd.DataFrame
            The DataFrame to search (e.g. a source's ``df_state`` or a
            ``get_df_metrics()`` result).
        decimals : int
            Rounding precision for the identity match — see
            ``dedup_decimals`` in :meth:`merge_datasets`.
        context : str
            Human-readable description of ``reference_df``, used only in
            error messages (e.g. ``"df_state of source 0"``).

        Returns
        -------
        pd.DataFrame
            ``len(flcc)`` rows of ``reference_df``, reindexed (0..n-1) to
            match ``flcc``'s case order exactly.

        Raises
        ------
        RuntimeError
            If ``reference_df`` is empty or missing one of ``design_vars``
            as a column; if an extracted case in ``flcc`` has **no**
            matching row in ``reference_df`` (the case was extracted but
            is absent from this table — a genuine inconsistency); or if it
            has **more than one** matching row (the case cannot be
            unambiguously identified in ``reference_df`` at the requested
            precision).

        Examples
        --------
        ::

            matched_state = FRODO._match_rows_by_identity(
                flcc=db.data_dict['CADGroup_3']['FlCc'],
                design_vars=['AoA', 'Mach'],
                reference_df=db.df_state,
                decimals=8,
                context="df_state of source 0",
            )
            assert len(matched_state) == db.data_dict['CADGroup_3']['FlCc'].shape[0]
        """
        if reference_df is None or not isinstance(reference_df, pd.DataFrame) \
                or reference_df.empty:
            raise RuntimeError(
                f"{context} is empty or unavailable; cannot match the "
                "extracted cases against it."
            )

        missing_cols = [v for v in design_vars if v not in reference_df.columns]
        if missing_cols:
            raise RuntimeError(
                f"{context} is missing design-variable column(s) "
                f"{missing_cols}; cannot match extracted cases against it."
            )

        ref_values  = reference_df[design_vars].to_numpy(dtype=np.float64)
        ref_rounded = np.round(ref_values, decimals=decimals)

        matched_positions = []
        for j in range(flcc.shape[0]):
            key = FRODO._case_identity_key(flcc[j], decimals)
            hits = np.where((ref_rounded == np.array(key)).all(axis=1))[0]

            if hits.size == 0:
                raise RuntimeError(
                    f"{context}: no matching row found for extracted case "
                    f"{j} with design-variable values {key}. The case is "
                    "present in FlCc (i.e. it was extracted) but absent "
                    f"from {context}; merging would misalign the "
                    "resulting DataFrames, so this is treated as an error "
                    "instead of silently dropping or padding the row."
                )
            if hits.size > 1:
                raise RuntimeError(
                    f"{context}: {hits.size} rows match extracted case "
                    f"{j} with design-variable values {key}; the case "
                    f"cannot be unambiguously identified in {context} at "
                    f"{decimals}-decimal precision. Increase "
                    "'dedup_decimals' or clean up duplicate rows in the "
                    "source table before merging."
                )
            matched_positions.append(int(hits[0]))

        return reference_df.iloc[matched_positions].reset_index(drop=True)

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def merge_datasets(
        root_dir: str,
        name: Union[str, None],
        sources: list,
        new_group_id: str,
        method: str = 'idw',
        k: int = 4,
        mesh_ref: int = 0,
        cache: bool = True,
        get_df_metrics_attr: dict = {},
        dedup_decimals: int = 8,
    ) -> 'FRODO':
        """
        Merge multiple FRODO datasets into a single unified one.

        Interpolates all source meshes onto a common reference mesh and
        concatenates the ``FlCc`` arrays of every source, then rebuilds
        ``df_cases`` (and, when applicable to the format, ``df_state`` and
        ``df_post``) so that they describe **exactly** the cases that were
        actually extracted into the sources (i.e. the rows already present
        in each source's ``data_dict[...]['FlCc']``) — never the full set
        of cases merely *defined* in a source's ``df_cases`` / ``df_state``.

        Design principles
        ------------------
        * **``FlCc`` is the ground truth of "which cases are being
          merged".** A source's ``df_cases`` / ``df_state`` may describe
          more cases than were ever passed through ``extract_inputs()`` /
          ``extract_outputs()``; only the cases actually present in
          ``FlCc`` end up in the merged database.
        * **No silent deduplication.** If the same physical case (by
          design-variable identity, see ``dedup_decimals``) appears in two
          different sources, :meth:`merge_datasets` raises ``ValueError``
          instead of picking one and discarding the other.
        * **Format-agnostic core.** Case-set resolution, duplicate
          checking, mesh homogenisation, ``FlCc``/``Vars`` concatenation
          and ``df_cases`` construction do not depend on the source
          format. Only the ``df_post`` step (which uses
          ``CODAResiduals.get_df_metrics``) is CODA-specific, and is
          skipped entirely for other formats.
        * **Alignment is guaranteed by construction, not by hoping.**
          ``df_state`` and ``df_post`` are built by matching each
          extracted case (a row of ``FlCc``) against the source table via
          design-variable identity (:meth:`_match_rows_by_identity`), so
          row ``i`` of ``FlCc``, ``df_cases``, ``df_state`` and ``df_post``
          always refers to the same physical case. If a case cannot be
          found (or found unambiguously) in one of these tables, that is
          treated as an error rather than a silently misaligned result.

        Parameters
        ----------
        root_dir : str
            Output root directory for the merged dataset.
        name : str or None
            Name of the merged dataset. If None, defaults to 'FRODO_Merged'.
        sources : list[tuple[FRODO, str]]
            (FRODO_instance, CADGroupID) pairs to merge.
            Example: ``[(db1, '3'), (db2, '3_fine')]``.
        new_group_id : str
            CADGroupID assigned to the merged group in the output.
        method : str
            Mesh interpolation method. Supported: 'idw'. Default 'idw'.
        k : int
            Nearest neighbours for IDW. Default 4.
        mesh_ref : int
            Index of the source used as reference mesh. Default 0.
        cache : bool
            Cache KDTree / interpolation results between sources.
        get_df_metrics_attr : dict
            Keyword arguments for ``db.residuals.get_df_metrics()``.
            Required for format 'CODA'; must be left empty for any other
            format (an explicit ``ValueError`` is raised otherwise, so
            that non-CODA formats never accidentally depend on it).
        dedup_decimals : int
            Number of decimal places used to round design-variable values
            when (a) checking that no case is duplicated across sources
            and (b) matching an extracted case (a row of ``FlCc``) back to
            its row in a source's ``df_state`` / ``get_df_metrics()``
            result. Default 8 (matches the precision previously used,
            implicitly, for case deduplication).

        Returns
        -------
        FRODO
            A new FRODO instance whose ``metadata['df_cases']`` and,
            when applicable, ``df_state`` describe exactly the merged
            cases, in the same order as
            ``data_dict['CADGroup_<new_group_id>']['FlCc']``. For CODA,
            a matching ``df_post`` is additionally computed and saved to
            ``<root_dir>/metadata/df_post.csv``.

        Raises
        ------
        ValueError
            If fewer than two sources are given; if sources do not share
            the same format; if ``mesh_ref`` is not a valid index into
            ``sources``; if ``FlCc`` arrays have incompatible column
            counts; if sources declare inconsistent ``design_vars``; if
            the same physical case is found in more than one source; or
            if ``get_df_metrics_attr`` is given for a non-CODA format (or
            omitted for CODA).
        KeyError
            If a requested CADGroup, or its ``FlCc``, is not present in a
            source's ``data_dict`` (i.e. ``extract_inputs()`` was not run
            for that group).
        RuntimeError
            If an extracted case cannot be found, or cannot be
            unambiguously identified, in a source's ``df_state`` or
            ``get_df_metrics()`` result — this signals that
            ``df_state``/``df_post`` would otherwise end up misaligned
            with ``FlCc``.

        Examples
        --------
        Merge two CODA sources whose cases were fully extracted::

            db_full = FRODO.merge_datasets(
                root_dir='/data/merged',
                name='merged_run',
                sources=[(db1, '3'), (db2, '3')],
                new_group_id='3_merged',
                get_df_metrics_attr={'var_metrics': ['CoefLift', 'CoefDrag']},
            )
            n = db_full.data_dict['CADGroup_3_merged']['FlCc'].shape[0]
            assert len(db_full.metadata['df_cases']) == n
            assert len(db_full.df_state) == n

        Merge a NUMPY-format source pair (no ``get_df_metrics_attr``
        needed, and no ``df_post`` is produced)::

            db_full = FRODO.merge_datasets(
                root_dir='/data/merged_numpy',
                name='merged_numpy',
                sources=[(db1, '3'), (db2, '3')],
                new_group_id='3_merged',
            )
        """
        # ── 0. Validate sources / format ────────────────────────────────────
        if len(sources) < 2:
            raise ValueError("At least 2 sources are required.")

        dbs     = [db for db, _ in sources]
        formats = [db.format for db in dbs]
        if len(set(formats)) != 1:
            raise ValueError("All sources must share the same format.")
        format_ref = formats[0]

        if not isinstance(mesh_ref, int) or isinstance(mesh_ref, bool):
            raise ValueError("mesh_ref must be an integer index.")
        if not (0 <= mesh_ref < len(sources)):
            raise ValueError(
                f"mesh_ref={mesh_ref} is out of range for "
                f"{len(sources)} sources (must be in [0, {len(sources) - 1}])."
            )

        if format_ref == 'CODA':
            if not get_df_metrics_attr:
                raise ValueError(
                    "get_df_metrics_attr must be provided for CODA format."
                )
        else:
            if get_df_metrics_attr:
                raise ValueError(
                    f"get_df_metrics_attr is only supported for format 'CODA' "
                    f"(got '{format_ref}')."
                )

        # ── 1. Validate groups exist and gather each source's extracted FlCc ──
        source_labels = [f"source {i} ('{db.name}', group '{gid}')"
                          for i, (db, gid) in enumerate(sources)]
        source_flcc: list = []
        for i, (db, gid) in enumerate(sources):
            key = f'CADGroup_{gid}'
            if key not in db.data_dict:
                raise KeyError(
                    f"{source_labels[i]}: group '{key}' not found in "
                    "data_dict. Run extract_inputs() for this group first."
                )
            if 'FlCc' not in db.data_dict[key] or db.data_dict[key]['FlCc'] is None:
                raise KeyError(
                    f"{source_labels[i]}: '{key}' has no 'FlCc' array; "
                    "no cases have been extracted for this group."
                )
            source_flcc.append(np.asarray(db.data_dict[key]['FlCc']))

        # ── 2. Validate FlCc column counts and design_vars consistency ──────
        flcc_dims = [f.shape[1] for f in source_flcc]
        if len(set(flcc_dims)) != 1:
            raise ValueError(
                "FlCc arrays have incompatible column counts across sources: "
                f"{dict(zip(source_labels, flcc_dims))}."
            )

        design_vars = dbs[mesh_ref].metadata.get('design_vars')
        if not design_vars or len(design_vars) != flcc_dims[0]:
            raise ValueError(
                "metadata['design_vars'] of the mesh_ref source "
                f"({source_labels[mesh_ref]}) is missing or does not match "
                f"the number of FlCc columns ({flcc_dims[0]})."
            )
        for i, db in enumerate(dbs):
            dv_i = db.metadata.get('design_vars')
            if dv_i is not None and list(dv_i) != list(design_vars):
                raise ValueError(
                    f"{source_labels[i]} declares design_vars={dv_i}, which "
                    f"differs from the mesh_ref source's design_vars="
                    f"{design_vars}. All sources must agree on the meaning "
                    "of each FlCc column before merging."
                )

        # ── 3. Duplicate-case validation (identity = rounded FlCc row) ──────
        FRODO._check_no_duplicate_cases(
            source_flcc, decimals=dedup_decimals, source_labels=source_labels,
        )

        # ── 4. Mesh homogenisation (unchanged behaviour) ─────────────────────
        ref_db, ref_gid = sources[mesh_ref]
        ref_group       = ref_db.data_dict[f'CADGroup_{ref_gid}']

        cache_interp: dict = {}
        processed = []
        for i, (db, gid) in enumerate(sources):
            if i == mesh_ref:
                processed.append((db, gid))
                continue

            ck = (id(db), gid, id(ref_group))
            if cache and ck in cache_interp:
                processed.append((db, cache_interp[ck]))
                continue

            new_id = f"{gid}_merge_tmp_{id(db)}"
            db.sets.interpolate_msh2msh(
                id_group_src=gid, new_group_id=new_id,
                new_mesh=ref_group, method=method, k=k,
            )
            if cache:
                cache_interp[ck] = new_id
            processed.append((db, new_id))

        # 'processed' groups carry the exact same FlCc as before homogenising
        # (interpolate_msh2msh copies 'FlCc' verbatim); re-read it from the
        # (possibly new) group key so downstream code has a single source of
        # truth tied to the group actually used for Vars/mesh concatenation.
        processed_flcc = [
            np.asarray(db.data_dict[f'CADGroup_{gid}']['FlCc'])
            for db, gid in processed
        ]
        flcc_all = np.vstack(processed_flcc)
        n_total  = flcc_all.shape[0]

        dataset_labels = []
        for i, arr in enumerate(processed_flcc):
            dataset_labels.extend([f'dataset_{i}'] * arr.shape[0])

        # ── 5. df_cases: built directly from the extracted FlCc, never from
        #      a source's full df_cases (which may describe cases that were
        #      never extracted). case_idx is freshly assigned for the merged
        #      dataset (0..n_total-1); source case_idx values are not reused.
        df_cases_new = pd.DataFrame(flcc_all, columns=list(design_vars))
        df_cases_new.insert(0, 'case_idx', np.arange(n_total, dtype=np.int32))
        df_cases_new['dataset'] = dataset_labels

        # ── 6. df_state: only built if every source actually has one; each
        #      source's df_state is filtered+reordered to match exactly the
        #      cases present in that source's (processed) FlCc.
        df_state_new = None
        sources_have_state = all(
            isinstance(getattr(db, 'df_state', None), pd.DataFrame)
            and not db.df_state.empty
            for db in dbs
        )
        if sources_have_state:
            state_chunks = []
            for i, ((db, _gid), flcc) in enumerate(zip(processed, processed_flcc)):
                state_chunks.append(
                    FRODO._match_rows_by_identity(
                        flcc=flcc, design_vars=list(design_vars),
                        reference_df=db.df_state, decimals=dedup_decimals,
                        context=f"df_state of {source_labels[i]}",
                    )
                )
            df_state_new = pd.concat(state_chunks, ignore_index=True)
            if len(df_state_new) != n_total:
                raise RuntimeError(
                    "Internal inconsistency: merged df_state has "
                    f"{len(df_state_new)} rows but FlCc has {n_total}."
                )
            df_state_new['case_idx'] = np.arange(n_total, dtype=np.int32)
            df_state_new['dataset']  = dataset_labels

        # ── 7. df_post: CODA-only. Built from get_df_metrics(), then
        #      filtered+reordered to match the extracted FlCc exactly, the
        #      same way df_state is — no independent case-selection logic.
        df_post_new = None
        if format_ref == 'CODA':
            design_vars_lower = [v.lower() for v in design_vars]
            post_chunks = []
            for i, ((db, _gid), flcc) in enumerate(zip(processed, processed_flcc)):
                df_post_full = db.residuals.get_df_metrics(**get_df_metrics_attr)
                df_post_full = df_post_full.rename(columns=str.lower)
                post_chunks.append(
                    FRODO._match_rows_by_identity(
                        flcc=flcc, design_vars=design_vars_lower,
                        reference_df=df_post_full, decimals=dedup_decimals,
                        context=f"get_df_metrics() result of {source_labels[i]}",
                    )
                )
            df_post_new = pd.concat(post_chunks, ignore_index=True)
            if len(df_post_new) != n_total:
                raise RuntimeError(
                    "Internal inconsistency: merged df_post has "
                    f"{len(df_post_new)} rows but FlCc has {n_total}."
                )
            df_post_new['case_idx'] = np.arange(n_total, dtype=np.int32)
            df_post_new['dataset']  = dataset_labels

        # ── 8. Final alignment sanity checks (defence in depth; should never
        #      trigger given the construction above, but a misalignment here
        #      is exactly the class of bug this refactor exists to prevent) ──
        if len(df_cases_new) != n_total:
            raise RuntimeError(
                f"Internal inconsistency: df_cases has {len(df_cases_new)} "
                f"rows but FlCc has {n_total}."
            )
        if df_state_new is not None and len(df_state_new) != len(df_cases_new):
            raise RuntimeError(
                "Internal inconsistency: df_state and df_cases row counts "
                f"differ ({len(df_state_new)} vs {len(df_cases_new)})."
            )
        if df_post_new is not None and len(df_post_new) != len(df_cases_new):
            raise RuntimeError(
                "Internal inconsistency: df_post and df_cases row counts "
                f"differ ({len(df_post_new)} vs {len(df_cases_new)})."
            )

        # ── 9. Build the new FRODO instance ──────────────────────────────────
        db_new              = FRODO.__new__(FRODO)
        db_new.format       = format_ref
        db_new.root_dir     = root_dir
        db_new.name         = name.replace(" ", "_") if name is not None else "FRODO_Merged"
        db_new.sim_metadata = {}
        db_new.kwargs       = {}
        db_new.df_state     = df_state_new

        for d in [root_dir,
                  os.path.join(root_dir, 'metadata'),
                  os.path.join(root_dir, 'outputs')]:
            os.makedirs(d, exist_ok=True)

        for db in dbs:
            for mk, mv in db.sim_metadata.items():
                db_new.sim_metadata.setdefault(mk, mv)

        db_new.metadata = copy.deepcopy(dbs[mesh_ref].metadata)
        db_new.metadata.pop('df_cases', None)
        db_new.metadata['df_cases'] = df_cases_new
        df_cases_new.to_csv(
            os.path.join(root_dir, 'metadata', 'df_cases.csv')
        )

        meta_save = copy.deepcopy(db_new.metadata)
        meta_save['df_cases'] = df_cases_new.to_dict(orient='list')
        with open(
            os.path.join(root_dir, 'metadata', 'cases_metadata.json'), 'w'
        ) as fh:
            json.dump(meta_save, fh, indent=4)

        db_new._set_subclasses()
        db_new.data_dict = {}
        ngk = f'CADGroup_{new_group_id}'
        db_new.data_dict[ngk] = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in ref_group.items()
            if k != "Vars"
        }
        db_new.data_dict[ngk]["FlCc"] = flcc_all

        if df_post_new is not None:
            df_post_new.to_csv(
                os.path.join(root_dir, 'metadata', 'df_post.csv'), sep=','
            )

        # ── 10. Vars: concatenated in the exact same source order as FlCc,
        #       with no post-hoc filtering (there is nothing to filter: every
        #       row of every processed source's FlCc is, by construction, a
        #       distinct case that belongs in the merged dataset). ──────────
        db_new.data_dict[ngk]["Vars"] = {}

        all_stages: set = set()
        for db, gid in processed:
            all_stages.update(db.data_dict[f'CADGroup_{gid}']["Vars"].keys())

        for stage in all_stages:
            db_new.data_dict[ngk]["Vars"][stage] = {}
            all_vars: set = set()
            for db, gid in processed:
                all_vars.update(
                    db.data_dict[f'CADGroup_{gid}']["Vars"].get(stage, {}).keys()
                )

            for var in all_vars:
                ref_shape = None
                for db, gid in processed:
                    vs = db.data_dict[f'CADGroup_{gid}']["Vars"].get(stage, {})
                    if var in vs:
                        ref_shape = vs[var].shape[:-1]
                        break
                if ref_shape is None:
                    continue

                var_list = []
                for db, gid in processed:
                    vs      = db.data_dict[f'CADGroup_{gid}']["Vars"].get(stage, {})
                    n_cases = db.data_dict[f'CADGroup_{gid}']["FlCc"].shape[0]
                    if var not in vs:
                        var_list.append(np.full(ref_shape + (n_cases,), np.nan))
                    else:
                        v = vs[var]
                        if v.ndim not in (2, 3):
                            raise ValueError(
                                f"Variable '{var}' has unsupported ndim {v.ndim}."
                            )
                        var_list.append(v)

                var_concat = np.concatenate(var_list, axis=-1)
                db_new.data_dict[ngk]["Vars"][stage][var] = var_concat

        return db_new