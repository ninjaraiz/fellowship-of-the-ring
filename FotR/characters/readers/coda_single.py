"""
readers/coda_single.py
======================
Reader for CODA_SINGLE datasets: CODA cases with one mesh per case.

Same on-disk content as CODA (``output_<stage>_*.vtu`` + monitor
``.dat`` files per case folder, ``metadata/cases_metadata.json`` with a
``mesh_map`` written by the ``coda_single`` ring), but every case owns
its geometry.  There is deliberately **no shared-mesh invariant**:

* ``parse_simulation_dirs`` joins folders to ``df_cases`` rows by the
  ``'folder'`` name (never by Euclidean distance in design-variable
  space, which collapses on names like ``f0.8``), and records each
  case's mesh from ``mesh_map``.
* ``extract_inputs`` / ``extract_outputs`` keep the CODA signatures
  (``id_groups`` selects physical ``CADGroupID`` zones from each case's
  own VTU) but store **one geometry per case**: ``Coord``,
  ``NodeCoord``, ``Conec`` … and ``Vars[stage][var]`` are lists aligned
  with ``case_order``, since point counts differ between meshes.

Subsets, ``case_per_idx``, ``load_vtu_from_stage``,
``print_available_cadgroup_ids`` and ``plot_state`` are inherited
unchanged from :class:`CODAReader`.
"""

import json
import logging
import os
import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd

from ..sam import SAM
from .coda import CODAReader

log = logging.getLogger(__name__)


class CODASingleReader(CODAReader):
    """
    Reader for CODA_SINGLE-format FRODO databases.

    A *case* is a single subdirectory of ``outputs/``.  Cases may (and
    usually do) use different meshes; only the folder name identifies a
    case, joined against ``metadata['df_cases']['folder']``.

    Attributes set after ``parse_simulation_dirs``
    -----------------------------------------------
    sim_metadata : dict
        Keyed by folder name.  Each value holds ``path``, ``stages``,
        ``mesh`` (``{'file', 'abspath'}`` from ``mesh_map``), and the
        design-variable values.
    df_state : pd.DataFrame
        One row per case, **ordered by ``case_idx``** so that row position
        and ``case_idx`` agree (folders whose name sorts differently from
        their ``df_cases`` position — ``f10`` vs ``f2`` — would otherwise
        desynchronise every consumer that indexes ``df_state`` by
        position). Columns: design vars + ``'mesh_file'`` (basename) +
        ``'stage'`` (completed stages) + ``folder``/``case_idx`` when
        known from ``df_cases``.
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, root_dir: str, **kwargs) -> None:
        """
        Build the reader and additionally load the ``mesh_map`` written by
        ``CODASingleRing``.

        ``CODAReader.__init__`` keeps only the four format-agnostic keys of
        ``cases_metadata.json`` (``eq_type``, ``folder_fmt``,
        ``design_vars``, ``num_stages``) plus ``df_cases``. ``mesh_map`` is
        specific to this format, so it is loaded here rather than widening
        the parent's contract.

        Sets ``self.metadata['mesh_map']`` to a ``{folder: abspath}`` dict
        (empty when the file is missing, unparseable or carries no
        ``mesh_map``), so :meth:`parse_simulation_dirs` can rely on the key
        always existing.
        """
        super().__init__(root_dir, **kwargs)
        self.metadata.setdefault('mesh_map', {})

        # Per CADGroup key, the (case, reason) pairs that extract_inputs
        # could not read. A study still running always has some.
        self.skipped_cases: dict = {}

        meta_path = os.path.join(root_dir, 'metadata', 'cases_metadata.json')
        try:
            with open(meta_path, 'r') as fh:
                cm = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            warnings.warn(
                f"Could not read 'mesh_map' from '{meta_path}' ({exc}). "
                "Each case's mesh will be discovered by scanning its "
                "output folder for a '*.msh' file.",
                UserWarning,
            )
            return

        mesh_map = cm.get('mesh_map') or {}
        if not isinstance(mesh_map, dict):
            warnings.warn(
                f"'mesh_map' in '{meta_path}' is not a dict "
                f"({type(mesh_map).__name__}); ignoring it.",
                UserWarning,
            )
            return

        self.metadata['mesh_map'] = {
            str(folder): path for folder, path in mesh_map.items()
        }

    # ── BaseReader interface ──────────────────────────────────────────

    def parse_simulation_dirs(self) -> None:
        """
        Walk ``outputs/``, join every folder to its ``df_cases`` row by
        name, and record per-case stages and mesh identity.

        Populates ``self.sim_metadata`` and ``self.df_state``.  Folders
        with no matching ``df_cases`` row are kept with empty
        design-vars and a warning instead of being dropped.
        """
        df_cases = self.metadata.get('df_cases')
        if df_cases is None or df_cases.empty:
            raise RuntimeError(
                "metadata['df_cases'] is missing or empty. "
                "CODA_SINGLE requires the cases_metadata.json written by "
                "the coda_single ring (folder-join, no numeric matching)."
            )
        if 'folder' not in df_cases.columns:
            raise RuntimeError(
                "metadata['df_cases'] has no 'folder' column. "
                "CODA_SINGLE joins folders to rows by name; regenerate "
                "the metadata with the coda_single ring."
            )
        design_vars = list(self.metadata.get('design_vars') or [])
        mesh_map = self.metadata.get('mesh_map', {}) or {}

        by_folder = {}
        for pos, folder in enumerate(df_cases['folder'].tolist()):
            if pd.notna(folder):
                by_folder[folder] = pos

        self.sim_metadata = {}
        state_rows: list = []

        for folder in sorted(os.listdir(self.output_dir)):
            full_path = os.path.join(self.output_dir, folder)
            if not os.path.isdir(full_path):
                continue

            pos = by_folder.get(folder)
            if pos is None:
                warnings.warn(
                    f"Folder '{folder}' has no matching row in df_cases; "
                    "keeping it with empty design variables.",
                    UserWarning,
                )
                dv_row = {}
                case_idx = None
            else:
                row = df_cases.iloc[pos]
                dv_row = {
                    var: row[var] for var in design_vars
                    if var in df_cases.columns
                }
                case_idx = (
                    int(row['case_idx'])
                    if 'case_idx' in df_cases.columns
                    and pd.notna(row['case_idx']) else pos
                )

            stage_dict = self._scan_case_stages(full_path)
            mesh_entry = self._resolve_case_mesh(folder, full_path, mesh_map)

            self.sim_metadata[folder] = {
                "folder":            folder,
                "path":              full_path,
                "stages":            stage_dict,
                "computation times": [],
                # NOTE: 'mesh_info', not 'mesh': 'mesh' may itself be a
                # design variable (mesh factor), which arrives below via
                # dv_row and must not clobber the identity dict.
                "mesh_info":         mesh_entry,
            }
            self.sim_metadata[folder].update(dv_row)

            state_row = dict(dv_row)
            state_row.update({
                'mesh_file': mesh_entry['file'],
                'stage':    len(stage_dict),
                'folder':   folder,
                'case_idx': case_idx,
            })
            state_rows.append(state_row)

            if pos is not None:
                df_cases.at[df_cases.index[pos], 'folder'] = folder

        log.info("%s simulations found.", len(self.sim_metadata))

        # df_state is ordered by case_idx, NOT by folder name. Folders are
        # walked in lexicographic order ('f10' < 'f2'), which does not match
        # the df_cases order; leaving df_state in listdir order silently
        # desynchronises every consumer that treats a df_state row position
        # as a case_idx (CODAResiduals.get_df_metrics, CODAReader.plot_state).
        # Cases with no df_cases row (case_idx None) go last, keeping their
        # discovery order.
        df_state = (
            pd.DataFrame.from_records(state_rows)
            if state_rows else pd.DataFrame()
        )
        if not df_state.empty and 'case_idx' in df_state.columns:
            order = df_state['case_idx'].astype('Int64')
            df_state = (
                df_state
                .assign(_sort_key=order.fillna(np.iinfo(np.int32).max))
                .sort_values('_sort_key', kind='stable')
                .drop(columns='_sort_key')
                .reset_index(drop=True)
            )
        self.df_state = df_state

    # ── Mesh helpers ──────────────────────────────────────────────────

    def _resolve_case_mesh(
        self,
        folder: str,
        full_path: str,
        mesh_map: dict,
    ) -> dict:
        """
        Resolve the mesh of one case, preferring the declared ``mesh_map``.

        ``CODASingleRing`` records in ``mesh_map`` the **source** path of
        each mesh (typically a shared ``meshes/`` directory outside
        ``outputs/``) while also copying the file into the case folder.
        Either copy can go missing — the shared directory may be cleaned or
        the dataset moved, and a re-run with ``overwrite=False`` updates
        ``mesh_map`` without recopying — so resolution falls back in order:

        1. the declared path, if it exists on disk;
        2. a file with the declared basename inside the case folder;
        3. any ``*.msh`` found in the case folder (:meth:`_find_case_mesh`).

        Parameters
        ----------
        folder : str
            Case folder name, used as the ``mesh_map`` key.
        full_path : str
            Absolute path of the case folder.
        mesh_map : dict
            ``{folder: abspath}`` from ``metadata['mesh_map']``.

        Returns
        -------
        dict
            ``{'file': basename or None, 'abspath': path or None,
            'source': 'mesh_map' | 'mesh_map_local_copy' | 'scan' | None,
            'declared': declared path or None}``.
        """
        declared = mesh_map.get(folder)

        if declared:
            if os.path.isfile(declared):
                return {'file': os.path.basename(declared),
                        'abspath': os.path.abspath(declared),
                        'source': 'mesh_map', 'declared': declared}

            local = os.path.join(full_path, os.path.basename(declared))
            if os.path.isfile(local):
                return {'file': os.path.basename(local),
                        'abspath': os.path.abspath(local),
                        'source': 'mesh_map_local_copy', 'declared': declared}

            warnings.warn(
                f"Case '{folder}': mesh declared in mesh_map "
                f"('{declared}') was not found, neither at that path nor "
                f"as '{os.path.basename(declared)}' inside the case "
                "folder. Falling back to scanning the folder.",
                UserWarning,
            )

        found = self._find_case_mesh(full_path)
        if found is None:
            if declared:
                warnings.warn(
                    f"Case '{folder}': no mesh file could be resolved.",
                    UserWarning,
                )
            return {'file': None, 'abspath': None,
                    'source': None, 'declared': declared}

        return {'file': os.path.basename(found),
                'abspath': os.path.abspath(found),
                'source': 'scan', 'declared': declared}

    @staticmethod
    def _scan_case_stages(full_path: str) -> dict:
        """Catalogue ``output_<stage>_*`` files of one case folder."""
        stage_dict: dict = {}
        for fname in os.listdir(full_path):
            if fname.startswith("output_"):
                parts = fname.split("_")
                if len(parts) >= 2:
                    stage_raw = os.path.splitext(parts[1])[0]
                    if stage_raw.isdigit():
                        stage = int(stage_raw)
                        ext = os.path.splitext(fname)[-1].lstrip(".")
                        stage_dict.setdefault(
                            stage, {"files": [], "types": set()}
                        )
                        stage_dict[stage]["files"].append(fname)
                        stage_dict[stage]["types"].add(ext)

        for stage in stage_dict:
            stage_dict[stage]["types"] = list(stage_dict[stage]["types"])
        return stage_dict

    # ── Per-case readers (shared by extract_inputs and CODASingleSets) ──

    #: Sorting functions accepted by ``method_to_sort``.
    SORT_FUNCTIONS = {
        'lexsort':     SAM.Weapons.sort_lexsort,
        'centroid':    SAM.Weapons.sort_by_centroid,
        'kdtree':      SAM.Weapons.sort_closed_curve_by_kdtree,
        'convex_hull': SAM.Weapons.sort_points_by_hull_projection,
    }

    @classmethod
    def resolve_sort_fn(cls, method_to_sort: str):
        """Return the sorting callable for ``method_to_sort``."""
        if method_to_sort not in cls.SORT_FUNCTIONS:
            raise ValueError(
                f"method_to_sort '{method_to_sort}' not supported. "
                f"Options: {list(cls.SORT_FUNCTIONS)}."
            )
        return cls.SORT_FUNCTIONS[method_to_sort]

    def read_case_geometry(
        self,
        case_name: str,
        stage: int,
        ids_to_combine,
        vtu_type: Literal['volume', 'surface'] = 'surface',
        sort_fn=None,
    ) -> dict:
        """
        Read one case's geometry for one stage, already sorted.

        This is the single place where a CODA_SINGLE geometry is derived
        from a ``.vtu``. :meth:`extract_inputs` calls it per stage, and the
        lazy accessors of ``CODASingleSets`` call it to re-read a case that
        was never materialised — so both paths cannot drift apart.

        Parameters
        ----------
        case_name : str
            Case folder name (key of ``sim_metadata``).
        stage : int
            Stage to read.
        ids_to_combine : tuple[int]
            ``CADGroupID`` values that make up the group.
        vtu_type : {'surface', 'volume'}
            Which ``.vtu`` to read. Default ``'surface'``.
        sort_fn : callable or None
            Sorting function (see :meth:`resolve_sort_fn`). ``None`` uses
            ``'lexsort'``.

        Returns
        -------
        dict
            ``Coord`` ``(n_cells, 3)``, ``NodeCoord`` ``(n_nodes, 3)``,
            ``Conec`` ``(n_cells, max_nodes)`` already remapped to the
            sorted node numbering, ``eltype`` ``(n_cells,)`` in sorted cell
            order, ``cellOrder``/``pointOrder`` (identity, CODA
            semantics), and the permutations ``idx`` / ``idx_nodes``.

        Raises
        ------
        ValueError
            If the mesh has no ``CADGroupID``, or no cell belongs to the
            requested group.

        Examples
        --------
        ::

            geo = db.reader.read_case_geometry('f10', 0, (3,))
            geo['Coord'].shape
        """
        if sort_fn is None:
            sort_fn = self.resolve_sort_fn('lexsort')

        mesh = self.load_vtu_from_stage(case_name, stage, vtu_type)
        if "CADGroupID" not in mesh.cell_data:
            raise ValueError("'CADGroupID' not found in mesh cell_data.")

        mask = np.isin(mesh.cell_data["CADGroupID"], ids_to_combine)
        if not mask.any():
            raise ValueError(
                f"No cell matches CADGroupID {ids_to_combine}."
            )

        celdas = mesh.extract_cells(mask)
        centroids = np.array(
            celdas.cell_centers().points, dtype=np.float64
        )
        nodes = np.array(celdas.points, dtype=np.float64)

        centroids_sorted, idx = sort_fn(points=centroids)
        nodes_sorted, idx_nodes = sort_fn(points=nodes)

        # Connectivity of the EXTRACTED cells, in their own cell order and
        # indexing celdas.points, then rewritten into the sorted numbering
        # so that it matches Coord/NodeCoord as stored.
        conec = self._reindex_connectivity(
            SAM.Backpack.cell_connectivity_in_order(celdas), idx, idx_nodes,
        )

        return {
            'Coord':      centroids_sorted,
            'NodeCoord':  nodes_sorted,
            'Conec':      conec,
            # eltype is row-aligned with Coord/Conec, so it follows the
            # same cell permutation.
            'eltype':     celdas.celltypes.copy()[idx],
            # cellOrder/pointOrder keep CODAReader's semantics (identity);
            # the permutation itself lives in idx / idx_nodes.
            'cellOrder':  np.arange(celdas.n_cells, dtype=np.float64),
            'pointOrder': np.arange(celdas.n_points, dtype=np.float64),
            'idx':        idx,
            'idx_nodes':  idx_nodes,
        }

    def read_case_vars(
        self,
        case_name: str,
        stage: int,
        ids_to_combine,
        sorter,
        vtu_type: Literal['volume', 'surface'] = 'surface',
        var_names=None,
    ) -> dict:
        """
        Read one case's cell variables for one stage, in sorted order.

        Companion of :meth:`read_case_geometry`, used by the same two
        callers.

        Parameters
        ----------
        case_name : str
            Case folder name.
        stage : int
            Stage to read.
        ids_to_combine : tuple[int]
            ``CADGroupID`` values that make up the group.
        sorter : array-like of int
            Cell permutation (``idx`` from :meth:`read_case_geometry`).
        vtu_type : {'surface', 'volume'}
            Which ``.vtu`` to read. Default ``'surface'``.
        var_names : list[str] or None
            Variables to read. ``None`` reads every ``cell_data`` array.

        Returns
        -------
        dict[str, np.ndarray]
            ``(n_cells,)`` for scalars, ``(n_cells, n_dims)`` for vectors.
            A requested variable missing from this case is absent from the
            result rather than ``None``, so the caller decides how to fill.

        Examples
        --------
        ::

            geo = db.reader.read_case_geometry('f10', 1, (3,))
            v = db.reader.read_case_vars('f10', 1, (3,), geo['idx'])
        """
        mesh = self.load_vtu_from_stage(case_name, stage, vtu_type)
        mask = np.isin(mesh.cell_data["CADGroupID"], ids_to_combine)
        sorter = np.asarray(sorter, dtype=np.int32)

        names = (
            list(mesh.cell_data.keys()) if var_names is None
            else [v for v in var_names if v in mesh.cell_data]
        )

        out = {}
        for var_name in names:
            data_masked = np.array(mesh.cell_data[var_name])[mask]
            if data_masked.ndim == 1:
                out[var_name] = data_masked[sorter].astype(np.float64)
            elif data_masked.ndim == 2:
                out[var_name] = data_masked[sorter, :].astype(np.float64)
            else:
                raise ValueError(
                    f"Variable '{var_name}' has unsupported "
                    f"ndim {data_masked.ndim}."
                )
        return out

    @staticmethod
    def _reindex_connectivity(
        conec: np.ndarray,
        idx_cells: np.ndarray,
        idx_nodes: np.ndarray,
    ) -> np.ndarray:
        """
        Rewrite a connectivity array into the sorted cell/node numbering.

        ``extract_inputs`` stores ``Coord`` and ``NodeCoord`` **sorted** by
        ``method_to_sort``. The connectivity read from the extracted mesh
        is in the mesh's own order and refers to unsorted node indices, so
        it needs both a row permutation (cells) and a value remap (nodes)
        to stay consistent with what is stored.

        Parameters
        ----------
        conec : np.ndarray, shape (n_cells, max_nodes)
            Padded connectivity in original cell order, indexing the
            unsorted node array. ``-1`` marks padding.
        idx_cells : array-like of int, shape (n_cells,)
            Permutation with ``centroids[idx_cells] == centroids_sorted``.
        idx_nodes : array-like of int, shape (n_nodes,)
            Permutation with ``nodes[idx_nodes] == nodes_sorted``.

        Returns
        -------
        np.ndarray
            Same shape as *conec*, rows in sorted-cell order and values
            indexing the sorted node array. Padding stays ``-1``.

        Examples
        --------
        ::

            conec = SAM.Backpack.cell_connectivity_in_order(celdas)
            conec = CODASingleReader._reindex_connectivity(
                conec, idx, idx_nodes
            )
            # conec[k] now indexes NodeCoord (sorted) for Coord row k
        """
        conec = np.asarray(conec)
        if conec.size == 0:
            return conec

        out = conec[np.asarray(idx_cells, dtype=np.int64)]

        idx_nodes = np.asarray(idx_nodes, dtype=np.int64)
        inverse = np.empty(idx_nodes.shape[0], dtype=np.int64)
        inverse[idx_nodes] = np.arange(idx_nodes.shape[0], dtype=np.int64)

        padding = out < 0
        out = inverse[np.where(padding, 0, out)]
        out[padding] = -1
        return out

    @staticmethod
    def _find_case_mesh(full_path: str) -> Union[str, None]:
        """Fallback mesh lookup: first ``*.msh`` file in the folder."""
        for fname in sorted(os.listdir(full_path)):
            if fname.endswith(".msh"):
                return os.path.join(full_path, fname)
        return None

    # ── Extraction (per-case geometry) ────────────────────────────────

    def extract_inputs(
        self,
        id_groups: Union[int, tuple],
        vtu_type: Literal['volume', 'surface'] = 'surface',
        method_to_sort: Literal[
            'lexsort', 'centroid', 'kdtree', 'convex_hull'
        ] = 'lexsort',
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        subset: Union[str, None] = None,
        verbose: bool = False,
    ) -> None:
        """
        Extract mesh and metadata for one or multiple CADGroup IDs.

        Same signature and subset semantics as
        :meth:`CODAReader.extract_inputs`, but every selected case keeps
        its own geometry: ``Coord`` / ``NodeCoord`` / ``Conec`` /
        ``eltype`` / ``cellOrder`` / ``pointOrder`` / ``idx_sort`` /
        ``idx_sort_nodes`` are stored as **lists aligned with
        ``case_order``** (point counts differ between meshes, so no
        shared ``Coord`` and no cross-case consistency check exist).

        Populates ``self.data_dict['CADGroup_<suffix>']`` with
        ``'FlCc'`` ``(n_cases, n_dvars)`` plus the per-case geometry
        lists, ``'mesh_files'``, ``'case_order'``, ``'stages_available'``
        and — only if absent — an empty ``'Vars'`` dict.
        ``'idx_sort'`` / ``'idx_sort_nodes'`` are lists aligned with
        ``case_order`` holding one ``{stage: sorter}`` dict per case.
        Records the resolved global positions in
        ``self.active_cases_idx[key]`` like the parent.

        Unfinished simulations
        ----------------------
        A case is extracted from the stages it **actually has on disk**
        (``sim_metadata[case]['stages']``), not from ``range(num_stages)``:
        a study still running has cases with only some stages written, and
        those must still yield their geometry instead of being dropped
        wholesale. A case with no readable stage at all is skipped, and
        every skipped case is reported in ``self.skipped_cases[key]``.

        Per-case atomicity
        ------------------
        Geometry is committed to the output lists **only once the whole
        case has been read**. A failure half-way (e.g. the inter-stage
        coordinate check below) therefore leaves no partial entry behind,
        so ``Coord[i]`` always describes ``case_order[i]``.
        """
        design_vars = self.metadata.get("design_vars", [])
        df_cases = self.metadata.get("df_cases", pd.DataFrame())

        cases_idx = self._resolve_cases_idx(cases_idx, subset)
        sim_keys = df_cases.loc[cases_idx, "folder"].tolist()
        num_stages = self.metadata.get("num_stages", 1)

        sort_fn = self.resolve_sort_fn(method_to_sort)

        for group_id in id_groups:
            ids_to_combine, key_suffix = self._parse_group_id(group_id)
            key = f"CADGroup_{key_suffix}"

            flcc_rows: list = []
            coords, node_coords, conecs = [], [], []
            eltypes, cell_orders, point_orders = [], [], []
            idx_sorts, idx_sort_nodes = [], []
            mesh_files, case_order, kept = [], [], []
            stages_available: list = []
            skipped: list = []

            for cont, case_i in enumerate(cases_idx):
                sim_key = sim_keys[cont]
                if verbose:
                    log.debug(
                        '\t cont=%s  case=%s  folder=%s',
                        cont, case_i, sim_key,
                    )

                # Stages actually written for this case, capped at
                # num_stages. A study still running has cases with only the
                # first stage(s) present; those are extracted from what
                # exists instead of being dropped.
                on_disk = self.sim_metadata.get(sim_key, {}).get('stages', {})
                stages_here = sorted(
                    s for s in on_disk if 0 <= int(s) < num_stages
                )
                if not stages_here:
                    skipped.append((sim_key, 'no stage available on disk'))
                    warnings.warn(
                        f"Case '{sim_key}' (group {group_id}) has no stage "
                        "available on disk; skipped.",
                        UserWarning,
                    )
                    continue

                # ── Per-case staging area: nothing is committed to the
                #    output lists until the whole case has been read. ──
                case_coord = case_nodes = case_conec = None
                case_eltype = case_cell_order = case_point_order = None
                sorters, sorters_nodes = {}, {}

                try:
                    for stage in stages_here:
                        geo = self.read_case_geometry(
                            sim_key, stage, ids_to_combine, vtu_type, sort_fn,
                        )

                        if case_coord is None:
                            case_coord = geo['Coord']
                            case_nodes = geo['NodeCoord']
                            case_conec = geo['Conec']
                            case_eltype = geo['eltype']
                            case_cell_order = geo['cellOrder']
                            case_point_order = geo['pointOrder']
                        else:
                            for base, current, label in [
                                (case_coord, geo['Coord'], "cell"),
                                (case_nodes, geo['NodeCoord'], "node"),
                            ]:
                                if base.shape != current.shape:
                                    raise ValueError(
                                        f"Inconsistent {label} count at "
                                        f"stage {stage}, case {sim_key}: "
                                        f"{current.shape} vs {base.shape} "
                                        "(same case, different stages)."
                                    )
                                if not SAM.Backpack.same_columns(
                                    np.stack([base, current], axis=0)
                                ):
                                    raise ValueError(
                                        f"Inconsistent {label} coordinates "
                                        f"at stage {stage}, case {sim_key} "
                                        "(same case, different stages)."
                                    )
                        sorters[stage] = geo['idx']
                        sorters_nodes[stage] = geo['idx_nodes']

                    flcc_rows.append([
                        self.sim_metadata[sim_key][p] for p in design_vars
                    ])
                except Exception as exc:
                    skipped.append((sim_key, str(exc)))
                    warnings.warn(
                        f"Error reading inputs for '{sim_key}', "
                        f"group {group_id}: {exc}. "
                        "This case was skipped.",
                        UserWarning,
                    )
                    continue

                # ── Commit: every list grows by exactly one entry. ──
                coords.append(case_coord)
                node_coords.append(case_nodes)
                conecs.append(case_conec)
                eltypes.append(case_eltype)
                cell_orders.append(case_cell_order)
                point_orders.append(case_point_order)
                idx_sorts.append(sorters)
                idx_sort_nodes.append(sorters_nodes)
                stages_available.append(list(stages_here))
                mesh_files.append(
                    self.sim_metadata[sim_key].get("mesh_info", {}).get("file")
                )
                case_order.append(sim_key)
                kept.append(case_i)

            flcc = (
                np.asarray(flcc_rows, dtype=np.float64)
                if flcc_rows
                else np.zeros((0, len(design_vars)), dtype=np.float64)
            )

            group = self.data_dict.setdefault(key, {})
            group.update({
                'Coord':            coords,
                'NodeCoord':        node_coords,
                'FlCc':             flcc,
                'Conec':            conecs,
                'idx_sort':         idx_sorts,
                'idx_sort_nodes':   idx_sort_nodes,
                'eltype':           eltypes,
                'cellOrder':        cell_orders,
                'pointOrder':       point_orders,
                'mesh_files':       mesh_files,
                'case_order':       case_order,
                'stages_available': stages_available,
                # Everything needed to re-read a case from disk without
                # guessing: CODASingleSets' lazy accessors rely on it.
                'group_ids':        tuple(ids_to_combine),
                'vtu_type':         vtu_type,
                'method_to_sort':   method_to_sort,
            })
            # 'Vars' is NOT reset: re-running extract_inputs (to add a
            # group, or after a failure) must not destroy fields already
            # read by extract_outputs. CODAReader behaves the same way.
            group.setdefault('Vars', {})

            self.active_cases_idx[key] = kept
            self.skipped_cases[key] = skipped

            if skipped:
                log.info(
                    "%s: %s/%s case(s) extracted, %s skipped.",
                    key, len(kept), len(cases_idx), len(skipped),
                )

    def extract_outputs(
        self,
        stage: int,
        id_groups: Union[int, tuple],
        vtu_type: Literal['volume', 'surface'] = 'surface',
        cases_idx: Union[list, tuple, range, int, str] = 'all',
        subset: Union[str, None] = None,
        var_name_excluded: Union[list, tuple, None] = None,
        verbose: bool = False,
    ) -> None:
        """
        Extract cell-based output variables, one geometry per case.

        Same signature and subset semantics as
        :meth:`CODAReader.extract_outputs` (the selection must match the
        one used in :meth:`extract_inputs` for the same groups).
        Requires ``extract_inputs`` first (per-case ``idx_sort`` lists).

        Populates ``self.data_dict[key]['Vars'][str(stage)][var]`` with
        **lists aligned with ``case_order``** — one array per case,
        ``(n_points_i,)`` for scalars or ``(n_points_i, n_dims)`` for
        vectors — since point counts differ between meshes.  Cases
        missing a variable found in the first case get ``None`` (with a
        warning) instead of aborting the whole extraction.

        Partially extracted groups
        --------------------------
        The requested selection does not have to match ``extract_inputs``
        exactly: cases that ``extract_inputs`` skipped (unfinished
        simulations) are simply not available here, and asking for them
        only warns. What *is* enforced is that the arrays written stay
        aligned with ``case_order``, so extraction always walks
        ``case_order`` and never the caller's selection. Every stored list
        therefore has ``len(case_order)`` entries; positions outside the
        requested selection keep whatever this stage already held (or
        ``None``), so reading a group case by case accumulates instead of
        overwriting.

        A case that lacks the requested ``stage`` stores ``None`` for
        every variable, with a warning; if **no** selected case has that
        stage, a ``ValueError`` is raised instead of failing later with a
        confusing ``FileNotFoundError``.
        """
        df_cases = self.metadata.get("df_cases", pd.DataFrame())
        cases_idx = self._resolve_cases_idx(cases_idx, subset)
        sim_keys = df_cases.loc[cases_idx, "folder"].tolist()

        for group_id in id_groups:
            ids_to_combine, key_suffix = self._parse_group_id(group_id)
            key = f"CADGroup_{key_suffix}"

            if key not in self.data_dict or \
                    'idx_sort' not in self.data_dict[key]:
                raise RuntimeError(
                    f"No idx_sort found for {key}. "
                    "Run extract_inputs first."
                )

            group = self.data_dict[key]
            case_order = list(group.get('case_order') or [])

            # Walk case_order (the order the stored arrays use), keeping
            # only the cases the caller asked for. Requested cases that are
            # not in the group were skipped by extract_inputs.
            requested = set(sim_keys)
            positions = [
                pos for pos, name in enumerate(case_order)
                if name in requested
            ]
            missing = [name for name in sim_keys if name not in case_order]
            if missing:
                warnings.warn(
                    f"{key}: case(s) {missing} were not extracted by "
                    "extract_inputs (unfinished or failed) and are skipped "
                    "here too.",
                    UserWarning,
                )
            if not positions:
                raise ValueError(
                    f"{key}: none of the requested cases {sim_keys} is "
                    f"present in case_order {case_order}. Run "
                    "extract_inputs for this selection first."
                )

            # Which of them actually have this stage on disk?
            with_stage = [
                pos for pos in positions
                if stage in group['idx_sort'][pos]
            ]
            if not with_stage:
                available = sorted({
                    s for pos in positions for s in group['idx_sort'][pos]
                })
                raise ValueError(
                    f"{key}: stage {stage} is not available for any of the "
                    f"selected cases. Stages present: {available}."
                )

            mesh0 = self.load_vtu_from_stage(
                case_order[with_stage[0]], stage, vtu_type
            )
            var_names = [
                v for v in mesh0.cell_data.keys()
                if var_name_excluded is None or v not in var_name_excluded
            ]
            var_storage: dict = {v: [] for v in var_names}

            # Anything already stored for this stage, so that re-reading a
            # subset does not discard the cases left out of this call.
            previous = group.get('Vars', {}).get(str(stage), {})

            def _previous(var_name: str, pos: int):
                stored = previous.get(var_name)
                if isinstance(stored, list) and len(stored) == len(case_order):
                    return stored[pos]
                return None

            selected = set(positions)

            # Walk the WHOLE case_order so every Vars list keeps the same
            # length and ordering as the geometry lists.
            for pos, sim_key in enumerate(case_order):
                if pos not in selected:
                    for var_name in var_names:
                        var_storage[var_name].append(_previous(var_name, pos))
                    continue

                if verbose:
                    log.debug('\t pos=%s  folder=%s', pos, sim_key)

                if stage not in group['idx_sort'][pos]:
                    warnings.warn(
                        f"{key}: case '{sim_key}' has no stage {stage} "
                        f"(has {sorted(group['idx_sort'][pos])}); storing "
                        "None for every variable.",
                        UserWarning,
                    )
                    for var_name in var_names:
                        var_storage[var_name].append(None)
                    continue

                read = self.read_case_vars(
                    sim_key, stage, ids_to_combine,
                    sorter=group['idx_sort'][pos][stage],
                    vtu_type=vtu_type, var_names=var_names,
                )
                for var_name in var_names:
                    if var_name not in read:
                        warnings.warn(
                            f"Variable '{var_name}' missing in "
                            f"'{sim_key}'; storing None.",
                            UserWarning,
                        )
                        var_storage[var_name].append(None)
                        continue
                    var_storage[var_name].append(read[var_name])

            self.data_dict.setdefault(key, {})
            self.data_dict[key].setdefault('Vars', {})
            self.data_dict[key]['Vars'][str(stage)] = var_storage
