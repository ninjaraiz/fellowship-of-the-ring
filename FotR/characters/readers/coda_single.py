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
        One row per case: design vars + ``'mesh_file'`` (basename) +
        ``'stage'`` (completed stages) + ``folder``/``case_idx`` when
        known from ``df_cases``.
    """

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

            mesh_abspath = mesh_map.get(folder)
            if mesh_abspath is None:
                mesh_abspath = self._find_case_mesh(full_path)
            mesh_entry = (
                {'file': os.path.basename(mesh_abspath),
                 'abspath': mesh_abspath}
                if mesh_abspath is not None else {'file': None, 'abspath': None}
            )

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

        self.df_state = (
            pd.DataFrame.from_records(state_rows)
            if state_rows else pd.DataFrame()
        )

    # ── Mesh helpers ──────────────────────────────────────────────────

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
        lists, ``'mesh_files'``, ``'case_order'`` and an empty
        ``'Vars'`` dict.  ``'idx_sort'`` / ``'idx_sort_nodes'`` are lists
        aligned with ``case_order`` holding one ``{stage: sorter}`` dict
        per case.  Records the resolved global positions in
        ``self.active_cases_idx[key]`` like the parent.
        """
        design_vars = self.metadata.get("design_vars", [])
        df_cases = self.metadata.get("df_cases", pd.DataFrame())

        cases_idx = self._resolve_cases_idx(cases_idx, subset)
        sim_keys = df_cases.loc[cases_idx, "folder"].tolist()
        num_stages = self.metadata.get("num_stages", 1)

        sort_fn_map = {
            'lexsort':     SAM.Weapons.sort_lexsort,
            'centroid':    SAM.Weapons.sort_by_centroid,
            'kdtree':      SAM.Weapons.sort_closed_curve_by_kdtree,
            'convex_hull': SAM.Weapons.sort_points_by_hull_projection,
        }
        if method_to_sort not in sort_fn_map:
            raise ValueError(
                f"method_to_sort '{method_to_sort}' not supported. "
                f"Options: {list(sort_fn_map)}."
            )
        sort_fn = sort_fn_map[method_to_sort]

        for group_id in id_groups:
            ids_to_combine, key_suffix = self._parse_group_id(group_id)
            key = f"CADGroup_{key_suffix}"

            FlCc = np.zeros((len(sim_keys), len(design_vars)), dtype=np.float64)
            coords, node_coords, conecs = [], [], []
            eltypes, cell_orders, point_orders = [], [], []
            idx_sorts, idx_sort_nodes = [], []
            mesh_files, case_order, kept = [], [], []

            for cont, case_i in enumerate(cases_idx):
                sim_key = sim_keys[cont]
                if verbose:
                    log.debug(
                        '\t cont=%s  case=%s  folder=%s',
                        cont, case_i, sim_key,
                    )
                try:
                    sorters, sorters_nodes = {}, {}
                    for stage in range(num_stages):
                        mesh = self.load_vtu_from_stage(
                            sim_key, stage, vtu_type
                        )
                        if "CADGroupID" not in mesh.cell_data:
                            raise ValueError(
                                "'CADGroupID' not found in mesh cell_data."
                            )
                        mask = np.isin(
                            mesh.cell_data["CADGroupID"], ids_to_combine
                        )
                        celdas = mesh.extract_cells(mask)
                        centroids = np.array(
                            celdas.cell_centers().points, dtype=np.float64
                        )
                        nodes = np.array(celdas.points, dtype=np.float64)
                        connectivity = (
                            SAM.Backpack.get_unified_connectivity(mesh)[mask]
                        )
                        centroids_sorted, idx = sort_fn(points=centroids)
                        nodes_sorted, idx_nodes = sort_fn(points=nodes)
                        if stage == 0:
                            coords.append(centroids_sorted)
                            node_coords.append(nodes_sorted)
                            conecs.append(connectivity)
                            eltypes.append(celdas.celltypes.copy())
                            cell_orders.append(
                                np.arange(celdas.n_cells, dtype=np.float64)
                            )
                            point_orders.append(
                                np.arange(celdas.n_points, dtype=np.float64)
                            )
                        else:
                            for base, current, label in [
                                (coords[-1], centroids_sorted, "cell"),
                                (node_coords[-1], nodes_sorted, "node"),
                            ]:
                                if not SAM.Backpack.same_columns(
                                    np.stack([base, current], axis=0)
                                ):
                                    raise ValueError(
                                        f"Inconsistent {label} coordinates "
                                        f"at stage {stage}, case {sim_key} "
                                        "(same case, different stages)."
                                    )
                        sorters[stage] = idx
                        sorters_nodes[stage] = idx_nodes

                    FlCc[len(kept)] = [
                        self.sim_metadata[sim_key][p]
                        for p in design_vars
                    ]
                    mesh_files.append(
                        self.sim_metadata[sim_key].get("mesh_info", {}).get("file")
                    )
                    case_order.append(sim_key)
                    idx_sorts.append(sorters)
                    idx_sort_nodes.append(sorters_nodes)
                    kept.append(case_i)
                except Exception as exc:
                    warnings.warn(
                        f"Error reading inputs for '{sim_key}', "
                        f"group {group_id}: {exc}. "
                        "This case was skipped.",
                        UserWarning,
                    )

            self.data_dict.setdefault(key, {}).update({
                'Coord':          coords,
                'NodeCoord':      node_coords,
                'FlCc':           FlCc[:len(kept)],
                'Conec':          conecs,
                'idx_sort':       idx_sorts,
                'idx_sort_nodes': idx_sort_nodes,
                'eltype':         eltypes,
                'cellOrder':      cell_orders,
                'pointOrder':     point_orders,
                'mesh_files':     mesh_files,
                'case_order':     case_order,
                'Vars':           {},
            })

            self.active_cases_idx[key] = kept

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
            if group.get('case_order') != sim_keys:
                raise ValueError(
                    f"Selection mismatch for {key}: extract_outputs cases "
                    f"{sim_keys} differ from the extract_inputs order "
                    f"{group.get('case_order')}. Use the same selection."
                )

            mesh0 = self.load_vtu_from_stage(sim_keys[0], stage, vtu_type)
            var_names = [
                v for v in mesh0.cell_data.keys()
                if var_name_excluded is None or v not in var_name_excluded
            ]
            var_storage: dict = {v: [] for v in var_names}

            for cont, sim_key in enumerate(sim_keys):
                if verbose:
                    log.debug(
                        '\t cont=%s  case=%s  folder=%s',
                        cont, cases_idx[cont], sim_key,
                    )

                mesh = self.load_vtu_from_stage(sim_key, stage, vtu_type)
                mask = np.isin(mesh.cell_data["CADGroupID"], ids_to_combine)

                sorter = np.asarray(
                    group['idx_sort'][cont][stage], dtype=np.int32
                )
                for var_name in var_names:
                    if var_name not in mesh.cell_data:
                        warnings.warn(
                            f"Variable '{var_name}' missing in "
                            f"'{sim_key}'; storing None.",
                            UserWarning,
                        )
                        var_storage[var_name].append(None)
                        continue
                    data_masked = np.array(mesh.cell_data[var_name])[mask]
                    if data_masked.ndim == 1:
                        var_storage[var_name].append(
                            data_masked[sorter].astype(np.float64)
                        )
                    elif data_masked.ndim == 2:
                        var_storage[var_name].append(
                            data_masked[sorter, :].astype(np.float64)
                        )
                    else:
                        raise ValueError(
                            f"Variable '{var_name}' has unsupported "
                            f"ndim {data_masked.ndim}."
                        )

            self.data_dict.setdefault(key, {})
            self.data_dict[key].setdefault('Vars', {})
            self.data_dict[key]['Vars'][str(stage)] = var_storage
