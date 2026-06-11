import time
import os, re
import json
import numpy as np
import torch
import copy
import pandas as pd
from typing import Literal, Union

import h5py
import pyvista as pv

from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
import glob
import warnings

import pyLOM as SMEAGOL

from ..EarendilsLight import EarendilsLight
from .sam import SAM


class FRODO():
    """
    Framework for Reusable Organized Data Output
    ------------------------------------------------
    One tool to rule them all, one tool to find them.

    FRODO is a lightweight yet powerful assistant for the management,
    organization, and long-term archiving of simulation data, crafted
    to handle multiple CFD cases with the care and precision of a hobbit
    recording tales in the Red Book of Westmarch.

    Supported formats
    -----------------
    - 'CODA'      : CODA CFD solver output (.vtu surface/volume files).
    - 'Airfoil'   : AASM airfoil database (.dat files).
    - 'NUMPYFILE' : Pre-processed numpy dictionaries (.npy files).
    - 'PYLOM'     : pyLOM HDF5 datasets (.h5 / .pkl files).
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Shortcut to Eärendil's Light help system."""
        return cls.light.help(name)

    def __str__(self):
        return f"{self.name}; root_dir: {self.root_dir}; format: {self.format}"

    def __getattr__(self, name):
        """
        Dynamic delegation: if FRODO does not have the attribute, search in
        self.sets, self.reader and self.residuals (in that order).

        This allows calling db.add_aux(...) when add_aux is implemented in
        db.sets without explicitly exposing it on FRODO.
        """
        for sub in ('sets', 'reader', 'residuals'):
            try:
                obj = object.__getattribute__(self, sub)
            except AttributeError:
                obj = None
            if obj is not None and hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __init__(
        self,
        root_dir: str,
        format: Literal['CODA', 'Airfoil', 'NUMPYFILE', 'PYLOM'],
        initial_parse: bool = True,
        **kwargs
    ):
        self.format = format
        self.root_dir = os.path.abspath(root_dir)
        self.sim_metadata = {}
        self.data_dict = {}
        self.kwargs = kwargs
        self.update_df_state = kwargs.pop("update_df_state", False)
        self.name = kwargs.pop("name", 'FRODO Database')

        self._set_subclasses()

        if initial_parse:
            inicio = time.perf_counter()
            self._parse()
            fin_parse = time.perf_counter()
            print(f"Parse taked: {fin_parse - inicio:.4f} seconds")

    def _set_subclasses(self):

        format = self.format

        # ── READER FACTORY ──────────────────────────────────────────────────
        reader_map = {
            "CODA":      self.READERS.CODAReader,
            "Airfoil":   self.READERS.AIRFOILReader,
            "NUMPYFILE": self.READERS.NUMPYFILEReader,
            "PYLOM":     self.READERS.PYLOMReader,
        }

        if format not in reader_map:
            raise ValueError(
                f"Format '{format}' is not supported. "
                f"Available formats: {list(reader_map.keys())}"
            )

        self.reader = reader_map[format](root_dir=self.root_dir, **self.kwargs)

        # ── SETS FACTORY ────────────────────────────────────────────────────
        sets_map = {
            "CODA":      self.SETS.CODASets,
            "Airfoil":   None,
            "NUMPYFILE": self.SETS.NUMPYFILESets,
            "PYLOM":     self.SETS.PYLOMSets,
        }

        sets_cls = sets_map.get(format)
        if sets_cls is not None:
            self.sets = sets_cls(db=self)
        else:
            self.sets = None
            print(
                "\n\tWARNING: This format does not have a Sets class implemented. "
                "Sets methods will not be available in this FRODO instance.\n"
            )

        # ── RESIDUALS FACTORY ───────────────────────────────────────────────
        residuals_map = {
            "CODA":      self.RESIDUALS.CODAResiduals,
            "Airfoil":   None,
            "NUMPYFILE": None,
            "PYLOM":     None,
        }

        residuals_cls = residuals_map.get(format)
        if residuals_cls is not None:
            self.residuals = residuals_cls(db=self)
        else:
            self.residuals = None
            print(
                "\n\tWARNING: This format does not have a Residuals class implemented. "
                "Residuals methods will not be available in this FRODO instance.\n"
            )

    def _parse(self):
        self.reader.parse_simulation_dirs()
        self.sim_metadata = self.reader.sim_metadata
        self.df_state = self.reader.df_state

    def _sync_reader(self):
        """Sync attributes computed by the reader back into the FRODO object."""
        if hasattr(self.reader, "sim_metadata"):
            self.sim_metadata = self.reader.sim_metadata
        if hasattr(self.reader, "df_state"):
            self.df_state = self.reader.df_state
        if hasattr(self.reader, "data_dict"):
            self.data_dict = self.reader.data_dict

    def extract_inputs(self, *args, **kwargs):
        self.reader.extract_inputs(*args, **kwargs)
        self._sync_reader()

    def extract_outputs(self, *args, **kwargs):
        self.reader.extract_outputs(*args, **kwargs)
        self._sync_reader()

    def summary_data(self):
        if hasattr(self, 'data_dict'):
            SAM.DictVisualizer.rich_tree(self.data_dict)
        else:
            raise KeyError(
                'Attribute data_dict not found. '
                'Please run extract_inputs() at least.'
            )

    def copy(self):
        """Create a deep copy of this FRODO object."""
        new_db = FRODO.__new__(FRODO)
        new_db.format = self.format
        new_db.root_dir = self.root_dir
        new_db.sim_metadata = copy.deepcopy(self.sim_metadata)
        new_db.data_dict = copy.deepcopy(self.data_dict)
        new_db.kwargs = copy.deepcopy(self.kwargs)
        new_db.update_df_state = self.update_df_state
        new_db.name = self.name + "_copy"
        new_db._set_subclasses()
        return new_db

    @staticmethod
    def merge_datasets(
        root_dir: str,
        sources: list,
        new_group_id: str,
        method: str = 'idw',
        k: int = 4,
        mesh_ref: int = 0,
        cache: bool = True,
        get_df_metrics_attr: dict = {},
    ) -> 'FRODO':
        """
        Merge multiple FRODO datasets into a single one with a unified mesh
        and FlCc, based on a reference mesh and spatial interpolation.

        Args:
            root_dir (str): Root directory for the merged dataset.
            sources (list[tuple]): List of (FRODO, CADGroupID) pairs to merge.
                Example: [(db1, '3'), (db2, '3_interp')].
            new_group_id (str): CADGroupID assigned to the merged group.
            method (str): Mesh interpolation method. Supported: 'idw'.
                Default 'idw'.
            k (int): Nearest neighbours for IDW. Default 4.
            mesh_ref (int): Index of the source used as reference mesh.
                Default 0.
            cache (bool): Cache KDTree and interpolation results. Default True.
            get_df_metrics_attr (dict): Passed to db.residuals.get_df_metrics()
                for each CODA source. Only supported for CODA format.

        Returns:
            FRODO: New FRODO instance with the merged dataset.
        """
        # ── 0. Validations ──────────────────────────────────────────────────
        if len(sources) < 2:
            raise ValueError("At least 2 datasets are required.")

        dbs = [db for db, _ in sources]
        formats = [db.format for db in dbs]
        if len(set(formats)) != 1:
            raise ValueError("All datasets must share the same format.")

        format_ref = formats[0]

        if format_ref == 'CODA':
            if get_df_metrics_attr:
                csvs = []
                for db, gid in sources:
                    df = db.residuals.get_df_metrics(**get_df_metrics_attr)
                    df.columns = df.columns.str.lower()
                    flcc = db.data_dict[f'CADGroup_{gid}']['FlCc']
                    design_vars_lower = [v.lower() for v in db.metadata['design_vars']]
                    df_flcc = pd.DataFrame(flcc, columns=design_vars_lower)
                    df = df_flcc.merge(df, on=design_vars_lower, how='left')
                    csvs.append(df)
                for i, csv in enumerate(csvs):
                    csv['dataset'] = f'dataset_{i}'
                df_post = pd.concat(csvs, ignore_index=True)
            else:
                raise ValueError(
                    "get_df_metrics_attr not provided for CODA format. "
                    "Please supply it with the parameters needed by "
                    "db.residuals.get_df_metrics()."
                )
        else:
            if get_df_metrics_attr:
                raise ValueError(
                    f"get_df_metrics_attr is only supported for CODA format "
                    f"(received format '{format_ref}')."
                )

        # ── 1. Reference mesh ────────────────────────────────────────────────
        if not isinstance(mesh_ref, int):
            raise ValueError("mesh_ref must be an integer index.")
        ref_db, ref_gid = sources[mesh_ref]
        ref_group = ref_db.data_dict[f'CADGroup_{ref_gid}']

        # ── 2. Validate FlCc column count ────────────────────────────────────
        flcc_dims = [
            db.data_dict[f'CADGroup_{gid}']["FlCc"].shape[1]
            for db, gid in sources
        ]
        if len(set(flcc_dims)) != 1:
            raise ValueError(
                "FlCc arrays have incompatible column counts across sources."
            )

        # ── 3. Cache dicts ───────────────────────────────────────────────────
        cache_kdtree = {}
        cache_interp = {}

        # ── 4. Homogenise meshes ─────────────────────────────────────────────
        processed = []
        for i, (db, gid) in enumerate(sources):
            if i == mesh_ref:
                processed.append((db, gid))
                continue

            cache_key = (id(db), gid, id(ref_group))
            if cache and cache_key in cache_interp:
                processed.append((db, cache_interp[cache_key]))
                continue

            if method == "idw":
                tree_key = (id(db), gid)
                if cache and tree_key in cache_kdtree:
                    tree = cache_kdtree[tree_key]
                else:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(db.data_dict[f'CADGroup_{gid}']["Coord"])
                    if cache:
                        cache_kdtree[tree_key] = tree

            new_id = f"{gid}_merge_tmp_{id(db)}"
            db.sets.interpolate_msh2msh(
                id_group_src=gid,
                new_group_id=new_id,
                new_mesh=ref_group,
                method=method,
                k=k,
            )
            if cache:
                cache_interp[cache_key] = new_id
            processed.append((db, new_id))

        # ── 5. Build new FRODO ───────────────────────────────────────────────
        db_new = FRODO.__new__(FRODO)
        db_new.format = format_ref
        db_new.root_dir = root_dir
        db_new.sim_metadata = {}
        db_new.kwargs = {}

        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(os.path.join(root_dir, 'metadata'), exist_ok=True)
        os.makedirs(os.path.join(root_dir, 'outputs'), exist_ok=True)

        for db in dbs:
            for meta_key, meta_val in db.sim_metadata.items():
                if meta_key not in db_new.sim_metadata:
                    db_new.sim_metadata[meta_key] = meta_val

        db_new.metadata = copy.deepcopy(dbs[mesh_ref].metadata)
        db_new.metadata.pop('df_cases', None)
        db_new.metadata['df_cases'] = (
            df_post[
                db_new.metadata['design_vars'] + ['case_idx', 'dataset']
            ].copy()
            if format_ref == 'CODA' else None
        )
        db_new.metadata['df_cases'].to_csv(
            os.path.join(root_dir, 'metadata', 'df_cases.csv')
        )

        metadata_to_save = copy.deepcopy(db_new.metadata)
        if metadata_to_save.get('df_cases') is not None:
            metadata_to_save['df_cases'] = (
                metadata_to_save['df_cases'].to_dict(orient='list')
            )
        with open(
            os.path.join(root_dir, 'metadata', 'cases_metadata.json'), 'w'
        ) as f:
            json.dump(metadata_to_save, f, indent=4)

        db_new._set_subclasses()
        db_new.data_dict = {}

        new_group_key = f'CADGroup_{new_group_id}'
        db_new.data_dict[new_group_key] = {}
        for key, value in ref_group.items():
            if key != "Vars":
                db_new.data_dict[new_group_key][key] = (
                    value.copy() if isinstance(value, np.ndarray) else value
                )

        # ── 6. FlCc with deduplication ───────────────────────────────────────
        flcc_list = []
        case_splits = []
        for db, gid in processed:
            flcc = db.data_dict[f'CADGroup_{gid}']["FlCc"]
            flcc_list.append(flcc)
            case_splits.append(flcc.shape[0])

        flcc_all = np.vstack(flcc_list)
        assert len(df_post) == flcc_all.shape[0], (
            "Length of df_post does not match the total number of cases in FlCc. "
            "Check get_df_metrics_attr configuration."
        )

        seen = {}
        keep_indices = []
        offset = 0
        for i, n_cases in enumerate(case_splits):
            flcc = flcc_list[i]
            for j in range(n_cases):
                key = tuple(np.round(flcc[j], decimals=8))
                global_idx = offset + j
                if key not in seen:
                    seen[key] = (i, global_idx)
                    keep_indices.append(global_idx)
                else:
                    prev_i, prev_idx = seen[key]
                    if prev_i == mesh_ref and i != mesh_ref:
                        if prev_idx in keep_indices:
                            keep_indices.remove(prev_idx)
                        keep_indices.append(global_idx)
                        seen[key] = (i, global_idx)
            offset += n_cases

        keep_indices = sorted(keep_indices)
        df_post = df_post.iloc[keep_indices].reset_index(drop=True)
        db_new.data_dict[new_group_key]["FlCc"] = flcc_all[keep_indices]
        df_post.to_csv(
            os.path.join(root_dir, 'metadata', 'df_post.csv'), sep=','
        )

        # ── 7. Vars ──────────────────────────────────────────────────────────
        db_new.data_dict[new_group_key]["Vars"] = {}

        all_stages = set()
        for db, gid in processed:
            all_stages.update(db.data_dict[f'CADGroup_{gid}']["Vars"].keys())

        for stage in all_stages:
            db_new.data_dict[new_group_key]["Vars"][stage] = {}

            all_vars = set()
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
                    vs = db.data_dict[f'CADGroup_{gid}']["Vars"].get(stage, {})
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

                if not var_list:
                    continue

                var_concat = np.concatenate(var_list, axis=-1)
                db_new.data_dict[new_group_key]["Vars"][stage][var] = (
                    var_concat[..., keep_indices]
                )

        return db_new

    # =========================================================================
    # READERS
    # =========================================================================
    class READERS:

        class CODAReader():

            def __init__(self, root_dir: str):
                self.root_dir = root_dir
                self.output_dir = os.path.join(self.root_dir, "outputs")
                print(f'\n NEW CODA SIMULATION WILL BE LOADED FROM {root_dir}')

                try:
                    metadata_path = os.path.join(
                        root_dir, 'metadata', 'cases_metadata.json'
                    )
                    with open(metadata_path, 'r') as f:
                        case_metadata = json.load(f)

                    self.metadata = {
                        'eq_type':     case_metadata.get('eq_type', None),
                        'folder_fmt':  case_metadata.get('folder_fmt', None),
                        'design_vars': case_metadata.get('design_vars', None),
                        'num_stages':  case_metadata.get('num_stages', None),
                    }

                    df_cases = pd.DataFrame.from_dict(
                        case_metadata.get('df_cases', {})
                    ).sort_values(
                        by=self.metadata['design_vars'][0],
                        ignore_index=True, axis=0
                    ).reset_index(drop=True)

                    if "case_idx" not in df_cases.columns:
                        df_cases.insert(
                            0, "case_idx", df_cases.index.astype(np.int32)
                        )
                    self.metadata['df_cases'] = df_cases

                except Exception as e:
                    print(
                        "WARNING: cases_metadata.json not found or could not be "
                        "loaded. Folder format will be inferred from folder names.\n"
                    )
                    print(e)
                    folders = os.listdir(self.output_dir)
                    possible_sep = ['_', '-']
                    sep_list, params_list, nfiles_outputs_stages = [], [], []

                    for folder in folders:
                        for sep in possible_sep:
                            if sep in folder:
                                parts = folder.split(sep)
                                params = [
                                    float(re.findall(r"-?\d+\.?\d*", p)[0])
                                    for p in parts
                                    if re.findall(r"-?\d+\.?\d*", p)
                                ]
                                params_list.append(params)
                                sep_list.append(sep)
                                nfiles_outputs_stages.append(len(
                                    SAM.Backpack.find_files(
                                        os.path.join(self.output_dir, folder),
                                        file_end='.h5',
                                        notinfile='ci',
                                    )
                                ))

                    if not params_list:
                        raise ValueError(
                            "No simulation folders found or no numeric parameters "
                            "detected in folder names."
                        )

                    if (
                        all(len(p) == len(params_list[0]) for p in params_list)
                        and all(s == sep_list[0] for s in sep_list)
                        and all(n == nfiles_outputs_stages[0]
                                for n in nfiles_outputs_stages)
                    ):
                        self.metadata = {
                            'eq_type': None,
                            'folder_fmt': sep_list[0].join([
                                p if not re.findall(r"-?\d+\.?\d*", p) else "{}"
                                for p in parts
                            ]),
                            'design_vars': [
                                parts[i] for i in range(0, len(parts), 2)
                            ],
                            'num_stages': nfiles_outputs_stages[0],
                        }
                        df_cases_array = np.zeros(
                            (len(folders), len(params_list[0])), dtype=float
                        )
                        for f_idx, folder in enumerate(folders):
                            df_cases_array[f_idx, :] = params_list[f_idx]

                        df_cases = pd.DataFrame(
                            df_cases_array,
                            columns=self.metadata['design_vars']
                        ).reset_index(drop=True)
                        df_cases.insert(
                            0, "case_idx", df_cases.index.astype(np.int32)
                        )
                        self.metadata['df_cases'] = df_cases
                    else:
                        raise ValueError(
                            "Inconsistent folder naming detected. "
                            "Please provide a valid cases_metadata.json file."
                        )

                self.sim_metadata = {}
                self.df_state = pd.DataFrame()
                self.data_dict = {}

            def parse_simulation_dirs(self):
                """
                Parse the output directory to build simulation metadata and
                state information.

                Populates:
                    - self.sim_metadata: dict mapping each folder to its metadata
                      (design variables, path, stages, computation times).
                    - self.df_state: DataFrame summarising design variables and
                      stage counts.
                """
                folder_fmt = self.metadata["folder_fmt"]
                pattern = SAM.Backpack.folder_fmt_to_pattern(folder_fmt)

                for folder in os.listdir(self.output_dir):
                    if not pattern.match(folder):
                        continue

                    pattern_nums = re.compile(r"[-\d\.]+")
                    nums_folder = np.array(
                        [float(x) for x in pattern_nums.findall(folder)],
                        dtype=float,
                    )
                    nums_df = self.metadata['df_cases'][
                        self.metadata['design_vars']
                    ].values
                    idx_closest = np.argmin(
                        np.linalg.norm(nums_df - nums_folder, axis=1)
                    )
                    nums = (
                        self.metadata['df_cases']
                        .iloc[idx_closest][self.metadata['design_vars']]
                        .values.tolist()
                    )
                    self.metadata['df_cases'].at[idx_closest, 'folder'] = folder

                    full_path = os.path.join(self.output_dir, folder)
                    if not os.path.isdir(full_path):
                        continue

                    stage_dict = {}
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

                    self.sim_metadata[folder] = {
                        "folder": folder,
                        "path": full_path,
                        "stages": stage_dict,
                        "computation times": [],
                    }
                    self.sim_metadata[folder].update(
                        {var: val for var, val in
                         zip(self.metadata["design_vars"], nums)}
                    )

                print(f"{len(self.sim_metadata)} simulations found.")

                n_dvars = len(self.metadata['design_vars'])
                state_array = np.zeros(
                    (len(self.sim_metadata), n_dvars + 1), dtype=float
                )
                for n_sim, sim_key in enumerate(self.sim_metadata):
                    sim = self.sim_metadata[sim_key]
                    for i, var in enumerate(self.metadata['design_vars']):
                        state_array[n_sim, i] = sim[var]
                    state_array[n_sim, -1] = len(sim["stages"])

                df_state = pd.DataFrame(
                    state_array,
                    columns=self.metadata['design_vars'] + ['stage'],
                ).sort_values(
                    by=self.metadata['design_vars'][0]
                ).reset_index(drop=True)

                self.df_state = pd.merge(
                    df_state,
                    self.metadata['df_cases'],
                    on=self.metadata['design_vars'],
                    how='left',
                )

            def print_available_cadgroup_ids(
                self,
                stage,
                vtu_type: Literal["surface", "volume"] = "surface",
            ):
                """
                Print available combinations of CADGroupIDs and cell_data keys
                across all simulations.

                Args:
                    stage (int): Stage number to inspect.
                    vtu_type (str): Type of .vtu file. Default 'surface'.
                """
                summary = defaultdict(list)
                for sim_key, sim in self.sim_metadata.items():
                    try:
                        mesh = self.load_vtu_from_stage(sim_key, stage, vtu_type)
                        cad_ids = tuple(sorted(
                            np.unique(mesh.cell_data["CADGroupID"])
                        ))
                        cell_keys = tuple(sorted(mesh.cell_data.keys()))
                        summary[(cad_ids, cell_keys)].append(sim["folder"])
                    except Exception as e:
                        print(f"Error in simulation {sim_key}: {e}")

                print("\nCADGroupID and cell_data combination summary:")
                for (cad_ids, cell_keys), folders in summary.items():
                    print(f"CADGroupIDs: {cad_ids}")
                    print(f"cell_data keys: {cell_keys}")
                    print(f"Simulations ({len(folders)}): {folders}\n")

            def load_vtu_from_stage(
                self,
                case_name: str,
                stage: int,
                vtu_type: Literal["surface", "volume"] = "surface",
                verbose: bool = False,
            ):
                """
                Load a .vtu file for a given simulation and stage.

                Args:
                    case_name (str): Key in self.sim_metadata.
                    stage (int): Stage number to load.
                    vtu_type (str): 'surface' or 'volume'. Default 'surface'.
                    verbose (bool): Print mesh details if True.

                Returns:
                    pyvista.UnstructuredGrid

                Raises:
                    FileNotFoundError: If no matching .vtu file is found.
                """
                sim = self.sim_metadata[case_name]
                files = sim["stages"].get(stage, {}).get("files", [])
                path = sim["path"]

                for fname in files:
                    if fname.endswith(".vtu") and vtu_type in fname:
                        mesh = pv.read(os.path.join(path, fname))
                        mesh = SAM.Backpack.ensure_cell_data(mesh)
                        if verbose:
                            print(f"Mesh from '{sim['folder']}' loaded")
                            print(f"  Points: {mesh.n_points}")
                            for k in mesh.cells_dict:
                                print(f"  {k}: {mesh.cells_dict[k].shape}")
                        return mesh

                raise FileNotFoundError(
                    f"No .vtu file with type '{vtu_type}' found in "
                    f"stage {stage} of simulation '{sim['folder']}'."
                )

            def extract_inputs(
                self,
                id_groups: Union[int, tuple],
                vtu_type: Literal['volume', 'surface'] = 'surface',
                method_to_sort: Literal[
                    "lexsort", "centroid", "kdtree", "convex_hull"
                ] = 'lexsort',
                cases_idx: Union[list, tuple, int, str] = 'all',
                verbose: bool = False,
            ):
                """
                Extract input mesh and metadata for one or multiple CADGroup IDs.

                Args:
                    id_groups: Tuple of group IDs (int) or tuples of IDs to combine.
                        Examples: (3,) for a single group;
                        ((1, 2), 3) to merge groups 1+2 and also process group 3.
                    vtu_type (str): '.vtu' file type to load. Default 'surface'.
                    method_to_sort (str): Cell-centroid sorting method.
                        Options: 'lexsort', 'centroid', 'kdtree', 'convex_hull'.
                    cases_idx: Subset of cases to process. Default 'all'.
                    verbose (bool): Print per-case progress.

                Populates self.data_dict[key] with:
                    'Coord', 'NodeCoord', 'FlCc', 'Conec',
                    'idx_sort', 'idx_sort_nodes', 'eltype', 'cellOrder', 'pointOrder'.
                """
                num_stages = self.metadata.get("num_stages", 1)
                design_vars = self.metadata.get("design_vars", [])
                df_cases = self.metadata.get("df_cases", pd.DataFrame())

                if isinstance(cases_idx, str):
                    cases_idx = (
                        list(range(len(df_cases)))
                        if cases_idx.lower() == "all"
                        else (_ for _ in ()).throw(
                            ValueError("Invalid string for cases_idx. Use 'all'.")
                        )
                    )
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

                if any(i >= len(df_cases) or i < 0 for i in cases_idx):
                    raise IndexError("cases_idx contains out-of-range values.")

                sim_keys = df_cases.loc[cases_idx, "folder"].tolist()
                ncases = len(sim_keys)

                sort_fn_map = {
                    'lexsort':     SAM.Weapons.sort_lexsort,
                    None:          SAM.Weapons.sort_lexsort,
                    'centroid':    SAM.Weapons.sort_by_centroid,
                    'kdtree':      SAM.Weapons.sort_closed_curve_by_kdtree,
                    'convex_hull': SAM.Weapons.sort_points_by_hull_projection,
                }
                if method_to_sort not in sort_fn_map:
                    raise ValueError(
                        f"method_to_sort '{method_to_sort}' not supported. "
                        f"Options: {list(sort_fn_map.keys())}."
                    )
                sort_function = sort_fn_map[method_to_sort]

                for group_id in id_groups:
                    if isinstance(group_id, tuple):
                        ids_to_combine = group_id
                        key_suffix = "_".join(map(str, ids_to_combine))
                    elif isinstance(group_id, int):
                        ids_to_combine = (group_id,)
                        key_suffix = str(group_id)
                    else:
                        raise TypeError(
                            f"id_groups elements must be int or tuple, "
                            f"got {type(group_id)}."
                        )

                    key = f"CADGroup_{key_suffix}"
                    Coord_base = NodeCoord_base = None
                    Conec_base = eltype_base = cellOrder_base = pointOrder_base = None
                    FlCc = idx_sort = idx_sort_nodes = None

                    for stage in range(num_stages):
                        if verbose:
                            print(f'Stage {stage}:')
                        for cont, case_i in enumerate(cases_idx):
                            sim_key = sim_keys[cont]
                            if verbose:
                                print(
                                    f'\t cont={cont}  case={case_i}  '
                                    f'folder={sim_key}'
                                )
                            try:
                                mesh = self.load_vtu_from_stage(
                                    sim_key, stage, vtu_type
                                )
                                if "CADGroupID" not in mesh.cell_data:
                                    raise ValueError(
                                        "'CADGroupID' not found in cell_data."
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
                                centroids_sorted, idx = sort_function(
                                    points=centroids
                                )
                                nodes_sorted, idx_nodes = sort_function(
                                    points=nodes
                                )

                                if stage == 0 and cont == 0:
                                    npoints = centroids.shape[0]
                                    nnodes = nodes.shape[0]
                                    idx_sort = np.zeros(
                                        (num_stages, ncases, npoints), dtype=np.int32
                                    )
                                    idx_sort_nodes = np.zeros(
                                        (num_stages, ncases, nnodes), dtype=np.int32
                                    )
                                    FlCc = np.zeros(
                                        (ncases, len(design_vars)), dtype=np.float64
                                    )
                                    Coord_base = centroids_sorted.copy()
                                    NodeCoord_base = nodes_sorted.copy()
                                    Conec_base = connectivity.copy()
                                    eltype_base = celdas.celltypes.copy()
                                    cellOrder_base = np.arange(
                                        celdas.n_cells, dtype=np.float64
                                    )
                                    pointOrder_base = np.arange(
                                        celdas.n_points, dtype=np.float64
                                    )
                                else:
                                    for base, current, label in [
                                        (Coord_base, centroids_sorted, "cell"),
                                        (NodeCoord_base, nodes_sorted, "node"),
                                    ]:
                                        if not SAM.Backpack.same_columns(
                                            np.stack([base, current], axis=0)
                                        ):
                                            raise ValueError(
                                                f"Inconsistent {label} coordinates "
                                                f"at stage {stage}, case {sim_key}."
                                            )

                                idx_sort[stage, cont] = idx
                                idx_sort_nodes[stage, cont] = idx_nodes
                                if stage == 0:
                                    FlCc[cont] = [
                                        self.sim_metadata[sim_key][p]
                                        for p in design_vars
                                    ]
                            except Exception as e:
                                print(
                                    f"Error reading inputs for '{sim_key}', "
                                    f"group {group_id}, stage {stage}: {e}"
                                )

                    self.data_dict.setdefault(key, {}).update({
                        'Coord':          Coord_base,
                        'NodeCoord':      NodeCoord_base,
                        'FlCc':           FlCc,
                        'Conec':          Conec_base,
                        'idx_sort':       idx_sort,
                        'idx_sort_nodes': idx_sort_nodes,
                        'eltype':         eltype_base,
                        'cellOrder':      cellOrder_base,
                        'pointOrder':     pointOrder_base,
                    })

            def extract_outputs(
                self,
                stage: int,
                id_groups: Union[int, tuple],
                vtu_type: Literal['volume', 'surface'] = 'surface',
                cases_idx: Union[list, tuple, int, str] = 'all',
                var_name_excluded: Union[list, tuple, None] = None,
                verbose: bool = False,
            ):
                """
                Extract cell-based output variables for one or multiple CADGroup IDs.

                Requirements:
                    extract_inputs must have been called first for the same groups.

                Args:
                    stage (int): Stage number to extract from.
                    id_groups: Same format as in extract_inputs.
                    vtu_type (str): '.vtu' file type. Default 'surface'.
                    cases_idx: Subset of cases. Default 'all'.
                    var_name_excluded (list or None): Variables to skip.
                    verbose (bool): Print progress.

                Populates self.data_dict[key]['Vars'][str(stage)] with arrays of
                shape (n_points, n_cases) for each variable.
                """
                df_cases = self.metadata.get("df_cases", pd.DataFrame())

                if isinstance(cases_idx, str):
                    cases_idx = (
                        list(range(len(df_cases)))
                        if cases_idx.lower() == "all"
                        else (_ for _ in ()).throw(
                            ValueError("Invalid string for cases_idx. Use 'all'.")
                        )
                    )
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

                if any(i >= len(df_cases) or i < 0 for i in cases_idx):
                    raise IndexError("cases_idx contains out-of-range values.")

                sim_keys = df_cases.loc[cases_idx, "folder"].tolist()

                for group_id in id_groups:
                    if isinstance(group_id, tuple):
                        ids_to_combine = group_id
                        key_suffix = "_".join(map(str, ids_to_combine))
                    elif isinstance(group_id, int):
                        ids_to_combine = (group_id,)
                        key_suffix = str(group_id)
                    else:
                        raise TypeError(
                            f"id_groups elements must be int or tuple, "
                            f"got {type(group_id)}."
                        )

                    key = f"CADGroup_{key_suffix}"
                    if key not in self.data_dict or \
                            'idx_sort' not in self.data_dict[key]:
                        raise RuntimeError(
                            f"No idx_sort found for {key}. "
                            "Run extract_inputs first."
                        )

                    idx_sort_all = self.data_dict[key]['idx_sort']
                    if stage >= idx_sort_all.shape[0]:
                        raise ValueError(
                            f"Stage {stage} out of bounds for idx_sort."
                        )

                    mesh0 = self.load_vtu_from_stage(
                        sim_keys[0], stage, vtu_type
                    )
                    var_names = [
                        v for v in mesh0.cell_data.keys()
                        if var_name_excluded is None or v not in var_name_excluded
                    ]
                    var_storage = {v: [] for v in var_names}
                    expected_ncells = None

                    for cont, case_i in enumerate(cases_idx):
                        sim_key = sim_keys[cont]
                        if verbose:
                            print(
                                f'\t cont={cont}  case={case_i}  '
                                f'folder={sim_key}'
                            )
                        mesh = self.load_vtu_from_stage(
                            sim_key, stage, vtu_type
                        )
                        mask = np.isin(
                            mesh.cell_data["CADGroupID"], ids_to_combine
                        )
                        ncells = np.sum(mask)

                        if expected_ncells is None:
                            expected_ncells = ncells
                        elif ncells != expected_ncells:
                            raise ValueError(
                                f"Inconsistent cell count for {key} in "
                                f"'{sim_key}': {ncells} vs {expected_ncells}."
                            )

                        sorter = idx_sort_all[stage, cont].astype(np.int32)
                        for var_name in var_names:
                            data_masked = np.array(
                                mesh.cell_data[var_name]
                            )[mask]
                            if data_masked.ndim == 1:
                                var_storage[var_name].append(data_masked[sorter])
                            elif data_masked.ndim == 2:
                                var_storage[var_name].append(
                                    data_masked[sorter, :]
                                )
                            else:
                                raise ValueError(
                                    f"Variable '{var_name}' has unsupported "
                                    f"ndim {data_masked.ndim}."
                                )

                    self.data_dict.setdefault(key, {})
                    self.data_dict[key].setdefault('Vars', {})
                    self.data_dict[key]['Vars'].setdefault(str(stage), {})

                    for var_name in var_names:
                        arr = np.stack(var_storage[var_name], axis=0)
                        if arr.ndim == 2:
                            self.data_dict[key]['Vars'][str(stage)][var_name] = (
                                arr.T.astype(np.float64)
                            )
                        elif arr.ndim == 3:
                            self.data_dict[key]['Vars'][str(stage)][var_name] = (
                                np.transpose(arr, (2, 1, 0)).astype(np.float64)
                            )
                        else:
                            raise RuntimeError(
                                "Unexpected stacked array dimension."
                            )

            def plot_state(self, figsize=(15, 7)):
                """
                Plot a scatter diagram of the design-variable space with
                simulation completion state colour-coded.

                Args:
                    figsize (tuple): Figure size. Default (15, 7).
                """
                num_states = self.metadata['num_stages']
                design_vars = self.metadata['design_vars']
                df_state = self.df_state

                cmap_custom = plt.get_cmap("RdYlGn", num_states)
                norm = BoundaryNorm(
                    np.arange(-0.5, num_states + 0.5, 1), cmap_custom.N
                )
                dvf = [v for v in design_vars if df_state[v].nunique() > 1]

                if len(dvf) < 2:
                    raise ValueError(
                        "At least two varying design variables are required."
                    )

                legend_handles = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               label='Not started', markerfacecolor='red',
                               markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w',
                               label='In progress', markerfacecolor='yellow',
                               markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w',
                               label='Finished', markerfacecolor='green',
                               markersize=10),
                ]

                if len(dvf) == 2:
                    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
                    ax[0].scatter(
                        df_state[dvf[0]], df_state[dvf[1]],
                        c=df_state['stage'], cmap=cmap_custom, norm=norm, s=100,
                    )
                    ax[0].set(
                        xlabel=dvf[0], ylabel=dvf[1], title="Status of cases"
                    )
                    ax[0].grid()
                    ax[0].legend(
                        handles=legend_handles, loc='lower center',
                        bbox_to_anchor=(1.15, 0.5), ncols=1,
                    )
                    for i, (x, y) in enumerate(
                        zip(df_state[dvf[0]].values, df_state[dvf[1]].values)
                    ):
                        offset = (0, 10) if df_state['stage'][i] != 0 else (0, -12)
                        ax[0].annotate(
                            f"{i}", (x, y), textcoords="offset points",
                            xytext=offset, ha='center', fontsize=8,
                        )

                    mask_finished = (
                        df_state['stage'].values
                        == np.max(df_state['stage'].values)
                    )
                    ax[1].scatter(
                        df_state[dvf[0]][mask_finished],
                        df_state[dvf[1]][mask_finished], s=100,
                    )
                    pct = (mask_finished.sum() / len(mask_finished)) * 100
                    ax[1].set(
                        xlabel=dvf[0], ylabel=dvf[1],
                        title=f"Finished cases {pct:.2f}%",
                    )
                    ax[1].grid()
                    fig.subplots_adjust(wspace=0.3)
                    fig.show()

                elif len(dvf) == 3:
                    fig, ax = plt.subplots(1, 1, figsize=figsize)
                    ax.scatter(
                        df_state[dvf[0]], df_state[dvf[1]], df_state[dvf[2]],
                        c=df_state['stage'], cmap=cmap_custom, norm=norm, s=100,
                    )
                    ax.set(
                        xlabel=dvf[0], ylabel=dvf[1], zlabel=dvf[2],
                        title="Status of cases",
                    )
                    ax.grid()
                    ax.legend(
                        handles=legend_handles, loc='lower center',
                        bbox_to_anchor=(1.15, 0.5), ncols=1,
                    )

            def case_per_idx(self, idx: int) -> str:
                """
                Return the sim_metadata folder name for a given df_state index.

                Tries df_cases['folder'] first; falls back to parameter matching.
                """
                df_cases = self.metadata.get('df_cases', None)
                if df_cases is not None and 'folder' in df_cases.columns:
                    if 'case_idx' in df_cases.columns:
                        match = df_cases.loc[
                            df_cases['case_idx'] == idx, 'folder'
                        ]
                        if not match.empty and pd.notna(match.iloc[0]):
                            return match.iloc[0]
                    try:
                        folder = df_cases.at[idx, 'folder']
                        if pd.notna(folder):
                            return folder
                    except Exception:
                        pass

                row = self.df_state.loc[idx]
                for case_name, sim in self.sim_metadata.items():
                    try:
                        if all(
                            np.isclose(float(sim[var]), float(row[var]))
                            for var in self.metadata['design_vars']
                        ):
                            return case_name
                    except Exception:
                        continue

                raise KeyError(
                    f"No case found in sim_metadata for df_state idx={idx}."
                )

            def plot_integrals_from_case(
                self,
                case_name: str = None,
                case_idx: Union[int, None] = None,
                stage: Union[list, tuple, str] = 'all',
                save_dir: Union[str, None] = None,
                **kwargs,
            ):
                if case_name is None:
                    if case_idx is None:
                        raise ValueError(
                            "Provide either case_name or case_idx."
                        )
                    case_name = self.case_per_idx(case_idx)

                case_path = self.sim_metadata[case_name]['path']
                stages = (
                    list(self.sim_metadata[case_name]['stages'].keys())
                    if stage == 'all' else stage
                )

                files = SAM.Backpack.find_files(
                    case_path, "_wall_boundary_integrals.dat"
                )
                if not files:
                    raise ValueError(
                        f"No '_wall_boundary_integrals.dat' files found in "
                        f"{case_path}."
                    )
                if stages is not None:
                    files = [
                        f for f in files
                        if any(f"_{s}_" in f for s in stages)
                    ]
                    if not files:
                        raise ValueError(
                            f"No files found for stages {stages} in {case_path}."
                        )

                df = SAM.Backpack.get_df_from_csv(files)
                if "Iteration" not in df.columns:
                    raise ValueError(
                        "Column 'Iteration' not found; cannot detect stages."
                    )
                df["stage"] = df["Iteration"].diff().lt(0).cumsum()

                titles = df.columns
                colors = plt.get_cmap("viridis")(
                    np.linspace(0, 1, 3)
                )

                fig, ax = plt.subplots(
                    1, 1, figsize=kwargs.get('figsize', (10, 8))
                )
                for color, cy in zip(colors, [1, 2, 3]):
                    ax.plot(
                        df[titles[0]], df[titles[cy]],
                        color=color, label=f"{titles[cy]}",
                    )
                for idx in df.index[df["stage"].diff().fillna(0) != 0]:
                    ax.axvline(
                        x=df.iloc[idx, 0], color="black",
                        linestyle="--", linewidth=1, alpha=0.5,
                    )
                ax.set_ylabel("Values")
                ax.legend()
                ax.grid(True)
                title = (
                    f"Wall Integrals – {case_name}"
                    if case_idx is None
                    else f"Wall Integrals – Case {case_idx}"
                )
                fig.suptitle(title, fontsize=16)

                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir,
                        f"{case_name}_stages_"
                        f"{'_'.join(map(str, stages))}_wall_integrals.png",
                    )
                    fig.savefig(save_path, bbox_inches='tight')
                    print(f"Figure saved to {save_path}")
                else:
                    plt.show()

        # ─────────────────────────────────────────────────────────────────────
        class AIRFOILReader():

            def __init__(self, root_dir: str):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_data = pd.DataFrame()
                self.data_dict = {}
                print(
                    "Format developed to read the Airfoil Database from AASM "
                    "(Applied Aerodynamics Surrogate Modeling).\n"
                )

            def parse_simulation_dirs(self):
                folders = [
                    os.path.join(self.root_dir, d)
                    for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d))
                ]
                state_array = []
                for folder in folders:
                    sample_files = list(glob.glob(os.path.join(folder, "*")))
                    folder_key = os.path.basename(folder)
                    self.sim_metadata[folder_key] = {}
                    rows = np.zeros((len(sample_files), 8), dtype=object)

                    for nsim, fname in enumerate(sample_files):
                        sd = FRODO.READERS.AIRFOILReader.read_dat(fname)
                        sk = f"Sample_{sd['sample_number']}"
                        self.sim_metadata[folder_key][sk] = sd['metadata']
                        self.sim_metadata[folder_key][sk]['path'] = fname
                        self.sim_metadata[folder_key][sk]['available_vars'] = (
                            list(sd['df_data'].columns)
                        )
                        m = self.sim_metadata[folder_key][sk]
                        rows[nsim, 0] = m['AoA']
                        rows[nsim, 1] = m['Mach']
                        rows[nsim, 2] = m['Re']
                        rows[nsim, 3] = m['Cl']
                        rows[nsim, 4] = m['Cd']
                        rows[nsim, 5] = m['Cmy']
                        rows[nsim, 6] = [[m[f'CST{i}'] for i in range(1, 10)]]
                        rows[nsim, 7] = folder_key

                    state_array.append(rows)

                self.df_state = pd.DataFrame(
                    np.concatenate(state_array, axis=0, dtype=object),
                    columns=[
                        "AoA", "Mach", "Re", "Cl", "Cd", "Cmy",
                        "CST Coord", "Folder",
                    ],
                ).sort_values(by="AoA").reset_index(drop=True)

            def extract_inputs(self):
                """
                Extract coordinates and global parameters from all samples.

                Populates self.data_dict with:
                    'Coord'      – (n_samples, n_points, 2)
                    'Norm_vector'– (n_samples, n_points, 2)
                    'FlCc'       – (n_samples, 3)  [AoA, Mach, Re]
                    'idx_sort'   – (n_samples, n_points)
                """
                coord_list, norm_list, flcc_list, idx_list = [], [], [], []
                folders = [
                    os.path.join(self.root_dir, d)
                    for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d))
                ]
                for folder in folders:
                    for fname in glob.glob(os.path.join(folder, "*")):
                        sd = FRODO.READERS.AIRFOILReader.read_dat(fname)
                        coords = np.stack(
                            [sd['df_data']['x'], sd['df_data']['z']], axis=1
                        )
                        norm = np.stack(
                            [sd['df_data']['nx'], sd['df_data']['nz']], axis=1
                        )
                        idx = np.lexsort((coords[:, 0], coords[:, 1]))
                        coord_list.append(coords[idx])
                        norm_list.append(norm[idx])
                        idx_list.append(idx)
                        flcc_list.append([
                            sd['metadata'].get('AoA', np.nan),
                            sd['metadata'].get('Mach', np.nan),
                            sd['metadata'].get('Re', np.nan),
                        ])

                self.data_dict = {
                    'Coord':       np.array(coord_list, dtype=float),
                    'Norm_vector': np.array(norm_list, dtype=float),
                    'FlCc':        np.array(flcc_list, dtype=float),
                    'idx_sort':    np.array(idx_list, dtype=int),
                }

            def extract_outputs(self):
                """
                Extract surface variables (cp, cfx, cfz, …) for all samples.

                Populates self.data_dict['Vars'] with arrays shaped
                (n_points, n_samples) per variable.
                """
                integrals_list, field_list = [], []
                folders = [
                    os.path.join(self.root_dir, d)
                    for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d))
                ]
                sd = None
                for folder in folders:
                    for fname in glob.glob(os.path.join(folder, "*")):
                        sd = FRODO.READERS.AIRFOILReader.read_dat(fname)
                        integrals_list.append([
                            v for k, v in sd['metadata'].items()
                            if k.startswith("C")
                        ])
                        field_list.append([
                            sd['df_data'][c].values
                            for c in sd['df_data'].columns
                            if c.startswith("c")
                        ])

                if sd is None:
                    return

                integrals_array = np.array(integrals_list, dtype=float)
                field_array = np.array(field_list, dtype=float)
                integrals_name = [
                    k for k in sd['metadata'] if k.startswith("C")
                ]
                field_name = [
                    c for c in sd['df_data'].columns if c.startswith("c")
                ]

                self.data_dict.setdefault('Vars', {})
                self.data_dict['Vars']['integrals'] = {
                    n: integrals_array[:, i]
                    for i, n in enumerate(integrals_name)
                }
                self.data_dict['Vars']['field'] = {
                    n: np.transpose(field_array[:, i, :])
                    for i, n in enumerate(field_name)
                }

            @staticmethod
            def read_dat(path_case: str) -> dict:
                """
                Parse an AASM .dat sample file.

                Args:
                    path_case (str): Full path to the .dat file.

                Returns:
                    dict with keys 'sample_number', 'metadata', 'df_data'.
                """
                with open(path_case, 'r') as fh:
                    first = fh.readline().strip()
                sample_number = int(
                    first.split("_")[0].split("Sample")[1]
                )
                metadata = {}
                ct = 1
                with open(path_case, 'r') as fh:
                    for line in fh:
                        if ":" in line:
                            k, v = line.strip().split(":", 1)
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                            metadata[k] = v
                            ct += 1

                data = np.loadtxt(path_case, skiprows=ct + 1)
                var_names = np.loadtxt(
                    path_case, skiprows=ct, max_rows=1, dtype=str
                )
                return {
                    'sample_number': sample_number,
                    'metadata': metadata,
                    'df_data': pd.DataFrame(data=data, columns=var_names),
                }

        # ─────────────────────────────────────────────────────────────────────
        class NUMPYFILEReader():

            def __init__(
                self,
                root_dir: str,
                file: Union[list, tuple, str],
            ):
                self.root_dir = root_dir

                if isinstance(file, str):
                    self.files = [file]
                elif isinstance(file, (list, tuple)):
                    if all(isinstance(f, str) for f in file):
                        self.files = list(file)
                    else:
                        raise TypeError(
                            "Every element in 'file' must be a str path."
                        )
                else:
                    raise TypeError(
                        "'file' must be a string or a list/tuple of strings."
                    )

                for f in self.files:
                    full = os.path.join(root_dir, f)
                    if not os.path.exists(full):
                        raise FileNotFoundError(f"File not found: {full}")

                self.sim_metadata = {}
                self.data_dict = {"inputs": {}, "outputs": {}, "aux": {}}
                self.npy_dict = {
                    f: np.load(
                        os.path.join(root_dir, f), allow_pickle=True
                    ).item()
                    for f in self.files
                }
                self.df_state = None

            def parse_simulation_dirs(self):
                """
                Analyse the data structure of the .npy file(s) and classify
                variables by shape.
                """
                for f in self.files:
                    self.sim_metadata[f] = {
                        "path": os.path.join(self.root_dir, f),
                        "keys": {
                            k: v.shape
                            for k, v in np.load(
                                os.path.join(self.root_dir, f),
                                allow_pickle=True,
                            ).item().items()
                        },
                    }
                    print(f"Parsed {f}")

            def extract_inputs(
                self,
                keys_inputs: dict,
                keys_aux: dict,
                method_to_sort: Literal[
                    "centroid", "kdtree", "concave_hull"
                ] = 'centroid',
                common: Union[list, None] = None,
                **kwargs,
            ):
                """
                Extract input and auxiliary variables from the .npy dictionary.

                Example::

                    db.extract_inputs(
                        keys_inputs={
                            'ptos': 'db.npy/Airfoil',
                            'aoa':  'db.npy/Alpha',
                        },
                        keys_aux={},
                        common=['ptos'],
                    )

                Args:
                    keys_inputs (dict): Mapping alias → 'file.npy/key'.
                        The alias 'ptos' (mesh coordinates) is mandatory.
                    keys_aux (dict): Mapping alias → 'file.npy/key'.
                    method_to_sort (str): Sorting method for coordinates.
                        Options: 'centroid', 'kdtree', 'concave_hull'.
                    common (list or None): Aliases shared across all cases.
                """
                if common is None:
                    common = []

                self.data_dict["inputs"] = {}
                self.data_dict["aux"] = {}

                for alias, key_path in keys_inputs.items():
                    file_key, key = key_path.split('/')
                    if key not in self.npy_dict[file_key]:
                        raise KeyError(
                            f"Key '{file_key}/{key}' not found in .npy dictionary."
                        )
                    arr = np.asarray(self.npy_dict[file_key][key])
                    if alias not in common and arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    self.data_dict["inputs"][alias] = arr

                if method_to_sort == 'centroid' or method_to_sort is None:
                    self.data_dict["inputs"]['ptos'], self.order_ptos = (
                        SAM.Weapons.sort_by_centroid(
                            points=self.data_dict["inputs"]['ptos']
                        )
                    )
                elif method_to_sort == 'kdtree':
                    self.data_dict["inputs"]['ptos'], self.order_ptos = (
                        SAM.Weapons.sort_closed_curve_by_kdtree(
                            self.data_dict["inputs"]['ptos'],
                            k=kwargs.get('k', 3),
                            start_index=kwargs.get('start_index', 0),
                            alpha=kwargs.get('alpha', 0.7),
                        )
                    )
                elif method_to_sort == 'concave_hull':
                    self.data_dict["inputs"]['ptos'], self.order_ptos = (
                        SAM.Weapons.sort_points_by_hull_projection(
                            self.data_dict["inputs"]['ptos']
                        )
                    )
                else:
                    raise ValueError(
                        f"method_to_sort '{method_to_sort}' not supported. "
                        "Options: 'centroid', 'kdtree', 'concave_hull'."
                    )

                for alias, key_path in keys_aux.items():
                    file_key, key = key_path.split('/')
                    if key not in self.npy_dict[file_key]:
                        raise KeyError(
                            f"Key '{key}' not found in .npy dictionary."
                        )
                    self.data_dict["aux"][alias] = np.asarray(
                        self.npy_dict[file_key][key][self.order_ptos]
                    )

                self.sim_metadata["keys_inputs"] = keys_inputs
                self.sim_metadata["keys_aux"] = keys_aux
                self.sim_metadata["common"] = common
                self._check_input_shapes()

            def extract_outputs(self, keys_outputs: dict):
                """
                Extract output variables from the .npy dictionary.

                Args:
                    keys_outputs (dict): Mapping alias → 'file.npy/key'.

                Example::

                    db.extract_outputs({'cp': 'db.npy/Cp'})
                """
                self.data_dict["outputs"] = {}
                shape_ref = self.data_dict['inputs'][
                    next(iter(self.data_dict["inputs"]))
                ].shape

                for alias, key_path in keys_outputs.items():
                    file_key, key = key_path.split('/')
                    if key not in self.npy_dict[file_key]:
                        raise KeyError(
                            f"Key '{file_key}/{key}' not found in .npy dictionary."
                        )
                    arr = np.asarray(self.npy_dict[file_key][key])
                    if arr.shape[0] == shape_ref[0]:
                        self.data_dict["outputs"][alias] = arr[self.order_ptos]
                    elif arr.shape[1] == shape_ref[0]:
                        self.data_dict["outputs"][alias] = arr.T[self.order_ptos]
                    else:
                        warnings.warn(
                            f"Output '{alias}' has shape {arr.shape}; "
                            "could not determine case/point axes automatically."
                        )
                self.sim_metadata["keys_outputs"] = keys_outputs

            def _check_input_shapes(self):
                """
                Verify that input arrays have at most two distinct first-dimension
                sizes. Warns if more than two are found.
                """
                inputs = self.data_dict.get("inputs", {})
                if not inputs:
                    warnings.warn("data_dict['inputs'] is empty.")
                    return

                first_dims = {
                    name: arr.shape[0]
                    for name, arr in inputs.items()
                    if isinstance(arr, np.ndarray)
                }
                unique_sizes = sorted(set(first_dims.values()))
                self.size_inputs = {
                    size: [n for n, s in first_dims.items() if s == size]
                    for size in unique_sizes
                }
                if len(unique_sizes) > 2:
                    warnings.warn(
                        f"{len(unique_sizes)} distinct first-dimension sizes "
                        f"detected ({unique_sizes}). Check dimensional consistency."
                    )

        # ─────────────────────────────────────────────────────────────────────
        class PYLOMReader():
            """
            Reader for pyLOM datasets stored as .h5 or .pkl files.

            Typical usage
            -------------
            ::

                db = FRODO(root_dir='/path/to/data', format='PYLOM', file='sim.h5')

                db.extract_inputs(
                    keys_inputs={'ptos': 'xyz', 'time': 'time'},
                    keys_aux={},
                )
                db.extract_outputs({'cp': 'Cp', 'vel': 'Velocity'})
            """

            def __init__(self, root_dir: str, file: Union[str, list, tuple],
                         **kwargs):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_state = pd.DataFrame()
                self.data_dict = {"inputs": {}, "outputs": {}, "aux": {}}

                if file is None:
                    raise ValueError("'file' must not be None.")
                if isinstance(file, str):
                    self.files = [file]
                elif isinstance(file, (list, tuple)):
                    if all(isinstance(f, str) for f in file):
                        self.files = list(file)
                    else:
                        raise TypeError(
                            "Every element in 'file' must be a str path."
                        )
                else:
                    raise TypeError(
                        "'file' must be a string or a list/tuple of strings."
                    )

                self.file = self.files[0]
                for f in self.files:
                    full = os.path.join(root_dir, f)
                    if not os.path.exists(full):
                        raise FileNotFoundError(f"File not found: {full}")

                self._dataset = None

            def _load_dataset(self) -> 'SMEAGOL.Dataset':
                """Load and cache the pyLOM Dataset from disk."""
                if self._dataset is None:
                    t0 = time.perf_counter()
                    self._dataset = SMEAGOL.Dataset.load(
                        os.path.join(self.root_dir, self.file)
                    )
                    print(
                        f"[PYLOMReader] Dataset loaded in "
                        f"{time.perf_counter() - t0:.3f} s"
                    )
                return self._dataset

            @staticmethod
            def _field_to_frodo(
                value: np.ndarray, ndim: int, npoints: int
            ) -> np.ndarray:
                """
                Convert a pyLOM field to FRODO storage convention.

                pyLOM layout             →  FRODO layout
                ─────────────────────────────────────────────────
                (ndim*npoints,)          →  (npoints,)
                (ndim*npoints, ncases)   →  (npoints, ncases)
                (ndim*npoints,) ndim>1   →  (ndim, npoints)
                (ndim*npoints, ncases)   →  (ndim, npoints, ncases)
                """
                single = value.ndim == 1
                if ndim == 1:
                    if single:
                        return value.reshape(npoints)
                    ncases = value.shape[1]
                    return value.reshape(npoints, ncases, order='C')
                else:
                    if single:
                        return value.reshape(npoints, ndim, order='C').T
                    ncases = value.shape[1]
                    return (
                        value.reshape(npoints, ndim, ncases, order='C')
                        .transpose(1, 0, 2)
                    )

            def parse_simulation_dirs(self):
                """
                Load the pyLOM Dataset and populate sim_metadata with a summary
                of variables and fields.
                """
                self.sim_metadata = {
                    "path": os.path.join(self.root_dir, self.file)
                }
                ds = self._load_dataset()
                print(ds)

                npoints = len(ds)
                self.sim_metadata["npoints"] = npoints
                self.sim_metadata["xyz_shape"] = ds.xyz.shape
                self.sim_metadata["Vars"] = {
                    vname: {"shape": vdata["value"].shape, "idim": vdata["idim"]}
                    for vname, vdata in ds.vars.items()
                }
                self.sim_metadata["Fields"] = {
                    fname: {"shape": fdata["value"].shape, "ndim": fdata["ndim"]}
                    for fname, fdata in ds.fields.items()
                }

                case_vars = {
                    k: v["value"]
                    for k, v in ds.vars.items()
                    if v["idim"] == 0
                }
                if case_vars:
                    df_dict = {
                        k: np.asarray(arr).ravel()
                        for k, arr in case_vars.items()
                    }
                    try:
                        self.df_state = pd.DataFrame(df_dict)
                    except ValueError:
                        self.df_state = pd.DataFrame(
                            {k: pd.Series(v) for k, v in df_dict.items()}
                        )

                print(
                    f"[PYLOMReader] Parsed '{self.file}'  "
                    f"({npoints} points, "
                    f"{len(ds.vars)} vars, "
                    f"{len(ds.fields)} fields)"
                )

            def extract_inputs(
                self,
                keys_inputs: dict,
                keys_aux: dict,
                filter_by_vars=None,
                filter_by_fields=None,
            ):
                """
                Extract mesh coordinates and parametric variables.

                Args:
                    keys_inputs (dict): Mapping alias → source_key.
                        Use 'xyz' for coordinates. All other keys must exist
                        in the dataset's vars dict.

                        Example::

                            keys_inputs = {
                                'ptos': 'xyz',
                                'time': 'time',
                                'mach': 'Mach',
                            }

                    keys_aux (dict): Mapping alias → field_key for auxiliary
                        spatial fields (from _fieldict).
                    filter_by_vars: Reserved for future use.
                    filter_by_fields: Reserved for future use.
                """
                ds = self._load_dataset()
                npoints = len(ds)
                self.data_dict["inputs"] = {}
                self.data_dict["aux"] = {}

                for alias, src_key in keys_inputs.items():
                    if src_key == "xyz":
                        self.data_dict["inputs"][alias] = ds.xyz.copy()
                    elif src_key in ds.vars:
                        arr = np.asarray(ds.vars[src_key]["value"])
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        self.data_dict["inputs"][alias] = arr
                    else:
                        raise KeyError(
                            f"[PYLOMReader.extract_inputs] Key '{src_key}' not "
                            f"found. Available vars: {list(ds.vars.keys())}  |  "
                            "Use 'xyz' for coordinates."
                        )

                for alias, field_key in keys_aux.items():
                    if field_key not in ds.fields:
                        raise KeyError(
                            f"[PYLOMReader.extract_inputs] Aux key '{field_key}' "
                            f"not found. Available: {list(ds.fields.keys())}"
                        )
                    fdata = ds.fields[field_key]
                    self.data_dict["aux"][alias] = self._field_to_frodo(
                        fdata["value"], fdata["ndim"], npoints
                    )

                self.sim_metadata["keys_inputs"] = keys_inputs
                self.sim_metadata["keys_aux"] = keys_aux

                scalar_inputs = {
                    k: v.ravel()
                    for k, v in self.data_dict["inputs"].items()
                    if k != "ptos" and isinstance(v, np.ndarray) and v.ndim <= 2
                }
                if scalar_inputs:
                    try:
                        self.df_state = pd.DataFrame(scalar_inputs)
                    except ValueError:
                        self.df_state = pd.DataFrame(
                            {k: pd.Series(v) for k, v in scalar_inputs.items()}
                        )

                print(
                    f"[PYLOMReader] extract_inputs done — "
                    f"inputs: {list(self.data_dict['inputs'].keys())}  |  "
                    f"aux: {list(self.data_dict['aux'].keys())}"
                )

            def extract_outputs(self, keys_outputs: dict):
                """
                Extract spatial fields from the dataset.

                Args:
                    keys_outputs (dict): Mapping alias → field_key.

                    Example::

                        db.extract_outputs({'cp': 'Cp', 'vel': 'Velocity'})

                Result shapes:
                    scalar (ndim=1) → (npoints, ncases)
                    vector (ndim>1) → (ndim, npoints, ncases)
                """
                ds = self._load_dataset()
                npoints = len(ds)
                self.data_dict["outputs"] = {}

                for alias, field_key in keys_outputs.items():
                    if field_key not in ds.fields:
                        raise KeyError(
                            f"[PYLOMReader.extract_outputs] Key '{field_key}' not "
                            f"found. Available: {list(ds.fields.keys())}"
                        )
                    fdata = ds.fields[field_key]
                    self.data_dict["outputs"][alias] = self._field_to_frodo(
                        fdata["value"], fdata["ndim"], npoints
                    )

                self.sim_metadata["keys_outputs"] = keys_outputs
                print(
                    f"[PYLOMReader] extract_outputs done — "
                    f"outputs: {list(self.data_dict['outputs'].keys())}"
                )

    # =========================================================================
    # RESIDUALS
    # =========================================================================
    class RESIDUALS:

        class CODAResiduals():

            def __init__(self, db: 'FRODO'):
                self.db = db

            def update_converged_state(
                self,
                threshold: float = 1e-4,
                exclude_residuals: Union[list, tuple] = ['MomentumYResidual'],
            ):
                df_res = self.db.get_all_final_residuals(
                    verbose=False, only_finished=True, load_in_metadata=False
                )
                cols = [
                    c for c in df_res.columns
                    if c.endswith('norm')
                    and all(r not in c for r in exclude_residuals)
                ]
                df_conv = df_res[
                    (df_res[cols] < threshold).all(axis=1)
                ][self.db.metadata['design_vars']]

                self.db.df_state["key"] = list(
                    zip(self.db.df_state[p]
                        for p in self.db.metadata['design_vars'])
                )
                df_conv["key"] = list(
                    zip(df_conv[p]
                        for p in self.db.metadata['design_vars'])
                )
                self.db.df_state["Converged"] = (
                    self.db.df_state["key"].isin(df_conv["key"]).astype(int)
                )
                self.db.df_state.drop(columns=["key"], inplace=True)

            @staticmethod
            def get_df_residuals_from_txt(
                case_path: str,
                verbose: bool = True,
                txt_from_end: int = 1,
            ):
                """
                Parse a CODA residual text file (-out.txt) into a DataFrame.

                Args:
                    case_path (str): Simulation folder path.
                    verbose (bool): Print warnings if True.
                    txt_from_end (int): Index from end of sorted file list.

                Returns:
                    pd.DataFrame or None: Columns
                        ['iters', 'cfl', 'rho_res', 'mom_res', 'energ_res'],
                        or None if no matching file was found.
                """
                files = SAM.Backpack.find_files(
                    case_path, "-out.txt", verbose=False
                )
                if not files:
                    if verbose:
                        print(
                            f"WARNING: No -out.txt file found in {case_path}."
                        )
                    return None

                if txt_from_end:
                    files = [files[-txt_from_end]] if isinstance(files, list) \
                        else [files]

                list_df = []
                regex = re.compile(
                    r"Iteration (\d+):\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
                    r"\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
                )
                for file in files:
                    if verbose:
                        print(f'Reading {file}')
                    with open(file, 'r') as fh:
                        content = fh.read()

                    rows = [
                        (i, float(m.group(2)), float(m.group(3)),
                         float(m.group(4)), float(m.group(5)))
                        for i, m in enumerate(
                            regex.search(line)
                            for line in content.splitlines()
                            if regex.search(line)
                        )
                    ]
                    counters, cfls, rhos, moms, energs = (
                        zip(*rows) if rows else ([], [], [], [], [])
                    )
                    list_df.append(pd.DataFrame({
                        "iters":     list(counters),
                        "cfl":       list(cfls),
                        "rho_res":   list(rhos),
                        "mom_res":   list(moms),
                        "energ_res": list(energs),
                    }))

                return pd.concat(list_df, axis=0, ignore_index=True)

            def get_df_metrics(
                self,
                var_metrics: Union[str, list, tuple],
                iter_var: int = 1000,
                save: bool = False,
            ):
                """
                Build a DataFrame with mean and variance of integral metrics
                over the last *iter_var* iterations for every case and stage.

                Args:
                    var_metrics (str or list[str]): Integral variable names to
                        track (must exist in _wall_boundary_integrals.dat files).
                    iter_var (int): Number of trailing iterations. Default 1000.
                    save (bool): Save df_post to metadata/df_post.csv.

                Returns:
                    pd.DataFrame: df_post with mean/var columns per variable
                        per stage.
                """
                db = self.db
                if isinstance(var_metrics, str):
                    var_metrics = [var_metrics]

                df_post = db.df_state.copy()
                design_vars = db.metadata['design_vars']
                n_stages = db.metadata['num_stages']

                for stage in range(n_stages):
                    for v in var_metrics:
                        df_post[f"{v}_mean_stage{stage}"] = np.nan
                        df_post[f"{v}_var_stage{stage}"] = np.nan

                for stage in range(n_stages):
                    df_finals = db.residuals.get_all_final_residuals(
                        verbose=False, stage=[stage],
                        only_finished=False, load_in_metadata=False,
                    ).copy()
                    df_finals.columns = (
                        df_finals.columns.astype(str).str.lower()
                    )
                    rename_dict = {
                        col: f"{col}_stage{stage}"
                        for col in df_finals.columns
                        if col not in [dv.lower() for dv in design_vars]
                    }
                    df_finals = df_finals.rename(columns=rename_dict)
                    df_post = df_post.merge(df_finals, on=design_vars, how="left")

                for irow in range(len(db.df_state)):
                    case_name = db.case_per_idx(irow)
                    output_path = os.path.join(
                        db.root_dir, 'outputs', case_name
                    )
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
                        stage = int(match.group(1))
                        df_int = SAM.Backpack.get_df_from_csv(
                            files_list=[os.path.join(output_path, fname)]
                        )
                        if not all(v in df_int.columns for v in var_metrics):
                            continue
                        df_tail = df_int[var_metrics].tail(iter_var)
                        for v in var_metrics:
                            df_post.loc[irow, f"{v}_mean_stage{stage}"] = (
                                df_tail[v].mean()
                            )
                            df_post.loc[irow, f"{v}_var_stage{stage}"] = (
                                df_tail[v].var()
                            )

                df_post = df_post.sort_values(
                    by=self.db.metadata['design_vars'][0],
                    ignore_index=True,
                ).reset_index(drop=True)

                if save:
                    df_post.to_csv(
                        os.path.join(db.root_dir, 'metadata', 'df_post.csv')
                    )
                return df_post

            def get_all_final_residuals(
                self,
                stage: Union[list, tuple, str] = 'all',
                verbose: bool = False,
                only_finished: bool = True,
                load_in_metadata: bool = True,
            ):
                df_all = []
                folder_fmt = self.db.metadata.get('folder_fmt', '')
                pattern = SAM.Backpack.folder_fmt_to_pattern(folder_fmt)

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
                    stages_done = len(
                        self.db.sim_metadata[folder]['stages'].keys()
                    )
                    if only_finished and \
                            stages_done < self.db.metadata['num_stages']:
                        if verbose:
                            print(
                                f"Simulation {folder} not finished. "
                                f"{stages_done}/{self.db.metadata['num_stages']}"
                            )
                        continue

                    df_one = self.get_df_residuals_from_case(
                        case_name=folder, stage=stage
                    )
                    if df_one is None:
                        if verbose:
                            print(f"Folder {folder}: no results.")
                        res = [float('nan')] * 26
                    else:
                        res = df_one.tail(1).values[0, :].reshape(1, -1)

                    fila = np.concatenate(
                        (res, np.expand_dims(np.array(params_float), axis=0)),
                        axis=1, dtype=np.float64,
                    )
                    df_all.append(fila)

                names = (
                    list(df_one.columns) + self.db.metadata['design_vars']
                )
                df_final = pd.DataFrame(np.vstack(df_all), columns=names)

                if load_in_metadata:
                    os.makedirs(
                        os.path.join(self.db.root_dir, 'metadata'),
                        exist_ok=True,
                    )
                    df_final.to_csv(
                        os.path.join(
                            self.db.root_dir,
                            'metadata',
                            'all_final_residuals.csv',
                        ),
                        index=False,
                    )
                return df_final

            def integrals_convergence_criteria(
                self,
                iterations_back: int = 1000,
                only_finished: bool = False,
                only_converged: bool = False,
                columns_to_remove: Union[list, tuple] = [
                    'total_iter', 'Iteration', 'Time'
                ],
                mode: Literal['2D', '3D'] = '3D',
                plot: bool = False,
                verbose: bool = False,
                **kwargs,
            ):
                """
                Analyse convergence of integral variables over the last
                *iterations_back* iterations, with optional plotting.

                Args:
                    iterations_back (int): Trailing iterations for analysis.
                    only_finished (bool): Consider only completed simulations.
                    only_converged (bool): Consider only residual-converged cases.
                    columns_to_remove (list): Columns excluded from statistics.
                    mode (str): '2D' scatter or '3D' surface plots.
                    plot (bool): Generate plots if True.
                    verbose (bool): Print progress.

                Returns:
                    tuple[pd.DataFrame, pd.DataFrame]: (result_mean, result_std)
                """
                if only_converged and not only_finished:
                    print(
                        "WARNING: only_converged requires only_finished=True. "
                        "Enabling it automatically."
                    )
                    only_finished = True

                all_means, all_std = [], []
                df_res = self.get_all_final_residuals(
                    verbose=False, only_finished=only_finished,
                    load_in_metadata=False,
                )
                cols = [
                    c for c in df_res.columns
                    if c.endswith('norm') and 'MomentumYResidual' not in c
                ]
                df_filtered = (
                    df_res[
                        (df_res[cols] < 1e-4).all(axis=1)
                    ][self.db.metadata['design_vars']]
                    if only_converged
                    else df_res[self.db.metadata['design_vars']]
                )

                folder_fmt = self.db.metadata.get('folder_fmt', '')
                pattern = SAM.Backpack.folder_fmt_to_pattern(folder_fmt)
                df_integrals_case = None

                for folder_name, dic in self.db.sim_metadata.items():
                    if not re.match(pattern, folder_name):
                        continue
                    valores = list(
                        map(float, re.findall(r"-?\d+\.\d+", folder_name))
                    )
                    stages_done = len(
                        self.db.sim_metadata[folder_name]['stages'].keys()
                    )
                    if only_finished and \
                            stages_done < self.db.metadata['num_stages']:
                        if verbose:
                            print(
                                f"Simulation {folder_name} not finished."
                            )
                        continue

                    mask = np.ones(len(df_filtered), dtype=bool)
                    for val, var in zip(
                        valores, self.db.metadata['design_vars']
                    ):
                        mask &= np.isclose(df_filtered[var].values, val)

                    if not mask.any():
                        if verbose:
                            print(
                                f"Simulation {folder_name} does not meet "
                                "convergence criteria."
                            )
                        continue

                    df_integrals_case = SAM.Backpack.get_df_from_csv(
                        files_list=SAM.Backpack.find_files(
                            path=dic['path'],
                            file_end='_wall_boundary_integrals.dat',
                            verbose=False,
                        )
                    )
                    last = df_integrals_case.tail(iterations_back).drop(
                        columns_to_remove, axis=1
                    )
                    all_means.append(valores + list(last.mean().values))
                    all_std.append(valores  + list(last.std().values))

                if df_integrals_case is None:
                    return pd.DataFrame(), pd.DataFrame()

                columns = (
                    self.db.metadata['design_vars']
                    + list(
                        df_integrals_case.drop(columns_to_remove, axis=1).columns
                    )
                )
                result_mean = pd.DataFrame(np.array(all_means), columns=columns)
                result_std  = pd.DataFrame(np.array(all_std),   columns=columns)

                if plot and len(self.db.metadata['design_vars']) == 2:
                    param1, param2 = self.db.metadata['design_vars']
                    if mode == '2D':
                        fig, ax = plt.subplots(
                            2, len(columns[2:]),
                            figsize=kwargs.get('figsize', (5 * len(columns[2:]), 8)),
                        )
                        ax = ax.flatten()
                        for i, col in enumerate(columns[2:]):
                            sc1 = ax[i].scatter(
                                result_mean[param1], result_mean[param2],
                                c=result_mean[col], cmap='viridis',
                                s=100, edgecolors='k',
                            )
                            plt.colorbar(sc1, ax=ax[i], label=f'Mean {col}')
                            ax[i].set(
                                xlabel=param1, ylabel=param2,
                                title=f'Mean {col}',
                            )
                            ax[i].grid(
                                True, which='both', linestyle='--', linewidth=0.5
                            )
                            sc2 = ax[i + len(columns[2:])].scatter(
                                result_std[param1], result_std[param2],
                                c=result_std[col], cmap='plasma',
                                s=100, edgecolors='k',
                                norm=mcolors.LogNorm(),
                            )
                            plt.colorbar(
                                sc2,
                                ax=ax[i + len(columns[2:])],
                                label=f'Std Dev {col}',
                            )
                            ax[i + len(columns[2:])].set(
                                xlabel=param1, ylabel=param2,
                                title=f'Std Dev {col}',
                            )
                            ax[i + len(columns[2:])].grid(
                                True, which='both', linestyle='--', linewidth=0.5
                            )

                    elif mode == '3D':
                        from scipy.interpolate import griddata
                        for col in columns[2:]:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter3d(
                                x=result_mean[param1], y=result_mean[param2],
                                z=result_mean[col], mode='markers',
                                name=f'Mean {col}',
                                marker=dict(size=5, color='blue'),
                                opacity=0.8,
                            ))
                            g1 = np.linspace(
                                result_mean[param1].min(),
                                result_mean[param1].max(), 50,
                            )
                            g2 = np.linspace(
                                result_mean[param2].min(),
                                result_mean[param2].max(), 50,
                            )
                            G1, G2 = np.meshgrid(g1, g2)
                            Z_mean = griddata(
                                (result_mean[param1], result_mean[param2]),
                                result_mean[col], (G1, G2), method='cubic',
                            )
                            fig.add_trace(go.Surface(
                                x=G1, y=G2, z=Z_mean,
                                colorscale='Blues', opacity=0.5,
                                showscale=False,
                            ))
                            Z_std = griddata(
                                (result_std[param1], result_std[param2]),
                                result_std[col], (G1, G2), method='cubic',
                            )
                            fig.add_trace(go.Scatter3d(
                                x=result_std[param1], y=result_std[param2],
                                z=result_std[col], mode='markers',
                                name=f'Std Dev {col}',
                                marker=dict(size=5, color='red', symbol='diamond'),
                                opacity=0.8,
                            ))
                            fig.add_trace(go.Surface(
                                x=G1, y=G2, z=Z_std,
                                colorscale='Reds', opacity=0.5, showscale=False,
                            ))
                            fig.update_layout(
                                title=f'Integral Variable: {col}',
                                scene=dict(
                                    xaxis_title=param1,
                                    yaxis_title=param2,
                                    zaxis_title=col,
                                ),
                                margin=dict(l=0, r=0, b=0, t=50),
                                width=kwargs.get('width', 1200),
                                height=kwargs.get('height', 800),
                            )
                            fig.show()

                return result_mean, result_std

            def get_df_residuals_from_case(
                self,
                case_name: str = None,
                case_idx: Union[int, None] = None,
                stage: Union[list, tuple, str] = 'all',
                verbose: bool = False,
            ):
                """
                Return a DataFrame with absolute, normalised and scaled residuals
                for all requested stages of the specified case.

                Args:
                    case_name (str): Folder name in sim_metadata.
                    case_idx (int or None): df_state index (used if case_name
                        is None).
                    stage: Stages to include. Default 'all'.
                    verbose (bool): Print per-stage information.

                Returns:
                    pd.DataFrame: Residual data with 'stage' and
                        'total_iterations' columns.
                """
                if case_name is None:
                    if case_idx is None:
                        raise ValueError(
                            "Provide either case_name or case_idx."
                        )
                    resolved = self.db.case_per_idx(case_idx)
                    case_path = self.db.sim_metadata[resolved]['path']
                    stages = (
                        list(self.db.sim_metadata[resolved]['stages'].keys())
                        if stage == 'all' else stage
                    )
                else:
                    case_path = self.db.sim_metadata[case_name]['path']
                    stages = (
                        list(self.db.sim_metadata[case_name]['stages'].keys())
                        if stage == 'all' else stage
                    )

                dfs_stage = []
                for s in stages:
                    df_abs = SAM.Backpack.get_df_from_csv([
                        os.path.join(
                            case_path,
                            f"output_{s}__monitors_TimeIntegration.dat",
                        )
                    ])
                    df_init = SAM.Backpack.get_df_from_csv([
                        os.path.join(
                            case_path,
                            f"output_{s}__monitors_stage{s}InitialResidual.dat",
                        )
                    ])
                    df_cfl = SAM.Backpack.get_df_from_csv([
                        os.path.join(
                            case_path,
                            f"output_{s}__monitors_CFLRamp.dat",
                        )
                    ])

                    r0_vals = df_init.iloc[0].to_dict()
                    ref_cols = [
                        c for c in df_cfl.columns
                        if c.startswith("SERReference")
                    ]
                    ref_vals = df_cfl[ref_cols].copy()
                    ref_vals.columns = [
                        re.sub(r"^SERReference", "", c)
                        for c in ref_cols
                    ]

                    df_stage = df_abs.copy()
                    for col in df_abs.columns:
                        if "Residual" in col:
                            r_abs = df_abs[col]
                            r0 = r0_vals.get(col, None)
                            S = ref_vals[col]
                            df_stage[f"{col}_norm"] = r_abs / r0
                            df_stage[f"{col}_scaled"] = r_abs / S

                    df_stage["stage"] = s
                    dfs_stage.append(df_stage)

                    if verbose:
                        print(
                            f"[INFO] Stage {s}: {len(df_stage)} iterations  |  "
                            f"vars: "
                            f"{[c for c in df_stage.columns if 'Residual' in c]}"
                        )

                df_all = pd.concat(dfs_stage, ignore_index=True)
                df_all['total_iterations'] = np.arange(len(df_all))
                return df_all

            def plot_residuals_from_case(
                self,
                case_name: str = None,
                case_idx: Union[int, None] = None,
                stage: Union[list, tuple, str] = 'all',
                mode: Literal['absolute', 'norm', 'scaled'] = 'scaled',
                save_dir: Union[str, None] = None,
                verbose: bool = False,
                **kwargs,
            ):
                if case_name is None:
                    if case_idx is None:
                        raise ValueError(
                            "Provide either case_name or case_idx."
                        )
                    case_name = self.db.case_per_idx(case_idx)

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
                    if 'Residual' in c and 'MomentumYResidual' not in c
                    and mode in c
                ]
                colors = cm.tab10.colors[:len(columns)]
                for ycol, color in zip(columns, colors):
                    df_res.plot(
                        x='total_iterations', y=ycol, s=3,
                        kind='scatter', ax=ax,
                        label=ycol.replace("Residual_" + mode, ''),
                        color=color, grid=True, logy=True,
                    )
                ax.set(
                    title=f'Case {case_name}',
                    ylim=(1e-8, 1e2),
                    ylabel=f'Residual ({mode})',
                )
                ax.legend(
                    bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3
                )

                divider = make_axes_locatable(ax)
                ax_cfl = divider.append_axes(
                    "bottom", size="35%", pad=0.1, sharex=ax
                )
                ax_cfl.scatter(
                    x=df_res['total_iterations'], y=df_res['CFL'],
                    color='black', s=1.5,
                )
                ax_cfl.set(ylabel="CFL", xlabel="Iterations")
                ax_cfl.set_yscale('log')
                ax_cfl.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)
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
            ):
                """
                Plot scatter maps of final residuals for all simulations.

                Args:
                    save_dir (str or None): Save to folder instead of displaying.
                    mode (str): 'scaled', 'norm' or 'absolute'.
                    stage: Stages to include.
                    only_finished (bool): Only finished simulations.
                    print_non_converged (bool): Print and save non-converged cases.
                    activate_idx (bool): Annotate points with df_state index.
                    ncols (int): Subplot columns. Default 2.
                    lim_converged (float): Residual convergence threshold.
                """
                df_finals = self.get_all_final_residuals(
                    verbose=False, stage=stage,
                    only_finished=only_finished, load_in_metadata=False,
                )
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
                axes = axes.flatten()
                converged_mask = (df_finals[columns].lt(lim_converged)).all(axis=1)
                norm = mcolors.LogNorm(vmin=lim_converged, vmax=1e0)
                color = kwargs.get('cmap', 'summer')
                dvf = [
                    v for v in self.db.metadata['design_vars']
                    if df_finals[v].nunique() > 1
                ]

                if len(dvf) == 2:
                    for i, col in enumerate(columns):
                        x = df_finals[dvf[0]]
                        y = df_finals[dvf[1]]
                        c = df_finals[col]
                        sc_nc = axes[i].scatter(
                            x[~converged_mask], y[~converged_mask],
                            c=c[~converged_mask], cmap=color, norm=norm,
                            s=60, edgecolor='k', label='Non-converged',
                        )
                        axes[i].scatter(
                            x[converged_mask], y[converged_mask],
                            c=c[converged_mask], cmap=color, norm=norm,
                            s=60, marker='*', linewidth=1.5, label='Converged',
                        )
                        if activate_idx:
                            for p in df_finals[dvf].values:
                                idx = np.where(
                                    (self.db.df_state.iloc[:, 0] == p[0])
                                    & (self.db.df_state.iloc[:, 1] == p[1])
                                )[0][0]
                                axes[i].annotate(
                                    f"{idx}", (p[0], p[1]),
                                    textcoords="offset points",
                                    xytext=(0, 7), ha='center', fontsize=8,
                                )
                        axes[i].set(
                            title=col, xlabel=dvf[0], ylabel=dvf[1]
                        )
                        fig.colorbar(
                            sc_nc, ax=axes[i]
                        ).ax.set_title(f'Residual stage {stage}')

                    handles, labels = axes[0].get_legend_handles_labels()
                    fig.legend(
                        handles, labels, loc='lower center',
                        frameon=False, ncols=2,
                    )

                if save_dir:
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
                            self.db.root_dir,
                            'metadata',
                            'non_converged_cases.csv',
                        ),
                        index=False,
                    )
                    for i, row in df_finals[~converged_mask].iterrows():
                        residuals_exp = " ".join(
                            f"{val:.2E}" for val in row[3:].values
                        )
                        print(
                            f"Case {i}: {dvf[0]}={row[dvf[0]]:.4f}, "
                            f"{dvf[1]}={row[dvf[1]]:.4f},  "
                            f"Residuals: {residuals_exp}"
                        )

            def plot_state_calculation(
                self,
                num_stages: int = 1,
                txt_from_end: int = 1,
                figsize: tuple = None,
            ):
                data_to_plot = []
                for name, case in self.db.sim_metadata.items():
                    if len(case['stages'].keys()) == num_stages:
                        df = FRODO.RESIDUALS.CODAResiduals.get_df_residuals_from_txt(
                            case_path=case['path'],
                            verbose=False,
                            txt_from_end=txt_from_end,
                        )
                        if df is not None:
                            data_to_plot.append((name, df))
                        else:
                            print(
                                f'\tWARNING: Case {name} has not started yet.\n'
                            )

                if not data_to_plot:
                    print("No data found for the specified criteria.")
                    return

                nrows = (len(data_to_plot) + 1) // 2
                ncols = 2 if len(data_to_plot) > 1 else 1
                if figsize is None:
                    figsize = (15, 6 * nrows)

                fig, ax = plt.subplots(
                    nrows, ncols, figsize=figsize, squeeze=False
                )
                ax_flat = ax.flatten()

                for i, (name, df_txt) in enumerate(data_to_plot):
                    cur = ax_flat[i]
                    for res, color in zip(
                        df_txt.columns.to_list()[2:],
                        ['blue', 'orange', 'green'],
                    ):
                        cur.scatter(
                            x=df_txt['iters'], y=df_txt[res],
                            color=color, label=res, s=1.5,
                        )
                    cur.set_yscale('log')
                    cur.set(title=name, ylabel="Residuals")
                    cur.grid(
                        which='both', linestyle='-', linewidth=0.5, alpha=0.3
                    )
                    cur.legend(
                        loc='upper right', fontsize='small', markerscale=4
                    )
                    cur.tick_params(labelbottom=False)

                    divider = make_axes_locatable(cur)
                    ax_cfl = divider.append_axes(
                        "bottom", size="35%", pad=0.1, sharex=cur
                    )
                    if 'cfl' in df_txt.columns:
                        ax_cfl.scatter(
                            x=df_txt['iters'], y=df_txt['cfl'],
                            color='black', s=1.5,
                        )
                    ax_cfl.set(ylabel="CFL", xlabel="Iterations")
                    ax_cfl.set_yscale('log')
                    ax_cfl.grid(
                        which='both', linestyle='-', linewidth=0.5, alpha=0.3
                    )

                for j in range(i + 1, len(ax_flat)):
                    ax_flat[j].axis('off')

                plt.tight_layout()
                plt.show()

    # =========================================================================
    # SETS
    # =========================================================================
    class SETS:

        class CODASets():

            def __init__(self, db: 'FRODO'):
                self.db = db

            def create_jset(
                self,
                stage: str,
                id_group: str,
                sol: Union[list, tuple, int, str] = 'all',
                idx_flcc: Union[list, tuple, str] = 'all',
                save_path: Union[str, None] = None,
                verbose: bool = False,
            ):
                key_group = f'CADGroup_{id_group}'
                data_dict = self.db.data_dict[key_group]

                if idx_flcc == 'all':
                    idx_flcc = list(range(data_dict['FlCc'].shape[0]))

                tensor_ptos = data_dict['Coord']
                tensor_flcc = data_dict['FlCc'][idx_flcc]

                tensors_aux = (
                    [
                        data_dict['Aux'][n][:, idx_flcc]
                        for n in data_dict['Aux']
                    ]
                    if 'Aux' in data_dict else None
                )

                sol_num = (
                    list(range(len(data_dict['Vars'][stage].keys())))
                    if sol == 'all'
                    else ([sol] if isinstance(sol, int) else sol)
                )

                tensors_out = []
                for i, (name, arr) in enumerate(
                    data_dict['Vars'][stage].items()
                ):
                    if i in sol_num:
                        print(name)
                        if arr.ndim == 2:
                            tensors_out.append(arr[:, idx_flcc])
                        elif arr.ndim == 3:
                            print(
                                f"WARNING: Variable '{name}' is a vector. "
                                "Only scalars are fully supported here."
                            )

                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, verbose=verbose,
                )

                if save_path:
                    if save_path.endswith('.h5'):
                        with h5py.File(save_path, "w") as hf:
                            hf.create_dataset("tensor", data=result['tensor'].numpy())
                            hf.create_dataset("scaled", data=result['scaled'].numpy())
                            hf.create_dataset("mins",   data=result['mins'].numpy())
                            hf.create_dataset("maxs",   data=result['maxs'].numpy())
                    elif save_path.endswith('.pt'):
                        torch.save(obj=result, f=save_path)
                    elif save_path.endswith('.npy'):
                        np.save(file=save_path, arr=result, allow_pickle=True)
                    else:
                        raise NameError(
                            "save_path extension not supported. "
                            "Choose .pt, .npy or .h5."
                        )
                    if verbose:
                        print(f"Jset saved in {save_path}\n")

                self.db.jset = result
                columns = (
                    ['x', 'z'] if data_dict['Coord'].shape[1] == 2
                    else ['x', 'y', 'z'] if data_dict['Coord'].shape[1] == 3
                    else (_ for _ in ()).throw(
                        ValueError("Unexpected coordinate shape.")
                    )
                )
                columns = list(columns)
                columns.extend(self.db.metadata['design_vars'])
                if 'Aux' in data_dict:
                    columns.extend(list(data_dict['Aux'].keys()))
                columns.extend([
                    n for i, n in enumerate(data_dict['Vars'][stage].keys())
                    if i in sol_num
                ])
                self.db.df_data = pd.DataFrame(
                    data=result['tensor'].numpy(), columns=columns
                )
                if verbose:
                    print(f'Loaded: {columns}')
                    print("\nJset loaded in db.jset\n")
                    print("\nDataframe loaded in db.df_data\n")

            def create_pylom_mesh(
                self, id_groups: Union[int, tuple]
            ):
                """
                Create pyLOM Mesh objects from stored CADGroup data.

                Args:
                    id_groups: IDs or tuple-combinations of IDs.

                Returns:
                    list[pyLOM.Mesh]
                """
                mesh_list = []
                for id in id_groups:
                    if isinstance(id, tuple):
                        key_suffix = "_".join(map(str, id))
                    elif isinstance(id, int):
                        key_suffix = str(id)
                    else:
                        raise TypeError("Unknown ID format.")

                    key = f"CADGroup_{key_suffix}"
                    xyz   = self.db.data_dict[key]["Coord"]
                    conec = self.db.data_dict[key]["Conec"]

                    eltype = np.array(
                        self.db.data_dict[key]["eltype"][0, :], copy=True
                    )
                    eltype[eltype == 5] = 2
                    eltype[eltype == 9] = 3

                    ptable = SMEAGOL.PartitionTable.new(
                        1, conec[0, :, :].shape[0], xyz.shape[0]
                    )
                    mesh = SMEAGOL.Mesh(
                        'UNSTRUCT', xyz, conec, eltype,
                        self.db.data_dict[key]["cellOrder"][0, :],
                        self.db.data_dict[key]["pointOrder"][0, :],
                        ptable,
                    )
                    mesh_list.append(mesh)
                    print(mesh)
                return mesh_list

            def create_NN_pylom(
                self,
                id_groups: Union[int, tuple, list],
                stage: int,
                idx_to_print: Union[int, list, str] = 'all',
                external_vars: Union[dict, None] = None,
                save_path: Union[bool, str] = False,
                nan_policy: Literal['fill', 'raise'] = 'fill',
                nan_fill_value: float = 0.0,
            ):
                """
                Create pyLOM Dataset objects combining mesh geometry and
                simulation field variables.

                Args:
                    id_groups: CADGroup IDs to export.
                    stage (int): Stage whose variables are exported.
                    idx_to_print: Case indices to include. Default 'all'.
                    external_vars (dict or None): Custom parametric variable
                        dict. If None, design variables from db.metadata are
                        used.
                    save_path (bool or str): If a directory path, saves each
                        dataset as <path>/<key>_stage_<stage>.h5.
                    nan_policy ('fill' or 'raise'): Action when NaN values are
                        found. 'fill' replaces with nan_fill_value; 'raise'
                        raises ValueError.
                    nan_fill_value (float): Replacement value for NaN when
                        nan_policy='fill'. Default 0.0.

                Returns:
                    list[pyLOM.Dataset]
                """
                if not self.db.data_dict:
                    raise AttributeError(
                        "data_dict is empty. Run extract_inputs() and "
                        "extract_outputs() first."
                    )
                if nan_policy not in ('fill', 'raise'):
                    raise ValueError("nan_policy must be 'fill' or 'raise'.")

                if isinstance(id_groups, int):
                    id_groups = [id_groups]

                d_list = []
                for id in id_groups:
                    key = f"CADGroup_{id}"
                    xyz   = self.db.data_dict[key]["Coord"]
                    conec = self.db.data_dict[key]["Conec"]
                    npoints = xyz.shape[0]

                    ptable = SMEAGOL.PartitionTable.new(
                        1, conec.shape[0], npoints
                    )
                    fields = [
                        n for n in
                        self.db.data_dict[key]['Vars'][str(stage)].keys()
                        if n not in ('GlobalNumber', 'CADGroupID')
                    ]

                    if idx_to_print == 'all':
                        idx_to_print = list(
                            range(self.db.data_dict[key]["FlCc"].shape[0])
                        )
                    elif isinstance(idx_to_print, int):
                        idx_to_print = [idx_to_print]

                    max_cases = self.db.data_dict[key]["FlCc"].shape[0]
                    if any(i >= max_cases or i < 0 for i in idx_to_print):
                        raise IndexError(
                            "idx_to_print contains out-of-range indices."
                        )
                    case_idx = np.asarray(idx_to_print, dtype=np.int64)

                    eltype = self.db.data_dict[key]["eltype"].copy()
                    eltype[eltype == 5] = 2
                    eltype[eltype == 9] = 3
                    cell_order = self.db.data_dict[key]["cellOrder"]

                    if external_vars is None:
                        param_dict = {
                            p: {
                                'idim': 0,
                                'value': self.db.df_state[p].iloc[
                                    idx_to_print
                                ].values,
                            }
                            for p in self.db.metadata['design_vars']
                        }
                    else:
                        param_dict = external_vars
                        for p, content in param_dict.items():
                            val = content['value']
                            idx = np.asarray(idx_to_print)
                            if idx.size > 0 and idx.max() >= len(val):
                                raise IndexError(
                                    f"idx_to_print out of range for param '{p}'."
                                )
                            content['value'] = val[idx]

                    def _sanitize(name, value):
                        if not np.issubdtype(value.dtype, np.floating):
                            return np.ascontiguousarray(value)
                        nan_mask = np.isnan(value)
                        if not np.any(nan_mask):
                            return np.ascontiguousarray(value)
                        n_nan = int(np.count_nonzero(nan_mask))
                        if nan_policy == 'raise':
                            raise ValueError(
                                f"Variable '{name}' contains {n_nan} NaN values."
                            )
                        warnings.warn(
                            f"Variable '{name}': replacing {n_nan} NaN values "
                            f"with {nan_fill_value}.",
                            RuntimeWarning,
                        )
                        value = value.copy()
                        value[nan_mask] = nan_fill_value
                        return np.ascontiguousarray(value)

                    field_dict = {}
                    for f in fields:
                        va = np.asarray(
                            self.db.data_dict[key]['Vars'][str(stage)][f]
                        )
                        if va.ndim == 2:
                            value = (
                                va[:, case_idx]
                                if va.shape[0] == npoints
                                else va[case_idx, :].T
                            )
                            field_dict[f] = {
                                'ndim': 1,
                                'value': _sanitize(f, value),
                            }
                        elif va.ndim == 3:
                            if va.shape[1] != npoints:
                                raise ValueError(
                                    f"Vector variable '{f}': axis 1 "
                                    f"({va.shape[1]}) != npoints ({npoints})."
                                )
                            value = va[:, :, case_idx]
                            ndim_v, np_, nc = value.shape
                            interleaved = (
                                value.transpose(1, 0, 2)
                                .reshape(np_ * ndim_v, nc, order='C')
                            )
                            field_dict[f] = {
                                'ndim': ndim_v,
                                'value': _sanitize(f, interleaved),
                            }
                        else:
                            raise ValueError(
                                f"Variable '{f}' has unsupported shape "
                                f"{va.shape}."
                            )

                    d = SMEAGOL.Dataset(
                        xyz=xyz, ptable=ptable, order=cell_order,
                        point=True, vars=param_dict, **field_dict,
                    )
                    print('DONE', flush=True)
                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                        out = os.path.join(
                            save_path, f"{key}_stage_{stage}.h5"
                        )
                        d.save(out)
                        print(f"Dataset saved to {out}")
                    d_list.append(d)

                return d_list

            def add_to_data_dict(
                self, arr: np.ndarray, id_group: str, array_name: str
            ):
                data = self.db.data_dict.copy()
                group_key = f'CADGroup_{id_group}'
                data[group_key].setdefault('Aux', {})
                data[group_key]['Aux'][array_name] = arr
                self.db.data_dict = data

            def change_order_coord(
                self,
                id_group: str,
                new_order: Union[str, list, tuple],
                new_nodes_order: Union[None, list, tuple] = None,
            ):
                data = copy.deepcopy(self.db.data_dict)
                key_group = 'CADGroup_' + id_group
                coord = data[key_group]['Coord']
                nodecoord = data[key_group]['NodeCoord']

                if isinstance(new_order, str):
                    sort_fn_map = {
                        'lexsort':     SAM.Weapons.sort_lexsort,
                        'centroid':    SAM.Weapons.sort_by_centroid,
                        'kdtree':      SAM.Weapons.sort_closed_curve_by_kdtree,
                        'convex_hull': SAM.Weapons.sort_points_by_hull_projection,
                    }
                    if new_order not in sort_fn_map:
                        raise ValueError(
                            f"new_order '{new_order}' not supported. "
                            f"Options: {list(sort_fn_map.keys())}."
                        )
                    fn = sort_fn_map[new_order]
                    _, idx_new = fn(coord)
                    _, idx_nodes_new = fn(nodecoord)
                elif isinstance(new_order, (list, tuple)):
                    if new_nodes_order is None:
                        raise ValueError(
                            "new_nodes_order must be provided when new_order "
                            "is a list or tuple."
                        )
                    idx_new = new_order
                    idx_nodes_new = new_nodes_order
                else:
                    raise TypeError(
                        "new_order must be a str or list/tuple of indices."
                    )

                for k in ['Coord', 'Conec', 'eltype', 'cellOrder']:
                    self.db.data_dict[key_group][k] = data[key_group][k][idx_new]
                for k in ['NodeCoord', 'pointOrder']:
                    self.db.data_dict[key_group][k] = (
                        data[key_group][k][idx_nodes_new]
                    )
                for k, idx in zip(
                    ['idx_sort', 'idx_sort_nodes'], [idx_new, idx_nodes_new]
                ):
                    for s in range(data[key_group][k].shape[0]):
                        for c in range(data[key_group][k].shape[1]):
                            self.db.data_dict[key_group][k][s, c, :] = (
                                data[key_group][k][s, c, idx]
                            )
                for s in data[key_group]['Vars']:
                    for var in data[key_group]['Vars'][s]:
                        self.db.data_dict[key_group]['Vars'][s][var] = (
                            data[key_group]['Vars'][s][var][idx_new]
                        )

            def save_to_npy(
                self,
                stage: Union[list, tuple, int],
                id_group: str,
                filepath: str,
                case_idx: Union[int, list, tuple, str] = 'all',
                ignore_vars: Union[list, tuple, None] = None,
                verbose: bool = False,
            ):
                """
                Save a stage and group subset to a .npy file.

                Args:
                    stage (int): Stage number to export.
                    id_group (str): CADGroup identifier (e.g. '3').
                    filepath (str): Output path (.npy added if absent).
                    case_idx: Cases to include. Default 'all'.
                    ignore_vars (list or None): Variables to exclude.
                    verbose (bool): Print progress.
                """
                if not isinstance(id_group, str):
                    raise ValueError("id_group must be a string.")

                group_key = f'CADGroup_{id_group}'
                gd = self.db.data_dict[group_key]
                stage_vars = gd["Vars"][str(stage)]
                aux_dict = gd.get('Aux', {})

                if isinstance(case_idx, str):
                    if case_idx != 'all':
                        raise ValueError("case_idx as string only accepts 'all'.")
                    case_idx = list(range(gd['FlCc'].shape[0]))
                    all_cases = True
                elif isinstance(case_idx, int):
                    case_idx = [case_idx]
                    all_cases = False
                elif isinstance(case_idx, (list, tuple)):
                    all_cases = False
                else:
                    raise ValueError(
                        "case_idx must be a tuple, list, int or 'all'."
                    )

                ncases = len(case_idx)
                npoints = gd["Coord"].shape[0]
                idx_sort_complete = gd["idx_sort"]

                idx_sort  = np.zeros((ncases, npoints), dtype=np.int32)
                eltype    = np.zeros((ncases, npoints), dtype=np.int32)
                cellOrder = np.zeros((ncases, npoints), dtype=np.int32)
                for ci, c in enumerate(case_idx):
                    idx_sort[ci]  = idx_sort_complete[stage, c, :]
                    eltype[ci]    = gd["eltype"][idx_sort_complete[stage, c, :]]
                    cellOrder[ci] = gd["cellOrder"][idx_sort_complete[stage, c, :]]

                out = {
                    'Coord': gd["Coord"], 'FlCc': gd['FlCc'],
                    'idx_sort': idx_sort, 'Conec': gd["Conec"],
                    'eltype': eltype, 'cellOrder': cellOrder,
                }
                for var_name, var_data in stage_vars.items():
                    if ignore_vars and var_name in ignore_vars:
                        continue
                    if var_data.ndim == 2 and \
                            var_data.shape[0] == gd["Coord"].shape[0]:
                        out[var_name] = np.transpose(var_data[:, case_idx])
                    elif var_data.ndim == 3 and \
                            var_data.shape[1] == gd["Coord"].shape[0]:
                        out[var_name] = var_data[:, :, case_idx]

                out.update(aux_dict)

                if not filepath.endswith('.npy'):
                    filepath += '.npy'
                np.save(filepath, out, allow_pickle=True)

                if verbose:
                    print(
                        f"\nSaved {'all cases' if all_cases else case_idx} "
                        f"to {filepath}"
                    )

            def save_to_h5(
                self,
                filepath: str,
                overwrite: bool = True,
                verbose: bool = True,
            ):
                """
                Save the full data_dict to a compressed HDF5 file.

                Args:
                    filepath (str): Destination path (.h5 added if absent).
                    overwrite (bool): Overwrite existing file. Default True.
                    verbose (bool): Print progress. Default True.
                """
                if os.path.exists(filepath):
                    if overwrite:
                        os.remove(filepath)
                    else:
                        raise FileExistsError(f"{filepath} already exists.")

                if not filepath.endswith('.h5'):
                    filepath += '.h5'

                def _compressed(group, name, data):
                    if isinstance(data, np.ndarray) and data.ndim in (2, 3):
                        chunks = (1, min(100000, data.shape[1])) + (
                            (data.shape[2],) if data.ndim == 3 else ()
                        )
                    else:
                        chunks = True
                    group.create_dataset(
                        name, data=data,
                        compression="gzip", compression_opts=4,
                        shuffle=True, chunks=chunks,
                    )

                with h5py.File(filepath, "w", libver="latest") as f:
                    for group_key, gd in self.db.data_dict.items():
                        if verbose:
                            print(f"\nSaving group {group_key}")
                        grp = f.create_group(group_key)

                        grp.create_dataset("Coord",     data=gd["Coord"])
                        grp.create_dataset("NodeCoord", data=gd["NodeCoord"])
                        grp.create_dataset("FlCc",      data=gd["FlCc"])
                        for attr in ("Conec", "eltype", "cellOrder", "pointOrder"):
                            _compressed(grp, attr, gd[attr])

                        vg = grp.create_group("Vars")
                        for stage, stage_vars in gd["Vars"].items():
                            if verbose:
                                print(f"  Stage {stage}")
                            sg = vg.create_group(str(stage))
                            scalars_g   = sg.create_group("Scalars")
                            vectors_g   = sg.create_group("Vectors")
                            gradients_g = sg.create_group("Gradients")

                            for vname, vdata in stage_vars.items():
                                if vdata.ndim == 2:
                                    to_save = vdata.T
                                elif vdata.ndim == 3:
                                    to_save = np.transpose(vdata, (2, 1, 0))
                                else:
                                    raise ValueError(
                                        f"Unexpected ndim {vdata.ndim} in '{vname}'."
                                    )
                                target = (
                                    gradients_g if "Grad" in vname
                                    else vectors_g if to_save.ndim == 3
                                    else scalars_g
                                )
                                _compressed(target, vname, to_save)
                                if verbose:
                                    print(f"    {vname}: {to_save.shape}")

                    f.swmr_mode = True

                if verbose:
                    print(
                        "\nFile saved with compression, chunking and SWMR."
                    )

            def crop_bounding_box(
                self,
                id_group: str,
                bbox: Union[list, None] = None,
                radius_center: Union[tuple, None] = None,
                new_group_suffix: str = "_crop",
            ):
                """
                Create a new CADGroup containing only cells within a bounding
                box or spherical region.

                Args:
                    id_group (str): Source CADGroup identifier.
                    bbox (list or None): [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
                    radius_center (tuple or None): (radius, center_array).
                    new_group_suffix (str): Suffix for the new group key.
                """
                key_old = f'CADGroup_{id_group}'
                key_new = f'{key_old}{new_group_suffix}'
                group = self.db.data_dict[key_old]
                coord = group['Coord']
                nodecoord = group['NodeCoord']

                if bbox is not None:
                    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
                    mask = (
                        (coord[:, 0] >= xmin) & (coord[:, 0] <= xmax) &
                        (coord[:, 1] >= ymin) & (coord[:, 1] <= ymax) &
                        (coord[:, 2] >= zmin) & (coord[:, 2] <= zmax)
                    )
                elif radius_center is not None:
                    radius, center = radius_center
                    mask = np.linalg.norm(coord - center, axis=1) <= radius
                else:
                    raise ValueError(
                        "Provide either bbox or radius_center."
                    )

                idx_cells = np.where(mask)[0]
                conec = group['Conec'][idx_cells]
                used_nodes = np.unique(conec)
                node_map = -np.ones(nodecoord.shape[0], dtype=np.int64)
                node_map[used_nodes] = np.arange(len(used_nodes))

                new_group = {
                    'Coord':     coord[idx_cells],
                    'NodeCoord': nodecoord[used_nodes],
                    'Conec':     node_map[conec],
                }
                for attr in ('eltype', 'cellOrder'):
                    if attr in group:
                        new_group[attr] = group[attr][idx_cells]
                if 'pointOrder' in group:
                    new_group['pointOrder'] = group['pointOrder'][used_nodes]
                if 'FlCc' in group:
                    new_group['FlCc'] = group['FlCc']
                if 'idx_sort' in group:
                    new_group['idx_sort'] = group['idx_sort'][:, :, idx_cells]
                if 'idx_sort_nodes' in group:
                    new_group['idx_sort_nodes'] = (
                        group['idx_sort_nodes'][:, :, used_nodes]
                    )

                new_group['Vars'] = {}
                for stage in group.get('Vars', {}):
                    new_group['Vars'][stage] = {}
                    for var, arr in group['Vars'][stage].items():
                        if arr.ndim == 2:
                            new_group['Vars'][stage][var] = arr[idx_cells]
                        elif arr.ndim == 3 and arr.shape[0] == 3:
                            new_group['Vars'][stage][var] = arr[:, idx_cells]
                        else:
                            new_group['Vars'][stage][var] = arr[idx_cells]

                self.db.data_dict[key_new] = new_group

            def interpolate_vol2surf(
                self,
                vol_group: str,
                surf_group: str,
                stage: str,
                vars: Union[str, list] = 'all',
                k: int = 4,
                eps: float = 1e-12,
            ):
                """
                Interpolate volume cell-centred fields onto surface centroids
                using inverse-distance weighting (IDW).

                Args:
                    vol_group (str): Source CADGroup identifier (volume).
                    surf_group (str): Target CADGroup identifier (surface).
                    stage (str): Vars stage key (e.g. '0').
                    vars ('all' or list[str]): Variables to interpolate.
                        'GlobalNumber' and 'CADGroupID' are always excluded.
                    k (int): IDW nearest neighbours. Default 4.
                    eps (float): Stability constant. Default 1e-12.
                """
                from scipy.spatial import cKDTree

                vol  = self.db.data_dict[f'CADGroup_{vol_group}']
                surf = self.db.data_dict[f'CADGroup_{surf_group}']

                tree = cKDTree(vol['Coord'])
                dist, idx = tree.query(surf['Coord'], k=k)
                w = 1.0 / (dist + eps)
                w /= w.sum(axis=1, keepdims=True)

                surf.setdefault('Vars', {}).setdefault(stage, {})

                if vars == 'all':
                    vars = [
                        v for v in vol['Vars'][stage]
                        if v not in ('GlobalNumber', 'CADGroupID')
                    ]

                for var in vars:
                    arr = vol['Vars'][stage][var]
                    if arr.ndim == 2:
                        surf['Vars'][stage][var + '_interp'] = np.einsum(
                            'ij,ijk->ik', w, arr[idx]
                        )
                    elif arr.ndim == 3 and arr.shape[0] == 3:
                        surf['Vars'][stage][var + '_interp'] = np.einsum(
                            'ij,lijk->lik', w, arr[:, idx, :]
                        )
                    else:
                        raise ValueError(
                            f"Unsupported shape for variable '{var}': "
                            f"{arr.shape}."
                        )

            def interpolate_msh2msh(
                self,
                id_group_src: str,
                new_group_id: str,
                new_mesh: dict,
                vars: Union[str, list] = 'all',
                method: str = 'idw',
                k: int = 4,
            ):
                """
                Interpolate all variables from a source CADGroup onto a
                different target mesh, storing the result as a new CADGroup.

                Args:
                    id_group_src (str): Source CADGroup identifier.
                    new_group_id (str): Identifier for the interpolated group.
                    new_mesh (dict): Target mesh dict (must contain 'Coord').
                    vars ('all' or list[str]): Variables to interpolate.
                    method (str): Interpolation method.
                        Supported: 'idw', 'griddata', 'pyvista'. Default 'idw'.
                    k (int): IDW neighbours. Default 4.
                """
                src = self.db.data_dict[f'CADGroup_{id_group_src}']
                coord_src = src["Coord"]
                vars_src  = src["Vars"]

                if "Coord" not in new_mesh:
                    raise ValueError("new_mesh must contain 'Coord'.")

                coord_dst = new_mesh["Coord"]
                conec_src = src.get("Conec")
                conec_dst = new_mesh.get("Conec")

                if np.shares_memory(coord_src, coord_dst):
                    print("WARNING: source and destination meshes share memory.")

                new_key = f'CADGroup_{new_group_id}'
                if new_key in self.db.data_dict:
                    print(f"WARNING: overwriting {new_key}.")
                self.db.data_dict[new_key] = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in new_mesh.items()
                }
                if "FlCc" in src:
                    self.db.data_dict[new_key]["FlCc"] = src["FlCc"].copy()
                self.db.data_dict[new_key]["Vars"] = {}

                if method == "idw":
                    from scipy.spatial import cKDTree
                    tree = cKDTree(coord_src)
                else:
                    tree = None

                for stage, stage_data in vars_src.items():
                    self.db.data_dict[new_key]["Vars"][stage] = {}

                    selected = (
                        [
                            v for v in stage_data
                            if v not in ("GlobalNumber", "CADGroupID")
                        ]
                        if vars == 'all' else vars
                    )

                    var_list, shapes, valid_names = [], [], []
                    for vname in selected:
                        if vname not in stage_data:
                            continue
                        v = stage_data[vname]
                        if v.ndim == 1:
                            v = v[:, None]
                        if v.ndim == 3:
                            nd, nc = v.shape[0], v.shape[2]
                            var_list.append(
                                v.transpose(1, 0, 2).reshape(v.shape[1], nd * nc)
                            )
                            shapes.append(('vec', nd, nc))
                        else:
                            var_list.append(v)
                            shapes.append(v.shape[1])
                        valid_names.append(vname)

                    if not var_list:
                        continue

                    src_stack = np.hstack(var_list)
                    if method == "idw":
                        dst_stack = SAM.Weapons._interpolate_idw_tree(
                            tree, coord_src, src_stack, coord_dst, k=k
                        )
                    elif method == "griddata":
                        dst_stack = SAM.Weapons._interpolate_griddata(
                            coord_src, src_stack, coord_dst
                        )
                    elif method == "pyvista":
                        if conec_src is None or conec_dst is None:
                            raise ValueError(
                                "PyVista requires 'Conec' in both meshes."
                            )
                        dst_stack = SAM.Weapons._interpolate_pyvista(
                            coord_src, conec_src, src_stack,
                            coord_dst, conec_dst,
                        )
                    else:
                        raise ValueError(
                            f"Unknown method '{method}'. "
                            "Supported: 'idw', 'griddata', 'pyvista'."
                        )

                    col = 0
                    for vname, shape in zip(valid_names, shapes):
                        if isinstance(shape, tuple):
                            _, nd, nc = shape
                            chunk = dst_stack[:, col:col + nd * nc]
                            self.db.data_dict[new_key]["Vars"][stage][vname] = (
                                chunk.reshape(chunk.shape[0], nd, nc)
                                .transpose(1, 0, 2)
                            )
                            col += nd * nc
                        else:
                            self.db.data_dict[new_key]["Vars"][stage][vname] = (
                                dst_stack[:, col:col + shape]
                            )
                            col += shape

        # ─────────────────────────────────────────────────────────────────────
        class NUMPYFILESets():

            def __init__(self, db: 'FRODO'):
                self.db = db

            def add_aux(
                self,
                array_name: str,
                array: np.ndarray,
                notes: str = None,
            ):
                """
                Store an auxiliary array in data_dict['aux'] and record
                metadata.

                Args:
                    array_name (str): Key for data_dict['aux'].
                    array (np.ndarray): Array to store.
                    notes (str or None): Human-readable description.
                """
                db = self.db
                db.data_dict.setdefault("aux", {})
                db.sim_metadata.setdefault('info_aux', []).append(notes)
                db.sim_metadata.setdefault('keys_aux', {})[array_name] = notes
                db.data_dict['aux'][array_name] = array

            def create_jset(
                self,
                sol: Union[list, tuple, int, str] = 'all',
                save_path: Union[bool, str] = False,
                verbose: bool = False,
            ):
                """
                Assemble inputs, aux and outputs into a joint ML tensor via
                SAM.Gardener.

                Args:
                    sol: Solution indices. Default 'all'.
                    save_path (bool or str): Save path (.h5, .pt or .npy).
                    verbose (bool): Print progress.

                Side-effects: Sets db.jset and db.df_data.
                """
                tensor_ptos = self.db.data_dict['inputs']['ptos']
                tensor_flcc = np.column_stack([
                    self.db.data_dict['inputs'][n]
                    for n in self.db.data_dict['inputs']
                    if n != 'ptos'
                ])
                tensors_aux = list(self.db.data_dict['aux'].values())
                tensors_out = list(self.db.data_dict['outputs'].values())

                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, verbose=verbose,
                )

                if save_path:
                    if save_path.endswith('.h5'):
                        with h5py.File(save_path, "w") as hf:
                            hf.create_dataset("tensor", data=result['tensor'].numpy())
                            hf.create_dataset("scaled", data=result['scaled'].numpy())
                            hf.create_dataset("mins",   data=result['mins'].numpy())
                            hf.create_dataset("maxs",   data=result['maxs'].numpy())
                    elif save_path.endswith('.pt'):
                        torch.save(obj=result, f=save_path)
                    elif save_path.endswith('.npy'):
                        np.save(file=save_path, arr=result, allow_pickle=True)
                    else:
                        raise NameError(
                            "save_path extension not supported. "
                            "Choose .pt, .npy or .h5."
                        )
                    if verbose:
                        print(f"Jset saved in {save_path}\n")

                self.db.jset = result
                columns = []
                for k in self.db.data_dict['inputs']:
                    arr = self.db.data_dict['inputs'][k]
                    if arr.shape[1] == 2:
                        columns.extend(['x', 'z'])
                    elif arr.shape[1] == 3:
                        columns.extend(['x', 'y', 'z'])
                    elif arr.shape[1] == 1:
                        columns.append(k)
                for section in ('aux', 'outputs'):
                    columns.extend(self.db.data_dict[section].keys())

                self.db.df_data = pd.DataFrame(
                    data=result['tensor'].numpy(), columns=columns
                )
                if verbose:
                    print("\nJset loaded in db.jset\n")
                    print("\nDataframe loaded in db.df_data\n")

        # ─────────────────────────────────────────────────────────────────────
        class PYLOMSets():
            """
            High-level operations on a FRODO database loaded from a pyLOM
            dataset.

            Access via ``db.sets`` after constructing FRODO with
            ``format='PYLOM'``.

            Quick-reference
            ---------------
            db.sets.get_xyz()                          → (npoints, ndim)
            db.sets.get_variable('time')               → (ncases,)
            db.sets.get_field('cp')                    → (npoints, ncases)
            db.sets.get_field('vel', idim=0)           → (npoints, ncases)
            db.sets.add_aux('mask', arr, 'bool mask')  → store auxiliary array
            db.sets.to_pylom_dataset()                 → reconstruct pyLOM Dataset
            db.sets.create_jset()                      → SAM.Gardener ML tensor
            db.sets.summary()                          → print data_dict overview
            """

            def __init__(self, db: 'FRODO'):
                self.db = db

            def get_xyz(self) -> np.ndarray:
                """Return mesh node coordinates, shape (npoints, ndim)."""
                inputs = self.db.data_dict.get("inputs", {})
                for cand in ("ptos", "xyz"):
                    if cand in inputs:
                        return inputs[cand]
                raise KeyError(
                    "Coordinates not found. Run extract_inputs with 'ptos': 'xyz'."
                )

            def get_variable(self, name: str) -> np.ndarray:
                """Return a parametric variable as a 1-D array (ncases,)."""
                inputs = self.db.data_dict.get("inputs", {})
                if name not in inputs:
                    raise KeyError(
                        f"Variable '{name}' not found. "
                        f"Available: {list(inputs.keys())}"
                    )
                return np.asarray(inputs[name]).ravel()

            def get_field(
                self,
                name: str,
                idim: int = None,
                section: Union[int, slice, None] = None,
            ) -> np.ndarray:
                """
                Return an output field with optional component/case selection.

                Args:
                    name (str): Alias from extract_outputs.
                    idim (int or None): Component index for vector fields.
                    section: Case subset (int or slice).

                Returns:
                    np.ndarray
                """
                outputs = self.db.data_dict.get("outputs", {})
                if name not in outputs:
                    raise KeyError(
                        f"Field '{name}' not found. "
                        f"Available: {list(outputs.keys())}"
                    )
                arr = outputs[name]
                if idim is not None:
                    if arr.ndim < 3:
                        raise ValueError(
                            f"Field '{name}' is scalar; idim requires a vector field."
                        )
                    arr = arr[idim]
                if section is not None:
                    arr = arr[..., section]
                return arr

            def field_names(self) -> list:
                """Return available output field aliases."""
                return list(self.db.data_dict.get("outputs", {}).keys())

            def variable_names(self) -> list:
                """Return input variable aliases (excluding 'ptos'/'xyz')."""
                return [
                    k for k in self.db.data_dict.get("inputs", {})
                    if k not in ("ptos", "xyz")
                ]

            def add_aux(
                self,
                array_name: str,
                array: np.ndarray,
                notes: str = None,
            ):
                """
                Store an auxiliary array in data_dict['aux'].

                Args:
                    array_name (str): Key for data_dict['aux'].
                    array (np.ndarray): Array to store.
                    notes (str or None): Description for sim_metadata.
                """
                db = self.db
                db.data_dict.setdefault("aux", {})
                db.sim_metadata.setdefault("info_aux", []).append(notes)
                db.sim_metadata.setdefault("keys_aux", {})[array_name] = notes
                db.data_dict["aux"][array_name] = array

            def to_pylom_dataset(self) -> 'SMEAGOL.Dataset':
                """
                Reconstruct a pyLOM Dataset from the current data_dict.

                Useful for passing data back to pyLOM reduction methods
                (POD, DMD, …) after processing inside FRODO.

                Returns:
                    SMEAGOL.Dataset
                """
                from pyLOM.dataset import Dataset
                from pyLOM.partition_table import PartitionTable

                inputs  = self.db.data_dict.get("inputs", {})
                outputs = self.db.data_dict.get("outputs", {})

                xyz = self.get_xyz()
                npoints = xyz.shape[0]
                ptable = PartitionTable.new(
                    nparts=1, nelems=0, npoints=npoints, has_master=False
                )

                var_dict = {}
                for i, (alias, arr) in enumerate(inputs.items()):
                    if alias in ("ptos", "xyz"):
                        continue
                    var_dict[alias] = {
                        "idim": i, "value": np.asarray(arr).ravel()
                    }

                field_dict = {}
                for alias, arr in outputs.items():
                    arr = np.asarray(arr)
                    if arr.ndim == 1:
                        field_dict[alias] = {
                            "ndim": 1, "value": arr.reshape(npoints)
                        }
                    elif arr.ndim == 2:
                        field_dict[alias] = {
                            "ndim": 1, "value": arr.reshape(npoints, arr.shape[1])
                        }
                    elif arr.ndim == 3:
                        nd, np_, nc = arr.shape
                        field_dict[alias] = {
                            "ndim": nd,
                            "value": arr.transpose(1, 0, 2).reshape(
                                np_ * nd, nc, order='C'
                            ),
                        }
                    else:
                        warnings.warn(
                            f"Field '{alias}' has unexpected shape {arr.shape}."
                        )
                        field_dict[alias] = {"ndim": 1, "value": arr}

                return Dataset(
                    xyz=xyz, ptable=ptable, vars=var_dict,
                    order=np.arange(npoints, dtype=np.int32),
                    point=True, **field_dict,
                )

            def create_jset(
                self,
                sol: Union[list, int, str] = "all",
                save_path: Union[bool, str] = False,
                idx_flcc: Union[list, tuple, str] = 'all',
                ref: Union[dict, None] = None,
                verbose: bool = False,
            ):
                """
                Assemble inputs, aux and outputs into a joint ML tensor via
                SAM.Gardener.

                Args:
                    sol: Case indices to include. Default 'all'.
                    save_path (bool or str): Optional save path (.h5 or .pt).
                    idx_flcc: FlCc row subset. Default 'all'.
                    ref (dict or None): Reference mins/maxs for normalisation.
                    verbose (bool): Print progress.

                Side-effects: Sets db.dict_tensors and db.df_data.
                """
                db = self.db
                dd = db.data_dict

                if not dd.get("inputs"):
                    raise ValueError(
                        "data_dict['inputs'] is empty. "
                        "Call extract_inputs() first."
                    )
                if not dd.get("outputs"):
                    raise ValueError(
                        "data_dict['outputs'] is empty. "
                        "Call extract_outputs() first."
                    )

                input_keys = list(dd["inputs"].keys())
                ptos_key = next(
                    (k for k in input_keys if k in ("ptos", "xyz")), None
                )
                if ptos_key is None:
                    raise ValueError(
                        "No coordinate array ('ptos' or 'xyz') in "
                        "data_dict['inputs']."
                    )

                tensor_ptos = torch.from_numpy(
                    np.asarray(dd["inputs"][ptos_key])
                )
                flcc_arrays = [
                    torch.from_numpy(
                        np.asarray(dd["inputs"][k]).reshape(-1, 1)
                        if np.asarray(dd["inputs"][k]).ndim == 1
                        else np.asarray(dd["inputs"][k])
                    )
                    for k in input_keys if k != ptos_key
                ]
                tensor_flcc = (
                    torch.cat(flcc_arrays, dim=1)
                    if flcc_arrays else torch.empty(0)
                )
                tensors_aux = [
                    torch.from_numpy(np.asarray(v))
                    for v in dd.get("aux", {}).values()
                ]
                tensors_out = [
                    torch.from_numpy(np.asarray(v))
                    for v in dd["outputs"].values()
                ]

                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, idx_flcc=idx_flcc, ref=ref, verbose=verbose,
                )

                if save_path:
                    if save_path.endswith(".h5"):
                        with h5py.File(save_path, "w") as hf:
                            hf.create_dataset("tensor", data=result["tensor"].numpy())
                            hf.create_dataset("scaled", data=result["scaled"].numpy())
                            hf.create_dataset("mins",   data=result["mins"].numpy())
                            hf.create_dataset("maxs",   data=result["maxs"].numpy())
                    elif save_path.endswith(".pt"):
                        torch.save(result, save_path)
                    else:
                        raise ValueError(
                            "save_path must end in '.h5' or '.pt'."
                        )
                    if verbose:
                        print(f"[PYLOMSets] jset saved → {save_path}")

                db.dict_tensors = result

                columns = []
                for k in input_keys:
                    arr = np.asarray(dd["inputs"][k])
                    if k == ptos_key:
                        ndim = arr.shape[1] if arr.ndim > 1 else 1
                        columns.extend(["x", "y", "z"][:ndim])
                    else:
                        nc = arr.shape[1] if arr.ndim > 1 else 1
                        columns.extend(
                            [k] if nc == 1 else [f"{k}_{i}" for i in range(nc)]
                        )
                for section in ("aux", "outputs"):
                    columns.extend(dd.get(section, {}).keys())

                try:
                    db.df_data = pd.DataFrame(
                        data=result["tensor"].numpy(), columns=columns
                    )
                except Exception:
                    db.df_data = pd.DataFrame(result["tensor"].numpy())

                if verbose:
                    print("[PYLOMSets] jset loaded in db.dict_tensors")
                    print("[PYLOMSets] dataframe loaded in db.df_data")

            def summary(self):
                """Print a compact overview of data_dict contents."""
                dd = self.db.data_dict
                print("── PYLOMSets summary ──────────────────────────────────")
                for section in ("inputs", "outputs", "aux"):
                    print(f"  [{section}]")
                    for k, v in dd.get(section, {}).items():
                        arr = np.asarray(v)
                        print(
                            f"    {k:25s}  shape={arr.shape}  dtype={arr.dtype}"
                        )
                print("───────────────────────────────────────────────────────")