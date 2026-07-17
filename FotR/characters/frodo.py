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
    ) -> 'FRODO':
        """
        Merge multiple FRODO datasets into a single unified one.

        Interpolates all source meshes onto a common reference mesh and
        concatenates FlCc arrays with optional deduplication.

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
            Required for format 'CODA'; ignored otherwise.

        Returns
        -------
        FRODO
        """
        # ── 0. Validate ──────────────────────────────────────────────────────
        if len(sources) < 2:
            raise ValueError("At least 2 sources are required.")

        dbs     = [db for db, _ in sources]
        formats = [db.format for db in dbs]
        if len(set(formats)) != 1:
            raise ValueError("All sources must share the same format.")
        format_ref = formats[0]

        if format_ref == 'CODA':
            if not get_df_metrics_attr:
                raise ValueError(
                    "get_df_metrics_attr must be provided for CODA format."
                )
            csvs_post = []
            csvs_state = []
            for db, gid in sources:
                df = db.residuals.get_df_metrics(**get_df_metrics_attr)
                df.columns = df.columns.str.lower()
                flcc   = db.data_dict[f'CADGroup_{gid}']['FlCc']
                dv_low = [v.lower() for v in db.metadata['design_vars']]
                csvs_post.append(
                    pd.DataFrame(flcc, columns=dv_low).merge(
                        df, on=dv_low, how='left'
                    )
                )
                df_state = db.df_state.copy()
                df_state.columns = df_state.columns.str.lower()
                csvs_state.append(df_state)
            for i, c in enumerate(csvs_post):
                c['dataset'] = f'dataset_{i}'
            df_post = pd.concat(csvs_post, ignore_index=True)
            df_state = pd.concat(csvs_state, ignore_index=True)
        else:
            if get_df_metrics_attr:
                raise ValueError(
                    f"get_df_metrics_attr is only supported for format 'CODA' "
                    f"(got '{format_ref}')."
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

        # ── 3. Cache ─────────────────────────────────────────────────────────
        cache_interp: dict = {}

        # ── 4. Homogenise meshes ─────────────────────────────────────────────
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

        # ── 5. Build new FRODO ───────────────────────────────────────────────
        db_new              = FRODO.__new__(FRODO)
        db_new.format       = format_ref
        db_new.root_dir     = root_dir
        db_new.name         = name.replace(" ", "_") if name is not None else "FRODO_Merged"
        db_new.sim_metadata = {}
        db_new.kwargs       = {}
        db_new.df_state     = df_state if format_ref == 'CODA' else None

        for d in [root_dir,
                  os.path.join(root_dir, 'metadata'),
                  os.path.join(root_dir, 'outputs')]:
            os.makedirs(d, exist_ok=True)

        for db in dbs:
            for mk, mv in db.sim_metadata.items():
                db_new.sim_metadata.setdefault(mk, mv)

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
        # db_new.df_cases = db_new.metadata['df_cases']
        
        meta_save = copy.deepcopy(db_new.metadata)
        if meta_save.get('df_cases') is not None:
            meta_save['df_cases'] = meta_save['df_cases'].to_dict(orient='list')
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

        # ── 6. FlCc with deduplication ───────────────────────────────────────
        flcc_list, case_splits = [], []
        for db, gid in processed:
            f = db.data_dict[f'CADGroup_{gid}']["FlCc"]
            flcc_list.append(f)
            case_splits.append(f.shape[0])

        flcc_all = np.vstack(flcc_list)
        assert len(df_post) == flcc_all.shape[0]

        seen: dict = {}
        keep: list = []
        offset = 0
        for i, n_cases in enumerate(case_splits):
            for j in range(n_cases):
                key_t = tuple(np.round(flcc_list[i][j], decimals=8))
                gidx  = offset + j
                if key_t not in seen:
                    seen[key_t] = (i, gidx)
                    keep.append(gidx)
                else:
                    prev_i, prev_idx = seen[key_t]
                    if prev_i == mesh_ref and i != mesh_ref:
                        if prev_idx in keep:
                            keep.remove(prev_idx)
                        keep.append(gidx)
                        seen[key_t] = (i, gidx)
            offset += n_cases

        keep     = sorted(keep)
        df_post  = df_post.iloc[keep].reset_index(drop=True)
        db_new.data_dict[ngk]["FlCc"] = flcc_all[keep]
        df_post.to_csv(
            os.path.join(root_dir, 'metadata', 'df_post.csv'), sep=','
        )

        # ── 7. Vars ──────────────────────────────────────────────────────────
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
                db_new.data_dict[ngk]["Vars"][stage][var] = (
                    var_concat[..., keep]
                )

        return db_new