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
    One tool to rule them all, one tool to find them, 

    FRODO is a lightweight yet powerful assistant for the management, 
    organization, and long-term archiving of simulation data, crafted 
    to handle multiple CFD cases with the care and precision of a hobbit 
    recording tales in the Red Book of Westmarch.

    Whether you've wandered through forests of folders or climbed mountains 
    of mesh files, FRODO helps you collect angle of attack, Mach numbers, 
    CADGroup-based outputs, and more — and forge them into one single 
    HDF5 database, ready for analysis, plotting, or machine learning.

    Inspired by the resilience of Frodo Bolsón, this tool carries the burden 
    of structuring your simulation results so you don't have to.

    Use it wisely. Even the smallest file can change the fate of a dataset.
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Atajo a Eärendil's Light."""
        return cls.light.help(name)

    def __str__(self):
        return f"{self.name}; root_dir: {self.root_dir}; format: {self.format}"
    
    def __getattr__(self, name):
        """
        Delegación dinámica: si FRODO no tiene el atributo, buscamos en
        self.sets, self.reader y self.residuals (en ese orden).
        Esto permite hacer db.add_aux(...) cuando add_aux está implementado
        en db.sets (p.ej. en NRL7301Sets), sin añadir add_aux a FRODO.
        """
        # Usar object.__getattribute__ para evitar recursión en getattr
        for sub in ('sets', 'reader', 'residuals'):
            try:
                obj = object.__getattribute__(self, sub)
            except AttributeError:
                obj = None
            if obj is not None and hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __init__(
        self,
        root_dir: str,
        format: Literal['CODA', 'Airfoil', 'PALMO', 'NRL7301', 'NUMPYFILE', 'FLUENT', 'PYLOM'],
        initial_parse:bool = True,
        **kwargs
        ):
        
        self.format = format
        self.root_dir = os.path.abspath(root_dir)
        self.sim_metadata = {}
        self.data_dict = {}
        self.kwargs = kwargs
        self.update_df_state = kwargs.pop("update_df_state", False)
        self.name = kwargs.pop("name", 'FRODO Database')
        # -------- READER FACTORY ----------
        reader_map = {
            "CODA": self.READERS.CODAReader,
            "Airfoil": self.READERS.AIRFOILReader,
            "NUMPYFILE": self.READERS.NUMPYFILEReader,
        }

        if format not in reader_map:
            raise ValueError(f"Format {format} not supported.")

        self.reader = reader_map[format](root_dir=self.root_dir, **kwargs)

        # -------- SETS FACTORY ------------
        sets_map = {
            "CODA": self.SETS.CODASets,
            "Airfoil": None,
            "NUMPYFILE": self.SETS.NUMPYFILESets,
        }
        
        sets_cls = sets_map.get(format)

        if sets_cls is not None:
            self.sets = sets_cls(db=self)
        else:
            self.sets = None
            print("\n\tWARNING: Actual format does not have sets class implemented. Sets methods will not be available in FRODO instance.\n")


        # -------- RESIDUALS FACTORY ------------
        residuals_map = {
            "CODA": self.RESIDUALS.CODAResiduals,
            "Airfoil": None,
            "NUMPYFILE": None
        }
        
        residuals_cls = residuals_map.get(format)
        
        if residuals_cls is not None:
            self.residuals = residuals_cls(db=self)
            # if format == 'CODA':
            #     for attr in dir(self.residuals):
            #         if not attr.startswith("_") and callable(getattr(self.residuals, attr)):
            #             setattr(self, attr, getattr(self.residuals, attr))
        else:
            self.residuals = None
            print("\n\tWARNING: Actual format does not have residuals class implemented. Residuals methods will not be available in FRODO instance.\n")
            
        if initial_parse:
            inicio = time.perf_counter()
            self._parse()
            fin_parse = time.perf_counter()
            
            print(f"Parse taked: {fin_parse - inicio:.4f} seconds")
                
    def _parse(self):
        self.reader.parse_simulation_dirs()

        self.sim_metadata = self.reader.sim_metadata
        self.df_state = self.reader.df_state
        
    def _sync_reader(self):
        """
        Sincroniza los atributos calculados por el reader
        con el objeto FRODO.
        """
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
            raise KeyError(f'Attribute data_dict not found. Please run extract_input() at least.')
        
    class READERS:

        class CODAReader():

            def __init__(self, root_dir: str):
                self.root_dir = root_dir
                self.output_dir = os.path.join(self.root_dir, "outputs")
                # os.makedirs(os.path.join(root_dir, 'metadata'), exist_ok=True)
                print(f'\n NEW CODA SIMULATION WILL BE LOADED FROM {root_dir}')
                
                try:
                    metadata_path = os.path.join(root_dir, 'metadata', 'cases_metadata.json')
                    with open(metadata_path, 'r') as f:
                        case_metadata = json.load(f)
                    
                    self.metadata = {
                        'eq_type': case_metadata.get('eq_type', None),
                        'folder_fmt': case_metadata.get('folder_fmt', None),
                        'design_vars': case_metadata.get('design_vars', None),
                        'num_stages': case_metadata.get('num_stages', None),
                    }
                    
                    df_cases = pd.DataFrame.from_dict(case_metadata.get('df_cases', {})).sort_values(by=self.metadata['design_vars'][0], ignore_index=True, axis=0)
                    
                    df_cases = df_cases.reset_index(drop=True)
                    df_cases.insert(0, "case_idx", df_cases.index.astype(np.int32))

                    self.metadata['df_cases'] = df_cases
                    
                except Exception as e:
                    print("WARNING: json metadata not found or loaded. Format folder will be guess from name's folders.\n\n")
                    print(e)
                    folders = os.listdir(self.output_dir)
                    possible_sep = ['_', '-']
                    sep_list = []
                    params_list = []
                    nfiles_outputs_stages = []
                    for f, folder in enumerate(folders):
                        for sep in possible_sep:
                            if sep in folder:
                                parts = folder.split(sep)
                                params = [float(re.findall(r"-?\d+\.?\d*", part)[0]) for part in parts if re.findall(r"-?\d+\.?\d*", part)]
                                params_list.append(params)
                                sep_list.append(sep)
                                nfiles_outputs_stages.append(len(
                                    SAM.Backpack.find_files(os.path.join(self.output_dir, folder), file_end = '.h5', notinfile='ci')
                                ))
                    if len(params_list) == 0:
                        raise ValueError("No simulation folders found or no parameters detected in folder names.")
                    
                    if all(len(params) == len(params_list[0]) for params in params_list) and all(sep == sep_list[0] for sep in sep_list) and all(nfiles == nfiles_outputs_stages[0] for nfiles in nfiles_outputs_stages):
                        self.metadata = {
                            'eq_type': None,
                            'folder_fmt': sep_list[0].join([parts[i] if not re.findall(r"-?\d+\.?\d*", parts[i]) else "{}" for i in range(len(parts))]),
                            'design_vars': [parts[i] for i in range(0, len(parts), 2)],
                            "num_stages": nfiles_outputs_stages[0]
                        }
                        
                        for f, folder in enumerate(folders):
                            if f == 0:
                                df_cases_array = np.zeros((len(folders), len(params_list[0])), dtype=float)
                            df_cases_array[f, :] = params_list[f]
                            
                        df_cases = pd.DataFrame(
                            df_cases_array,
                            columns=self.metadata['design_vars']
                        )

                        df_cases = df_cases.reset_index(drop=True)
                        df_cases.insert(0, "case_idx", df_cases.index.astype(np.int32))

                        self.metadata['df_cases'] = df_cases
                        # self.metadata['df_cases'] = pd.DataFrame(df_cases_array, columns=self.metadata['design_vars'])
                    else:
                        raise ValueError("Inconsistent folder naming detected. Please provide a valid cases_metadata.json file.")
                    
                self.sim_metadata = {}
                
                self.df_state = pd.DataFrame()
                self.data_dict = {}
                
            def parse_simulation_dirs(self):
                """
                Parse the output directory to build simulation metadata and state information.

                Populates:
                    - self.sim_metadata: Dictionary mapping each simulation folder to its metadata
                    (AoA, Mach, folder name, path, stages, computation times).
                    - self.df_state: Pandas DataFrame summarizing AoA, Mach, and stage counts.

                No arguments are required. Prints the number of simulations found.
                """

                folder_fmt = self.metadata["folder_fmt"]
                pattern = SAM.Backpack.folder_fmt_to_pattern(folder_fmt)


                # pattern = re.compile(r"aoa_([-\d\.]+)_mach_([\d\.]+)")
                n_sim=0
                for folder in os.listdir(self.output_dir):
                    if not pattern.match(folder):
                        continue
                    
                    pattern_nums = re.compile(r"[-\d\.]+")
                    nums_folder = pattern_nums.findall(folder) # Aquí cogemos el aoa y mach con la precisión de escritura en el nombre de la carpeta. Es mejor que busquemos los valores más cercanos en el df_cases (metadata).
                    # Convert token strings to floats so numeric operations work correctly
                    nums_folder = np.array([float(x) for x in pattern_nums.findall(folder)], dtype=float)
                    # Aquí cogemos el aoa y mach con la precisión de escritura en el nombre de la carpeta. Es mejor que busquemos los valores más cercanos en el df_cases (metadata).
                    # Encontrar la fila de df_cases con isclosed
                    nums_df_cases = self.metadata['df_cases'][self.metadata['design_vars']].values
                    idx_closest = np.argmin(np.linalg.norm(nums_df_cases - nums_folder, axis=1))
                    
                    nums = self.metadata['df_cases'].iloc[idx_closest][self.metadata['design_vars']].values.tolist() # Aquí ya cogemos los valores de AoA y Mach con la precisión del df_cases, que es la que realmente nos interesa para luego hacer los plots. Además, si el formato de las carpetas no es consistente, esto nos permite igual coger los valores aunque no estén escritos con la misma precisión o incluso en el mismo orden (si el folder_fmt es algo como "aoa_{}_mach_{}" o "mach_{}_aoa_{}").
                    # guardar nombre de carpeta en df_cases para referencia futura
                    self.metadata['df_cases'].at[idx_closest, 'folder'] = folder
                    
                    full_path = os.path.join(self.output_dir, folder)

                    if not os.path.isdir(full_path):
                        continue

                    files = os.listdir(full_path)
                    stage_dict = {}

                    for fname in files:
                        if fname.startswith("output_"):
                            parts = fname.split("_")
                            if len(parts) >= 2:
                                stage_raw = os.path.splitext(parts[1])[0]
                                if stage_raw.isdigit():
                                    stage = int(stage_raw)
                                    ext = os.path.splitext(fname)[-1].lstrip(".")
                                    if stage not in stage_dict:
                                        stage_dict[stage] = {
                                            "files": [],
                                            "types": set(),
                                        }
                                    stage_dict[stage]["files"].append(fname)
                                    stage_dict[stage]["types"].add(ext)
                    
                    for stage in stage_dict.keys():
                        stage_dict[stage]["types"] = list(stage_dict[stage]["types"])
                        
                    self.sim_metadata[folder] = {
                        "folder": folder,
                        "path": full_path,
                        "stages": stage_dict,
                        "computation times": []
                    }
                    self.sim_metadata[folder].update(
                        {var: val for var, val in zip(self.metadata["design_vars"], nums)})
                    n_sim+=1

                print(f"{len(self.sim_metadata)} simulations found.")

                state_array = np.zeros((len(self.sim_metadata), len(self.metadata['design_vars'])+1), dtype=float)
                for n_sim, sim_key in enumerate(self.sim_metadata.keys()):
                    sim = self.sim_metadata[sim_key]
                    for i, var in enumerate(self.metadata['design_vars']):
                        state_array[n_sim, i] = sim[var]
                    state_array[n_sim, -1] = len(sim["stages"])

                df_state = pd.DataFrame(state_array, columns=self.metadata['design_vars'] + ['stage']).sort_values(by=self.metadata['design_vars'][0]).reset_index(drop=True)
                self.df_state = pd.merge(df_state, self.metadata['df_cases'], on=self.metadata['design_vars'], how='left')
                # Propuesta de meter df_post (resultado de db.residuals.get_df_metrics()) en df_state
                # # Esta parte está ahora mismo en desuso. El bucle de stage tiene poco sentido y se puede invertir el orden con el de sim_metadata.
                # for stage in range(10):
                #     for sim_key, sim in self.sim_metadata.items():
                #         path = sim['path']
                #         if stage not in sim['stages']:
                #             continue
                #         try:
                #             res_time = self.parse_cfd_log(path)  # Creo que perdí esta función en una de las versiones anteriores de FRODO. Al menos tiene que estar en alguna de 11/2025
                #             # self.sim_metadata[sim_key]['computation times'][stage] = res_time['stage_times_hours'][stage]
                #             self.sim_metadata[sim_key]['computation times'].append(res_time['stage_times_hours'][stage])
                #         except:
                #             self.sim_metadata[sim_key]['computation times'].append(np.nan)

            def print_available_cadgroup_ids(
                self, stage,
                vtu_type:Literal["surface", "volume"] = "surface"
                ):
                """
                Print available combinations of CADGroupIDs and cell_data keys across all simulations.

                Args:
                    stage (int): Stage number to inspect.
                    vtu_type (Literal["surface","volume"]): Type of .vtu file to use (default "surface").

                Returns:
                    None. Prints a summary grouping simulations by CADGroupIDs and available cell_data keys.
                """

                summary = defaultdict(list)

                for sim_key, sim in self.sim_metadata.items():
                    try:
                        mesh = self.load_vtu_from_stage(
                            sim_key,
                            stage,
                            vtu_type
                        )
                        cad_ids = tuple(sorted(np.unique(mesh.cell_data["CADGroupID"])))
                        cell_keys = tuple(sorted(mesh.cell_data.keys()))
                        key = (cad_ids, cell_keys)
                        summary[key].append(sim["folder"])
                    except Exception as e:
                        print(f"Error en simulación {sim_key}: {e}")

                print("\nResumen de combinaciones CADGroupID y cell_data:")
                for (cad_ids, cell_keys), folders in summary.items():
                    print(f"CADGroupIDs: {cad_ids}")
                    print(f"cell_data keys: {cell_keys}")
                    print(f"Simulaciones ({len(folders)}): {folders}\n")

            def load_vtu_from_stage(
                self,
                case_name,
                stage:int,
                vtu_type:Literal["surface", "volume"] = "surface",
                verbose:bool = False
                ):
                """
                Load a .vtu file for a given simulation and stage.

                Args:
                    sim (str): Key of the simulation in self.sim_metadata.
                    stage (int): Stage number to load (e.g., 0, 1).
                    vtu_type (Literal["surface","volume"]): Type of file to load, default is "surface".
                    verbose (bool): If True, prints details about the loaded mesh.

                Returns:
                    pyvista.UnstructuredGrid: The loaded VTU mesh for the specified simulation and stage.

                Raises:
                    FileNotFoundError: If no matching .vtu file is found.
                """
                sim = self.sim_metadata[case_name]
                files = sim["stages"].get(stage, {}).get("files", [])
                path = sim["path"]

                for fname in files:
                    if fname.endswith(".vtu") and vtu_type in fname:
                        full_path = os.path.join(path, fname)
                        mesh = pv.read(full_path)
                        mesh = SAM.Backpack.ensure_cell_data(mesh) #Pasar datos a celdas.
                        if verbose:
                            print(f"Mesh from {sim['folder']} loaded\n")
                            print(f"No of points:\t {mesh.n_points}")
                            # print(f"No of cells:\t {mesh.cell_data["CADGroupID"].shape}")
                            print(f"Shape of conectivities arrays:")
                            for key in mesh.cells_dict.keys():
                                print(f"{key}: {mesh.cells_dict[key].shape}")
                        return mesh

                raise FileNotFoundError(f"No se encontró archivo .vtu con tipo '{vtu_type}' en stage {stage} de la simulación {sim['folder']}")
            
            def extract_inputs(
                self,
                id_groups: Union[int, tuple[int]],
                vtu_type: Literal['volume', 'surface'] = 'surface',
                method_to_sort: Literal["centroid", "kdtree"] = 'lexsort',
                cases_idx:Union[list[int], tuple[int], int, 'all']='all',
                verbose:bool = False,
                
                ):
                
                """
                Extract input mesh and metadata for one or multiple CADGroup IDs.

                Args:
                    id_groups (tuple): Tuple of IDs or tuples of IDs.
                        - Example: (3,) for a single group.
                        - Example: ((1,2), 3) to combine groups 1 and 2 into one key and also process group 3.
                    vtu_type (Literal['volume', 'surface']): Type of .vtu file to load, default is 'surface'.
                    cases_idx (Union[list[int], tuple[int], int, 'all']): Indices of simulations to process.
                        - Example: [0, 2] to process the first and third simulations.
                        
                Populates self.data_dict with:
                    - 'Coord', 'NodeCoord': Coordinates of centroids and nodes.
                    - 'FlCc': Array of AoA and Mach for each simulation.
                    - 'Conec': Unified connectivity arrays.
                    - 'idx_sort', 'idx_sort_nodes': Sorting indices for centroids and nodes.
                    - 'eltype', 'cellOrder', 'pointOrder': Mesh type and ordering arrays.

                Returns:
                    None. Updates self.data_dict in place.
                """
                
                num_stages = self.metadata.get("num_stages", 1)
                design_vars = self.metadata.get("design_vars", [])
                df_cases = self.metadata.get("df_cases", pd.DataFrame())
                
                df_cases = self.metadata.get("df_cases", pd.DataFrame())

                # --- Normalización fuerte de cases_idx ---
                if isinstance(cases_idx, str):
                    if cases_idx.lower() == "all":
                        cases_idx = list(range(len(df_cases)))
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

                # Validación
                max_cases = len(df_cases)
                if any(i >= max_cases or i < 0 for i in cases_idx):
                    raise IndexError("cases_idx contains out-of-range values.")

                # Obtener sim_keys alineados
                sim_keys = df_cases.loc[cases_idx, "folder"].tolist()

                ncases = len(sim_keys)
                
                if method_to_sort == 'lexsort' or (method_to_sort is None):
                    sort_function = SAM.Weapons.sort_lexsort
                elif method_to_sort == 'centroid':
                    sort_function = SAM.Weapons.sort_by_centroid
                elif method_to_sort == 'kdtree':
                    sort_function = SAM.Weapons.sort_closed_curve_by_kdtree
                elif method_to_sort == 'convex_hull':
                    sort_function = SAM.Weapons.sort_points_by_hull_projection
                else:
                    raise ValueError('method_to_sort not supported. Available method: lexsort, centroid, kdtree and convex_hull')
                
                for group_id in id_groups:

                    if isinstance(group_id, tuple):
                        ids_to_combine = group_id
                        key_suffix = "_".join(map(str, ids_to_combine))
                    elif isinstance(group_id, int):
                        ids_to_combine = (group_id,)
                        key_suffix = str(group_id)

                    key = f"CADGroup_{key_suffix}"

                    Coord_base = None
                    NodeCoord_base = None

                    Conec_base = None
                    eltype_base = None
                    cellOrder_base = None
                    pointOrder_base = None

                    FlCc = None
                    idx_sort = None
                    idx_sort_nodes = None

                    for stage in range(num_stages):
                        if verbose:
                            print(f'Stage {stage}:\n')
                        for cont, case_i in enumerate(cases_idx):

                            sim_key = sim_keys[cont]
                            if verbose:
                                print(f'\t Cont: {cont}\tCase: {case_i}\tFolder: {sim_key}\n')
                            try:
                                mesh = self.load_vtu_from_stage(
                                    case_name=sim_key,
                                    stage=stage,
                                    vtu_type=vtu_type
                                )

                                if "CADGroupID" not in mesh.cell_data:
                                    raise ValueError("No se encuentra 'CADGroupID' en cell_data.")

                                groups = mesh.cell_data["CADGroupID"]
                                mask = np.isin(groups, ids_to_combine)

                                celdas = mesh.extract_cells(mask)
                                centroids = np.array(celdas.cell_centers().points, dtype=np.float64)
                                nodes = np.array(celdas.points, dtype=np.float64)

                                connectivity = SAM.Backpack.get_unified_connectivity(mesh)[mask]

                                centroids_sorted, idx = sort_function(points=centroids)
                                nodes_sorted, idx_nodes = sort_function(points=nodes)

                                if stage == 0 and cont == 0:

                                    npoints = centroids.shape[0]
                                    nnodes = nodes.shape[0]

                                    idx_sort = np.zeros((num_stages, ncases, npoints), dtype=np.int32)
                                    idx_sort_nodes = np.zeros((num_stages, ncases, nnodes), dtype=np.int32)

                                    FlCc = np.zeros((ncases, len(design_vars)), dtype=np.float64)

                                    Coord_base = centroids_sorted.copy()
                                    NodeCoord_base = nodes_sorted.copy()

                                    Conec_base = connectivity.copy()
                                    eltype_base = celdas.celltypes.copy()
                                    cellOrder_base = np.arange(celdas.n_cells, dtype=np.float64)
                                    pointOrder_base = np.arange(celdas.n_points, dtype=np.float64)

                                else:
                                    if not SAM.Backpack.same_columns(
                                        np.stack([Coord_base, centroids_sorted], axis=0)
                                        ):
                                        raise ValueError(
                                            f"Inconsistent cell coordinates at stage {stage}, case {sim_key}"
                                        )

                                    if not SAM.Backpack.same_columns(
                                        np.stack([NodeCoord_base, nodes_sorted], axis=0)
                                        ):
                                        raise ValueError(
                                            f"Inconsistent node coordinates at stage {stage}, case {sim_key}"
                                        )
                                    
                                idx_sort[stage, cont] = idx
                                idx_sort_nodes[stage, cont] = idx_nodes

                                if stage == 0:
                                    FlCc[cont] = [self.sim_metadata[sim_key][p] for p in design_vars]

                            except Exception as e:
                                print(f"Error reading simulation's inputs {sim_key}, group {group_id}, stage {stage}: {e}")

                    if key not in self.data_dict:
                        self.data_dict[key] = {}

                    self.data_dict[key].update({
                        'Coord': Coord_base,
                        'NodeCoord': NodeCoord_base,
                        'FlCc': FlCc,
                        'Conec': Conec_base,
                        'idx_sort': idx_sort,
                        'idx_sort_nodes': idx_sort_nodes,
                        'eltype': eltype_base,
                        'cellOrder': cellOrder_base,
                        'pointOrder': pointOrder_base
                    })
                    
            def extract_inputs_ant(
                self,
                id_groups: Union[int, tuple[int]],
                vtu_type: Literal['volume', 'surface'] = 'surface',
                cases_idx:Union[list[int], tuple[int], int, 'all']='all'
                
                ):
                
                """
                Extract input mesh and metadata for one or multiple CADGroup IDs.

                Args:
                    id_groups (tuple): Tuple of IDs or tuples of IDs.
                        - Example: (3,) for a single group.
                        - Example: ((1,2), 3) to combine groups 1 and 2 into one key and also process group 3.
                    vtu_type (Literal['volume', 'surface']): Type of .vtu file to load, default is 'surface'.
                    cases_idx (Union[list[int], tuple[int], int, 'all']): Indices of simulations to process.
                        - Example: [0, 2] to process the first and third simulations.
                        
                Populates self.data_dict with:
                    - 'Coord', 'NodeCoord': Coordinates of centroids and nodes.
                    - 'FlCc': Array of AoA and Mach for each simulation.
                    - 'Conec': Unified connectivity arrays.
                    - 'idx_sort', 'idx_sort_nodes': Sorting indices for centroids and nodes.
                    - 'eltype', 'cellOrder', 'pointOrder': Mesh type and ordering arrays.

                Returns:
                    None. Updates self.data_dict in place.
                """
                
                num_stages = self.metadata.get("num_stages", 1)
                design_vars = self.metadata.get("design_vars", [])
                
                sim_keys_total = list(self.sim_metadata.keys())
                    
                # if cases_idx == 'all':
                #     sim_keys = sim_keys_total
                #     self.metadata['sim_keys'] = 'all'
                # elif isinstance(cases_idx, int):
                #     sim_keys = [self.case_per_idx(cases_idx)]
                #     self.metadata['sim_keys'] = [cases_idx]
                # elif isinstance(cases_idx, (list, tuple)):
                #     sim_keys = [self.case_per_idx(i) for i in cases_idx]
                #     self.metadata['sim_keys'] = cases_idx
                # else:
                #     raise ValueError("Invalid type for cases_idx. Must be 'all', int, list[int], or tuple[int].")
                
                if cases_idx == 'all':
                    sim_keys = sim_keys_total
                    self.metadata['sim_keys'] = 'all'
                elif isinstance(cases_idx, int):
                    sim_keys = [self.case_per_idx(cases_idx)]
                    self.metadata['sim_keys'] = [cases_idx]
                elif isinstance(cases_idx, (list, tuple)):
                    sim_keys = [self.case_per_idx(i) for i in cases_idx]
                    self.metadata['sim_keys'] = cases_idx
                else:
                    raise ValueError("Invalid type for cases_idx. Must be 'all', int, list[int], or tuple[int].")
                
                
                ncases = len(sim_keys)
                
                for group_id in id_groups:

                    if isinstance(group_id, tuple):
                        ids_to_combine = group_id
                        key_suffix = "_".join(map(str, ids_to_combine))
                    elif isinstance(group_id, int):
                        ids_to_combine = (group_id,)
                        key_suffix = str(group_id)

                    key = f"CADGroup_{key_suffix}"

                    Coord_base = None
                    NodeCoord_base = None

                    Conec_base = None
                    eltype_base = None
                    cellOrder_base = None
                    pointOrder_base = None

                    FlCc = None
                    idx_sort = None
                    idx_sort_nodes = None

                    for stage in range(num_stages):

                        for case_i, sim_key in enumerate(sim_keys):

                            try:
                                mesh = self.load_vtu_from_stage(
                                    case_name=sim_key,
                                    stage=stage,
                                    vtu_type=vtu_type
                                )

                                if "CADGroupID" not in mesh.cell_data:
                                    raise ValueError("No se encuentra 'CADGroupID' en cell_data.")

                                groups = mesh.cell_data["CADGroupID"]
                                mask = np.isin(groups, ids_to_combine)

                                celdas = mesh.extract_cells(mask)
                                centroids = np.array(celdas.cell_centers().points)
                                nodes = np.array(celdas.points)

                                connectivity = SAM.Backpack.get_unified_connectivity(mesh)[mask]

                                idx = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
                                idx_nodes = np.lexsort((nodes[:, 2], nodes[:, 1], nodes[:, 0]))

                                centroids_sorted = centroids[idx]
                                nodes_sorted = nodes[idx_nodes]

                                if stage == 0 and case_i == 0:

                                    npoints = centroids.shape[0]
                                    nnodes = nodes.shape[0]

                                    idx_sort = np.zeros((num_stages, ncases, npoints), dtype=np.int32)
                                    idx_sort_nodes = np.zeros((num_stages, ncases, nnodes), dtype=np.int32)

                                    FlCc = np.zeros((ncases, len(design_vars)), dtype=np.float32)

                                    Coord_base = centroids_sorted.copy()
                                    NodeCoord_base = nodes_sorted.copy()

                                    Conec_base = connectivity.copy()
                                    eltype_base = celdas.celltypes.copy()
                                    cellOrder_base = np.arange(celdas.n_cells)
                                    pointOrder_base = np.arange(celdas.n_points)

                                else:
                                    if not SAM.Backpack.same_columns(
                                        np.stack([Coord_base, centroids_sorted], axis=0)
                                    ):
                                        raise ValueError(
                                            f"Inconsistent cell coordinates at stage {stage}, case {sim_key}"
                                        )

                                    if not SAM.Backpack.same_columns(
                                        np.stack([NodeCoord_base, nodes_sorted], axis=0)
                                    ):
                                        raise ValueError(
                                            f"Inconsistent node coordinates at stage {stage}, case {sim_key}"
                                        )

                                idx_sort[stage, case_i] = idx
                                idx_sort_nodes[stage, case_i] = idx_nodes

                                if stage == 0:
                                    FlCc[case_i] = [self.sim_metadata[sim_key][p] for p in design_vars]

                            except Exception as e:
                                print(f"Error reading simulation's inputs {sim_key}, group {group_id}, stage {stage}: {e}")

                    if key not in self.data_dict:
                        self.data_dict[key] = {}

                    self.data_dict[key].update({
                        'Coord': Coord_base,
                        'NodeCoord': NodeCoord_base,
                        'FlCc': FlCc,
                        'Conec': Conec_base,
                        'idx_sort': idx_sort,
                        'idx_sort_nodes': idx_sort_nodes,
                        'eltype': eltype_base,
                        'cellOrder': cellOrder_base,
                        'pointOrder': pointOrder_base
                    })
   
            def extract_outputs(
                self,
                stage: int,
                id_groups: Union[int, tuple[int]],
                vtu_type: Literal['volume', 'surface'] = 'surface',
                cases_idx:Union[list[int], tuple[int], int, 'all']='all',
                var_name_excluded: Union[list[str], tuple[str], None] = None,
                verbose:bool = False,
                ):

                """
                Extract cell-based output variables for one or multiple CADGroup IDs.

                Args:
                    stage (int): Stage number from which to extract variables.
                    id_groups (tuple): Tuple of IDs or tuples of IDs, similar to extract_inputs.
                    vtu_type (Literal['volume', 'surface']): Type of .vtu file to use, default is "surface".
                    var_name_excluded (list or tuple or None): List of variable names to exclude from extraction.

                Requirements:
                    extract_inputs must be executed first for the same groups (to provide idx_sort).

                Populates self.data_dict[key]['Vars'][stage] with:
                    - Variable arrays transposed to shape (n_sims, n_points).

                Returns:
                    None. Updates self.data_dict in place.
                """
                num_stages = self.metadata.get("num_stages", 1)
                design_vars = self.metadata.get("design_vars", [])
                df_cases = self.metadata.get("df_cases", pd.DataFrame())

                if isinstance(cases_idx, str):
                    if cases_idx.lower() == "all":
                        cases_idx = list(range(len(df_cases)))
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

                # Validación
                max_cases = len(df_cases)
                if any(i >= max_cases or i < 0 for i in cases_idx):
                    raise IndexError("cases_idx contains out-of-range values.")

                # Ahora sí: lista limpia
                sim_keys = df_cases.loc[cases_idx, "folder"].tolist()
                for group_id in id_groups:

                    if isinstance(group_id, tuple):
                        ids_to_combine = group_id
                        key_suffix = "_".join(map(str, ids_to_combine))
                    elif isinstance(group_id, int):
                        ids_to_combine = (group_id,)
                        key_suffix = str(group_id)
                    else:
                        raise TypeError("Invalid id_groups format")

                    key = f"CADGroup_{key_suffix}"

                    if key not in self.data_dict or 'idx_sort' not in self.data_dict[key]:
                        raise RuntimeError(
                            f"No idx_sort found for {key}. Run extract_inputs first."
                        )

                    idx_sort_all = self.data_dict[key]['idx_sort']

                    if stage >= idx_sort_all.shape[0]:
                        raise ValueError(f"Stage {stage} out of bounds for idx_sort.")

                    first_sim = sim_keys[0]
                    mesh0 = self.load_vtu_from_stage(
                        case_name=first_sim,
                        stage=stage,
                        vtu_type=vtu_type
                    )

                    var_names = [
                        v for v in mesh0.cell_data.keys()
                        if (var_name_excluded is None or v not in var_name_excluded)
                    ]

                    var_storage = {v: [] for v in var_names}

                    expected_ncells = None

                    for cont, case_i in enumerate(cases_idx):

                        sim_key = sim_keys[cont]

                        if verbose:
                            print(f'\tCont: {cont}\tCase: {case_i}\tFolder: {sim_key}\n')
                        mesh = self.load_vtu_from_stage(
                            case_name=sim_key,
                            stage=stage,
                            vtu_type=vtu_type
                        )

                        groups = mesh.cell_data["CADGroupID"]
                        mask = np.isin(groups, ids_to_combine)

                        ncells = np.sum(mask)

                        if expected_ncells is None:
                            expected_ncells = ncells
                        elif ncells != expected_ncells:
                            raise ValueError(
                                f"Inconsistent number of cells for {key} "
                                f"in simulation {sim_key}: "
                                f"{ncells} vs expected {expected_ncells}"
                            )

                        sorter = idx_sort_all[stage, cont].astype(np.int32)

                        for var_name in var_names:

                            data = np.array(mesh.cell_data[var_name])
                            data_masked = data[mask]

                            if data_masked.ndim == 1:
                                data_sorted = data_masked[sorter]
                            elif data_masked.ndim == 2:
                                data_sorted = data_masked[sorter, :]
                            else:
                                raise ValueError(
                                    f"Variable {var_name} has unsupported ndim {data_masked.ndim}"
                                )

                            var_storage[var_name].append(data_sorted)

                    if key not in self.data_dict:
                        self.data_dict[key] = {}

                    if 'Vars' not in self.data_dict[key]:
                        self.data_dict[key]['Vars'] = {}

                    if str(stage) not in self.data_dict[key]['Vars']:
                        self.data_dict[key]['Vars'][str(stage)] = {}

                    for var_name in var_names:

                        arr = np.stack(var_storage[var_name], axis=0)

                        if arr.ndim == 2:
                            self.data_dict[key]['Vars'][str(stage)][var_name] = (
                                arr.T.astype(np.float64)
                            )
                        elif arr.ndim == 3:
                            arr = np.transpose(arr, (2, 1, 0))
                            self.data_dict[key]['Vars'][str(stage)][var_name] = (
                                arr.astype(np.float64)
                            )
                        else:
                            raise RuntimeError("Unexpected stacked array dimension.")

            def plot_state(self, figsize = (15, 7)):
                """
                Plot a scatter diagram showing the AoA/Mach space of all simulations and their completion state.

                Args:
                    figsize (tuple): Figure size for the two-panel plot. Default is (15, 7).

                Returns:
                    None. Displays the figure with annotated points representing each simulation.
                """
                num_states = self.metadata['num_stages']
                design_vars = self.metadata['design_vars']
                df_state = self.df_state
                #tantos colores como estados haya, asignados de forma que el estado 0 sea rojo, el intermedio amarillo y el final verde.
                cmap_custom = plt.get_cmap("RdYlGn", num_states)
                norm=BoundaryNorm(np.arange(-0.5, num_states+0.5, 1), cmap_custom.N)

                # Omitir valores de desing_vars que sean constantes en todas las simulaciones, ya que no aportan información al gráfico de estado.
                design_vars_filtered = [var for var in design_vars if df_state[var].nunique() > 1]
                
                
                if design_vars_filtered is None or len(design_vars_filtered) < 2:
                    raise ValueError("Se requieren al menos dos variables de diseño para el gráfico de estado.")
                elif len(design_vars_filtered) == 2: # pintando scatter 2D con AoA en x y Mach en y, coloreando por estado de convergencia (última columna de state_array)
                    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
                    ax[0].scatter(
                        df_state[design_vars_filtered[0]],
                        df_state[design_vars_filtered[1]],
                        c=df_state['stage'],
                        cmap=cmap_custom,
                        norm = norm,
                        s=100
                        )
                    ax[0].set_xlabel(design_vars_filtered[0])
                    ax[0].set_ylabel(design_vars_filtered[1])
                    ax[0].set_title("Status of cases")
                    ax[0].grid()
                    ax[0].legend(handles=[
                        plt.Line2D([0], [0], marker='o', color='w', label='Not started', markerfacecolor='red', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='In progress', markerfacecolor='yellow', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Finished', markerfacecolor='green', markersize=10)
                    ], loc='lower center', bbox_to_anchor=(1.15, 0.5), ncols = 1)
                    
                    for i, (x,y) in enumerate(zip(df_state[design_vars_filtered[0]].values, df_state[design_vars_filtered[1]].values)): 
                        if df_state['stage'][i] != 0:
                            ax[0].annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                        else:
                            #ax.text(x, y, str(i), fontsize=8)
                            ax[0].annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8)

                    mask_finished = df_state['stage'].values == np.max(df_state['stage'].values)
                    ax[1].scatter(
                        df_state[design_vars_filtered[0]][mask_finished],
                        df_state[design_vars_filtered[1]][mask_finished],
                        s=100
                        )
                    for i, (x, y) in enumerate(zip(df_state[design_vars_filtered[0]].values, df_state[design_vars_filtered[1]].values)):
                        ax[1].annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)
                        
                    ax[1].set_xlabel(design_vars_filtered[0])
                    ax[1].set_ylabel(design_vars_filtered[1])
                    ax[1].set_title(f"Finished cases {df_state['stage'][df_state['stage'].values == num_states].shape[0] / df_state['stage'].values.shape[0] * 100:.2f}%")
                    ax[1].grid()
                    fig.subplots_adjust(wspace=0.3, hspace=0.2)
                    fig.show()
                    
                elif len(design_vars_filtered) == 3:
                    fig, ax = plt.subplots(1, 1, figsize=figsize)
                    sc = ax.scatter(
                        df_state[design_vars_filtered[0]],
                        df_state[design_vars_filtered[1]],
                        df_state[design_vars_filtered[2]],
                        c=df_state['stage'],
                        cmap=cmap_custom,
                        norm=norm,
                        s=100
                    )
                    ax.set_xlabel(design_vars_filtered[0])
                    ax.set_ylabel(design_vars_filtered[1])
                    ax.set_zlabel(design_vars_filtered[2])
                    ax.set_title("Status of cases")
                    ax.grid()
                    ax.legend(handles=[
                            plt.Line2D([0], [0], marker='o', color='w', label='Not started', markerfacecolor='red', markersize=10),
                            plt.Line2D([0], [0], marker='o', color='w', label='In progress', markerfacecolor='yellow', markersize=10),
                            plt.Line2D([0], [0], marker='o', color='w', label='Finished', markerfacecolor='green', markersize=10)
                        ], loc='lower center', bbox_to_anchor=(1.15, 0.5), ncols = 1)

                    for i, (x, y, z) in enumerate(zip(df_state[design_vars_filtered[0]].values, df_state[design_vars_filtered[1]].values, df_state[design_vars_filtered[2]].values)):
                        ax.annotate(f"{i}", (x, y, z), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)

            def case_per_idx_GPT(self, case_idx: int):

                sim_keys = list(self.sim_metadata.keys())

                if case_idx >= len(sim_keys):
                    raise IndexError("case_idx out of range")

                case_name = sim_keys[case_idx]

                return case_name

            def case_per_idx(self, idx: int):
                """
                Dado un índice de df_state, devuelve el nombre del caso
                correspondiente en sim_metadata.

                Ahora primero intenta resolver el nombre de carpeta usando
                self.metadata['df_cases'].folder (si existe). Si no está,
                cae al método previo de buscar por igualdad de parámetros.
                """
                # 1) Intentar obtener carpeta desde df_cases (metadatos)
                df_cases = self.metadata.get('df_cases', None)
                if df_cases is not None:
                    # Si df_cases tiene columna 'folder', preferimos usarla
                    if 'folder' in df_cases.columns:
                        # Buscar por columna case_idx si existe
                        if 'case_idx' in df_cases.columns:
                            match = df_cases.loc[df_cases['case_idx'] == idx, 'folder']
                            if not match.empty and pd.notna(match.iloc[0]):
                                return match.iloc[0]
                        # Fallback: intentar por index igual a idx
                        try:
                            folder = df_cases.at[idx, 'folder']
                            if pd.notna(folder):
                                return folder
                        except Exception:
                            pass

                # 2) Si no hay carpeta en df_cases, usar la lógica antigua (comparar parámetros)
                row = self.df_state.loc[idx]
                for case_name, sim in self.sim_metadata.items():
                    try:
                        if all(
                            np.isclose(float(sim[var]), float(row[var]))
                            for var in self.metadata['design_vars']
                        ):
                            return case_name
                    except Exception:
                        # en caso de datos faltantes o tipos incompatibles, seguir buscando
                        continue

                raise KeyError(f"No case found in sim_metadata for df_state idx={idx}")

            def plot_integrals_from_case(
                self,
                case_name: str = None,
                case_idx: Union[int, None] = None,
                stage: Union[list, tuple, 'all'] = 'all',
                save_dir: Union[None, str] = None,
                **kwargs
                ):
                
                if case_name == None:
                    if case_idx == None:
                        raise ValueError("You must provide either case_name or case_idx.")
                    else:
                        case_name = self.case_per_idx(case_idx)
                        case_path = self.sim_metadata[case_name]['path']
                        if stage == 'all':
                            stages = list(self.sim_metadata[case_name]['stages'].keys())
                        elif stage == None:
                            raise ValueError("Stage cannot be None.")
                        else:
                            stages = stage
                else:
                    case_path = self.sim_metadata[case_name]['path']
                    if stage == 'all':
                        stages = list(self.sim_metadata[case_name]['stages'].keys())
                    elif stage == None:
                        raise ValueError("Stage cannot be None.")
                    else:
                        stages = stage
                
                files = SAM.Backpack.find_files(case_path, "_wall_boundary_integrals.dat")
                if len(files) == 0:
                    raise ValueError(f'No files found in {case_path} with the ending "_wall_boundary_integrals.dat"')

                if stages is not None:
                    files = [f for f in files if any(f"_{s}_" in f for s in stages)]
                    if len(files) == 0:
                        raise ValueError(f'No files found in {case_path} for the specified stages {stages}.')
                    
                df = SAM.Backpack.get_df_from_csv(files)
                # --- Detectar stages por reinicio de Iteration ---
                if "Iteration" in df.columns:
                    df["stage"] = df["Iteration"].diff().lt(0).cumsum()
                else:
                    raise ValueError("Column 'Iteration' not found in dataframe, cannot detect stages.")

                # df = csv_residuals_to_df(files)
                titles = df.columns
                cx = 0
                cy_list = [1, 2, 3]
                cmap = plt.get_cmap("viridis")
                colors = cmap(np.linspace(0, 1, len(cy_list)))

                # Crear subplots con 2 filas, compartir eje X
                fig, ax_main = plt.subplots(1, 1, figsize=kwargs.get('figsize', (10, 8)))

                # Gráfico principal
                for color, cy in zip(colors, cy_list):
                    ax_main.plot(
                        df[titles[cx]],
                        df[titles[cy]],
                        color=color,
                        label=f"{titles[cy]}"
                    )
                stage_changes = df.index[df["stage"].diff().fillna(0) != 0]

                for idx in stage_changes:
                    x_val = df.iloc[idx, cx]
                    ax_main.axvline(
                        x=x_val,
                        color="black",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.5
                    )
                    
                ax_main.set_ylabel("Values")
                # ax_main.set_ylim(top=1.5)
                # ax_main.set_yscale('log')
                ax_main.legend()
                ax_main.grid(True)

                fig.suptitle(f"Wall Integrals Boundary Values - {case_name}" if case_idx is None else f"Wall Integrals Boundary Values - Case {case_idx}", fontsize=16)
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{case_name}_stages_{'_'.join(map(str, stages))}_wall_integrals.png")
                    fig.savefig(save_path, bbox_inches='tight')
                    print(f"Figure saved to {save_path}")
                else:
                    plt.show()

        class AIRFOILReader():

            def __init__(self, root_dir:str):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_data = pd.DataFrame()
                self.data_dict = {}

                print("Format developed to read Airfoil Database from AASM (Applied Aerodynamics Surrogate Modeling).\n")
                
            def parse_simulation_dirs(self):
                folders = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
                state_array=[]
                for folder in folders:
                    sample_files = list(glob.glob(os.path.join(folder, "*")))
                    self.sim_metadata[os.path.basename(folder)]={}
                    
                    state_array_folder = np.zeros((len(sample_files), 8), dtype=object)
                    for nsim, fname in enumerate(sample_files):
                        sim_data = FRODO.READERS.AIRFOILReader.read_dat(fname)
                        sample_key = f"Sample_{sim_data['sample_number']}"
                        self.sim_metadata[os.path.basename(folder)][sample_key] = sim_data['metadata']
                        self.sim_metadata[os.path.basename(folder)][sample_key]['path'] = fname
                        self.sim_metadata[os.path.basename(folder)][sample_key]['available_vars'] = list(sim_data['df_data'].columns)
                        
                        # for i, key in enumerate(list(sim_data['metadata'].keys())):
                        #     state_array_folder[nsim, i] = sim_data['metadata'][key]
                        #     u=i
                        
                        # poner como bucle para ahorrar lineas y generalizar
                        state_array_folder[nsim, 0] = self.sim_metadata[os.path.basename(folder)][sample_key]['AoA']
                        state_array_folder[nsim, 1] = self.sim_metadata[os.path.basename(folder)][sample_key]['Mach']
                        state_array_folder[nsim, 2] = self.sim_metadata[os.path.basename(folder)][sample_key]['Re']
                        state_array_folder[nsim, 3] = self.sim_metadata[os.path.basename(folder)][sample_key]['Cl']
                        state_array_folder[nsim, 4] = self.sim_metadata[os.path.basename(folder)][sample_key]['Cd']
                        state_array_folder[nsim, 5] = self.sim_metadata[os.path.basename(folder)][sample_key]['Cmy']
                        
                        state_array_folder[nsim, 6] = [[self.sim_metadata[os.path.basename(folder)][sample_key][f'CST{i}'] for i in range(1, 10)]]
                        state_array_folder[nsim, 7] = os.path.basename(folder)

                    state_array.append(state_array_folder)
                self.df_state = pd.DataFrame(np.concatenate(state_array, axis=0, dtype=object), columns=["AoA", "Mach", "Re", "Cl", "Cd", "Cmy", "CST Coord", "Folder"]).sort_values(by="AoA").reset_index(drop=True)
                          
            def extract_inputs(self):
                """
                Extrae las coordenadas y parámetros globales de todos los samples.
                Llena self.data_dict con:
                    - 'Coord': array (n_points, 2) de coordenadas (x, z)
                    - 'FlCc': array (n_samples, 2) con AoA y Mach
                    - 'idx_sort': índices de orden lexicográfico para las coordenadas
                """
                coord_list = []
                norm_vector = []
                flcc_list = []
                idx_sort_list = []
                folders = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

                for folder in folders:
                    sample_files = list(glob.glob(os.path.join(folder, "*")))
                    for nsim, fname in enumerate(sample_files):
                        sim_data = FRODO.READERS.AIRFOILReader.read_dat(fname)
                            
                        x = sim_data['df_data']['x']
                        z = sim_data['df_data']['z']
                        nx = sim_data['df_data']['nx']
                        nz = sim_data['df_data']['nz']

                        coords = np.stack([x, z], axis=1)
                        norm = np.stack([nx, nz], axis=1)
                        idx_sort = np.lexsort((coords[:, 0], coords[:, 1]))
                        coord_list.append(coords[idx_sort])
                        norm_vector.append(norm[idx_sort])
                        idx_sort_list.append(idx_sort)
                        flcc_list.append([sim_data['metadata'].get('AoA', np.nan), sim_data['metadata'].get('Mach', np.nan), sim_data['metadata'].get('Re', np.nan)])

                coords_array = np.array(coord_list, dtype=float)
                norm_array = np.array(norm_vector, dtype=float)
                flcc_array = np.array(flcc_list, dtype=float)
                idx_sort_array = np.array(idx_sort_list, dtype=int)

                self.data_dict = {
                    'Coord': coords_array, # Hay varias geometrías. No podemos coger solo la primera como en CODA.
                    'Norm_vector' : norm_array,
                    'FlCc': flcc_array,
                    'idx_sort': idx_sort_array,
                }

            def extract_outputs(self):
                """
                Extrae las variables de superficie (cp, cfx, cfz, nx, nz) para todos los samples.
                Llena self.data_dict['Vars'] con arrays (n_samples, n_points) para cada variable.
                """
                integrals_list=[]
                field_list=[]
                folders = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
                Vars={}
                for folder in folders:
                    sample_files = list(glob.glob(os.path.join(folder, "*")))
                    for _, fname in enumerate(sample_files):
                        sim_data = FRODO.READERS.AIRFOILReader.read_dat(fname)
                        integrals_list.append([value for key, value in sim_data['metadata'].items() if key.startswith("C")])
                        field_one=[]
                        for column in list(sim_data['df_data'].columns):
                            if column.startswith("c"):
                                field_one.append(sim_data['df_data'][column])
                        field_list.append(field_one)

                integrals_array = np.array(integrals_list, dtype=float)
                field_array = np.array(field_list, dtype=float)
                integrals_name = [key for key in sim_data['metadata'].keys() if key.startswith("C")]
                field_name = [column for column in sim_data['df_data'].columns if column.startswith("c")]

                if 'Vars' not in self.data_dict:
                    self.data_dict['Vars'] = {}
                
                self.data_dict['Vars']['integrals'] = {}
                for i, int_name in enumerate(integrals_name):
                    self.data_dict['Vars']['integrals'][int_name]=integrals_array[:, i]

                self.data_dict['Vars']['field'] = {}
                for i, fil_name in enumerate(field_name):
                    self.data_dict['Vars']['field'][fil_name]=np.transpose(field_array[:, i, :])

            @staticmethod
            def read_dat(path_case):
                sim_data={}
                with open(path_case, 'r') as file:
                    first_line = file.readline().strip()
                    sample_number = int(first_line.split("_")[0].split("Sample")[1])
                    
                ct = 1
                sim_data['sample_number'] = sample_number
                sim_data['metadata']={}
                # Leer atributos globales
                with open(path_case, 'r') as file:
                    for line in file:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                            sim_data['metadata'][key] = value
                            ct += 1

                # Leer datos de superficie
                data = np.loadtxt(path_case, skiprows=ct+1)
                var_names = np.loadtxt(path_case, skiprows=ct, max_rows=1, dtype=str)
                sim_data['df_data'] = pd.DataFrame(data=data, columns=var_names)
                
                return sim_data
    
        class NUMPYFILEReader():
            
            def __init__(self, root_dir: str, file: Union[list[str], tuple[str], str]):

                self.root_dir = root_dir
                
                for f in file:
                    if not os.path.exists(os.path.join(root_dir, f)):
                        raise FileNotFoundError(f"File {os.path.join(root_dir, f)} not found.")
                
                if file is None:
                    raise ValueError("File mustn't be None")
                elif isinstance(file, str):
                    self.files = [file]
                elif isinstance(file, (list, tuple)) or isinstance(file, tuple):
                    if all(isinstance(f, str) for f in file):
                        self.files = file
                    else:
                        raise TypeError("Every element in 'file' must be a str path.")
                else:
                    raise TypeError("The 'file' argument must be a string or a list/tuple of strings (paths to .npy files).")

                self.sim_metadata = {}
                self.data_dict = {"inputs": {}, "outputs": {}, "aux": {}}

                self.npy_dict = {}
                for f in file:
                    self.npy_dict[f] = np.load(os.path.join(root_dir, f), allow_pickle=True).item()
                self.df_state = None

            def parse_simulation_dirs(self):
                """
                Analyses data structure of .npy file and classifies variables according their shape.
                """
                for f in self.files:
                    self.sim_metadata[f] = {}
                    self.sim_metadata[f]["path"] = os.path.join(self.root_dir, f)
                    content = np.load(self.sim_metadata[f]["path"], allow_pickle=True).item()

                    shapes = {key: content[key].shape for key in content.keys()}
                    self.sim_metadata[f]["keys"] = shapes

                    print(f"Parsed {f}")
                
            def extract_inputs(
                self,
                keys_inputs: dict,
                keys_aux: dict,
                method_to_sort: Literal["centroid", "kdtree"] = 'centroid',
                common: Union[list[str], None] = None,
                **kwargs
                ):
                """
                Extracts input and auxiliary variables from the .npy dictionary. Example of required format:
                    db.extract_inputs(
                        keys_inputs={
                            'ptos': 'db_random.npy/Airfoil',
                            'aoa': 'db_random.npy/Alpha',
                            'vel': 'db_random.npy/Vinf'
                        },
                        keys_aux={},
                        common=['ptos']
                    )

                'ptos' in keys_inputs is mandatory.
                Args:
                    keys_inputs (dict): Mapping of desired input variable names to their keys in the .npy files.
                    keys_aux (dict): Mapping of desired auxiliary variable names to their keys in the .npy files.
                    method_to_sort (str): Method to sort points. Options: 'centroid', 'kdtree', 'concave_hull' or None.
                    common (list[str] or None): List of variable names in keys_inputs that are common across all cases (e.g., geometry).
                """
                if common is None:
                    common = []

                self.data_dict["inputs"] = {}
                self.data_dict["aux"] = {}

                for alias, key_path in keys_inputs.items():
                    
                    file_key, key = key_path.split(sep='/')
                    
                    if key not in self.npy_dict[file_key]:
                        raise KeyError(f"Key '{file_key}/{key}' not found in .npy dictionary.")

                    arr = np.asarray(self.npy_dict[file_key][key])

                    if alias in common:
                        # Variable común (por ejemplo, geometría)
                        self.data_dict["inputs"][alias] = arr
                    else:
                        # Variables dependientes de casos (e.g., Alpha, Mach)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        self.data_dict["inputs"][alias] = arr
                        
                # Método para ordenar
                if method_to_sort == 'centroid' or (method_to_sort == None):
                    
                    self.data_dict["inputs"]['ptos'], self.order_ptos = SAM.Weapons.sort_by_centroid(points=self.data_dict["inputs"]['ptos'])
                    # centroid = self.data_dict["inputs"]['ptos'].mean(axis=0)
                    # shifted = self.data_dict["inputs"]['ptos'] - centroid
                    # self.order_ptos = np.argsort(np.arctan2(shifted[:, 1], shifted[:, 0]))
                
                elif method_to_sort == 'kdtree':
                    self.data_dict["inputs"]['ptos'], self.order_ptos = SAM.Weapons.sort_closed_curve_by_kdtree(
                        self.data_dict["inputs"]['ptos'], k=kwargs.get('k', 3), start_index=kwargs.get('start_index', 0), alpha=kwargs.get('alpha', 0.7)
                    )
                
                elif method_to_sort == 'concave_hull':
                    # hull_indices = SAM.Weapons.sort_profile_by_concave_hull(self.data_dict["inputs"]['ptos'], alpha=1.5)
                    self.data_dict["inputs"]['ptos'], self.order_ptos = SAM.Weapons.sort_points_by_hull_projection(self.data_dict["inputs"]['ptos'], self.data_dict["inputs"]['ptos'][hull_indices])
                    
                else:
                    raise ValueError("method_to_sort must be 'centroid', 'kdtree', 'concave_hull' or None.")
                # self.data_dict["inputs"]['ptos'] = self.data_dict["inputs"]['ptos'][self.order_ptos]
                
                for alias, key_path in keys_aux.items():
                    file_key, key = key_path.split(sep='/')
                    if key not in self.npy_dict[file_key]:
                        raise KeyError(f"Key '{key}' not found in .npy dictionary.")
                    self.data_dict["aux"][alias] = np.asarray(self.npy_dict[file_key][key][self.order_ptos])

                # Actualizar metadatos
                self.sim_metadata["keys_inputs"] = keys_inputs
                self.sim_metadata["keys_aux"] = keys_aux
                self.sim_metadata["common"] = common

                self.check_input_shapes()
                
                # df_dict = {}
                # for key, value in self.data_dict['inputs'].items():
                #     if key == 'ptos':
                #         continue
                #     df_dict[key] = value.squeeze()
                    
                # self.df_state = pd.DataFrame.from_dict(df_dict, dtype=float)
                                 
            def extract_outputs(self, keys_outputs: dict):
                """
                Extracts output variables from the .npy dictionary.
                Distinguishes between surface and field outputs based on their shape.
                Example:
                    db.extract_outputs(
                        keys_outputs={'cp': 'db_random.npy/Cp'}
                    )
                    
                Args:
                    keys_outputs (dict): Mapping of desired output variable names to their keys in the .npy files.
                """
                
                self.data_dict["outputs"] = {}
                
                shape_ref = self.data_dict['inputs'][list(self.data_dict["inputs"].keys())[0]].shape
                for alias, key_path in keys_outputs.items():
                    file_key, key = key_path.split(sep='/')
                    if key not in self.npy_dict[file_key]:
                        raise KeyError(f"Key '{file_key}/{key}' not found in .npy dictionary.")

                    arr = np.asarray(self.npy_dict[file_key][key])
                    # shape = arr.shape

                    # (n_cases, n_points) → (n_points, n_cases) → campo superficial
                    if arr.shape[0] == shape_ref[0]:
                        self.data_dict["outputs"][alias] = arr[self.order_ptos]
                    elif arr.shape[1] == shape_ref[0]:
                        self.data_dict["outputs"][alias] = arr.T[self.order_ptos]
                    else:
                        print('CASO EXTRAÑO')

                self.sim_metadata["keys_outputs"] = keys_outputs
                
            def check_input_shapes(self):
                """
                Verifica que los arrays en self.data_dict['inputs'] presentan solo dos tamaños
                distintos en su primera dimensión (número de puntos y número de casos).

                Emite un warning si se detectan más de dos tamaños distintos.
                """

                inputs = self.data_dict.get("inputs", {})
                if not inputs:
                    warnings.warn("No se encontraron datos en self.data_dict['inputs'].")
                    return None

                first_dims = {}
                for name, arr in inputs.items():
                    if isinstance(arr, np.ndarray):
                        first_dims[name] = arr.shape[0]
                    else:
                        warnings.warn(f"'{name}' no es un ndarray (tipo: {type(arr)})")

                unique_sizes = sorted(set(first_dims.values()))
                arrays_by_size = {
                    size: [name for name, s in first_dims.items() if s == size]
                    for size in unique_sizes
                }

                self.size_inputs = arrays_by_size
                # # Mensaje resumen
                # print("📏 Tamaños en la primera dimensión de los inputs:")
                # for size, names in arrays_by_size.items():
                #     print(f"  - {size:>6} → {names}")

                if len(unique_sizes) > 2:
                    warnings.warn(
                        f"{len(unique_sizes)} differents sizes were detected."
                        f"({unique_sizes}). Revisa la coherencia dimensional."
                    )
                else:
                    warnings.warn(
                        "Only one size was detected. May be some variables are missed at differents levels."
                    )

                # return {
                #     "unique_sizes": unique_sizes,
                #     "arrays_by_size": arrays_by_size,
                # }
                
        class NUMPYFILEReader_mio():
            
            def __init__(self, root_dir:str, file:str, key_geom:str):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_state = pd.DataFrame()
                self.data_dict = {}
                
                if file == None:
                    raise ValueError("File mustn't be None")
                else:
                    self.file_name = file
                    
                if key_geom == None:
                    raise ValueError("Key_geom mustn't be None")
                else:
                    self.key_geom = key_geom
                
                print(f"Reader developed for numpy files.")
                print(f"Geometry file must contain 'airfoil' in its name.\n")
            
            def parse_simulation_dirs_ant(self):
                self.sim_metadata['path'] = os.path.join(self.root_dir, self.file_name)
                content = np.load(self.sim_metadata['path'], allow_pickle=True).item()
                
                shapes = {key: content[key].shape for key in content.keys()}
                self.sim_metadata['keys'] = shapes
                unique_shapes = set(shapes.values())
                shape_dict = {}
                
                try:
                    us_geom = shapes[self.key_geom]
                except:
                    raise ValueError(f"Key {self.key_geom} not found in file {self.file_name}")
                
                for us in unique_shapes:
                    keys_with_shape = [key for key, shape in shapes.items() if shape == us]
                            
                    if us == us_geom:
                        continue
                    
                    if len(us) == 1 or us[1] == 1:
                        shape_dict['integrals'] = keys_with_shape
                        us_integrals = us
                        
                for us in unique_shapes:
                    keys_with_shape = [key for key, shape in shapes.items() if shape == us]
                    keys_filtered = [item for item in keys_with_shape if item not in shape_dict['integrals']]
                    if us == (us_integrals[0], us_geom[0]):
                        shape_dict['surface'] = keys_filtered
                        
                    elif len(us)>1 and us[0] == us_integrals[0] and us[1] > us_geom[0]:
                        shape_dict['field'] = keys_filtered
                
                self.sim_metadata['keys_integrals'] = shape_dict.get('integrals', [])
                self.sim_metadata['keys_surface'] = shape_dict.get('surface', [])
                self.sim_metadata['keys_field'] = shape_dict.get('field', [])
                        
            def make_one_branch(self, keys: Union[list[str], str], name:str):
                if set(keys).issubset(list(self.sim_metadata['keys'].keys())):
                    pass
                else:
                    raise ValueError(f"Keys {keys} not found in file {self.file_name}.")
                
                content = np.load(self.sim_metadata['path'], allow_pickle=True).item()
                
                shapes = {content[key].shape for key in keys}
                
                if len(shapes) > 1:
                    raise ValueError(f"Keys must have the same size.")
                self.data_dict[name] = {key: content[key] for key in keys}
            
            def make_data_dict(self):
                for name in ['inputs', 'outputs']:
                    pass
                
            def extract_inputs(self, keys_inputs:Union[list[str], str]):
                pass
            
            def extract_outputs(self):
                pass
            
        class NRL7301Reader_pylom():
            
            def __init__(self, root_dir:str, keys:dict, names:dict = {}, **kwargs):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_state = None
                self.data_dict = {}
                self.keys = keys

                if names == {}:
                    self.names = self.keys
                else:
                    self.names = names
                try:
                    self.common_tensors = self.kwargs.get('common_tensors', {})
                except:
                    self.common_tensors = []
                
                self.SAM = SAM
                print('Reader developed for NRL7301 DLR Database (pylom format)')

            def parse_simulation_dirs(self):
                files = list(glob.glob(os.path.join(self.root_dir, "*.h5")))
                self.files = []
                for f in files:
                    fname = os.path.basename(f).lower()
                    if any(x in fname for x in ["train", "test", "val", "valid"]):
                        self.files.append(f)
                print(f"{len(self.files)} files found.")

                # Completar sim_metadata con las características de los archivos encontrados
                self.sim_metadata = {}
                self.sim_metadata['files'] = {}
                for f in self.files:
                    with h5py.File(f, 'r') as h5file:
                        keys_in_file = []
                        h5file.visit(lambda name: keys_in_file.append(name) if isinstance(h5file.get(name, None), h5py.Dataset) else None)
                    
                    # self.sim_metadata['files'][os.path.basename(f)] = {}
                    self.sim_metadata['files'][os.path.basename(f)] = {
                        "path": f,
                        "keys": keys_in_file
                    }
                self.df_state = None  # Not used for this format

            def extract_inputs(self):
                self.data_dict = {key: {} for key in list(self.names.keys())}
                
                ayuda = {}
                for label in ['inputs', 'aux']:
                    keys = self.keys[label]
                    names = self.names[label]
                    for fname, meta in self.sim_metadata['files'].items():
                        reader = self.SAM.HDF5reader(meta["path"])
                        dic = {}
                        for key, name in zip(keys, names):
                            dic[name] = reader.load_to_tensor(key, False)
                            
                        if fname not in ayuda:
                            ayuda[fname] = {}
                        ayuda[fname][label] = dic

                # Recorremos archivos
                for fname, file_content in ayuda.items():
                    for section, section_content in file_content.items():  # aux / inputs / outputs
                        for key, tensor in section_content.items():
                            if key not in self.data_dict[section]:
                                self.data_dict[section][key] = []
                            self.data_dict[section][key].append(tensor)

                # Concatenación inteligente
                for section in self.data_dict:
                    for key, tensors in self.data_dict[section].items():
                        shapes = [t.shape for t in tensors]

                        # Caso especial: todos iguales → no concateno, me quedo con uno
                        if all(s == shapes[0] for s in shapes):
                            self.data_dict[section][key] = tensors[0]
                            continue

                        # Si la primera dimensión coincide → concatenamos en la segunda
                        if all(s[0] == shapes[0][0] for s in shapes):
                            self.data_dict[section][key] = torch.cat(tensors, dim=1)
                        else:
                            # Si no, concatenamos en la primera
                            self.data_dict[section][key] = torch.cat(tensors, dim=0)

            def extract_outputs(self):
                self.data_dict["outputs"]={} # <-- igual que en extract_inputs

                ayuda = {}
                keys = self.keys['outputs']
                names = self.names['outputs']
                for fname, meta in self.sim_metadata['files'].items():
                    reader = self.SAM.HDF5reader(meta["path"])
                    dic = {}
                    for key, name in zip(keys, names):
                        dic[name] = reader.load_to_tensor(key, False)
                    if fname not in ayuda:
                        ayuda[fname] = {}
                    ayuda[fname]['outputs'] = dic

                # Recorremos archivos
                for fname, file_content in ayuda.items():
                    for section, section_content in file_content.items():  # solo "outputs"
                        for key, tensor in section_content.items():
                            if key not in self.data_dict[section]:
                                self.data_dict[section][key] = []
                            self.data_dict[section][key].append(tensor)

                # Concatenación inteligente
                for section in ["outputs"]:  # aquí solo outputs
                    for key, tensors in self.data_dict[section].items():
                        shapes = [t.shape for t in tensors]

                        if all(s == shapes[0] for s in shapes):
                            self.data_dict[section][key] = tensors[0]
                            continue

                        if all(s[0] == shapes[0][0] for s in shapes):
                            self.data_dict[section][key] = torch.cat(tensors, dim=1)
                        else:
                            self.data_dict[section][key] = torch.cat(tensors, dim=0)

        class NRL7301Reader():
            
            def __init__(self, root_dir:str, **kwargs):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_state = None
                self.data_dict = {key: {} for key in ['inputs', 'outputs', 'aux']}
                
                self.SAM = SAM
                
            def parse_simulation_dirs(self):
                files = list(glob.glob(os.path.join(self.root_dir, "*.h5")))
                self.files = []
                for f in files:
                    fname = os.path.basename(f).lower()
                    if any(x in fname for x in ["train", "test", "val", "valid"]):
                        self.files.append(f)
                print(f"{len(self.files)} files found.")

                # Completar sim_metadata con las características de los archivos encontrados
                self.sim_metadata = {}
                self.sim_metadata['files']={}
                for f in self.files:
                    with h5py.File(f, 'r') as h5file:
                        keys_in_file = []
                        h5file.visit(lambda name: keys_in_file.append(name) if isinstance(h5file.get(name, None), h5py.Dataset) else None)
                    
                    # self.sim_metadata['files'][os.path.basename(f)] = {}
                    self.sim_metadata['files'][os.path.basename(f)] = {
                        "path": f,
                        "keys": keys_in_file
                    }
                    
                self.df_state = None  # Not used for this format
                
            def extract_inputs(self, keys_inputs:dict, keys_aux:dict, common:Union[list[str], None] = None):
                # Procesar inputs
                for key_inp, key_inp_file in keys_inputs.items():
                    list_array_inp = []
                    for file in self.sim_metadata['files'].keys():
                        path = self.sim_metadata['files'][file]['path']
                        reader = SAM.HDF5reader(file_path=path, verbose=False)
                        array_inp = reader.load_to_numpy(key_inp_file)
                        list_array_inp.append(np.expand_dims(array_inp, axis=len(array_inp.shape)))
                        
                    if key_inp in common:
                        lista_norm = [a.reshape(1, a.shape[0], -1) for a in list_array_inp]
                        apto = SAM.Backpack.same_columns(np.concatenate(lista_norm, axis=0))
                            
                        if apto:
                            self.data_dict['inputs'][key_inp] = list_array_inp[0].squeeze()
                    else:
                        self.data_dict['inputs'][key_inp]= np.concatenate(list_array_inp, axis=0).squeeze()
                            
                    # self.data_dict['inputs'][key_inp] = np.concatenate(list_array_inp, axis=1 if list_array_inp[0].ndim > 1 else 0)
                self.sim_metadata['keys_inputs'] = keys_inputs
                # Procesar aux
                for key_aux, key_aux_file in keys_aux.items():
                    list_array_aux = []
                    for file in self.sim_metadata['files'].keys():
                        path = self.sim_metadata['files'][file]['path']
                        reader = SAM.HDF5reader(file_path=path, verbose=False)
                        array_aux = reader.load_to_numpy(key_aux_file)
                        list_array_aux.append(np.expand_dims(array_aux, axis=len(array_aux.shape)))
                        
                    if key_aux in common:
                        lista_norm = [a.reshape(1, a.shape[0], -1) for a in list_array_aux]
                        apto = SAM.Backpack.same_columns(np.concatenate(lista_norm, axis=0))
                            
                        if apto:
                            self.data_dict['aux'][key_aux] = list_array_aux[0].squeeze()
                    else:
                        self.data_dict['aux'][key_aux]= np.concatenate(list_array_aux, axis=0).squeeze()
                self.sim_metadata['keys_aux'] = keys_aux
                
                self.check_input_shapes()
                
            def extract_outputs(self, keys_outputs:dict):
                key = list(self.data_dict['inputs'].keys())[0]
                tam = self.data_dict['inputs'][key].shape
                for (key_out, key_out_file) in keys_outputs.items():
                    array_out = []
                    for file in self.sim_metadata['files'].keys():
                        path = self.sim_metadata['files'][file]['path']
                        
                        reader = SAM.HDF5reader(file_path=path, verbose=False)
                        array_out.append(reader.load_to_numpy(key_out_file))
                        
                    self.data_dict['outputs'][key_out]= np.concatenate(array_out, axis=0).reshape(tam[0], -1)
                self.sim_metadata['keys_outputs'] = keys_outputs
            
            def check_input_shapes(self):
                """
                Verifica que los arrays en self.data_dict['inputs'] presentan solo dos tamaños
                distintos en su primera dimensión (número de puntos y número de casos).

                Emite un warning si se detectan más de dos tamaños distintos.

                Returns:
                    dict con:
                        - unique_sizes: lista de tamaños únicos
                        - arrays_by_size: agrupación de nombres por tamaño
                """

                inputs = self.data_dict.get("inputs", {})
                if not inputs:
                    warnings.warn("No se encontraron datos en self.data_dict['inputs'].")
                    return None

                first_dims = {}
                for name, arr in inputs.items():
                    if isinstance(arr, np.ndarray):
                        first_dims[name] = arr.shape[0]
                    else:
                        warnings.warn(f"'{name}' no es un ndarray (tipo: {type(arr)})")

                unique_sizes = sorted(set(first_dims.values()))
                arrays_by_size = {
                    size: [name for name, s in first_dims.items() if s == size]
                    for size in unique_sizes
                }

                self.size_inputs = arrays_by_size
                # # Mensaje resumen
                # print("📏 Tamaños en la primera dimensión de los inputs:")
                # for size, names in arrays_by_size.items():
                #     print(f"  - {size:>6} → {names}")

                if len(unique_sizes) > 2:
                    warnings.warn(
                        f"{len(unique_sizes)} differents sizes were detected."
                        f"({unique_sizes}). Revisa la coherencia dimensional."
                    )
                else:
                    warnings.warn(
                        "Only one size was detected. May be some variables are missed at differents levels."
                    )

                # return {
                #     "unique_sizes": unique_sizes,
                #     "arrays_by_size": arrays_by_size,
                # }
             
        class FLUENTReader():
            
            def __init__(self, root_dir:str, **kwargs):
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_state = pd.DataFrame()
                
            def parse_simulation_dirs(self):
                folders = [folder for folder in os.listdir(self.root_dir) if os.path.isdir(folder)]
                
                for folder in folders:
                    full_path = os.path.join(self.root_dir, folder)
                    self.sim_metadata[folder]={}
                    self.sim_metadata[folder]['full_path'] = full_path
                    
                    SYS_list = os.listdir(os.path.join(full_path, 'SYS'))
                    mesh_name = [f for f in SYS_list if f.endswith('.msh')]
                    
                    if len(mesh_name) == 0:
                        raise FileNotFoundError(f"No mesh file found in {os.path.join(full_path, 'SYS')}")
                    elif len(mesh_name) > 1:
                        raise ValueError(f"Multiple mesh files found in {os.path.join(full_path, 'SYS')}: {mesh_name}")
                    self.sim_metadata[folder]['mesh_name'] = mesh_name[0]
                    
                    stat_files = os.listdir(os.path.join(full_path, 'Statistics'))
                    cases_files = [f for f in stat_files if f.endswith('.cas.h5')]
                    data_files = [f for f in stat_files if f.endswith('.dat.h5')]
                    
                    if len(cases_files) != len(data_files):
                        raise ValueError(f"Number of case files and data files do not match in {os.path.join(full_path, 'Statistics')}")
                    else:
                        self.sim_metadata[folder]['n_stages'] = len(cases_files)
                    
                    #registro de report-files (son aquellos que guardan datos de series temporales)
                    outputs_files = os.listdir(os.path.join(full_path, 'Outputs'))
                    report_files = [f for f in outputs_files if f.startswith('report-') and f.endswith('.out')]
                    self.sim_metadata[folder]['report_files'] = report_files
                       
            def extract_inputs(self):
                return super().extract_inputs()
            
            def extract_outputs(self):
                return super().extract_outputs()
            
            @staticmethod
            def read_report_pressure_points(folder_path):
                with open(os.path.join(folder_path, 'Outputs', 'report-pressure-points.out')) as f:
                    f.readline()
                    f.readline()
                    names = re.findall(r'"(.*?)"', f.readline().strip('\n'))

                return pd.read_csv(os.path.join(folder_path, 'Outputs', 'report-pressure-points.out'), sep=r"\s+", engine='python', header=None, names=names, skiprows=3)
        
        class PYLOMReader():
            
            def __init__(self, root_dir:str, file: str, **kwargs):
                """
                Initialize the PYLOMReader, in order to load data from pylom datasets stored as .h5 files.
                Args:
                    root_dir (str): Root path where .h5 files are located.
                    file (Union[str, list[str], tuple[str]]): Path(s) to .h5 file(s) relative to root_dir.
                    **kwargs: Additional keyword arguments.
                """
                self.root_dir = root_dir
                self.sim_metadata = {}
                self.df_state = pd.DataFrame()
                self.file = file
                
                for f in file:
                    if not os.path.exists(os.path.join(root_dir, f)):
                        raise FileNotFoundError(f"File {os.path.join(root_dir, f)} not found.")
                
                if file is None:
                    raise ValueError("File mustn't be None")
                elif isinstance(file, str):
                    self.files = [file]
                elif isinstance(file, list) or isinstance(file, tuple):
                    if all(isinstance(f, str) for f in file):
                        self.files = file
                    else:
                        raise TypeError("Every element in 'file' must be a str path.")
                else:
                    raise TypeError("The 'file' argument must be a string or a list/tuple of strings (paths to .npy files).")

            def parse_simulation_dirs(self):
                
                self.sim_metadata = {}
                self.sim_metadata["path"] = os.path.join(self.root_dir, self.file)
                
                inicio_leer = time.perf_counter()
                content = SMEAGOL.Dataset.load(os.path.join(self.root_dir, self.file))
                fin_leer = time.perf_counter()
                print(f'Tiempo de lectura: {fin_leer - inicio_leer} segundos')
                print(content)
                self.sim_metadata['Vars'] = {}
                self.sim_metadata['Fields'] = {}
                for var in content._vardict.keys():
                    self.sim_metadata['Vars'][var] = content._vardict[var]['value'].shape

                for field in content._fieldict.keys():
                    self.sim_metadata['Fields'][field] = content._fieldict[field]['value'].shape
                    
                print(f"Parsed {self.file}")
                    
            def extract_inputs(
                self,
                keys_inputs,
                keys_aux,
                filter_by_vars,
                filter_by_fields
                ):
                
                pass
            
            def extract_outputs(self):
                return super().extract_outputs()
            
    class KEEPERS:

        class AIRFOILKeeper():

            def save_to_h5(self, db:'FRODO', save_path:str):

                with h5py.File(save_path, 'w') as h5file:
                    # Guardar metadatos de simulaciones
                    sim_metadata_group = h5file.create_group('sim_metadata')
                    for folder_key, folder_data in db.sim_metadata.items():
                        folder_group = sim_metadata_group.create_group(str(folder_key))
                        for sample_key, sample_data in folder_data.items():
                            sample_group = folder_group.create_group(str(sample_key))
                            for meta_key, meta_value in sample_data.items():
                                # Guardar solo datos escalar o listas simples
                                if isinstance(meta_value, (float, int)):
                                    sample_group.create_dataset(meta_key, data=meta_value)
                                elif isinstance(meta_value, str):
                                    # Guardar string como atributo
                                    sample_group.attrs[meta_key] = meta_value
                                elif isinstance(meta_value, list):
                                    arr = np.array(meta_value)
                                    if arr.dtype.kind in {'U', 'O'}:  # Si es string, convertir a bytes
                                        arr = arr.astype('S')
                                    sample_group.create_dataset(meta_key, data=arr)
                                elif meta_key == "available_vars":
                                    arr = np.array(meta_value)
                                    arr = arr.astype('S')
                                    sample_group.create_dataset(meta_key, data=arr)
                                elif meta_key == "path":
                                    sample_group.attrs[meta_key] = str(meta_value)

                    # Guardar datos principales
                    airfoil_group = h5file.create_group('airfoil_data')
                    for key in ['Coord', 'Norm_vector', 'FlCc', 'idx_sort']:
                        if key in db.data_dict:
                            airfoil_group.create_dataset(key, data=db.data_dict[key])

                    # Guardar variables
                    if 'Vars' in db.data_dict:
                        vars_group = airfoil_group.create_group('Vars')
                        for var_type, var_dict in db.data_dict['Vars'].items():
                            var_type_group = vars_group.create_group(var_type)
                            for var_name, arr in var_dict.items():
                                var_type_group.create_dataset(var_name, data=arr)

                print(f"Base de datos Airfoil guardada en: {save_path}")

            def save_pylom_dataset(self):
                raise NotImplementedError("save_pylom_dataset method is not implemented for AIRFOILKeeper.")

            def create_pylom_mesh(self):
                raise NotImplementedError("create_pylom_mesh method is not implemented for AIRFOILKeeper.")
            
        class NRL7301Keeper_pylom():
            
            def save_to_h5(self, save_path):
                raise NotImplementedError("save_to_h5 method is not implemented for NRL7301Keeper.")
            
            def save_pylom_dataset(self, save_path):
                # TENEMOS CONECTIVIDAD EN LOS ARCHIVOS ORIGINALES. PODEMOS HACER ESTE MÉTODO
                raise NotImplementedError("save_pylom_dataset method is not implemented for NRL7301Keeper.")
            
    class RESIDUALS:
        
        class CODAResiduals():
            
            def __init__(self, db: 'FRODO'):
                self.db = db
            
            def update_converged_state(self, threshold:float = 1e-4, exclude_residuals:Union[tuple[str], list[str]]=['MomentumYResidual',]):
                df_res = self.db.get_all_final_residuals(verbose=False, only_finished=True, load_in_metadata=False)
                cols = [c for c in list(df_res.columns) if c.endswith('norm') and all(resi not in c for resi in exclude_residuals)]
                df_params_converged = df_res[(df_res[cols] < threshold).all(axis=1)][self.db.metadata['design_vars']]
                
                self.db.df_state["key"] = list(zip(self.db.df_state[param] for param in self.db.metadata['design_vars']))
                df_params_converged["key"] = list(zip(df_params_converged[param] for param in self.db.metadata['design_vars']))

                self.db.df_state["Converged"] = self.db.df_state["key"].isin(df_params_converged["key"]).astype(int)

                self.db.df_state.drop(columns=["key"], inplace=True)
                
            @staticmethod
            def get_df_residuals_from_txt(
                case_path: str, verbose: bool = True, txt_from_end:int = 1
                ):
                """
                Parse a residual text file (-out.txt) into a pandas DataFrame.

                Args:
                    case_path (str): Path to the simulation folder.
                    verbose (bool): If True, prints warnings or file information.

                Returns:
                    pd.DataFrame: Columns include ['iters', 'cfl', 'rho_res', 'mom_res', 'energ_res'].
                """

                files = SAM.Backpack.find_files(case_path, "-out.txt", verbose=False)
                if len(files) == 0:
                    if verbose:
                        print(f"WARNING: No files found in {case_path} with the ending -out.txt")
                    return None
                else:
                    list_df = []

                    if txt_from_end:
                        if isinstance(files, list):
                            files = [files[-txt_from_end]]
                        elif isinstance(files, str):
                            files = [files]
                    for file in files:
                        if verbose:
                            print(f'Leyendo {file}')

                        with open(file, 'r') as f:
                            contenido = f.read()

                        cfl = []
                        rho_res = []
                        mom_res = []
                        energ_res = []
                        iters = []
                        cont = 0

                        regex_residuos = re.compile(
                            r"Iteration (\d+):\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
                        )

                        for line in contenido.splitlines():
                            match_res = regex_residuos.search(line)
                            if match_res:
                                iters.append(cont)
                                cont += 1
                                cfl.append(float(match_res.group(2)))
                                rho_res.append(float(match_res.group(3)))
                                mom_res.append(float(match_res.group(4)))
                                energ_res.append(float(match_res.group(5)))

                        list_df.append(
                            pd.DataFrame({
                                "iters": iters,
                                "cfl": cfl,
                                "rho_res": rho_res,
                                "mom_res": mom_res,
                                "energ_res": energ_res
                            })
                        )
                    df = pd.concat(list_df, axis=0, ignore_index=True, )
                return df
            
            def get_df_metrics(
                self,
                var_metrics: Union[str, list[str], tuple[str]],
                iter_var: int = 1000,
                save: bool = False,
                ):

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
                        verbose=False,
                        stage=[stage,],
                        only_finished=False,
                        load_in_metadata=False
                    ).copy()

                    df_finals.columns = df_finals.columns.astype(str).str.lower()


                    rename_dict = {
                        col: f"{col}_stage{stage}"
                        for col in df_finals.columns
                        if col not in design_vars
                    }

                    df_finals = df_finals.rename(columns=rename_dict)

                    df_post = df_post.merge(
                        df_finals,
                        on=design_vars,
                        # how="left"
                    )

                for irow in range(len(db.df_state)):

                    case_name = db.case_per_idx(irow)
                    output_path = os.path.join(db.root_dir, 'outputs', case_name)

                    if not os.path.exists(output_path):
                        continue

                    files_list = [
                        f for f in os.listdir(output_path)
                        if f.endswith("_monitors_wall_boundary_integrals.dat")
                    ]

                    if len(files_list) == 0:
                        continue

                    for fname in files_list:

                        match = re.search(r"output_(\d+)__", fname)
                        if not match:
                            continue

                        stage = int(match.group(1))

                        full_path = os.path.join(output_path, fname)

                        df_int = SAM.Backpack.get_df_from_csv(files_list=[full_path])

                        if not all(v in df_int.columns for v in var_metrics):
                            continue

                        df_tail = df_int[var_metrics].tail(iter_var)

                        mean_series = df_tail.mean()
                        var_series = df_tail.var()

                        for v in var_metrics:
                            df_post.loc[irow, f"{v}_mean_stage{stage}"] = mean_series.get(v, np.nan)
                            df_post.loc[irow, f"{v}_var_stage{stage}"] = var_series.get(v, np.nan)

                df_post = df_post.sort_values(by=self.db.metadata['design_vars'][0], ignore_index=True, axis=0).reset_index(drop=True)

                if save:
                    df_post.to_csv(
                        os.path.join(db.root_dir, 'metadata', 'df_post.csv'),
                    )

                return df_post

            def get_all_final_residuals(
                self,
                stage:Union[list, tuple, 'all'] = 'all',
                verbose:bool = False,
                only_finished:bool = True,
                load_in_metadata:bool = True
                ):
                
                df_all=[]
                
                folder_fmt = self.db.metadata.get('folder_fmt', r"aoa_-?\d+\.\d+_mach_-?\d+\.\d+")
                pattern = SAM.Backpack.folder_fmt_to_pattern(folder_fmt)
                
                # design_vars = self.db.metadata['design_vars'] #[param1, param2, ...]
                
                for folder in self.db.sim_metadata.keys():
                    if re.match(pattern, folder):
                        # params = re.findall(r"-?\d+\.\d+", folder)
                        # #nombres de los params en self.metadata['design_vars']
                        # params_float = list(map(float, params))
                        
                        params_float = self.db.metadata['df_cases'][self.db.metadata['design_vars']][self.db.metadata['df_cases']['folder']== folder].values.squeeze().tolist()
                        stages_done = len(self.db.sim_metadata[folder]['stages'].keys())
                        if only_finished and stages_done < self.db.metadata['num_stages']:
                            if verbose:
                                print(f'Simulation {folder} not finished. Stages done: {stages_done}/{self.db.num_stages}')
                            continue
                        
                        df_one = self.get_df_residuals_from_case(case_name=folder, stage=stage)
                        if df_one is None:
                            if verbose:
                                print(f"Folder {folder} without results")
                            res=[float('nan')] * 26
                        else:
                            res = df_one.tail(1).values[0,:].reshape(1, -1)
                        
                        fila = np.concatenate((res, np.expand_dims(np.array(params_float), axis=0)), axis=1, dtype=np.float64)
                        df_all.append(fila)

                names = list(df_one.columns) + self.db.metadata['design_vars']

                df_final_residuals = pd.DataFrame(np.vstack(df_all), columns = names)

                if load_in_metadata:
                    os.makedirs(os.path.join(self.db.root_dir, 'metadata'), exist_ok=True)
                    df_final_residuals.to_csv(os.path.join(self.db.root_dir, 'metadata', 'all_final_residuals.csv'), index=False)
                    
                return df_final_residuals
                                   
            def integrals_convergence_criteria(
                self, iterations_back:int = 1000,
                only_finished:bool = False,
                only_converged:bool = False,
                columns_to_remove:Union[list[str], tuple[str]]=['total_iter', 'Iteration', 'Time'],
                mode:Literal['2D', '3D'] = '3D',
                plot:bool = False,
                verbose:bool = False,
                **kwargs
                ):
                """
                Analyze the convergence of integral variables based on the last iterations of the simulations, and optionally plot the results.
                Args:
                    iterations_back (int): Number of last iterations to consider for convergence analysis.
                    only_finished (bool): If True, only considers simulations that have completed all stages.
                    only_converged (bool): If True, only considers simulations that meet the convergence criteria based on final residuals.
                    columns_to_remove (list[str] or tuple[str]): Columns to exclude from the analysis when calculating means and standard deviations.
                    mode (str): '2D' for 2D scatter plots, '3D' for 3D scatter plots with surfaces.
                    plot (bool): If True, generates the specified plots.
                    verbose (bool): If True, prints detailed information during the process.
                
                Returns:
                    result_mean (pd.DataFrame): DataFrame containing mean values of integral variables for each case.
                    result_std (pd.DataFrame): DataFrame containing standard deviation of integral variables for each case.
                """
                all_means = []
                all_std = []
                
                if only_converged:
                    if not only_finished:
                        print('WARNING: To get only converged cases, only_finished must be activated.')
                        only_finished=True
                        
                df_res = self.get_all_final_residuals(verbose=False, only_finished=only_finished, load_in_metadata=False)
                cols = [c for c in list(df_res.columns) if c.endswith('norm') and ('MomentumYResidual' not in c)]
                # columnas_finales = [cols.remove(col) for col in columns_to_remove]
                
                if only_converged:
                    df_filtered = df_res[(df_res[cols] < 1e-4).all(axis=1)][self.db.metadata['design_vars']]
                else:
                    df_filtered = df_res[self.db.metadata['design_vars']]
                
                folder_fmt = self.db.metadata.get('folder_fmt', r"aoa_-?\d+\.\d+_mach_-?\d+\.\d+")
                pattern = SAM.Backpack.folder_fmt_to_pattern(folder_fmt)
                
                for folder_name, dic in self.db.sim_metadata.items():
                    if re.match(pattern, folder_name):
                        params = re.findall(r"-?\d+\.\d+", folder_name)
                        #nombres de los params en self.metadata['design_vars']
                        valores = list(map(float, params))
                        stages_done = len(self.db.sim_metadata[folder_name]['stages'].keys())
                        if only_finished and stages_done < self.db.metadata['num_stages']: #Ha terminado?
                            if verbose:
                                print(f'Simulation {folder_name} not finished. Stages done: {stages_done}/{self.db.num_stages}')
                            continue

                        mask = np.ones(len(df_filtered), dtype=bool)
                        for val, var in zip(valores, self.db.metadata['design_vars']):
                            mask &= np.isclose(df_filtered[var].values, val)

                        if mask.any():  # Ha convergido

                            df_integrals_case = SAM.Backpack.get_df_from_csv(
                                files_list = SAM.Backpack.find_files(
                                    path = dic['path'], file_end = '_wall_boundary_integrals.dat', verbose = False
                                    )
                            )
                            
                            last_values = df_integrals_case.tail(iterations_back).drop(columns_to_remove, axis=1)
                            
                            mean_values = list(last_values.mean().values)
                            std_values = list(last_values.std().values)
                            
                            mean_complete = valores + mean_values
                            std_complete = valores + std_values
                            
                            all_means.append(mean_complete)
                            all_std.append(std_complete)
                        else:
                            if verbose:
                                print(f'Simulation {folder_name} does not meet convergence criteria')
                            continue
                
                columns = self.db.metadata['design_vars'] + list(df_integrals_case.drop(columns_to_remove, axis=1).columns)

                result_mean = pd.DataFrame(np.array(all_means), columns = columns)
                result_std = pd.DataFrame(np.array(all_std), columns = columns)
                
                if plot and len(self.db.metadata['design_vars']) == 2:
                    param1, param2 = self.db.metadata['design_vars']
                    if mode == '2D':
                        fig, ax = plt.subplots(2, len(columns[2:]), figsize=kwargs.get('figsize', (5*len(columns[2:]), 8)))
                        ax = ax.flatten()
                        for i, col in enumerate(columns[2:]):
                            # Scatter de medias
                            sc1 = ax[i].scatter(result_mean[param1], result_mean[param2], c=result_mean[col], cmap='viridis', s=100, edgecolors='k')
                            plt.colorbar(sc1, ax=ax[i], label=f'Mean {col}')
                            ax[i].set_xlabel(param1)
                            ax[i].set_ylabel(param2)
                            ax[i].set_title(f'Mean {col}')
                            ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
                            # Scatter de desviaciones (en logaritmo)
                            norm = mcolors.LogNorm()
                            sc2 = ax[i+len(columns[2:])].scatter(result_std[param1], result_std[param2], c=result_std[col], cmap='plasma', s=100, edgecolors='k', norm=norm)
                            plt.colorbar(sc2, ax=ax[i+len(columns[2:])], label=f'Std Dev {col}')
                            ax[i+len(columns[2:])].set_xlabel(param1)
                            ax[i+len(columns[2:])].set_ylabel(param2)
                            ax[i+len(columns[2:])].set_title(f'Std Dev {col}')
                            ax[i+len(columns[2:])].grid(True, which='both', linestyle='--', linewidth=0.5)
                            
                            
                    elif mode == '3D':
                        for col in columns[2:]:
                            # Figura base
                            fig = go.Figure()

                            # === Scatter de medias ===
                            fig.add_trace(go.Scatter3d(
                                x=result_mean[param1],
                                y=result_mean[param2],
                                z=result_mean[col],
                                mode='markers',
                                name=f'Mean {col}',
                                marker=dict(size=5, color='blue', symbol='circle'),
                                opacity=0.8
                            ))

                            # === Superficie interpolada para las medias ===
                            # Crear grilla de interpolación
                            aoa_grid = np.linspace(result_mean[param1].min(), result_mean[param1].max(), 50)
                            mach_grid = np.linspace(result_mean[param2].min(), result_mean[param2].max(), 50)
                            AOA, MACH = np.meshgrid(aoa_grid, mach_grid)
                            
                            # Interpolación con griddata
                            from scipy.interpolate import griddata
                            Z_mean = griddata(
                                (result_mean[param1], result_mean[param2]),
                                result_mean[col],
                                (AOA, MACH),
                                method='cubic'
                            )

                            fig.add_trace(go.Surface(
                                x=AOA, y=MACH, z=Z_mean,
                                colorscale='Blues',
                                opacity=0.5,
                                showscale=False,
                                name=f'Mean Surface {col}'
                            ))

                            # === Scatter de desviaciones ===
                            fig.add_trace(go.Scatter3d(
                                x=result_std[param1],
                                y=result_std[param2],
                                z=result_std[col],
                                mode='markers',
                                name=f'Std Dev {col}',
                                marker=dict(size=5, color='red', symbol='diamond'),
                                opacity=0.8
                            ))

                            # === Superficie interpolada para desviaciones ===
                            Z_std = griddata(
                                (result_std[param1], result_std[param2]),
                                result_std[col],
                                (AOA, MACH),
                                method='cubic'
                            )

                            fig.add_trace(go.Surface(
                                x=AOA, y=MACH, z=Z_std,
                                colorscale='Reds',
                                opacity=0.5,
                                showscale=False,
                                name=f'Std Surface {col}'
                            ))

                            # === Configuración final ===
                            fig.update_layout(
                                title=f'Integral Variable: {col}',
                                scene=dict(
                                    xaxis_title=param1,
                                    yaxis_title=param2,
                                    zaxis_title=col,
                                ),
                                legend=dict(
                                    x=0.02, y=0.98,
                                    bgcolor='rgba(255,255,255,0.6)',
                                    bordercolor='rgba(0,0,0,0.3)',
                                    borderwidth=1
                                ),
                                margin=dict(l=0, r=0, b=0, t=50),
                                width=kwargs.get('width', 1200),
                                height=kwargs.get('height', 800),
                            )

                            fig.show()
                        
                return result_mean, result_std
            
            def get_df_residuals_from_case(
                self,
                case_name:str=None,
                case_idx:Union[int, None]=None,
                stage:Union[list, tuple, 'all'] = 'all',
                verbose:bool = False
                ):
                """
                Devuelve un DataFrame con residuos absolutos, normalizados y escalados
                para todas las etapas del caso indicado por case_idx.
                """
                if case_name == None:
                    if case_idx == None:
                        raise ValueError("You must provide either case_name or case_idx.")
                    else:
                        case_path = self.db.sim_metadata[self.db.case_per_idx(case_idx)]['path']
                        if stage == 'all':
                            stages = list(self.db.sim_metadata[self.db.case_per_idx(case_idx)]['stages'].keys())
                        elif stage == None:
                            raise ValueError("Stage cannot be None.")
                        else:
                            stages = stage
                else:
                    case_path = self.db.sim_metadata[case_name]['path']
                    if stage == 'all':
                        stages = list(self.db.sim_metadata[case_name]['stages'].keys())
                    elif stage == None:
                        raise ValueError("Stage cannot be None.")
                    else:
                        stages = stage

                dfs_stage = []  # para concatenar luego todas las etapas
                for s in stages:
                    # --- Archivos ---
                    file_time = os.path.join(case_path, f"output_{s}__monitors_TimeIntegration.dat")
                    file_init = os.path.join(case_path, f"output_{s}__monitors_stage{s}InitialResidual.dat")
                    file_cfl  = os.path.join(case_path, f"output_{s}__monitors_CFLRamp.dat")

                    # --- Lectura de datos ---
                    df_abs_res = SAM.Backpack.get_df_from_csv([file_time])
                    df_initial = SAM.Backpack.get_df_from_csv([file_init])
                    df_cfl     = SAM.Backpack.get_df_from_csv([file_cfl])

                    # --- Selección de valores iniciales (primera fila) ---
                    r0_vals = df_initial.iloc[0].to_dict()

                    # --- Escalas (Reference...) ---
                    ref_cols = [c for c in df_cfl.columns if c.startswith("SERReference")]
                    ref_vals = df_cfl[ref_cols] # .iloc[0].to_dict()
                    # ref_vals_clean = {re.sub(r"^SERReference", "", k): v for k, v in ref_vals.items()}
                    ref_vals.columns=list(re.sub(r"^SERReference", "", k) for k, _ in ref_vals.items())

                    # --- Crear DataFrame salida ---
                    df_stage = df_abs_res.copy()
                    for col in df_abs_res.columns:
                        if "Residual" in col:
                            base = col.replace("Residual", "")
                            r_abs = df_abs_res[col]
                            r0 = r0_vals.get(col, None)
                            # S  = ref_vals_clean.get(col, None)
                            S = ref_vals[col]

                            df_stage[f"{col}_norm"] = r_abs / r0
                            df_stage[f"{col}_scaled"] = r_abs / S

                    df_stage["stage"] = s
                    dfs_stage.append(df_stage)

                    if verbose:
                        print(f"[INFO] Stage {s}: {len(df_stage)} iteraciones procesadas")
                        print(f"       Variables detectadas: {[c for c in df_stage.columns if 'Residual' in c]}")

                # --- Concatenar todas las etapas ---
                df_all = pd.concat(dfs_stage, ignore_index=True)
                df_all['total_iterations']= np.array(range(len(df_all)))
                return df_all
            
            def plot_residuals_from_case(
                self,
                case_name: str = None,
                case_idx: Union[int, None] = None,
                stage: Union[list, tuple, 'all'] = 'all',
                mode: Literal['absolute', 'norm', 'scaled'] = 'scaled',
                save_dir: Union[None, str] = None,
                verbose:bool = False,
                **kwargs
                ):    
                
                if case_name == None:
                    if case_idx == None:
                        raise ValueError("You must provide either case_name or case_idx.")
                    else:
                        case_name = self.db.case_per_idx(case_idx)
                if stage == 'all':
                    stages = list(self.db.sim_metadata[case_name]['stages'].keys())
                elif stage == None:
                    raise ValueError("Stage cannot be None.")
                else:
                    stages = stage
                
                df_res = self.get_df_residuals_from_case(case_name = case_name, stage=stages, verbose=verbose)    
                    
                _, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))

                columns_all = [col for col in df_res.columns if 'Residual' in col and 'MomentumYResidual' not in col]
                columns = [col for col in columns_all if mode in col]
                
                colors = cm.tab10.colors[:len(columns)]
                for ycol, color in zip(columns, colors):
                    df_res.plot(
                        x='total_iterations',
                        y=ycol,
                        s=3,
                        kind='scatter',
                        ax=ax,
                        label=ycol.replace("Residual_" + mode,''),
                        color=color,
                        grid=True,
                        logy=True
                    )

                ax.set_title(f'Case {case_name}')
                ax.set_ylim((1e-8, 1e2))
                ax.set_ylabel(f'Residual ({mode})')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
                
                divider = make_axes_locatable(ax)
                ax_cfl = divider.append_axes("bottom", size="35%", pad=0.1, sharex=ax)
                
                ax_cfl.scatter(
                    x=df_res['total_iterations'],
                    y=df_res['CFL'],
                    color='black',
                    s=1.5
                )
                
                ax_cfl.set_ylabel("CFL")
                ax_cfl.set_yscale('log')
                ax_cfl.set_xlabel("Iterations")
                ax_cfl.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)
                plt.show()
                
            def plot_all_final_residuals(
                self,
                save_dir: Union[None, str] = None,
                mode: Literal['absolute', 'norm', 'scaled'] = 'scaled',
                stage: Union[tuple, list, 'all'] = 'all',
                # xlabel:str = 'Iterations',
                only_finished: bool = False,
                print_non_converged: bool = False,
                activate_idx: bool = True,
                ncols:int = 2,
                lim_converged:float = 1e-5,
                **kwargs
                ):
                """
                Plot scatter maps of final residuals for all simulations.

                Args:
                    save_dir (str or None): If provided, saves the plot in this folder instead of displaying it.
                    mode (Literal["scaled", "normalized"]): Determines if residuals are scaled or normalized.
                    print_non_converged (bool): If True, prints and saves non-converged cases according to specific values (1e-6 in scaled case).

                Returns:
                    None. Displays or saves a multi-panel scatter plot of residual values.
                """
                
                df_finals = self.get_all_final_residuals(verbose=False, stage = stage, only_finished=only_finished, load_in_metadata=False)
                # if xlabel not in list(df_finals.columns):
                #     raise ValueError('xlabel must be one of the columns in the final residuals dataframe.')

                columns_all = [col for col in df_finals.columns if 'Residual' in col and 'MomentumYResidual' not in col and 'TurbulentSANuTilde' not in col]
                columns = [col for col in columns_all if mode in col]
                # ncols = ncols
                nrows = int(np.ceil(len(columns) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), constrained_layout=True)

                axes = axes.flatten()
                converged_mask = (df_finals[columns].lt(lim_converged)).all(axis=1)
                
                norm = mcolors.LogNorm(vmin=lim_converged, vmax=1e0)
                color = kwargs.get('cmap', 'summer')
                # cmap = 'summer'  # colormap para residuos
                design_vars = self.db.metadata['design_vars']
                # for var in design_vars:
                #     # Si alguna de las vars es constante en tood df_finals, omitirla de los ejes
                #     if df_finals[var].nunique() <= 1:
                #         print(f"Variable {var} is constant across all cases. It will be omitted from the axes.")
                #         df_finals = df_finals.drop(columns=[var])
                design_vars_filtered = [var for var in design_vars if df_finals[var].nunique() > 1]
                if len(design_vars_filtered) == 2:
                    for i, col in enumerate(columns):
                        x = df_finals[design_vars_filtered[0]]
                        y = df_finals[design_vars_filtered[1]]
                        c = df_finals[col]
                        
                        sc_nc = axes[i].scatter(
                            x[~converged_mask],
                            y[~converged_mask],
                            c=c[~converged_mask],
                            cmap=color,
                            norm=norm,
                            s=60,
                            edgecolor='k',
                            label= 'Non-converged',
                        )
                        sc_c = axes[i].scatter(
                            x[converged_mask],
                            y[converged_mask],
                            c=c[converged_mask],
                            cmap=color,
                            norm=norm,
                            s=60,
                            marker='*',
                            linewidth=1.5,
                            label='Converged'
                        )
                        
                        if activate_idx:
                            for p in df_finals[design_vars_filtered].values:
                                idx = np.where((self.db.df_state.iloc[:,0] == p[0]) & (self.db.df_state.iloc[:,1] == p[1]))[0][0]
                                axes[i].annotate(f"{idx}", (p[0], p[1]),  textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)
                        # Configuración de ejes
                        axes[i].set_title(f'{col}')
                        axes[i].set_xlabel('AoA')
                        axes[i].set_ylabel('Mach')
                        
                        #Colorbar
                        cbar = fig.colorbar(sc_nc, ax=axes[i])
                        cbar.ax.set_title(f'Residual Stage {stage}')
                        
                    # Agregar leyenda única fuera del área de los plots
                    handles, labels = axes[0].get_legend_handles_labels()
                    fig.legend(
                        handles,
                        labels,
                        loc='lower center',
                        frameon=False,
                        ncols=2
                    )
                
                # Guardar o mostrar
                if save_dir:
                    fig.savefig(os.path.join(save_dir, "residuals_all_cases.png"), dpi=150, bbox_inches='tight')
                else:
                    plt.show()

                # Mostrar casos no convergidos
                if print_non_converged:
                    print("Non-converged cases:")
                    df_finals[columns][~converged_mask].to_csv(os.path.join(self.db.root_dir, 'metadata', 'non_converged_cases.csv'), index=False)
                    for i, row in df_finals[~converged_mask].iterrows():
                        residuals_exp = " ".join(f"{val:.2E}" for val in row[3:].values)
                        print(
                            f"Case {i}: AoA: {row['aoa']:.4f}, Mach: {row['mach']:.4f}, Residuals: {residuals_exp}"
                        )
            
            def plot_state_calculation(self, num_stages: int = 1, txt_from_end: int = 1, figsize: tuple = None):
                data_to_plot = []
                for name, case in self.db.sim_metadata.items():
                    if len(case['stages'].keys()) == num_stages:
                        path = case['path']
                        df = FRODO.RESIDUALS.CODAResiduals.get_df_residuals_from_txt(
                            case_path=path, verbose=False, txt_from_end=txt_from_end
                        )
                        if df is not None:
                            data_to_plot.append((name, df))
                        else:
                            print(f'\tWARNING: Case {name} has not started yet. Skipping to next.\n')
                num_plots = len(data_to_plot)
                if num_plots == 0:
                    print("No se encontraron datos para los criterios especificados.")
                    return

                nrows = (num_plots + 1) // 2
                ncols = 2 if num_plots > 1 else 1
                
                # Aumentamos un poco el alto por defecto ya que ahora cada bloque tiene dos gráficos
                if figsize is None:
                    figsize = (15, 6 * nrows)

                fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
                ax_flat = ax.flatten()

                for i, (name, df_txt) in enumerate(data_to_plot):
                    current_ax = ax_flat[i]
                    
                    # --- Gráfico de Residuos (Superior) ---
                    for res, color in zip(df_txt.columns.to_list()[2:], ['blue', 'orange', 'green']):
                        current_ax.scatter(
                            x=df_txt['iters'],
                            y=df_txt[res],
                            color=color,
                            label=res,
                            s=1.5
                        )
                    
                    current_ax.set_yscale('log')
                    current_ax.set_title(name, fontsize=12)
                    current_ax.set_ylabel("Residuals")
                    current_ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)
                    current_ax.legend(loc='upper right', fontsize='small', markerscale=4)
                    
                    # Quitamos los labels del eje X del gráfico superior para que no se solapen
                    current_ax.tick_params(labelbottom=False)

                    # --- Crear eje para CFL (Inferior) ---
                    # Dividimos el axis actual para crear uno debajo que comparta el eje X
                    divider = make_axes_locatable(current_ax)
                    ax_cfl = divider.append_axes("bottom", size="35%", pad=0.1, sharex=current_ax)

                    # Gráfico de CFL
                    if 'cfl' in df_txt.columns:
                        ax_cfl.scatter(
                            x=df_txt['iters'],
                            y=df_txt['cfl'],
                            color='black',
                            s=1.5
                        )
                    
                    ax_cfl.set_ylabel("CFL")
                    ax_cfl.set_xlabel("Iterations")
                    ax_cfl.set_yscale('log')
                    ax_cfl.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.3)

                # Ocultar huecos vacíos si el número de casos es impar
                for j in range(i + 1, len(ax_flat)):
                    ax_flat[j].axis('off')

                plt.tight_layout()
                plt.show()
                  
    class SETS:
        
        class CODASets():
            
            def __init__(self, db:'FRODO'):
                self.db = db
                
            def create_jset(
                self,
                stage:str,
                id_group:str,
                sol: Union[list[int], tuple[int], int, 'all'] = 'all',
                idx_flcc: Union[list[int], tuple[int], 'all']='all',
                save_path: Union[str, None] = None,
                verbose: bool = False
                ):
                
                key_group = f'CADGroup_{id_group}'
                data_dict = self.db.data_dict[key_group]
                # --- Inputs ---

                if idx_flcc == 'all':
                    idx_flcc = list(range(data_dict['FlCc'].shape[0]))
                    
                tensor_ptos = data_dict['Coord']
                tensor_flcc = data_dict['FlCc'][idx_flcc]
                
                if 'Aux' in data_dict.keys():
                    # --- Aux ---
                    tensors_aux = [data_dict['Aux'][name][:, idx_flcc] for name in data_dict['Aux'].keys()]
                else:
                    tensors_aux = None
                    
                # --- Outputs ---
                if sol == 'all':
                    sol_num = list(range(len(data_dict['Vars'][stage].keys())))
                elif isinstance(sol, int):
                    sol_num = [sol]
                else:
                    sol_num = sol
                    
                tensors_out = []
                for i, (name, arr) in enumerate(data_dict['Vars'][stage].items()):
                    if i in sol_num:
                        print(name)
                        if len(arr.shape) == 2: #(nptos, ncasos)
                            tensors_out.append(arr[:, idx_flcc])
                        elif len(arr.shape) == 3:
                            print(f"WARNING: Variable {name} in data_dict has be described as a vector. It must be scalar.")
                
                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, verbose=verbose
                )
                # --- Guardar si hace falta ---
                if save_path:
                    if save_path.endswith('.h5'):
                        with h5py.File(save_path, "w") as h5file:
                            h5file.create_dataset("tensor", data=result['tensor'].numpy())
                            h5file.create_dataset("scaled", data=result['scaled'].numpy())
                            h5file.create_dataset("mins", data=result['mins'].numpy())
                            h5file.create_dataset("maxs", data=result['maxs'].numpy())

                    elif save_path.endswith('.pt'):
                        torch.save(obj=result, f=save_path)
                    elif save_path.endswith('.npy'):
                        np.save(file=save_path, arr = result, allow_pickle=True)
                    else:
                        raise NameError('save_path extension not supported. Please choose between .pt, .npy or .h5.')
                    
                    if verbose:
                        print(f"Jset saved in {save_path}\n")
                        
                self.db.jset = result
                columns = []
                
                if data_dict['Coord'].shape[1] == 2:
                    columns.extend(['x', 'z'])
                    
                elif data_dict['Coord'].shape[1] == 3:
                    columns.extend(['x', 'y', 'z'])
                    
                else:
                    raise ValueError('Error in coord array shape. Check FRODO.SETS.CODASets.create_jset() or ask ChatGPT.')
                
                columns.extend(self.db.metadata['design_vars'])
                
                if 'Aux' in data_dict.keys():
                    columns.extend([name for name in data_dict['Aux'].keys()])
                
                
                # columns.extend([key for key in list(data_dict['Vars'][stage].keys())[sol_num]]) #probar de otra forma, que esta no funciona
                columns.extend([name for i, name in enumerate(data_dict['Vars'][stage].keys()) if i in sol_num])
                
                self.db.df_data = pd.DataFrame(data = result['tensor'].numpy(), columns = columns)
                
                if verbose:
                    print(f'Loaded:\n {columns}')
                    print("\nJset loaded in db.jset\n")
                    print("\nDataframe with main tensor loaded in db.df_data\n")
            
            def create_pylom_mesh(
                self, id_groups:Union[int, tuple[int]]
                ):
                """
                Create pyLOM Mesh objects from stored CADGroup data.

                Args:
                id_groups (tuple): IDs or tuple-combinations of IDs to convert into pyLOM Mesh objects.

                Returns:
                    list[pyLOM.Mesh]: List of pyLOM Mesh objects, one per requested CADGroup or combination.
                """

                mesh_list = []
                for id in id_groups:
                    if isinstance(id, tuple):
                        ids_to_combine = id
                        key_suffix = "_".join(map(str, ids_to_combine))
                    elif isinstance(id, int):
                        ids_to_combine = (id,)  # convertir a tupla para uniformidad
                        key_suffix = str(id)
                    else:
                        raise TypeError("ID format unknowed")

                    key = f"CADGroup_{key_suffix}"
                    
                    xyz = self.db.data_dict[key]["Coord"]
                    conec = self.db.data_dict[key]["Conec"]
                    
                    #pylom tiene distinta nomenclatura para el tipo de elementos
                    eltype = np.array(self.db.data_dict[key]["eltype"][0,:], copy=True)
                    eltype[eltype == 5] = 2
                    eltype[eltype == 9] = 3
                    #---
                    
                    ptable = SMEAGOL.PartitionTable.new(1,conec[0,:,:].shape[0],xyz.shape[0])
                    
                    mesh = SMEAGOL.Mesh('UNSTRUCT',xyz,conec,eltype, self.db.data_dict[key]["cellOrder"][0,:], self.db.data_dict[key]["pointOrder"][0,:],ptable)
                    mesh_list.append(mesh)
                    print(mesh)
                
                return mesh_list
            
            def create_NN_pylom(
                self, id_groups:Union[int, tuple[str], list[str]],
                stage:int,
                idx_to_print:Union[int, list[int], 'all'] = 'all',
                external_vars:Union[dict, None] = None,
                save_path:Union[bool, str]=False
                
                ):
                """
                Create pyLOM Dataset objects combining mesh and simulation variables.

                Args:
                    db (FRODO): FRODO instance containing data_dict with extracted inputs/outputs.
                    id_groups (tuple): IDs or tuple-combinations of IDs to convert into pyLOM Datasets.
                    save_path (Union[bool, str]): If True, saves datasets to a file; if str, saves to that path; if False, does not save.

                Returns:
                    list[pyLOM.Dataset]: List of pyLOM Dataset objects ready for pyLOM processing/export.
                """
                
                if self.db.data_dict == {}:
                    raise AttributeError("FRODO instance must have a 'data_dict' attribute to use CODASets. Please run extract_inputs() and extract_outputs() method first.")
                
                d_list = []
                if isinstance(id_groups, int):
                    id_groups = [id_groups]
                    
                for id in id_groups:
                    key = f"CADGroup_{id}" #f"CADGroup_{key_suffix}"
                    
                    idx_sort = self.db.data_dict[key]['idx_sort'] # (nstages, ncasos, nptos) 
                    xyz = self.db.data_dict[key]["Coord"]
                    conec = self.db.data_dict[key]["Conec"]
                
                    ptable = SMEAGOL.PartitionTable.new(1,conec.shape[0],xyz.shape[0])

                    fields = list(self.db.data_dict[key]['Vars'][str(stage)].keys())
                    fields.remove('GlobalNumber')
                    fields.remove('CADGroupID')
                    
                    if idx_to_print == 'all':
                        idx_to_print = list(range(self.db.data_dict[key]["FlCc"].shape[0]))

                    elif isinstance(idx_to_print, int):
                        idx_to_print = [idx_to_print]
                        
                    max_cases = self.db.data_dict[key]["FlCc"].shape[0]
                    if any(i >= max_cases for i in idx_to_print):
                        raise IndexError("idx_to_print contains indices out of range.")
                    
                    eltype = self.db.data_dict[key]["eltype"]
                    cell_order = self.db.data_dict[key]["cellOrder"]
                    
                    eltype[eltype == 5] = 2
                    eltype[eltype == 9] = 3
                    
                    if external_vars is None:
                        param_dict = {}

                        for parameter in self.db.metadata['design_vars']:
                            param_dict[parameter] = {
                                'idim': 0,
                                'value': self.db.df_state[parameter].iloc[idx_to_print].values
                            }
                    else:
                        param_dict = external_vars
                        
                        for parameter, content in param_dict.items():
                            value = content['value']
                            
                            if idx_to_print is not None:
                                idx = np.asarray(idx_to_print)
                                if idx.max() >= len(value):
                                    raise IndexError(
                                        f"idx_to_print contiene índices fuera de rango. "
                                        f"Max idx: {idx.max()}, tamaño: {len(value)}"
                                    )
                                content['value'] = value[idx]
                        
                    field_dict = {}

                    for f in fields:
                        var_array = self.db.data_dict[key]['Vars'][str(stage)][f]

                        if len(var_array.shape) == 2: #escalar

                            value = var_array[:, idx_to_print]#[idx_points, cols]
                            field_dict[f] = {
                                'ndim': 1,
                                'value': value
                            }

                        elif len(var_array.shape) == 3: # vectorial
                            raise TypeError('Sin hacer')

                    
                        
                    d = SMEAGOL.Dataset(
                        xyz=xyz,
                        ptable=ptable,
                        order=cell_order,
                        point=True,
                        vars = param_dict,
                        **field_dict
                    )
                    print('DONE',flush=True)
                    if save_path:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        d.save(os.path.join(save_path, f"{key}_stage_{stage}.h5"))
                        print(f'Dataset saved to {os.path.join(save_path, f"{key}_stage_{stage}.h5")}')
                        
                    d_list.append(d)
                    
                return d_list
            
            def add_to_data_dict(
                self, arr:np.array, id_group:str, array_name:str
                ):
                data = self.db.data_dict.copy()
                
                if 'Aux' not in data[f'CADGroup_{id_group}'].keys():
                    data[f'CADGroup_{id_group}']['Aux'] = {}
                
                data[f'CADGroup_{id_group}']['Aux'].update({array_name: arr})
                    
                self.db.data_dict = data
            
            def change_order_coord(
                self, id_group:str,
                new_order: Union[str, list[int], tuple[int]],
                new_nodes_order: Union[None, list[int], tuple[int]] = None,
                ):
                
                data = copy.deepcopy(self.db.data_dict)
                key_group = 'CADGroup_' + id_group

                coord = data[key_group]['Coord']
                nodecoord = data[key_group]['NodeCoord']
                
                if isinstance(new_order, str):
                    if new_order == 'lexsort':
                        sort_function = SAM.Weapons.sort_lexsort
                    elif new_order == 'centroid':
                        sort_function = SAM.Weapons.sort_by_centroid
                    elif new_order == 'kdtree':
                        sort_function = SAM.Weapons.sort_closed_curve_by_kdtree
                    elif new_order == 'convex_hull':
                        sort_function = SAM.Weapons.sort_points_by_hull_projection
                    else:
                        raise ValueError('new_order format not supported. Available method: lexsort, centroid, kdtree and convex_hull')

                    _, idx_new = sort_function(coord)
                    _, idx_nodes_new = sort_function(nodecoord)
                    
                elif isinstance(new_order, (tuple, list)):
                    idx_new = new_order
                    
                    if new_nodes_order is not None:
                        idx_nodes_new = new_nodes_order
                    else:
                        raise ValueError(f'new_nodes_order must be provided if new_order is a tuple or list.')
                
                for key in ['Coord', 'Conec', 'eltype', 'cellOrder']:
                    self.db.data_dict[key_group].update({key:data[key_group][key][idx_new]})
                
                
                for key in ['NodeCoord', 'pointOrder']:
                    self.db.data_dict[key_group].update({key:data[key_group][key][idx_nodes_new]})

                for key, idx in zip(['idx_sort', 'idx_sort_nodes'], [idx_new, idx_nodes_new]):
                    for stage in range(data[key_group][key].shape[0]):
                        for case in range(data[key_group][key].shape[1]):
                            self.db.data_dict[key_group][key][stage, case, :] = data[key_group][key][stage, case, idx]

                # Ordenar variables:
                
                for stage in data[key_group]['Vars']:
                    for var in data[key_group]['Vars'][stage]:
                        self.db.data_dict[key_group]['Vars'][stage][var] = \
                            data[key_group]['Vars'][stage][var][idx_new]
                                    
            def save_to_npy(
                self,
                stage:Union[list[int], tuple[int], int],
                id_group:str,
                filepath:str,
                case_idx:Union[int, list[int], tuple[int], 'all'] = 'all',
                ignore_vars:Union[list[str], tuple[str], None] = None,
                verbose:bool = False
                ):
                
                """
                """
                
                if not isinstance(id_group, str):
                    raise ValueError('id_group must be a string')
                
                group_key = f'CADGroup_{id_group}'
                group_data = self.db.data_dict[group_key]
                stage_vars = group_data["Vars"][str(stage)]
                
                try:
                    aux_dict = group_data['Aux']
                except KeyError:
                    aux_dict = {}

                Coord = group_data["Coord"]
                idx_sort_complete = group_data["idx_sort"]  # tamaño original (nstages, ncases, npuntos)
                conec = group_data["Conec"]
                flcc = group_data['FlCc']
                
                if isinstance(case_idx, str):
                    if case_idx == 'all':
                        case_idx = list(range(0, group_data['FlCc'].shape[0]))
                        witness_cases = True
                    else:
                        raise ValueError(f'case_idx can not be general string. Only "all" or numerical values.')
                elif isinstance(case_idx, int):
                    case_idx = [case_idx]
                    witness_cases = False
                elif isinstance(case_idx, (tuple, list)):
                    witness_cases = False
                    pass
                else:
                    raise ValueError('case_idx must be tuple, list, int or "all".')
                
                ncases = len(case_idx)
                npoints = Coord.shape[0]

                idx_sort = np.zeros((ncases,npoints), dtype=np.int32)
                eltype = np.zeros((ncases, npoints), dtype=np.int32) # tamaño original (npuntos)
                cellOrder = np.zeros((ncases, npoints), dtype=np.int32)  # tamaño original (npuntos)
                for ci,c in enumerate(case_idx):
                    eltype[ci] = group_data["eltype"][idx_sort_complete[stage,c, :]]
                    cellOrder[ci] = group_data["cellOrder"][idx_sort_complete[stage,c, :]]

                    idx_sort[ci]= idx_sort_complete[stage, c, :]   # ---> (ncases, npuntos)
                    
                diccionario = {}
                diccionario.update({'Coord': Coord, 'FlCc': flcc, 'idx_sort': idx_sort, 'Conec': conec, 'eltype': eltype, 'cellOrder': cellOrder})

                for var_name, var_data in stage_vars.items():
                    if ignore_vars is not None and var_name in ignore_vars:
                        continue

                    if var_data.ndim == 2:
                        if var_data.shape[0] == group_data["Coord"].shape[0]:
                            diccionario.update({var_name: np.transpose(var_data[:, case_idx])})
                    elif var_data.ndim == 3:
                        if var_data.shape[1] == group_data["Coord"].shape[0]:
                            diccionario.update({var_name: var_data[:, :, case_idx]})

                for aux_name, aux_data in aux_dict.items():
                    diccionario.update({aux_name: aux_data})
                    
                if not filepath.endswith('.npy'):
                    filepath = filepath + '.npy'

                np.save(filepath, diccionario, allow_pickle=True)

                if verbose:
                    print(f"\nSaved case {case_idx} to {filepath}" if witness_cases else f"\nSaved all cases to {filepath}")
                              
            def save_to_h5(
                self,
                filepath: str,
                overwrite: bool = True,
                verbose: bool = True
                ):

                if os.path.exists(filepath):
                    if overwrite:
                        os.remove(filepath)
                    else:
                        raise FileExistsError(f"{filepath} already exists.")

                if not filepath.endswith('.h5'):
                    filepath = filepath + '.h5'
                    
                def create_dataset_compressed(group, name, data):

                    # Determinar chunking óptimo por caso
                    if isinstance(data, np.ndarray):

                        if data.ndim == 2:
                            ncases, ncells = data.shape
                            chunks = (1, min(100000, ncells))

                        elif data.ndim == 3:
                            ncases, ncells, ncomp = data.shape
                            chunks = (1, min(100000, ncells), ncomp)

                        else:
                            chunks = True  # fallback seguro

                    else:
                        chunks = True

                    group.create_dataset(
                        name,
                        data=data,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        chunks=chunks
                    )

                with h5py.File(filepath, "w", libver="latest") as f:

                    for group_key, group_data in self.db.data_dict.items():

                        if verbose:
                            print(f"\nSaving group {group_key}")

                        grp = f.create_group(group_key)

                        # =====================================================
                        # GEOMETRY
                        # =====================================================
                        grp.create_dataset("Coord", data=group_data["Coord"])
                        grp.create_dataset("NodeCoord", data=group_data["NodeCoord"])
                        grp.create_dataset("FlCc", data=group_data["FlCc"])

                        create_dataset_compressed(grp, "Conec", group_data["Conec"])
                        create_dataset_compressed(grp, "eltype", group_data["eltype"])
                        create_dataset_compressed(grp, "cellOrder", group_data["cellOrder"])
                        create_dataset_compressed(grp, "pointOrder", group_data["pointOrder"])

                        # =====================================================
                        # VARIABLES
                        # =====================================================
                        vars_group = grp.create_group("Vars")

                        for stage, stage_vars in group_data["Vars"].items():

                            if verbose:
                                print(f"  Stage {stage}")

                            stage_grp = vars_group.create_group(str(stage))

                            # Subgrupos
                            scalars_grp = stage_grp.create_group("Scalars")
                            vectors_grp = stage_grp.create_group("Vectors")
                            gradients_grp = stage_grp.create_group("Gradients")

                            for var_name, var_data in stage_vars.items():

                                # -------------------------------------------------
                                # Reordenación final (case-major)
                                # -------------------------------------------------
                                if var_data.ndim == 2:
                                    # (cells, cases) → (cases, cells)
                                    var_to_save = var_data.T

                                elif var_data.ndim == 3:
                                    # (3, cells, cases) → (cases, cells, 3)
                                    var_to_save = np.transpose(var_data, (2, 1, 0))

                                else:
                                    raise ValueError(
                                        f"Unexpected dimension {var_data.ndim} in variable {var_name}"
                                    )

                                # -------------------------------------------------
                                # Separación por tipo
                                # -------------------------------------------------
                                if "Grad" in var_name:
                                    target_group = gradients_grp
                                elif var_to_save.ndim == 3:
                                    target_group = vectors_grp
                                else:
                                    target_group = scalars_grp

                                create_dataset_compressed(
                                    target_group,
                                    var_name,
                                    var_to_save
                                )

                                if verbose:
                                    print(f"    {var_name}: {var_to_save.shape}")

                    # Activar modo lectura paralela segura
                    f.swmr_mode = True

                if verbose:
                    print("\nFile saved successfully with compression, chunking and SWMR enabled.")
            
            def crop_bounding_box(
                self,
                id_group: str,
                bbox: Union[list, None] = None,
                radius_center: Union[tuple, None] = None,
                new_group_suffix: str = "_crop"
                ):

                key_old = f'CADGroup_{id_group}'
                key_new = f'{key_old}{new_group_suffix}'

                group = self.db.data_dict[key_old]

                coord = group['Coord']
                nodecoord = group['NodeCoord']

                # Posibilidad de hacer un círculo con radius_center = (radio, centro) o una caja de límites bbox
                if bbox is not None:
                    xmin, xmax = bbox[0]
                    ymin, ymax = bbox[1]
                    zmin, zmax = bbox[2]

                    # -------------------------
                    # 1. seleccionar celdas
                    # -------------------------

                    mask_cells = (
                        (coord[:,0] >= xmin) & (coord[:,0] <= xmax) &
                        (coord[:,1] >= ymin) & (coord[:,1] <= ymax) &
                        (coord[:,2] >= zmin) & (coord[:,2] <= zmax)
                    )

                    idx_cells = np.where(mask_cells)[0]

                elif radius_center is not None:
                    radius, center = radius_center
                    dist = np.linalg.norm(coord - center, axis=1)
                    mask_cells = dist <= radius
                    idx_cells = np.where(mask_cells)[0]

                else:
                    raise ValueError("Either bbox or radius_center must be provided.")

                # -------------------------
                # 2. nodos usados
                # -------------------------

                conec = group['Conec'][idx_cells]

                used_nodes = np.unique(conec)

                # mapa viejo -> nuevo
                node_map = -np.ones(nodecoord.shape[0], dtype=np.int64)
                node_map[used_nodes] = np.arange(len(used_nodes))

                # nueva conectividad
                conec_new = node_map[conec]

                # -------------------------
                # 3. construir nuevo grupo
                # -------------------------

                new_group = {}

                new_group['Coord'] = coord[idx_cells]
                new_group['NodeCoord'] = nodecoord[used_nodes]

                new_group['Conec'] = conec_new

                if 'eltype' in group:
                    new_group['eltype'] = group['eltype'][idx_cells]

                if 'cellOrder' in group:
                    new_group['cellOrder'] = group['cellOrder'][idx_cells]

                if 'pointOrder' in group:
                    new_group['pointOrder'] = group['pointOrder'][used_nodes]

                if 'FlCc' in group:
                    new_group['FlCc'] = group['FlCc']

                # -------------------------
                # 4. idx_sort
                # -------------------------

                if 'idx_sort' in group:
                    new_group['idx_sort'] = group['idx_sort'][:,:,idx_cells]

                if 'idx_sort_nodes' in group:
                    new_group['idx_sort_nodes'] = group['idx_sort_nodes'][:,:,used_nodes]

                # -------------------------
                # 5. variables
                # -------------------------

                new_group['Vars'] = {}

                if 'Vars' in group:

                    for stage in group['Vars']:

                        new_group['Vars'][stage] = {}

                        for var, arr in group['Vars'][stage].items():

                            if arr.ndim == 2:  # (Ncells, Ncases)
                                new_group['Vars'][stage][var] = arr[idx_cells]

                            elif arr.ndim == 3:

                                if arr.shape[0] == 3:  # vector
                                    new_group['Vars'][stage][var] = arr[:, idx_cells]

                                else:
                                    new_group['Vars'][stage][var] = arr[idx_cells]

                # -------------------------
                # 6. guardar nuevo grupo
                # -------------------------

                self.db.data_dict[key_new] = new_group

                # return idx_cells, used_nodes
            
            def interpolate_volume_to_surface(
                self,
                vol_group: str,
                surf_group: str,
                stage: str,
                vars: Union['all', list[str]] = 'all',
                k: int = 4,
                eps: float = 1e-12
                ):

                from scipy.spatial import cKDTree

                key_vol = f'CADGroup_{vol_group}'
                key_surf = f'CADGroup_{surf_group}'

                vol = self.db.data_dict[key_vol]
                surf = self.db.data_dict[key_surf]

                pts_vol = vol['Coord']
                pts_surf = surf['Coord']

                # -------------------------
                # KDTree
                # -------------------------

                tree = cKDTree(pts_vol)

                dist, idx = tree.query(pts_surf, k=k)

                w = 1.0 / (dist + eps)
                w /= w.sum(axis=1, keepdims=True)

                # -------------------------
                # preparar Vars superficie
                # -------------------------

                if 'Vars' not in surf:
                    surf['Vars'] = {}

                if stage not in surf['Vars']:
                    surf['Vars'][stage] = {}

                # -------------------------
                # interpolación
                # -------------------------

                if vars == 'all':
                    vars = list(vol['Vars'][stage].keys())
                    vars.remove('GlobalNumber')
                    vars.remove('CADGroupID')
                for var in vars:

                    arr = vol['Vars'][stage][var]

                    # -----------------
                    # ESCALAR
                    # shape (Ncells, Ncases)
                    # -----------------

                    if arr.ndim == 2:

                        # gather vecinos
                        vals = arr[idx]              # (Nsurf, k, Ncases)

                        # combinación ponderada
                        interp = np.einsum('ij,ijk->ik', w, vals)

                        surf['Vars'][stage][var + '_interp'] = interp


                    # -----------------
                    # VECTOR
                    # shape (3, Ncells, Ncases)
                    # -----------------

                    elif arr.ndim == 3 and arr.shape[0] == 3:

                        # gather vecinos
                        vals = arr[:, idx, :]        # (3, Nsurf, k, Ncases)

                        # combinación ponderada
                        interp = np.einsum('ij,lijk->lik', w, vals)

                        surf['Vars'][stage][var + '_interp'] = interp

                    else:

                        raise ValueError(f"Formato no soportado para {var}")
                    
            def interpolate_volume_to_surface_ant(
                self,
                vol_group: str,
                surf_group: str,
                vars: list,
                stage: str,
                k: int = 4,
                eps: float = 1e-12
                ):
                
                from scipy.spatial import cKDTree

                key_vol = f'CADGroup_{vol_group}'
                key_surf = f'CADGroup_{surf_group}'

                vol = self.db.data_dict[key_vol]
                surf = self.db.data_dict[key_surf]

                pts_vol = vol['Coord']
                pts_surf = surf['Coord']

                # -------------------------
                # KDTree
                # -------------------------

                tree = cKDTree(pts_vol)

                dist, idx = tree.query(pts_surf, k=k)

                w = 1.0 / (dist + eps)
                w /= w.sum(axis=1, keepdims=True)

                # -------------------------
                # preparar Vars superficie
                # -------------------------

                if 'Vars' not in surf:
                    surf['Vars'] = {}

                if stage not in surf['Vars']:
                    surf['Vars'][stage] = {}

                # -------------------------
                # interpolación
                # -------------------------

                for var in vars:

                    arr = vol['Vars'][stage][var]

                    # -----------------
                    # escalar
                    # -----------------

                    if arr.ndim == 2:

                        # (Ncells, Ncases)
                        values = arr[:, :]

                        interp = np.sum(values[idx] * w[..., None], axis=1)

                        surf['Vars'][stage][var + '_interp'] = interp

                    # -----------------
                    # vector
                    # -----------------

                    elif arr.ndim == 3 and arr.shape[0] == 3:

                        # (3, Ncells, Ncases)
                        values = arr

                        interp = np.sum(values[:, idx, :] * w[None, :, :, None], axis=2)

                        surf['Vars'][stage][var + '_interp'] = interp

                    else:

                        raise ValueError(f"Formato no soportado para {var}")


        class NRL7301Sets_pylom():
            
            def __init__(self, db:'FRODO'):
                self.db = db
                self.keys = db.kwargs.get('keys', {})
                names = db.kwargs.get('names', {})
                
                if names == {}:
                    self.names = self.keys
                else:
                    self.names = names
                    
            def add_aux(
                self,
                notes: str,
                tensor_name: str,
                tensor: torch.Tensor
                ):
                # Accedemos a la base de datos a través de self.db
                db = self.db

                # Asegurar que existe la clave 'aux'
                if not hasattr(db, "data_dict") or db.data_dict is None:
                    db.data_dict = {"inputs": {}, "aux": {}, "outputs": {}}
                elif "aux" not in db.data_dict:
                    db.data_dict["aux"] = {}

                # Añadir la nota en sim_metadata a nivel global
                if 'info_aux' not in db.sim_metadata:
                    db.sim_metadata['info_aux'] = []
                db.sim_metadata['info_aux'].append(notes)

                # Añadir el tensor
                db.data_dict['aux'][tensor_name] = tensor
                
            # @staticmethod
            # def add_aux(
            #     db: 'FRODO',
            #     notes: str = None,
            #     tensor_name: str,
            #     tensor: torch.Tensor
            #     ):
            #     # Asegurar que existe la clave 'aux'
            #     if not hasattr(db, "data_dict"):
            #         db.data_dict = {"inputs": {}, "aux": {}, "outputs": {}}
            #     elif "aux" not in db.data_dict:
            #         db.data_dict["aux"] = {}

            #     # Añadir la nota en sim_metadata a nivel global
            #     if 'info_aux' not in db.sim_metadata:
            #         db.sim_metadata['info_aux'] = []
            #     db.sim_metadata['info_aux'].append(notes)

            #     # Añadir el tensor
            #     db.data_dict['aux'][tensor_name] = tensor
                
            def create_jset(
                self,
                sol='all',
                n=None,
                save_path: bool | str = False,
                verbose: bool = False
                ):

                # --- Inputs ---
                tensors_inputs = [self.db.data_dict['inputs'][name] for name in self.names['inputs']]
                tensor_ptos = tensors_inputs[0]  # asumimos que ptos es el primero en names['inputs']
                tensor_flcc = torch.stack(tensors_inputs[1:], dim=1)  # resto de inputs apilados

                # --- Aux ---
                tensors_aux = [self.db.data_dict['aux'][name] for name in self.db.data_dict['aux'].keys()]
                print(tuple(tensor.shape for tensor in tensors_aux))
                # --- Outputs ---
                tensors_out = [self.db.data_dict['outputs'][name] for name in self.names['outputs']]

                # --- Crear dataset ---
                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, n=n, verbose=verbose
                )

                # --- Guardar si hace falta ---
                if save_path:
                    if save_path.endswith('h5'):
                        with h5py.File(save_path, "w") as h5file:
                            h5file.create_dataset("tensor", data=result['tensor'].numpy())
                            h5file.create_dataset("scaled", data=result['scaled'].numpy())
                            h5file.create_dataset("mins", data=result['mins'].numpy())
                            h5file.create_dataset("maxs", data=result['maxs'].numpy())

                    if save_path.endswith('pt'):
                        torch.save(obj=result, f=save_path)

                    if verbose:
                        print(f"Jset saved in {save_path}")

                self.db.dict_tensors = result
                print("jset loaded in db.dict_tensors")

            def create_split(self):
                pass
            
            def create_NN_pylom(self, *args, **kwargs):
                raise NotImplementedError("Función para exportar a pyLOM no implementada todavía.")
            
        class NRL7301Sets():
            
            def __init__(self, db:'FRODO'):
                self.db = db
            
            def add_aux(
                self,
                array_name: str,
                array: np.ndarray,
                notes: str = None,
                ):
                
                db = self.db
                # Asegurar que existe la clave 'aux'
                if not hasattr(db, "data_dict") or db.data_dict is None:
                    db.data_dict = {"inputs": {}, "aux": {}, "outputs": {}}
                elif "aux" not in db.data_dict:
                    db.data_dict["aux"] = {}

                # Añadir la nota en sim_metadata a nivel global
                if 'info_aux' not in db.sim_metadata:
                    db.sim_metadata['info_aux'] = []
                db.sim_metadata['info_aux'].append(notes)
                
                if 'keys_aux' not in list(db.sim_metadata.keys()):
                    db.sim_metadata['keys_aux']={}
                    
                db.sim_metadata['keys_aux'][array_name] = notes

                # Añadir el array
                db.data_dict['aux'][array_name] = array
            
            def define_split(
                self,
                split_by: list[str],
                columns_in_tensor: Union[list, tuple],
                values: Union[torch.Tensor, np.ndarray],
                ):
                
                if not hasattr(self.db, "dict_tensors"):
                    raise ValueError("No jset created yet in db.dict_tensors. Create it before defining a split.")
                elif not hasattr(self.db, "df_data"):
                    raise ValueError("No df_data created in db.df_data with create_jset().")
                
                print("FALTA POR HACER ESTE MÉTODO, AUNQUE NO ES PRIORITARIO PORQUE SE PUEDE HACER A MANO FÁCILMENTE. PRIORIDAD PARA HACER LOS CLUSTERS Y EL ANÁLISIS DEL BIC.")
                
            def create_NN_pylom(self, *args, **kwargs):
                pass
            
            def create_jset(
                self,
                sol='all',
                n=None,
                save_path: bool | str = False,
                verbose: bool = False
                ):
                # --- Inputs ---
                tensor_inputs = [torch.from_numpy(self.db.data_dict['inputs'][name]) for name in self.db.data_dict['inputs']]
                tensor_ptos = tensor_inputs[0]  # asumimos que ptos es el primero en names['inputs']
                tensor_flcc = torch.stack(tensor_inputs[1:], axis=1)  # resto de inputs apilados
                
                # --- Aux ---
                tensors_aux = [torch.from_numpy(self.db.data_dict['aux'][name]) for name in self.db.data_dict['aux'].keys()]

                # --- Outputs ---
                tensors_out = [torch.from_numpy(self.db.data_dict['outputs'][name]) for name in self.db.data_dict['outputs']]

                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, n=n, verbose=verbose
                )
                # --- Guardar si hace falta ---
                if save_path:
                    if save_path.endswith('h5'):
                        with h5py.File(save_path, "w") as h5file:
                            h5file.create_dataset("tensor", data=result['tensor'].numpy())
                            h5file.create_dataset("scaled", data=result['scaled'].numpy())
                            h5file.create_dataset("mins", data=result['mins'].numpy())
                            h5file.create_dataset("maxs", data=result['maxs'].numpy())

                    if save_path.endswith('pt'):
                        torch.save(obj=result, f=save_path)

                    if verbose:
                        print(f"Jset saved in {save_path}")
                self.db.dict_tensors = result
                columns = []
                for key2 in list(self.db.data_dict['inputs'].keys()):
                    try:
                        if self.db.data_dict['inputs'][key2].shape[1] == 2:
                            columns.extend(['x', 'z'])
                            
                        elif self.db.data_dict['inputs'][key2].shape[1] == 3:
                            columns.extend(['x', 'y', 'z'])
                    except:
                        columns.append(key2)
                        
                for key in ['aux', 'outputs']:
                    for key2 in list(self.db.data_dict[key].keys()):
                        # print(db.data_dict[key][key2].shape)
                        columns.append(key2)
                    
                self.db.df_data = pd.DataFrame(data = result['tensor'].numpy(), columns = columns)
                print("\njset loaded in db.dict_tensors\n")
                print("\ndataframe with main tensor loaded in db.df_data\n")
        
        class NUMPYFILESets():
            
            def __init__(self, db:'FRODO'):
                self.db = db
            
            def add_aux(
                self,
                array_name: str,
                array: np.ndarray,
                notes: str = None,
                ):
                
                db = self.db
                # Asegurar que existe la clave 'aux'
                if not hasattr(db, "data_dict") or db.data_dict is None:
                    db.data_dict = {"inputs": {}, "aux": {}, "outputs": {}}
                elif "aux" not in db.data_dict:
                    db.data_dict["aux"] = {}

                # Añadir la nota en sim_metadata a nivel global
                if 'info_aux' not in db.sim_metadata:
                    db.sim_metadata['info_aux'] = []
                db.sim_metadata['info_aux'].append(notes)
                
                if 'keys_aux' not in list(db.sim_metadata.keys()):
                    db.sim_metadata['keys_aux']={}
                    
                db.sim_metadata['keys_aux'][array_name] = notes

                # Añadir el array
                db.data_dict['aux'][array_name] = array
            
            def create_jset(
                self,
                sol: Union[list[int], tuple[int], int, 'all'] = 'all',
                n=None,
                save_path: bool | str = False,
                verbose: bool = False
                ):
                # --- Inputs ---

                tensor_ptos = self.db.data_dict['inputs']['ptos']
                tensor_flcc = np.column_stack([self.db.data_dict['inputs'][name] for name in self.db.data_dict['inputs'] if name != 'ptos'])
                
                # --- Aux ---
                tensors_aux = [self.db.data_dict['aux'][name] for name in self.db.data_dict['aux'].keys()]

                # --- Outputs ---
                tensors_out = [self.db.data_dict['outputs'][name] for name in self.db.data_dict['outputs'].keys()]

                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
                    sol=sol, n=n, verbose=verbose
                )
                # --- Guardar si hace falta ---
                if save_path:
                    if save_path.endswith('.h5'):
                        with h5py.File(save_path, "w") as h5file:
                            h5file.create_dataset("tensor", data=result['tensor'].numpy())
                            h5file.create_dataset("scaled", data=result['scaled'].numpy())
                            h5file.create_dataset("mins", data=result['mins'].numpy())
                            h5file.create_dataset("maxs", data=result['maxs'].numpy())

                    elif save_path.endswith('.pt'):
                        torch.save(obj=result, f=save_path)
                    elif save_path.endswith('.npy'):
                        np.save(file=save_path, arr = result, allow_pickle=True)
                    else:
                        raise NameError('save_path extension not supported. Please choose between .pt, .npy or .h5.')
                    
                    if verbose:
                        print(f"Jset saved in {save_path}\n")
                        
                self.db.jset = result
                columns = []
                for key2 in list(self.db.data_dict['inputs'].keys()):
                    if self.db.data_dict['inputs'][key2].shape[1] == 2:
                        columns.extend(['x', 'z'])
                        
                    elif self.db.data_dict['inputs'][key2].shape[1] == 3:
                        columns.extend(['x', 'y', 'z'])
                        
                    elif self.db.data_dict['inputs'][key2].shape[1] == 1:
                        columns.append(key2)
                        
                for key in ['aux', 'outputs']:
                    for key2 in list(self.db.data_dict[key].keys()):
                        # print(db.data_dict[key][key2].shape)
                        columns.append(key2)
                self.db.df_data = pd.DataFrame(data = result['tensor'].numpy(), columns = columns)
                
                if verbose:
                    print("\nJset loaded in db.jset\n")
                    print("\nDataframe with main tensor loaded in db.df_data\n")
            
            def create_NN_pylom(self, *args, **kwargs):
                pass
        
        class FLUENTSets():
            
            def __init__(self, db:'FRODO'):
                self.db = db
                
            def create_jset(self, *args, **kwargs):
                return super().create_jset(*args, **kwargs)
            
            def create_NN_pylom(self, *args, **kwargs):
                return super().create_NN_pylom(*args, **kwargs)
            
            def create_split(self, *args, **kwargs):
                return super().create_split(*args, **kwargs)