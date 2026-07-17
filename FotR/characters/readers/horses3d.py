"""
readers/horses3d.py
===================
Reader for Horses3D Solver output.

Expected directory layout
--------------------------
::

    COMPLETAR
    
    
"""
import os
import re
import json

from typing import Literal, Union

import numpy as np
import pandas as pd

from ..sam import SAM
from .base import BaseReader

class HORSES3DReader(BaseReader):
    """

    Reader for HORSES3D CFD solver output.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root directory (passed by FRODO).
    """
    
    def __init__(self, root_dir: str, **kwargs):
        super().__init__(root_dir, **kwargs)
        self.output_dir = os.path.join(self.root_dir, "outputs")
        print(f'\n NEW CODA SIMULATION WILL BE LOADED FROM {root_dir}')

        try:
            meta_path = os.path.join(root_dir, 'metadata', 'cases_metadata.json')
            with open(meta_path, 'r') as fh:
                cm = json.load(fh)

            self.metadata = {
                'eq_type':     cm.get('eq_type',    None),
                'folder_fmt':  cm.get('folder_fmt', None),
                'design_vars': cm.get('design_vars', None),
                'num_stages':  cm.get('num_stages', None),
            }

            df_cases = (
                pd.DataFrame.from_dict(cm.get('df_cases', {}))
                .sort_values(by=self.metadata['design_vars'][0],
                             ignore_index=True, axis=0)
                .reset_index(drop=True)
            )
            if "case_idx" not in df_cases.columns:
                df_cases.insert(0, "case_idx", df_cases.index.astype(np.int32))
            self.metadata['df_cases'] = df_cases

        except Exception as exc:
            print(
                "WARNING: cases_metadata.json not found or could not be "
                "loaded. Folder format will be inferred from folder names.\n"
            )
            print(exc)
            self._infer_metadata_from_folders()
            
    # ── Metadata inference fallback ───────────────────────────────────────────

    def _infer_metadata_from_folders(self):
        """Infer metadata from output folder names when JSON is missing."""
        folders       = os.listdir(self.output_dir)
        possible_sep  = ['_', '-']
        sep_list, params_list, nfiles_list = [], [], []

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
                    files = SAM.Backpack.pattern_pocket.find_files(
                            os.path.join(self.output_dir, folder, 'RESULTS'),
                            endswith=r"_p\d+\.hsol",
                            literal=False
                        )
                    nfiles_list.append(len(files))
        if not params_list:
            raise ValueError(
                "No simulation folders found or no numeric parameters "
                "detected in folder names."
            )

        if not (
            all(len(p) == len(params_list[0]) for p in params_list)
            and all(s == sep_list[0] for s in sep_list)
            and all(n == nfiles_list[0] for n in nfiles_list)
        ):
            raise ValueError(
                "Inconsistent folder naming. "
                "Please provide a valid cases_metadata.json file."
            )

        parts = folders[0].split(sep_list[0])
        self.metadata = {
            'eq_type': None,
            'folder_fmt': sep_list[0].join([
                p if not re.findall(r"-?\d+\.?\d*", p) else "{}"
                for p in parts
            ]),
            'design_vars': [parts[i] for i in range(0, len(parts), 2)],
            'num_stages':  nfiles_list[0],
        }

        df_arr = np.zeros((len(folders), len(params_list[0])), dtype=float)
        for f_idx, folder in enumerate(folders):
            df_arr[f_idx, :] = params_list[f_idx]

        df_cases = pd.DataFrame(
            df_arr, 
            columns=self.metadata['design_vars']
        ).reset_index(drop=True)
        df_cases.insert(0, "case_idx", df_cases.index.astype(np.int32))
        self.metadata['df_cases'] = df_cases
        
    def parse_simulation_dirs(self) -> None:
        """
        Walk ``outputs/`` and build ``sim_metadata`` and ``df_state``.

        Matches folder names against the format pattern, maps each folder
        to the closest entry in ``df_cases`` (by Euclidean distance in the
        design-variable space), and counts available stages.

        Populates
        ---------
        self.sim_metadata : dict
            Keyed by folder name. Values: path, stages dict, design-var
            values, computation_times list.
        self.df_state : pd.DataFrame
            One row per matched simulation.
        """
        pattern = SAM.Backpack.pattern_pocket.FilenamePattern.from_template(
            self.metadata["folder_fmt"], numeric=True
        ).compiled
        
        poly_calculated = False * np.ones((len(os.listdir(self.output_dir)),7), dtype=bool)
        for i, folder in enumerate(os.listdir(self.output_dir)):
            if not pattern.match(folder):
                continue

            nums_folder = np.array(
                [float(x) for x in re.compile(r"[-\d\.]+").findall(folder)],
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
            
            control_files_found = SAM.Backpack.pattern_pocket.find_files(
                os.path.join(self.output_dir, folder,),
                endswith='.control'
            ) #stages que hemos programado. tenemos que pillar el patrón del nombre y usarlo para buscar los archivos de cada stage.
            # motivo por el que se crea SAM...pattern_pocket -> conseguir patrón de archivos .control y buscar archivos de cada stage.
            
            ## CONTINUAR POR AQUÍ
                
            files = SAM.Backpack.pattern_pocket.find_files(
                os.path.join(self.output_dir, folder, 'RESULTS'),
                endswith=r"_p\d+\.hsol",
                literal=False
            )
            
            list_poly_degree = []
            for file in files:
                # sacar grado polinomio del nombre de cada file: *_p<grado>.hsol
                poly_degree = int(re.findall(r"_p(\d+)\.hsol", file)[0])
                list_poly_degree.append(poly_degree)
            # Cambiar posiciones list_poly_degree de poly_calculated a True
            for degree in list_poly_degree:
                if degree < poly_calculated.shape[1]:
                    poly_calculated[i,degree-1] = True
                    
            full_path = os.path.join(self.output_dir, folder)
            if not os.path.isdir(full_path):
                continue

            stage_dict: dict = {}
            # for fname in os.listdir(full_path):
            #     if fname.startswith("output_"):
            #         parts = fname.split("_")
            #         if len(parts) >= 2:
            #             stage_raw = os.path.splitext(parts[1])[0]
            #             if stage_raw.isdigit():
            #                 stage = int(stage_raw)
            #                 ext   = os.path.splitext(fname)[-1].lstrip(".")
            #                 stage_dict.setdefault(
            #                     stage, {"files": [], "types": set()}
            #                 )
            #                 stage_dict[stage]["files"].append(fname)
            #                 stage_dict[stage]["types"].add(ext)

            # for stage in stage_dict:
            #     stage_dict[stage]["types"] = list(stage_dict[stage]["types"])

            self.sim_metadata[folder] = {
                "folder":            folder,
                "path":              full_path,
                "stages":            stage_dict,
                "computation times": [],
            }
            self.sim_metadata[folder].update(
                {var: val for var, val in
                 zip(self.metadata["design_vars"], nums)}
            )

        self.df_state = pd.concat(
            (
                self.metadata['df_cases'],
                pd.DataFrame(
                    poly_calculated, columns = [f'p{p}' for p in range(1,8)], dtype=bool)
            ),
            axis=1, ignore_index=False,
        )

        print(f"{len(self.sim_metadata)} simulations found." if len(self.sim_metadata) > 1 else f"{len(self.sim_metadata)} simulation found.")
    
    def extract_inputs(self, *args, **kwargs):
        pass
    
    def extract_outputs(self, *args, **kwargs):
        pass