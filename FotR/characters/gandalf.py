import os
import json
import numpy as np
import torch

import pandas as pd
from typing import Union
from scipy.stats import truncnorm, qmc

import shutil
import subprocess

from ..EarendilsLight import EarendilsLight

class GANDALF:
    
    
    light = EarendilsLight(__name__)
    
    @classmethod
    def some_light(cls, name=None):
        """Atajo a Eärendil's Light."""
        return cls.light.help(name)


    def __init__(
        self, root_dir: str,
        eq_type: str = "rans",
        num_stages:int=None,
        **kwargs
        ):
        """
        Initializes the CFD case manager.
        """
        self.root_dir = root_dir
        self.eq_type = eq_type.lower()
        assert self.eq_type in ["euler", "rans"], "eq_type must be 'Euler' o 'RANS'."
        self.case_tensor = None
        self.design_vars = None
        if num_stages is None:
            raise ValueError('Number of stages must be provided.')
        else:
            self.num_stages = num_stages
        self.version = kwargs.get('version', None)
        # script_dir = f'/home/m.jaraiz/repos/CETACEO_UPM/cetaceo/data/{case}/sources/'

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "metadata"), exist_ok=True)
        
        print(f'New simulation folder will be created in {os.path.abspath(self.root_dir)}.')

    def define_geom_file(
        self,
        geom_file_path:str,
        cols_idx:list[int] = [0,2],
        normalize:bool = False,
        **kwargs
        ):
        """
        Define the geometry file (airfoil coordinates) with pandas library.
        Args:
            geom_file_path (str): path to the geometry CSV file.
            cols_idx (list[int]): list of column indices to extract (default: [0,2] for x and y).
            normalize (bool): whether to normalize the airfoil coordinates.
            **kwargs: additional arguments for pd.read_csv.
        
        """
        self.df_geom = pd.read_csv(geom_file_path, **kwargs)
        if normalize:
            self.array_ptos = GANDALF.Backpack.normalize_airfoil(self.df_geom.values[:,cols_idx])
        else:
            self.array_ptos = self.df_geom.values[:,cols_idx]
        
    def define_cases(
        self,
        method: str,
        bounds: dict = None,
        n_samples: int = 100,
        peak_ranges: dict = None,
        range_sigma: float = 2.0,
        external_dataframe: pd.DataFrame = None,
        seed: int = None,
        ):
        """
        Define the tensor of CFD cases (e.g., AoA, Mach).
        
        Args:
            method (str): 'halton', 'lhs', or 'external'.
            bounds (dict): {'AoA': (0, 5), 'Mach': (0.3, 1.5), 'h': 11000}.
            n_samples (int): number of points to generate (for halton/lhs).
            peak_ranges (dict): Ranges of bounds where higher point density is needed. Default None. Example: {'AoA': None, 'Mach': (0.7, 1.2)}.
            range_sigma (float): controls concentration (LHS only).
            external_dataframe (pd.DataFrame): DataFrame with external cases (for 'external' method).
            seed (int): seed for reproducibility.
        """

        method = method.lower()
        if method not in ["halton", "lhs", "external"]:
            raise ValueError("method must be 'halton', 'lhs', or 'external'.")

        # ---------------------------------------
        # Caso external
        # ---------------------------------------
        if method == "external":
            if external_dataframe is None:
                raise ValueError("An external dataframe must be provided for method 'external'.")
            self.case_tensor = external_dataframe.values.astype(float)
            self.design_vars = external_dataframe.columns.tolist()
            self.df_cases = external_dataframe
            
            return None

        # ---------------------------------------
        # Halton / LHS
        # ---------------------------------------
        if bounds is None or n_samples is None:
            raise ValueError("bounds and n_samples must be provided for 'halton' or 'lhs' methods.")

        # Separar variables en muestreadas y constantes
        vars_to_sample = []
        vars_constant = {}

        for key, val in bounds.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                vars_to_sample.append(key)
            elif isinstance(val, (tuple, list)) and len(val) == 1:
                vars_constant[key] = float(val)
            elif isinstance(val, (int, float)):
                vars_constant[key] = float(val)
            # elif isinstance(val, range):
            #     vars_to_sample.append(key)
            else:
                raise ValueError(f"Invalid bound for variable '{key}': {val}. Must be a tuple (min, max) or a single value.")

        dims = len(vars_to_sample)
        if dims == 0:
            raise ValueError("At least one variable must be defined with a (min, max) bound for sampling. Or you can define an external dataframe.")

        # Sampler Halton o LHS
        sampler = qmc.Halton(d=dims, seed=seed) if method == "halton" else qmc.LatinHypercube(d=dims, seed=seed)
        u = sampler.random(n=n_samples)

        # peak_ranges por defecto solo para las variables muestreadas
        if peak_ranges is None:
            peak_ranges = {k: None for k in vars_to_sample}

        # Aplicar warp solo a las variables muestreadas
        sampled_columns = []
        for i, var in enumerate(vars_to_sample):
            warped = GANDALF.Backpack._warp_variable(
                u[:, i],
                bounds[var],
                peak_range=peak_ranges.get(var, None),
                range_sigma=range_sigma,
            )
            sampled_columns.append(warped)

        sampled_np = np.column_stack(sampled_columns)

        # Construir el tensor final respetando el orden original de bounds
        final_columns = []
        for var in bounds.keys():
            if var in vars_to_sample:
                idx = vars_to_sample.index(var)
                final_columns.append(sampled_np[:, idx])
            else:
                final_columns.append(np.full(n_samples, vars_constant[var], dtype=float))

        self.case_tensor = np.column_stack(final_columns)
        self.design_vars = list(bounds.keys())
        self.df_cases = pd.DataFrame(data=self.case_tensor, columns = self.design_vars, dtype=np.float32)

    def add_param(
        self,
        name:str = "param",
        data:np.ndarray = None
        ):
        """
        Add a new parameter to the cases DataFrame. The data must be provided as a external numpy array. For calculated data, use compute_param().
        """
        df = self.df_cases
        if data == None:
            raise ValueError('Data must be provided to add the parameter.')
        df[name] = np.asarray(data, dtype=float)
        self.df_cases = df
        
    def compute_param(self, name:str = 'param', formula:str = None, externals:dict = None):
        """
        Compute a new parameter based on a formula and add it to the cases DataFrame.
        
        Args:
            name (str): Name of the new parameter.
            formula (str): Formula to compute the parameter, using DataFrame column names as variables.
            externals (dict): External variables to include in the formula.
        """
        
        df = self.df_cases
        
        if formula is None:
            raise ValueError('A formula must be provided to compute the parameter.')
        
        env = {}

        # Variables del DataFrame
        for col in df.columns:
            env[col] = df[col].values

        # Variables externas
        if externals is not None:
            for key, value in externals.items():
                # permitir escalar o array
                if np.isscalar(value):
                    env[key] = value
                else:
                    env[key] = np.asarray(value)

        # Funciones matemáticas seguras
        env.update({
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "pi": np.pi,
        })

        try:
            param = eval(formula, {"__builtins__": None}, env)
        except Exception as e:
            print(f"Error evaluating formula: {formula}")
            print(f"   {e}")
            return None
        
        df[name] = np.asarray(param, dtype=float)
        self.df_cases = df
        
    def generate_folders(
        self,
        base_files: list[str],
        mesh_path:str,
        script_dir: str,
        folder_fmt: str = "aoa_{AoA:.2f}_mach_{Mach:.3f}_h_{h:.0f}",
        overwrite: bool = False,
        update_base_files:bool = False,
        data_to_update:dict = {},
        ):
        """
        Generate subfolders for each defined CFD case.
        
        Args:
            base_files (list): List of base files to copy into each subfolder.
            script_dir (str): Directory where base files are located.
            folder_fmt (str): Folder name format. Names must match those used in bounds in define_cases. Example:
                "AoA_{AoA:.2f}_Mach_{Mach:.3f}_h_{h:.0f}".
            overwrite (bool): If True, overwrite existing folders.
            update_base_files (bool): If True, you can update placeholders in base file with "data_to_update". By default, False.
            data_to_update (dict): Dictionary with data to update in base files. Key must be the string to change, and value the label of the variable in self.df_cases. Example: {"AOA_PLACEHOLDER": AoA}.
        """
        
        if self.case_tensor is None: # self.case_tensor sirve para hacer self.df_cases, así que tienen el mismo orden
            raise RuntimeError("Cases must be defined before generating folders. Use define_cases().")

        if self.design_vars is None:
            raise RuntimeError("Design variables not defined.")

        if update_base_files and data_to_update == {}:
            raise ValueError('No data found to update base files.')
        
        self.folder_fmt = folder_fmt
        self.folders_name = []
        n_cases = self.case_tensor.shape[0]
        print(f"Creating {n_cases} simulation folders in {self.root_dir}")

        for i, row in enumerate(self.case_tensor):

            case_dict = {var: float(row[j]) for j, var in enumerate(self.design_vars)}

            try:
                folder_name = folder_fmt.format(**case_dict)
            except KeyError as e:
                raise KeyError(f"Variable {e} not found in design_vars ({self.design_vars})")
            
            self.folders_name.append(folder_name)
            case_dir = os.path.join(self.root_dir, 'outputs', folder_name)
            if os.path.exists(case_dir):
                if overwrite:
                    shutil.rmtree(case_dir)
                else:
                    print(f"Folder {case_dir} already exists. Skipping...")
                    continue

            os.makedirs(case_dir, exist_ok=True)

            for f in base_files:
                src = os.path.join(script_dir, f)
                dst = os.path.join(case_dir, f)
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Base file not found: {src}")
                shutil.copyfile(src, dst)
                
                with open(dst, 'r') as file:
                    content = file.read()
                    content = content.replace("MESH_PLACEHOLDER", '"' + os.path.basename(mesh_path) + '"')
                    
                    if update_base_files:
                        for placeholder, var_name in data_to_update.items():
                            if var_name not in self.df_cases.columns:
                                content = content.replace(placeholder, var_name)
                            else:
                                content = content.replace(placeholder, str(self.df_cases[data_to_update[placeholder]].iloc[i]))
                                
                with open(dst, 'w') as file:
                    file.write(content)
                    
            dst_mesh = os.path.join(case_dir, os.path.basename(mesh_path))
            shutil.copyfile(mesh_path, dst_mesh)
        
        # Save self params (folder_fmt, df_cases, ...) in os.path.join(root_dir, 'metadata') como un archivo json
        metadata_path = os.path.join(self.root_dir, 'metadata', 'cases_metadata.json')
        metadata = {
            'root_dir': self.root_dir,
            'eq_type': self.eq_type,
            'folder_fmt': self.folder_fmt,
            'design_vars': self.design_vars,
            'num_stages': self.num_stages,
            'df_cases': self.df_cases.to_dict(orient='list'),
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        self.df_cases.to_csv(os.path.join(self.root_dir, 'metadata','df_cases.csv'), sep=',', index=False)
                    
    def assign_jobs(
        self, file_sh:str,
        nodes:list[str],
        cpus_per_job:int,
        submit:bool = False
        ):
        """
        Assign cases to available nodes in a balanced way.
        
        Args:
            file_sh (str): Name of the shell script file to modify for each case.
            nodes (list[str]): List of available node names.
            cpus_per_job (int): Number of CPUs to assign per job.
            submit (bool): If True, modify the shell script files with assigned node and CPUs.
        """
        casos = self.folders_name
        self.file_sh = file_sh
        cpus_por_nodo = {}
        output_dir = os.path.join(self.root_dir, 'outputs')

        print("\n--- Trabajos en ejecución ---")
        _ = GANDALF.Backpack.squeue_terminal()
        
        print("\n--- Estado de los nodos ---")
        sinfo = GANDALF.Backpack.sinfocpu_terminal()
        
        for linea in sinfo.strip().split("\n")[1:]:
            partes = linea.split()
            nodo = partes[0]
            if nodo not in nodes:
                continue
            try:
                # Formato esperado: "n005  alloc/idle/other/total"
                datos_cpu = partes[1].split("/")
                cpus_disponibles = int(datos_cpu[1])  # CPUs inactivas
                cpus_por_nodo[nodo] = cpus_disponibles
            except (IndexError, ValueError):
                print(f"⚠️ Formato inesperado en línea: {linea}")

        if not cpus_por_nodo:
            raise ValueError("❌ No se encontraron nodos válidos en la lista filtrada.")

        # Ordenar nodos por CPUs disponibles
        nodos_ordenados = sorted(cpus_por_nodo.items(), key=lambda x: x[1], reverse=True)
        asignaciones = {nodo: [] for nodo, _ in nodos_ordenados}

        # Asignar casos de forma balanceada
        for caso in casos:
            nodo_menos_cargado = min(
                asignaciones.keys(),
                key=lambda n: len(asignaciones[n])
            )
            asignaciones[nodo_menos_cargado].append(
                (caso, cpus_per_job)
            )

        print("\n✅ Asignaciones completadas:\n")
        for nodo, trabajos in asignaciones.items():
            print(f"{nodo}: {len(trabajos)} tareas")
            for caso, cpus in trabajos:
                print(f"  └─ {caso} ({cpus} CPUs)")
                if submit:
                    # if data_to_update == {}:
                    #     raise ValueError('No data found in data_to_update')
                    run_sh_path = os.path.join(output_dir, caso, file_sh)
                    with open(run_sh_path, "r") as f:
                        contenido = f.read()

                    contenido = contenido.replace("%NODO", f"{nodo}")
                    separado = caso.split('_')
                    contenido = contenido.replace(
                        "%NAME", ''.join(separado[i][0] + f"{float(separado[i+1]):.2f}" for i in range(0, len(separado), 2))
                        )
                    contenido = contenido.replace("%CPUs", str(cpus))

                    with open(run_sh_path, "w") as f:
                        f.write(contenido)

                    print(f"📄 Modificado {os.path.basename(run_sh_path)} → Nodo: {nodo}, CPUs: {cpus}")
                    
    def submit_cases(self):
        """
        Submit all case jobs using their respective shell scripts.
        """
        output_dir = os.path.join(self.root_dir, 'outputs')
        casos = self.folders_name
        if not casos:
            raise RuntimeError("No cases found in 'outputs/'.")

        print("\n--- Submitting jobs ---\n")
        for caso in casos:
            run_sh_path = os.path.join(output_dir, caso, self.file_sh)
            case_dir = os.path.join(output_dir, caso)
            print(run_sh_path)
            if not os.path.isfile(run_sh_path):
                print(f"Execution file {run_sh_path} not found. Skipping...")
                continue

            try:
                result = subprocess.run(
                    ["sbatch", run_sh_path],
                    capture_output=True,
                    cwd=case_dir,
                    text=True,
                    check=True
                )

                print(f"{caso}: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"{caso}: {e.stderr.strip()}")
                
    class Backpack:
        
        @staticmethod
        def _warp_variable(u, bounds, peak_range=None, range_sigma=3):
            """
            Aplica una transformación no lineal (warp) a u∈[0,1] para concentrar
            puntos dentro de un rango (peak_range). Si peak_range es None, hace un mapeo lineal.
            """
            v_min, v_max = bounds

            if peak_range is None:
                # Mapeo lineal simple
                return v_min + u * (v_max - v_min)

            p_min, p_max = peak_range
            mean = (p_min + p_max) / 2
            std = (p_max - p_min) / range_sigma

            a, b = (v_min - mean) / std, (v_max - mean) / std
            trunc_gauss = truncnorm(a=a, b=b, loc=mean, scale=std)

            u_clipped = np.clip(u, 1e-6, 1 - 1e-6)
            return trunc_gauss.ppf(u_clipped)
        
        @staticmethod
        def squeue_terminal():
            try:
                squeue = subprocess.run(
                    ["squeue", "-t",  "R"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(squeue.stdout)
                
                return squeue.stdout
            
            except Exception as e:
                print(f"Error ejecutando squeue: {e}")
                
        @staticmethod
        def sinfocpu_terminal():
            try:
                resultado = subprocess.run(
                    ["sinfo", "-N", "-o", "%n  %C"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(resultado.stdout)
                return resultado.stdout
            
            except Exception as e:
                print(f"Error ejecutando sinfo: {e}")
        
        @staticmethod
        def normalize_airfoil(coords: Union[torch.Tensor, np.ndarray]):
            """
            Normaliza un perfil aerodinámico Nx2 para que la cuerda sea 1
            y el borde de ataque esté en x = 0.
            """
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("El tensor debe tener forma [N, 2].")

            x_min = coords[:, 0].min()
            x_max = coords[:, 0].max()
            chord = x_max - x_min

            # coords_shifted = coords.copy()
            coords[:, 0] -= x_min  # mueve el borde de ataque a x=0
            coords_scaled = coords / chord  # cuerda unitaria

            return coords_scaled
        
        @staticmethod
        def isa_atmosphere(h):
            
            """
            Calculate the static pressure, temperature and density according to ISA model.
            
            Args:
                h (float or np.ndarray): Geometric altitude [m]
            
            Returns:
                T (float or np.ndarray): Temperature [K]
                P (float or np.ndarray): Pressure [Pa]
                rho (float or np.ndarray): Density [kg/m^3]
            """

            # Constantes
            R = 287.05       # J/(kg·K), constante específica del gas para el aire
            g0 = 9.80665     # m/s², gravedad
            P0 = 101325.0    # Pa, presión al nivel del mar
            T0 = 288.15      # K, temperatura al nivel del mar
            # rho0 = 1.225     # kg/m³, densidad al nivel del mar

            # Capas ISA hasta 86 km
            h_layers = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852]
            L = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002]  # gradientes [K/m]
            
            T = T0
            P = P0

            if isinstance(h, (float, int)):
                h = np.array([h])
                scalar_input = True
            else:
                h = np.asarray(h)
                scalar_input = False

            T_out = np.zeros_like(h, dtype=float)
            P_out = np.zeros_like(h, dtype=float)
            rho_out = np.zeros_like(h, dtype=float)

            for i in range(len(h)):
                hi = h[i]
                T = T0
                P = P0
                for j in range(len(L)):
                    hb = h_layers[j]
                    ht = h_layers[j+1]
                    if hi <= ht:
                        L_j = L[j]
                        if L_j == 0.0:
                            T = T
                            P *= np.exp(-g0 * (hi - hb) / (R * T))
                        else:
                            T = T + L_j * (hi - hb)
                            P *= (T / (T - L_j * (hi - hb))) ** (-g0 / (R * L_j))
                        break
                    else:
                        L_j = L[j]
                        h_diff = ht - hb
                        if L_j == 0.0:
                            P *= np.exp(-g0 * h_diff / (R * T))
                        else:
                            T = T + L_j * h_diff
                            P *= (T / (T - L_j * h_diff)) ** (-g0 / (R * L_j))
                rho = P / (R * T)
                T_out[i] = T
                P_out[i] = P
                rho_out[i] = rho

            if scalar_input:
                return T_out[0], P_out[0], rho_out[0]
            else:
                return T_out, P_out, rho_out
         
        @staticmethod   
        def Sutherland_law(mu0, T, Treference):
            """
            """
            return mu0 * (T/Treference)**1.5 * (Treference + 110.4) / (T + 110.4)