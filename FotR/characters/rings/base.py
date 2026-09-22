"""
rings/base.py
=============
Shared base for every GANDALF case-generation ring.

``BaseRing`` owns everything that does not depend on how meshes are
assigned to cases: case-space sampling (``define_cases``), parameter
columns (``add_param`` / ``compute_param``), SLURM submission
(``assign_jobs`` / ``submit_cases`` / ``recover_pending_jobs``) and the
``Backpack`` utilities.  Concrete rings (``coda``, ``coda_single``)
only implement ``generate_folders`` with their own mesh policy.
"""

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, qmc

from ...EarendilsLight import EarendilsLight

log = logging.getLogger(__name__)


class BaseRing:
    """
    Shared case-generation logic for all rings.

    Parameters
    ----------
    root_dir : str
        Dataset root directory (``outputs/`` + ``metadata/`` are created).
    eq_type : str
        ``'euler'`` or ``'rans'``.
    num_stages : int
        Number of solver stages (required).
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Shortcut to Earendil's Light help system."""
        return cls.light.help(name)

    def __init__(
        self, root_dir: str,
        eq_type: str = "rans",
        num_stages: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initializes the CFD case manager.
        """
        self.root_dir = root_dir
        self.eq_type = eq_type.lower()
        if self.eq_type not in ["euler", "rans"]:
            raise ValueError("eq_type must be 'Euler' o 'RANS'.")
        self.case_tensor = None
        self.design_vars = None
        self.df_cases = None
        self.df_geom = None
        if num_stages is None:
            raise ValueError('Number of stages must be provided.')
        else:
            self.num_stages = num_stages
        self.version = kwargs.get('version', None)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "metadata"), exist_ok=True)

        log.info(
            'GANDALF initialized with root_dir: %s.',
            os.path.abspath(self.root_dir),
        )

    def define_geom_file(
        self,
        geom_file_path: str,
        cols_idx: list = [0, 2],
        normalize: bool = False,
        **kwargs
    ) -> None:
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
            self.array_ptos = BaseRing.Backpack.normalize_airfoil(
                self.df_geom.values[:, cols_idx]
            )
        else:
            self.array_ptos = self.df_geom.values[:, cols_idx]

    def define_cases(
        self,
        method: str,
        bounds: Optional[dict] = None,
        n_samples: int = 100,
        peak_ranges: Optional[dict] = None,
        range_sigma: float = 2.0,
        external_dataframe: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Define the tensor of CFD cases (e.g., AoA, Mach).

        Args:
            method (str): 'halton', 'lhs', or 'external'.
            bounds (dict): {'AoA': (0, 5), 'Mach': (0.3, 1.5), 'h': 11000}.
            n_samples (int): number of points to generate (for halton/lhs).
            peak_ranges (dict): Ranges of bounds where higher point density is needed. Default None. Example: {'AoA': None, 'Mach': (0.7, 1.2)}.
            range_sigma (float): controls concentration of the warp.
            external_dataframe (pd.DataFrame): DataFrame with external cases (for 'external' method).
            seed (int): seed for reproducibility.
        """

        method = method.lower()
        if method not in ["halton", "lhs", "external"]:
            raise ValueError("method must be 'halton', 'lhs', or 'external'.")

        # ---------------------------------------
        # External cases
        # ---------------------------------------
        if method == "external":
            if external_dataframe is None:
                raise ValueError("An external dataframe must be provided for method 'external'.")
            external_dataframe = external_dataframe.copy()
            self.case_tensor = external_dataframe.values.astype(float)
            self.design_vars = external_dataframe.columns.tolist()
            self.df_cases = external_dataframe

            return None

        # ---------------------------------------
        # Halton / LHS
        # ---------------------------------------
        if bounds is None or n_samples is None:
            raise ValueError("bounds and n_samples must be provided for 'halton' or 'lhs' methods.")
        if not isinstance(n_samples, (int, np.integer)) or int(n_samples) <= 0:
            raise ValueError(f"n_samples must be a positive int, got {n_samples!r}.")
        n_samples = int(n_samples)

        # Split variables into sampled and constant ones
        vars_to_sample = []
        vars_constant = {}

        for key, val in bounds.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                lo, hi = float(val[0]), float(val[1])
                if not lo < hi:
                    raise ValueError(
                        f"Invalid bound for variable '{key}': ({lo}, {hi}). "
                        "Need min < max."
                    )
                vars_to_sample.append(key)
            elif isinstance(val, (tuple, list)) and len(val) == 1:
                vars_constant[key] = float(val[0])
            elif isinstance(val, (int, float, np.integer, np.floating)):
                vars_constant[key] = float(val)
            else:
                raise ValueError(f"Invalid bound for variable '{key}': {val}. Must be a tuple (min, max) or a single value.")

        dims = len(vars_to_sample)
        if dims == 0:
            raise ValueError("At least one variable must be defined with a (min, max) bound for sampling. Or you can define an external dataframe.")

        # Halton or LHS sampler
        sampler = qmc.Halton(d=dims, seed=seed) if method == "halton" else qmc.LatinHypercube(d=dims, seed=seed)
        u = sampler.random(n=n_samples)

        # Default peak_ranges, only for sampled variables
        if peak_ranges is None:
            peak_ranges = {k: None for k in vars_to_sample}

        for var, pr in peak_ranges.items():
            if pr is None:
                continue
            if var not in vars_to_sample:
                raise ValueError(
                    f"peak_ranges['{var}'] refers to a variable that is not "
                    f"sampled (sampled: {vars_to_sample})."
                )
            if (
                not isinstance(pr, (tuple, list)) or len(pr) != 2
                or not all(isinstance(v, (int, float, np.integer, np.floating)) for v in pr)
            ):
                raise ValueError(
                    f"peak_ranges['{var}'] must be a (min, max) pair, got {pr!r}."
                )
            p_min, p_max = float(pr[0]), float(pr[1])
            if not p_min < p_max:
                raise ValueError(
                    f"peak_ranges['{var}'] needs min < max, got {pr!r}."
                )
            v_min, v_max = bounds[var]
            if p_min < v_min or p_max > v_max:
                raise ValueError(
                    f"peak_ranges['{var}']={pr!r} lies outside the sampling "
                    f"bounds {(v_min, v_max)}."
                )

        # Warp only the sampled variables
        sampled_columns = []
        for i, var in enumerate(vars_to_sample):
            warped = BaseRing.Backpack._warp_variable(
                u[:, i],
                bounds[var],
                peak_range=peak_ranges.get(var, None),
                range_sigma=range_sigma,
            )
            sampled_columns.append(warped)

        sampled_np = np.column_stack(sampled_columns)

        # Final tensor respecting the original bounds order
        final_columns = []
        for var in bounds.keys():
            if var in vars_to_sample:
                idx = vars_to_sample.index(var)
                final_columns.append(sampled_np[:, idx])
            else:
                final_columns.append(np.full(n_samples, vars_constant[var], dtype=float))

        self.case_tensor = np.column_stack(final_columns)
        self.design_vars = list(bounds.keys())
        self.df_cases = pd.DataFrame(
            data=self.case_tensor, columns=self.design_vars, dtype=np.float64
        )

    def add_param(
        self,
        name: str = "param",
        data: Optional[np.ndarray] = None
    ) -> None:
        """
        Add a new parameter to the cases DataFrame. The data must be provided as a external numpy array. For calculated data, use compute_param().
        """
        if self.df_cases is None:
            raise RuntimeError(
                "Cases must be defined before adding parameters. "
                "Use define_cases()."
            )
        if data is None:
            raise ValueError('Data must be provided to add the parameter.')
        arr = np.asarray(data, dtype=float).ravel()
        if arr.shape[0] != len(self.df_cases):
            raise ValueError(
                f"Data has {arr.shape[0]} rows but there are "
                f"{len(self.df_cases)} cases."
            )
        df = self.df_cases.copy()
        df[name] = arr
        self.df_cases = df

    def compute_param(
        self, name: str = 'param', formula: Optional[str] = None,
        externals: Optional[dict] = None
    ) -> None:
        """
        Compute a new parameter based on a formula and add it to the cases DataFrame.

        The formula is evaluated by a restricted expression parser (no
        ``eval``): column names, ``externals`` entries, numeric constants,
        ``+ - * / ** %`` and the functions ``sin cos exp log sqrt`` plus
        the ``np`` namespace (``np.sin`` etc.) and ``pi``.

        Args:
            name (str): Name of the new parameter.
            formula (str): Formula to compute the parameter, using DataFrame column names as variables.
            externals (dict): External variables to include in the formula.

        Raises:
            ValueError: If a name in the formula is neither a column, an
                external, nor an allowed function/constant.
        """

        df = self.df_cases
        if df is None:
            raise RuntimeError(
                "Cases must be defined before computing parameters. "
                "Use define_cases()."
            )

        if formula is None:
            raise ValueError('A formula must be provided to compute the parameter.')

        env = {}

        # DataFrame variables
        for col in df.columns:
            env[col] = df[col].values

        # External variables (scalar or array)
        if externals is not None:
            for key, value in externals.items():
                if np.isscalar(value):
                    env[key] = value
                else:
                    env[key] = np.asarray(value)

        allowed_funcs = {
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
        }

        try:
            param = self._eval_formula(formula, env, allowed_funcs)
        except Exception as e:
            log.error("Error evaluating formula: %s (%s)", formula, e)
            raise ValueError(
                f"Error evaluating formula: {formula} ({e})"
            ) from e

        arr = np.asarray(param, dtype=float).ravel()
        if arr.shape[0] != len(df):
            raise ValueError(
                f"Formula produced {arr.shape[0]} values but there are "
                f"{len(df)} cases."
            )
        df = df.copy()
        df[name] = arr
        self.df_cases = df

    @staticmethod
    def _eval_formula(formula: str, env: dict, allowed_funcs: dict):
        """Evaluate an arithmetic expression without ``eval``.

        Thin delegate to :func:`SAM.Backpack.eval_formula`, where the
        implementation now lives so that other subpackages (notably
        ``sets``) can reuse it without importing ``rings``.

        The import is local so that ``rings`` does not pull SAM's heavy
        dependency stack (torch, pyvista, sklearn) at module import time.
        """
        from ..sam import SAM

        return SAM.Backpack.eval_formula(formula, env, allowed_funcs)

    # ── Folder / metadata writing shared by every ring ────────────────────

    @staticmethod
    def _expand_nodelist(spec: str) -> list:
        """Expand a Slurm nodelist spec into plain node names.

        Handles ``n003``, comma lists and bracket ranges such as
        ``n[003-005]`` or ``n[001,003]``. Anything unparseable is kept
        verbatim so matching degrades gracefully instead of failing.
        """
        if not spec or spec == "(null)":
            return []
        nodes = []
        # Split on commas that are not inside [...] ranges.
        chunks, depth, current = [], 0, ""
        for ch in spec:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                chunks.append(current)
                current = ""
            else:
                current += ch
        chunks.append(current)
        token_re = re.compile(r"([A-Za-z_-]*)(\[[^\]]+\])?")
        for chunk in chunks:
            chunk = chunk.strip()
            m = token_re.fullmatch(chunk)
            if not m or not m.group(2):
                nodes.append(chunk)
                continue
            prefix, bracket = m.group(1), m.group(2)[1:-1]
            for part in bracket.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-", 1)
                    width = max(len(start), len(end))
                    try:
                        for i in range(int(start), int(end) + 1):
                            nodes.append(f"{prefix}{i:0{width}d}")
                    except ValueError:
                        nodes.append(f"{prefix}[{part}]")
                else:
                    nodes.append(f"{prefix}{part}")
        return nodes
    @staticmethod
    def _job_name(caso: str) -> str:
        """Derive a short Slurm ``%NAME`` from a case folder name.

        ``name_value`` pairs become ``<initial><value:.2f>`` (e.g.
        ``aoa_3.50`` → ``a3.50``); anything else falls back to the
        folder name with non-alphanumeric characters stripped.
        """
        try:
            separado = caso.split('_')
            if len(separado) % 2 != 0:
                raise ValueError("odd number of '_'-separated tokens")
            return ''.join(
                separado[i][0] + f"{float(separado[i + 1]):.2f}"
                for i in range(0, len(separado), 2)
            )
        except (ValueError, IndexError):
            return re.sub(r'[^0-9A-Za-z]+', '', caso)

    def _write_cases_metadata(
        self, df_cases: pd.DataFrame, extra: Optional[dict] = None
    ) -> None:
        """Persist folder_fmt/design_vars/df_cases for FRODO readers."""
        metadata_path = os.path.join(self.root_dir, 'metadata', 'cases_metadata.json')
        metadata = {
            'root_dir': self.root_dir,
            'eq_type': self.eq_type,
            'folder_fmt': self.folder_fmt,
            'design_vars': self.design_vars,
            'num_stages': self.num_stages,
            'df_cases': df_cases.to_dict(orient='list'),
        }
        if extra:
            metadata.update(extra)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        self.df_cases = df_cases
        self.df_cases.to_csv(os.path.join(self.root_dir, 'metadata', 'df_cases.csv'), sep=',', index=False)

    def assign_jobs(
        self, file_sh: str,
        nodes: list,
        cpus_per_job: int,
        submit: bool = False
    ) -> None:
        """
        Assign cases to available nodes in a balanced way.

        Args:
            file_sh (str): Name of the shell script file to modify for each case.
            nodes (list[str]): List of available node names.
            cpus_per_job (int): Number of CPUs to assign per job.
            submit (bool): If True, modify the shell script files with assigned node and CPUs.
        """
        casos = getattr(self, 'folders_name', None)
        self.file_sh = file_sh
        if not casos:
            raise RuntimeError(
                "No cases found. Run generate_folders() first."
            )
        if (
            self.df_cases is None
            or 'folder' not in self.df_cases.columns
            or 'exist' not in self.df_cases.columns
        ):
            raise RuntimeError(
                "df_cases has no 'folder'/'exist' columns. "
                "Run generate_folders() first."
            )
        cpus_por_nodo = {}
        output_dir = os.path.join(self.root_dir, 'outputs')

        log.info("--- Trabajos en ejecución ---")
        _ = BaseRing.Backpack.squeue_terminal(opt=[['-t', 'R'], ])

        log.info("--- Estado de los nodos ---")
        sinfo = BaseRing.Backpack.sinfocpu_terminal()
        if not sinfo or not sinfo.strip():
            raise RuntimeError(
                "Could not query Slurm nodes ('sinfo' gave no output). "
                "Are you on a Slurm cluster?"
            )

        for linea in sinfo.strip().split("\n")[1:]:
            partes = linea.split()
            if len(partes) < 2:
                continue
            nodo = partes[0]
            if nodo not in nodes:
                continue
            try:
                # Expected format: "n005  alloc/idle/other/total"
                datos_cpu = partes[1].split("/")
                cpus_disponibles = int(datos_cpu[1])
                cpus_por_nodo[nodo] = cpus_disponibles
            except (IndexError, ValueError):
                log.warning("Formato inesperado en línea: %s", linea)

        if not cpus_por_nodo:
            raise ValueError("No se encontraron nodos válidos en la lista filtrada.")

        nodos_ordenados = sorted(cpus_por_nodo.items(), key=lambda x: x[1], reverse=True)
        asignaciones = {nodo: [] for nodo, _ in nodos_ordenados}

        for caso in casos:
            exists = self.df_cases.loc[
                self.df_cases['folder'] == caso,
                'exist'
            ]

            if not exists.empty and bool(exists.iloc[0] is True):
                log.info("Folder %s already exists. Skipping...", caso)
                continue
            nodo_menos_cargado = min(
                asignaciones.keys(),
                key=lambda n: len(asignaciones[n])
            )
            asignaciones[nodo_menos_cargado].append(
                (caso, cpus_per_job)
            )

        log.info("Asignaciones completadas:")
        for nodo, trabajos in asignaciones.items():
            log.info("%s: %s tareas", nodo, len(trabajos))
            for caso, cpus in trabajos:
                log.info("  └─ %s (%s CPUs)", caso, cpus)
                if submit:
                    run_sh_path = os.path.join(output_dir, caso, file_sh)
                    try:
                        with open(run_sh_path, "r") as f:
                            contenido = f.read()
                    except OSError as e:
                        log.warning(
                            "Execution file %s not found (%s). Skipping...",
                            run_sh_path, e,
                        )
                        continue

                    contenido = contenido.replace("%NODO", f"{nodo}")
                    contenido = contenido.replace(
                        "%NAME", self._job_name(caso)
                    )
                    contenido = contenido.replace("%CPUs", str(cpus))

                    with open(run_sh_path, "w") as f:
                        f.write(contenido)

                    log.info(
                        "Modificado %s → Nodo: %s, CPUs: %s",
                        os.path.basename(run_sh_path), nodo, cpus,
                    )

    def submit_cases(self) -> None:
        """
        Submit all case jobs using their respective shell scripts.
        """
        output_dir = os.path.join(self.root_dir, 'outputs')
        casos = getattr(self, 'folders_name', None)
        if not casos:
            raise RuntimeError(
                "No cases found. Run generate_folders() first."
            )
        if not getattr(self, 'file_sh', None):
            raise RuntimeError(
                "No shell script selected. Run assign_jobs() first."
            )

        log.info("--- Submitting jobs ---")
        for caso in casos:
            run_sh_path = os.path.join(output_dir, caso, self.file_sh)
            case_dir = os.path.join(output_dir, caso)
            log.info(run_sh_path)
            if not os.path.isfile(run_sh_path):
                log.warning("Execution file %s not found. Skipping...", run_sh_path)
                continue

            try:
                result = subprocess.run(
                    ["sbatch", run_sh_path],
                    capture_output=True,
                    cwd=case_dir,
                    text=True,
                    check=True
                )

                log.info("%s: %s", caso, result.stdout.strip())
            except FileNotFoundError:
                raise RuntimeError(
                    "'sbatch' not found. Are you on a Slurm cluster?"
                )
            except subprocess.CalledProcessError as e:
                log.warning("%s: %s", caso, (e.stderr or "").strip())

    def recover_pending_jobs(
        self,
        broken_node: str,
        replacement_node: str,
        file_sh: Optional[str] = None,
        dry_run: bool = True,
    ):
        """
        Reassign pending Slurm jobs that are constrained to a failed node.

        The cases are identified using the metadata stored in
        ``<root_dir>/metadata/cases_metadata.json`` and the working directory
        reported by Slurm for each pending job.

        The method:
            1. Reads the simulation metadata.
            2. Searches the user's PENDING Slurm jobs.
            3. Identifies jobs requesting ``broken_node``.
            4. Associates each job with its simulation directory.
            5. Shows the proposed changes if ``dry_run=True``.
            6. Otherwise cancels the old job, modifies its ``run.sh`` and
            resubmits it with ``sbatch``.

        Args:
            broken_node (str):
                Node that is unavailable, e.g. ``"n003"``.

            replacement_node (str):
                Node to which the affected simulations will be reassigned,
                e.g. ``"n005"``.

            file_sh (str):
                Name of the Slurm submission script inside each simulation
                folder. Defaults to the ``file_sh`` given to
                :meth:`assign_jobs`, falling back to ``"run.sh"``.

            dry_run (bool):
                If True, only display the proposed operations.
                If False, actually cancel, modify and resubmit the jobs.

        Returns:
            list[dict]:
                Information about the jobs found and, when ``dry_run=False``,
                their resubmission.
        """

        if not broken_node:
            raise ValueError("broken_node must be provided.")

        if not replacement_node:
            raise ValueError("replacement_node must be provided.")

        if broken_node == replacement_node:
            raise ValueError(
                "broken_node and replacement_node must be different."
            )

        if file_sh is None:
            file_sh = getattr(self, 'file_sh', None) or "run.sh"

        # ==============================================================
        # 1. Load simulation metadata
        # ==============================================================

        metadata_path = os.path.join(
            self.root_dir,
            "metadata",
            "cases_metadata.json",
        )

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Simulation metadata not found: {metadata_path}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        required_metadata = {
            "root_dir",
            "folder_fmt",
            "design_vars",
            "df_cases",
        }

        missing = required_metadata - metadata.keys()

        if missing:
            raise ValueError(
                f"Invalid cases_metadata.json. Missing fields: "
                f"{sorted(missing)}"
            )

        # ==============================================================
        # 2. Define simulation directory
        # ==============================================================

        output_dir = os.path.join(
            self.root_dir,
            "outputs",
        )

        if not os.path.isdir(output_dir):
            raise FileNotFoundError(
                f"Simulation output directory not found: {output_dir}"
            )

        # ==============================================================
        # 3. Reconstruct valid case directories from metadata
        # ==============================================================

        folder_fmt = metadata["folder_fmt"]
        design_vars = metadata["design_vars"]
        df_cases = pd.DataFrame(metadata["df_cases"])

        case_dirs = {}

        for _, row in df_cases.iterrows():

            try:
                case_dict = {
                    var: float(row[var])
                    for var in design_vars
                }
                folder_name = folder_fmt.format(**case_dict)
            except KeyError as e:
                raise ValueError(
                    f"Variable {e} from folder_fmt was not found in "
                    f"design_vars: {design_vars}"
                ) from e
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Could not build a folder name from row "
                    f"{row.to_dict()}: {e}"
                ) from e

            case_dir = os.path.join(
                output_dir,
                folder_name,
            )

            case_dirs[os.path.realpath(case_dir)] = {
                "folder": folder_name,
                "path": case_dir,
            }

        # ==============================================================
        # 4. Header
        # ==============================================================

        log.info("=" * 70)
        log.info("SLURM PENDING JOB RECOVERY")
        log.info("=" * 70)

        log.info("Root directory    : %s", self.root_dir)
        log.info("Broken node       : %s", broken_node)
        log.info("Replacement node  : %s", replacement_node)
        log.info("Script            : %s", file_sh)

        if dry_run:
            log.info("Mode              : DRY RUN (no changes will be made)")
        else:
            log.info("Mode              : EXECUTION")

        log.info("=" * 70)

        # ==============================================================
        # 5. Get user's pending jobs
        # ==============================================================

        user = os.environ.get("USER", "")
        if not user:
            raise RuntimeError(
                "Could not determine the user ('$USER' is empty); "
                "cannot query Slurm jobs."
            )
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "-h",
                    "-u",
                    user,
                    "-t",
                    "PENDING",
                    "-o",
                    "%A",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "'squeue' not found. Are you on a Slurm cluster?"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Could not obtain pending Slurm jobs:\n"
                f"{e.stderr.strip()}"
            ) from e

        job_ids = [
            job_id.strip()
            for job_id in result.stdout.splitlines()
            if job_id.strip()
        ]

        if not job_ids:
            log.info("No pending jobs found.")
            return []

        # ==============================================================
        # 6. Inspect pending jobs
        # ==============================================================

        affected_jobs = []

        for job_id in job_ids:

            try:
                result = subprocess.run(
                    [
                        "scontrol",
                        "show",
                        "job",
                        "-o",
                        job_id,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "'scontrol' not found. Are you on a Slurm cluster?"
                )
            except subprocess.CalledProcessError:
                log.warning(
                    "Could not inspect job %s. Skipping.", job_id
                )
                continue

            job_info = result.stdout.strip()

            # ----------------------------------------------------------
            # Extract Slurm information
            # ----------------------------------------------------------

            state_match = re.search(
                r"\bJobState=([^\s]+)",
                job_info,
            )

            req_node_match = re.search(
                r"\bReqNodeList=([^\s]+)",
                job_info,
            )

            workdir_match = re.search(
                r"\bWorkDir=(\S+)",
                job_info,
            )

            job_name_match = re.search(
                r"\bJobName=(\S+)",
                job_info,
            )

            if state_match is None:
                continue

            if state_match.group(1) != "PENDING":
                continue

            if req_node_match is None:
                continue

            requested_nodes = self._expand_nodelist(req_node_match.group(1))

            if broken_node not in requested_nodes:
                continue

            if workdir_match is None:
                log.warning(
                    "Job %s requests %s, but WorkDir could not "
                    "be determined. Skipping.",
                    job_id, broken_node,
                )
                continue

            workdir = os.path.realpath(
                workdir_match.group(1)
            )

            # ----------------------------------------------------------
            # Verify that WorkDir belongs to this dataset
            # ----------------------------------------------------------

            if workdir not in case_dirs:

                log.warning(
                    "Job %s has WorkDir outside the cases described "
                    "by metadata: %s. Skipping.",
                    job_id, workdir,
                )

                continue

            case_info = case_dirs[workdir]

            run_sh_path = os.path.join(
                workdir,
                file_sh,
            )

            if not os.path.isfile(run_sh_path):
                log.warning(
                    "run script not found for job %s: %s",
                    job_id, run_sh_path,
                )
                continue

            job_name = (
                job_name_match.group(1)
                if job_name_match is not None
                else "unknown"
            )

            affected_jobs.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "state": "PENDING",
                    "requested_node": req_node_match.group(1),
                    "folder": case_info["folder"],
                    "workdir": workdir,
                    "run_sh": run_sh_path,
                }
            )

        # ==============================================================
        # 7. Nothing to recover
        # ==============================================================

        if not affected_jobs:

            log.info(
                "No PENDING jobs requesting %s were found in this dataset.",
                broken_node,
            )

            return []

        # ==============================================================
        # 8. Show affected jobs
        # ==============================================================

        log.info("Found %s affected jobs:", len(affected_jobs))

        for job in affected_jobs:

            log.info(
                "  Job %8s | %s",
                job['job_id'], job['folder'],
            )

            log.info(
                "             WorkDir : %s",
                job['workdir'],
            )

            log.info(
                "             Node    : %s → %s",
                job['requested_node'], replacement_node,
            )

        # ==============================================================
        # 9. DRY RUN
        # ==============================================================

        if dry_run:

            log.info("-" * 70)
            log.info("DRY RUN — proposed operations")
            log.info("-" * 70)

            for job in affected_jobs:

                log.info("Job %s — %s", job['job_id'], job['folder'])

                log.info("  1. Cancel: scancel %s", job['job_id'])

                log.info("  2. Modify: %s", job['run_sh'])

                log.info("     #SBATCH --nodelist=%s", broken_node)
                log.info("       ↓")
                log.info("     #SBATCH --nodelist=%s", replacement_node)
                log.info("  3. Submit: sbatch %s", job['run_sh'])

            log.info("-" * 70)
            log.info("No changes have been made.")
            log.info("Use dry_run=False to execute the operations.")
            log.info("-" * 70)

            return affected_jobs

        # ==============================================================
        # 10. EXECUTION
        # ==============================================================

        log.info("-" * 70)
        log.info("Executing recovery")
        log.info("-" * 70)

        recovered_jobs = []

        for job in affected_jobs:

            job_id = job["job_id"]
            run_sh_path = job["run_sh"]
            case_dir = job["workdir"]

            log.info("--- %s (job %s) ---", job['folder'], job_id)

            # ----------------------------------------------------------
            # 10.1 Patch run.sh first: cancelling before knowing the
            # script can be rewritten would lose the job.
            # ----------------------------------------------------------

            try:

                with open(run_sh_path, "r") as f:
                    content = f.read()

                pattern = (
                    r"^(\s*#SBATCH\s+--nodelist=)"
                    + re.escape(broken_node)
                    + r"(\s*(?:#.*)?)$"
                )

                new_content, n_replacements = re.subn(
                    pattern,
                    rf"\g<1>{replacement_node}\g<2>",
                    content,
                    flags=re.MULTILINE,
                )

                if n_replacements == 0:

                    log.warning(
                        "Could not find #SBATCH --nodelist=%s in %s; "
                        "leaving job %s untouched.",
                        broken_node, run_sh_path, job_id,
                    )

                    continue

                with open(run_sh_path, "w") as f:
                    f.write(new_content)

                log.info(
                    "Modified %s: %s → %s",
                    file_sh, broken_node, replacement_node,
                )

            except OSError as e:

                log.warning(
                    "Could not modify %s: %s",
                    run_sh_path, e,
                )

                continue

            # ----------------------------------------------------------
            # 10.2 Cancel old job
            # ----------------------------------------------------------

            try:

                subprocess.run(
                    ["scancel", job_id],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                log.info("Cancelled job %s", job_id)

            except FileNotFoundError:
                raise RuntimeError(
                    "'scancel' not found. Are you on a Slurm cluster?"
                )
            except subprocess.CalledProcessError as e:

                log.warning(
                    "Could not cancel job %s: %s",
                    job_id, (e.stderr or "").strip(),
                )

                continue

            # ----------------------------------------------------------
            # 10.3 Submit modified script
            # ----------------------------------------------------------

            try:

                result = subprocess.run(
                    [
                        "sbatch",
                        run_sh_path,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=case_dir,
                    check=True,
                )
                submission = result.stdout.strip()

                log.info("Resubmitted: %s", submission)

                recovered_jobs.append(
                    {
                        **job,
                        "replacement_node": replacement_node,
                        "submission": submission,
                    }
                )

            except FileNotFoundError:
                raise RuntimeError(
                    "'sbatch' not found. Are you on a Slurm cluster?"
                )
            except subprocess.CalledProcessError as e:

                log.warning(
                    "Could not submit %s: %s",
                    run_sh_path, (e.stderr or "").strip(),
                )

        # ==============================================================
        # 11. Summary
        # ==============================================================

        log.info("=" * 70)
        log.info("RECOVERY SUMMARY")
        log.info("=" * 70)

        log.info("Jobs found       : %s", len(affected_jobs))

        log.info("Jobs recovered   : %s", len(recovered_jobs))

        log.info("Broken node      : %s", broken_node)

        log.info("Replacement node : %s", replacement_node)

        log.info("=" * 70)

        return recovered_jobs

    class Backpack:

        @staticmethod
        def _warp_variable(u, bounds, peak_range=None, range_sigma=3):
            """
            Apply a non-linear warp to u in [0,1] to concentrate points
            inside a range (peak_range). With peak_range None, linear mapping.
            """
            v_min, v_max = bounds

            if peak_range is None:
                return v_min + u * (v_max - v_min)

            p_min, p_max = peak_range
            mean = (p_min + p_max) / 2
            std = (p_max - p_min) / range_sigma

            a, b = (v_min - mean) / std, (v_max - mean) / std
            trunc_gauss = truncnorm(a=a, b=b, loc=mean, scale=std)

            u_clipped = np.clip(u, 1e-6, 1 - 1e-6)
            return trunc_gauss.ppf(u_clipped)

        @staticmethod
        def squeue_terminal(opt: Union[list, tuple]):
            """
            Show the Slurm queue in the terminal with the given options.
            Args:
                opt (list[list] | tuple[tuple] | list[tuple] | tuple[list]): Options for the squeue command.
            """
            try:
                args = ["squeue"]
                if isinstance(opt, (list, tuple)):
                    for item in opt:
                        if isinstance(item, (list, tuple)):
                            args.extend(item)
                        else:
                            args.append(item)
                else:
                    raise ValueError("opt must be a list or tuple of lists/tuples.")

                resultado = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    check=True
                )
                log.info(resultado.stdout)
                return resultado.stdout
            except Exception as e:
                log.warning("Error ejecutando squeue: %s", e)

        @staticmethod
        def sinfocpu_terminal():
            """Query Slurm node CPU states (``sinfo -N -o '%n %C'``)."""
            try:
                resultado = subprocess.run(
                    ["sinfo", "-N", "-o", "%n  %C"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                log.info(resultado.stdout)
                return resultado.stdout

            except Exception as e:
                log.warning("Error ejecutando sinfo: %s", e)

        @staticmethod
        def normalize_airfoil(coords: Union[np.ndarray, "torch.Tensor"]):
            """
            Normalise an Nx2 airfoil to chord 1 with the leading edge at x = 0.
            Works on a copy: the input array is never modified.
            """
            import torch as _torch

            arr = np.asarray(coords, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("El tensor debe tener forma [N, 2].")

            x_min = arr[:, 0].min()
            x_max = arr[:, 0].max()
            chord = x_max - x_min

            arr[:, 0] -= x_min
            coords_scaled = arr / chord

            if isinstance(coords, _torch.Tensor):
                return _torch.as_tensor(coords_scaled)
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

            # Constants
            R = 287.05
            g0 = 9.80665
            P0 = 101325.0
            T0 = 288.15

            h_layers = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852]
            L = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002]

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
                    ht = h_layers[j + 1]
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
            Sutherland viscosity law with reference values.
            """
            return mu0 * (T / Treference) ** 1.5 * (Treference + 110.4) / (T + 110.4)
