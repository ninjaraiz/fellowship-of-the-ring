import os
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Callable, Union

log = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Tipos de script aceptados
# ---------------------------------------------------------------------------
# • str              → se escribe tal cual al fichero de entrada
# • list[str]        → se unen con '\\n' y se escriben
# • Path             → se lee el fichero y se usa su contenido
# • objeto           → se llama a .build_commands() y el resultado (list[str])
#                      se une con '\\n'   [compatibilidad con XFoilCase, etc.]

ScriptLike = Union[str, list, Path, Any]

class RunResult:
    """
    Contenedor ligero para el resultado de una ejecución.
 
    Atributos
    ---------
    run_id      : identificador de la ejecución
    program     : ejecutable usado
    returncode  : código de retorno del proceso
    stdout      : salida estándar en bruto
    stderr      : salida de error en bruto
    elapsed     : tiempo de ejecución en segundos
    data        : dict con los datos parseados (vacío si no hay parser)
    success     : True si returncode == 0
    """
 
    def __init__(
        self,
        run_id: str,
        program: str,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed: float,
        data: dict,
    ):
        self.run_id = run_id
        self.program = program
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed
        self.data = data
 
    @property
    def success(self) -> bool:
        return self.returncode == 0
 
    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERR({self.returncode})"
        return (
            f"<RunResult id={self.run_id!r} program={self.program!r} "
            f"status={status} t={self.elapsed:.2f}s "
            f"data_keys={list(self.data.keys())}>"
        )
    
class ARAGORN:
    """
    ARAGORN — Automated Runner for Aerodynamic and General Operational
            Research Numerical-solvers.
    Ejecutor genérico de programas de terminal. Recibe un script con las
    órdenes del usuario, el ejecutable a usar y, opcionalmente, una función
    de parseo para estructurar los resultados. Los datos obtenidos se
    almacenan internamente en la instancia.
    
    
    Uso básico
    ----------
    >>> runner = ARAGORN()
    >>> result = runner.run(
    ...     script=my_xfoil_script,   # str | list[str] | Path | objeto con .build_commands()
    ...     program="xfoil",
    ...     run_id="naca2412_Re1e6",
    ...     parser=my_parser_fn,      # callable(stdout, workdir) -> dict
    ... )
    >>> runner.outputs["naca2412_Re1e6"]  # datos parseados

    El ciclo de vida de una ejecución es:
        1. _resolve_script()  →  convierte ScriptLike a str
        2. _write_script()    →  escribe el script en un fichero temporal
        3. _execute()         →  lanza el proceso con subprocess
        4. _parse()           →  aplica el parser al stdout + workdir
        5. _store()           →  guarda RunResult en self.outputs
 
    Parámetros de instancia
    -----------------------
    default_program : programa por defecto si run() no especifica uno
    default_workdir : directorio de trabajo por defecto ('.' si None)
    default_parser  : parser por defecto si run() no especifica uno
    timeout         : segundos máximos de espera por ejecución (None = sin límite)
    """
 
    def __init__(
        self,
        default_program: str = "xfoil",
        default_workdir: Union[str, Path, None] = None,
        default_parser: Union[Callable, None] = None,
        timeout: Union[float, None] = 120.0,
    ):
        self.default_program = default_program
        self.default_workdir = Path(default_workdir) if default_workdir else Path(".")
        self.default_parser = default_parser
        self.timeout = timeout
 
        # Almacén principal de resultados
        # outputs[run_id] = RunResult
        self.outputs: dict[str, RunResult] = {}
 
    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------
 
    def run(
        self,
        script: ScriptLike,
        program: Union[str, None] = None,
        run_id: Union[str, None] = None,
        parser: Union[Callable, None] = None,
        workdir: Union[str, Path, None] = None,
        env: Union[dict, None] = None,
        stdin_mode: bool = True,
        extra_args: Union[list, None] = None,
        overwrite: bool = True,
    ) -> RunResult:
        """
        Ejecuta un programa con el script dado.
 
        Parámetros
        ----------
        script      : comandos para el programa. Acepta:
                        - str         (script completo)
                        - list[str]   (líneas de comandos)
                        - Path        (ruta a un fichero de script)
                        - cualquier objeto con método .build_commands() -> list[str]
        program     : ejecutable a lanzar (hereda default_program si None)
        run_id      : clave para self.outputs. Se genera automáticamente si None.
        parser      : callable(stdout: str, workdir: Path) -> dict
                      Recibe la salida estándar y el directorio de trabajo
                      (donde el programa habrá escrito sus ficheros de salida)
                      y devuelve un diccionario con los datos extraídos.
                      Hereda default_parser si None.
        workdir     : directorio donde correrá el proceso. Si None, usa
                      default_workdir. El directorio se crea si no existe.
        env         : variables de entorno adicionales para el proceso.
        stdin_mode  : si True (por defecto), el script se pasa por stdin.
                      Si False, se pasa como primer argumento posicional.
        extra_args  : argumentos adicionales a la llamada del ejecutable.
        overwrite   : si False y run_id ya existe en outputs, lanza ValueError.
 
        Retorna
        -------
        RunResult con el resultado completo de la ejecución.
        """
        # Resolver parámetros con herencia de defaults
        program  = program  or self.default_program
        parser   = parser   or self.default_parser
        workdir  = Path(workdir) if workdir else self.default_workdir
        run_id   = run_id   or self._auto_id(program)
 
        if not overwrite and run_id in self.outputs:
            raise ValueError(
                f"run_id '{run_id}' ya existe en outputs. "
                "Usa overwrite=True para sobreescribir."
            )
 
        workdir.mkdir(parents=True, exist_ok=True)
 
        # 1. Resolver el script a un string plano
        script_str = self._resolve_script(script)
 
        # 2. Escribir el script en un fichero temporal
        script_path = self._write_script(script_str, workdir)
 
        # 3. Ejecutar el programa
        t0 = time.perf_counter()
        proc = self._execute(
            program=program,
            script_path=script_path,
            workdir=workdir,
            env=env,
            stdin_mode=stdin_mode,
            extra_args=extra_args or [],
        )
        elapsed = time.perf_counter() - t0
 
        # 4. Parsear resultados
        data = self._parse(parser, proc.stdout, workdir)
 
        # 5. Construir y almacenar RunResult
        result = RunResult(
            run_id=run_id,
            program=program,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed=elapsed,
            data=data,
        )
        self.outputs[run_id] = result
 
        # Limpiar fichero temporal de script
        try:
            script_path.unlink()
        except OSError:
            pass
 
        log.info(
            "run '%s' (%s) → %s  [%.2fs]",
            run_id, program,
            "OK" if result.success else f"ERR({result.returncode})",
            elapsed,
        )
        return result
 
    # ------------------------------------------------------------------
    # Consulta y gestión de outputs
    # ------------------------------------------------------------------
 
    def get(self, run_id: str) -> RunResult:
        """Devuelve el RunResult almacenado para run_id."""
        if run_id not in self.outputs:
            raise KeyError(f"No hay resultado para run_id='{run_id}'")
        return self.outputs[run_id]
 
    def get_data(self, run_id: str) -> dict:
        """Devuelve directamente el dict de datos parseados."""
        return self.get(run_id).data
 
    def list_runs(self) -> list[str]:
        """Lista todos los run_id almacenados."""
        return list(self.outputs.keys())
 
    def clear(self, run_id: Union[str, None] = None) -> None:
        """
        Elimina resultados del almacén.
        Si run_id es None, limpia todos los resultados.
        """
        if run_id is None:
            self.outputs.clear()
        else:
            self.outputs.pop(run_id, None)
 
    def summary(self) -> str:
        """Devuelve un resumen de texto de todas las ejecuciones."""
        if not self.outputs:
            return "ARAGORN: sin ejecuciones almacenadas."
        lines = [f"{'run_id':<40} {'program':<16} {'status':<8} {'t(s)':<8} data_keys"]
        lines.append("-" * 90)
        for rid, r in self.outputs.items():
            status = "OK" if r.success else f"ERR({r.returncode})"
            lines.append(
                f"{rid:<40} {r.program:<16} {status:<8} "
                f"{r.elapsed:<8.2f} {list(r.data.keys())}"
            )
        return "\n".join(lines)
 
    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------
 
    def _resolve_script(self, script: ScriptLike) -> str:
        """
        Convierte cualquier formato de script a un string plano.
 
        - str        → devuelve tal cual
        - list[str]  → une con '\\n'
        - Path       → lee el fichero
        - objeto     → llama a .build_commands() y une con '\\n'
        """
        if isinstance(script, str):
            return script
 
        if isinstance(script, list):
            return "\n".join(str(line) for line in script)
 
        if isinstance(script, Path):
            return script.read_text(encoding="utf-8")
 
        # Protocolo genérico: cualquier objeto que sepa construir comandos
        if hasattr(script, "build_commands"):
            cmds = script.build_commands()
            return "\n".join(str(c) for c in cmds)
 
        raise TypeError(
            f"Tipo de script no soportado: {type(script)}. "
            "Usa str, list[str], Path o un objeto con .build_commands()."
        )
 
    def _write_script(self, script_str: str, workdir: Path) -> Path:
        """
        Escribe el script en un fichero temporal dentro de workdir.
        Retorna la ruta del fichero.
        """
        fd, path = tempfile.mkstemp(
            suffix=".script",
            prefix="aragorn_",
            dir=workdir,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script_str)
        return Path(path)
 
    def _execute(
        self,
        program: str,
        script_path: Path,
        workdir: Path,
        env: Union[dict, None],
        stdin_mode: bool,
        extra_args: list,
    ) -> subprocess.CompletedProcess:
        """
        Lanza el proceso externo.
 
        En stdin_mode=True el script se pasa por stdin (XFoil, Construct2D...).
        En stdin_mode=False el fichero de script se pasa como primer argumento
        (flujos más parecidos a un batch file).
        """
        # Construir entorno del proceso
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
 
        if stdin_mode:
            cmd = [program] + extra_args
            stdin_source = open(script_path, "r", encoding="utf-8")
        else:
            cmd = [program, str(script_path)] + extra_args
            stdin_source = subprocess.DEVNULL
 
        try:
            proc = subprocess.run(
                cmd,
                stdin=stdin_source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workdir,
                env=proc_env,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            log.warning("Timeout (%ss) ejecutando '%s'", self.timeout, program)
            # Devolver un resultado sintético de error para no interrumpir flujos batch
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout=exc.stdout or "",
                stderr=f"TIMEOUT after {self.timeout}s",
            )
        except FileNotFoundError:
            log.error("Ejecutable no encontrado: '%s'", program)
            raise
        finally:
            if stdin_mode and hasattr(stdin_source, "close"):
                stdin_source.close()
 
        return proc
 
    def _parse(
        self,
        parser: Union[Callable, None],
        stdout: str,
        workdir: Path,
    ) -> dict:
        """
        Aplica el parser si existe. Captura excepciones para no interrumpir
        el flujo aunque el parseo falle.
 
        El parser recibe (stdout: str, workdir: Path) y debe devolver un dict.
        """
        if parser is None:
            return {}
 
        try:
            data = parser(stdout, workdir)
            if not isinstance(data, dict):
                log.warning("El parser debe devolver un dict, recibido: %s", type(data))
                return {"raw": data}
            return data
        except Exception as exc:
            log.error("Error en el parser: %s", exc, exc_info=True)
            return {"parse_error": str(exc)}
 
    def _auto_id(self, program: str) -> str:
        """Genera un run_id único basado en el programa y el timestamp."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{Path(program).stem}_{ts}"
 
    # ------------------------------------------------------------------
    # Protocolo de representación
    # ------------------------------------------------------------------
 
    def __repr__(self) -> str:
        return (
            f"<ARAGORN program={self.default_program!r} "
            f"runs={len(self.outputs)} "
            f"timeout={self.timeout}s>"
        )
 
    def __len__(self) -> int:
        return len(self.outputs)
 
    def __contains__(self, run_id: str) -> bool:
        return run_id in self.outputs
    
# class XFoilCase:
    
#     def __init__(
#             self, workdir:Union[str, None],
#             save_geom:bool = True,
#             visc_condition: Literal['incompressible', 'compressible'] = 'compressible',
#             ):
#         # Airfoil
#         self._mode = None
#         self._naca = None
#         self._file = None
#         self._name = "airfoil"
#         self.save_geom = save_geom
#         self.visc_condition = visc_condition

#         # # Condiciones
#         # if self.visc_condition == 'incompressible':
#         #     self.re = None
#         #     self.mach = None
#         # elif self.visc_condition == 'compressible':
#         #     self.re = 1e6
#         #     self.mach = 0.0
        
#         self.alpha_start = -5
#         self.alpha_end = 10
#         self.alpha_step = 1.0
#         self.iter = 100

#         # Outputs
#         self.workdir = workdir if workdir is not None else "xfoil_out"
#         os.makedirs(self.workdir, exist_ok=True)

#         self.foil_file = "airfoil.dat"
#         self.polar_file = "polar.dat"
#         self.cp_file = "cp.dat"
#         self.bl_file = "bl.dat"

#     def naca(self, code: str):
#         self._mode = "naca"
#         self._naca = code
#         self._name = f"NACA{code}"
#         return self

#     def load(self, filepath: str, name="airfoil"):
#         self._mode = "file"
#         self._file = filepath
#         self._name = name
#         return self

#     def _fullpath(self, filename):
#         return os.path.join(self.workdir, filename)

#     def set_conditions(self, re=1e6, mach=0.0, alpha=(-5, 10, 1)):

#         self.re = re
#         self.mach = mach

#         if isinstance(alpha, (tuple, list)):
#             self.alpha_start, self.alpha_end, self.alpha_step = alpha
#         elif isinstance(alpha, int):
#             self.alpha_start = alpha
#             self.alpha_end = alpha
#             self.alpha_step = 1
#         else:
#             raise ValueError("alpha must be a tuple (start, end, step) or an int for a single alpha.")
#         return self

#     def get_executable(self):
#         return "xfoil"
    
#     def build_commands(self):
#         cmds = []

#         # Airfoil
#         if self._mode == "naca":
#             cmds.append(f"NACA {self._naca}")
#         elif self._mode == "file":
#             cmds += [f"LOAD {self._file}", self._name]
#         else:
#             raise ValueError("Define un airfoil")

#         if self.save_geom:
#             cmds.append(f"SAVE {self.foil_file}")
        
#         cmds.append("PANE")

#         # Operación
#         if self.visc_condition == 'compressible':
#             list_visc = [
#                 "OPER",
#                 f"Visc {self.re}",
#                 f"MACH {self.mach}",
#                 f"ITER {self.iter}",
#             ]
#         else:
#             list_visc = [
#                 "OPER",
#                 f"MACH {self.mach}",
#                 f"ITER {self.iter}",
#             ]

#         cmds += list_visc

#         # Polar
#         cmds += [
#             "PACC",
#             self.polar_file,
#             ""
#         ]

#         for a in np.arange(self.alpha_start,
#                         self.alpha_end + self.alpha_step,
#                         self.alpha_step):
#             cmds.append(f"ALFA {a}")

#         cmds += ["PACC", ""]

#         # 🔥 Cp multi-alpha
#         for a in self.cp_alphas:
#             cp_name = f"cp_{a:.2f}.dat"

#             print(a, cp_name)
#             cmds += ["PANE", f"ALFA {a}", f"CPWR {cp_name}"]

#         if self.visc_condition == 'compressible':
#             # Boundary layer dump
#             cmds += [
#                 "DUMP",
#                 self.bl_file
#             ]
#         cmds.append("QUIT")

#         return cmds

#     def get_airfoil(self):
#         filepath = self._fullpath(self.foil_file)

#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"No existe: {filepath}")

#         data = []
#         with open(filepath) as f:
#             lines = f.readlines()[1:]

#         for l in lines:
#             parts = l.split()
#             if len(parts) == 2:
#                 data.append([float(parts[0]), float(parts[1])])

#         return np.array(data)

#     def get_airfoil_df(self):
#         arr = self.get_airfoil()
#         return pd.DataFrame(arr, columns=["x", "y"])

#     def get_polar(self):
#         return pd.read_csv(
#             self._fullpath(self.polar_file),
#             delim_whitespace=True,
#             skiprows=12
#         )
    
#     def get_cp(self):
#         return pd.read_csv(
#             self._fullpath(self.cp_file),
#             delim_whitespace=True,
#             skiprows=2
#         )
    
#     def get_id(self):
#         return f"{self._name}_Re{self.re:.0e}"

#     def get_bl(self):
#         return pd.read_csv(
#             self._fullpath(self.bl_file),
#             delim_whitespace=True
#         )

#     def get_clmax(self):
#         polar = self.get_polar()

#         idx = polar["CL"].idxmax()

#         return {
#             "CLmax": polar.loc[idx, "CL"],
#             "alpha": polar.loc[idx, "alpha"]
#         }
#     def set_cp_alphas(self, alphas):
#         self.cp_alphas = alphas
#         return self

#     def parse_all(self):
#         data = {}

#         # Airfoil
#         data["airfoil"] = self.get_airfoil()

#         # Polar
#         try:
#             data["polar"] = self.get_polar()
#         except:
#             data["polar"] = None

#         # Cp multi
#         cp_data = {}
#         for a in self.cp_alphas:
#             fname = f"cp_{a:.2f}.dat"
#             path = self._fullpath(fname)

#             if os.path.exists(path):
#                 cp_data[a] = pd.read_csv(
#                     path,
#                     delim_whitespace=True,
#                     skiprows=2
#                 )

#         data["cp"] = cp_data

#         # BL (opcional)
#         try:
#             data["bl"] = self.get_bl()
#         except:
#             data["bl"] = None

#         return data

#     def export_airfoil_csv(self, filepath):
#         df = self.get_airfoil_df()
#         df.to_csv(filepath, index=False)

#     def export_airfoil_dat(self, filepath):
#         coords = self.get_airfoil()

#         with open(filepath, "w") as f:
#             f.write(f"{self._name}\n")

#             for x, y in coords:
#                 f.write(f"{x:.6f} {y:.6f}\n")

#     def plot_airfoil(self):
#         import matplotlib.pyplot as plt
#         xy = self.get_airfoil()
#         plt.plot(xy[:, 0], xy[:, 1])
#         plt.axis("equal")
#         plt.title(self._name)
#         plt.show()

#     def plot_polar(self):
#         import matplotlib.pyplot as plt
#         df = self.get_polar()
#         plt.plot(df["alpha"], df["CL"])
#         plt.xlabel("alpha")
#         plt.ylabel("CL")
#         plt.show()

#     def plot_cp(self):
#         import matplotlib.pyplot as plt
#         df = self.get_cp()
#         plt.plot(df["x"], df["Cp"])
#         plt.gca().invert_yaxis()
#         plt.show()

# def xfoil_parser(stdout: str, workdir: Path) -> dict:
#     """
#     Ejemplo de parser compatible con ARAGORN para resultados de XFoil.
 
#     Parámetros
#     ----------
#     stdout  : salida estándar de XFoil
#     workdir : directorio donde XFoil escribió los ficheros de salida
 
#     Retorna
#     -------
#     dict con claves: 'polar', 'cp', 'airfoil', 'bl'
#     """
 
#     data = {}
 
#     # -- Polar (fichero polar.dat generado por PACC)
#     polar_path = workdir / "polar.dat"
#     if polar_path.exists():
#         try:
#             data["polar"] = pd.read_csv(
#                 polar_path,
#                 sep=r"\s+",
#                 skiprows=12,
#                 engine="python",
#             )
#         except Exception:
#             data["polar"] = None
 
#     # -- Distribuciones de Cp (ficheros cp_<alpha>.dat)
#     cp_files = sorted(workdir.glob("cp_*.dat"))
#     cp_dict = {}
#     for cp_path in cp_files:
#         try:
#             # Extraer alpha del nombre: cp_5.00.dat → 5.0
#             alpha_str = cp_path.stem.replace("cp_", "")
#             alpha = float(alpha_str)
#             cp_dict[alpha] = pd.read_csv(
#                 cp_path,
#                 sep=r"\s+",
#                 skiprows=2,
#                 engine="python",
#             )
#         except Exception:
#             pass
#     data["cp"] = cp_dict
 
#     # -- Geometría del perfil (airfoil.dat)
#     geom_path = workdir / "airfoil.dat"
#     if geom_path.exists():
#         try:
#             raw = np.loadtxt(geom_path, skiprows=1)
#             data["airfoil"] = raw
#         except Exception:
#             data["airfoil"] = None
 
#     # -- Capa límite (bl.dat generado por DUMP)
#     bl_path = workdir / "bl.dat"
#     if bl_path.exists():
#         try:
#             data["bl"] = pd.read_csv(bl_path, sep=r"\s+", engine="python")
#         except Exception:
#             data["bl"] = None
 
#     return data