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
        default_workdir: Union[str, Path, None] = None,
        default_parser: Union[Callable, None] = None,
        timeout: Union[float, None] = 120,
    ):
        self.default_workdir = Path(default_workdir) if default_workdir else Path(".")
        self.default_parser = default_parser
        self.timeout = timeout
 
        # Almacén principal de resultados
        # outputs[run_id] = RunResult
        self.outputs: dict[str, RunResult] = {}
    
    def _inject_cpu_env(self, env: dict, n_cpus: int) -> dict:
        """
        Configura variables de entorno para paralelismo.
        """
        env = env.copy() if env else {}

        env["OMP_NUM_THREADS"] = str(n_cpus)
        env["OPENBLAS_NUM_THREADS"] = str(n_cpus)
        env["MKL_NUM_THREADS"] = str(n_cpus)
        env["NUMEXPR_NUM_THREADS"] = str(n_cpus)

        return env

    def get_available_cpus(self):
        """
        Detecta CPUs disponibles (SLURM-aware).
        """
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        elif "SLURM_JOB_CPUS_PER_NODE" in os.environ:
            return int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
        else:
            return os.cpu_count()
        
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
        n_cpus: Union[int, None] = None
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
        program     : ejecutable a lanzar.
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
        n_cpus      : si se especifica, inyecta variables de entorno para limitar el número de CPUs usados por el proceso (OMP_NUM_THREADS, etc.).
 
        Retorna
        -------
        RunResult con el resultado completo de la ejecución.
        """
        if n_cpus is not None:
            env = self._inject_cpu_env(env, n_cpus)
        # Resolver parámetros con herencia de defaults
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
