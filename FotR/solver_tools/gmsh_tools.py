from pathlib import Path
import shutil


class GmshCase:
    def __init__(
        self,
        geo_file,
        workdir="gmsh_out",
        mesh_name="mesh.msh",
        dim=3,
        format="msh2",
    ):
        self.geo_file = Path(geo_file)
        self.workdir = Path(workdir)
        self.mesh_name = mesh_name
        self.n_cpus = None
        
        self.dim = dim              # 2 o 3
        self.format = format        # msh2 recomendado
        self.extra_args = []

        self.workdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Configuración fluida (estilo corporate builder)
    # -----------------------------
    def set_dimension(self, dim: int):
        self.dim = dim
        return self

    def set_format(self, fmt: str):
        self.format = fmt
        return self

    def set_n_cpus(self, n: int):
        self.n_cpus = n
        return self

    def add_args(self, *args):
        self.extra_args.extend(args)
        return self

    def set_mesh_name(self, name):
        self.mesh_name = name
        return self

    # -----------------------------
    # Core: comandos para Gmsh
    # -----------------------------
    def build_args(self):

        output_path = (self.workdir / self.mesh_name).resolve()

        args = [
            f"-{self.dim}",
            "-format", self.format,
            "-o", str(output_path),
        ]

        # CPUs para gmsh
        if self.n_cpus is not None:
            args += ["-nt", str(self.n_cpus)]

        args += self.extra_args

        return args

    def get_mesh_path(self):
        return self.workdir / self.mesh_name

    def get_executable(self):
        """
        Puedes overridear esto si quieres abstraer más.
        """
        self.executable = shutil.which("gmsh")
        if self.executable is None:
            raise FileNotFoundError("No se encontró el ejecutable de Gmsh en el PATH.")
        else:
            return self.executable
    # -----------------------------
    # Runner integration
    # -----------------------------
    def run(self, runner, run_id=None):

        geo_abs = self.geo_file.resolve()

        return runner.run(
            script=geo_abs,
            program=self.get_executable(),
            run_id=run_id,
            workdir=self.geo_file.parent,
            stdin_mode=False,
            extra_args=self.build_args(),
        )
    # -----------------------------
    # Postproceso
    # -----------------------------
    def load_mesh(self):
        import pyvista as pv

        mesh_path = self.get_mesh_path()
        if not mesh_path.exists():
            raise FileNotFoundError(f"No existe la malla: {mesh_path}")

        return pv.read(mesh_path)

    def summary(self):
        return {
            "geo": str(self.geo_file),
            "mesh": str(self.get_mesh_path()),
            "dim": self.dim,
            "format": self.format,
        }
    
    def _in_jupyter(self):
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except Exception:
            return False
        
    def plot(self, mode="auto", screenshot=None, show_edges=True):

        import os
        import pyvista as pv

        mesh = self.load_mesh()

        # -------------------------
        # AUTO
        # -------------------------
        if mode == "auto":

            if self._in_jupyter():
                mode = "notebook"

            elif "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                mode = "interactive"

            else:
                mode = "offscreen"

        # -------------------------
        # NOTEBOOK
        # -------------------------
        if mode == "notebook":

            import nest_asyncio
            nest_asyncio.apply()

            pv.set_jupyter_backend("trame")

            plotter = pv.Plotter(notebook=True)

            plotter.add_mesh(
                mesh,
                show_edges=show_edges
            )

            plotter.show(jupyter_backend="trame")

            return plotter

        # -------------------------
        # OFFSCREEN
        # -------------------------
        elif mode == "offscreen":

            pv.OFF_SCREEN = True

            plotter = pv.Plotter(off_screen=True)

            plotter.add_mesh(
                mesh,
                show_edges=show_edges
            )

            if screenshot is None:
                screenshot = self.workdir / "mesh.png"

            plotter.show(screenshot=str(screenshot))

            return screenshot

        # -------------------------
        # GUI
        # -------------------------
        elif mode == "interactive":

            pv.OFF_SCREEN = False

            plotter = pv.Plotter()

            plotter.add_mesh(
                mesh,
                show_edges=show_edges
            )

            plotter.show()

            return plotter

        else:
            raise ValueError(f"Modo desconocido: {mode}")