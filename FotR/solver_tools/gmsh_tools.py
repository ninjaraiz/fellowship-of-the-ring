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
        args = [
            f"-{self.dim}",
            "-format", self.format,
            "-o", self.mesh_name
        ]
        args += self.extra_args
        return args

    # def prepare_case(self):
    #     """
    #     Copia el .geo al working dir para evitar problemas de rutas.
    #     """
    #     target_geo = self.workdir / self.geo_file.name
    #     shutil.copy(self.geo_file, target_geo)
    #     return target_geo

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
        return runner.run(
            script=self.geo_file,
            program=self.get_executable(),
            run_id=run_id,
            workdir=self.geo_file.parent,   # 🔥 clave
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
    
    def plot(self, mode="auto", screenshot=None, show_edges=True):
        import os
        import pyvista as pv

        mesh = self.load_mesh()

        # -------------------------
        # Auto-detección entorno
        # -------------------------
        if mode == "auto":
            if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                mode = "interactive"
            else:
                mode = "notebook"

        # -------------------------
        # MODO NOTEBOOK (seguro HPC)
        # -------------------------
        if mode == "notebook":
            pv.set_jupyter_backend("static")
            return mesh.plot(show_edges=show_edges)

        # -------------------------
        # MODO OFFSCREEN (batch)
        # -------------------------
        elif mode == "offscreen":
            pv.OFF_SCREEN = True

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(mesh, show_edges=show_edges)

            if screenshot is None:
                screenshot = self.workdir / "mesh.png"

            plotter.show(screenshot=str(screenshot))
            return screenshot

        # -------------------------
        # MODO INTERACTIVO (GUI)
        # -------------------------
        elif mode == "interactive":
            pv.OFF_SCREEN = False

            plotter = pv.Plotter()
            plotter.add_mesh(mesh, show_edges=show_edges)
            plotter.show()

        else:
            raise ValueError(f"Modo desconocido: {mode}")