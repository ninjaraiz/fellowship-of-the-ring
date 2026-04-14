import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Literal
class XFoilCase:
    
    def __init__(
            self, workdir:Union[str, None],
            save_geom:bool = True,
            visc_condition: Literal['incompressible', 'compressible'] = 'compressible',
            ):
        # Airfoil
        self._mode = None
        self._naca = None
        self._file = None
        self._name = "airfoil"
        self.save_geom = save_geom
        self.visc_condition = visc_condition
        self.cp_alphas = []

        # # Condiciones
        # if self.visc_condition == 'incompressible':
        #     self.re = None
        #     self.mach = None
        # elif self.visc_condition == 'compressible':
        #     self.re = 1e6
        #     self.mach = 0.0
        
        self.alpha_start = -5
        self.alpha_end = 10
        self.alpha_step = 1.0
        self.iter = 100

        # Outputs
        self.workdir = workdir if workdir is not None else "xfoil_out"
        os.makedirs(self.workdir, exist_ok=True)

        self.foil_file = "airfoil.dat"
        self.polar_file = "polar.dat"
        self.cp_file = "cp.dat"
        self.bl_file = "bl.dat"

    def naca(self, code: str):
        self._mode = "naca"
        self._naca = code
        self._name = f"NACA{code}"
        return self

    def load(self, filepath: str, name="airfoil"):
        self._mode = "file"
        self._file = filepath
        self._name = name
        return self

    def _fullpath(self, filename):
        return os.path.join(self.workdir, filename)

    def set_conditions(self, re=1e6, mach=0.0, alpha=(-5, 10, 1)):
        self.re = re
        self.mach = mach

        if isinstance(alpha, (tuple, list)):
            self.alpha_start, self.alpha_end, self.alpha_step = alpha
        elif isinstance(alpha, int):
            self.alpha_start = alpha
            self.alpha_end = alpha
            self.alpha_step = 1
        else:
            raise ValueError("alpha must be a tuple (start, end, step) or an int.")

        # Sincronizar cp_alphas con el barrido completo por defecto
        self.cp_alphas = list(np.arange(self.alpha_start,
                                        self.alpha_end + self.alpha_step,
                                        self.alpha_step))
        return self

    def get_executable(self):
        return "xfoil"
    
    def build_commands(self):
        cmds = []

        # Airfoil
        if self._mode == "naca":
            cmds.append(f"NACA {self._naca}")
        elif self._mode == "file":
            cmds += [f"LOAD {self._file}", self._name]
        else:
            raise ValueError("Define un airfoil")

        if self.save_geom:
            cmds.append(f"SAVE {self.foil_file}")
        
        cmds.append("PANE")

        # Operación
        if self.visc_condition == 'compressible':
            list_visc = [
                "OPER",
                f"Visc {self.re}",
                f"MACH {self.mach}",
                f"ITER {self.iter}",
            ]
        else:
            list_visc = [
                "OPER",
                f"MACH {self.mach}",
                f"ITER {self.iter}",
            ]

        cmds += list_visc

        # Polar
        cmds += [
            "PACC",
            self.polar_file,
            ""
        ]

        for a in np.arange(self.alpha_start,
                        self.alpha_end + self.alpha_step,
                        self.alpha_step):
            cmds.append(f"ALFA {a}")

        cmds += ["PACC", ""]

        # 🔥 Cp multi-alpha
        for a in self.cp_alphas:
            cp_name = f"cp_{a:.2f}.dat"

            print(a, cp_name)
            cmds += ["PANE", f"ALFA {a}", f"CPWR {cp_name}"]

        if self.visc_condition == 'compressible':
            # Boundary layer dump
            cmds += [
                "DUMP",
                self.bl_file
            ]
        cmds.append("QUIT")

        return cmds

    def get_airfoil(self):
        filepath = self._fullpath(self.foil_file)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No existe: {filepath}")

        data = []
        with open(filepath) as f:
            lines = f.readlines()[1:]

        for l in lines:
            parts = l.split()
            if len(parts) == 2:
                data.append([float(parts[0]), float(parts[1])])

        return np.array(data)

    def get_airfoil_df(self):
        arr = self.get_airfoil()
        return pd.DataFrame(arr, columns=["x", "y"])

    def get_polar(self):
        return pd.read_csv(
            self._fullpath(self.polar_file),
            delim_whitespace=True,
            skiprows=12
        )
    
    def get_cp(self):
        return pd.read_csv(
            self._fullpath(self.cp_file),
            delim_whitespace=True,
            skiprows=2
        )
    
    def get_id(self):
        return f"{self._name}_Re{self.re:.0e}"

    def get_bl(self):
        return pd.read_csv(
            self._fullpath(self.bl_file),
            delim_whitespace=True
        )

    def get_clmax(self):
        polar = self.get_polar()

        idx = polar["CL"].idxmax()

        return {
            "CLmax": polar.loc[idx, "CL"],
            "alpha": polar.loc[idx, "alpha"]
        }
    def set_cp_alphas(self, alphas):
        self.cp_alphas = alphas
        return self

    def parse_all(self):
        data = {}

        # Airfoil
        data["airfoil"] = self.get_airfoil()

        # Polar
        try:
            data["polar"] = self.get_polar()
        except:
            data["polar"] = None

        # Cp multi
        cp_data = {}
        for a in self.cp_alphas:
            fname = f"cp_{a:.2f}.dat"
            path = self._fullpath(fname)

            if os.path.exists(path):
                cp_data[a] = pd.read_csv(
                    path,
                    delim_whitespace=True,
                    skiprows=2
                )

        data["cp"] = cp_data

        # BL (opcional)
        try:
            data["bl"] = self.get_bl()
        except:
            data["bl"] = None

        return data

    def export_airfoil_csv(self, filepath):
        df = self.get_airfoil_df()
        df.to_csv(filepath, index=False)

    def export_airfoil_dat(self, filepath):
        coords = self.get_airfoil()

        with open(filepath, "w") as f:
            f.write(f"{self._name}\n")

            for x, y in coords:
                f.write(f"{x:.6f} {y:.6f}\n")

    def plot_airfoil(self):
        import matplotlib.pyplot as plt
        xy = self.get_airfoil()
        plt.plot(xy[:, 0], xy[:, 1])
        plt.axis("equal")
        plt.title(self._name)
        plt.show()

    def plot_polar(self):
        import matplotlib.pyplot as plt
        df = self.get_polar()
        plt.plot(df["alpha"], df["CL"])
        plt.xlabel("alpha")
        plt.ylabel("CL")
        plt.show()

    def plot_cp(self):
        import matplotlib.pyplot as plt
        df = self.get_cp()
        plt.plot(df["x"], df["Cp"])
        plt.gca().invert_yaxis()
        plt.show()

def xfoil_parser(stdout: str, workdir: Path) -> dict:
    """
    Ejemplo de parser compatible con ARAGORN para resultados de XFoil.
 
    Parámetros
    ----------
    stdout  : salida estándar de XFoil
    workdir : directorio donde XFoil escribió los ficheros de salida
 
    Retorna
    -------
    dict con claves: 'polar', 'cp', 'airfoil', 'bl'
    """
 
    data = {}
 
    # -- Polar (fichero polar.dat generado por PACC)
    polar_path = workdir / "polar.dat"
    if polar_path.exists():
        try:
            data["polar"] = pd.read_csv(
                polar_path,
                sep=r"\s+",
                skiprows=12,
                engine="python",
            )
        except Exception:
            data["polar"] = None
 
    # -- Distribuciones de Cp (ficheros cp_<alpha>.dat)
    cp_files = sorted(workdir.glob("cp_*.dat"))
    cp_dict = {}
    for cp_path in cp_files:
        try:
            # Extraer alpha del nombre: cp_5.00.dat → 5.0
            alpha_str = cp_path.stem.replace("cp_", "")
            alpha = float(alpha_str)
            cp_dict[alpha] = pd.read_csv(
                cp_path,
                sep=r"\s+",
                skiprows=2,
                engine="python",
            )
        except Exception:
            pass
    data["cp"] = cp_dict
 
    # -- Geometría del perfil (airfoil.dat)
    geom_path = workdir / "airfoil.dat"
    if geom_path.exists():
        try:
            raw = np.loadtxt(geom_path, skiprows=1)
            data["airfoil"] = raw
        except Exception:
            data["airfoil"] = None
 
    # -- Capa límite (bl.dat generado por DUMP)
    bl_path = workdir / "bl.dat"
    if bl_path.exists():
        try:
            data["bl"] = pd.read_csv(bl_path, sep=r"\s+", engine="python")
        except Exception:
            data["bl"] = None
 
    return data