import numpy as np
import torch
from typing import Literal, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings


class LEGOLAS:
    """
    LEGOLAS — Lightweight Exploratory Graphics Of Loaded Aerodynamic Simulations
    ---------------------------------------------------------------------------
    Estructura principal organizada por dominios de plot:
    - fields      -> variables contenidas en data_dict
    - residuals   -> residuos y métricas de convergencia
    - params      -> parámetros de simulación y espacio de diseño
    - state       -> estado de cálculo / completitud de casos
    """

    def __init__(self, db: 'FRODO'):
        self.db = db
        self.format = getattr(db, "format", None)
        self.data_dict = getattr(db, "data_dict", {})

        self._check_data_dict()

        # Submódulos de plots
        self.fields = LEGOLAS.Fields(self)
        self.residuals = LEGOLAS.Residuals(self)
        self.params = LEGOLAS.Parameters(self)
        self.state = LEGOLAS.State(self)

    def _check_data_dict(self):
        if not hasattr(self.db, 'data_dict'):
            warnings.warn(
                "Database without data dictionary. Run extract_inputs() and "
                "extract_outputs() to enable field plots."
            )
        elif not self.db.data_dict:
            warnings.warn(
                "Data dictionary empty. Run extract_inputs() and "
                "extract_outputs() to enable field plots."
            )
        else:
            self.data_dict = self.db.data_dict

    def sync(self):
        """Sync local references after db updates."""
        self.data_dict = getattr(self.db, "data_dict", {})
        self.fields._sync()

    def __str__(self):
        return f"Legolas fighting for you. Database loaded: {self.db}; format: {self.format}"

    # ------------------------------------------------------------------
    # Convenience passthroughs (compatibilidad rápida)
    # ------------------------------------------------------------------
    def plot_field(self, *args, **kwargs):
        return self.fields.plot_field(*args, **kwargs)

    def plot_vector_field(self, *args, **kwargs):
        return self.fields.plot_vector_field(*args, **kwargs)

    def plot_distribution(self, *args, **kwargs):
        return self.fields.plot_distribution(*args, **kwargs)

    def plot_correlation_matrix(self, *args, **kwargs):
        return self.fields.plot_correlation_matrix(*args, **kwargs)

    def compare_cases(self, *args, **kwargs):
        return self.fields.compare_cases(*args, **kwargs)

    def list_available_variables(self, *args, **kwargs):
        return self.fields.list_available_variables(*args, **kwargs)

    # ==========================================================
    # FIELDS
    # ==========================================================
    class Fields:
        def __init__(self, parent: 'LEGOLAS'):
            self.parent = parent
            self.db = parent.db
            self.format = parent.format
            self.data_dict = parent.data_dict
            self.active_group = None
            self.active_stage = None
            self._auto_select()

        def _sync(self):
            """Sync data_dict with parent."""
            self.data_dict = self.parent.data_dict

        def _auto_select(self):
            """
            Automatically select the first CADGroup and stage.
            """
            if self.format == "CODA":
            
                groups = [k for k in self.data_dict.keys() if k.startswith("CADGroup_")]
                if groups:
                    self.active_group = groups[0]
                    self.active_stage = "0"

        def select_group(self, group_key: str, stage: Union[int, str] = 0):
            """
            Select a specific CADGroup and stage for subsequent field plots.
             - group_key: e.g., "CADGroup_0"
             - stage: e.g., 0 or "0"
             If not called, it will use the first available group and stage by default.
            """
            
            if group_key not in self.data_dict:
                raise KeyError(f"Group {group_key} not found in data_dict.")
            self.active_group = group_key
            self.active_stage = str(stage)

        # --------------------------
        # Internal data utilities
        # --------------------------
        @staticmethod
        def _to_numpy(arr):
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        def _get_coda_variable(self, group_key, stage, var_name, case_idx):
            """ 
            Extract variable from CODA data_dict structure:
             - group_key: e.g., "CADGroup_0"
             - stage: e.g., 0 or "0"
             - var_name: e.g., "p", "u", etc.
             - case_idx: index of the case to extract (for 2D/3D arrays)
             Returns a 1D array for the specified case.
             CODA structure: data_dict[group_key]["Vars"][stage][var_name] -> shape (n_points, n_cases) or (n_points, n_dims, n_cases)
             This function selects the appropriate slice for the given case_idx.
             If the variable is 2D (n_points, n_cases), it returns arr[:, case_idx].
             If the variable is 3D (n_points, n_dims, n_cases), it returns arr[:, :, case_idx].
             Raises ValueError if the variable has an unsupported number of dimensions.
            """
            arr = self.data_dict[group_key]["Vars"][str(stage)][var_name]
            if arr.ndim == 2:
                return arr[:, case_idx]
            if arr.ndim == 3:
                return arr[:, :, case_idx]
            raise ValueError(f"Unsupported ndim {arr.ndim} for CODA variable {var_name}")

        def _get_numpy_variable(self, section, var_name, case_idx):
            """
            Extract a variable from the numpy data_dict structure.

            Args:
                section (str): The section of the data_dict (e.g., "outputs").
                var_name (str): The name of the variable to extract.
                case_idx (int): The index of the case to extract (for 2D/3D arrays).

            Returns:
                np.ndarray: The extracted variable as a numpy array.
            """
            arr = self.data_dict[section][var_name]
            arr = self._to_numpy(arr)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2:
                # Heurística: (n_points, n_cases) -> seleccionar columna
                if case_idx < arr.shape[1]:
                    return arr[:, case_idx]
                if case_idx < arr.shape[0]:
                    return arr[case_idx, :]
            return arr

        def _get_variable(self, var_name, case_idx, group_key=None, stage=None, section=None):
            """
            Extract a variable from the data_dict structure.

            Args:
                var_name (str): The name of the variable to extract.
                case_idx (int): The index of the case to extract.
                group_key (str, optional): The key of the group to extract from. Defaults to None.
                stage (str, optional): The stage to extract from. Defaults to None.
                section (str, optional): The section to extract from. Defaults to None.

            Raises:
                ValueError: If the variable cannot be found.
                KeyError: If the group, stage, or section is not found.

            Returns:
                np.ndarray: The extracted variable as a numpy array.
            """
 
            if self.format == "CODA":
                if group_key is None:
                    group_key = self.active_group
                if stage is None:
                    stage = self.active_stage
                if group_key is None or stage is None:
                    raise ValueError("CODA requires group_key and stage (or select_group).")
                return self._get_coda_variable(group_key, stage, var_name, case_idx)

            # NUMPYFILE / otros: buscar por sección
            if section is None:
                for sec in ("outputs", "aux", "inputs"):
                    if sec in self.data_dict and var_name in self.data_dict[sec]:
                        section = sec
                        break
            if section is None:
                raise KeyError(f"Variable '{var_name}' not found in data_dict.")
            return self._get_numpy_variable(section, var_name, case_idx)

        def _find_coords(self, group_key=None, section="inputs"):
            """
            Find the coordinates for the specified group and section.
            """
            if self.format == "CODA":
                if group_key is None:
                    group_key = self.active_group
                if group_key is None:
                    raise ValueError("CODA requires group_key (or select_group).")
                return self._to_numpy(self.data_dict[group_key]["Coord"])

            # NUMPYFILE / otros
            if "inputs" in self.data_dict and "ptos" in self.data_dict["inputs"]:
                return self._to_numpy(self.data_dict["inputs"]["ptos"])
            if "Coord" in self.data_dict:
                return self._to_numpy(self.data_dict["Coord"])
            # Buscar cualquier input 2D/3D
            if section in self.data_dict:
                for v in self.data_dict[section].values():
                    arr = self._to_numpy(v)
                    if arr.ndim == 2 and arr.shape[1] in (2, 3):
                        return arr
            raise ValueError("No coordinates found for plotting.")

        def list_available_variables(self):
            """
            List all available variables in the data_dict.
            """
            if self.format == "CODA":
                print("Available CADGroups and variables:")
                for key in self.data_dict:
                    if not key.startswith("CADGroup_"):
                        continue
                    print(f"\n[{key}]")
                    vars_stage = self.data_dict[key].get("Vars", {})
                    for stage, vdict in vars_stage.items():
                        print(f"  Stage {stage}: {list(vdict.keys())}")
            else:
                print("Available variables:")
                for section in ("inputs", "outputs", "aux"):
                    if section in self.data_dict:
                        print(f"\n[{section.upper()}]")
                        for k, v in self.data_dict[section].items():
                            arr = self._to_numpy(v)
                            print(f"  • {k:20s}  shape={arr.shape}")

        # --------------------------
        # Plots
        # --------------------------
        def plot_field(
            self,
            var_name: Union[str, list[str], tuple[str]],
            case_idx: int = 0,
            group_key: Union[str, None] = None,
            stage: Union[int, str, None] = None,
            section: Union[str, None] = None,
            coord_idx: tuple[int, int] = (0, 1),
            cmap: str = "viridis",
            s: int = 2
            ):
            """
            Plot scalar fields over a 2D projection of the coordinates.

            Args:
                var_name (str | list[str] | tuple[str]): Variable(s) to plot.
                case_idx (int): Case index to extract.
                group_key (str | None): CADGroup key (CODA). Uses selected group if None.
                stage (int | str | None): Stage to extract (CODA). Uses selected stage if None.
                section (str | None): Section for non-CODA data ("inputs", "outputs", "aux").
                coord_idx (tuple[int,int]): Coordinate indices to project (e.g., (0,1) or (0,2)).
                cmap (str): Matplotlib colormap.
                s (int): Marker size.

            Returns:
                None. Displays the figure.
            """
            if isinstance(var_name, str):
                var_list = [var_name]
            else:
                var_list = list(var_name)

            coords = self._find_coords(group_key=group_key, section=section)
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError("Coordinates must be 2D or 3D to use plot_field.")
            if max(coord_idx) >= coords.shape[1]:
                raise IndexError(f"coord_idx {coord_idx} out of bounds for coords with shape {coords.shape}.")
            x = coords[:, coord_idx[0]]
            y = coords[:, coord_idx[1]]

            for v in var_list:
                field = self._get_variable(
                    v, case_idx, group_key=group_key, stage=stage, section=section
                )
                plt.figure(figsize=(8, 6))
                sc = plt.scatter(
                    x,
                    y,
                    c=field,
                    cmap=cmap,
                    s=s
                )
                plt.colorbar(sc)
                plt.xlabel(f"Coord[{coord_idx[0]}]")
                plt.ylabel(f"Coord[{coord_idx[1]}]")
                plt.title(f"{v} | Case {case_idx} | coords={coord_idx}")
                plt.tight_layout()
                plt.show()

        def plot_vector_field(
            self,
            var_name: str,
            case_idx: int = 0,
            group_key: Union[str, None] = None,
            stage: Union[int, str, None] = None,
            section: Union[str, None] = None,
            coord_idx: tuple[int, int] = (0, 1),
            vec_comp_idx: Union[tuple[int, int], None] = None,
            stride: int = 50
            ):
            """
            Plot a vector field over a 2D projection of the coordinates.

            Args:
                var_name (str): Vector variable name.
                case_idx (int): Case index to extract.
                group_key (str | None): CADGroup key (CODA). Uses selected group if None.
                stage (int | str | None): Stage to extract (CODA). Uses selected stage if None.
                section (str | None): Section for non-CODA data ("inputs", "outputs", "aux").
                coord_idx (tuple[int,int]): Coordinate indices to project.
                vec_comp_idx (tuple[int,int] | None): Vector component indices to plot.
                stride (int): Subsampling stride for quiver.

            Returns:
                None. Displays the figure.
            """
            coords = self._find_coords(group_key=group_key, section=section)
            vec = self._get_variable(
                var_name, case_idx, group_key=group_key, stage=stage, section=section
            )

            if vec.ndim != 2:
                raise ValueError("Vector variable must be (n_comp, n_points)")
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError("Coordinates must be 2D or 3D to use plot_vector_field.")
            if max(coord_idx) >= coords.shape[1]:
                raise IndexError(f"coord_idx {coord_idx} out of bounds for coords with shape {coords.shape}.")

            if vec_comp_idx is None:
                vec_comp_idx = coord_idx
            if max(vec_comp_idx) >= vec.shape[0]:
                raise IndexError(f"vec_comp_idx {vec_comp_idx} out of bounds for vec with shape {vec.shape}.")

            plt.figure(figsize=(8, 6))
            plt.quiver(
                coords[::stride, coord_idx[0]],
                coords[::stride, coord_idx[1]],
                vec[vec_comp_idx[0], ::stride],
                vec[vec_comp_idx[1], ::stride],
            )
            plt.xlabel(f"Coord[{coord_idx[0]}]")
            plt.ylabel(f"Coord[{coord_idx[1]}]")
            plt.title(f"{var_name} vector field | Case {case_idx} | coords={coord_idx}")
            plt.tight_layout()
            plt.show()

        def plot_line_field(
            self,
            var_name: str,
            case_idx: int = 0,
            group_key: Union[str, None] = None,
            stage: Union[int, str, None] = None,
            section: Union[str, None] = None,
            x_axis: Literal["index", "coord", "arc"] = "arc",
            coord_idx: int = 0,
            linewidth: float = 2.0
            ):
            """
            Plot 1D groups as a line: variable vs. a 1D coordinate.

            Args:
                var_name (str): Variable name to plot.
                case_idx (int): Case index to extract.
                group_key (str | None): CADGroup key (CODA). Uses selected group if None.
                stage (int | str | None): Stage to extract (CODA). Uses selected stage if None.
                section (str | None): Section for non-CODA data ("inputs", "outputs", "aux").
                x_axis (str): "index", "coord" (uses coord_idx), or "arc" (curvilinear).
                coord_idx (int): Coordinate index to use if x_axis="coord".
                linewidth (float): Line width.

            Returns:
                None. Displays the figure.
            """
            coords = self._find_coords(group_key=group_key, section=section)
            field = self._get_variable(
                var_name, case_idx, group_key=group_key, stage=stage, section=section
            )

            if coords.ndim == 1:
                coord_array = coords
            elif coords.ndim == 2:
                if x_axis == "coord":
                    if coord_idx >= coords.shape[1]:
                        raise IndexError(f"coord_idx {coord_idx} out of bounds for coords with shape {coords.shape}.")
                    coord_array = coords[:, coord_idx]
                elif x_axis == "arc":
                    diffs = np.diff(coords, axis=0)
                    ds = np.linalg.norm(diffs, axis=1)
                    coord_array = np.concatenate(([0.0], np.cumsum(ds)))
                else:
                    coord_array = np.arange(coords.shape[0])
            else:
                raise ValueError("Unsupported coordinate array for 1D plotting.")

            plt.figure(figsize=(8, 4))
            plt.plot(coord_array, field, linewidth=linewidth)
            plt.xlabel("s" if x_axis == "arc" else f"Coord[{coord_idx}]" if x_axis == "coord" else "Index")
            plt.ylabel(var_name)
            plt.title(f"{var_name} | Case {case_idx} | 1D plot")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.show()

        def plot_distribution(
            self,
            var_name: str,
            case_idx: int = 0,
            group_key: Union[str, None] = None,
            stage: Union[int, str, None] = None,
            section: Union[str, None] = None,
            save_path:Union[str, None] = None,
            **kwargs
            ):
            """
            Plot distribution of a variable using seaborn.
            Args:
                var_name (str): Variable name to plot.
                case_idx (int): Case index to extract.
                group_key (str | None): CADGroup key (CODA). Uses selected group if None.
                stage (int | str | None): Stage to extract (CODA). Uses selected stage if None.
                section (str | None): Section for non-CODA data ("inputs", "outputs", "aux").
                save_path (str | None): Path to save the figure.
                **kwargs: Additional keyword arguments for seaborn's histplot.

            Returns:
                None. Displays the figure or saves it in save_path.
            """
            field = self._get_variable(
                var_name, case_idx, group_key=group_key, stage=stage, section=section
            )
            plt.figure(figsize=(8, 5))
            sns.histplot(field, **kwargs)
            plt.title(f"Distribution of {var_name}")
            plt.yscale(kwargs.get("yscale", False) and "log" or "linear")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        def plot_correlation_matrix(
            self,
            group_key: Union[str, None] = None,
            stage: Union[int, str, None] = None,
            case_idx: int = 0,
            variables: Union[list[str], None] = None
            ):
            if self.format != "CODA":
                raise NotImplementedError("Correlation for non-CODA formats pending.")

            if group_key is None:
                group_key = self.active_group
            if stage is None:
                stage = self.active_stage

            vars_dict = self.data_dict[group_key]["Vars"][str(stage)]
            if variables is None:
                variables = list(vars_dict.keys())

            data = []
            for v in variables:
                arr = vars_dict[v][:, case_idx]
                if arr.ndim == 1:
                    data.append(arr)

            data = np.array(data).T
            corr = np.corrcoef(data, rowvar=False)

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.show()

        def compare_cases(
            self,
            var_name: str,
            case_indices: Union[list[int], tuple[int]],
            group_key: Union[str, None] = None,
            stage: Union[int, str, None] = None
            ):
            if self.format != "CODA":
                raise NotImplementedError("compare_cases is only available for CODA.")

            if group_key is None:
                group_key = self.active_group
            if stage is None:
                stage = self.active_stage

            plt.figure(figsize=(8, 6))
            for case_idx in case_indices:
                field = self._get_variable(
                    var_name, case_idx, group_key=group_key, stage=stage
                )
                plt.plot(field, label=f"Case {case_idx}")

            plt.legend()
            plt.title(f"Comparison of {var_name}")
            plt.tight_layout()
            plt.show()

    # ==========================================================
    # RESIDUALS
    # ==========================================================
    class Residuals:
        def __init__(self, parent: 'LEGOLAS'):
            self.parent = parent
            self.db = parent.db

        def _require_residuals(self):
            if getattr(self.db, "residuals", None) is None:
                raise NotImplementedError("Residuals not available for this database format.")

        def plot_case(self, *args, **kwargs):
            self._require_residuals()
            return self.db.residuals.plot_residuals_from_case(*args, **kwargs)

        def plot_map(self, *args, **kwargs):
            self._require_residuals()
            return self.db.residuals.plot_all_final_residuals(*args, **kwargs)

        def final_table(self, *args, **kwargs):
            self._require_residuals()
            return self.db.residuals.get_all_final_residuals(*args, **kwargs)

        def plot_vs_params(
            self,
            residual_name: Union[str, None] = None,
            mode: Literal['absolute', 'norm', 'scaled'] = 'scaled',
            stage: Union[list, tuple, 'all'] = 'all',
            only_finished: bool = False,
            params: Union[list[str], tuple[str], None] = None,
            figsize: tuple = (8, 6),
            s: int = 80,
            cmap: str = "viridis"
        ):
            """
            Plot residual values against the design parameter space.
            """
            self._require_residuals()
            df_finals = self.db.residuals.get_all_final_residuals(
                stage=stage, only_finished=only_finished, load_in_metadata=False
            )

            # Detect residual columns
            cols_mode = [c for c in df_finals.columns if "Residual" in c and mode in c]
            if not cols_mode:
                raise ValueError(f"No residual columns found for mode '{mode}'.")

            if residual_name is None:
                target_col = cols_mode[0]
            else:
                if residual_name in df_finals.columns:
                    target_col = residual_name
                else:
                    # Try to match by partial name + mode
                    matches = [c for c in cols_mode if residual_name in c]
                    if not matches:
                        raise KeyError(
                            f"Residual '{residual_name}' not found. "
                            f"Available: {cols_mode}"
                        )
                    target_col = matches[0]

            if params is None:
                params = getattr(self.db, "metadata", {}).get("design_vars", None)
            if params is None or len(params) < 2:
                raise ValueError("At least two parameters required to plot residuals vs params.")

            params = list(params)

            if len(params) == 2:
                x, y = params
                plt.figure(figsize=figsize)
                sc = plt.scatter(df_finals[x], df_finals[y], c=df_finals[target_col], s=s, cmap=cmap)
                plt.colorbar(sc, label=target_col)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(f"{target_col} vs parameters")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.tight_layout()
                plt.show()
            elif len(params) >= 3:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
                x, y, z = params[:3]
                sc = ax.scatter(df_finals[x], df_finals[y], df_finals[z],
                                c=df_finals[target_col], cmap=cmap, s=s)
                fig.colorbar(sc, ax=ax, label=target_col)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_zlabel(z)
                ax.set_title(f"{target_col} vs parameters (3D)")
                plt.tight_layout()
                plt.show()

        def plot_integral_metrics(
            self,
            var_metrics: Union[str, list[str], tuple[str]],
            stage: int = 0,
            iter_var: int = 1000,
            only_finished: bool = False,
            figsize: tuple = (10, 5),
            s: int = 80,
            cmap: str = "viridis"
        ):
            """
            Plot mean and variance of integral metrics over the design space.
            """
            self._require_residuals()

            df_post = self.db.residuals.get_df_metrics(
                var_metrics=var_metrics, iter_var=iter_var, save=False
            )

            if isinstance(var_metrics, str):
                var_metrics = [var_metrics]

            params = getattr(self.db, "metadata", {}).get("design_vars", None)
            if params is None or len(params) < 2:
                raise ValueError("At least two parameters required to plot integral metrics.")

            x, y = params[:2]

            for v in var_metrics:
                mean_col = f"{v}_mean_stage{stage}"
                var_col = f"{v}_var_stage{stage}"
                if mean_col not in df_post.columns or var_col not in df_post.columns:
                    raise KeyError(f"Columns {mean_col} or {var_col} not found in df_post.")

                fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
                sc1 = axes[0].scatter(df_post[x], df_post[y], c=df_post[mean_col], s=s, cmap=cmap)
                plt.colorbar(sc1, ax=axes[0], label=mean_col)
                axes[0].set_xlabel(x)
                axes[0].set_ylabel(y)
                axes[0].set_title(f"Mean {v} (stage {stage})")
                axes[0].grid(True, linestyle="--", alpha=0.4)

                sc2 = axes[1].scatter(df_post[x], df_post[y], c=df_post[var_col], s=s, cmap=cmap)
                plt.colorbar(sc2, ax=axes[1], label=var_col)
                axes[1].set_xlabel(x)
                axes[1].set_title(f"Var {v} (stage {stage})")
                axes[1].grid(True, linestyle="--", alpha=0.4)

                plt.tight_layout()
                plt.show()

    # ==========================================================
    # PARAMETERS / DESIGN SPACE
    # ==========================================================
    class Parameters:
        def __init__(self, parent: 'LEGOLAS'):
            self.parent = parent
            self.db = parent.db

        def _get_df_state(self):
            df_state = getattr(self.db, "df_state", None)
            if df_state is None:
                raise AttributeError("db.df_state not available. Run parse_simulation_dirs() or extract_inputs().")
            return df_state

        def plot_space(
            self,
            vars: Union[list[str], tuple[str], None] = None,
            color: Union[str, None] = None,
            figsize: tuple = (8, 6),
            s: int = 80
        ):
            df_state = self._get_df_state()

            if vars is None:
                vars = getattr(self.db, "metadata", {}).get("design_vars", None)
                if vars is None:
                    # Fallback: numeric columns
                    vars = list(df_state.select_dtypes(include=[np.number]).columns)
                    vars = [v for v in vars if v != "stage"]

            if len(vars) < 2:
                raise ValueError("At least two variables required to plot parameter space.")

            if len(vars) == 2:
                x, y = vars
                plt.figure(figsize=figsize)
                if color and color in df_state:
                    sc = plt.scatter(df_state[x], df_state[y], c=df_state[color], s=s, cmap="viridis")
                    plt.colorbar(sc, label=color)
                else:
                    plt.scatter(df_state[x], df_state[y], s=s)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title("Parameter space")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.tight_layout()
                plt.show()
            else:
                # Pairplot para más dimensiones
                sns.pairplot(df_state[list(vars)], diag_kind="hist")

    # ==========================================================
    # STATE
    # ==========================================================
    class State:
        def __init__(self, parent: 'LEGOLAS'):
            self.parent = parent
            self.db = parent.db

        def plot(self, *args, **kwargs):
            # Preferir implementación específica si existe
            if hasattr(self.db, "plot_state") and callable(getattr(self.db, "plot_state")):
                return self.db.plot_state(*args, **kwargs)

            # Fallback: usar df_state si existe
            df_state = getattr(self.db, "df_state", None)
            if df_state is None:
                raise AttributeError("db.plot_state not available and df_state missing.")

            vars = getattr(self.db, "metadata", {}).get("design_vars", None)
            if vars is None:
                vars = list(df_state.select_dtypes(include=[np.number]).columns)
                vars = [v for v in vars if v != "stage"]

            if len(vars) >= 2 and "stage" in df_state:
                x, y = vars[:2]
                plt.figure(figsize=kwargs.get("figsize", (8, 6)))
                sc = plt.scatter(df_state[x], df_state[y], c=df_state["stage"], cmap="RdYlGn", s=80)
                plt.colorbar(sc, label="stage")
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title("State of cases")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.tight_layout()
                plt.show()
            else:
                raise ValueError("Insufficient data to plot state.")
            
            
    class Carcaj:
            
        @staticmethod
        def _df_to_image_matplotlib(
            df: pd.DataFrame,
            filename: str,
            figsize: tuple = (12, 4),
            dpi: int = 300,
            fontsize: int = 10,
            header_color: str = "#D9EAF7",
            row_colors: tuple = ("#FFFFFF", "#F5F5F5"),
            edge_color: str = "black",
            column_formats=None
        ):
            """
            Export a pandas DataFrame as a PNG table using Matplotlib.
            """

            import matplotlib.pyplot as plt

            df_plot = df.copy()

            if column_formats is None:
                column_formats = {}

            for col in df_plot.columns:

                fmt = column_formats.get(col)

                if fmt is None:
                    continue

                df_plot[col] = df_plot[col].map(
                    lambda x: fmt.format(x) if pd.notna(x) else ""
                )

            fig, ax = plt.subplots(figsize=figsize)
            ax.axis("off")

            table = ax.table(
                cellText=df_plot.values,
                colLabels=df_plot.columns,
                rowLabels=df_plot.index,
                cellLoc="center",
                loc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.scale(1.2, 1.5)

            for (row, col), cell in table.get_celld().items():

                cell.set_edgecolor(edge_color)

                if row == 0:
                    cell.set_facecolor(header_color)
                    cell.set_text_props(weight="bold")

                elif row > 0:
                    cell.set_facecolor(row_colors[(row - 1) % 2])

            fig.tight_layout()

            fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches="tight",
            )

            plt.close(fig)
            
        @staticmethod
        def _df_to_image_plotly(
            df: pd.DataFrame,
            filename: str,
            column_formats=None
        ):
            """
            Export a DataFrame as a PNG table using Plotly.
            Requires kaleido.
            """

            import plotly.graph_objects as go

            df_plot = df.copy()

            if column_formats is None:
                column_formats = {}

            for col in df_plot.columns:

                fmt = column_formats.get(col)

                if fmt is None:
                    continue

                df_plot[col] = df_plot[col].map(
                    lambda x: fmt.format(x) if pd.notna(x) else ""
                )

            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=["<b>"+str(c)+"</b>" for c in df_plot.columns],
                            align="center",
                        ),
                        cells=dict(
                            values=[df_plot[c] for c in df_plot.columns],
                            align="center",
                        ),
                    )
                ]
            )

            fig.write_image(
                filename,
                scale=2,
            )
            
        @staticmethod
        def _df_to_image_dataframe_image(
            df: pd.DataFrame,
            filename: str,
            cmap: str = "Blues",
            column_formats=None
        ):
            """
            Export a styled DataFrame as a PNG using dataframe_image.
            """

            import dataframe_image as dfi

            if column_formats is None:
                column_formats = {}

            styled = df.style

            numeric_cols = df.select_dtypes(include=np.number).columns

            if len(numeric_cols):
                styled = styled.background_gradient(
                    cmap=cmap,
                    subset=numeric_cols,
                )

            if column_formats:
                styled = styled.format(column_formats)

            dfi.export(
                styled,
                filename,
            )