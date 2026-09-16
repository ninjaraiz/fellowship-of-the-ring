"""
rings/coda_single.py
====================
CODA ring with one mesh per case.

Same behaviour as :class:`CODARing` except the mesh: instead of a
single shared ``mesh_path``, a parallel ``mesh_paths`` list assigns one
mesh file to each case (position ``i`` ↔ case ``i``).  Each case folder
receives its own mesh and ``MESH_PLACEHOLDER`` is replaced by that
case's mesh basename.  The folder→mesh assignment is persisted as
``mesh_map`` in ``cases_metadata.json``.
"""

import logging
import os
import shutil
from typing import Optional

from .base import BaseRing
from .coda import _replace_mesh_placeholder, _resolve_update_value

log = logging.getLogger(__name__)


class CODASingleRing(BaseRing):
    """
    CODA case generator with a per-case mesh.

    Parameters
    ----------
    root_dir, eq_type, num_stages
        See :class:`BaseRing`.
    """

    def generate_folders(
        self,
        base_files: list,
        mesh_paths: list,
        script_dir: str,
        folder_fmt: str = "aoa_{AoA:.2f}_mach_{Mach:.3f}_h_{h:.0f}",
        overwrite: bool = False,
        update_base_files: bool = False,
        data_to_update: Optional[dict] = None,
    ) -> None:
        """
        Generate subfolders for each defined CFD case, one mesh per case.

        Args:
            base_files (list): List of base files to copy into each subfolder.
            mesh_paths (list[str]): Mesh file per case, in case order
                (position ``i`` ↔ case ``i``). Each file is copied into
                its case folder.
            script_dir (str): Directory where base files are located.
            folder_fmt (str): Folder name format. Names must match those used in bounds in define_cases.
            overwrite (bool): If True, overwrite existing folders.
            update_base_files (bool): If True, placeholders in base files are
                replaced using ``data_to_update``. By default, False.
            data_to_update (dict): ``{PLACEHOLDER: df_cases column or literal}``.
        """

        if self.case_tensor is None:
            raise RuntimeError("Cases must be defined before generating folders. Use define_cases().")

        if self.design_vars is None:
            raise RuntimeError("Design variables not defined.")

        if update_base_files and not data_to_update:
            raise ValueError('No data found to update base files.')

        n_cases = self.case_tensor.shape[0]
        if (
            not isinstance(mesh_paths, (list, tuple))
            or len(mesh_paths) != n_cases
        ):
            n_given = (
                len(mesh_paths)
                if isinstance(mesh_paths, (list, tuple)) else mesh_paths
            )
            raise ValueError(
                f"mesh_paths must be a list with one mesh per case "
                f"({n_cases} cases, got {n_given!r})."
            )
        for mesh_path in mesh_paths:
            if not isinstance(mesh_path, str) or not os.path.isfile(mesh_path):
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        basenames = [os.path.basename(m) for m in mesh_paths]
        if len(set(basenames)) != len(basenames):
            raise ValueError(
                f"Mesh basenames must be unique, got {basenames}."
            )

        self.folder_fmt = folder_fmt
        self.folders_name = []
        self.mesh_paths = list(mesh_paths)
        df_cases = self.df_cases.copy()
        df_cases['exist'] = None
        df_cases['folder'] = None
        log.info("Creating %s simulation folders in %s", n_cases, self.root_dir)

        mesh_map = {}
        for i, row in enumerate(self.case_tensor):
            case_dict = {var: float(row[j]) for j, var in enumerate(self.design_vars)}

            try:
                folder_name = folder_fmt.format(**case_dict)
            except KeyError as e:
                raise KeyError(f"Variable {e} not found in design_vars ({self.design_vars})")

            self.folders_name.append(folder_name)
            case_dir = os.path.join(self.root_dir, 'outputs', folder_name)
            mesh_path = mesh_paths[i]
            if os.path.exists(case_dir):
                if overwrite:
                    shutil.rmtree(case_dir)
                else:
                    df_cases.loc[i, 'exist'] = True
                    df_cases.loc[i, 'folder'] = folder_name
                    mesh_map[folder_name] = os.path.abspath(mesh_path)
                    log.info("Folder %s already exists. Skipping...", case_dir)
                    continue

            os.makedirs(case_dir, exist_ok=True)
            df_cases.loc[i, 'exist'] = False
            df_cases.loc[i, 'folder'] = folder_name
            mesh_map[folder_name] = os.path.abspath(mesh_path)
            for f in base_files:
                src = os.path.join(script_dir, f)
                dst = os.path.join(case_dir, f)
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Base file not found: {src}")
                shutil.copyfile(src, dst)

                with open(dst, 'r') as file:
                    content = file.read()
                    content = _replace_mesh_placeholder(
                        content, os.path.basename(mesh_path)
                    )

                    if update_base_files:
                        for placeholder, var_name in data_to_update.items():
                            content = content.replace(
                                placeholder,
                                _resolve_update_value(
                                    self.df_cases, var_name, case_dict, i,
                                ),
                            )

                with open(dst, 'w') as file:
                    file.write(content)

            dst_mesh = os.path.join(case_dir, os.path.basename(mesh_path))
            shutil.copyfile(mesh_path, dst_mesh)

        self._write_cases_metadata(df_cases, extra={'mesh_map': mesh_map})
