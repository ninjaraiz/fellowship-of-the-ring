"""
rings/coda.py
=============
CODA ring with a single shared mesh for every case.

This is the original GANDALF behaviour: one ``mesh_path`` is copied
into every case folder and ``MESH_PLACEHOLDER`` is replaced by its
basename.  Per-case values or literals are substituted through
``data_to_update`` (``{PLACEHOLDER: df_cases column or literal}``),
e.g. ``{"AOA_PLACEHOLDER": "aoa", "CHORD_PLACEHOLDER": str(c)}``.
"""

import logging
import os
import shutil
from typing import Optional

from .base import BaseRing

log = logging.getLogger(__name__)

_MESH_TOKEN = "MESH_PLACEHOLDER"


def _replace_mesh_placeholder(content: str, mesh_basename: str) -> str:
    """Substitute ``MESH_PLACEHOLDER``, quoting it only when needed.

    An already-quoted token keeps single quotes; a bare token gains
    them. Both variants are handled when they coexist.
    """
    content = content.replace(
        f'"{_MESH_TOKEN}"', f'"{mesh_basename}"'
    )
    return content.replace(_MESH_TOKEN, f'"{mesh_basename}"')


def _resolve_update_value(
    df_cases, var_name: str, case_dict: dict, i: int
) -> str:
    """Resolve a ``data_to_update`` value for one case.

    Values naming a ``df_cases`` column give that case's value
    (preferred via ``case_dict`` when the column is a design variable,
    else positionally); anything else is used as a literal string.
    """
    cols = getattr(df_cases, 'columns', [])
    if var_name not in cols:
        return var_name
    if var_name in case_dict:
        return str(case_dict[var_name])
    try:
        return str(df_cases[var_name].iloc[i])
    except (IndexError, KeyError):
        return var_name


class CODARing(BaseRing):
    """
    CODA case generator with one shared mesh.

    Parameters
    ----------
    root_dir, eq_type, num_stages
        See :class:`BaseRing`.
    """

    def generate_folders(
        self,
        base_files: list,
        mesh_path: str,
        script_dir: str,
        folder_fmt: str = "aoa_{AoA:.2f}_mach_{Mach:.3f}_h_{h:.0f}",
        overwrite: bool = False,
        update_base_files: bool = False,
        data_to_update: Optional[dict] = None,
    ) -> None:
        """
        Generate subfolders for each defined CFD case.

        Args:
            base_files (list): List of base files to copy into each subfolder.
            mesh_path (str): Shared mesh file, copied into every case folder.
            script_dir (str): Directory where base files are located.
            folder_fmt (str): Folder name format. Names must match those used in bounds in define_cases. Example:
                "AoA_{AoA:.2f}_Mach_{Mach:.3f}_h_{h:.0f}".
            overwrite (bool): If True, overwrite existing folders.
            update_base_files (bool): If True, placeholders in base files are
                replaced using ``data_to_update``. By default, False.
            data_to_update (dict): ``{PLACEHOLDER: df_cases column or literal}``.
                Values naming a ``df_cases`` column are replaced per case;
                anything else is used as a literal string. Example:
                ``{"AOA_PLACEHOLDER": "AoA", "CHORD_PLACEHOLDER": str(c)}``.
        """

        if self.case_tensor is None:
            raise RuntimeError("Cases must be defined before generating folders. Use define_cases().")

        if self.design_vars is None:
            raise RuntimeError("Design variables not defined.")

        if update_base_files and not data_to_update:
            raise ValueError('No data found to update base files.')

        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        self.folder_fmt = folder_fmt
        self.folders_name = []
        self.mesh_path = mesh_path
        df_cases = self.df_cases.copy()
        df_cases['exist'] = None
        df_cases['folder'] = None
        n_cases = self.case_tensor.shape[0]
        log.info("Creating %s simulation folders in %s", n_cases, self.root_dir)

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
                    df_cases.loc[i, 'exist'] = True
                    df_cases.loc[i, 'folder'] = folder_name
                    log.info("Folder %s already exists. Skipping...", case_dir)
                    continue

            os.makedirs(case_dir, exist_ok=True)
            df_cases.loc[i, 'exist'] = False
            df_cases.loc[i, 'folder'] = folder_name
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

        self._write_cases_metadata(df_cases)
