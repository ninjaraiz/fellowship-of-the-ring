"""
sets/numpy_file.py
==================
Sets class for the NUMPYFILE format.

Provides ML-tensor assembly and auxiliary data management for FRODO
instances loaded from pre-processed numpy dictionaries.

The class operates on the three-bucket ``data_dict`` layout produced by
``NUMPYFILEReader``::

    data_dict = {
        'inputs':  {'ptos': ..., 'aoa': ..., ...},
        'outputs': {'cp': ..., ...},
        'aux':     {'normals': ..., ...},
    }
"""

import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import h5py

from ..sam import SAM
from .base import BaseSets

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..frodo import FRODO


class NUMPYFILESets(BaseSets):
    """
    Sets class for NUMPYFILE-format FRODO databases.

    Assembles the contents of ``data_dict['inputs']``,
    ``data_dict['aux']`` and ``data_dict['outputs']`` into a single flat
    ML-ready joint tensor via ``SAM.Gardener.create_final_tensor``.

    The tensor layout per row is::

        [coordinates | parametric_vars | aux_features | output_fields]

    Each row corresponds to one (point, case) pair.

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance whose ``data_dict`` already contains
        ``'inputs'`` (populated by ``NUMPYFILEReader.extract_inputs``).

    Examples
    --------
    Full workflow from FRODO::

        db = FRODO(
            root_dir='/data/numpy_db',
            format='NUMPYFILE',
            file='db_random.npy',
        )
        db.extract_inputs(
            keys_inputs={'ptos': 'db_random.npy/Airfoil',
                         'aoa':  'db_random.npy/Alpha'},
            keys_aux={'normals': 'db_random.npy/Normals'},
        )
        db.extract_outputs({'cp': 'db_random.npy/Cp'})

        result = db.sets.create_jset(save_path='/out/jset.h5', verbose=True)
        print(result['tensor'].shape)   # (n_points * n_cases, n_cols)
    """

    def __init__(self, db: 'FRODO'):
        super().__init__(db)

    # =========================================================================
    # BaseSets interface
    # =========================================================================

    def create_jset(
        self,
        sol: Union[list, tuple, int, str] = 'all',
        save_path: Union[bool, str] = False,
        verbose: bool = False,
    ) -> dict:
        """
        Assemble mesh coordinates, parametric variables, auxiliary arrays
        and output fields into a single flat ML-ready joint tensor.

        This method collects:

        * **Coordinates** — ``data_dict['inputs']['ptos']``
          (shape ``(n_points, n_dim)``).
        * **Parametric variables** — all remaining arrays in
          ``data_dict['inputs']`` concatenated column-wise into a matrix of
          shape ``(n_cases, n_params)``.
        * **Auxiliary features** — all arrays in ``data_dict['aux']``
          (spatial features shared across cases, e.g. surface normals).
        * **Output fields** — all arrays in ``data_dict['outputs']``.

        All arrays are forwarded to :func:`SAM.Gardener.create_final_tensor`
        which handles broadcasting, flattening and min-max normalisation.

        The result is stored in ``db.jset`` and a column-named DataFrame is
        stored in ``db.df_data`` for quick inspection.

        Parameters
        ----------
        sol : 'all', int or list[int]
            Which output channel(s) to include.  ``'all'`` keeps every
            output; an int selects a single channel index; a list selects
            multiple channels.  Default ``'all'``.
        save_path : bool or str
            If a non-empty string, the result is saved to this path.
            Supported extensions:

            * ``.h5``  — GZIP-compressed HDF5 (recommended for large datasets).
            * ``.pt``  — PyTorch serialised dict (``torch.save``).
            * ``.npy`` — NumPy serialised dict (``np.save``).

            Default ``False`` (no saving).
        verbose : bool
            If True, prints the shape of the assembled tensor and confirms
            the save path. Default False.

        Returns
        -------
        dict
            Result from ``SAM.Gardener.create_final_tensor`` with keys:

            * ``'tensor'`` — raw joint tensor, shape ``(n_pts×n_cases, n_cols)``.
            * ``'scaled'`` — min-max normalised tensor, same shape.
            * ``'mins'``   — per-column minimum values (1-D torch.Tensor).
            * ``'maxs'``   — per-column maximum values (1-D torch.Tensor).
            * ``'info'``   — dict with ``'ninputs'`` and ``'noutputs'``.

        Raises
        ------
        KeyError
            If ``data_dict['inputs']`` or ``data_dict['outputs']`` is empty
            or missing.
        NameError
            If ``save_path`` has an unsupported file extension.

        Side-effects
        ------------
        * Sets ``db.jset`` to the returned result dict.
        * Sets ``db.df_data`` to a ``pd.DataFrame`` with named columns.

        Notes
        -----
        Column names in ``db.df_data`` are inferred from the shapes of the
        input arrays:

        * 2-column input → ``['x', 'z']``
        * 3-column input → ``['x', 'y', 'z']``
        * 1-column input → the alias string (e.g. ``'aoa'``)

        Examples
        --------
        Assemble and inspect inline::

            result = db.sets.create_jset(verbose=True)
            print(db.df_data.head())
            print(db.df_data.columns.tolist())

        Assemble, keep only channel 0, and save to HDF5::

            result = db.sets.create_jset(
                sol=0,
                save_path='/output/jset.h5',
                verbose=True,
            )

        Load back from HDF5::

            reader = SAM.HDF5reader('/output/jset.h5')
            tensor = reader.load_to_tensor('tensor')
            print(tensor.shape)
        """
        dd = self.db.data_dict

        if not dd.get("inputs"):
            raise KeyError(
                "data_dict['inputs'] is empty. "
                "Call extract_inputs() before create_jset()."
            )
        if not dd.get("outputs"):
            raise KeyError(
                "data_dict['outputs'] is empty. "
                "Call extract_outputs() before create_jset()."
            )

        # ── Coordinates ───────────────────────────────────────────────────
        tensor_ptos = dd['inputs']['ptos']

        # ── Parametric variables (all inputs except 'ptos') ───────────────
        param_arrays = [
            dd['inputs'][k]
            for k in dd['inputs']
            if k != 'ptos'
        ]
        tensor_flcc = (
            np.column_stack(param_arrays)
            if param_arrays
            else np.empty((tensor_ptos.shape[0], 0))
        )

        # ── Auxiliary and output arrays ───────────────────────────────────
        tensors_aux = list(dd.get('aux', {}).values())
        tensors_out = list(dd['outputs'].values())

        result = SAM.Gardener.create_final_tensor(
            tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
            sol=sol, verbose=verbose,
        )

        # ── Optional save ─────────────────────────────────────────────────
        if save_path:
            self._save_result(result, save_path)
            if verbose:
                print(f"Jset saved to {save_path}\n")

        self.db.jset = result

        # ── Build column names ────────────────────────────────────────────
        columns: list = []
        for alias, arr in dd['inputs'].items():
            arr = np.asarray(arr)
            n   = arr.shape[1] if arr.ndim > 1 else 1
            if n == 2:
                columns.extend(['x', 'z'])
            elif n == 3:
                columns.extend(['x', 'y', 'z'])
            elif n == 1:
                columns.append(alias)
            else:
                columns.extend([f"{alias}_{i}" for i in range(n)])

        for section in ('aux', 'outputs'):
            columns.extend(dd.get(section, {}).keys())

        try:
            self.db.df_data = pd.DataFrame(
                data=result['tensor'].numpy(), columns=columns
            )
        except Exception:
            self.db.df_data = pd.DataFrame(result['tensor'].numpy())

        if verbose:
            print(f"Tensor shape : {result['tensor'].shape}")
            print(f"Columns      : {columns}")
            print("\nJset loaded into db.jset")
            print("DataFrame loaded into db.df_data\n")

        return result

    # =========================================================================
    # Auxiliary data
    # =========================================================================

    def add_aux(
        self,
        array_name: str,
        array: np.ndarray,
        notes: str = None,
    ) -> None:
        """
        Store an auxiliary spatial array in ``data_dict['aux']``.

        Unlike the CODA format's per-group ``'Aux'`` bucket, NUMPYFILE
        stores all auxiliary arrays in a flat dict under
        ``data_dict['aux']``.  This method mirrors the signature of
        ``BaseSets.add_aux`` and additionally records a human-readable
        description in ``sim_metadata``.

        Parameters
        ----------
        array_name : str
            Key used in ``data_dict['aux']`` (e.g. ``'normals'``).
        array : np.ndarray
            Array to store.  Typically shape ``(n_points,)`` or
            ``(n_points, n_dim)``.
        notes : str or None
            Human-readable description stored in
            ``sim_metadata['keys_aux']``. Default None.

        Side-effects
        ------------
        * Adds or overwrites ``data_dict['aux'][array_name]``.
        * Appends ``notes`` to ``sim_metadata['info_aux']``.
        * Sets ``sim_metadata['keys_aux'][array_name] = notes``.

        Examples
        --------
        Add surface normals computed externally::

            normals = compute_normals(db.data_dict['inputs']['ptos'])
            db.sets.add_aux(
                'surface_normals',
                normals,
                notes='Outward unit normals at cell centroids',
            )
            print(db.data_dict['aux']['surface_normals'].shape)

        The new array is picked up automatically by the next call to
        ``create_jset``::

            result = db.sets.create_jset()
        """
        db = self.db
        db.data_dict.setdefault("aux", {})
        db.sim_metadata.setdefault("info_aux", []).append(notes)
        db.sim_metadata.setdefault("keys_aux", {})[array_name] = notes
        db.data_dict["aux"][array_name] = np.asarray(array)

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _save_result(result: dict, save_path: str) -> None:
        """
        Persist a ``SAM.Gardener`` result dict to disk.

        Supports three formats selected by the file extension:

        * ``.h5``  — GZIP-compressed HDF5 via h5py.
        * ``.pt``  — PyTorch binary format via ``torch.save``.
        * ``.npy`` — NumPy pickle format via ``np.save``.

        Parameters
        ----------
        result : dict
            Dict with keys ``'tensor'``, ``'scaled'``, ``'mins'``, ``'maxs'``
            as ``torch.Tensor`` objects.
        save_path : str
            Destination path including extension.

        Raises
        ------
        NameError
            If the file extension is not ``.h5``, ``.pt`` or ``.npy``.

        Examples
        --------
        ::

            NUMPYFILESets._save_result(result, '/output/jset.h5')
        """
        if save_path.endswith('.h5'):
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("tensor", data=result['tensor'].numpy())
                hf.create_dataset("scaled", data=result['scaled'].numpy())
                hf.create_dataset("mins",   data=result['mins'].numpy())
                hf.create_dataset("maxs",   data=result['maxs'].numpy())
        elif save_path.endswith('.pt'):
            torch.save(result, save_path)
        elif save_path.endswith('.npy'):
            np.save(save_path, result, allow_pickle=True)
        else:
            raise NameError(
                "save_path extension not supported. "
                "Use '.h5', '.pt' or '.npy'."
            )