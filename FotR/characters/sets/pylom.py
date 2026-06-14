"""
sets/pylom.py
=============
Sets class for the PYLOM format.

Provides higher-level operations on FRODO instances whose reader is
``PYLOMReader``.  The class covers:

* ML tensor assembly (``create_jset``)
* Round-trip conversion back to a pyLOM ``Dataset`` (``to_pylom_dataset``)
* Field and variable accessor helpers (``get_xyz``, ``get_field``, …)
* Auxiliary array management (``add_aux``)
* Data overview (``summary``)

All methods operate on the three-bucket layout produced by
``PYLOMReader``::

    data_dict = {
        'inputs':  {'ptos': ..., 'time': ..., ...},
        'outputs': {'cp': ..., 'vel': ..., ...},
        'aux':     {'normals': ..., ...},
    }
"""

import os
import warnings
from typing import Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import h5py

import pyLOM as SMEAGOL

from ..sam import SAM
from .base import BaseSets

if TYPE_CHECKING:
    from ..frodo import FRODO


class PYLOMSets(BaseSets):
    """
    Sets class for PYLOM-format FRODO databases.

    All methods read from and write to ``self.db.data_dict``, which is
    assumed to use the three-bucket layout populated by ``PYLOMReader``:
    ``'inputs'``, ``'outputs'`` and ``'aux'``.

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance whose ``data_dict`` is already populated by
        ``PYLOMReader.parse_simulation_dirs``.

    Quick-reference
    ---------------
    ::

        db.sets.get_xyz()                           → (n_points, n_dim)
        db.sets.get_variable('time')                → (n_cases,)
        db.sets.get_field('cp')                     → (n_points, n_cases)
        db.sets.get_field('vel', idim=0)            → (n_points, n_cases)
        db.sets.add_aux('normals', arr, 'normals')  → stores in data_dict['aux']
        db.sets.to_pylom_dataset()                  → pyLOM.Dataset
        db.sets.create_jset()                       → SAM.Gardener ML tensor
        db.sets.field_names()                       → list of output aliases
        db.sets.variable_names()                    → list of input aliases
        db.sets.summary()                           → print data overview
    """

    def __init__(self, db: 'FRODO'):
        super().__init__(db)

    # =========================================================================
    # BaseSets interface
    # =========================================================================

    def create_jset(
        self,
        sol: Union[list, int, str] = 'all',
        save_path: Union[bool, str] = False,
        idx_flcc: Union[list, tuple, str] = 'all',
        ref: Union[dict, None] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Assemble mesh coordinates, parametric variables, auxiliary arrays
        and output fields into a single flat ML-ready joint tensor.

        The method collects three data groups from ``db.data_dict``:

        1. **Coordinates** — ``data_dict['inputs']['ptos']`` or ``'xyz'``
           (whichever is present), shape ``(n_points, n_dim)``.
        2. **Parametric variables** — all remaining arrays in
           ``data_dict['inputs']`` stacked column-wise, giving a matrix of
           shape ``(n_cases, n_params)``.
        3. **Output fields** — all arrays in ``data_dict['outputs']``.
        4. **Auxiliary features** (optional) — all arrays in
           ``data_dict['aux']``.

        All tensors are forwarded to
        :func:`SAM.Gardener.create_final_tensor`, which handles
        broadcasting, flattening and optional min-max normalisation.

        The result is stored in ``db.dict_tensors`` and a column-named
        ``DataFrame`` is stored in ``db.df_data``.

        Parameters
        ----------
        sol : 'all', int or list[int]
            Which output channel(s) to include.  ``'all'`` keeps every
            output; an int selects a single channel; a list selects
            multiple channels. Default ``'all'``.
        save_path : bool or str
            Destination file path.  Supported extensions:

            * ``.h5``  — GZIP-compressed HDF5 (recommended).
            * ``.pt``  — PyTorch serialised dict.

            Default ``False`` (no saving).
        idx_flcc : 'all' or list[int]
            Subset of case (snapshot) indices to include.  Applied before
            channel selection.  Default ``'all'``.
        ref : dict or None
            Reference normalisation dict with keys ``'mins'`` and ``'maxs'``
            (both ``torch.Tensor``).  If None, min and max are computed from
            the current data.  Pass a previously computed ``ref`` to apply
            the same normalisation to a test split.  Default None.
        verbose : bool
            Print tensor shapes and save confirmation. Default False.

        Returns
        -------
        dict
            Result from :func:`SAM.Gardener.create_final_tensor` with keys:

            * ``'tensor'`` — raw joint tensor, shape ``(n_pts×n_cases, n_cols)``.
            * ``'scaled'`` — min-max normalised tensor, same shape.
            * ``'mins'``   — per-column minimums (1-D ``torch.Tensor``).
            * ``'maxs'``   — per-column maximums (1-D ``torch.Tensor``).
            * ``'info'``   — dict with ``'ninputs'`` and ``'noutputs'``.

        Raises
        ------
        ValueError
            If ``data_dict['inputs']`` or ``data_dict['outputs']`` is empty,
            or if no coordinate array (``'ptos'`` / ``'xyz'``) is found.
        ValueError
            If ``save_path`` has an unsupported extension.

        Side-effects
        ------------
        * Sets ``db.dict_tensors`` to the returned result dict.
        * Sets ``db.df_data`` to a ``pd.DataFrame`` with named columns.

        Notes
        -----
        Column names for ``db.df_data`` are inferred from the data shape:

        * 2-column coordinates → ``['x', 'z']``
        * 3-column coordinates → ``['x', 'y', 'z']``
        * 1-column parametric  → the alias string (e.g. ``'time'``)

        Examples
        --------
        Full workflow::

            db = FRODO(root_dir='/data', format='PYLOM', file='sim.h5')
            db.extract_inputs({'ptos': 'xyz', 'time': 'time'}, {})
            db.extract_outputs({'cp': 'Cp'})

            result = db.sets.create_jset(verbose=True)
            print(result['tensor'].shape)    # (n_pts * n_cases, n_cols)
            print(db.df_data.columns.tolist())

        Use the first 50 snapshots only::

            result = db.sets.create_jset(idx_flcc=list(range(50)))

        Apply a pre-computed normalisation to a test set::

            train_result = db_train.sets.create_jset()
            test_result  = db_test.sets.create_jset(ref=train_result)

        Save to HDF5 and load back::

            db.sets.create_jset(save_path='/out/jset.h5')
            reader = SAM.HDF5reader('/out/jset.h5')
            tensor = reader.load_to_tensor('tensor')
        """
        dd = self.db.data_dict

        if not dd.get("inputs"):
            raise ValueError(
                "data_dict['inputs'] is empty. Call extract_inputs() first."
            )
        if not dd.get("outputs"):
            raise ValueError(
                "data_dict['outputs'] is empty. Call extract_outputs() first."
            )

        # ── Locate coordinate array ───────────────────────────────────────
        input_keys = list(dd["inputs"].keys())
        ptos_key   = next(
            (k for k in input_keys if k in ("ptos", "xyz")), None
        )
        if ptos_key is None:
            raise ValueError(
                "No coordinate array found in data_dict['inputs']. "
                "Expected alias 'ptos' or 'xyz'."
            )

        tensor_ptos = torch.from_numpy(np.asarray(dd["inputs"][ptos_key]))

        # ── Parametric variable matrix ────────────────────────────────────
        flcc_arrays = [
            torch.from_numpy(
                np.asarray(dd["inputs"][k]).reshape(-1, 1)
                if np.asarray(dd["inputs"][k]).ndim == 1
                else np.asarray(dd["inputs"][k])
            )
            for k in input_keys if k != ptos_key
        ]
        tensor_flcc = (
            torch.cat(flcc_arrays, dim=1)
            if flcc_arrays
            else torch.empty(0)
        )

        # ── Auxiliary and output tensors ──────────────────────────────────
        tensors_aux = [
            torch.from_numpy(np.asarray(v))
            for v in dd.get("aux", {}).values()
        ]
        tensors_out = [
            torch.from_numpy(np.asarray(v))
            for v in dd["outputs"].values()
        ]

        result = SAM.Gardener.create_final_tensor(
            tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
            sol=sol, idx_flcc=idx_flcc, ref=ref, verbose=verbose,
        )

        # ── Optional save ─────────────────────────────────────────────────
        if save_path:
            self._save_result(result, save_path)
            if verbose:
                print(f"[PYLOMSets] jset saved → {save_path}")

        self.db.dict_tensors = result

        # ── Build column names ────────────────────────────────────────────
        columns: list = []
        for k in input_keys:
            arr  = np.asarray(dd["inputs"][k])
            n    = arr.shape[1] if arr.ndim > 1 else 1
            if k == ptos_key:
                columns.extend(["x", "y", "z"][:n])
            elif n == 1:
                columns.append(k)
            else:
                columns.extend([f"{k}_{i}" for i in range(n)])

        for section in ("aux", "outputs"):
            columns.extend(dd.get(section, {}).keys())

        try:
            self.db.df_data = pd.DataFrame(
                data=result["tensor"].numpy(), columns=columns
            )
        except Exception:
            self.db.df_data = pd.DataFrame(result["tensor"].numpy())

        if verbose:
            print(f"[PYLOMSets] Tensor shape : {result['tensor'].shape}")
            print(f"[PYLOMSets] Columns      : {columns}")
            print("[PYLOMSets] Result stored in db.dict_tensors and db.df_data")

        return result

    # =========================================================================
    # Field and variable accessors
    # =========================================================================

    def get_xyz(self) -> np.ndarray:
        """
        Return the mesh node coordinates array.

        Searches ``data_dict['inputs']`` for an array registered under the
        alias ``'ptos'`` or ``'xyz'`` (whichever is present).

        Returns
        -------
        np.ndarray, shape (n_points, n_dim)

        Raises
        ------
        KeyError
            If no coordinate array is found.

        Examples
        --------
        ::

            xyz = db.sets.get_xyz()
            print(xyz.shape)   # (5000, 3) for a 3-D mesh
        """
        inputs = self.db.data_dict.get("inputs", {})
        for cand in ("ptos", "xyz"):
            if cand in inputs:
                return inputs[cand]
        raise KeyError(
            "Coordinate array not found in data_dict['inputs']. "
            "Run extract_inputs with 'ptos': 'xyz' or 'xyz': 'xyz'."
        )

    def get_variable(self, name: str) -> np.ndarray:
        """
        Return a parametric (case-level) variable as a 1-D array.

        Parameters
        ----------
        name : str
            Alias used in ``extract_inputs`` (e.g. ``'time'``, ``'Mach'``).

        Returns
        -------
        np.ndarray, shape (n_cases,)

        Raises
        ------
        KeyError
            If ``name`` is not found in ``data_dict['inputs']``.

        Examples
        --------
        ::

            time_vec = db.sets.get_variable('time')
            print(time_vec.shape)   # (100,) for 100 snapshots

            mach_vec = db.sets.get_variable('Mach')
        """
        inputs = self.db.data_dict.get("inputs", {})
        if name not in inputs:
            raise KeyError(
                f"Variable '{name}' not found in data_dict['inputs']. "
                f"Available: {list(inputs)}."
            )
        return np.asarray(inputs[name]).ravel()

    def get_field(
        self,
        name: str,
        idim: int = None,
        section: Union[int, slice, None] = None,
    ) -> np.ndarray:
        """
        Return an output field with optional component and case selection.

        Parameters
        ----------
        name : str
            Alias used in ``extract_outputs`` (e.g. ``'cp'``, ``'vel'``).
        idim : int or None
            For vector fields (``ndim > 1``), select a single spatial
            component (0-indexed).  E.g. ``idim=0`` returns the x-component
            of a velocity field stored as ``(3, n_points, n_cases)``.
            Raises ``ValueError`` for scalar fields.  Default None (return
            all components).
        section : int, slice or None
            Select a subset of cases from the trailing axis of the array.
            E.g. ``section=slice(0, 50)`` returns the first 50 snapshots.
            Default None (return all cases).

        Returns
        -------
        np.ndarray
            Shape depends on ``idim`` and ``section``:

            * No selection, scalar  → ``(n_points, n_cases)``
            * No selection, vector  → ``(ndim, n_points, n_cases)``
            * ``idim`` set, vector  → ``(n_points, n_cases)``
            * ``section`` set       → trailing axis restricted to ``section``

        Raises
        ------
        KeyError
            If ``name`` is not found in ``data_dict['outputs']``.
        ValueError
            If ``idim`` is set but the field is scalar (ndim == 1).

        Examples
        --------
        Get the full pressure coefficient field::

            cp = db.sets.get_field('cp')
            print(cp.shape)   # (5000, 100)

        Get the x-component of a velocity vector field::

            vel_x = db.sets.get_field('vel', idim=0)
            print(vel_x.shape)   # (5000, 100)

        Get only the first 20 snapshots::

            cp_train = db.sets.get_field('cp', section=slice(0, 20))
            print(cp_train.shape)   # (5000, 20)

        Combine both selections::

            vel_x_train = db.sets.get_field('vel', idim=0, section=slice(0, 20))
        """
        outputs = self.db.data_dict.get("outputs", {})
        if name not in outputs:
            raise KeyError(
                f"Field '{name}' not found in data_dict['outputs']. "
                f"Available: {list(outputs)}."
            )
        arr = outputs[name]

        if idim is not None:
            if arr.ndim < 3:
                raise ValueError(
                    f"Field '{name}' is scalar (shape {arr.shape}); "
                    "idim selection is only valid for vector fields (ndim > 1)."
                )
            arr = arr[idim]

        if section is not None:
            arr = arr[..., section]

        return arr

    def field_names(self) -> list:
        """
        Return the list of available output field aliases.

        Returns
        -------
        list[str]
            Aliases registered in ``data_dict['outputs']``.

        Examples
        --------
        ::

            print(db.sets.field_names())   # ['cp', 'vel']
        """
        return list(self.db.data_dict.get("outputs", {}).keys())

    def variable_names(self) -> list:
        """
        Return the list of available parametric variable aliases.

        Excludes the coordinate aliases ``'ptos'`` and ``'xyz'``.

        Returns
        -------
        list[str]
            Aliases registered in ``data_dict['inputs']`` (excluding
            coordinate arrays).

        Examples
        --------
        ::

            print(db.sets.variable_names())   # ['time', 'Mach']
        """
        return [
            k for k in self.db.data_dict.get("inputs", {})
            if k not in ("ptos", "xyz")
        ]

    # =========================================================================
    # Round-trip to pyLOM
    # =========================================================================

    def to_pylom_dataset(self) -> 'SMEAGOL.Dataset':
        """
        Reconstruct a ``pyLOM.Dataset`` from the current ``data_dict``.

        Converts the FRODO three-bucket layout back to pyLOM's interleaved
        storage format so the data can be passed directly to pyLOM reduction
        methods (POD, DMD, SPOD, …) or saved with ``Dataset.save()``.

        The conversion is the inverse of ``PYLOMReader._field_to_frodo``:

        * Scalar output ``(n_points, n_cases)`` →
          ``{'ndim': 1, 'value': (n_points, n_cases)}``.
        * Vector output ``(ndim, n_points, n_cases)`` →
          ``{'ndim': ndim, 'value': (ndim*n_points, n_cases)}`` (interleaved).

        Returns
        -------
        SMEAGOL.Dataset
            A fully constructed pyLOM Dataset that mirrors the current
            ``data_dict`` contents.

        Raises
        ------
        KeyError
            If no coordinate array is found (``get_xyz`` fails).
        UserWarning
            If an output array has an unexpected number of dimensions.

        Examples
        --------
        Convert and run a POD reduction::

            ds = db.sets.to_pylom_dataset()
            ds.save('/output/processed_sim.h5')

        Pass to pyLOM ROM::

            ds  = db.sets.to_pylom_dataset()
            pod = pyLOM.POD.run(ds, nModes=10)
        """
        from pyLOM.dataset       import Dataset
        from pyLOM.partition_table import PartitionTable

        inputs  = self.db.data_dict.get("inputs", {})
        outputs = self.db.data_dict.get("outputs", {})

        xyz     = self.get_xyz()
        npoints = xyz.shape[0]
        ptable  = PartitionTable.new(
            nparts=1, nelems=0, npoints=npoints, has_master=False
        )

        # ── Parametric variables ──────────────────────────────────────────
        var_dict: dict = {}
        for i, (alias, arr) in enumerate(inputs.items()):
            if alias in ("ptos", "xyz"):
                continue
            var_dict[alias] = {
                "idim":  i,
                "value": np.asarray(arr).ravel(),
            }

        # ── Output fields ─────────────────────────────────────────────────
        field_dict: dict = {}
        for alias, arr in outputs.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                field_dict[alias] = {
                    "ndim":  1,
                    "value": arr.reshape(npoints),
                }
            elif arr.ndim == 2:
                field_dict[alias] = {
                    "ndim":  1,
                    "value": arr.reshape(npoints, arr.shape[1]),
                }
            elif arr.ndim == 3:
                ndim_f, np_, nc = arr.shape
                field_dict[alias] = {
                    "ndim":  ndim_f,
                    "value": (
                        arr.transpose(1, 0, 2)
                        .reshape(np_ * ndim_f, nc, order='C')
                    ),
                }
            else:
                warnings.warn(
                    f"[PYLOMSets.to_pylom_dataset] Field '{alias}' has "
                    f"unexpected shape {arr.shape}. Stored as-is.",
                    UserWarning,
                )
                field_dict[alias] = {"ndim": 1, "value": arr}

        return Dataset(
            xyz=xyz,
            ptable=ptable,
            vars=var_dict,
            order=np.arange(npoints, dtype=np.int32),
            point=True,
            **field_dict,
        )

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

        Auxiliary arrays are spatial features that are not prediction targets
        (e.g. surface normals, wall distances, mask arrays).  They are picked
        up automatically by ``create_jset`` and inserted between the
        parametric variables and the output fields in the joint tensor.

        Parameters
        ----------
        array_name : str
            Key used in ``data_dict['aux']``.
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
        Add pre-computed surface normals::

            normals = compute_normals(db.sets.get_xyz())   # (n_points, 3)
            db.sets.add_aux('normals', normals, 'outward unit normals')

        Add a boolean mask array::

            mask = np.zeros(n_points)
            mask[boundary_indices] = 1.0
            db.sets.add_aux('boundary_mask', mask)

        Verify it was stored::

            print(list(db.data_dict['aux'].keys()))   # ['normals', 'boundary_mask']
        """
        db = self.db
        db.data_dict.setdefault("aux", {})
        db.sim_metadata.setdefault("info_aux", []).append(notes)
        db.sim_metadata.setdefault("keys_aux", {})[array_name] = notes
        db.data_dict["aux"][array_name] = np.asarray(array)

    # =========================================================================
    # Overview
    # =========================================================================

    def summary(self) -> None:
        """
        Print a compact tabular overview of ``data_dict`` contents.

        For each of the three buckets (``inputs``, ``outputs``, ``aux``),
        lists every array with its shape and dtype.

        Examples
        --------
        ::

            db.sets.summary()

        Sample output::

            ── PYLOMSets summary ────────────────────────────────────
              [inputs]
                ptos                       shape=(5000, 3)  dtype=float64
                time                       shape=(100, 1)   dtype=float64
              [outputs]
                cp                         shape=(5000, 100) dtype=float64
                vel                        shape=(3, 5000, 100) dtype=float64
              [aux]
            ─────────────────────────────────────────────────────────
        """
        dd = self.db.data_dict
        print("── PYLOMSets summary ────────────────────────────────────")
        for section in ("inputs", "outputs", "aux"):
            print(f"  [{section}]")
            for k, v in dd.get(section, {}).items():
                arr = np.asarray(v)
                print(
                    f"    {k:<30s}  shape={arr.shape}  dtype={arr.dtype}"
                )
        print("─────────────────────────────────────────────────────────")

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _save_result(result: dict, save_path: str) -> None:
        """
        Persist a ``SAM.Gardener`` result dict to disk.

        Parameters
        ----------
        result : dict
            Dict with keys ``'tensor'``, ``'scaled'``, ``'mins'``, ``'maxs'``
            as ``torch.Tensor`` objects.
        save_path : str
            Destination path. Supported extensions: ``.h5``, ``.pt``.

        Raises
        ------
        ValueError
            If the file extension is not ``.h5`` or ``.pt``.

        Examples
        --------
        ::

            PYLOMSets._save_result(result, '/output/jset.h5')
            PYLOMSets._save_result(result, '/output/jset.pt')
        """
        if save_path.endswith(".h5"):
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("tensor", data=result["tensor"].numpy())
                hf.create_dataset("scaled", data=result["scaled"].numpy())
                hf.create_dataset("mins",   data=result["mins"].numpy())
                hf.create_dataset("maxs",   data=result["maxs"].numpy())
        elif save_path.endswith(".pt"):
            torch.save(result, save_path)
        else:
            raise ValueError(
                "save_path extension not supported for PYLOMSets. "
                "Use '.h5' or '.pt'."
            )