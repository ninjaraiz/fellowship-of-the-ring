"""
readers/pylom.py
================
Reader for pyLOM datasets stored as HDF5 (``.h5``) or pickle (``.pkl``)
files.

Expected layout
---------------
::

    root_dir/
    └── simulation.h5    ← single pyLOM Dataset file

The reader wraps ``pyLOM.Dataset.load`` and exposes the internal variable
and field dictionaries through the standard FRODO ``data_dict`` layout::

    data_dict = {
        'inputs': {
            'ptos': np.ndarray (n_points, n_dim),   # mesh coordinates
            'time': np.ndarray (n_cases, 1),         # parametric vars
            ...
        },
        'outputs': {
            'cp':  np.ndarray (n_points, n_cases),
            'vel': np.ndarray (3, n_points, n_cases),
        },
        'aux': {},
    }

pyLOM field storage convention
-------------------------------
pyLOM stores fields in a flat interleaved layout:

* Scalar field (ndim=1) → ``(n_points, n_cases)`` or ``(n_points,)``
* Vector field (ndim>1) → ``(ndim*n_points, n_cases)`` or
  ``(ndim*n_points,)``

This reader converts them to FRODO's convention on the fly via the static
method ``_field_to_frodo``:

* Scalar single snap → ``(n_points,)``
* Scalar multi snap  → ``(n_points, n_cases)``
* Vector single snap → ``(ndim, n_points)``
* Vector multi snap  → ``(ndim, n_points, n_cases)``
"""

import os
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd

import pyLOM as SMEAGOL

from .base import BaseReader


class PYLOMReader(BaseReader):
    """
    Reader for pyLOM datasets (``.h5`` / ``.pkl`` files).

    The dataset is loaded lazily on the first access via ``_load_dataset``
    and cached in ``self._dataset``.  Subsequent calls to
    ``extract_inputs`` or ``extract_outputs`` reuse the cached object
    without touching the disk again.

    Parameters
    ----------
    root_dir : str
        Directory where the dataset file resides.
    file : str, list[str] or tuple[str]
        Relative path(s) to the dataset file(s) inside ``root_dir``.
        When multiple files are given, only the first is used as the
        primary dataset; support for multi-file merging is reserved for
        future work.

    Attributes
    ----------
    file : str
        Relative path of the primary dataset file.
    files : list[str]
        All provided file paths (normalised to a list).
    _dataset : pyLOM.Dataset or None
        Cached dataset object.  None until the first call to
        ``_load_dataset``.

    Raises
    ------
    FileNotFoundError
        If any of the requested files do not exist on disk.
    TypeError
        If ``file`` is not a string or list/tuple of strings.
    ValueError
        If ``file`` is None.

    Examples
    --------
    Construct via FRODO (recommended)::

        from FotR.characters.frodo import FRODO

        db = FRODO(
            root_dir='/data/pylom',
            format='PYLOM',
            file='simulation.h5',
        )

    Construct and use the reader directly::

        from FotR.characters.readers.pylom import PYLOMReader

        reader = PYLOMReader(root_dir='/data/pylom', file='simulation.h5')
        reader.parse_simulation_dirs()

        reader.extract_inputs(
            keys_inputs={'ptos': 'xyz', 'time': 'time'},
            keys_aux={},
        )
        reader.extract_outputs({'cp': 'Cp', 'vel': 'Velocity'})
        print(reader.data_dict['outputs']['cp'].shape)
    """

    def __init__(self, root_dir: str, file: Union[str, list, tuple], **kwargs):
        super().__init__(root_dir, **kwargs)

        # ── Normalise ``file`` argument ────────────────────────────────────
        if file is None:
            raise ValueError("'file' must not be None.")
        if isinstance(file, str):
            self.files = [file]
        elif isinstance(file, (list, tuple)):
            if not all(isinstance(f, str) for f in file):
                raise TypeError("Every element in 'file' must be a str path.")
            self.files = list(file)
        else:
            raise TypeError(
                "'file' must be a string or a list/tuple of strings."
            )

        # ── Verify files exist ────────────────────────────────────────────
        for f in self.files:
            full = os.path.join(root_dir, f)
            if not os.path.exists(full):
                raise FileNotFoundError(f"File not found: {full}")

        self.file     = self.files[0]
        self.data_dict = {"inputs": {}, "outputs": {}, "aux": {}}
        self._dataset  = None

    # =========================================================================
    # Lazy dataset loading
    # =========================================================================

    def _load_dataset(self) -> 'SMEAGOL.Dataset':
        """
        Load the pyLOM Dataset from disk and cache it.

        The first call reads the file via ``pyLOM.Dataset.load`` and stores
        the result in ``self._dataset``.  Subsequent calls return the cached
        object immediately without touching the disk.

        Returns
        -------
        SMEAGOL.Dataset
            The loaded (and cached) pyLOM Dataset object.

        Examples
        --------
        Internal usage::

            ds = reader._load_dataset()
            print(ds.xyz.shape)         # (n_points, 3)
            print(list(ds.vars.keys()))  # parametric variable names
            print(list(ds.fields.keys())) # field names
        """
        if self._dataset is None:
            t0             = time.perf_counter()
            self._dataset  = SMEAGOL.Dataset.load(
                os.path.join(self.root_dir, self.file)
            )
            elapsed = time.perf_counter() - t0
            print(f"[PYLOMReader] Dataset loaded in {elapsed:.3f} s")
        return self._dataset

    # =========================================================================
    # BaseReader interface
    # =========================================================================

    def parse_simulation_dirs(self) -> None:
        """
        Load the pyLOM Dataset and populate ``sim_metadata`` with a summary
        of all variables and fields.

        Unlike CODA and NUMPYFILE, there are no "simulation folders" to
        walk; the dataset is a single file.  This method records metadata
        (shapes, ndim, idim) and builds ``df_state`` from the zero-dimensional
        parametric variables (case-level scalars such as AoA or Mach).

        Populates
        ---------
        self.sim_metadata : dict
            Structure::

                {
                    'path': '/data/pylom/simulation.h5',
                    'npoints': 5000,
                    'xyz_shape': (5000, 3),
                    'Vars': {
                        'time': {'shape': (100,), 'idim': 0},
                        ...
                    },
                    'Fields': {
                        'Cp':       {'shape': (5000, 100), 'ndim': 1},
                        'Velocity': {'shape': (15000, 100), 'ndim': 3},
                        ...
                    },
                }

        self.df_state : pd.DataFrame
            DataFrame built from all case-level variables (``idim == 0``).
            Each column is one variable; each row is one snapshot / case.
            Set to an empty DataFrame if no case-level variables are found.

        Examples
        --------
        ::

            reader.parse_simulation_dirs()
            print(reader.sim_metadata['npoints'])
            print(reader.df_state.head())
        """
        self.sim_metadata = {
            "path": os.path.join(self.root_dir, self.file)
        }

        ds      = self._load_dataset()
        print(ds)
        npoints = len(ds)

        self.sim_metadata["npoints"]   = npoints
        self.sim_metadata["xyz_shape"] = ds.xyz.shape

        self.sim_metadata["Vars"] = {
            vname: {"shape": vdata["value"].shape, "idim": vdata["idim"]}
            for vname, vdata in ds.vars.items()
        }
        self.sim_metadata["Fields"] = {
            fname: {"shape": fdata["value"].shape, "ndim": fdata["ndim"]}
            for fname, fdata in ds.fields.items()
        }

        # Build df_state from case-level variables (idim == 0)
        case_vars = {
            k: np.asarray(v["value"]).ravel()
            for k, v in ds.vars.items()
            if v["idim"] == 0
        }
        if case_vars:
            try:
                self.df_state = pd.DataFrame(case_vars)
            except ValueError:
                self.df_state = pd.DataFrame(
                    {k: pd.Series(v) for k, v in case_vars.items()}
                )
        else:
            self.df_state = pd.DataFrame()

        print(
            f"[PYLOMReader] Parsed '{self.file}'  "
            f"({npoints} points, "
            f"{len(ds.vars)} vars, "
            f"{len(ds.fields)} fields)"
        )

    def extract_inputs(
        self,
        keys_inputs: dict,
        keys_aux: dict,
        filter_by_vars=None,
        filter_by_fields=None,
    ) -> None:
        """
        Extract mesh coordinates and parametric variables from the pyLOM
        Dataset into ``self.data_dict['inputs']`` and
        ``self.data_dict['aux']``.

        Coordinates
        ~~~~~~~~~~~
        Use the special source key ``'xyz'`` to request the mesh node
        coordinates (``ds.xyz``).  This is the only key that does not come
        from ``ds.vars``.

        Parametric variables
        ~~~~~~~~~~~~~~~~~~~~
        Any other source key must match a key in ``ds.vars``.  1-D arrays
        are reshaped to ``(n, 1)`` for consistency with downstream code.

        Auxiliary fields
        ~~~~~~~~~~~~~~~~
        Keys in ``keys_aux`` must match entries in ``ds.fields``.  They are
        converted from pyLOM's interleaved layout to FRODO's convention via
        :meth:`_field_to_frodo`.

        Parameters
        ----------
        keys_inputs : dict
            Mapping from alias to source key.  The alias ``'ptos'`` should
            map to ``'xyz'`` for mesh coordinates.  All other aliases map to
            variable names in ``ds.vars``.

            Example::

                keys_inputs = {
                    'ptos': 'xyz',     # mesh node coordinates
                    'time': 'time',    # case-level variable
                    'mach': 'Mach',    # case-level variable
                }

        keys_aux : dict
            Mapping from alias to a field name in ``ds.fields``.  These
            are spatial fields used as auxiliary features (not prediction
            targets).

            Example::

                keys_aux = {
                    'wall_dist': 'WallDistance',
                }

        filter_by_vars : any
            Reserved for future use. Currently ignored.
        filter_by_fields : any
            Reserved for future use. Currently ignored.

        Raises
        ------
        KeyError
            If a source key in ``keys_inputs`` is not ``'xyz'`` and is not
            found in ``ds.vars``, or if a source key in ``keys_aux`` is not
            found in ``ds.fields``.

        Side-effects
        ------------
        * Populates ``self.data_dict['inputs']`` and ``self.data_dict['aux']``.
        * Rebuilds ``self.df_state`` from the loaded scalar input variables.
        * Stores ``keys_inputs`` and ``keys_aux`` in ``self.sim_metadata``.

        Examples
        --------
        Extract coordinates, a time variable and a wall-distance aux field::

            reader.extract_inputs(
                keys_inputs={
                    'ptos': 'xyz',
                    'time': 'time',
                },
                keys_aux={
                    'wall_dist': 'WallDistance',
                },
            )
            print(reader.data_dict['inputs']['ptos'].shape)
            print(reader.data_dict['aux']['wall_dist'].shape)
        """
        ds      = self._load_dataset()
        npoints = len(ds)

        self.data_dict["inputs"] = {}
        self.data_dict["aux"]    = {}

        # ── Inputs ────────────────────────────────────────────────────────
        for alias, src_key in keys_inputs.items():
            if src_key == "xyz":
                self.data_dict["inputs"][alias] = ds.xyz.copy()
            elif src_key in ds.vars:
                arr = np.asarray(ds.vars[src_key]["value"])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                self.data_dict["inputs"][alias] = arr
            else:
                raise KeyError(
                    f"[PYLOMReader.extract_inputs] Key '{src_key}' not found. "
                    f"Available vars: {list(ds.vars)}  |  "
                    "Use 'xyz' for mesh coordinates."
                )

        # ── Auxiliary fields ──────────────────────────────────────────────
        for alias, field_key in keys_aux.items():
            if field_key not in ds.fields:
                raise KeyError(
                    f"[PYLOMReader.extract_inputs] Aux key '{field_key}' not "
                    f"found in ds.fields. Available: {list(ds.fields)}."
                )
            fdata = ds.fields[field_key]
            self.data_dict["aux"][alias] = self._field_to_frodo(
                fdata["value"], fdata["ndim"], npoints
            )

        self.sim_metadata["keys_inputs"] = keys_inputs
        self.sim_metadata["keys_aux"]    = keys_aux

        # ── Rebuild df_state from scalar inputs ───────────────────────────
        scalar_inputs = {
            k: v.ravel()
            for k, v in self.data_dict["inputs"].items()
            if k not in ("ptos", "xyz") and isinstance(v, np.ndarray)
            and v.ndim <= 2
        }
        if scalar_inputs:
            try:
                self.df_state = pd.DataFrame(scalar_inputs)
            except ValueError:
                self.df_state = pd.DataFrame(
                    {k: pd.Series(v) for k, v in scalar_inputs.items()}
                )

        print(
            f"[PYLOMReader] extract_inputs done — "
            f"inputs: {list(self.data_dict['inputs'])}  |  "
            f"aux: {list(self.data_dict['aux'])}"
        )

    def extract_outputs(self, keys_outputs: dict) -> None:
        """
        Extract spatial field variables from the pyLOM Dataset into
        ``self.data_dict['outputs']``.

        Field arrays are converted from pyLOM's interleaved format to
        FRODO's convention by :meth:`_field_to_frodo`:

        * Scalar field → ``(n_points, n_cases)``
        * Vector field → ``(ndim, n_points, n_cases)``

        Parameters
        ----------
        keys_outputs : dict
            Mapping from alias to field name in ``ds.fields``.

            Example::

                keys_outputs = {
                    'cp':  'Cp',
                    'vel': 'Velocity',
                }

        Raises
        ------
        RuntimeError
            If ``extract_inputs`` has not been called first (``data_dict``
            does not yet contain a ``'ptos'`` array for shape inference).
        KeyError
            If a field name is not found in ``ds.fields``.

        Side-effects
        ------------
        * Populates ``self.data_dict['outputs']``.
        * Stores ``keys_outputs`` in ``self.sim_metadata``.

        Examples
        --------
        Extract a scalar pressure field and a 3-D velocity field::

            reader.extract_outputs({
                'cp':  'Cp',        # scalar → (n_points, n_cases)
                'vel': 'Velocity',  # vector → (3, n_points, n_cases)
            })
            print(reader.data_dict['outputs']['cp'].shape)
            print(reader.data_dict['outputs']['vel'].shape)
        """
        ds      = self._load_dataset()
        npoints = len(ds)

        self.data_dict["outputs"] = {}

        for alias, field_key in keys_outputs.items():
            if field_key not in ds.fields:
                raise KeyError(
                    f"[PYLOMReader.extract_outputs] Key '{field_key}' not "
                    f"found in ds.fields. Available: {list(ds.fields)}."
                )
            fdata = ds.fields[field_key]
            self.data_dict["outputs"][alias] = self._field_to_frodo(
                fdata["value"], fdata["ndim"], npoints
            )

        self.sim_metadata["keys_outputs"] = keys_outputs

        print(
            f"[PYLOMReader] extract_outputs done — "
            f"outputs: {list(self.data_dict['outputs'])}"
        )

    # =========================================================================
    # Static conversion helpers
    # =========================================================================

    @staticmethod
    def _field_to_frodo(
        value: np.ndarray,
        ndim: int,
        npoints: int,
    ) -> np.ndarray:
        """
        Convert a pyLOM field array to FRODO's storage convention.

        pyLOM stores spatial fields in an interleaved flat layout where the
        spatial components of a vector field are concatenated along the first
        axis::

            scalar (ndim=1):
                single snap → (n_points,)
                multi snap  → (n_points, n_cases)

            vector (ndim>1):
                single snap → (ndim * n_points,)
                multi snap  → (ndim * n_points, n_cases)

        FRODO uses a more intuitive layout where the component axis comes
        first::

            scalar (ndim=1):
                single snap → (n_points,)
                multi snap  → (n_points, n_cases)

            vector (ndim>1):
                single snap → (ndim, n_points)
                multi snap  → (ndim, n_points, n_cases)

        The conversion is carried out by a ``reshape`` followed by a
        ``transpose``, with no data copying beyond what numpy requires.

        Parameters
        ----------
        value : np.ndarray
            Raw field array as returned by pyLOM.
        ndim : int
            Number of spatial components (1 for scalars, 3 for 3-D vectors).
        npoints : int
            Number of mesh points (returned by ``len(ds)``).

        Returns
        -------
        np.ndarray
            Converted array in FRODO's convention.

        Examples
        --------
        Convert a scalar field with 100 snapshots::

            raw   = np.random.rand(5000, 100)   # (n_points, n_cases)
            frodo = PYLOMReader._field_to_frodo(raw, ndim=1, npoints=5000)
            assert frodo.shape == (5000, 100)   # unchanged for scalars

        Convert a 3-D vector field with 100 snapshots::

            raw   = np.random.rand(15000, 100)  # (3 * n_points, n_cases)
            frodo = PYLOMReader._field_to_frodo(raw, ndim=3, npoints=5000)
            assert frodo.shape == (3, 5000, 100)

        Convert a single-snapshot 3-D vector::

            raw   = np.random.rand(15000)
            frodo = PYLOMReader._field_to_frodo(raw, ndim=3, npoints=5000)
            assert frodo.shape == (3, 5000)
        """
        single = (value.ndim == 1)

        if ndim == 1:
            # Scalar: layout is already correct
            if single:
                return value.reshape(npoints)
            return value.reshape(npoints, value.shape[1], order='C')

        # Vector field
        if single:
            # (ndim * npoints,) → (ndim, npoints)
            return value.reshape(npoints, ndim, order='C').T
        # (ndim * npoints, ncases) → (ndim, npoints, ncases)
        ncases = value.shape[1]
        return (
            value.reshape(npoints, ndim, ncases, order='C')
            .transpose(1, 0, 2)
        )