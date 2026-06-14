"""
readers/numpy_file.py
=====================
Reader for pre-processed numpy dictionary datasets.

Expected layout
---------------
Each dataset consists of one or more ``.npy`` files, each containing a
Python dict with string keys and ``np.ndarray`` values::

    root_dir/
    ├── db_random.npy        ← dict with keys 'Airfoil', 'Alpha', 'Cp', …
    └── db_structured.npy    ← optional second file

The reader loads every file into memory at construction time via
``np.load(..., allow_pickle=True).item()`` and exposes them through a
unified ``data_dict`` that mirrors the CODA layout as closely as possible:

.. code-block:: text

    data_dict = {
        'inputs': {
            'ptos': np.ndarray (n_points, n_dim),   # coordinates (mandatory)
            'aoa':  np.ndarray (n_cases, 1),
            ...
        },
        'outputs': {
            'cp':   np.ndarray (n_points, n_cases),
            ...
        },
        'aux': {
            'normals': np.ndarray (n_points, n_dim),
            ...
        },
    }
"""

import os
import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd

from ..sam import SAM
from .base import BaseReader


class NUMPYFILEReader(BaseReader):
    """
    Reader for pre-processed numpy dictionary datasets (``.npy`` files).

    Each ``.npy`` file is expected to contain a Python dict (saved with
    ``np.save(..., allow_pickle=True)``).  Variable references follow the
    pattern ``'<filename>/<key>'``, e.g. ``'db_random.npy/Cp'``.

    Parameters
    ----------
    root_dir : str
        Directory where the ``.npy`` files reside.
    file : str, list[str] or tuple[str]
        Relative path(s) to the ``.npy`` file(s) inside ``root_dir``.
        A single string is accepted as well as a list / tuple of strings.

    Attributes
    ----------
    files : list[str]
        Normalised list of relative file paths.
    npy_dict : dict[str, dict]
        Mapping from filename to the loaded numpy dict content.
    order_ptos : np.ndarray
        Sorting permutation applied to the coordinate array by
        ``extract_inputs``.  Set to ``None`` before the first call.

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
    Construct via FRODO::

        from FotR.characters.frodo import FRODO

        db = FRODO(
            root_dir='/data/numpy_db',
            format='NUMPYFILE',
            file='db_random.npy',
            initial_parse=True,
        )

    Construct the reader directly (for testing)::

        from FotR.characters.readers.numpy_file import NUMPYFILEReader

        reader = NUMPYFILEReader(
            root_dir='/data/numpy_db',
            file=['db_random.npy', 'db_structured.npy'],
        )
        reader.parse_simulation_dirs()
        print(reader.sim_metadata)
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

        # ── Load all dictionaries eagerly ────────────────────────────────
        self.npy_dict: dict = {
            f: np.load(
                os.path.join(root_dir, f), allow_pickle=True
            ).item()
            for f in self.files
        }

        self.data_dict   = {"inputs": {}, "outputs": {}, "aux": {}}
        self.order_ptos: np.ndarray = None

    # =========================================================================
    # BaseReader interface
    # =========================================================================

    def parse_simulation_dirs(self) -> None:
        """
        Inspect the loaded ``.npy`` dictionaries and record the shape of
        every array in ``self.sim_metadata``.

        This is a lightweight operation: no data is copied or transformed.
        Its main purpose is to satisfy the FRODO parse contract so that
        ``db.sim_metadata`` is populated before the user calls
        ``extract_inputs`` / ``extract_outputs``.

        Populates
        ---------
        self.sim_metadata : dict
            One entry per file. Structure::

                {
                    'db_random.npy': {
                        'path': '/data/numpy_db/db_random.npy',
                        'keys': {
                            'Airfoil': (500, 2),
                            'Alpha':   (100,),
                            'Cp':      (500, 100),
                            ...
                        },
                    },
                    ...
                }

        self.df_state : pd.DataFrame
            Always set to an empty DataFrame for this format (there are no
            individual "simulation folders" to enumerate).

        Examples
        --------
        ::

            reader.parse_simulation_dirs()
            for fname, meta in reader.sim_metadata.items():
                print(fname, meta['keys'])
        """
        for f in self.files:
            content = self.npy_dict[f]
            self.sim_metadata[f] = {
                "path": os.path.join(self.root_dir, f),
                "keys": {
                    k: np.asarray(v).shape
                    for k, v in content.items()
                },
            }
            print(f"Parsed '{f}'  —  keys: {list(content.keys())}")

        self.df_state = pd.DataFrame()

    def extract_inputs(
        self,
        keys_inputs: dict,
        keys_aux: dict,
        method_to_sort: Literal[
            'centroid', 'kdtree', 'concave_hull', 'lexsort'
        ] = 'centroid',
        common: Union[list, None] = None,
        **kwargs,
    ) -> None:
        """
        Extract input arrays and auxiliary arrays from the ``.npy``
        dictionaries into ``self.data_dict['inputs']`` and
        ``self.data_dict['aux']``.

        The alias ``'ptos'`` is **mandatory** in ``keys_inputs`` and must
        map to a coordinate array of shape ``(n_points, n_dim)``.  After
        loading, the coordinate array is sorted according to
        ``method_to_sort`` and the resulting permutation index is stored in
        ``self.order_ptos`` so that ``extract_outputs`` can apply the same
        ordering.

        All other input arrays are stored as-is (1-D arrays are reshaped
        to ``(n, 1)`` unless they appear in ``common``).

        Auxiliary arrays (e.g. surface normals) are loaded and immediately
        permuted by ``self.order_ptos``.

        Parameters
        ----------
        keys_inputs : dict
            Mapping from alias to ``'<filename>/<key>'``.

            The alias ``'ptos'`` is mandatory and must point to the
            coordinate array.  Example::

                keys_inputs = {
                    'ptos': 'db_random.npy/Airfoil',   # (n_points, 2)
                    'aoa':  'db_random.npy/Alpha',      # (n_cases,)
                    'vel':  'db_random.npy/Vinf',       # (n_cases,)
                }

        keys_aux : dict
            Mapping from alias to ``'<filename>/<key>'`` for spatial
            auxiliary arrays that should be sorted together with ``'ptos'``::

                keys_aux = {
                    'normals': 'db_random.npy/Normals',  # (n_points, 2)
                }

        method_to_sort : str
            Sorting algorithm applied to the ``'ptos'`` coordinate array.
            Options:

            * ``'centroid'`` — sort by polar angle around the centroid
              (default, robust for closed curves such as airfoils).
            * ``'kdtree'``   — nearest-neighbour chain traversal (better
              for open or noisy curves; accepts ``k``, ``start_index``,
              ``alpha`` via ``**kwargs``).
            * ``'concave_hull'`` — project onto the concave hull and sort
              by arc length.
            * ``'lexsort'`` — lexicographic sort (stable, fast, no
              geometric assumptions).

        common : list[str] or None
            Aliases in ``keys_inputs`` whose arrays are identical across
            all cases (e.g. a shared geometry).  These arrays are stored
            without reshaping.

        **kwargs
            Extra arguments forwarded to the sorting function when
            ``method_to_sort='kdtree'``:

            * ``k`` (int)           — number of nearest neighbours (default 3).
            * ``start_index`` (int) — starting point index (default 0).
            * ``alpha`` (float)     — tangent smoothing weight (default 0.7).

        Raises
        ------
        KeyError
            If a key path is not found in the corresponding ``.npy`` dict.
        ValueError
            If ``method_to_sort`` is not one of the supported options, or if
            ``'ptos'`` is absent from ``keys_inputs``.

        Side-effects
        ------------
        * Populates ``self.data_dict['inputs']`` and ``self.data_dict['aux']``.
        * Sets ``self.order_ptos`` (permutation index for later use by
          ``extract_outputs``).
        * Stores ``keys_inputs``, ``keys_aux`` and ``common`` in
          ``self.sim_metadata`` for reference.
        * Calls ``self._check_input_shapes()`` as a consistency check.

        Examples
        --------
        Basic usage with centroid sorting::

            reader.extract_inputs(
                keys_inputs={
                    'ptos': 'db_random.npy/Airfoil',
                    'aoa':  'db_random.npy/Alpha',
                },
                keys_aux={
                    'normals': 'db_random.npy/Normals',
                },
                method_to_sort='centroid',
            )
            print(reader.data_dict['inputs']['ptos'].shape)   # (n_points, 2)
            print(reader.data_dict['inputs']['aoa'].shape)    # (n_cases, 1)

        KDTree sorting with custom parameters::

            reader.extract_inputs(
                keys_inputs={'ptos': 'db.npy/pts', 'time': 'db.npy/t'},
                keys_aux={},
                method_to_sort='kdtree',
                k=10,
                alpha=0.8,
            )
        """
        if 'ptos' not in keys_inputs:
            raise ValueError(
                "'ptos' (coordinate array) is mandatory in keys_inputs."
            )
        if common is None:
            common = []

        self.data_dict["inputs"] = {}
        self.data_dict["aux"]    = {}

        # ── Load input arrays ─────────────────────────────────────────────
        for alias, key_path in keys_inputs.items():
            file_key, key = self._split_key_path(key_path)
            arr = np.asarray(self.npy_dict[file_key][key])
            if alias not in common and arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self.data_dict["inputs"][alias] = arr

        # ── Sort coordinate array ─────────────────────────────────────────
        ptos = self.data_dict["inputs"]["ptos"]

        sort_dispatch = {
            'centroid':     lambda: SAM.Weapons.sort_by_centroid(points=ptos),
            'lexsort':      lambda: SAM.Weapons.sort_lexsort(points=ptos),
            'kdtree':       lambda: SAM.Weapons.sort_closed_curve_by_kdtree(
                ptos,
                k=kwargs.get('k', 3),
                start_index=kwargs.get('start_index', 0),
                alpha=kwargs.get('alpha', 0.7),
            ),
            'concave_hull': lambda: SAM.Weapons.sort_points_by_hull_projection(ptos),
        }
        if method_to_sort not in sort_dispatch:
            raise ValueError(
                f"method_to_sort '{method_to_sort}' not supported. "
                f"Options: {list(sort_dispatch)}."
            )

        sorted_ptos, self.order_ptos = sort_dispatch[method_to_sort]()
        self.data_dict["inputs"]["ptos"] = sorted_ptos

        # ── Load and sort auxiliary arrays ────────────────────────────────
        for alias, key_path in keys_aux.items():
            file_key, key = self._split_key_path(key_path)
            arr = np.asarray(self.npy_dict[file_key][key])
            self.data_dict["aux"][alias] = arr[self.order_ptos]

        # ── Store metadata for reference ──────────────────────────────────
        self.sim_metadata["keys_inputs"] = keys_inputs
        self.sim_metadata["keys_aux"]    = keys_aux
        self.sim_metadata["common"]      = common

        self._check_input_shapes()

    def extract_outputs(self, keys_outputs: dict) -> None:
        """
        Extract output field arrays from the ``.npy`` dictionaries into
        ``self.data_dict['outputs']``.

        The method auto-detects which axis corresponds to ``n_points`` by
        comparing both axes of each loaded array against the shape of the
        already-loaded ``'ptos'`` array, and applies ``self.order_ptos`` to
        reorder the points consistently with ``extract_inputs``.

        Parameters
        ----------
        keys_outputs : dict
            Mapping from alias to ``'<filename>/<key>'``.  Example::

                keys_outputs = {
                    'cp': 'db_random.npy/Cp',   # (n_points, n_cases) or
                                                 # (n_cases, n_points)
                }

        Raises
        ------
        RuntimeError
            If ``extract_inputs`` has not been called first (``order_ptos``
            is None).
        KeyError
            If a key path is not found in the corresponding ``.npy`` dict.
        UserWarning
            If the shape of an output array does not clearly match
            ``(n_points, n_cases)`` or ``(n_cases, n_points)``.

        Side-effects
        ------------
        * Populates ``self.data_dict['outputs']``.
        * Stores ``keys_outputs`` in ``self.sim_metadata``.

        Examples
        --------
        ::

            reader.extract_outputs({'cp': 'db_random.npy/Cp'})
            print(reader.data_dict['outputs']['cp'].shape)
            # → (n_points, n_cases)

        Multiple outputs::

            reader.extract_outputs({
                'cp':  'db_random.npy/Cp',
                'cfx': 'db_random.npy/Cfx',
            })
        """
        if self.order_ptos is None:
            raise RuntimeError(
                "extract_inputs must be called before extract_outputs."
            )

        self.data_dict["outputs"] = {}
        n_points = self.data_dict["inputs"]["ptos"].shape[0]

        for alias, key_path in keys_outputs.items():
            file_key, key = self._split_key_path(key_path)
            arr = np.asarray(self.npy_dict[file_key][key])

            if arr.shape[0] == n_points:
                self.data_dict["outputs"][alias] = arr[self.order_ptos]
            elif arr.ndim > 1 and arr.shape[1] == n_points:
                self.data_dict["outputs"][alias] = arr.T[self.order_ptos]
            else:
                warnings.warn(
                    f"Output '{alias}' has shape {arr.shape}; "
                    "could not determine point/case axes automatically. "
                    "Storing as-is.",
                    UserWarning,
                )
                self.data_dict["outputs"][alias] = arr

        self.sim_metadata["keys_outputs"] = keys_outputs

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _split_key_path(self, key_path: str) -> tuple:
        """
        Split a ``'<filename>/<key>'`` string into its two components.

        Parameters
        ----------
        key_path : str
            Reference of the form ``'db_random.npy/Alpha'``.

        Returns
        -------
        tuple[str, str]
            ``(file_key, key)`` where ``file_key`` is the filename (relative
            path used as key in ``self.npy_dict``) and ``key`` is the array
            name inside that file.

        Raises
        ------
        KeyError
            If ``file_key`` is not in ``self.npy_dict`` or ``key`` is not in
            the corresponding dict.

        Examples
        --------
        ::

            file_key, key = reader._split_key_path('db_random.npy/Alpha')
            # file_key → 'db_random.npy'
            # key      → 'Alpha'
        """
        file_key, key = key_path.split('/', 1)
        if file_key not in self.npy_dict:
            raise KeyError(
                f"File '{file_key}' not found in loaded dictionaries. "
                f"Available: {list(self.npy_dict)}."
            )
        if key not in self.npy_dict[file_key]:
            raise KeyError(
                f"Key '{key}' not found in '{file_key}'. "
                f"Available keys: {list(self.npy_dict[file_key])}."
            )
        return file_key, key

    def _check_input_shapes(self) -> None:
        """
        Verify that arrays in ``self.data_dict['inputs']`` have at most two
        distinct first-dimension sizes (corresponding to ``n_points`` and
        ``n_cases``).

        Emits a ``UserWarning`` if more than two sizes are detected (likely
        a misconfiguration) or if only one size is found (possibly missing
        variables at one level).

        Stores the grouped shape information in ``self.size_inputs`` for
        downstream inspection.

        Side-effects
        ------------
        Sets ``self.size_inputs``: a dict mapping each unique first-dimension
        size to the list of array aliases that have that size.

        Examples
        --------
        ::

            reader._check_input_shapes()
            # self.size_inputs might be:
            # {500: ['ptos', 'normals'], 100: ['aoa', 'vel']}
        """
        inputs = self.data_dict.get("inputs", {})
        if not inputs:
            warnings.warn("data_dict['inputs'] is empty.", UserWarning)
            return

        first_dims = {
            name: arr.shape[0]
            for name, arr in inputs.items()
            if isinstance(arr, np.ndarray)
        }
        unique_sizes = sorted(set(first_dims.values()))
        self.size_inputs: dict = {
            size: [n for n, s in first_dims.items() if s == size]
            for size in unique_sizes
        }

        if len(unique_sizes) > 2:
            warnings.warn(
                f"{len(unique_sizes)} distinct first-dimension sizes detected "
                f"({unique_sizes}). "
                "Check that 'ptos' and parametric arrays have consistent shapes.",
                UserWarning,
            )
        elif len(unique_sizes) == 1:
            warnings.warn(
                "Only one first-dimension size detected across all inputs. "
                "Some variables at a different level may be missing.",
                UserWarning,
            )