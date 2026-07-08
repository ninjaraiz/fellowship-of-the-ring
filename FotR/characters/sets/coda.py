"""
sets/coda.py
============
Sets class for the CODA CFD solver format.

Provides higher-level operations on an already-populated FRODO instance
whose reader is CODAReader:

* ML tensor assembly  (``create_jset``)
* pyLOM mesh / dataset creation (``create_pylom_mesh``, ``create_NN_pylom``)
* Mesh-to-mesh interpolation (``interpolate_vol2surf``, ``interpolate_msh2msh``)
* Spatial cropping (``crop_bounding_box``)
* Coordinate reordering (``change_order_coord``)
* I/O helpers (``save_to_npy``, ``save_to_h5``, ``add_to_data_dict``)
"""

import os
import copy
import warnings
from typing import Literal, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import h5py

import pyLOM as SMEAGOL

from ..sam import SAM
from .base import BaseSets

if TYPE_CHECKING:
    from ..frodo import FRODO


class CODASets(BaseSets):
    """
    Sets class for CODA-format FRODO databases.

    All methods operate on ``self.db.data_dict``, which is assumed to be
    structured as::

        data_dict = {
            'CADGroup_<id>': {
                'Coord':          np.ndarray (n_points, n_dim),
                'NodeCoord':      np.ndarray (n_nodes,  n_dim),
                'FlCc':           np.ndarray (n_cases,  n_dvars),
                'Conec':          np.ndarray (n_cells,  max_nodes),
                'idx_sort':       np.ndarray (n_stages, n_cases, n_points),
                'idx_sort_nodes': np.ndarray (n_stages, n_cases, n_nodes),
                'eltype':         np.ndarray (n_points,),
                'cellOrder':      np.ndarray (n_points,),
                'pointOrder':     np.ndarray (n_nodes,),
                'Vars': {
                    '<stage>': {
                        '<var_name>': np.ndarray (n_points, n_cases),
                        ...
                    },
                    ...
                },
                'Aux': {           # optional
                    '<name>': np.ndarray,
                    ...
                },
            },
            ...
        }

    Parameters
    ----------
    db : FRODO
        Parent FRODO instance.
    """

    def __init__(self, db: 'FRODO'):
        super().__init__(db)

    # =========================================================================
    # create_jset
    # =========================================================================

    def create_jset(
        self,
        stage: str,
        id_group: str,
        sol: Union[list, tuple, int, str] = 'all',
        idx_flcc: Union[list, tuple, str] = 'all',
        save_path: Union[str, None] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Assemble mesh coordinates, flight conditions, auxiliary arrays and
        output field variables into a single flat ML-ready joint tensor.

        The tensor layout (per row) is::

            [x_coords | flight_conditions | aux_features | field_variables]

        Each row corresponds to one (point, case) pair so the total number
        of rows is ``n_points × n_cases``.

        The result is stored in ``db.jset`` and a convenience DataFrame is
        stored in ``db.df_data``.

        Parameters
        ----------
        stage : str
            Stage key in ``data_dict[key]['Vars']`` (e.g. ``'0'`` or ``'1'``).
        id_group : str
            CADGroup identifier string (e.g. ``'3'`` or ``'1_2'``).
        sol : 'all', int or list[int]
            Which output variable indices to include. ``'all'`` keeps every
            variable in the stage. An int selects a single variable index.
            A list selects multiple indices.
        idx_flcc : 'all' or list[int]
            Subset of case indices (rows of FlCc) to include. Default 'all'.
        save_path : str or None
            If provided, saves the result to this path. Supported extensions:
            ``.h5``, ``.pt``, ``.npy``. Default None (no saving).
        verbose : bool
            If True, prints the list of included variables and confirms
            where the jset was saved.

        Returns
        -------
        dict
            Result from ``SAM.Gardener.create_final_tensor`` with keys:
            ``'tensor'``, ``'scaled'``, ``'mins'``, ``'maxs'``, ``'info'``.

        Side-effects
        ------------
        * Sets ``db.jset`` to the result dict.
        * Sets ``db.df_data`` to a pandas DataFrame with named columns.

        Raises
        ------
        KeyError
            If ``id_group`` is not found in ``db.data_dict``.
        NameError
            If ``save_path`` has an unsupported extension.

        Examples
        --------
        Minimal usage (all cases, all variables, no saving)::

            db = FRODO(root_dir='/data/sim', format='CODA')
            db.extract_inputs(id_groups=(3,))
            db.extract_outputs(stage=0, id_groups=(3,))
            result = db.sets.create_jset(stage='0', id_group='3')
            print(result['tensor'].shape)   # (n_points * n_cases, n_cols)

        Select specific variables and save to HDF5::

            result = db.sets.create_jset(
                stage='0',
                id_group='3',
                sol=[0, 2],
                idx_flcc=list(range(50)),
                save_path='/output/jset.h5',
                verbose=True,
            )

        Load a previously saved jset::

            reader = SAM.HDF5reader('/output/jset.h5')
            tensor = reader.load_to_tensor('tensor')
        """
        key_group = f'CADGroup_{id_group}'
        dd        = self.db.data_dict[key_group]

        if idx_flcc == 'all':
            idx_flcc = list(range(dd['FlCc'].shape[0]))

        tensor_ptos = dd['Coord']
        tensor_flcc = dd['FlCc'][idx_flcc]

        tensors_aux = (
            [dd['Aux'][n][:, idx_flcc] for n in dd['Aux']]
            if 'Aux' in dd else []
        )

        sol_num = (
            list(range(len(dd['Vars'][stage].keys())))
            if sol == 'all'
            else ([sol] if isinstance(sol, int) else list(sol))
        )

        tensors_out = []
        var_names_selected = []
        for i, (name, arr) in enumerate(dd['Vars'][stage].items()):
            if i in sol_num:
                if verbose:
                    print(f'  Including variable: {name}')
                if arr.ndim == 2:
                    tensors_out.append(arr[:, idx_flcc])
                    var_names_selected.append(name)
                elif arr.ndim == 3:
                    warnings.warn(
                        f"Variable '{name}' is a vector field; "
                        "only scalar variables are fully supported in create_jset. "
                        "Skipping.",
                        UserWarning,
                    )

        result = SAM.Gardener.create_final_tensor(
            tensor_ptos, tensor_flcc, tensors_out, tensors_aux,
            sol=sol, verbose=verbose,
        )

        # ── Optional save ─────────────────────────────────────────────────────
        if save_path:
            self._save_result(result, save_path)
            if verbose:
                print(f"Jset saved to {save_path}\n")

        self.db.jset = result

        # ── Build column names for df_data ────────────────────────────────────
        coord_cols = (
            ['x', 'z']      if dd['Coord'].shape[1] == 2
            else ['x', 'y', 'z'] if dd['Coord'].shape[1] == 3
            else [f'coord_{i}' for i in range(dd['Coord'].shape[1])]
        )
        columns  = coord_cols + self.db.metadata['design_vars']
        if 'Aux' in dd:
            columns += list(dd['Aux'].keys())
        columns += var_names_selected

        self.db.df_data = pd.DataFrame(
            data=result['tensor'].numpy(), columns=columns
        )
        if verbose:
            print("\nJset loaded into db.jset")
            print("DataFrame loaded into db.df_data\n")

        return result

    # =========================================================================
    # pyLOM helpers
    # =========================================================================

    def create_pylom_mesh(
        self,
        id_groups: Union[int, tuple],
    ) -> list:
        """
        Create pyLOM ``Mesh`` objects from stored CADGroup geometry data.

        Converts the FRODO internal connectivity and element-type arrays into
        the format expected by pyLOM (SMEAGOL), mapping VTK cell types to pyLOM
        equivalents (VTK triangle 5 → pyLOM 2; VTK quad 9 → pyLOM 3).

        Parameters
        ----------
        id_groups : int, str or tuple
            CADGroup IDs to convert. A tuple of ints merges those groups.
            Examples: ``3`` for group 3; ``(1, 2)`` for merged groups 1 and 2.

        Returns
        -------
        list[SMEAGOL.Mesh]
            One Mesh object per requested group.

        Examples
        --------
        ::

            meshes = db.sets.create_pylom_mesh(id_groups=(3,))
            mesh   = meshes[0]
            print(mesh)

        Multiple groups::

            meshes = db.sets.create_pylom_mesh(id_groups=((1, 2), 3))
        """
        # id_groups debe ser una lista de string
        # if isinstance(id_groups, (int, tuple)):
        #     id_groups = [id_groups]

        if isinstance(id_groups, int):
            id_groups = [str(id_groups)]
        elif isinstance(id_groups, tuple):
            id_groups = [str(i) for i in id_groups]
        elif isinstance(id_groups, str):
            id_groups = [id_groups]
        elif isinstance(id_groups, list):
            id_groups = [str(i) for i in id_groups]
        else:
            raise TypeError(
                "id_groups must be an int, str, tuple, or list of ints/strs."
            )
            
        mesh_list = []
        for id in id_groups:
            key_suffix = (
                "_".join(map(str, id)) if isinstance(id, tuple) else str(id)
            )
            key   = f"CADGroup_{key_suffix}"
            xyz   = self.db.data_dict[key]["Coord"]
            conec = self.db.data_dict[key]["Conec"]

            eltype = np.array(
                self.db.data_dict[key]["eltype"][0, :], copy=True
            )
            eltype[eltype == 5] = 2   # VTK triangle → pyLOM
            eltype[eltype == 9] = 3   # VTK quad     → pyLOM

            ptable = SMEAGOL.PartitionTable.new(
                1, conec[0, :, :].shape[0], xyz.shape[0]
            )
            mesh = SMEAGOL.Mesh(
                'UNSTRUCT', xyz, conec, eltype,
                self.db.data_dict[key]["cellOrder"][0, :],
                self.db.data_dict[key]["pointOrder"][0, :],
                ptable,
            )
            mesh_list.append(mesh)
            print(mesh)

        return mesh_list

    def create_NN_pylom(
        self,
        id_groups: Union[int, tuple, list],
        stage: int,
        idx_to_print: Union[int, list, str] = 'all',
        external_vars: Union[dict, None] = None,
        save_path: Union[bool, str] = False,
        nan_policy: Literal['fill', 'raise'] = 'fill',
        nan_fill_value: float = 0.0,
    ) -> list:
        """
        Create pyLOM ``Dataset`` objects combining mesh geometry and
        simulation field variables ready for ROM / ML workflows.

        Each variable in ``data_dict[key]['Vars'][stage]`` is packed into
        the pyLOM interleaved storage format.  Scalar fields produce
        ``(n_points, n_cases)`` arrays; vector fields produce
        ``(n_points × n_dim, n_cases)`` interleaved arrays.

        Parameters
        ----------
        id_groups : int, tuple or list
            CADGroup IDs to export. A tuple of ints merges those groups.
            An int is treated as a single group.
        stage : int
            Stage number whose variables are exported (e.g. ``0``).
        idx_to_print : 'all', int or list[int]
            Case indices to include. ``'all'`` exports every available case.
            Default ``'all'``.
        external_vars : dict or None
            Custom parametric variable dict injected instead of design
            variables from ``db.metadata``. Format::

                {
                    'param_name': {
                        'idim': 0,
                        'value': np.ndarray,  # shape (n_cases,)
                    }
                }

            If None, ``db.metadata['design_vars']`` are used.
        save_path : bool or str
            If a directory path string, saves each dataset as
            ``<path>/<key>_stage_<stage>.h5``. Default False (no saving).
        nan_policy : 'fill' or 'raise'
            Action when NaN values are found in a field array:
            ``'fill'`` replaces them with ``nan_fill_value`` and emits a
            ``RuntimeWarning``; ``'raise'`` raises ``ValueError``.
            Default ``'fill'``.
        nan_fill_value : float
            Replacement value used when ``nan_policy='fill'``. Default 0.0.

        Returns
        -------
        list[SMEAGOL.Dataset]
            One Dataset per requested CADGroup.

        Raises
        ------
        AttributeError
            If ``data_dict`` is empty (extract_inputs not yet called).
        IndexError
            If ``idx_to_print`` contains out-of-range indices.
        ValueError
            If a field has an unsupported shape or ``nan_policy='raise'``
            and NaN values are found.

        Examples
        --------
        Export all cases for group 3, stage 0::

            datasets = db.sets.create_NN_pylom(
                id_groups=3, stage=0, save_path='/output/pylom/'
            )
            ds = datasets[0]
            ds.save('/output/pylom/CADGroup_3_stage_0.h5')

        Export a subset of cases with a custom parametric variable::

            datasets = db.sets.create_NN_pylom(
                id_groups=3,
                stage=0,
                idx_to_print=list(range(20)),
                external_vars={
                    'time': {'idim': 0, 'value': time_array},
                },
            )
        """
        if not self.db.data_dict:
            raise AttributeError(
                "data_dict is empty. Run extract_inputs() and "
                "extract_outputs() first."
            )
        if nan_policy not in ('fill', 'raise'):
            raise ValueError("nan_policy must be 'fill' or 'raise'.")

        # if isinstance(id_groups, int):
        #     id_groups = [id_groups]
        
        if isinstance(id_groups, int):
            id_groups = [str(id_groups)]
        elif isinstance(id_groups, tuple):
            id_groups = [str(i) for i in id_groups]
        elif isinstance(id_groups, str):
            id_groups = [id_groups]
        elif isinstance(id_groups, list):
            id_groups = [str(i) for i in id_groups]
        else:
            raise TypeError(
                "id_groups must be an int, str, tuple, or list of ints/strs."
            )

        d_list = []
        for id in id_groups:
            key     = f"CADGroup_{id}"
            xyz     = self.db.data_dict[key]["Coord"]
            conec   = self.db.data_dict[key]["Conec"]
            npoints = xyz.shape[0]

            ptable = SMEAGOL.PartitionTable.new(1, conec.shape[0], npoints)

            fields = [
                n for n in self.db.data_dict[key]['Vars'][str(stage)]
                if n not in ('GlobalNumber', 'CADGroupID')
            ]

            if idx_to_print == 'all':
                idx_to_print = list(
                    range(self.db.data_dict[key]["FlCc"].shape[0])
                )
            elif isinstance(idx_to_print, int):
                idx_to_print = [idx_to_print]

            max_cases = self.db.data_dict[key]["FlCc"].shape[0]
            if any(i >= max_cases or i < 0 for i in idx_to_print):
                raise IndexError(
                    "idx_to_print contains out-of-range indices."
                )
            case_idx = np.asarray(idx_to_print, dtype=np.int64)

            eltype     = self.db.data_dict[key]["eltype"].copy()
            eltype[eltype == 5] = 2
            eltype[eltype == 9] = 3
            cell_order = self.db.data_dict[key]["cellOrder"]

            # ── Parametric variables ─────────────────────────────────────────
            if external_vars is None:
                param_dict = {
                    p: {
                        'idim':  0,
                        'value': self.db.df_state[p].iloc[idx_to_print].values,
                    }
                    for p in self.db.metadata['design_vars']
                }
            else:
                param_dict = external_vars
                for p, content in param_dict.items():
                    val = content['value']
                    idx = np.asarray(idx_to_print)
                    if idx.size > 0 and idx.max() >= len(val):
                        raise IndexError(
                            f"idx_to_print out of range for param '{p}'."
                        )
                    content['value'] = val[idx]

            # ── NaN sanitiser ────────────────────────────────────────────────
            def _sanitize(name: str, value: np.ndarray) -> np.ndarray:
                if not np.issubdtype(value.dtype, np.floating):
                    return np.ascontiguousarray(value)
                nan_mask = np.isnan(value)
                if not np.any(nan_mask):
                    return np.ascontiguousarray(value)
                n_nan = int(np.count_nonzero(nan_mask))
                if nan_policy == 'raise':
                    raise ValueError(
                        f"Variable '{name}' contains {n_nan} NaN values. "
                        "Use nan_policy='fill' or clean the source data."
                    )
                warnings.warn(
                    f"Variable '{name}': replacing {n_nan} NaN values "
                    f"with {nan_fill_value}.",
                    RuntimeWarning,
                )
                value          = value.copy()
                value[nan_mask] = nan_fill_value
                return np.ascontiguousarray(value)

            # ── Pack field variables ─────────────────────────────────────────
            field_dict = {}
            for f in fields:
                va = np.asarray(
                    self.db.data_dict[key]['Vars'][str(stage)][f]
                )

                if va.ndim == 2:
                    # Shape (n_points, n_cases) or (n_cases, n_points)
                    value = (
                        va[:, case_idx]
                        if va.shape[0] == npoints
                        else va[case_idx, :].T
                    )
                    field_dict[f] = {
                        'ndim':  1,
                        'value': _sanitize(f, value),
                    }

                elif va.ndim == 3:
                    # Shape (n_dim, n_points, n_cases)
                    if va.shape[1] != npoints:
                        raise ValueError(
                            f"Vector variable '{f}': axis 1 ({va.shape[1]}) "
                            f"!= npoints ({npoints})."
                        )
                    value        = va[:, :, case_idx]   # (ndim, npts, nc)
                    nd, np_, nc  = value.shape
                    interleaved  = (
                        value.transpose(1, 0, 2)         # (npts, ndim, nc)
                        .reshape(np_ * nd, nc, order='C')
                    )
                    field_dict[f] = {
                        'ndim':  nd,
                        'value': _sanitize(f, interleaved),
                    }

                else:
                    raise ValueError(
                        f"Variable '{f}' has unsupported shape {va.shape}."
                    )

            d = SMEAGOL.Dataset(
                xyz=xyz, ptable=ptable, order=cell_order,
                point=True, vars=param_dict, **field_dict,
            )
            print('DONE', flush=True)

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                out = os.path.join(save_path, f"{key}_stage_{stage}.h5")
                d.save(out)
                print(f"Dataset saved to {out}")

            d_list.append(d)

        return d_list

    # =========================================================================
    # Auxiliary data helpers
    # =========================================================================

    def add_to_data_dict(
        self,
        arr: np.ndarray,
        id_group: str,
        array_name: str,
    ) -> None:
        """
        Store an auxiliary array inside a CADGroup's ``'Aux'`` sub-dict.

        Unlike ``BaseSets.add_aux``, which writes into the global
        ``data_dict['aux']`` bucket, this method stores the array directly
        in ``data_dict['CADGroup_<id_group>']['Aux']``.  This is the CODA
        convention for per-group auxiliary features (e.g. surface normals,
        wall distance).

        Parameters
        ----------
        arr : np.ndarray
            Array to store.  Typically shape (n_points, n_cases) or
            (n_points,) for geometry-only features.
        id_group : str
            CADGroup identifier string (e.g. ``'3'``).
        array_name : str
            Key under which the array is stored in ``'Aux'``.

        Examples
        --------
        ::

            normals = np.load('normals.npy')
            db.sets.add_to_data_dict(normals, id_group='3', array_name='nx')
        """
        group_key = f'CADGroup_{id_group}'
        self.db.data_dict[group_key].setdefault('Aux', {})
        self.db.data_dict[group_key]['Aux'][array_name] = arr

    # =========================================================================
    # Coordinate reordering
    # =========================================================================

    def change_order_coord(
        self,
        id_group: str,
        new_order: Union[str, list, tuple],
        new_nodes_order: Union[None, list, tuple] = None,
    ) -> None:
        """
        Re-sort cell centroids (and nodes) of a CADGroup in-place.

        Reorders ``Coord``, ``Conec``, ``eltype``, ``cellOrder``,
        ``NodeCoord``, ``pointOrder``, the multi-dimensional sort-index
        arrays ``idx_sort`` / ``idx_sort_nodes``, and all field variables
        in ``Vars``.

        Parameters
        ----------
        id_group : str
            CADGroup identifier string (e.g. ``'3'``).
        new_order : str, list or tuple
            If str, one of ``'lexsort'``, ``'centroid'``, ``'kdtree'``,
            ``'convex_hull'``.  The corresponding sorting function from
            ``SAM.Weapons`` is applied to the current ``Coord`` array.
            If list or tuple, treated as an explicit permutation index array
            for the cells.  In this case ``new_nodes_order`` is mandatory.
        new_nodes_order : list, tuple or None
            Explicit permutation index for the nodes.  Required when
            ``new_order`` is a list or tuple; ignored otherwise.

        Raises
        ------
        ValueError
            If ``new_order`` is a list/tuple and ``new_nodes_order`` is None.
        TypeError
            If ``new_order`` is not a recognised str, list, or tuple.

        Examples
        --------
        Sort by lexicographic order::

            db.sets.change_order_coord(id_group='3', new_order='lexsort')

        Apply a custom permutation::

            idx_cells = np.argsort(coords[:, 0])
            idx_nodes = np.argsort(node_coords[:, 0])
            db.sets.change_order_coord(
                id_group='3',
                new_order=idx_cells,
                new_nodes_order=idx_nodes,
            )
        """
        key_group = f'CADGroup_{id_group}'
        data      = copy.deepcopy(self.db.data_dict)
        coord     = data[key_group]['Coord']
        nodecoord = data[key_group]['NodeCoord']

        sort_fn_map = {
            'lexsort':     SAM.Weapons.sort_lexsort,
            'centroid':    SAM.Weapons.sort_by_centroid,
            'kdtree':      SAM.Weapons.sort_closed_curve_by_kdtree,
            'convex_hull': SAM.Weapons.sort_points_by_hull_projection,
        }

        if isinstance(new_order, str):
            if new_order not in sort_fn_map:
                raise ValueError(
                    f"new_order '{new_order}' not supported. "
                    f"Options: {list(sort_fn_map)}."
                )
            fn              = sort_fn_map[new_order]
            _, idx_new      = fn(coord)
            _, idx_nodes_new = fn(nodecoord)
        elif isinstance(new_order, (list, tuple)):
            if new_nodes_order is None:
                raise ValueError(
                    "new_nodes_order must be provided when new_order is "
                    "a list or tuple."
                )
            idx_new       = np.asarray(new_order)
            idx_nodes_new = np.asarray(new_nodes_order)
        else:
            raise TypeError(
                "new_order must be a str or list/tuple of indices."
            )

        for k in ('Coord', 'Conec', 'eltype', 'cellOrder'):
            self.db.data_dict[key_group][k] = data[key_group][k][idx_new]
        for k in ('NodeCoord', 'pointOrder'):
            self.db.data_dict[key_group][k] = data[key_group][k][idx_nodes_new]

        for k, idx in (('idx_sort', idx_new), ('idx_sort_nodes', idx_nodes_new)):
            for s in range(data[key_group][k].shape[0]):
                for c in range(data[key_group][k].shape[1]):
                    self.db.data_dict[key_group][k][s, c, :] = (
                        data[key_group][k][s, c, idx]
                    )

        for s in data[key_group]['Vars']:
            for var in data[key_group]['Vars'][s]:
                self.db.data_dict[key_group]['Vars'][s][var] = (
                    data[key_group]['Vars'][s][var][idx_new]
                )

    # =========================================================================
    # Spatial cropping
    # =========================================================================

    def crop_bounding_box(
        self,
        id_group: str,
        bbox: Union[list, None] = None,
        radius_center: Union[tuple, None] = None,
        new_group_suffix: str = "_crop",
    ) -> None:
        """
        Create a new CADGroup containing only cells within a bounding box
        or a spherical region, derived from an existing CADGroup.

        The new group inherits all arrays from the source group, restricted
        to the selected cell (and node) indices.  Connectivity is re-mapped
        so that node indices remain valid.

        Parameters
        ----------
        id_group : str
            Source CADGroup identifier (e.g. ``'3'``).
        bbox : list or None
            Axis-aligned bounding box as
            ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.
            Mutually exclusive with ``radius_center``.
        radius_center : tuple or None
            ``(radius, center)`` where ``center`` is an array-like of
            shape ``(n_dim,)``.  Selects cells whose centroid is within
            ``radius`` of ``center``.
            Mutually exclusive with ``bbox``.
        new_group_suffix : str
            Suffix appended to the source key to name the new group.
            Default ``'_crop'``.  Example: ``'CADGroup_3_crop'``.

        Raises
        ------
        ValueError
            If neither or both of ``bbox`` / ``radius_center`` are provided.

        Examples
        --------
        Crop to a bounding box::

            db.sets.crop_bounding_box(
                id_group='3',
                bbox=[[-1.0, 1.0], [-0.5, 0.5], [0.0, 0.1]],
            )
            # New key: 'CADGroup_3_crop'

        Crop to a sphere of radius 0.5 centred at the origin::

            db.sets.crop_bounding_box(
                id_group='3',
                radius_center=(0.5, np.array([0.0, 0.0, 0.0])),
                new_group_suffix='_sphere',
            )
        """
        key_old   = f'CADGroup_{id_group}'
        key_new   = f'{key_old}{new_group_suffix}'
        group     = self.db.data_dict[key_old]
        coord     = group['Coord']
        nodecoord = group['NodeCoord']

        if bbox is not None and radius_center is not None:
            raise ValueError(
                "Provide either bbox or radius_center, not both."
            )
        if bbox is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
            mask = (
                (coord[:, 0] >= xmin) & (coord[:, 0] <= xmax) &
                (coord[:, 1] >= ymin) & (coord[:, 1] <= ymax) &
                (coord[:, 2] >= zmin) & (coord[:, 2] <= zmax)
            )
        elif radius_center is not None:
            radius, center = radius_center
            mask = np.linalg.norm(coord - np.asarray(center), axis=1) <= radius
        else:
            raise ValueError("Provide either bbox or radius_center.")

        idx_cells  = np.where(mask)[0]
        conec      = group['Conec'][idx_cells]
        used_nodes = np.unique(conec[conec >= 0])
        node_map   = np.full(nodecoord.shape[0], -1, dtype=np.int64)
        node_map[used_nodes] = np.arange(len(used_nodes))

        new_group: dict = {
            'Coord':     coord[idx_cells],
            'NodeCoord': nodecoord[used_nodes],
            'Conec':     node_map[conec],
            'FlCc':      group['FlCc'],
        }
        for attr in ('eltype', 'cellOrder'):
            if attr in group:
                new_group[attr] = group[attr][idx_cells]
        if 'pointOrder' in group:
            new_group['pointOrder'] = group['pointOrder'][used_nodes]
        if 'idx_sort' in group:
            new_group['idx_sort'] = group['idx_sort'][:, :, idx_cells]
        if 'idx_sort_nodes' in group:
            new_group['idx_sort_nodes'] = (
                group['idx_sort_nodes'][:, :, used_nodes]
            )

        new_group['Vars'] = {}
        for stage in group.get('Vars', {}):
            new_group['Vars'][stage] = {}
            for var, arr in group['Vars'][stage].items():
                if arr.ndim == 2:
                    new_group['Vars'][stage][var] = arr[idx_cells]
                elif arr.ndim == 3:
                    new_group['Vars'][stage][var] = (
                        arr[:, idx_cells]
                        if arr.shape[0] == 3
                        else arr[idx_cells]
                    )

        self.db.data_dict[key_new] = new_group

    # =========================================================================
    # Mesh-to-mesh interpolation
    # =========================================================================

    def interpolate_vol2surf(
        self,
        vol_group: str,
        surf_group: str,
        stage: str,
        vars: Union[str, list] = 'all',
        k: int = 4,
        eps: float = 1e-12,
    ) -> None:
        """
        Interpolate volume cell-centred fields onto surface cell centroids
        using inverse-distance weighting (IDW).

        For each surface point the *k* nearest volume points are found and
        a weighted average (weights proportional to 1/distance) is computed.
        Interpolated arrays are stored in the surface group's ``'Vars'``
        dict with the suffix ``'_interp'``.

        Parameters
        ----------
        vol_group : str
            Source CADGroup identifier (volume mesh, e.g. ``'5'``).
        surf_group : str
            Target CADGroup identifier (surface mesh, e.g. ``'3'``).
        stage : str
            Vars stage key (e.g. ``'0'``).
        vars : 'all' or list[str]
            Variables to interpolate. ``'GlobalNumber'`` and
            ``'CADGroupID'`` are always excluded. Default ``'all'``.
        k : int
            Number of nearest neighbours for IDW. Default 4.
        eps : float
            Small constant added to distances for numerical stability.
            Default 1e-12.

        Side-effects
        ------------
        Adds ``<var>_interp`` entries to
        ``db.data_dict['CADGroup_<surf_group>']['Vars'][stage]``.

        Raises
        ------
        ValueError
            If a variable has an unsupported array shape.

        Examples
        --------
        Interpolate pressure and velocity from volume to surface::

            db.sets.interpolate_vol2surf(
                vol_group='5',
                surf_group='3',
                stage='0',
                vars=['Pressure', 'Velocity'],
                k=6,
            )
            # Access result:
            cp_interp = db.data_dict['CADGroup_3']['Vars']['0']['Pressure_interp']
        """
        from scipy.spatial import cKDTree

        vol  = self.db.data_dict[f'CADGroup_{vol_group}']
        surf = self.db.data_dict[f'CADGroup_{surf_group}']

        tree       = cKDTree(vol['Coord'])
        dist, idx  = tree.query(surf['Coord'], k=k)
        w          = 1.0 / (dist + eps)
        w         /= w.sum(axis=1, keepdims=True)

        surf.setdefault('Vars', {}).setdefault(stage, {})

        var_list = (
            [
                v for v in vol['Vars'][stage]
                if v not in ('GlobalNumber', 'CADGroupID')
            ]
            if vars == 'all' else vars
        )

        for var in var_list:
            arr = vol['Vars'][stage][var]
            if arr.ndim == 2:
                # (n_vol_points, n_cases)  →  (n_surf_points, n_cases)
                surf['Vars'][stage][var + '_interp'] = np.einsum(
                    'ij,ijk->ik', w, arr[idx]
                )
            elif arr.ndim == 3 and arr.shape[0] == 3:
                # (3, n_vol_points, n_cases)  →  (3, n_surf_points, n_cases)
                surf['Vars'][stage][var + '_interp'] = np.einsum(
                    'ij,lijk->lik', w, arr[:, idx, :]
                )
            else:
                raise ValueError(
                    f"Unsupported shape for variable '{var}': {arr.shape}."
                )

    def interpolate_msh2msh(
        self,
        id_group_src: str,
        new_group_id: str,
        new_mesh: dict,
        vars: Union[str, list] = 'all',
        method: str = 'idw',
        k: int = 4,
    ) -> None:
        """
        Interpolate all field variables from a source CADGroup onto a
        different target mesh, storing the result as a new CADGroup.

        All stages and all (non-excluded) variables are processed in a
        single pass per stage.  Variables are stacked into a single matrix
        before interpolation to minimise KDTree queries.

        Parameters
        ----------
        id_group_src : str
            Source CADGroup identifier (e.g. ``'3'``).
        new_group_id : str
            Identifier for the new interpolated CADGroup.
        new_mesh : dict
            Target mesh dict. Must contain at least ``'Coord'`` and,
            for ``method='pyvista'``, also ``'Conec'``.
        vars : 'all' or list[str]
            Variables to interpolate.  ``'GlobalNumber'`` and
            ``'CADGroupID'`` are always excluded.
        method : str
            Interpolation method. Options:

            * ``'idw'``      – inverse-distance weighting (default).
            * ``'griddata'`` – scipy griddata (linear by default).
            * ``'pyvista'``  – PyVista ``sample`` probe (requires Conec).

        k : int
            Nearest neighbours for IDW. Default 4.

        Side-effects
        ------------
        Creates ``db.data_dict['CADGroup_<new_group_id>']`` with the
        interpolated arrays under ``'Vars'`` and the target mesh geometry.

        Raises
        ------
        ValueError
            If ``new_mesh`` does not contain ``'Coord'``, if an unknown
            ``method`` is specified, or if PyVista method is requested
            without ``'Conec'``.

        Examples
        --------
        Interpolate from a coarse mesh (group 3) onto a fine mesh::

            fine_mesh = db_fine.data_dict['CADGroup_3']
            db_coarse.sets.interpolate_msh2msh(
                id_group_src='3',
                new_group_id='3_fine',
                new_mesh=fine_mesh,
                method='idw',
                k=6,
            )
        """
        src       = self.db.data_dict[f'CADGroup_{id_group_src}']
        coord_src = src["Coord"]
        vars_src  = src["Vars"]
        conec_src = src.get("Conec")

        if "Coord" not in new_mesh:
            raise ValueError("new_mesh must contain 'Coord'.")

        coord_dst = new_mesh["Coord"]
        conec_dst = new_mesh.get("Conec")

        if np.shares_memory(coord_src, coord_dst):
            warnings.warn(
                "Source and destination meshes share memory. "
                "This may produce unexpected results.",
                UserWarning,
            )

        new_key = f'CADGroup_{new_group_id}'
        if new_key in self.db.data_dict:
            warnings.warn(
                f"Overwriting existing group '{new_key}'.", UserWarning
            )

        self.db.data_dict[new_key] = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in new_mesh.items()
        }
        if "FlCc" in src:
            self.db.data_dict[new_key]["FlCc"] = src["FlCc"].copy()
        self.db.data_dict[new_key]["Vars"] = {}

        # ── Build KDTree once for IDW ────────────────────────────────────────
        if method == "idw":
            from scipy.spatial import cKDTree
            tree = cKDTree(coord_src)
        else:
            tree = None

        for stage, stage_data in vars_src.items():
            self.db.data_dict[new_key]["Vars"][stage] = {}

            selected = (
                [v for v in stage_data
                 if v not in ("GlobalNumber", "CADGroupID")]
                if vars == 'all' else list(vars)
            )

            var_list:    list = []
            shapes:      list = []
            valid_names: list = []

            for vname in selected:
                if vname not in stage_data:
                    continue
                v = stage_data[vname]
                if v.ndim == 1:
                    v = v[:, None]
                if v.ndim == 3:
                    nd, nc = v.shape[0], v.shape[2]
                    var_list.append(
                        v.transpose(1, 0, 2).reshape(v.shape[1], nd * nc)
                    )
                    shapes.append(('vec', nd, nc))
                else:
                    var_list.append(v)
                    shapes.append(v.shape[1])
                valid_names.append(vname)

            if not var_list:
                continue

            src_stack = np.hstack(var_list)

            if method == "idw":
                dst_stack = SAM.Weapons._interpolate_idw_tree(
                    tree, coord_src, src_stack, coord_dst, k=k,
                )
            elif method == "griddata":
                dst_stack = SAM.Weapons._interpolate_griddata(
                    coord_src, src_stack, coord_dst,
                )
            elif method == "pyvista":
                if conec_src is None or conec_dst is None:
                    raise ValueError(
                        "PyVista interpolation requires 'Conec' in both meshes."
                    )
                dst_stack = SAM.Weapons._interpolate_pyvista(
                    coord_src, conec_src, src_stack, coord_dst, conec_dst,
                )
            else:
                raise ValueError(
                    f"Unknown interpolation method '{method}'. "
                    "Supported: 'idw', 'griddata', 'pyvista'."
                )

            col = 0
            for vname, shape in zip(valid_names, shapes):
                if isinstance(shape, tuple):
                    _, nd, nc = shape
                    chunk = dst_stack[:, col:col + nd * nc]
                    self.db.data_dict[new_key]["Vars"][stage][vname] = (
                        chunk.reshape(chunk.shape[0], nd, nc).transpose(1, 0, 2)
                    )
                    col += nd * nc
                else:
                    self.db.data_dict[new_key]["Vars"][stage][vname] = (
                        dst_stack[:, col:col + shape]
                    )
                    col += shape

    # =========================================================================
    # I/O helpers
    # =========================================================================

    def save_to_npy(
        self,
        stage: int,
        id_group: str,
        filepath: str,
        case_idx: Union[int, list, tuple, str] = 'all',
        ignore_vars: Union[list, tuple, None] = None,
        verbose: bool = False,
    ) -> None:
        """
        Save a stage and group subset of ``data_dict`` to a ``.npy`` file.

        The saved dict contains ``'Coord'``, ``'FlCc'``, ``'idx_sort'``,
        ``'Conec'``, ``'eltype'``, ``'cellOrder'``, all field variables from
        the specified stage (transposed to (n_cases, n_points)), and any
        auxiliary arrays from ``'Aux'``.

        Parameters
        ----------
        stage : int
            Stage number to export (e.g. ``0``).
        id_group : str
            CADGroup identifier string (e.g. ``'3'``).
        filepath : str
            Destination path.  The extension ``.npy`` is appended if absent.
        case_idx : int, list, tuple or 'all'
            Cases to include. Default ``'all'``.
        ignore_vars : list, tuple or None
            Variable names to exclude from the output. Default None.
        verbose : bool
            Print a confirmation message after saving.

        Examples
        --------
        Save all cases, all variables::

            db.sets.save_to_npy(stage=0, id_group='3', filepath='/out/data')

        Save first 50 cases, excluding 'GlobalNumber'::

            db.sets.save_to_npy(
                stage=0, id_group='3', filepath='/out/data_50',
                case_idx=list(range(50)), ignore_vars=['GlobalNumber'],
                verbose=True,
            )
        """
        if not isinstance(id_group, str):
            raise ValueError("id_group must be a string.")

        group_key  = f'CADGroup_{id_group}'
        gd         = self.db.data_dict[group_key]
        stage_vars = gd["Vars"][str(stage)]
        aux_dict   = gd.get('Aux', {})

        if isinstance(case_idx, str):
            if case_idx != 'all':
                raise ValueError("case_idx as str only accepts 'all'.")
            case_idx  = list(range(gd['FlCc'].shape[0]))
            all_cases = True
        elif isinstance(case_idx, int):
            case_idx  = [case_idx]
            all_cases = False
        elif isinstance(case_idx, (list, tuple)):
            all_cases = False
        else:
            raise ValueError("case_idx must be 'all', int, list or tuple.")

        ncases         = len(case_idx)
        npoints        = gd["Coord"].shape[0]
        idx_sort_full  = gd["idx_sort"]

        idx_sort  = np.zeros((ncases, npoints), dtype=np.int32)
        eltype    = np.zeros((ncases, npoints), dtype=np.int32)
        cellOrder = np.zeros((ncases, npoints), dtype=np.int32)
        for ci, c in enumerate(case_idx):
            idx_sort[ci]  = idx_sort_full[stage, c, :]
            eltype[ci]    = gd["eltype"][idx_sort_full[stage, c, :]]
            cellOrder[ci] = gd["cellOrder"][idx_sort_full[stage, c, :]]

        out: dict = {f'CADGroup_{id_group}':{
            'Coord':     gd["Coord"],
            'FlCc':      gd['FlCc'],
            'idx_sort':  idx_sort,
            'Conec':     gd["Conec"],
            'eltype':    eltype,
            'cellOrder': cellOrder,}
        }
        out[f'CADGroup_{id_group}']['Vars'] = {str(stage): {}}

        for var_name, var_data in stage_vars.items():
            if ignore_vars and var_name in ignore_vars:
                continue
            if var_data.ndim == 2 and var_data.shape[0] == npoints:
                out[f'CADGroup_{id_group}']['Vars'][str(stage)][var_name] = np.transpose(var_data[:, case_idx])
            elif var_data.ndim == 3 and var_data.shape[1] == npoints:
                out[f'CADGroup_{id_group}']['Vars'][str(stage)][var_name] = var_data[:, :, case_idx]

        out[f'CADGroup_{id_group}'].update(aux_dict)

        if not filepath.endswith('.npy'):
            filepath += '.npy'
        np.save(filepath, out, allow_pickle=True)

        if verbose:
            label = 'all cases' if all_cases else f'cases {case_idx}'
            print(f"\nSaved {label} to {filepath}")

    def save_to_h5(
        self,
        filepath: str,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Save the full ``data_dict`` to a compressed HDF5 file.

        Variables are stored under a three-level hierarchy:
        ``<CADGroup_key>/Vars/<stage>/{Scalars,Vectors,Gradients}/<var>``.
        Arrays with GZIP compression (level 4), shuffle filter and
        appropriate chunking are used throughout.  The file is opened in
        SWMR mode for safer concurrent reading.

        Parameters
        ----------
        filepath : str
            Destination file path.  The extension ``.h5`` is appended if
            absent.
        overwrite : bool
            If True (default), an existing file is removed before writing.
            If False, raises ``FileExistsError``.
        verbose : bool
            Print group and variable names as they are written.

        Raises
        ------
        FileExistsError
            If the file already exists and ``overwrite=False``.

        Examples
        --------
        ::

            db.sets.save_to_h5('/output/database.h5', verbose=True)

        Loading back::

            reader = SAM.HDF5reader('/output/database.h5')
            reader.print_keys()
        """
        if os.path.exists(filepath):
            if overwrite:
                os.remove(filepath)
            else:
                raise FileExistsError(f"File already exists: {filepath}")

        if not filepath.endswith('.h5'):
            filepath += '.h5'

        def _compressed(grp, name: str, data: np.ndarray) -> None:
            if isinstance(data, np.ndarray) and data.ndim in (2, 3):
                chunks = (1, min(100_000, data.shape[1])) + (
                    (data.shape[2],) if data.ndim == 3 else ()
                )
            else:
                chunks = True
            grp.create_dataset(
                name, data=data,
                compression="gzip", compression_opts=4,
                shuffle=True, chunks=chunks,
            )

        with h5py.File(filepath, "w", libver="latest") as f:
            for group_key, gd in self.db.data_dict.items():
                if verbose:
                    print(f"\nSaving group '{group_key}'")
                grp = f.create_group(group_key)

                grp.create_dataset("Coord",     data=gd["Coord"])
                grp.create_dataset("NodeCoord", data=gd["NodeCoord"])
                grp.create_dataset("FlCc",      data=gd["FlCc"])
                for attr in ("Conec", "eltype", "cellOrder", "pointOrder"):
                    _compressed(grp, attr, gd[attr])

                vg = grp.create_group("Vars")
                for stage, stage_vars in gd["Vars"].items():
                    if verbose:
                        print(f"  Stage {stage}")
                    sg         = vg.create_group(str(stage))
                    scalars_g  = sg.create_group("Scalars")
                    vectors_g  = sg.create_group("Vectors")
                    grads_g    = sg.create_group("Gradients")

                    for vname, vdata in stage_vars.items():
                        to_save = (
                            vdata.T                          if vdata.ndim == 2
                            else np.transpose(vdata, (2, 1, 0))
                        )
                        if to_save.ndim not in (2, 3):
                            raise ValueError(
                                f"Unexpected ndim {vdata.ndim} for '{vname}'."
                            )
                        target = (
                            grads_g   if "Grad" in vname
                            else vectors_g if to_save.ndim == 3
                            else scalars_g
                        )
                        _compressed(target, vname, to_save)
                        if verbose:
                            print(f"    {vname}: {to_save.shape}")

            f.swmr_mode = True

        if verbose:
            print("\nFile saved with GZIP compression, chunking and SWMR.")

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
            Must contain keys ``'tensor'``, ``'scaled'``, ``'mins'``,
            ``'maxs'`` as torch tensors.
        save_path : str
            Destination path. Supported extensions: ``.h5``, ``.pt``, ``.npy``.

        Raises
        ------
        NameError
            If the extension is not supported.
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