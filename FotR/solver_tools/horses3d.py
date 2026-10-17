"""Read HORSES3D binary files (``.hsol`` solutions and ``.hmesh`` meshes).

This module only needs numpy, so that it can run where the rest of FotR
cannot (ParaView's ``pvpython`` on a cluster has numpy but not torch). It
must not import anything else from the package.

File format
-----------
Written by ``HexMesh_SaveSolution`` and ``HexMesh_Export``
(``Solver/src/libs/mesh/HexMesh.f90``) through ``SolutionFile.f90``, with
Fortran ``access="stream"``: no record markers, little-endian.

Header, 204 bytes:

========  =====================================================
offset    content
========  =====================================================
0         file name, 128 chars (written trimmed: the rest of the
          field is zeros or spaces)
128       int32 file type (see :data:`FILE_TYPES`)
132       int32 node type: 1 Gauss, 2 Gauss-Lobatto
136       int32 number of elements
140       int32 iteration
144       float64 non-dimensional time
152       6 float64 references: gamma, R, V_ref, rho_ref, T_ref, Mach
          (all zero in a mesh file)
200       int32 = 88, end of the header
========  =====================================================

Then one record per element, in global element order:

* int32 ``4`` and four int32 dimensions ``(n_vars, Nx+1, Ny+1, Nz+1)``,
  followed by the array in Fortran order. ``n_vars`` is 5 conservative
  variables in a solution, and 3 coordinates (times ``Lref``, so in metres
  when ``Lref = 1``) in a mesh;
* with gradients (types 3 and 9), ``U_x``, ``U_y`` and ``U_z`` with no
  header, ``n_grad x (N+1)^3`` each;
* with sensor (types 8 and 9), one float64 per element;
* with ``save les with solution``, the total and the subgrid viscosity,
  ``(N+1)^3`` each. The header does not say so: it is deduced from the size.

The file ends with an int32 ``99``. Its absence means that the solver is
still writing it, or that it was truncated.

Array layout
------------
A Fortran array ``Q(var, i, j, k)`` read in C order is ``[k, j, i, var]``,
and that is how it is returned: per element ``(Nz+1, Ny+1, Nx+1, n_vars)``,
without copying. ``i`` runs along xi, ``j`` along eta, ``k`` along zeta.
Flattening an element gives the local node ``i + (Nx+1) j + (Nx+1)(Ny+1) k``,
which is what :func:`to_nodes` does.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

__all__ = [
    'FILE_TYPES', 'NODE_TYPES', 'HorsesHeader', 'HorsesLayout',
    'IncompleteFileError', 'read_header', 'scan', 'read_solution',
    'read_mesh', 'to_nodes', 'primitive_variables', 'find_solutions',
]

HEADER_SIZE = 204
BEGINNING_DATA = 88
END_OF_FILE = 99

#: File type codes of ``SolutionFile.f90``.
FILE_TYPES = {
    1: 'MESH',
    2: 'SOLUTION',
    3: 'SOLUTION_AND_GRADIENTS',
    4: 'STATS',
    5: 'ZONE_MESH',
    6: 'ZONE_SOLUTION',
    7: 'ZONE_SOLUTION_AND_DOT',
    8: 'SOLUTION_AND_SENSOR',
    9: 'SOLUTION_AND_GRADIENTS_AND_SENSOR',
}

#: Node type codes (``Discretization nodes`` in the control file).
NODE_TYPES = {1: 'Gauss', 2: 'Gauss-Lobatto'}

_VOLUME_TYPES = (1, 2, 3, 8, 9)
_WITH_GRADIENTS = (3, 9)
_WITH_SENSOR = (8, 9)
_REF_NAMES = ('gamma', 'R', 'V_ref', 'rho_ref', 'T_ref', 'Mach')
_RECORD_HEAD = np.dtype([('nd', '<i4'), ('dims', '<i4', 4)])


class IncompleteFileError(ValueError):
    """The file has no end marker: still being written, or truncated."""


@dataclass(frozen=True)
class HorsesHeader:
    """Header of a ``.hsol`` or ``.hmesh`` file.

    Attributes
    ----------
    path : str
        File read.
    name : str
        File name stored inside the file (the one it had when written).
    file_type : int
        Code of :data:`FILE_TYPES`.
    node_type : int
        Code of :data:`NODE_TYPES`.
    n_elements : int
        Number of elements of the whole mesh.
    iteration : int
        Iteration at which the file was written.
    time : float
        Non-dimensional time, in units of ``Lref / V_ref``.
    refs : dict
        ``gamma``, ``R``, ``V_ref``, ``rho_ref``, ``T_ref`` and ``Mach``.
        All zero in a mesh file, and only ``rho_ref``/``V_ref`` are set by
        the incompressible solver.
    size : int
        File size in bytes.
    complete : bool
        Whether the file ends with the end marker (99).
    """

    path: str
    name: str
    file_type: int
    node_type: int
    n_elements: int
    iteration: int
    time: float
    refs: dict
    size: int
    complete: bool

    @property
    def file_type_name(self) -> str:
        return FILE_TYPES[self.file_type]

    @property
    def node_type_name(self) -> str:
        return NODE_TYPES.get(self.node_type, f'unknown ({self.node_type})')

    @property
    def is_mesh(self) -> bool:
        return self.file_type in (1, 5)

    def physical_time(self, lref: float = 1.0) -> float:
        """Time in seconds, ``time * lref / V_ref``.

        Parameters
        ----------
        lref : float, optional
            ``reference length (m)`` of the control file (default 1).
        """
        if self.refs['V_ref'] == 0.0:
            raise ValueError(f"{self.path}: no reference velocity in the header (mesh file?)")
        return self.time * lref / self.refs['V_ref']


@dataclass(frozen=True, eq=False)
class HorsesLayout:
    """Where each element record is in a volume file.

    Attributes
    ----------
    header : HorsesHeader
    n_vars : int
        Variables per node in the main array: 5 in a solution, 3 in a mesh.
    n_grad : int
        Gradient variables per direction, 0 if the file has no gradients.
    has_sensor : bool
    has_les : bool
        Whether each record ends with the total and subgrid viscosities.
    orders : numpy.ndarray
        ``(n_elements, 3)`` polynomial order of each element along xi,
        eta, zeta.
    offsets : numpy.ndarray
        ``(n_elements,)`` byte offset of each record.
    """

    header: HorsesHeader
    n_vars: int
    n_grad: int
    has_sensor: bool
    has_les: bool
    orders: np.ndarray
    offsets: np.ndarray

    @property
    def uniform(self) -> bool:
        """Whether every element has the same orders (fast path)."""
        return bool(np.all(self.orders == self.orders[0]))

    def record_dtype(self, order) -> np.dtype:
        """Structured dtype of one record of an element of the given orders."""
        nx, ny, nz = (int(n) + 1 for n in order)
        fields = [('nd', '<i4'), ('dims', '<i4', 4), ('q', '<f8', (nz, ny, nx, self.n_vars))]
        if self.n_grad:
            fields.append(('grad', '<f8', (3, nz, ny, nx, self.n_grad)))
        if self.has_sensor:
            fields.append(('sensor', '<f8'))
        if self.has_les:
            fields.append(('les', '<f8', (2, nz, ny, nx)))
        return np.dtype(fields)


def read_header(path) -> HorsesHeader:
    """Read the 204-byte header and check the end marker.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    HorsesHeader

    Raises
    ------
    ValueError
        If the file is too short or the header does not end with 88, i.e.
        it is not a HORSES3D binary file.
    """
    path = os.fspath(path)
    size = os.path.getsize(path)
    if size < HEADER_SIZE + 4:
        raise ValueError(f"{path}: {size} bytes, too short for a HORSES3D file")
    with open(path, 'rb') as f:
        raw = f.read(HEADER_SIZE)
        f.seek(size - 4)
        tail = f.read(4)

    terminator = int(np.frombuffer(raw, '<i4', 1, 200)[0])
    if terminator != BEGINNING_DATA:
        raise ValueError(f"{path}: header ends with {terminator}, not {BEGINNING_DATA}: "
                         "not a HORSES3D binary file")
    file_type, node_type, n_elements, iteration = (int(v) for v in np.frombuffer(raw, '<i4', 4, 128))
    if file_type not in FILE_TYPES:
        raise ValueError(f"{path}: unknown file type {file_type}")
    refs = np.frombuffer(raw, '<f8', 6, 152)

    return HorsesHeader(
        path=path,
        name=raw[:128].rstrip(b'\x00 ').decode('latin-1'),
        file_type=file_type,
        node_type=node_type,
        n_elements=n_elements,
        iteration=iteration,
        time=float(np.frombuffer(raw, '<f8', 1, 144)[0]),
        refs={k: float(v) for k, v in zip(_REF_NAMES, refs)},
        size=size,
        complete=int(np.frombuffer(tail, '<i4')[0]) == END_OF_FILE,
    )


def _record_bytes(nodes, n_vars, n_grad, sensor, les):
    return 20 + 8 * nodes * (n_vars + 3 * n_grad + 2 * les) + 8 * sensor


def scan(path, n_grad: int | None = None) -> HorsesLayout:
    """Locate every element record of a volume file.

    If all elements have the same orders the offsets follow from the size
    and a sample of records is checked. Otherwise (p-adaptation) the
    record headers are walked one by one.

    Parameters
    ----------
    path : str or path-like
    n_grad : int, optional
        Gradient variables per direction. Only needed for files with
        gradients and different orders per element; by default it is
        deduced from the size (uniform orders) or taken equal to the
        number of conservative variables, which holds for NS and iNS.

    Returns
    -------
    HorsesLayout

    Raises
    ------
    IncompleteFileError
        If the file has no end marker.
    NotImplementedError
        For surface (zone) and statistics files, types 4 to 7.
    ValueError
        If no layout matches the file size.
    """
    header = read_header(path)
    if header.file_type not in _VOLUME_TYPES:
        raise NotImplementedError(f"{header.path}: file type {header.file_type} "
                                  f"({header.file_type_name}) is not supported yet")
    if not header.complete:
        raise IncompleteFileError(f"{header.path}: no end marker ({END_OF_FILE}); the solver may "
                                  "still be writing it")

    ne = header.n_elements
    with open(header.path, 'rb') as f:
        fd = f.fileno()
        first = np.frombuffer(os.pread(fd, 20, HEADER_SIZE), _RECORD_HEAD)[0]
        if first['nd'] != 4:
            raise ValueError(f"{header.path}: first record has rank {first['nd']}, expected 4")
        n_vars = int(first['dims'][0])
        if header.is_mesh and n_vars != 3:
            raise ValueError(f"{header.path}: mesh file with {n_vars} coordinates per node")
        sensor = header.file_type in _WITH_SENSOR
        gradients = header.file_type in _WITH_GRADIENTS

        # Uniform orders: the size alone gives the layout
        nodes = int(np.prod(first['dims'][1:]))
        per_element, rem = divmod(header.size - HEADER_SIZE - 4, ne)
        if rem == 0:
            fit = _layout_from_size(per_element, nodes, n_vars, gradients, sensor, n_grad)
            if header.is_mesh and fit != (0, False):
                fit = None
            if fit is not None:
                ng, les = fit
                step = _record_bytes(nodes, n_vars, ng, sensor, les)
                offsets = HEADER_SIZE + np.arange(ne, dtype=np.int64) * step
                if _sample_matches(fd, offsets, first['dims'], header.size):
                    orders = np.tile(first['dims'][1:] - 1, (ne, 1)).astype(np.int32)
                    return HorsesLayout(header, n_vars, ng, sensor, les, orders, offsets)

        # Different orders: walk the record headers
        candidates = [(0, False), (0, True)]
        if gradients:
            ng = n_grad if n_grad is not None else n_vars
            candidates = [(ng, False), (ng, True)]
        if header.is_mesh:
            candidates = [(0, False)]
        for ng, les in candidates:
            walked = _walk(fd, ne, n_vars, ng, sensor, les, header.size)
            if walked is not None:
                orders, offsets = walked
                return HorsesLayout(header, n_vars, ng, sensor, les, orders, offsets)

    raise ValueError(f"{header.path}: no record layout matches the file size "
                     f"({header.size} bytes, {ne} elements)")


def _layout_from_size(per_element, nodes, n_vars, gradients, sensor, n_grad):
    """(n_grad, les) such that every record takes per_element bytes, or None."""
    data, rem = divmod(per_element - 20 - 8 * sensor, 8 * nodes)
    if rem or data < n_vars:
        return None
    extra = data - n_vars                       # 3*n_grad + 2*les
    if not gradients:
        return {0: (0, False), 2: (0, True)}.get(extra)
    les = extra % 3 == 2
    ng = (extra - 2 * les) // 3
    if ng < 1 or (n_grad is not None and ng != n_grad):
        return None
    return ng, bool(les)


def _sample_matches(fd, offsets, dims, size, n_sample=64):
    """Check the record headers of the first, last and a spread of elements."""
    if offsets[-1] + 20 > size:
        return False
    picks = np.unique(np.linspace(0, len(offsets) - 1, min(n_sample, len(offsets))).astype(np.int64))
    for e in picks:
        head = np.frombuffer(os.pread(fd, 20, int(offsets[e])), _RECORD_HEAD)[0]
        if head['nd'] != 4 or np.any(head['dims'] != dims):
            return False
    return True


def _walk(fd, ne, n_vars, n_grad, sensor, les, size):
    orders = np.empty((ne, 3), dtype=np.int32)
    offsets = np.empty(ne, dtype=np.int64)
    pos = HEADER_SIZE
    for e in range(ne):
        if pos + 20 > size - 4:
            return None
        head = np.frombuffer(os.pread(fd, 20, pos), _RECORD_HEAD)[0]
        if head['nd'] != 4 or head['dims'][0] != n_vars or np.any(head['dims'][1:] < 1):
            return None
        offsets[e] = pos
        orders[e] = head['dims'][1:] - 1
        pos += _record_bytes(int(np.prod(head['dims'][1:])), n_vars, n_grad, sensor, les)
    if pos != size - 4:
        return None
    return orders, offsets


def _selection(elements, ne):
    if elements is None:
        return np.arange(ne, dtype=np.int64)
    idx = np.atleast_1d(np.asarray(elements, dtype=np.int64))
    if idx.size and (idx.min() < 0 or idx.max() >= ne):
        raise IndexError(f"element indices must be in [0, {ne - 1}]")
    return idx


def _read_records(layout, idx):
    """Structured array of the selected records (uniform) or a list of them."""
    path = layout.header.path
    if layout.uniform:
        dtype = layout.record_dtype(layout.orders[0])
        mm = np.memmap(path, dtype=dtype, mode='r', offset=HEADER_SIZE, shape=(layout.header.n_elements,))
        recs = mm[idx].view(np.ndarray)        # reads only the selected records
        del mm
        bad = (recs['nd'] != 4) | np.any(recs['dims'][:, 1:] != layout.orders[0] + 1, axis=1)
        if np.any(bad):
            raise ValueError(f"{path}: corrupt record for element {int(idx[np.argmax(bad)])}")
        return recs
    recs = []
    with open(path, 'rb') as f:
        for e in idx:
            f.seek(int(layout.offsets[e]))
            rec = np.fromfile(f, dtype=layout.record_dtype(layout.orders[e]), count=1)[0]
            if rec['nd'] != 4 or np.any(rec['dims'][1:] != layout.orders[e] + 1):
                raise ValueError(f"{path}: corrupt record for element {int(e)}")
            recs.append(rec)
    return recs


def _field(recs, name, uniform):
    if uniform:
        return recs[name]
    return [r[name] for r in recs]


def read_solution(path, elements=None, layout: HorsesLayout | None = None) -> dict:
    """Read the solution of all elements or of a selection.

    Parameters
    ----------
    path : str or path-like
        ``.hsol`` volume file (types 2, 3, 8, 9).
    elements : array-like of int, optional
        Global element indices, starting at 0 (``globID - 1``), in the
        order wanted. Only those records are read. Default: all.
    layout : HorsesLayout, optional
        Result of :func:`scan` for this file, to avoid scanning it again
        when reading several selections.

    Returns
    -------
    dict
        * ``header``: :class:`HorsesHeader`; ``layout``: :class:`HorsesLayout`;
          ``elements``: the indices read.
        * ``Q``: conservative variables, non-dimensional,
          ``(n_sel, Nz+1, Ny+1, Nx+1, 5)``; a list of per-element arrays if
          the orders differ between elements.
        * ``U_x``, ``U_y``, ``U_z``: gradients, same shape with ``n_grad``
          variables (only in types 3 and 9).
        * ``sensor``: ``(n_sel,)`` (types 8 and 9).
        * ``mu_total``, ``mu_sgs``: ``(n_sel, Nz+1, Ny+1, Nx+1)`` (LES).
    """
    layout = layout or scan(path)
    if layout.header.is_mesh:
        raise ValueError(f"{layout.header.path} is a mesh file: use read_mesh")
    idx = _selection(elements, layout.header.n_elements)
    recs = _read_records(layout, idx)
    uniform = layout.uniform
    out = {'header': layout.header, 'layout': layout, 'elements': idx,
           'Q': _field(recs, 'q', uniform)}
    if layout.n_grad:
        grad = _field(recs, 'grad', uniform)
        for d, name in enumerate(('U_x', 'U_y', 'U_z')):
            out[name] = grad[:, d] if uniform else [g[d] for g in grad]
    if layout.has_sensor:
        sensor = _field(recs, 'sensor', uniform)
        out['sensor'] = np.asarray(sensor, dtype=np.float64)
    if layout.has_les:
        les = _field(recs, 'les', uniform)
        out['mu_total'] = les[:, 0] if uniform else [m[0] for m in les]
        out['mu_sgs'] = les[:, 1] if uniform else [m[1] for m in les]
    return out


def read_mesh(path, elements=None, layout: HorsesLayout | None = None) -> dict:
    """Read the solution-node coordinates from a ``.hmesh`` file.

    These are the exact positions of the solution nodes used by the solver
    (``Lref`` times the mapped geometry), not the nodes of the input mesh.

    Parameters
    ----------
    path : str or path-like
        ``MESH/<solution>_<iteration>.hmesh`` (type 1).
    elements, layout
        As in :func:`read_solution`.

    Returns
    -------
    dict
        ``header``, ``layout``, ``elements`` and ``x``:
        ``(n_sel, Nz+1, Ny+1, Nx+1, 3)``, or a list if the orders differ.
    """
    layout = layout or scan(path)
    if layout.header.file_type != 1:
        raise ValueError(f"{layout.header.path} is not a mesh file "
                         f"(type {layout.header.file_type_name})")
    idx = _selection(elements, layout.header.n_elements)
    recs = _read_records(layout, idx)
    return {'header': layout.header, 'layout': layout, 'elements': idx,
            'x': _field(recs, 'q', layout.uniform)}


def to_nodes(field) -> np.ndarray:
    """Flatten per-element arrays to ``(n_nodes, n_vars)``.

    Nodes are ordered by element, and inside each element by the local index
    ``i + (Nx+1) j + (Nx+1)(Ny+1) k``.

    Parameters
    ----------
    field : numpy.ndarray or list of numpy.ndarray
        ``(n_el, Nz+1, Ny+1, Nx+1, n_vars)`` or a list of per-element
        ``(Nz+1, Ny+1, Nx+1, n_vars)``. Arrays without the variable axis
        (``mu_total``, ``mu_sgs``) give ``(n_nodes,)``.
    """
    if isinstance(field, np.ndarray):
        scalar = field.ndim == 4
        return field.reshape(-1) if scalar else field.reshape(-1, field.shape[-1])
    if not field:
        return np.empty((0,))
    scalar = field[0].ndim == 3
    parts = [f.reshape(-1) if scalar else f.reshape(-1, f.shape[-1]) for f in field]
    return np.concatenate(parts)


def primitive_variables(Q, refs: dict, physical: bool = False) -> dict:
    """Primitive variables from the conservative ones.

    Parameters
    ----------
    Q : numpy.ndarray
        Conservative variables with the variable axis last (any leading
        shape), as returned by :func:`read_solution`.
    refs : dict
        ``header.refs`` of the solution file.
    physical : bool, optional
        Return SI units instead of non-dimensional values: rho in kg/m3,
        velocities in m/s, pressure in Pa, temperature in K.

    Returns
    -------
    dict
        Compressible NS (``gamma > 0`` in ``refs``): ``rho``, ``u``, ``v``,
        ``w``, ``p``, ``T``, ``Mach`` and ``Cp``, with
        ``p = (gamma-1)(rhoE - rho|u|^2/2)``, ``T = gamma M^2 p / rho`` and
        ``Cp = (p - p_inf) / (1/2)``, ``p_inf = 1/(gamma M^2)``, all
        non-dimensional before scaling.

        Incompressible NS (``gamma = 0``, where ``Q = (rho, rho u, rho v,
        rho w, p)``): ``rho``, ``u``, ``v``, ``w``, ``p`` and ``Cp``.
    """
    Q = np.asarray(Q)
    if Q.shape[-1] != 5:
        raise ValueError(f"expected 5 conservative variables on the last axis, got {Q.shape[-1]}")
    rho = Q[..., 0]
    u, v, w = (Q[..., d] / rho for d in (1, 2, 3))
    gamma, mach = refs['gamma'], refs['Mach']
    rho_ref, v_ref = refs['rho_ref'], refs['V_ref']

    if gamma > 0.0:
        p = (gamma - 1.0) * (Q[..., 4] - 0.5 * rho * (u * u + v * v + w * w))
        T = gamma * mach ** 2 * p / rho
        out = {'rho': rho, 'u': u, 'v': v, 'w': w, 'p': p, 'T': T,
               'Mach': np.sqrt(u * u + v * v + w * w) / np.sqrt(gamma * p / rho),
               'Cp': (p - 1.0 / (gamma * mach ** 2)) / 0.5}
    else:
        p = Q[..., 4]
        out = {'rho': rho, 'u': u, 'v': v, 'w': w, 'p': p, 'Cp': p / 0.5}

    if physical:
        if rho_ref == 0.0 or v_ref == 0.0:
            raise ValueError("refs without rho_ref/V_ref: cannot convert to SI units")
        out['rho'] = rho * rho_ref
        for name in ('u', 'v', 'w'):
            out[name] = out[name] * v_ref
        # compressible: absolute pressure (p_inf becomes rho_ref R T_ref); iNS: gauge
        out['p'] = p * rho_ref * v_ref ** 2
        if 'T' in out:
            out['T'] = out['T'] * refs['T_ref']
    return out


_SNAPSHOT = re.compile(r'^(?P<stem>.+?)(?:_(?P<iter>\d{10}))?\.hsol$')


def find_solutions(directory, stem: str | None = None) -> list:
    """List the volume solutions of a ``RESULTS`` folder, sorted by iteration.

    Parameters
    ----------
    directory : str or path-like
    stem : str, optional
        Solution name without extension (``RESULTS/<stem>.hsol`` and its
        autosaves ``<stem>_<iteration>.hsol``). Default: every ``.hsol``.

    Returns
    -------
    list of HorsesHeader
        Incomplete files are included with ``complete = False``; surface
        and statistics files are left out.
    """
    directory = os.fspath(directory)
    found = []
    for fname in os.listdir(directory):
        m = _SNAPSHOT.match(fname)
        if m is None or fname.endswith('.surf.hsol') or '.stats' in fname:
            continue
        if stem is not None and m.group('stem') != stem:
            continue
        try:
            header = read_header(os.path.join(directory, fname))
        except ValueError:
            continue
        if header.file_type in _VOLUME_TYPES and not header.is_mesh:
            found.append(header)
    return sorted(found, key=lambda h: (h.iteration, h.path))
