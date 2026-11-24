"""Node orderings: the canonical layout of LEGOLAS and how each format maps to it.

Canonical layout
----------------
* **Hexahedra and quadrilaterals of any order** are stored in *tensor* order,
  ``i`` fastest: node ``(i, j, k)`` of an order-``N`` hexahedron is at
  position ``i + (N+1) j + (N+1)**2 k``. This is the layout of HOPR/PyHOPE
  ``*_mesh.h5`` files and of HORSES3D, and it makes interpolation and
  Jacobians a matter of tensor products. It applies to linear hexahedra too,
  so the 8 corners are **not** in VTK order (use :func:`hex_corners`).
* **Tetrahedra, pyramids and prisms** are linear only and keep the VTK/CGNS
  corner order (they coincide for these elements).

Local faces of a hexahedron follow the CGNS face order, which is also the
local side order of HOPR (``SideInfo``)::

    0: zeta = 0 (k = 0)   1: eta = 0 (j = 0)   2: xi = 1 (i = N)
    3: eta = 1 (j = N)    4: xi = 0 (i = 0)    5: zeta = 1 (k = N)

Permutation tables
------------------
Each table ``perm`` maps a native element to the canonical one:
``canonical = native[perm]``. They were measured, not transcribed: gmsh
4.13.1 meshed a unit cube with a single element of order 1-4 and wrote it as
``.msh`` (v2.2 and v4.1) and ``.cgns``; rounding each node's coordinates to
the ``1/N`` grid gives its tensor index. The same reference files live in
``FotR/tests/data/legolas/`` and the tests repeat the measurement.

Two traps the tables encode:

* meshio reorders ``hexahedron27`` to the VTK layout but leaves
  ``hexahedron64`` and ``hexahedron125`` in gmsh's native order, so the
  meshio tables differ from the gmsh ones at order 2 only.
* CGNS and gmsh agree on corners and on the order-1 element, and differ
  from order 2 on (edges, faces and interior are numbered differently).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'tensor_index', 'linear_conn', 'hex_corners', 'hex_face_corners', 'hex_face_grid',
    'quad_corners', 'MESHIO_HEX', 'CGNS_HEX', 'GMSH_HEX', 'HEX_FACES_VTK',
    'LINEAR_FACES', 'NODES_PER_KIND', 'tensor_monotonic', 'VTK_HEX',
    'VTK_QUAD', 'GMSH_QUAD', 'detect_hex_order',
]

#: Corner-level faces of a hexahedron in CGNS/HOPR order, as indices into
#: the 8 VTK-ordered corners. Each is outward-oriented for a right-handed
#: element (counter-clockwise seen from outside).
HEX_FACES_VTK = (
    (0, 3, 2, 1),   # zeta = 0
    (0, 1, 5, 4),   # eta = 0
    (1, 2, 6, 5),   # xi = 1
    (2, 3, 7, 6),   # eta = 1
    (0, 4, 7, 3),   # xi = 0
    (4, 5, 6, 7),   # zeta = 1
)

#: Faces of the linear non-tensor elements, as indices into their VTK
#: corners. Orientation is not relied upon anywhere: code that needs an
#: outward normal checks it against the cell centroid.
LINEAR_FACES = {
    'tet': ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),
    'pyramid': ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    'prism': ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4),
              (2, 0, 3, 5)),
}

#: Nodes of a linear element of each non-tensor kind.
NODES_PER_KIND = {'tet': 4, 'pyramid': 5, 'prism': 6, 'tri': 3}

# --- measured permutations (canonical = native[perm]) ----------------------

_HEX1 = [0, 1, 3, 2, 4, 5, 7, 6]

#: gmsh native order, by order N (gmsh element types 5, 12, 92, 93).
GMSH_HEX = {
    1: _HEX1,
    2: [0, 8, 1, 9, 20, 11, 3, 13, 2, 10, 21, 12, 22, 26, 23, 15, 24, 14,
        4, 16, 5, 17, 25, 18, 7, 19, 6],
    3: [0, 8, 9, 1, 10, 32, 35, 14, 11, 33, 34, 15, 3, 19, 18, 2, 12, 36,
        37, 16, 40, 56, 57, 44, 43, 59, 58, 45, 22, 49, 48, 20, 13, 39, 38,
        17, 41, 60, 61, 47, 42, 63, 62, 46, 23, 50, 51, 21, 4, 24, 25, 5,
        26, 52, 53, 28, 27, 55, 54, 29, 7, 31, 30, 6],
    4: [0, 8, 9, 10, 1, 11, 44, 51, 47, 17, 12, 48, 52, 50, 18, 13, 45, 49,
        46, 19, 3, 25, 24, 23, 2, 14, 53, 57, 54, 20, 62, 98, 106, 99, 71,
        69, 107, 118, 109, 75, 65, 101, 111, 100, 72, 29, 81, 84, 80, 26,
        15, 60, 61, 58, 21, 66, 108, 119, 110, 78, 70, 120, 124, 121, 79,
        68, 113, 122, 112, 76, 30, 85, 88, 87, 27, 16, 56, 59, 55, 22, 63,
        102, 114, 103, 74, 67, 115, 123, 116, 77, 64, 105, 117, 104, 73, 31,
        82, 86, 83, 28, 4, 32, 33, 34, 5, 35, 89, 93, 90, 38, 36, 96, 97, 94,
        39, 37, 92, 95, 91, 40, 7, 43, 42, 41, 6],
}

#: What meshio 5.3 hands back for a gmsh file, by meshio cell type.
MESHIO_HEX = {
    'hexahedron': _HEX1,
    'hexahedron27': [0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 22, 17, 20, 26, 21,
                     19, 23, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6],
    'hexahedron64': GMSH_HEX[3],
    'hexahedron125': GMSH_HEX[4],
}

#: CGNS order, by CGNS ElementType_t code (HEXA_8, HEXA_27, HEXA_64,
#: HEXA_125).
CGNS_HEX = {
    17: _HEX1,
    19: [0, 8, 1, 11, 20, 9, 3, 10, 2, 12, 21, 13, 24, 26, 22, 15, 23, 14,
         4, 16, 5, 19, 25, 17, 7, 18, 6],
    39: [0, 8, 9, 1, 15, 32, 33, 10, 14, 35, 34, 11, 3, 13, 12, 2, 16, 36,
         37, 18, 49, 56, 57, 40, 48, 59, 58, 41, 22, 45, 44, 20, 17, 39, 38,
         19, 50, 60, 61, 43, 51, 63, 62, 42, 23, 46, 47, 21, 4, 24, 25, 5,
         31, 52, 53, 26, 30, 55, 54, 27, 7, 29, 28, 6],
    56: [0, 8, 9, 10, 1, 19, 44, 45, 46, 11, 18, 51, 52, 47, 12, 17, 50, 49,
         48, 13, 3, 16, 15, 14, 2, 20, 53, 54, 55, 23, 82, 98, 99, 100, 62,
         81, 105, 106, 101, 63, 80, 104, 103, 102, 64, 29, 73, 72, 71, 26,
         21, 60, 61, 56, 24, 83, 107, 108, 109, 69, 88, 114, 115, 110, 70,
         87, 113, 112, 111, 65, 30, 74, 79, 78, 27, 22, 59, 58, 57, 25, 84,
         116, 117, 118, 68, 85, 123, 124, 119, 67, 86, 122, 121, 120, 66,
         31, 75, 76, 77, 28, 4, 32, 33, 34, 5, 43, 89, 90, 91, 35, 42, 96,
         97, 92, 36, 41, 95, 94, 93, 37, 7, 40, 39, 38, 6],
}


def linear_conn(kind: str, corners: np.ndarray) -> np.ndarray:
    """Canonical order-1 connectivity from VTK-ordered corners.

    Only hexahedra change (VTK → tensor); the other kinds are returned as
    they come. Use it whenever a reader keeps just the corners of an
    element (serendipity or unsupported high-order types).
    """
    corners = np.asarray(corners)
    if kind == 'hex':
        return corners[:, _HEX1]
    return corners


def tensor_index(i, j, k, order: int):
    """Position of node ``(i, j, k)`` in a canonical hexahedron of *order*."""
    n = order + 1
    return np.asarray(i) + n * np.asarray(j) + n * n * np.asarray(k)


def _corner_ijk(order: int) -> np.ndarray:
    """``(8, 3)`` tensor indices of the corners, in VTK corner order."""
    a = order
    return np.array([(0, 0, 0), (a, 0, 0), (a, a, 0), (0, a, 0),
                     (0, 0, a), (a, 0, a), (a, a, a), (0, a, a)])


def hex_corners(order: int) -> np.ndarray:
    """Canonical positions of the 8 corners of a hexahedron, in VTK order.

    Parameters
    ----------
    order : int
        Geometric order ``N`` of the element.

    Returns
    -------
    numpy.ndarray
        ``(8,)`` indices into the ``(N+1)**3`` canonical nodes.
    """
    c = _corner_ijk(order)
    return tensor_index(c[:, 0], c[:, 1], c[:, 2], order)


def quad_corners(order: int) -> np.ndarray:
    """Canonical positions of the 4 corners of a tensor quadrilateral.

    Returns ``[0, N, (N+1)**2 - 1, N (N+1)]``: counter-clockwise from the
    first node.
    """
    n = order + 1
    return np.array([0, order, n * n - 1, order * n])


def hex_face_corners(order: int) -> np.ndarray:
    """``(6, 4)`` canonical corner positions of each local face.

    Faces in CGNS/HOPR order, each counter-clockwise seen from outside.
    """
    return hex_corners(order)[np.asarray(HEX_FACES_VTK)]


def hex_face_grid(order: int, face: int) -> np.ndarray:
    """Canonical positions of all the nodes of one local face, as a grid.

    Parameters
    ----------
    order : int
        Geometric order ``N``.
    face : int
        Local face, 0-5 (CGNS/HOPR order).

    Returns
    -------
    numpy.ndarray
        ``(N+1, N+1)`` array ``g``; ``g[a, b]`` runs from the face's first
        corner along its first edge with ``a`` and along its last edge with
        ``b``, so ``(d/da) x (d/db)`` points outward.
    """
    c = _corner_ijk(order)[list(HEX_FACES_VTK[face])]
    a = np.arange(order + 1)[:, None, None]
    b = np.arange(order + 1)[None, :, None]
    e1 = (c[1] - c[0]) // order
    e2 = (c[3] - c[0]) // order
    ijk = c[0] + a * e1 + b * e2
    return tensor_index(ijk[..., 0], ijk[..., 1], ijk[..., 2], order)


def tensor_monotonic(points: np.ndarray, conn: np.ndarray, order: int,
                     sample: int | None = 20000, seed: int = 0) -> np.ndarray:
    """Check that hexahedra really are in tensor order.

    Along every grid line of a correctly ordered element, consecutive nodes
    advance in the direction of that line (``x[i+1] - x[i]`` has a positive
    projection on ``x[N] - x[0]``), however curved the element is. A wrong
    permutation sends some node backwards. It is a cheap screen, not a proof:
    a permutation that happens to keep every line monotone would pass.

    Parameters
    ----------
    points : numpy.ndarray
        ``(n_points, 3)`` node coordinates.
    conn : numpy.ndarray
        ``(n_cells, (N+1)**3)`` connectivity, supposedly canonical.
    order : int
        Geometric order ``N``. Order 1 always passes (no inner nodes).
    sample : int or None
        Check at most this many elements, picked at random; ``None`` checks
        all of them.
    seed : int
        Seed of the random pick.

    Returns
    -------
    numpy.ndarray
        Indices (into *conn*) of the checked elements that fail.
    """
    if order < 2 or len(conn) == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.arange(len(conn))
    if sample is not None and len(conn) > sample:
        idx = np.sort(np.random.default_rng(seed).choice(len(conn), sample,
                                                         replace=False))
    n = order + 1
    x = points[conn[idx]].reshape(len(idx), n, n, n, 3)   # [e, k, j, i, :]
    bad = np.zeros(len(idx), dtype=bool)
    for axis in (1, 2, 3):
        d = np.diff(x, axis=axis)
        span = np.take(x, [-1], axis=axis) - np.take(x, [0], axis=axis)
        bad |= ((d * span).sum(-1) <= 0).reshape(len(idx), -1).any(1)
    return idx[bad]

#: VTK Lagrange hexahedron (cell type 72), by order: ``vtk = canonical[perm]``.
#: Measured from VTK 9.5 (``vtkLagrangeHexahedron.GetParametricCoords``).
VTK_HEX = {
    1: _HEX1,
    2: [0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15,
        12, 14, 10, 16, 4, 22, 13],
    3: [0, 3, 15, 12, 48, 51, 63, 60, 1, 2, 7, 11, 13, 14, 4, 8, 49, 50, 55,
        59, 61, 62, 52, 56, 16, 32, 19, 35, 31, 47, 28, 44, 20, 24, 36, 40, 23,
        27, 39, 43, 17, 18, 33, 34, 29, 30, 45, 46, 5, 6, 9, 10, 53, 54, 57,
        58, 21, 22, 25, 26, 37, 38, 41, 42],
    4: [0, 4, 24, 20, 100, 104, 124, 120, 1, 2, 3, 9, 14, 19, 21, 22, 23, 5,
        10, 15, 101, 102, 103, 109, 114, 119, 121, 122, 123, 105, 110, 115, 25,
        50, 75, 29, 54, 79, 49, 74, 99, 45, 70, 95, 30, 35, 40, 55, 60, 65, 80,
        85, 90, 34, 39, 44, 59, 64, 69, 84, 89, 94, 26, 27, 28, 51, 52, 53, 76,
        77, 78, 46, 47, 48, 71, 72, 73, 96, 97, 98, 6, 7, 8, 11, 12, 13, 16, 17,
        18, 106, 107, 108, 111, 112, 113, 116, 117, 118, 31, 32, 33, 36, 37,
        38, 41, 42, 43, 56, 57, 58, 61, 62, 63, 66, 67, 68, 81, 82, 83, 86, 87,
        88, 91, 92, 93],
}

#: VTK Lagrange quadrilateral (cell type 70): position ``i + (N+1) j`` of
#: each VTK point, with ``i`` along the first edge (``a`` of
#: :func:`hex_face_grid`) and ``j`` along the last (``b``).
VTK_QUAD = {
    1: [0, 1, 3, 2],
    2: [0, 2, 8, 6, 1, 5, 7, 3, 4],
    3: [0, 3, 15, 12, 1, 2, 7, 11, 13, 14, 4, 8, 5, 6, 9, 10],
    4: [0, 4, 24, 20, 1, 2, 3, 9, 14, 19, 21, 22, 23, 5, 10, 15, 6, 7, 8, 11,
        12, 13, 16, 17, 18],
}

#: gmsh quadrangle of order N (types 3, 10, 36, 37): position
#: ``a (N+1) + b`` in a :func:`hex_face_grid` of each gmsh node. Measured
#: on the faces of the reference cubes in ``FotR/tests/data/legolas/``.
GMSH_QUAD = {
    1: [0, 2, 3, 1],
    2: [0, 6, 8, 2, 3, 7, 5, 1, 4],
    3: [0, 12, 15, 3, 4, 8, 13, 14, 11, 7, 2, 1, 5, 9, 10, 6],
    4: [0, 20, 24, 4, 5, 10, 15, 21, 22, 23, 19, 14, 9, 3, 2, 1, 6, 16, 18,
        8, 11, 17, 13, 7, 12],
}


def _trilinear_inverse(c, x, iters=30):
    """Reference coordinates in [0, 1]^3 of points *x* ``(m, k, 3)`` in the
    trilinear cells with VTK corners *c* ``(m, 8, 3)``."""
    s = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    u = np.full(x.shape, 0.5)
    for _ in range(iters):
        w = np.where(s[None, None] == 1, u[:, :, None, :],
                     1 - u[:, :, None, :])                 # (m, k, 8, 3)
        N = w.prod(-1)                                    # (m, k, 8)
        F = np.einsum('mkc,mcd->mkd', N, c) - x
        dN = np.empty(N.shape + (3,))
        for d in range(3):
            sign = np.where(s[:, d] == 1, 1.0, -1.0)
            others = np.delete(w, d, axis=-1).prod(-1)
            dN[..., d] = sign * others
        J = np.einsum('mkcd,mce->mked', dN, c)            # (m, k, 3, 3)
        try:
            step = np.linalg.solve(J, -F[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        u = u + step
        if np.abs(step).max() < 1e-12:
            break
    return u


def detect_hex_order(points, conn, order, sample=400, seed=0) -> dict:
    """Find the node permutation of hexahedra whose file order is unknown.

    The corners must be right (they are whenever faces and boundaries match,
    which is what the topology checks); each other node is placed in the
    reference cube by inverting the trilinear map of the corners and rounded
    to the ``1/N`` grid. Over a sample of cells, the most common position of
    every node wins. Mildly curved cells give the same answer; the vote
    absorbs the rest.

    Parameters
    ----------
    points : numpy.ndarray
        ``(n_points, 3)``.
    conn : numpy.ndarray
        ``(n_cells, (N+1)**3)`` connectivity as read; corners already in the
        canonical positions (:func:`hex_corners`).
    order : int
        N.
    sample, seed : int
        Cells used for the vote, and the seed that picks them.

    Returns
    -------
    dict
        ``perm`` (``correct = conn[:, perm]``, or None if no permutation
        comes out), ``agreement`` (share of sampled cells that agree with the
        winner on every node) and ``identity`` (whether nothing has to move).
    """
    n = order + 1
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(conn), min(sample, len(conn)), replace=False)
    c = points[conn[idx][:, hex_corners(order)]]
    x = points[conn[idx]]
    u = _trilinear_inverse(c, x)
    ijk = np.clip(np.rint(u * order), 0, order).astype(int)
    pos = ijk[..., 0] + n * ijk[..., 1] + n * n * ijk[..., 2]   # (m, n^3)
    winner = np.empty(n ** 3, dtype=np.int64)
    for k in range(n ** 3):
        winner[k] = np.bincount(pos[:, k], minlength=n ** 3).argmax()
    if sorted(winner) != list(range(n ** 3)):
        return {'perm': None, 'agreement': 0.0, 'identity': False}
    perm = np.argsort(winner)          # node now at canonical position t
    agreement = float((pos == winner).all(1).mean())
    return {'perm': perm, 'agreement': agreement,
            'identity': bool((perm == np.arange(n ** 3)).all())}

