"""HORSES3D ``.hmesh`` files: the solution nodes the solver actually used.

The file stores, per element, the ``(N+1)**3`` solution nodes in tensor
order and nothing else: no connectivity, no boundary names. LEGOLAS rebuilds
the connectivity by merging coincident nodes, which only works for
Gauss-Lobatto nodes (they sit on the element faces); with Gauss nodes the
elements share nothing, so every face counts as exterior and a warning says
so. The order here is the polynomial order ``p`` of the solution, not the
geometric order of the mesh. Reading goes through
:func:`FotR.solver_tools.horses3d.read_mesh`.
"""

from __future__ import annotations

import numpy as np

from ..mesh import CellBlock, Mesh
from ..topology import coincident_labels


def merge_points(x: np.ndarray, rtol: float = 1e-9) -> tuple:
    """Merge points closer than ``rtol`` times the bounding-box diagonal.

    Returns
    -------
    points, index : numpy.ndarray
        Unique points and, for every input point, its row in ``points``.
    """
    label, first, _ = coincident_labels(x, rtol)
    return x[first], label


def read(path: str, rtol: float = 1e-9) -> Mesh:
    """Read a ``.hmesh`` file.

    Parameters
    ----------
    path : str
        ``MESH/<solution>_<iteration>.hmesh`` (HORSES3D file type 1).
    rtol : float
        Merge tolerance relative to the bounding-box diagonal.

    Returns
    -------
    Mesh
        One hexahedral block of order ``p``; no boundaries.
    """
    from FotR.solver_tools import horses3d as h3d

    data = h3d.read_mesh(path)
    x = data['x']
    if isinstance(x, list):
        raise NotImplementedError(f"{path}: elements of different orders "
                                  "(p-adaptation) are not supported")
    ne, nz, ny, nx, _ = x.shape
    if not nz == ny == nx:
        raise NotImplementedError(f"{path}: anisotropic orders "
                                  f"{(nx - 1, ny - 1, nz - 1)}")
    points, index = merge_points(x.reshape(-1, 3), rtol)
    conn = index.reshape(ne, -1)
    header = data['header']
    mesh = Mesh(points, blocks=[CellBlock('hex', nx - 1, conn)],
                zones={0: 'zone_0'},
                meta={'format': 'hmesh', 'ngeo': nx - 1,
                      'node_type': header.node_type_name,
                      'note': 'order is the solution order p'})
    if header.node_type != 2:
        mesh.warn(f"{header.node_type_name} nodes do not lie on element "
                  "faces: elements share no nodes and every face is exterior")
    return mesh
