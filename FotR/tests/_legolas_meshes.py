"""Small meshes built in memory for the LEGOLAS tests (not a test module)."""

from __future__ import annotations

import numpy as np

from FotR.characters.legolas import Boundary, CellBlock, Mesh
from FotR.characters.legolas import topology


def box_mesh(nx=3, ny=2, nz=4, order=2, with_bcs=True):
    """Structured box [0,nx]x[0,ny]x[0,nz] of hexahedra of *order*."""
    n = order
    gx, gy, gz = (np.linspace(0, m, m * n + 1) for m in (nx, ny, nz))
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing='ij')
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    sx, sy = nx * n + 1, ny * n + 1
    conn = []
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                kk, jj, ii = np.meshgrid(*(np.arange(n + 1),) * 3,
                                         indexing='ij')
                conn.append(((ei * n + ii) + sx * (ej * n + jj)
                             + sx * sy * (ek * n + kk)).ravel())
    mesh = Mesh(pts, [CellBlock('hex', order, np.array(conn))],
                zones={0: 'fluid'}, meta={'format': 'test'})
    if with_bcs:
        elem, face = topology.exterior_faces(mesh)
        b, local = mesh.locate_cell(elem)
        x = mesh.points[mesh.blocks[0].conn[local][:, 0], 0]
        inlet = (face == 4) & (x == 0)
        mesh.boundaries['inlet'] = Boundary('inlet', elem[inlet], face[inlet],
                                            'BCInflow')
        mesh.boundaries['walls'] = Boundary('walls', elem[~inlet],
                                            face[~inlet], 'BCWall')
    return mesh
