"""HOPR / PyHOPE ``*_mesh.h5`` meshes (the input mesh of HORSES3D and FLEXI).

Layout (checked against PyHOPE 0.11 files on 20/11/2026):

* ``NodeCoords`` repeats every node once per element that uses it;
  ``GlobalNodeIDs`` (1-based) says which unique node each row is.
* ``ElemInfo`` rows: ``[type, zone, first side, last side, first node,
  last node]``, ranges half-open and 0-based. Types: ``104/204`` tetra,
  ``105/115/205`` pyramid, ``106/116/206`` prism, ``108/118/208`` hexa
  (``1xx`` straight, ``2xx`` curved).
* Hexahedron nodes are in tensor order, ``i`` fastest, **also for the linear
  108/118** — LEGOLAS's canonical layout, no permutation needed. Verified by
  matching every element face with its neighbour's (100 % on the sphere
  g1/g2 and the wind-turbine g4 meshes).
* ``SideInfo`` rows: ``[side type, global side, neighbour elem (1-based,
  0 none), 10*neighbour local side + flip, BC id (1-based, 0 none)]``.
  The sides of an element come in CGNS face order.
* ``BCNames`` / ``BCType`` (``[type, curved, state, alpha]``); type 100 is
  an inner zone (it marks faces with an element on each side). Such a face
  carries the BC id on **both** of its sides; LEGOLAS keeps it once, on the
  element with the lower index.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Boundary, CellBlock, Mesh

_KINDS = {4: 'tet', 5: 'pyramid', 6: 'prism', 8: 'hex'}
_LINEAR_NODES = {'tet': 4, 'pyramid': 5, 'prism': 6, 'hex': 8}


def _decode(names) -> list:
    out = []
    for n in np.atleast_1d(names):
        if isinstance(n, bytes):
            n = n.decode(errors='replace')
        out.append(str(n).strip())
    return out


def read(path: str) -> Mesh:
    """Read a HOPR/PyHOPE mesh.

    Parameters
    ----------
    path : str
        ``*_mesh.h5`` file.

    Returns
    -------
    Mesh
        ``meta`` keeps ``ngeo``, the file's root attributes
        (``meta['attrs']``), ``bc_type_codes`` (``BCType`` rows) and
        ``file_cell_index`` (LEGOLAS cell → 0-based element of the file;
        add 1 for the element number HORSES3D prints).
    """
    import h5py

    with h5py.File(path, 'r') as fh:
        X = fh['NodeCoords'][()]
        gid = fh['GlobalNodeIDs'][()].astype(np.int64).ravel() - 1
        E = fh['ElemInfo'][()].astype(np.int64)
        S = fh['SideInfo'][()].astype(np.int64)
        names = _decode(fh['BCNames'][()]) if 'BCNames' in fh else []
        types = fh['BCType'][()] if 'BCType' in fh else np.zeros((0, 4), int)
        attrs = {k: (v.item() if hasattr(v, 'item') and np.ndim(v) == 0 else v)
                 for k, v in fh.attrs.items()}

    points = np.zeros((gid.max() + 1, 3))
    points[gid] = X
    mesh = Mesh(points, meta={'format': 'hopr', 'attrs': attrs,
                              'ngeo': int(attrs.get('Ngeo', 1)),
                              'bc_type_codes': np.asarray(types).tolist()})

    n_nodes = E[:, 5] - E[:, 4]
    n_sides = E[:, 3] - E[:, 2]
    order_of_elem = np.arange(len(E))
    bc_elem, bc_face, bc_id, file_index = [], [], [], []
    start = 0
    # group by (type, nodes per element), keeping the file's element order
    # inside each group
    for etype in np.unique(E[:, 0]):
        for nn in np.unique(n_nodes[E[:, 0] == etype]):
            sel = order_of_elem[(E[:, 0] == etype) & (n_nodes == nn)]
            kind = _KINDS.get(int(etype) % 10)
            if kind is None:
                raise ValueError(f"{path}: unknown HOPR element type {etype}")
            if kind == 'hex':
                order = int(round(nn ** (1 / 3))) - 1
                if (order + 1) ** 3 != nn:
                    raise ValueError(f"{path}: hexahedron with {nn} nodes")
            else:
                if nn != _LINEAR_NODES[kind]:
                    raise NotImplementedError(
                        f"{path}: curved {kind} elements (type {etype}, {nn} "
                        "nodes) are not supported yet")
                order = 1
                mesh.warn(f"HOPR {kind} node order assumed to be the CGNS "
                          "corner order (not verified on a real file)")
            conn = gid[E[sel, 4][:, None] + np.arange(nn)]
            block = CellBlock(kind, order, conn, zone=E[sel, 1])
            mesh.blocks.append(block)
            file_index.append(sel)

            nf = block.n_faces
            plain = n_sides[sel] == nf
            if not plain.all():
                mesh.warn(f"{(~plain).sum()} {kind} elements have mortar "
                          "sides; their boundary faces are skipped")
            rows = E[sel[plain], 2][:, None] + np.arange(nf)
            ids = S[rows, 4]
            # an inner BC (type 100) is written on both sides of the face:
            # keep it on the element with the lower file index only
            nb = S[rows, 2] - 1
            here = sel[plain][:, None]
            e_loc, f_loc = np.nonzero((ids > 0) & ((nb < 0) | (here < nb)))
            global_ids = start + np.nonzero(plain)[0][e_loc]
            bc_elem.append(global_ids)
            bc_face.append(f_loc)
            bc_id.append(ids[e_loc, f_loc])
            start += len(sel)

    if bc_elem:
        bc_elem = np.concatenate(bc_elem)
        bc_face = np.concatenate(bc_face)
        bc_id = np.concatenate(bc_id)
        for i, name in enumerate(names):
            m = bc_id == i + 1
            code = (str(int(types[i][0])) if len(types) > i else None)
            mesh.boundaries[name] = Boundary(name, bc_elem[m], bc_face[m],
                                             bc_type=code)
    # global LEGOLAS cell -> element of the file (0-based); identity unless
    # the file mixes element types
    mesh.meta['file_cell_index'] = np.concatenate(file_index)
    for z in np.unique(E[:, 1]):
        mesh.zones[int(z)] = f'zone_{int(z)}'
    return mesh
