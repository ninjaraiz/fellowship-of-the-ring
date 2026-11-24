"""Geometry for the viewer: what gets drawn, as plain numpy arrays.

Everything here is independent of HTTP and of the browser, so it is tested
without either. The server (:mod:`.viewer`) only packs these arrays.

Surfaces are lists of element faces (see :class:`.mesh.Boundary`). A
high-order face is drawn through its own node grid, ``N x N`` quadrilaterals
split into two triangles each, so curvature shows at the nodes without any
interpolation. Next to the triangles go the **element edges** (the four
sides of every face, following the nodes), which is what shows the
connectivity; the diagonals of the triangulation are never drawn.

Positions are sent relative to :func:`origin` (the centre of the bounding
box) and in float32, which keeps sub-millimetre detail on domains of
kilometres.
"""

from __future__ import annotations

import json

import numpy as np

from . import ordering, topology
from .mesh import Mesh

__all__ = ['origin', 'zone_faces', 'zone_list', 'face_geometry',
           'section_faces', 'cell_info', 'pack', 'EXTERIOR', 'UNASSIGNED',
           'MAX_TRIANGLES']

#: Pseudo-zone: every exterior face of the mesh.
EXTERIOR = '__exterior__'
#: Pseudo-zone: exterior faces that belong to no boundary.
UNASSIGNED = '__unassigned__'
#: Default budget of triangles per request (~55 MB with edges); above it,
#: faces are decimated.
MAX_TRIANGLES = 1_000_000


def origin(mesh: Mesh) -> np.ndarray:
    """Centre of the bounding box, subtracted from every position sent."""
    if 'origin' not in mesh._cache:
        p = mesh.points
        mesh._cache['origin'] = 0.5 * (p.min(0) + p.max(0))
    return mesh._cache['origin']


def _tris_per_face(block) -> int:
    if block.kind == 'hex':
        return 2 * block.order ** 2
    return 1 if len(ordering.LINEAR_FACES[block.kind][0]) == 3 else 2


def _row_boundary(mesh: Mesh) -> np.ndarray:
    """Boundary index of every row of :func:`.topology.all_faces` (-1 none).

    A face in several boundaries keeps the first one.
    """
    if 'row_bc' not in mesh._cache:
        rows = np.full(len(topology.all_faces(mesh)['elem']), -1, np.int64)
        for i, bc in reversed(list(enumerate(mesh.boundaries.values()))):
            rows[topology.face_rows(mesh, bc.elem, bc.face)] = i
        mesh._cache['row_bc'] = rows
    return mesh._cache['row_bc']


def zone_faces(mesh: Mesh, name: str) -> tuple:
    """``(elem, face)`` of a boundary, or of a pseudo-zone.

    Parameters
    ----------
    name : str
        A key of ``mesh.boundaries``, :data:`EXTERIOR` or
        :data:`UNASSIGNED`.
    """
    if name in mesh.boundaries:
        bc = mesh.boundaries[name]
        return bc.elem, bc.face
    elem, face = topology.exterior_faces(mesh)
    if name == EXTERIOR:
        return elem, face
    if name == UNASSIGNED:
        rows = topology.face_rows(mesh, elem, face)
        free = _row_boundary(mesh)[rows] < 0
        return elem[free], face[free]
    raise KeyError(f"no boundary {name!r}")


def zone_list(mesh: Mesh) -> list:
    """The zones the viewer offers, with their sizes.

    Returns
    -------
    list[dict]
        ``name``, ``bc_type``, ``n_faces``, ``interior`` (faces with a cell
        on each side) and ``triangles`` (what drawing it costs). Includes
        :data:`UNASSIGNED` when there are such faces, and :data:`EXTERIOR`
        when the mesh has no boundaries at all.
    """
    idx = topology.face_index(mesh)
    count = idx['count'][idx['inverse']]
    out = []
    names = list(mesh.boundaries)
    if not names:
        names.append(EXTERIOR)
    elif len(zone_faces(mesh, UNASSIGNED)[0]):
        names.append(UNASSIGNED)
    for name in names:
        elem, face = zone_faces(mesh, name)
        rows = topology.face_rows(mesh, elem, face)
        b_of, _ = mesh.locate_cell(elem)
        tris = sum(int((b_of == b).sum()) * _tris_per_face(blk)
                   for b, blk in enumerate(mesh.blocks))
        bc = mesh.boundaries.get(name)
        out.append({'name': name,
                    'bc_type': bc.bc_type if bc is not None else None,
                    'n_faces': int(len(elem)),
                    'interior': int((count[rows] >= 2).sum()),
                    'triangles': int(tris)})
    return out


def face_geometry(mesh: Mesh, elem, face, max_triangles: int = MAX_TRIANGLES
                  ) -> dict:
    """Triangles and element edges of a set of faces.

    Parameters
    ----------
    mesh : Mesh
    elem, face : array_like
        Global cell index and local face of every face to draw.
    max_triangles : int
        Budget. Above it every ``stride``-th face is kept.

    Returns
    -------
    dict
        ``positions`` ``(nv, 3)`` float32 relative to :func:`origin`;
        ``indices`` ``(nt, 3)`` uint32; ``tri_cell`` ``(nt,)`` uint32 cell of
        each triangle (for picking); ``vert_index`` ``(nv,)`` uint32, the
        position of each vertex in a per-(cell, node) array
        (:meth:`.mesh.Mesh.cell_node_offsets` plus the local node), for
        node fields; ``edges`` ``(ns, 2, 3)`` float32 segment
        end points; ``stride`` (1 unless decimated) and ``n_faces`` drawn.
    """
    elem = np.asarray(elem, dtype=np.int64)
    face = np.asarray(face, dtype=np.int64)
    b_of, local = mesh.locate_cell(elem)
    total = sum(int((b_of == b).sum()) * _tris_per_face(blk)
                for b, blk in enumerate(mesh.blocks))
    stride = max(1, -(-total // max_triangles)) if max_triangles else 1
    if stride > 1:
        elem, face, b_of, local = (a[::stride] for a in (elem, face, b_of,
                                                          local))
    P = mesh.points - origin(mesh)
    node_off = mesh.cell_node_offsets()
    pos, idx, tcell, edges, vidx = [], [], [], [], []
    nv = 0
    for b, block in enumerate(mesh.blocks):
        for f in range(block.n_faces):
            sel = np.nonzero((b_of == b) & (face == f))[0]
            if len(sel) == 0:
                continue
            cells = elem[sel]
            nodes = block.face_nodes(f)[local[sel]]
            if block.kind == 'hex':
                n = block.order + 1
                m = len(sel)
                x = P[nodes]                                   # (m, n, n, 3)
                pos.append(x.reshape(-1, 3))
                vidx.append((node_off[cells][:, None]
                             + ordering.hex_face_grid(block.order, f).ravel()
                             [None]).ravel())
                a, c = np.meshgrid(np.arange(n - 1), np.arange(n - 1),
                                   indexing='ij')
                v00 = (a * n + c).ravel()
                v10, v11, v01 = v00 + n, v00 + n + 1, v00 + 1
                quad = np.stack([v00, v10, v11, v00, v11, v01], 1).reshape(-1, 3)
                base = nv + n * n * np.arange(m)[:, None, None]
                idx.append((quad[None] + base).reshape(-1, 3))
                tcell.append(np.repeat(cells, len(quad)))
                rims = [x[:, 0, :], x[:, -1, :], x[:, :, 0], x[:, :, -1]]
                for r in rims:                                 # (m, n, 3)
                    edges.append(np.stack([r[:, :-1], r[:, 1:]], 2)
                                 .reshape(-1, 2, 3))
                nv += m * n * n
            else:
                k = nodes.shape[1]
                x = P[nodes]                                   # (m, k, 3)
                pos.append(x.reshape(-1, 3))
                vidx.append((node_off[cells][:, None]
                             + np.array(ordering.LINEAR_FACES[block.kind][f])
                             [None]).ravel())
                local_tris = ([[0, 1, 2]] if k == 3
                              else [[0, 1, 2], [0, 2, 3]])
                base = nv + k * np.arange(len(sel))[:, None, None]
                idx.append((np.array(local_tris)[None] + base).reshape(-1, 3))
                tcell.append(np.repeat(cells, len(local_tris)))
                ring = np.roll(np.arange(k), -1)
                edges.append(np.stack([x, x[:, ring]], 2).reshape(-1, 2, 3))
                nv += len(sel) * k
    if not pos:
        return {'positions': np.zeros((0, 3), np.float32),
                'indices': np.zeros((0, 3), np.uint32),
                'tri_cell': np.zeros(0, np.uint32),
                'vert_index': np.zeros(0, np.uint32),
                'edges': np.zeros((0, 2, 3), np.float32),
                'stride': stride, 'n_faces': 0}
    return {'positions': np.vstack(pos).astype(np.float32),
            'indices': np.vstack(idx).astype(np.uint32),
            'tri_cell': np.concatenate(tcell).astype(np.uint32),
            'vert_index': np.concatenate(vidx).astype(np.uint32),
            'edges': np.vstack(edges).astype(np.float32),
            'stride': stride, 'n_faces': int(len(elem))}


def _corner_range(mesh: Mesh, axis: int) -> tuple:
    """Per-cell min, max and mean of the corner coordinates along *axis*."""
    key = f'corner_range_{axis}'
    if key not in mesh._cache:
        lo, hi, mid = [], [], []
        for block in mesh.blocks:
            c = mesh.points[block.corners(), axis]
            lo.append(c.min(1))
            hi.append(c.max(1))
            mid.append(c.mean(1))
        mesh._cache[key] = tuple(np.concatenate(v) if v else np.empty(0)
                                 for v in (lo, hi, mid))
    return mesh._cache[key]


def section_faces(mesh: Mesh, axis: int, value: float, mode: str = 'slice',
                  side: int = -1) -> tuple:
    """Faces that show the volume mesh around a plane ``x[axis] = value``.

    Parameters
    ----------
    axis : int
        0, 1 or 2.
    value : float
        Plane position, in mesh coordinates.
    mode : str
        ``'slice'``: the layer of cells the plane crosses (a *crinkle*
        slice), drawn through all its outer faces — one cell thick, so
        every cell and its neighbours along the plane are visible.
        ``'clip'``: the cells whose corner centroid lies on *side* of the
        plane; only the faces where the cut opened the mesh are returned
        (the boundaries draw the rest), so the volume shows as a staircase.
    side : int
        -1 keeps ``x < value``, +1 keeps ``x > value`` (``'clip'`` only).

    Returns
    -------
    elem, face : numpy.ndarray
    """
    lo, hi, mid = _corner_range(mesh, axis)
    if mode == 'slice':
        keep = (lo <= value) & (hi >= value)
    elif mode == 'clip':
        keep = mid < value if side < 0 else mid > value
    else:
        raise ValueError(f"mode must be 'slice' or 'clip', not {mode!r}")
    faces, idx = topology.all_faces(mesh), topology.face_index(mesh)
    rows = np.nonzero(keep[faces['elem']])[0]
    uid = idx['inverse'][rows]
    in_subset = np.bincount(uid, minlength=len(idx['count']))[uid]
    outer = in_subset == 1
    if mode == 'clip':
        outer &= idx['count'][uid] >= 2       # opened by the cut, not a BC
    rows = rows[outer]
    return faces['elem'][rows], faces['face'][rows]


def _hex_edges(order: int) -> list:
    """Canonical node positions along the 12 edges of a hexahedron."""
    n = order + 1
    line = np.arange(n)
    out = []
    for fixed in ((1, 2), (0, 2), (0, 1)):         # edges along i, j, k
        run = ({0, 1, 2} - set(fixed)).pop()
        for a in (0, order):
            for b in (0, order):
                ijk = np.zeros((n, 3), dtype=np.int64)
                ijk[:, run] = line
                ijk[:, fixed[0]] = a
                ijk[:, fixed[1]] = b
                out.append(ordering.tensor_index(ijk[:, 0], ijk[:, 1],
                                                 ijk[:, 2], order))
    return out


def cell_info(mesh: Mesh, cell: int) -> dict:
    """What the viewer shows when a cell is picked.

    Returns
    -------
    dict
        ``cell``, ``kind``, ``order``, ``block``, ``local``, ``file_cell``
        (index in the file, when the reader keeps it), ``zone``,
        ``centroid`` and ``corners`` (mesh coordinates), ``faces`` (per
        local face: ``neighbour`` cell or None and ``boundary`` name or
        None) and ``outline``: the element edges as segments relative to
        :func:`origin`, for highlighting.
    """
    cell = int(cell)
    if not 0 <= cell < mesh.n_cells:
        raise ValueError(f"cell {cell} out of range 0..{mesh.n_cells - 1}")
    b, local = mesh.locate_cell([cell])
    b, local = int(b[0]), int(local[0])
    block = mesh.blocks[b]
    nodes = block.conn[local]
    corners = mesh.points[block.corners()[local]]
    faces, idx = topology.all_faces(mesh), topology.face_index(mesh)
    rows = topology.face_rows(mesh, np.full(block.n_faces, cell),
                              np.arange(block.n_faces))
    uid = idx['inverse'][rows]
    # rows sharing each unique face: the other one is the neighbour
    if 'rows_by_face' not in mesh._cache:
        o = np.argsort(idx['inverse'], kind='stable')
        mesh._cache['rows_by_face'] = (o, idx['inverse'][o])
    order, sorted_uid = mesh._cache['rows_by_face']
    starts = np.searchsorted(sorted_uid, uid)
    bc_rows = _row_boundary(mesh)
    names = list(mesh.boundaries)
    info_faces = []
    for k, (r, u, s) in enumerate(zip(rows, uid, starts)):
        neighbour = None
        for j in order[s:s + idx['count'][u]]:
            if j != r:
                neighbour = int(faces['elem'][j])
        bc = int(bc_rows[r])
        info_faces.append({'face': k, 'neighbour': neighbour,
                           'boundary': names[bc] if bc >= 0 else None})
    P = mesh.points[nodes] - origin(mesh)
    if block.kind == 'hex':
        segs = [np.stack([P[e[:-1]], P[e[1:]]], 1)
                for e in _hex_edges(block.order)]
    else:
        pairs = {tuple(sorted((f[i], f[(i + 1) % len(f)])))
                 for f in ordering.LINEAR_FACES[block.kind]
                 for i in range(len(f))}
        segs = [np.stack([P[[a]], P[[c]]], 1) for a, c in sorted(pairs)]
    fidx = mesh.meta.get('file_cell_index')
    return {
        'cell': cell, 'kind': block.kind, 'order': block.order, 'block': b,
        'local': local,
        'file_cell': int(fidx[cell]) if fidx is not None else None,
        'zone': mesh.zones.get(int(block.zone[local]),
                               f'zone_{int(block.zone[local])}'),
        'centroid': corners.mean(0).tolist(),
        'corners': corners.tolist(),
        'faces': info_faces,
        'outline': np.vstack(segs).astype(np.float32).tolist(),
    }


def pack(header: dict, arrays: dict) -> bytes:
    """Binary payload: ``uint32`` header length, JSON header, then arrays.

    Every array starts on a 4-byte boundary, little-endian. The header gets
    an ``arrays`` entry ``{name: {dtype, offset, count, shape}}`` with the
    byte offset from the start of the payload, so the browser can wrap each
    one in a typed array without copying.
    """
    meta = dict(header)
    meta['arrays'] = {}
    blobs = []
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr)
        if a.dtype not in (np.float32, np.uint32, np.int32):
            raise TypeError(f"{name}: dtype {a.dtype} is not sent")
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        meta['arrays'][name] = {'dtype': a.dtype.name, 'count': int(a.size),
                                'shape': list(a.shape), 'offset': 0}
        blobs.append((name, a.tobytes()))

    def encode(m):
        raw = json.dumps(m, allow_nan=False).encode('utf-8')
        return raw + b' ' * (-len(raw) % 4)

    # offsets depend on the header length, which depends on the offsets:
    # iterate until stable (two passes in practice)
    for _ in range(5):
        head = encode(meta)
        pos = 4 + len(head)
        changed = False
        for name, blob in blobs:
            if meta['arrays'][name]['offset'] != pos:
                meta['arrays'][name]['offset'] = pos
                changed = True
            pos += len(blob)
        if not changed:
            break
    head = encode(meta)
    return (np.uint32(len(head)).astype('<u4').tobytes() + head
            + b''.join(blob for _, blob in blobs))
