"""Gmsh ``.msh`` meshes: v2.2 with an own parser, v4 through meshio.

Why an own parser for v2.2: ANSA writes binary v2.2 files in **big-endian**
byte order. The format allows it (the integer ``1`` after the header tells
the byte order, and gmsh honours it), but meshio assumes little-endian and
fails with a bare "Couldn't read file". The v2.2 parser below reads ASCII
and binary files in either byte order; v4 files go to meshio.

Physical groups give the names: dimension-3 groups become volume zones and
dimension-2 groups become boundaries (a physical surface declared without
elements becomes an empty boundary). Surfaces without a physical group are
grouped by their geometrical entity (``entity_<tag>``).

Node orders: the v2.2 parser keeps gmsh's native order
(:data:`..ordering.GMSH_HEX`); meshio hands back ``hexahedron27`` in VTK
order but ``hexahedron64``/``hexahedron125`` in gmsh's
(:data:`..ordering.MESHIO_HEX`). Serendipity hexahedra and high-order
tetrahedra, pyramids and prisms keep their corners only, with a warning.

Files exported by ANSA at order 4 have been seen with a node ordering of
their own (neither gmsh nor CGNS); the tensor-order screen in
:func:`..io.read` then warns.
"""

from __future__ import annotations

import numpy as np

from ..mesh import CellBlock, Mesh
from ..ordering import GMSH_HEX, MESHIO_HEX, linear_conn
from .cgns import _add_boundaries, _merge_block, _stack_corners

#: gmsh element type -> (kind, dimension, nodes, hex order if tensor)
GMSH_TYPES = {
    1: ('line', 1, 2, None), 8: ('line', 1, 3, None),
    26: ('line', 1, 4, None), 27: ('line', 1, 5, None),
    15: ('point', 0, 1, None),
    2: ('tri', 2, 3, None), 9: ('tri', 2, 6, None), 20: ('tri', 2, 9, None),
    21: ('tri', 2, 10, None), 22: ('tri', 2, 12, None),
    23: ('tri', 2, 15, None),
    3: ('quad', 2, 4, None), 16: ('quad', 2, 8, None),
    10: ('quad', 2, 9, None), 36: ('quad', 2, 16, None),
    37: ('quad', 2, 25, None),
    4: ('tet', 3, 4, None), 11: ('tet', 3, 10, None), 29: ('tet', 3, 20, None),
    30: ('tet', 3, 35, None),
    7: ('pyramid', 3, 5, None), 14: ('pyramid', 3, 14, None),
    19: ('pyramid', 3, 13, None),
    6: ('prism', 3, 6, None), 13: ('prism', 3, 18, None),
    18: ('prism', 3, 15, None),
    5: ('hex', 3, 8, 1), 12: ('hex', 3, 27, 2), 17: ('hex', 3, 20, None),
    92: ('hex', 3, 64, 3), 93: ('hex', 3, 125, 4),
}
_CORNERS = {'hex': 8, 'tet': 4, 'pyramid': 5, 'prism': 6, 'quad': 4,
            'tri': 3}


# --- v2.2 parser ------------------------------------------------------------

def _sections(raw: bytes) -> dict:
    """Byte range of the body of every ``$Section``: {name: (start, end)}."""
    out = {}
    pos = 0
    while True:
        i = raw.find(b'$', pos)
        if i < 0:
            break
        j = raw.find(b'\n', i)
        tag = raw[i + 1:j].strip().decode(errors='replace')
        end = raw.find(b'\n$End' + tag.encode(), j - 1)
        if end < 0:
            raise ValueError(f"section ${tag} is not closed")
        out[tag] = (j + 1, end + 1)
        pos = raw.find(b'\n', end + 1) + 1
        if pos == 0:
            break
    return out


def read_msh22(path: str) -> dict:
    """Parse a v2.2 ``.msh`` file (ASCII or binary, either byte order).

    Parameters
    ----------
    path : str
        ``.msh`` file whose ``$MeshFormat`` says 2.x.

    Returns
    -------
    dict
        ``points`` ``(n, 3)``; ``elements``: list of ``(gmsh type, conn,
        physical, elementary)`` with ``conn`` 0-based rows of ``points`` in
        gmsh's native node order; ``names`` ``{(dim, tag): name}``;
        ``binary`` and ``byteorder`` (``'<'`` or ``'>'``).
    """
    with open(path, 'rb') as fh:
        raw = fh.read()
    sec = _sections(raw)
    a, b = sec['MeshFormat']
    head = raw[a:b].split(b'\n', 1)
    version, ftype, _ = head[0].split()
    if not version.startswith(b'2'):
        raise ValueError(f"{path}: msh version {version.decode()}, not 2.x")
    binary = ftype == b'1'
    order = '<'
    if binary:
        one = head[1][:4]
        if np.frombuffer(one, '<i4')[0] == 1:
            order = '<'
        elif np.frombuffer(one, '>i4')[0] == 1:
            order = '>'
        else:
            raise ValueError(f"{path}: cannot tell the byte order")

    names = {}
    if 'PhysicalNames' in sec:
        a, b = sec['PhysicalNames']
        lines = raw[a:b].decode(errors='replace').strip().splitlines()
        for line in lines[1:]:
            dim, tag, name = line.split(maxsplit=2)
            names[(int(dim), int(tag))] = name.strip().strip('"')

    # nodes
    a, b = sec['Nodes']
    nl = raw.index(b'\n', a)
    n_nodes = int(raw[a:nl])
    if binary:
        rec = np.dtype([('id', order + 'i4'), ('x', order + 'f8', 3)])
        nodes = np.frombuffer(raw, rec, n_nodes, nl + 1)
        ids = nodes['id'].astype(np.int64)
        xyz = nodes['x'].astype(np.float64)
    else:
        arr = np.array(raw[nl + 1:b].split(), dtype=np.float64).reshape(
            n_nodes, 4)
        ids, xyz = arr[:, 0].astype(np.int64), arr[:, 1:]
    lookup = np.full(ids.max() + 1, -1, dtype=np.int64)
    lookup[ids] = np.arange(n_nodes)

    # elements
    a, b = sec['Elements']
    nl = raw.index(b'\n', a)
    n_elem = int(raw[a:nl])
    groups = {}
    if binary:
        pos, done = nl + 1, 0
        i4 = np.dtype(order + 'i4')
        while done < n_elem:
            etype, count, ntags = (int(v) for v in
                                   np.frombuffer(raw, i4, 3, pos))
            pos += 12
            width = 1 + ntags + _npe(path, etype)
            block = np.frombuffer(raw, i4, count * width, pos).reshape(
                count, width).astype(np.int64)
            pos += 4 * count * width
            done += count
            tags = block[:, 1:1 + ntags]
            groups.setdefault(etype, []).append(
                (block[:, 1 + ntags:], _tag(tags, 0), _tag(tags, 1)))
    else:
        by_type = {}
        for line in raw[nl + 1:b].decode().splitlines():
            v = line.split()
            if v:
                by_type.setdefault(int(v[1]), []).append(v)
        for etype, vals in by_type.items():
            npe = _npe(path, etype)
            ntags = int(vals[0][2])
            arr = np.array(vals, dtype=np.int64)
            if arr.ndim != 2 or arr.shape[1] != 3 + ntags + npe:
                raise ValueError(f"{path}: elements of type {etype} with a "
                                 "varying number of tags")
            tags = arr[:, 3:3 + ntags]
            groups.setdefault(etype, []).append(
                (arr[:, 3 + ntags:], _tag(tags, 0), _tag(tags, 1)))

    elements = []
    for etype, parts in groups.items():
        conn = lookup[np.vstack([p[0] for p in parts])]
        if (conn < 0).any():
            raise ValueError(f"{path}: elements reference undefined nodes")
        elements.append((etype, conn, np.concatenate([p[1] for p in parts]),
                         np.concatenate([p[2] for p in parts])))
    return {'points': xyz, 'elements': elements, 'names': names,
            'binary': binary, 'byteorder': order}


def _npe(path, etype) -> int:
    if etype not in GMSH_TYPES:
        raise NotImplementedError(f"{path}: gmsh element type {etype} is "
                                  "not supported")
    return GMSH_TYPES[etype][2]


def _tag(tags, i) -> np.ndarray:
    if tags.shape[1] > i:
        return tags[:, i]
    return np.zeros(len(tags), dtype=np.int64)


# --- v4 through meshio -------------------------------------------------------

_MESHIO_KIND = (('hexahedron', 'hex', 3), ('tetra', 'tet', 3),
                ('pyramid', 'pyramid', 3), ('wedge', 'prism', 3),
                ('quad', 'quad', 2), ('triangle', 'tri', 2))


def _read_meshio(path) -> dict:
    """Cells of a v4 file as ``(kind, dim, order, conn, phys, geom, label)``.

    ``order`` is set (and ``conn`` canonical) for tensor hexahedra only.
    """
    import warnings
    import meshio

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = meshio.read(path, file_format='gmsh')
    names = {}
    for name, v in (m.field_data or {}).items():
        tag, dim = (int(x) for x in np.ravel(v)[:2])
        names[(dim, tag)] = name
    phys = m.cell_data.get('gmsh:physical')
    geom = m.cell_data.get('gmsh:geometrical')
    cells = []
    for i, block in enumerate(m.cells):
        kind = dim = None
        for prefix, k, d in _MESHIO_KIND:
            if block.type.startswith(prefix):
                kind, dim = k, d
        if kind is None:
            continue
        conn = np.asarray(block.data, dtype=np.int64)
        p = (np.asarray(phys[i], np.int64) if phys is not None
             else np.zeros(len(conn), np.int64))
        g = (np.asarray(geom[i], np.int64) if geom is not None
             else np.zeros(len(conn), np.int64))
        if kind == 'hex' and block.type in MESHIO_HEX:
            order = round(conn.shape[1] ** (1 / 3)) - 1
            cells.append((kind, dim, order, conn[:, MESHIO_HEX[block.type]],
                          p, g, block.type))
        else:
            cells.append((kind, dim, None, conn, p, g, block.type))
    return {'points': np.asarray(m.points, np.float64), 'cells': cells,
            'names': names}


# --- common -----------------------------------------------------------------

def read(path: str) -> Mesh:
    """Read a gmsh mesh.

    Parameters
    ----------
    path : str
        ``.msh`` file, version 2.2 (own parser) or 4.x (meshio).

    Returns
    -------
    Mesh
        ``meta['msh_version']`` and, for binary v2.2, ``meta['byteorder']``.
    """
    with open(path, 'rb') as fh:
        fh.readline()
        version = fh.readline().split()[0].decode()
    meta = {'format': 'gmsh', 'msh_version': version}
    if version.startswith('2'):
        data = read_msh22(path)
        if data['binary']:
            meta['byteorder'] = ('big-endian' if data['byteorder'] == '>'
                                 else 'little-endian')
        cells = []
        for etype, conn, p, g in data['elements']:
            kind, dim, _, order = GMSH_TYPES[etype]
            if order is not None:
                conn = conn[:, GMSH_HEX[order]]
            cells.append((kind, dim, order, conn, p, g, f'gmsh type {etype}'))
    else:
        data = _read_meshio(path)
        cells = data['cells']
    names = data['names']
    points = data['points']
    if points.shape[1] < 3:
        points = np.hstack([points, np.zeros((len(points),
                                              3 - points.shape[1]))])
    mesh = Mesh(points[:, :3], meta=meta)

    surfaces = {}
    for kind, dim, order, conn, ptag, gtag, label in cells:
        if dim == 3:
            for t in np.unique(ptag):
                mesh.zones[int(t)] = names.get((3, int(t)), f'volume_{t}')
            if order is not None:
                cb = CellBlock('hex', order, conn, ptag)
            else:
                if conn.shape[1] != _CORNERS[kind]:
                    mesh.warn(f"{label} ({kind}, {conn.shape[1]} nodes): only "
                              "the corners are kept")
                cb = CellBlock(kind, 1,
                               linear_conn(kind, conn[:, :_CORNERS[kind]]),
                               ptag)
            _merge_block(mesh, cb)
        elif dim == 2:
            corners = conn[:, :_CORNERS[kind]]
            for t in np.unique(ptag[ptag > 0]):
                name = names.get((2, int(t)), f'surface_{t}')
                surfaces.setdefault(name, []).append(corners[ptag == t])
            for t in np.unique(gtag[ptag <= 0]):
                sel = (ptag <= 0) & (gtag == t)
                surfaces.setdefault(f'entity_{t}', []).append(corners[sel])
    mesh.meta['ngeo'] = mesh.ngeo
    for (dim, _), name in names.items():
        if dim == 2:
            surfaces.setdefault(name, [])
    _add_boundaries(mesh, [(name, None, _stack_corners(parts))
                           for name, parts in surfaces.items()])
    return mesh
