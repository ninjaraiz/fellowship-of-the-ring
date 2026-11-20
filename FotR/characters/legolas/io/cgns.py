"""CGNS meshes, read with h5py (CGNS/HDF5 only; no CGNS library needed).

A CGNS/HDF5 file is an HDF5 tree where every CGNS node is a group with
attributes ``label`` and ``name`` and, if it carries data, a dataset called
``" data"`` (leading space). Strings are stored as int8 arrays.

Supported: unstructured zones, element sections of any fixed type or
``MIXED`` (with or without ``ElementStartOffset``), and boundaries given
by ``ZoneBC/BC_t`` (``PointList``/``PointRange`` on faces or on vertices).
Without ``ZoneBC``, each surface section becomes a boundary named after
it, which is what ANSA writes. Not supported: structured zones,
``NGON_n``/``NFACE_n`` polyhedra, CGNS/ADF files.

Hexahedra in tensor-product types (HEXA_8/27/64/125) are permuted to the
canonical layout; serendipity hexahedra (HEXA_20/32/56/44/98) and
high-order tetrahedra, pyramids and prisms keep their corners only, with a
warning.
"""

from __future__ import annotations

import numpy as np

from .. import topology
from ..mesh import Boundary, CellBlock, Mesh
from ..ordering import CGNS_HEX, linear_conn

# code -> (dimension, kind, nodes per element)
_TYPES = {
    2: (0, 'node', 1), 3: (1, 'bar', 2), 4: (1, 'bar', 3),
    24: (1, 'bar', 4), 40: (1, 'bar', 5),
    5: (2, 'tri', 3), 6: (2, 'tri', 6), 25: (2, 'tri', 9),
    26: (2, 'tri', 10), 41: (2, 'tri', 12), 42: (2, 'tri', 15),
    7: (2, 'quad', 4), 8: (2, 'quad', 8), 9: (2, 'quad', 9),
    27: (2, 'quad', 12), 28: (2, 'quad', 16), 43: (2, 'quad', 16),
    44: (2, 'quad', 25),
    10: (3, 'tet', 4), 11: (3, 'tet', 10), 29: (3, 'tet', 16),
    30: (3, 'tet', 20), 45: (3, 'tet', 22), 46: (3, 'tet', 34),
    47: (3, 'tet', 35),
    12: (3, 'pyramid', 5), 13: (3, 'pyramid', 14), 21: (3, 'pyramid', 13),
    31: (3, 'pyramid', 21), 32: (3, 'pyramid', 29), 33: (3, 'pyramid', 30),
    48: (3, 'pyramid', 29), 49: (3, 'pyramid', 50), 50: (3, 'pyramid', 55),
    14: (3, 'prism', 6), 15: (3, 'prism', 15), 16: (3, 'prism', 18),
    34: (3, 'prism', 24), 35: (3, 'prism', 38), 36: (3, 'prism', 40),
    51: (3, 'prism', 33), 52: (3, 'prism', 66), 53: (3, 'prism', 75),
    17: (3, 'hex', 8), 18: (3, 'hex', 20), 19: (3, 'hex', 27),
    37: (3, 'hex', 32), 38: (3, 'hex', 56), 39: (3, 'hex', 64),
    54: (3, 'hex', 44), 55: (3, 'hex', 98), 56: (3, 'hex', 125),
}
_CORNERS = {'tri': 3, 'quad': 4, 'tet': 4, 'pyramid': 5, 'prism': 6,
            'hex': 8, 'bar': 2, 'node': 1}
_MIXED, _NGON, _NFACE = 20, 22, 23


def _label(g) -> str:
    v = g.attrs.get('label', b'')
    return v.decode() if isinstance(v, bytes) else str(v)


def _name(g, key) -> str:
    v = g.attrs.get('name', None)
    if v is None:
        return key
    return v.decode().strip() if isinstance(v, bytes) else str(v).strip()


def _string(g) -> str:
    if ' data' not in g:
        return ''
    return g[' data'][()].tobytes().decode(errors='replace').strip('\x00 ')


def _children(g, label):
    for key in g:
        if key.startswith(' '):
            continue
        child = g[key]
        if hasattr(child, 'attrs') and _label(child) == label:
            yield key, child


def _section(sec) -> dict:
    """Elements of one ``Elements_t``: {code: (ids, conn)} with 1-based ids."""
    etype = int(sec[' data'][()][0])
    first, last = (int(v) for v in np.ravel(sec['ElementRange'][' data'][()]))
    conn = np.ravel(sec['ElementConnectivity'][' data'][()]).astype(np.int64)
    n = last - first + 1
    ids = np.arange(first, last + 1, dtype=np.int64)
    if etype in (_NGON, _NFACE):
        raise NotImplementedError("NGON_n/NFACE_n polyhedral sections are not "
                                  "supported")
    if etype != _MIXED:
        return {etype: (ids, conn.reshape(n, -1))}
    # MIXED: each element is [type, nodes...]
    if 'ElementStartOffset' in sec:
        offs = np.ravel(sec['ElementStartOffset'][' data'][()]).astype(np.int64)
        starts = offs[:-1]
    else:
        t0 = int(conn[0])
        npe = _TYPES[t0][2]
        if conn.size == n * (npe + 1) and (conn[::npe + 1] == t0).all():
            return {t0: (ids, conn.reshape(n, npe + 1)[:, 1:])}
        starts = np.empty(n, dtype=np.int64)
        pos = 0
        for i in range(n):
            starts[i] = pos
            pos += _TYPES[int(conn[pos])][2] + 1
    codes = conn[starts]
    out = {}
    for code in np.unique(codes):
        rows = np.nonzero(codes == code)[0]
        npe = _TYPES[int(code)][2]
        out[int(code)] = (ids[rows],
                          conn[starts[rows][:, None] + 1 + np.arange(npe)])
    return out


def read(path: str) -> Mesh:
    """Read every unstructured zone of a CGNS/HDF5 file.

    Parameters
    ----------
    path : str
        ``.cgns`` file in HDF5 flavour.

    Returns
    -------
    Mesh
        Zones of the file become volume zones; with several zones their
        nodes are not merged (a warning says so), so the interfaces between
        zones show up as exterior faces.
    """
    import h5py

    pts, vol, surf, bcs, zones = [], {}, [], [], {}
    warnings = []
    offset = 0
    with h5py.File(path, 'r') as fh:
        bases = list(_children(fh, 'CGNSBase_t'))
        if not bases:
            raise ValueError(f"{path}: no CGNSBase_t node")
        zid = 0
        for _, base in bases:
            for zkey, zone in _children(base, 'Zone_t'):
                zname = _name(zone, zkey)
                ztype = [_string(g) for _, g in _children(zone, 'ZoneType_t')]
                if ztype and ztype[0] != 'Unstructured':
                    raise NotImplementedError(
                        f"{path}: zone {zname} is {ztype[0]}; only "
                        "unstructured zones are supported")
                gc = zone['GridCoordinates']
                xyz = np.stack([np.ravel(gc[c][' data'][()]).astype(np.float64)
                                for c in ('CoordinateX', 'CoordinateY',
                                          'CoordinateZ')], axis=1)
                pts.append(xyz)
                zones[zid] = zname
                # elements: global (zone, id) -> corners for surfaces
                zsurf = {}
                for skey, sec in _children(zone, 'Elements_t'):
                    sname = _name(sec, skey)
                    for code, (ids, conn) in _section(sec).items():
                        dim, kind, _ = _TYPES[code]
                        conn = conn - 1 + offset
                        if dim == 3:
                            vol.setdefault(code, []).append(
                                (conn, np.full(len(conn), zid)))
                        elif dim == 2:
                            corners = conn[:, :_CORNERS[kind]]
                            zsurf.setdefault(sname, []).append((ids, corners))
                # boundaries
                zbc = list(_children(zone, 'ZoneBC_t'))
                id_map_ids = []
                id_map_corners = []
                for parts in zsurf.values():
                    for ids, corners in parts:
                        id_map_ids.append(ids)
                        id_map_corners.append(corners)
                all_ids = (np.concatenate(id_map_ids) if id_map_ids
                           else np.empty(0, np.int64))
                if zbc:
                    for bkey, bc in _children(zbc[0][1], 'BC_t'):
                        bname = _name(bc, bkey)
                        loc = [_string(g) for _, g in
                               _children(bc, 'GridLocation_t')]
                        loc = loc[0] if loc else 'Vertex'
                        pl = None
                        for key in ('PointList', 'ElementList'):
                            if key in bc:
                                pl = np.ravel(bc[key][' data'][()])
                        for key in ('PointRange', 'ElementRange'):
                            if pl is None and key in bc:
                                a, b = np.ravel(bc[key][' data'][()])[:2]
                                pl = np.arange(a, b + 1)
                        if pl is None:
                            warnings.append(f"BC {bname}: no point list, "
                                            "skipped")
                            continue
                        pl = pl.astype(np.int64)
                        if loc in ('Vertex',):
                            bcs.append((bname, _string(bc), 'nodes',
                                        pl - 1 + offset))
                        else:
                            bcs.append((bname, _string(bc), 'elements',
                                        (zid, pl)))
                else:
                    for sname, parts in zsurf.items():
                        corners = [c for _, c in parts]
                        bcs.append((sname, None, 'corners',
                                    _stack_corners(corners)))
                surf.append((zid, all_ids, id_map_corners))
                offset += len(xyz)
                zid += 1

    mesh = Mesh(np.vstack(pts), zones=zones, meta={'format': 'cgns'})
    for w in warnings:
        mesh.warn(w)
    if len(zones) > 1:
        mesh.warn(f"{len(zones)} zones: nodes on zone interfaces are not "
                  "merged, so interfaces count as exterior faces")
    for code in sorted(vol):
        conn = np.vstack([c for c, _ in vol[code]])
        zone = np.concatenate([z for _, z in vol[code]])
        _, kind, npe = _TYPES[code]
        if kind == 'hex' and code in CGNS_HEX:
            order = round(npe ** (1 / 3)) - 1
            block = CellBlock('hex', order, conn[:, CGNS_HEX[code]], zone)
        else:
            if npe != _CORNERS[kind]:
                mesh.warn(f"CGNS type {code} ({kind}, {npe} nodes): only the "
                          "corners are kept")
            block = CellBlock(kind, 1,
                              linear_conn(kind, conn[:, :_CORNERS[kind]]), zone)
        _merge_block(mesh, block)
    mesh.meta['ngeo'] = mesh.ngeo

    # resolve boundaries
    surf_by_zone = {}
    for zid, ids, corners in surf:
        if len(ids):
            order = np.argsort(ids)
            surf_by_zone[zid] = (ids[order], _stack_corners(corners)[order])
    pending = []
    for bname, btype, how, data in bcs:
        if how == 'elements':
            zid, pl = data
            ids, corners = surf_by_zone.get(zid, (np.empty(0, np.int64),
                                                  np.empty((0, 4), np.int64)))
            pos = np.searchsorted(ids, pl)
            ok = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == pl)
            if not ok.all():
                mesh.warn(f"BC {bname}: {(~ok).sum()} entries are not "
                          "surface elements (volume element ids?); ignored")
            pending.append((bname, btype, corners[pos[ok]]))
        elif how == 'corners':
            pending.append((bname, btype, data))
        else:   # vertex list: exterior faces whose corners are all listed
            inside = np.zeros(mesh.n_points, dtype=bool)
            inside[data] = True
            elem, face = _face_corner_check(
                mesh, *topology.exterior_faces(mesh), inside)
            mesh.boundaries[bname] = Boundary(bname, elem, face,
                                              bc_type=btype or None)
    _add_boundaries(mesh, pending)
    return mesh


def _stack_corners(parts) -> np.ndarray:
    """Stack corner arrays of triangles and quads into ``(m, 4)``, -1 pad."""
    rows = []
    for c in parts:
        if c.shape[1] == 3:
            c = np.hstack([c, np.full((len(c), 1), -1, np.int64)])
        rows.append(c)
    return np.vstack(rows) if rows else np.empty((0, 4), np.int64)


def _add_boundaries(mesh, items) -> None:
    """Turn surface elements into boundaries, matching all of them at once.

    Parameters
    ----------
    mesh : Mesh
    items : list of (name, bc_type, corners)
        ``corners`` is ``(m, 4)``, triangles padded with ``-1``. Entries with
        the same name are joined.
    """
    if not items:
        return
    corners = np.vstack([np.asarray(c, dtype=np.int64).reshape(-1, 4)
                         for _, _, c in items])
    elem, face, _ = topology.match_faces(mesh, corners)
    start = 0
    for name, btype, c in items:
        n = len(c)
        e, f = elem[start:start + n], face[start:start + n]
        start += n
        found = e >= 0
        if not found.all():
            mesh.warn(f"BC {name}: {(~found).sum()} faces match no element "
                      "face")
        e, f = e[found], f[found]
        if name in mesh.boundaries:
            old = mesh.boundaries[name]
            e, f = np.concatenate([old.elem, e]), np.concatenate([old.face, f])
        mesh.boundaries[name] = Boundary(name, e, f, bc_type=btype or None)


def _face_corner_check(mesh, elem, face, inside):
    b_of, local = mesh.locate_cell(elem)
    keep = np.zeros(len(elem), bool)
    for b, block in enumerate(mesh.blocks):
        for f in range(block.n_faces):
            sel = np.nonzero((b_of == b) & (face == f))[0]
            if len(sel):
                keep[sel] = inside[block.face_corners(f)[local[sel]]].all(1)
    return elem[keep], face[keep]


def _merge_block(mesh, block) -> None:
    """Append *block*, joining it to an existing one of the same kind/order."""
    for i, b in enumerate(mesh.blocks):
        if b.kind == block.kind and b.order == block.order:
            mesh.blocks[i] = CellBlock(b.kind, b.order,
                                       np.vstack([b.conn, block.conn]),
                                       np.concatenate([b.zone, block.zone]))
            mesh._cache.clear()
            return
    mesh.blocks.append(block)
    mesh._cache.clear()
