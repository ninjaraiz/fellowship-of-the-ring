"""Faces, exterior faces and face matching, by corner nodes.

Two faces are the same face when they share the same corner nodes; the key
of a face is its sorted corners (a triangle is padded with ``-1``). That
works for every order, because high-order nodes never decide identity.
Results are cached on the mesh (``mesh._cache``).
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh

__all__ = ['face_keys', 'group_keys', 'all_faces', 'face_index', 'face_rows', 'exterior_faces',
           'match_faces', 'face_area', 'used_nodes']


def face_keys(corners: np.ndarray) -> np.ndarray:
    """``(m, 4)`` sorted keys of faces given by 3 or 4 corner nodes."""
    corners = np.asarray(corners, dtype=np.int64)
    if corners.shape[1] == 3:
        corners = np.hstack([np.full((len(corners), 1), -1, np.int64),
                             corners])
    return np.sort(corners, axis=1)


def group_keys(keys: np.ndarray) -> tuple:
    """Group identical face keys.

    Each ``(m, 4)`` key is packed into two uint64 words (node ids up to
    2**32 - 2), which sorts about four times faster than
    ``np.unique(axis=0)``.

    Returns
    -------
    inverse, count : numpy.ndarray
        Group of each row (groups numbered in sorted key order) and the
        size of each group.
    """
    if len(keys) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    k = (np.asarray(keys, dtype=np.int64) + 1).astype(np.uint64)
    hi = (k[:, 0] << np.uint64(32)) | k[:, 1]
    lo = (k[:, 2] << np.uint64(32)) | k[:, 3]
    order = np.lexsort((lo, hi))
    hs, ls = hi[order], lo[order]
    new = np.empty(len(k), dtype=bool)
    new[0] = True
    new[1:] = (hs[1:] != hs[:-1]) | (ls[1:] != ls[:-1])
    group = np.cumsum(new) - 1
    inverse = np.empty(len(k), dtype=np.int64)
    inverse[order] = group
    return inverse, np.bincount(group)


def all_faces(mesh: Mesh) -> dict:
    """Every face of every cell, interior faces twice.

    Returns
    -------
    dict
        ``elem`` (global cell), ``face`` (local face) and ``key`` (see
        :func:`face_keys`), one row per (cell, face).
    """
    if 'faces' in mesh._cache:
        return mesh._cache['faces']
    elem, face, keys = [], [], []
    off = mesh.offsets
    for b, block in enumerate(mesh.blocks):
        ids = np.arange(len(block), dtype=np.int64) + off[b]
        for f in range(block.n_faces):
            elem.append(ids)
            face.append(np.full(len(block), f, dtype=np.int64))
            keys.append(face_keys(block.face_corners(f)))
    if elem:
        out = {'elem': np.concatenate(elem), 'face': np.concatenate(face),
               'key': np.concatenate(keys)}
    else:
        out = {'elem': np.empty(0, np.int64), 'face': np.empty(0, np.int64),
               'key': np.empty((0, 4), np.int64)}
    mesh._cache['faces'] = out
    return out


def face_index(mesh: Mesh) -> dict:
    """Unique faces of the mesh and how many cells share each.

    Returns
    -------
    dict
        ``inverse`` (row of :func:`all_faces` → unique face) and ``count``
        (cells per unique face: 1 exterior, 2 interior, more non-manifold).
    """
    if 'face_index' in mesh._cache:
        return mesh._cache['face_index']
    inverse, count = group_keys(all_faces(mesh)['key'])
    out = {'inverse': inverse, 'count': count}
    mesh._cache['face_index'] = out
    return out


def face_rows(mesh: Mesh, elem, face) -> np.ndarray:
    """Row of :func:`all_faces` that holds each (elem, face) pair."""
    elem = np.asarray(elem, dtype=np.int64)
    face = np.asarray(face, dtype=np.int64)
    b, local = mesh.locate_cell(elem)
    sizes = np.array([len(blk) for blk in mesh.blocks], dtype=np.int64)
    nfaces = np.array([blk.n_faces for blk in mesh.blocks], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(sizes * nfaces)])[:-1]
    return start[b] + face * sizes[b] + local


def exterior_faces(mesh: Mesh) -> tuple:
    """``(elem, face)`` of the faces that belong to a single cell."""
    faces, idx = all_faces(mesh), face_index(mesh)
    mask = idx['count'][idx['inverse']] == 1
    return faces['elem'][mask], faces['face'][mask]


def match_faces(mesh: Mesh, corners: np.ndarray) -> tuple:
    """Find the cell faces that have the given corners.

    Parameters
    ----------
    mesh : Mesh
    corners : numpy.ndarray
        ``(m, 3)`` or ``(m, 4)`` corner nodes of surface elements; in the
        4-column form a triangle carries ``-1`` in one column.

    Returns
    -------
    elem, face, shared : numpy.ndarray
        For each surface element, the first cell face found (``-1`` when
        none matches) and how many cells share that face (2 means the
        surface lies inside the domain).
    """
    faces, idx = all_faces(mesh), face_index(mesh)
    if len(corners) == 0:
        e = np.empty(0, np.int64)
        return e, e.copy(), e.copy()
    q = face_keys(corners)
    nv = len(faces['key'])
    inv, _ = group_keys(np.vstack([faces['key'], q]))
    first = np.full(inv.max() + 1, -1, dtype=np.int64)
    first[inv[:nv][::-1]] = np.arange(nv, dtype=np.int64)[::-1]
    row = first[inv[nv:]]
    ok = row >= 0
    elem = np.where(ok, faces['elem'][row], -1)
    face = np.where(ok, faces['face'][row], -1)
    shared = np.where(ok, idx['count'][idx['inverse'][row]], 0)
    return elem, face, shared


def _tri_area(a, b, c) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)


def face_area(mesh: Mesh, elem: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Area of each (elem, face), following the high-order nodes.

    High-order faces are split along their node grid into ``N**2``
    quadrilaterals and each into two triangles: exact for planar faces,
    a close approximation for curved ones (it converges with ``N``).
    """
    elem = np.asarray(elem, dtype=np.int64)
    face = np.asarray(face, dtype=np.int64)
    area = np.zeros(len(elem))
    if len(elem) == 0:
        return area
    b_of, local = mesh.locate_cell(elem)
    P = mesh.points
    for b, block in enumerate(mesh.blocks):
        for f in range(block.n_faces):
            sel = np.nonzero((b_of == b) & (face == f))[0]
            if len(sel) == 0:
                continue
            nodes = block.face_nodes(f)[local[sel]]
            if block.kind == 'hex':
                x = P[nodes]                       # (m, n, n, 3)
                p00, p10 = x[:, :-1, :-1], x[:, 1:, :-1]
                p11, p01 = x[:, 1:, 1:], x[:, :-1, 1:]
                a = _tri_area(p00, p10, p11) + _tri_area(p00, p11, p01)
                area[sel] = a.sum(axis=(1, 2))
            else:
                x = P[nodes]
                a = _tri_area(x[:, 0], x[:, 1], x[:, 2])
                if nodes.shape[1] == 4:
                    a = a + _tri_area(x[:, 0], x[:, 2], x[:, 3])
                area[sel] = a
    return area


def used_nodes(mesh: Mesh) -> np.ndarray:
    """Boolean mask of the nodes referenced by at least one cell."""
    used = np.zeros(mesh.n_points, dtype=bool)
    for block in mesh.blocks:
        used[block.conn.ravel()] = True
    return used
