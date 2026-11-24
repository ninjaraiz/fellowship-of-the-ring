"""``LEGOLAS.Convert``: fix the node order, and write VTK and gmsh files.

* :meth:`Convert.detect_order` / :meth:`Convert.fix_order` -- find and undo
  a node permutation the reader could not know about (ANSA's order-4
  ``.msh``), voting with the geometry (:func:`.ordering.detect_hex_order`).
* :meth:`Convert.to_vtu` -- the volume as a VTK XML unstructured grid:
  Lagrange cells (VTK 72, ParaView shows them curved) or each element cut
  into ``N**3`` linear hexahedra; per-cell fields go as cell data, per-node
  fields as point data.
* :meth:`Convert.to_vtm` -- a multiblock with the volume and one block per
  boundary, which is how ParaView lists the BCs by name.
* :meth:`Convert.write_msh` -- gmsh 2.2 (binary), in gmsh's own node order,
  with the zones and boundaries as physical groups: what PyHOPE reads.

Writers use numpy only (VTK is not needed to write VTK files). Node orders
come from :mod:`.ordering`, measured on VTK 9.5 and gmsh 4.13.
"""

from __future__ import annotations

import os

import numpy as np

from . import ordering
from .mesh import CellBlock

__all__ = ['Convert', 'write_vtu', 'VTK_TYPES']

#: VTK cell type of each kind: (linear, Lagrange).
VTK_TYPES = {'hex': (12, 72), 'tet': (10, None), 'pyramid': (14, None),
             'prism': (13, None), 'quad': (9, 70), 'tri': (5, None)}

_GMSH_HEX_TYPE = {1: 5, 2: 12, 3: 92, 4: 93}
_GMSH_QUAD_TYPE = {1: 3, 2: 10, 3: 36, 4: 37}
_GMSH_LINEAR = {'tet': 4, 'pyramid': 7, 'prism': 6, 'tri': 2}


# -- VTK XML -----------------------------------------------------------------

def _vtk_type(a: np.ndarray) -> str:
    return {np.dtype('float64'): 'Float64', np.dtype('float32'): 'Float32',
            np.dtype('int64'): 'Int64', np.dtype('int32'): 'Int32',
            np.dtype('uint8'): 'UInt8'}[a.dtype]


def write_vtu(path, points, conn_list, types, cell_data=None,
              point_data=None) -> None:
    """Write a VTK XML unstructured grid with raw appended binary data.

    Parameters
    ----------
    path : str or path-like
        ``.vtu`` file.
    points : numpy.ndarray
        ``(n_points, 3)``.
    conn_list : list[numpy.ndarray]
        One ``(n_cells_k, n_nodes_k)`` array per group of cells of the same
        size; rows are VTK-ordered node indices.
    types : list[int]
        VTK cell type of each group.
    cell_data, point_data : dict, optional
        ``{name: array}``; arrays may be ``(n,)`` or ``(n, components)``.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    conn = np.concatenate([c.ravel() for c in conn_list]).astype(np.int64) \
        if conn_list else np.zeros(0, np.int64)
    sizes = np.concatenate([np.full(len(c), c.shape[1], np.int64)
                            for c in conn_list]) if conn_list \
        else np.zeros(0, np.int64)
    offsets = np.cumsum(sizes).astype(np.int64)
    ctypes = np.concatenate([np.full(len(c), t, np.uint8)
                             for c, t in zip(conn_list, types)]) \
        if conn_list else np.zeros(0, np.uint8)
    n_cells = len(ctypes)
    blobs = []
    offset = 0

    def array(name, a, comps=None):
        nonlocal offset
        a = np.ascontiguousarray(a)
        if a.dtype == np.float32 or a.dtype == np.float64:
            pass
        elif a.dtype.kind == 'f':
            a = a.astype(np.float64)
        elif a.dtype.kind in 'iub' and a.dtype != np.uint8:
            a = a.astype(np.int64)
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        if comps is None:
            comps = 1 if a.ndim == 1 else a.shape[1]
        raw = a.tobytes()
        tag = (f'<DataArray type="{_vtk_type(a.astype(a.dtype.newbyteorder("=")))}"'
               + (f' Name="{name}"' if name else '')
               + f' NumberOfComponents="{comps}" format="appended"'
               f' offset="{offset}"/>')
        blobs.append(np.uint64(len(raw)).astype('<u8').tobytes() + raw)
        offset += 8 + len(raw)
        return tag

    xml = ['<?xml version="1.0"?>',
           # 2.2: the version that brought VTK's current Lagrange edge order;
           # with an older one VTK swaps two edges of every hexahedron on read
           '<VTKFile type="UnstructuredGrid" version="2.2" '
           'byte_order="LittleEndian" header_type="UInt64">',
           '<UnstructuredGrid>',
           f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{n_cells}">',
           '<Points>', array('', points, 3), '</Points>',
           '<Cells>', array('connectivity', conn), array('offsets', offsets),
           array('types', ctypes), '</Cells>']
    for tag, data in (('CellData', cell_data), ('PointData', point_data)):
        if data:
            xml.append(f'<{tag}>')
            for name, a in data.items():
                xml.append(array(name, np.asarray(a)))
            xml.append(f'</{tag}>')
    xml += ['</Piece>', '</UnstructuredGrid>',
            '<AppendedData encoding="raw">']
    with open(path, 'wb') as fh:
        fh.write('\n'.join(xml).encode() + b'\n_')
        for b in blobs:
            fh.write(b)
        fh.write(b'\n</AppendedData>\n</VTKFile>\n')


def _orient_linear(points, kind, conn):
    """Turn linear cells round when most of them are left-handed for VTK."""
    x = points[conn]
    if kind == 'tet':
        s = np.einsum('ij,ij->i', np.cross(x[:, 1] - x[:, 0],
                                           x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
        swap = [0, 2, 1, 3]
    elif kind == 'pyramid':
        n = np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 1])
        s = np.einsum('ij,ij->i', n, x[:, 4] - x[:, 0])
        swap = [0, 3, 2, 1, 4]
    elif kind == 'prism':
        # VTK: the normal of (0, 1, 2) points away from (3, 4, 5)
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        s = -np.einsum('ij,ij->i', n, x[:, 3] - x[:, 0])
        swap = [0, 2, 1, 3, 5, 4]
    else:
        return conn
    return conn[:, swap] if (s < 0).mean() > 0.5 else conn


def _sub_hexes(order: int) -> np.ndarray:
    """``(N**3, 8)`` canonical positions of the linear sub-cells, VTK order."""
    n = order + 1
    out = []
    for k in range(order):
        for j in range(order):
            for i in range(order):
                c = [(i, j, k), (i + 1, j, k), (i + 1, j + 1, k),
                     (i, j + 1, k), (i, j, k + 1), (i + 1, j, k + 1),
                     (i + 1, j + 1, k + 1), (i, j + 1, k + 1)]
                out.append([a + n * b + n * n * c_ for a, b, c_ in c])
    return np.array(out)


class Convert:
    """Node-order repair and writers (VTK XML, gmsh 2.2).

    Reached as ``lg.convert``.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.
    """

    def __init__(self, legolas):
        self._lg = legolas

    @property
    def mesh(self):
        """The mesh being written."""
        return self._lg.mesh

    # -- node order ----------------------------------------------------------
    def detect_order(self, sample: int = 400) -> list:
        """Check every high-order hexahedral block and find its permutation.

        Returns
        -------
        list[dict]
            Per block: ``block``, ``order``, ``fails`` (cells failing the
            tensor-order screen, all of them checked), and from
            :func:`.ordering.detect_hex_order` ``perm``, ``agreement`` and
            ``identity``; ``fails_after`` counts the failures once the
            permutation is applied.
        """
        out = []
        m = self.mesh
        for b, blk in enumerate(m.blocks):
            if blk.kind != 'hex' or blk.order < 2:
                continue
            fails = ordering.tensor_monotonic(m.points, blk.conn, blk.order,
                                              sample=None)
            r = ordering.detect_hex_order(m.points, blk.conn, blk.order,
                                          sample=sample)
            after = None
            if r['perm'] is not None:
                after = len(ordering.tensor_monotonic(
                    m.points, blk.conn[:, r['perm']], blk.order, sample=None))
            out.append({'block': b, 'order': blk.order, 'fails': len(fails),
                        'fails_after': after, **r})
        return out

    def fix_order(self, min_agreement: float = 0.9) -> list:
        """Apply the permutation of :meth:`detect_order` where it helps.

        A block is changed only when it fails the screen now, the vote agrees
        on at least ``min_agreement`` of the sampled cells, and no cell fails
        afterwards. The mesh is changed in place and every cached result
        (census, quality, resolution) is dropped.

        Returns
        -------
        list[dict]
            :meth:`detect_order`'s entries, each with ``applied``.

        Examples
        --------
        ::

            lg = LEGOLAS('Wind_turbine_disk_fluid_g4.msh')   # ANSA's own order
            lg.convert.fix_order()
            lg.convert.write_msh('Wind_turbine_disk_fluid_g4_gmsh.msh')
        """
        m = self.mesh
        report = self.detect_order()
        changed = False
        for r in report:
            ok = (r['fails'] > 0 and r['perm'] is not None
                  and r['agreement'] >= min_agreement and r['fails_after'] == 0)
            r['applied'] = bool(ok)
            if ok:
                blk = m.blocks[r['block']]
                m.blocks[r['block']] = CellBlock(blk.kind, blk.order,
                                                 blk.conn[:, r['perm']],
                                                 blk.zone)
                changed = True
        if changed:
            m.meta['warnings'] = [w for w in m.meta.get('warnings', [])
                                  if 'tensor-order' not in w]
            m.meta['node_order_fixed'] = [
                {'block': r['block'], 'perm': r['perm'].tolist()}
                for r in report if r['applied']]
            reset = getattr(self._lg, '_reset', None)
            if reset:
                reset()
            else:
                m._cache.clear()
        return report

    # -- VTK -----------------------------------------------------------------
    def to_vtu(self, path, mode: str = 'lagrange', fields: dict | None = None,
               point_fields: dict | None = None, quality: bool = False) -> str:
        """Write the volume as a ``.vtu`` file.

        Parameters
        ----------
        path : str or path-like
            Output file.
        mode : str
            ``'lagrange'`` (one VTK Lagrange cell per hexahedron, curved in
            ParaView) or ``'linear'`` (``N**3`` linear hexahedra per element,
            which any reader understands).
        fields : dict, optional
            Per-cell arrays ``{name: (n_cells,) or (n_cells, k)}`` written as
            cell data (repeated on the sub-cells in linear mode); for example
            ``{k: v['values'] for k, v in lg.resolution.fields(p=3).items()}``.
        point_fields : dict, optional
            Per-node arrays ``{name: (n_points,) or (n_points, k)}``.
        quality : bool
            Add every :mod:`.quality` metric as cell data.

        Returns
        -------
        str
            The path written. Cell data always carry ``cell`` (LEGOLAS cell
            index) and ``zone``.
        """
        if mode not in ('lagrange', 'linear'):
            raise ValueError("mode must be 'lagrange' or 'linear'")
        m = self.mesh
        conns, types, repeat = [], [], []
        for blk in m.blocks:
            if blk.kind == 'hex' and mode == 'lagrange' and blk.order > 1:
                conns.append(blk.conn[:, ordering.VTK_HEX[blk.order]])
                types.append(VTK_TYPES['hex'][1])
                repeat.append(1)
            elif blk.kind == 'hex':
                sub = _sub_hexes(blk.order)
                conns.append(blk.conn[:, sub].reshape(-1, 8))
                types.append(VTK_TYPES['hex'][0])
                repeat.append(len(sub))
            else:
                conns.append(_orient_linear(m.points, blk.kind, blk.conn))
                types.append(VTK_TYPES[blk.kind][0])
                repeat.append(1)
        cell = np.arange(m.n_cells)
        zone = np.concatenate([b.zone for b in m.blocks]) if m.blocks \
            else np.zeros(0, np.int64)
        data = {'cell': cell, 'zone': zone}
        if quality:
            data.update(self._lg.quality.metrics())
        if fields:
            for k, v in fields.items():
                v = np.asarray(v)
                if len(v) != m.n_cells:
                    raise ValueError(f"field {k!r}: {len(v)} values for "
                                     f"{m.n_cells} cells")
                data[k] = v
        per_block = []
        starts = m.offsets
        for name, v in data.items():
            parts = [np.repeat(v[starts[i]:starts[i + 1]], r, axis=0)
                     for i, r in enumerate(repeat)]
            per_block.append((name, np.concatenate(parts) if parts else v))
        if point_fields:
            for k, v in point_fields.items():
                if len(v) != m.n_points:
                    raise ValueError(f"point field {k!r}: {len(v)} values "
                                     f"for {m.n_points} nodes")
        write_vtu(path, m.points, conns, types, dict(per_block), point_fields)
        return os.fspath(path)

    def _boundary_vtu(self, path, name, mode) -> None:
        m = self.mesh
        bc = m.boundaries[name]
        b_of, local = m.locate_cell(bc.elem)
        conns, types = [], []
        for b, blk in enumerate(m.blocks):
            for f in range(blk.n_faces):
                sel = (b_of == b) & (bc.face == f)
                if not sel.any():
                    continue
                if blk.kind == 'hex':
                    grid = blk.face_nodes(f)[local[sel]]        # (k, n, n)
                    N = blk.order
                    if mode == 'lagrange' and N > 1:
                        flat = np.swapaxes(grid, 1, 2).reshape(len(grid), -1)
                        conns.append(flat[:, ordering.VTK_QUAD[N]])
                        types.append(VTK_TYPES['quad'][1])
                    else:
                        q = np.stack([grid[:, :-1, :-1], grid[:, 1:, :-1],
                                      grid[:, 1:, 1:], grid[:, :-1, 1:]], -1)
                        conns.append(q.reshape(-1, 4))
                        types.append(VTK_TYPES['quad'][0])
                else:
                    c = blk.face_corners(f)[local[sel]]
                    conns.append(c)
                    types.append(VTK_TYPES['quad' if c.shape[1] == 4
                                           else 'tri'][0])
        used = np.unique(np.concatenate([c.ravel() for c in conns])) \
            if conns else np.zeros(0, np.int64)
        remap = np.full(m.n_points, -1, np.int64)
        remap[used] = np.arange(len(used))
        write_vtu(path, m.points[used], [remap[c] for c in conns], types)

    def to_vtm(self, path, mode: str = 'lagrange', fields: dict | None = None,
               quality: bool = False) -> str:
        """Write a VTK multiblock: the volume plus one block per boundary.

        Parameters
        ----------
        path : str or path-like
            ``<stem>.vtm``; the pieces go to the folder ``<stem>/``
            (``volume.vtu`` and ``<boundary>.vtu``).
        mode, fields, quality
            As in :meth:`to_vtu`.

        Returns
        -------
        str
            The ``.vtm`` path.
        """
        path = os.fspath(path)
        stem = path[:-4] if path.endswith('.vtm') else path
        path = stem + '.vtm'
        folder = os.path.basename(stem)
        os.makedirs(stem, exist_ok=True)
        self.to_vtu(os.path.join(stem, 'volume.vtu'), mode, fields,
                    quality=quality)
        rows = [f'<DataSet index="0" name="volume" file="{folder}/volume.vtu"/>',
                '<Block index="1" name="boundaries">']
        k = 0
        for name, bc in self.mesh.boundaries.items():
            if not len(bc):
                continue
            safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_'
                           for ch in name)
            self._boundary_vtu(os.path.join(stem, safe + '.vtu'), name, mode)
            rows.append(f'<DataSet index="{k}" name="{name}" '
                        f'file="{folder}/{safe}.vtu"/>')
            k += 1
        rows.append('</Block>')
        with open(path, 'w') as fh:
            fh.write('<?xml version="1.0"?>\n<VTKFile '
                     'type="vtkMultiBlockDataSet" version="1.0">\n'
                     '<vtkMultiBlockDataSet>\n' + '\n'.join(rows)
                     + '\n</vtkMultiBlockDataSet>\n</VTKFile>\n')
        return path

    # -- gmsh ----------------------------------------------------------------
    def write_msh(self, path) -> str:
        """Write a gmsh 2.2 binary file in gmsh's own node order.

        Every volume zone becomes a 3-D physical group and every boundary a
        2-D one, with the faces as quadrangles or triangles of the same
        order as the cells. Inner zones are kept (PyHOPE wants them, with
        type 100). Binary little-endian, as gmsh writes it.

        Returns
        -------
        str
            The path written.
        """
        import struct
        m = self.mesh
        path = os.fspath(path)
        zone_tags = {}
        for blk in m.blocks:
            for z in np.unique(blk.zone):
                zone_tags.setdefault(int(z), len(zone_tags) + 1)
        bc_tags = {name: i + 1 for i, name in enumerate(m.boundaries)}
        groups = []                      # (gmsh type, conn 1-based, tag)
        for blk in m.blocks:
            ztag = np.array([zone_tags[int(z)] for z in blk.zone], np.int64)
            if blk.kind == 'hex':
                if blk.order not in _GMSH_HEX_TYPE:
                    raise NotImplementedError(f"hexahedra of order {blk.order}")
                native = np.empty_like(blk.conn)
                native[:, ordering.GMSH_HEX[blk.order]] = blk.conn
                gtype = _GMSH_HEX_TYPE[blk.order]
            else:
                native, gtype = blk.conn, _GMSH_LINEAR[blk.kind]
            for t in np.unique(ztag):
                groups.append((gtype, native[ztag == t] + 1, int(t)))
        for name, bc in m.boundaries.items():
            if not len(bc):
                continue
            b_of, local = m.locate_cell(bc.elem)
            for b, blk in enumerate(m.blocks):
                for f in range(blk.n_faces):
                    sel = (b_of == b) & (bc.face == f)
                    if not sel.any():
                        continue
                    if blk.kind == 'hex':
                        grid = blk.face_nodes(f)[local[sel]]
                        flat = grid.reshape(len(grid), -1)
                        rows = flat[:, ordering.GMSH_QUAD[blk.order]]
                        gtype = _GMSH_QUAD_TYPE[blk.order]
                    else:
                        rows = blk.face_corners(f)[local[sel]]
                        gtype = 3 if rows.shape[1] == 4 else 2
                    groups.append((gtype, rows + 1, bc_tags[name]))
        n_elem = sum(len(g[1]) for g in groups)
        zone_names = {z: m.zones.get(z, f'zone_{z}') for z in zone_tags}
        with open(path, 'wb') as fh:
            w = fh.write
            w(b'$MeshFormat\n2.2 1 8\n' + struct.pack('<i', 1)
              + b'\n$EndMeshFormat\n')
            w(b'$PhysicalNames\n%d\n' % (len(zone_tags) + len(bc_tags)))
            for z, tag in zone_tags.items():
                w(b'3 %d "%s"\n' % (tag, zone_names[z].encode()))
            for name, tag in bc_tags.items():
                w(b'2 %d "%s"\n' % (tag, name.encode()))
            w(b'$EndPhysicalNames\n$Nodes\n%d\n' % m.n_points)
            rec = np.zeros(m.n_points, dtype=[('id', '<i4'), ('x', '<f8', 3)])
            rec['id'] = np.arange(1, m.n_points + 1)
            rec['x'] = m.points
            w(rec.tobytes() + b'\n$EndNodes\n$Elements\n%d\n' % n_elem)
            eid = 1
            for gtype, rows, tag in groups:
                k = len(rows)
                w(np.array([gtype, k, 2], '<i4').tobytes())
                arr = np.hstack([np.arange(eid, eid + k)[:, None],
                                 np.full((k, 2), tag), rows])
                w(arr.astype('<i4').tobytes())
                eid += k
            w(b'\n$EndElements\n')
        return path

    def __repr__(self) -> str:
        return f"<LEGOLAS.Convert of {self.mesh!r}>"
