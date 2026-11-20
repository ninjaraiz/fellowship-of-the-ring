"""LEGOLAS L1: canonical layout, readers (HOPR, CGNS, gmsh, hmesh) and census.

Three layers of evidence:

1. **Ground truth from gmsh.** ``data/legolas/hex_p{1..4}`` are a unit cube
   meshed by gmsh 4.13.1 as one element of order 1-4 and written as msh
   v2.2, msh v4.1 and CGNS (``cube.geo``; ``gmsh cube.geo -3 -order N
   -format msh22|msh41|cgns``). After reading, canonical node ``(i, j, k)``
   must sit at ``(i, j, k) / N``. That checks every permutation table
   against gmsh, not against itself.
2. **Synthetic curved meshes** written here in the four formats from one
   canonical mesh: the readers must agree on everything.
3. **Real meshes** from COMAC when they are on disk (skipped otherwise).
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

from FotR.characters.legolas import LEGOLAS, Mesh, CellBlock, read  # noqa: E402
from FotR.characters.legolas import ordering, topology  # noqa: E402
from FotR.characters.legolas.io import detect_format  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data', 'legolas')
COMAC = os.path.expanduser('~/Documentos/COMAC')
TURBINE = os.path.join(COMAC, 'wind_turbine_ansa', 'sources')
SPHERE = os.path.join(COMAC, 'my_ESPHERE_1', 'outputs')


# ---------------------------------------------------------------------------
# 1. ground truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize('suffix', ['_v22.msh', '_v41.msh', '.cgns'])
def test_reference_element_lands_on_tensor_grid(order, suffix):
    lg = LEGOLAS(os.path.join(DATA, f'hex_p{order}{suffix}'))
    (block,) = lg.mesh.blocks
    assert block.kind == 'hex' and block.order == order and len(block) == 1
    x = lg.mesh.points[block.conn[0]]
    n = order + 1
    k, j, i = np.meshgrid(*(np.arange(n),) * 3, indexing='ij')
    expected = np.stack([i, j, k], -1).reshape(-1, 3) / order
    np.testing.assert_allclose(x, expected, atol=1e-12)


@pytest.mark.parametrize('suffix', ['_v22.msh', '_v41.msh', '.cgns'])
def test_reference_element_census(suffix):
    s = LEGOLAS(os.path.join(DATA, f'hex_p3{suffix}')).census.summary()
    assert s['exterior_faces'] == 6 and s['unassigned_faces'] == 0
    if suffix != '.cgns':      # gmsh's CGNS writer names its own sections
        areas = {b['name']: b['n_faces'] for b in s['boundaries']}
        assert areas == {'bottom': 1, 'top': 1, 'sides': 4}
    assert sum(b['area'] for b in s['boundaries']) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# canonical layout helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_face_grids_point_outward(order):
    n = order + 1
    k, j, i = np.meshgrid(*(np.arange(n),) * 3, indexing='ij')
    x = np.stack([i, j, k], -1).reshape(-1, 3) / order
    normals = [(0, 0, -1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0),
               (0, 0, 1)]
    for f, nrm in enumerate(normals):
        g = x[ordering.hex_face_grid(order, f)]
        d = np.cross(g[1, 0] - g[0, 0], g[0, 1] - g[0, 0])
        np.testing.assert_allclose(d / np.linalg.norm(d), nrm)
        grid = ordering.hex_face_grid(order, f)
        assert list(ordering.hex_face_corners(order)[f]) == [
            grid[0, 0], grid[-1, 0], grid[-1, -1], grid[0, -1]]


def test_tensor_monotonic_flags_a_wrong_permutation():
    x, conn = _grid(2, 2, 2, order=3)
    assert len(ordering.tensor_monotonic(x, conn, 3, sample=None)) == 0
    bad = conn[:, np.r_[1, 0, 2:conn.shape[1]]]
    assert len(ordering.tensor_monotonic(x, bad, 3, sample=None)) == len(conn)


def test_group_keys_matches_numpy_unique():
    rng = np.random.default_rng(1)
    keys = np.sort(rng.integers(-1, 50, size=(2000, 4)), axis=1)
    inv, cnt = topology.group_keys(keys)
    _, inv2, cnt2 = np.unique(keys, axis=0, return_inverse=True,
                              return_counts=True)
    assert (inv == inv2.ravel()).all() and (cnt == cnt2).all()


# ---------------------------------------------------------------------------
# 2. synthetic curved mesh in every format
# ---------------------------------------------------------------------------

NX, NY, NZ = 3, 2, 4


def _grid(nx, ny, nz, order):
    """Curved structured hex block: points and canonical connectivity."""
    n = order
    gx, gy, gz = (np.linspace(0, 1, m * n + 1) for m in (nx, ny, nz))
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing='ij')
    # curved inside, but the inlet x = 0 stays the flat unit square
    x = X + 0.04 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    y = Y + 0.04 * np.sin(np.pi * X) * np.sin(np.pi * Z)
    pts = np.stack([x, y, Z], -1).reshape(-1, 3)
    sx, sy = nx * n + 1, ny * n + 1
    conn = []
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                kk, jj, ii = np.meshgrid(*(np.arange(n + 1),) * 3,
                                         indexing='ij')
                gi, gj, gk = ei * n + ii, ej * n + jj, ek * n + kk
                conn.append((gi + sx * gj + sx * sy * gk).ravel())
    return pts, np.array(conn)


def _structured_faces(nx, ny, nz):
    """(elem, face) per boundary and the inner plane k = nz // 2."""
    out = {'inlet': [], 'outlet': [], 'walls': [], 'inner': []}
    e = 0
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                if ei == 0:
                    out['inlet'].append((e, 4))
                if ei == nx - 1:
                    out['outlet'].append((e, 2))
                for cond, f in ((ej == 0, 1), (ej == ny - 1, 3),
                                (ek == 0, 0), (ek == nz - 1, 5)):
                    if cond:
                        out['walls'].append((e, f))
                if ek == nz // 2 - 1:
                    out['inner'].append((e, 5))
                e += 1
    return out


def _neighbour(e, f, nx, ny, nz):
    ei, ej, ek = e % nx, (e // nx) % ny, e // (nx * ny)
    d = {0: (0, 0, -1, 5), 1: (0, -1, 0, 3), 2: (1, 0, 0, 4),
         3: (0, 1, 0, 1), 4: (-1, 0, 0, 2), 5: (0, 0, 1, 0)}[f]
    i, j, k = ei + d[0], ej + d[1], ek + d[2]
    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
        return i + nx * (j + ny * k), d[3]
    return -1, -1


def write_hopr(path, pts, conn, order, faces):
    names = ['inlet', 'outlet', 'walls', 'inner']
    ne, nn = conn.shape
    E = np.zeros((ne, 6), np.int64)
    S = np.zeros((6 * ne, 5), np.int64)
    bc_of = {}
    for b, name in enumerate(names):
        for e, f in faces[name]:
            bc_of[(e, f)] = b + 1
            if name == 'inner':
                nb, lf = _neighbour(e, f, NX, NY, NZ)
                bc_of[(nb, lf)] = b + 1           # HOPR writes both sides
    for e in range(ne):
        E[e] = [108 if order == 1 else 208, 1, 6 * e, 6 * e + 6, nn * e,
                nn * (e + 1)]
        for f in range(6):
            nb, lf = _neighbour(e, f, NX, NY, NZ)
            S[6 * e + f] = [4, 0, nb + 1, 10 * (lf + 1) if nb >= 0 else 0,
                            bc_of.get((e, f), 0)]
    with h5py.File(path, 'w') as fh:
        fh['NodeCoords'] = pts[conn].reshape(-1, 3)
        fh['GlobalNodeIDs'] = conn.ravel() + 1
        fh['ElemInfo'] = E
        fh['SideInfo'] = S
        fh['BCNames'] = np.array([n.ljust(255).encode() for n in names])
        fh['BCType'] = np.array([[3, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0],
                                 [100, 0, 0, 0]])
        fh.attrs['Ngeo'] = order


def _face_rows(conn, order, faces):
    """All nodes of each listed face, corners first (counter-clockwise)."""
    rows = []
    for e, f in faces:
        g = conn[e][ordering.hex_face_grid(order, f)]
        corners = [g[0, 0], g[-1, 0], g[-1, -1], g[0, -1]]
        rest = [v for v in g.ravel() if v not in corners]
        rows.append(corners + rest)
    return np.array(rows, dtype=np.int64)


def _inverse(perm):
    inv = np.empty(len(perm), dtype=np.int64)
    inv[np.asarray(perm)] = np.arange(len(perm))
    return inv


def _cgns_node(parent, name, label, dtype=b'MT', data=None):
    g = parent.create_group(name)
    g.attrs['name'] = np.bytes_(name.ljust(33, '\0')[:33])
    g.attrs['label'] = np.bytes_(label.ljust(33, '\0')[:33])
    g.attrs['type'] = np.bytes_(dtype)
    g.attrs['flags'] = np.array([1], dtype=np.int32)
    if data is not None:
        g[' data'] = data
    return g


def _cgns_str(s):
    return np.frombuffer(s.encode(), dtype=np.int8)


def write_cgns(path, pts, conn, order, faces, mixed=False, zonebc=True):
    vol_code = {1: 17, 2: 19, 3: 39, 4: 56}[order]
    quad_code = {1: 7, 2: 9, 3: 28, 4: 44}[order]
    native = conn[:, _inverse(ordering.CGNS_HEX[vol_code])] + 1
    with h5py.File(path, 'w') as fh:
        _cgns_node(fh, 'CGNSLibraryVersion', 'CGNSLibraryVersion_t', b'R4',
                   np.array([4.2], np.float32))
        base = _cgns_node(fh, 'Base', 'CGNSBase_t', b'I4',
                          np.array([3, 3], np.int32))
        z = _cgns_node(base, 'fluid', 'Zone_t', b'I8',
                       np.array([[len(pts)], [len(conn)], [0]]))
        _cgns_node(z, 'ZoneType', 'ZoneType_t', b'C1',
                   _cgns_str('Unstructured'))
        gc = _cgns_node(z, 'GridCoordinates', 'GridCoordinates_t')
        for c, col in zip(('CoordinateX', 'CoordinateY', 'CoordinateZ'),
                          pts.T):
            _cgns_node(gc, c, 'DataArray_t', b'R8', col)
        first = 1
        if mixed:
            data = np.hstack([np.full((len(native), 1), vol_code),
                              native]).ravel()
            code = 20
        else:
            data, code = native.ravel(), vol_code
        sec = _cgns_node(z, 'Volume', 'Elements_t', b'I4',
                         np.array([code, 0], np.int32))
        _cgns_node(sec, 'ElementRange', 'IndexRange_t', b'I8',
                   np.array([first, first + len(conn) - 1]))
        _cgns_node(sec, 'ElementConnectivity', 'DataArray_t', b'I8', data)
        first += len(conn)
        bc_root = _cgns_node(z, 'ZoneBC', 'ZoneBC_t') if zonebc else None
        for name in ('inlet', 'outlet', 'walls', 'inner'):
            rows = _face_rows(conn, order, faces[name]) + 1
            sec = _cgns_node(z, name, 'Elements_t', b'I4',
                             np.array([quad_code, 0], np.int32))
            _cgns_node(sec, 'ElementRange', 'IndexRange_t', b'I8',
                       np.array([first, first + len(rows) - 1]))
            _cgns_node(sec, 'ElementConnectivity', 'DataArray_t', b'I8',
                       rows.ravel())
            if zonebc:
                bc = _cgns_node(bc_root, name, 'BC_t', b'C1',
                                _cgns_str('BCWall'))
                _cgns_node(bc, 'PointList', 'IndexArray_t', b'I8',
                           np.arange(first, first + len(rows))[:, None])
                _cgns_node(bc, 'GridLocation', 'GridLocation_t', b'C1',
                           _cgns_str('FaceCenter'))
            first += len(rows)


def write_msh22(path, pts, conn, order, faces, binary=False, byteorder='<'):
    vol_type = {1: 5, 2: 12, 3: 92, 4: 93}[order]
    quad_type = {1: 3, 2: 10, 3: 36, 4: 37}[order]
    native = conn[:, _inverse(ordering.GMSH_HEX[order])] + 1
    names = ['inlet', 'outlet', 'walls', 'inner']
    blocks = [(vol_type, native, 1)]
    blocks += [(quad_type, _face_rows(conn, order, faces[n]) + 1, i + 1)
               for i, n in enumerate(names)]
    with open(path, 'wb') as fh:
        w = fh.write
        w(b'$MeshFormat\n2.2 %d 8\n' % (1 if binary else 0))
        if binary:
            w(struct.pack(byteorder + 'i', 1) + b'\n')
        w(b'$EndMeshFormat\n$PhysicalNames\n5\n')
        for i, n in enumerate(names):
            w(b'2 %d "%s"\n' % (i + 1, n.encode()))
        w(b'3 1 "fluid"\n$EndPhysicalNames\n$Nodes\n%d\n' % len(pts))
        ids = np.arange(1, len(pts) + 1)
        if binary:
            rec = np.zeros(len(pts), dtype=[('id', byteorder + 'i4'),
                                             ('x', byteorder + 'f8', 3)])
            rec['id'], rec['x'] = ids, pts
            w(rec.tobytes() + b'\n')
        else:
            for i, p in zip(ids, pts):
                w(b'%d %.17g %.17g %.17g\n' % (i, *p))
        n_el = sum(len(b[1]) for b in blocks)
        w(b'$EndNodes\n$Elements\n%d\n' % n_el)
        eid = 1
        for etype, rows, tag in blocks:
            if binary:
                w(np.array([etype, len(rows), 2], byteorder + 'i4').tobytes())
                arr = np.hstack([np.arange(eid, eid + len(rows))[:, None],
                                 np.full((len(rows), 2), tag), rows])
                w(arr.astype(byteorder + 'i4').tobytes())
            else:
                for r in rows:
                    w(b'%d %d 2 %d %d ' % (eid, etype, tag, tag) +
                      b' '.join(b'%d' % v for v in r) + b'\n')
                    eid += 1
                continue
            eid += len(rows)
        if binary:
            w(b'\n')
        w(b'$EndElements\n')


def _write_all(tmp_path, order):
    pts, conn = _grid(NX, NY, NZ, order)
    faces = _structured_faces(NX, NY, NZ)
    files = {
        'hopr': tmp_path / f'm_g{order}_mesh.h5',
        'cgns': tmp_path / f'm_g{order}.cgns',
        'cgns_mixed': tmp_path / f'm_g{order}_mixed.cgns',
        'cgns_sections': tmp_path / f'm_g{order}_sections.cgns',
        'msh_ascii': tmp_path / f'm_g{order}_a.msh',
        'msh_le': tmp_path / f'm_g{order}_le.msh',
        'msh_be': tmp_path / f'm_g{order}_be.msh',
    }
    write_hopr(files['hopr'], pts, conn, order, faces)
    write_cgns(files['cgns'], pts, conn, order, faces)
    write_cgns(files['cgns_mixed'], pts, conn, order, faces, mixed=True)
    write_cgns(files['cgns_sections'], pts, conn, order, faces, zonebc=False)
    write_msh22(files['msh_ascii'], pts, conn, order, faces)
    write_msh22(files['msh_le'], pts, conn, order, faces, binary=True)
    write_msh22(files['msh_be'], pts, conn, order, faces, binary=True,
                byteorder='>')
    return pts, conn, faces, files


@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_all_formats_agree_on_a_curved_mesh(tmp_path, order):
    pts, conn, faces, files = _write_all(tmp_path, order)
    ref = None
    for kind, path in files.items():
        lg = LEGOLAS(path)
        m = lg.mesh
        assert m.meta['warnings'] == [], (kind, m.meta['warnings'])
        (block,) = m.blocks
        assert block.kind == 'hex' and block.order == order
        # canonical connectivity: same element coordinates as written
        np.testing.assert_allclose(m.points[block.conn], pts[conn],
                                   err_msg=kind)
        s = lg.census.summary()
        got = {b['name']: (b['n_faces'], round(b['area'], 10),
                           b['interior_faces']) for b in s['boundaries']}
        if ref is None:
            ref = got
        assert got == ref, kind
        assert s['exterior_faces'] == 2 * (NX * NY + NY * NZ + NX * NZ)
        assert s['unassigned_faces'] == 0 and s['nonmanifold_faces'] == 0
        assert s['orphan_nodes'] == 0
    assert ref['inner'] == (NX * NY, ref['inner'][1], NX * NY)
    assert ref['inlet'][0] == NY * NZ and ref['inlet'][1] == pytest.approx(1)


def test_msh_byte_order_is_reported(tmp_path):
    *_, files = _write_all(tmp_path, 2)
    assert read(files['msh_be']).meta['byteorder'] == 'big-endian'
    assert read(files['msh_le']).meta['byteorder'] == 'little-endian'
    assert 'byteorder' not in read(files['msh_ascii']).meta


def test_wrong_node_order_is_flagged(tmp_path):
    pts, conn, faces, _ = _write_all(tmp_path, 3)
    shuffled = conn[:, np.r_[0, 2, 1, 3:conn.shape[1]]]
    path = tmp_path / 'bad.msh'
    write_msh22(path, pts, shuffled, 3, faces)
    warnings = read(path).meta['warnings']
    assert any('tensor-order' in w for w in warnings)


def test_mixed_linear_kinds(tmp_path):
    # one hexahedron and one prism glued on its x = 1 face, as CGNS MIXED
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1],
                  [1, 0, 1], [1, 1, 1], [0, 1, 1], [2, 0, 0], [2, 0, 1]],
                 float)
    hexa = [1, 2, 3, 4, 5, 6, 7, 8]          # CGNS/VTK corner order
    prism = [2, 9, 3, 6, 10, 7]
    data = np.array([17, *hexa, 14, *prism])
    path = tmp_path / 'mixed.cgns'
    with h5py.File(path, 'w') as fh:
        _cgns_node(fh, 'CGNSLibraryVersion', 'CGNSLibraryVersion_t', b'R4',
                   np.array([4.2], np.float32))
        base = _cgns_node(fh, 'Base', 'CGNSBase_t', b'I4',
                          np.array([3, 3], np.int32))
        z = _cgns_node(base, 'z', 'Zone_t', b'I8', np.array([[10], [2], [0]]))
        gc = _cgns_node(z, 'GridCoordinates', 'GridCoordinates_t')
        for c, col in zip(('CoordinateX', 'CoordinateY', 'CoordinateZ'), p.T):
            _cgns_node(gc, c, 'DataArray_t', b'R8', col)
        sec = _cgns_node(z, 'Cells', 'Elements_t', b'I4',
                         np.array([20, 0], np.int32))
        _cgns_node(sec, 'ElementRange', 'IndexRange_t', b'I8',
                   np.array([1, 2]))
        _cgns_node(sec, 'ElementConnectivity', 'DataArray_t', b'I8', data)
    lg = LEGOLAS(path)
    kinds = sorted((b.kind, len(b)) for b in lg.mesh.blocks)
    assert kinds == [('hex', 1), ('prism', 1)]
    s = lg.census.summary()
    assert s['exterior_faces'] == 6 + 5 - 2     # the shared quad is interior
    hexblock = [b for b in lg.mesh.blocks if b.kind == 'hex'][0]
    # linear hexahedra are tensor-ordered too
    np.testing.assert_allclose(lg.mesh.points[hexblock.conn[0]][[1, 2, 4]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_detect_format_errors(tmp_path):
    adf = tmp_path / 'old.cgns'
    adf.write_bytes(b'ADF Database Version A02011>')
    with pytest.raises(ValueError, match='ADF'):
        detect_format(str(adf))
    with pytest.raises(ValueError, match='cannot tell'):
        detect_format(str(tmp_path / 'mesh.xyz'))
    with pytest.raises(FileNotFoundError):
        read(tmp_path / 'missing.msh')


def test_mesh_from_memory_and_repr():
    pts, conn = _grid(1, 1, 1, order=2)
    lg = LEGOLAS(Mesh(pts, [CellBlock('hex', 2, conn)]))
    assert lg.name == '<memory>'
    assert lg.census.summary()['exterior_faces'] == 6
    assert 'hex2' in repr(lg.mesh)
    assert 'LEGOLAS.Census' in repr(lg.census)
    assert isinstance(lg.census, LEGOLAS.Census)


def test_import_is_light():
    code = ("import sys, FotR.characters.legolas as L; "
            "print(','.join(m for m in ('meshio', 'scipy', 'torch', 'pandas',"
            " 'pyvista', 'vtk') if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == ''


# ---------------------------------------------------------------------------
# 3. real meshes (skipped when absent)
# ---------------------------------------------------------------------------

def _need(*paths):
    for p in paths:
        if not os.path.isfile(p):
            pytest.skip(f'{p} not available')


def _bc(s):
    return {b['name'].lower(): (b['n_faces'], pytest.approx(b['area'],
                                                            rel=1e-6),
                                b['interior_faces'])
            for b in s['boundaries']}


@pytest.mark.parametrize('g', [1, 2])
def test_turbine_three_formats_agree(g):
    files = [os.path.join(TURBINE, f) for f in (
        f'Wind_turbine_disk_fluid_cgns_g{g}_mesh.h5',
        f'Wind_turbine_disk_fluid_g{g}.cgns',
        f'Wind_turbine_disk_fluid_g{g}.msh')]
    _need(*files)
    sums = [LEGOLAS(f).census.summary() for f in files]
    assert _bc(sums[0]) == _bc(sums[1]) == _bc(sums[2])
    for s in sums:
        assert s['n_cells'] == 38718 and s['unassigned_faces'] == 0
        assert not [w for w in s['warnings'] if 'tensor-order' in w]
    assert _bc(sums[0])['rotor_cylinder'][2] == 2560     # inner zone, once


def test_turbine_g4_ansa_msh_has_its_own_node_order():
    good = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_g4.cgns')
    bad = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_g4.msh')
    _need(good, bad)
    assert not LEGOLAS(good).mesh.meta['warnings']
    warnings = LEGOLAS(bad).mesh.meta['warnings']
    assert any('tensor-order' in w for w in warnings)


def test_sphere_hmesh_matches_its_mesh():
    h5 = os.path.join(SPHERE, 'M_0.3_re_400', 'MESH',
                      'my_esphere_v2g2_mesh.h5')
    hm = os.path.join(SPHERE, 'M_0.3_re_400', 'MESH',
                      'my_esphere_v2g2p2_0000000000.hmesh')
    _need(h5, hm)
    a, b = LEGOLAS(h5), LEGOLAS(hm)
    assert a.mesh.n_points == b.mesh.n_points
    assert (a.census.summary()['exterior_faces']
            == b.census.summary()['exterior_faces'])
