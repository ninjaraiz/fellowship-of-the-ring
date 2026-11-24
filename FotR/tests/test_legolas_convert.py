"""LEGOLAS L6: node-order repair, VTK and gmsh writers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS, Boundary, CellBlock, Mesh, read
from FotR.characters.legolas import ordering
from FotR.characters.legolas.io.gmsh import read_msh22

from _legolas_meshes import box_mesh

DATA = os.path.join(os.path.dirname(__file__), 'data', 'legolas')
TURBINE = os.path.expanduser('~/Documentos/COMAC/wind_turbine_ansa/sources')


def curved_box(order, nx=2, ny=2, nz=2):
    m = box_mesh(nx, ny, nz, order=order)
    x, y, z = m.points.T
    m.points = np.stack([x + 0.04 * np.sin(np.pi * y) * np.sin(np.pi * z / nz),
                         y + 0.03 * np.sin(np.pi * x / nx), z], 1)
    m._cache.clear()
    return m


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_vtk_tables_match_vtk(order):
    vtk = pytest.importorskip('vtk')
    n = order + 1
    for cls, npts, table, dims in (
            (vtk.vtkLagrangeHexahedron, n ** 3, ordering.VTK_HEX, 3),
            (vtk.vtkLagrangeQuadrilateral, n ** 2, ordering.VTK_QUAD, 2)):
        c = cls()
        c.GetPointIds().SetNumberOfIds(npts)
        c.GetPoints().SetNumberOfPoints(npts)
        c.SetOrder(*([order] * dims))
        c.Initialize()
        pc = c.GetParametricCoords()
        pc = np.array([pc[i] for i in range(3 * npts)]).reshape(-1, 3)
        ijk = np.rint(pc * order).astype(int)
        flat = ijk[:, 0] + n * ijk[:, 1] + (n * n * ijk[:, 2] if dims == 3 else 0)
        assert flat.tolist() == list(table[order])


@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_gmsh_quad_table_matches_the_reference_cube(order):
    d = read_msh22(os.path.join(DATA, f'hex_p{order}_v22.msh'))
    qtype = {1: 3, 2: 10, 3: 36, 4: 37}[order]
    conn = [c for t, c, _, _ in d['elements'] if t == qtype][0]
    for q in conn:
        x = d['points'][q]
        c = x[:4]
        e1, e3 = c[1] - c[0], c[3] - c[0]
        a = np.rint((x - c[0]) @ e1 / (e1 @ e1) * order).astype(int)
        b = np.rint((x - c[0]) @ e3 / (e3 @ e3) * order).astype(int)
        assert (a * (order + 1) + b).tolist() == ordering.GMSH_QUAD[order]


# ---------------------------------------------------------------------------
# node order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [2, 3, 4])
def test_fix_order_undoes_a_permutation(order):
    good = curved_box(order, 3, 2, 2)
    n3 = (order + 1) ** 3
    rng = np.random.default_rng(order)
    corners = set(ordering.hex_corners(order).tolist())
    inner = [t for t in range(n3) if t not in corners]
    perm = np.arange(n3)
    perm[inner] = rng.permutation(inner)
    bad = Mesh(good.points, [CellBlock('hex', order,
                                       good.blocks[0].conn[:, perm])],
               dict(good.boundaries))
    lg = LEGOLAS(bad)
    rep = lg.convert.fix_order()
    assert rep[0]['applied'] and rep[0]['fails'] > 0
    assert rep[0]['fails_after'] == 0 and rep[0]['agreement'] == 1.0
    np.testing.assert_array_equal(lg.mesh.blocks[0].conn, good.blocks[0].conn)
    # nothing to do on a sound mesh
    rep = LEGOLAS(curved_box(order)).convert.fix_order()
    assert not rep[0]['applied'] and rep[0]['identity']


def test_fix_order_drops_cached_results():
    good = curved_box(3)
    conn = good.blocks[0].conn.copy()
    conn[:, [1, 2]] = conn[:, [2, 1]]
    lg = LEGOLAS(Mesh(good.points, [CellBlock('hex', 3, conn)],
                      dict(good.boundaries)))
    # nodes 1 and 2 lie on an x edge of the 'walls' faces, not on 'inlet'
    walls = lambda: [b['area'] for b in lg.census.summary()['boundaries']
                     if b['name'] == 'walls'][0]
    before = walls()
    ratio = lg.quality.summary()['jacobian_ratio']['min']
    lg.convert.fix_order()
    good_ratio = LEGOLAS(good).quality.summary()['jacobian_ratio']['min']
    assert lg.quality.summary()['jacobian_ratio']['min'] == pytest.approx(
        good_ratio)
    assert ratio < good_ratio
    assert walls() != before
    assert walls() == pytest.approx(
        [b['area'] for b in LEGOLAS(good).census.summary()['boundaries']
         if b['name'] == 'walls'][0])


# ---------------------------------------------------------------------------
# VTK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_vtu_lagrange_is_read_back_by_vtk(tmp_path, order):
    pv = pytest.importorskip('pyvista')
    import vtk
    m = curved_box(order)
    lg = LEGOLAS(m)
    g = pv.read(lg.convert.to_vtu(tmp_path / 'm.vtu', 'lagrange',
                                  quality=True))
    assert g.n_cells == m.n_cells and g.n_points == m.n_points
    assert {'cell', 'zone', 'scaled_jacobian'} <= set(g.cell_data)
    # VTK's evaluation of every cell equals the LEGOLAS mapping
    from FotR.characters.legolas.quality import lagrange
    n = order + 1
    nodes = np.linspace(-1, 1, n)
    for cid in range(m.n_cells):
        cell = g.GetCell(cid)
        X = m.points[m.blocks[0].conn[cid]].reshape(n, n, n, 3)
        for pc in ([0.5, 0.5, 0.5], [0.2, 0.7, 0.9]):
            x = [0.0] * 3
            w = [0.0] * cell.GetNumberOfPoints()
            cell.EvaluateLocation(vtk.mutable(0), pc, x, w)
            L = [lagrange(nodes, [2 * p - 1])[0][0] for p in pc]
            ref = np.einsum('k,j,i,kjid->d', L[2], L[1], L[0], X)
            np.testing.assert_allclose(x, ref, atol=1e-12)


def test_vtu_linear_and_point_data(tmp_path):
    pv = pytest.importorskip('pyvista')
    m = curved_box(3)
    lg = LEGOLAS(m)
    g = pv.read(lg.convert.to_vtu(
        tmp_path / 'm.vtu', 'linear', fields={'f': np.arange(m.n_cells)},
        point_fields={'x': m.points[:, 0]}))
    assert g.n_cells == m.n_cells * 27
    np.testing.assert_array_equal(g.cell_data['f'],
                                  np.repeat(np.arange(m.n_cells), 27))
    np.testing.assert_allclose(g.point_data['x'], m.points[:, 0])
    vol = g.compute_cell_sizes(length=False, area=False)['Volume'].sum()
    assert vol == pytest.approx(lg.quality.field('volume').sum(), rel=1e-3)
    with pytest.raises(ValueError):
        lg.convert.to_vtu(tmp_path / 'x.vtu', 'linear', fields={'f': [1, 2]})


@pytest.mark.parametrize('mode', ['linear', 'lagrange'])
def test_vtm_has_the_volume_and_one_block_per_boundary(tmp_path, mode):
    pv = pytest.importorskip('pyvista')
    m = box_mesh(3, 2, 2, order=2)
    lg = LEGOLAS(m)
    path = lg.convert.to_vtm(tmp_path / 'case', mode)
    assert path.endswith('case.vtm') and os.path.isdir(tmp_path / 'case')
    mb = pv.read(path)
    assert list(mb.keys()) == ['volume', 'boundaries']
    names = list(mb['boundaries'].keys())
    assert names == ['inlet', 'walls']
    areas = {b['name']: b['area'] for b in lg.census.summary()['boundaries']}
    for name, block in zip(names, mb['boundaries']):
        a = block.compute_cell_sizes(length=False, volume=False)['Area'].sum()
        assert a == pytest.approx(areas[name])


def test_linear_kinds_are_oriented_for_vtk(tmp_path):
    pv = pytest.importorskip('pyvista')
    s3 = np.sqrt(3)
    # a gmsh-ordered prism (normal of (0, 1, 2) towards (3, 4, 5))
    p = np.array([[0, 0, 0], [1, 0, 0], [.5, s3 / 2, 0], [0, 0, 1], [1, 0, 1],
                  [.5, s3 / 2, 1]], float)
    lg = LEGOLAS(Mesh(p, [CellBlock('prism', 1, np.arange(6)[None])]))
    g = pv.read(lg.convert.to_vtu(tmp_path / 'p.vtu'))
    vol = g.compute_cell_sizes(length=False, area=False)['Volume'][0]
    assert vol == pytest.approx(s3 / 4)


# ---------------------------------------------------------------------------
# gmsh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_write_msh_round_trip(tmp_path, order):
    m = curved_box(order, 3, 2, 2)
    m.boundaries['empty'] = Boundary('empty', [], [])
    lg = LEGOLAS(m)
    path = lg.convert.write_msh(tmp_path / 'm.msh')
    back = read(path)
    assert back.meta['byteorder'] == 'little-endian'
    assert back.meta['warnings'] == []
    (blk,) = back.blocks
    np.testing.assert_allclose(back.points[blk.conn], m.points[m.blocks[0].conn])
    c = LEGOLAS(back).census.compare(lg)
    assert c['same_corners'] and c['issues'] == []
    # a boundary with no faces survives: PyHOPE wants every declared name
    assert set(back.boundaries) == {'empty', 'inlet', 'walls'}
    assert len(back.boundaries['empty']) == 0


def test_write_msh_is_read_by_meshio(tmp_path):
    meshio = pytest.importorskip('meshio')
    m = curved_box(2)
    path = LEGOLAS(m).convert.write_msh(tmp_path / 'm.msh')
    mm = meshio.read(path)
    assert {c.type for c in mm.cells} == {'hexahedron27', 'quad9'}
    assert set(mm.field_data) == {'fluid', 'inlet', 'walls'}


def test_ansa_g4_msh_becomes_the_cgns_mesh(tmp_path):
    bad = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_g4.msh')
    good = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_g4.cgns')
    for f in (bad, good):
        if not os.path.isfile(f):
            pytest.skip(f'{f} not available')
    lg = LEGOLAS(bad)
    assert any('tensor-order' in w for w in lg.mesh.meta['warnings'])
    rep = lg.convert.fix_order()
    assert rep[0]['applied'] and rep[0]['agreement'] == 1.0
    ref = LEGOLAS(good)
    c = lg.census.compare(ref)
    assert c['same_corners'] and c['issues'] == []
    out = lg.convert.write_msh(tmp_path / 'g4.msh')
    again = LEGOLAS(out)
    assert again.mesh.meta['warnings'] == []
    assert again.census.compare(ref)['same_corners']
