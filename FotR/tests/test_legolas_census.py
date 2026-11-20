"""LEGOLAS L3: BC files, check_bcs, watertight, duplicates, compare, check."""

from __future__ import annotations

import os

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS, Boundary, CellBlock, Mesh
from FotR.characters.legolas.bcfiles import read_control_bcs, read_ini_bcs

from _legolas_meshes import box_mesh

TURBINE = os.path.expanduser('~/Documentos/COMAC/wind_turbine_ansa/sources')
NX, NY, NZ = 3, 2, 4


def box_with_inner(fmt='hopr', order=1):
    """Box with BCs inlet/walls and an inner zone on the plane z = 2."""
    mesh = box_mesh(NX, NY, NZ, order=order)
    cells = [ei + NX * (ej + NY * 1) for ej in range(NY) for ei in range(NX)]
    mesh.boundaries['inner'] = Boundary('inner', cells, [5] * len(cells))
    mesh.meta['format'] = fmt
    return mesh


CONTROL = """\
!#define boundary inlet__walls__inner
!    type = Inflow
!#end

#define boundary Inlet        ! comment after the name
  type = Inflow
#end

#define boundary walls__ghost
  type = FreeSlipWall   ! trailing comment
#end
"""

INI = """\
BoundaryName = inlet
BoundaryType = (/2,0,0,0/)
BoundaryName = walls      ! comment
BoundaryType = (/4,0,0,0/)
!BoundaryName = commented
!BoundaryType = (/3,0,0,0/)
BoundaryName = inner
BoundaryType = (/4,0,0,0/)
"""


@pytest.fixture
def files(tmp_path):
    c = tmp_path / 'case.control'
    c.write_text(CONTROL)
    i = tmp_path / 'parameter.ini'
    i.write_text(INI)
    return c, i


def _levels(issues, level):
    return [i['message'] for i in issues if i['level'] == level]


# ---------------------------------------------------------------------------
# file parsers
# ---------------------------------------------------------------------------

def test_read_control_bcs(files):
    entries = read_control_bcs(files[0])
    assert [(e['name'], e['type']) for e in entries] == [
        ('Inlet', 'Inflow'), ('walls', 'FreeSlipWall'),
        ('ghost', 'FreeSlipWall')]
    assert entries[1]['group'] == ['walls', 'ghost']
    assert entries[0]['line'] == 5


def test_read_control_bcs_unclosed(tmp_path):
    p = tmp_path / 'bad.control'
    p.write_text('#define boundary inlet\n  type = Inflow\n')
    with pytest.raises(ValueError, match="no '#end'"):
        read_control_bcs(p)


def test_read_ini_bcs(files):
    entries = read_ini_bcs(files[1])
    assert [(e['name'], e['type']) for e in entries] == [
        ('inlet', (2, 0, 0, 0)), ('walls', (4, 0, 0, 0)),
        ('inner', (4, 0, 0, 0))]


# ---------------------------------------------------------------------------
# check_bcs
# ---------------------------------------------------------------------------

def test_zone_status():
    st = LEGOLAS(box_with_inner()).census.zone_status()
    assert st['inlet']['status'] == 'boundary'
    assert st['inner'] == {'n_faces': NX * NY, 'interior_faces': NX * NY,
                           'status': 'interior'}


def test_check_bcs_finds_each_problem(files):
    lg = LEGOLAS(box_with_inner())
    r = lg.census.check_bcs(control=files[0], ini=files[1])
    errors = _levels(r['issues'], 'error')
    # inner zone missing from the control: HORSES3D stops
    assert any("'inner'" in m and 'not in the control' in m for m in errors)
    # inner zone with a type other than 100 in the ini: PyHOPE stops
    assert any("'inner'" in m and 'wants 100' in m for m in errors)
    warnings = _levels(r['issues'], 'warning')
    assert any("'ghost'" in m and 'not in the mesh' in m for m in warnings)
    infos = _levels(r['issues'], 'info')
    assert any("'inlet' is 'Inlet'" in m for m in infos)
    names = [row['name'] for row in r['rows']]
    assert names == ['inlet', 'walls', 'inner', 'ghost']
    assert r['rows'][0]['control'] == 'Inflow'
    assert r['rows'][2]['ini'] == (4, 0, 0, 0)


def test_check_bcs_clean_case(tmp_path):
    c = tmp_path / 'ok.control'
    c.write_text('#define boundary inlet\n type = Inflow\n#end\n'
                 '#define boundary walls__inner\n type = FreeSlipWall\n#end\n')
    i = tmp_path / 'ok.ini'
    i.write_text('BoundaryName = inlet\nBoundaryType = (/2,0,0,0/)\n'
                 'BoundaryName = walls\nBoundaryType = (/4,0,0,0/)\n'
                 'BoundaryName = inner\nBoundaryType = (/100,0,0,0/)\n')
    r = LEGOLAS(box_with_inner()).census.check_bcs(control=c, ini=i)
    assert r['issues'] == []


def test_check_bcs_type_100_on_the_skin_and_missing_ini(tmp_path):
    i = tmp_path / 'p.ini'
    i.write_text('BoundaryName = inlet\nBoundaryType = (/100,0,0,0/)\n')
    for fmt, level in (('cgns', 'error'), ('hopr', 'warning')):
        r = LEGOLAS(box_with_inner(fmt)).census.check_bcs(ini=i)
        assert any("type 100" in m for m in _levels(r['issues'], 'error'))
        missing = [m for m in _levels(r['issues'], level)
                   if "'walls'" in m and 'not in parameter.ini' in m]
        assert missing, (fmt, r['issues'])


def test_empty_zone_needs_nothing(tmp_path):
    mesh = box_with_inner()
    mesh.boundaries['fluid'] = Boundary('fluid', [], [])
    c = tmp_path / 'c.control'
    c.write_text('#define boundary inlet__walls__inner\n type = Inflow\n#end\n')
    r = LEGOLAS(mesh).census.check_bcs(control=c)
    assert _levels(r['issues'], 'error') == []
    assert any("'fluid'" in m for m in _levels(r['issues'], 'info'))


# ---------------------------------------------------------------------------
# watertight
# ---------------------------------------------------------------------------

def test_watertight_closed_box():
    w = LEGOLAS(box_with_inner()).census.watertight()
    assert w['closed'] and w['issues'] == []
    assert w['exterior_faces'] == 2 * (NX * NY + NY * NZ + NX * NZ)


def test_watertight_flags_unassigned_and_mixed():
    mesh = box_with_inner()
    walls = mesh.boundaries.pop('walls')
    # a BC that is half on the skin and half inner
    mesh.boundaries['inner'] = Boundary(
        'inner', np.concatenate([mesh.boundaries['inner'].elem,
                                 walls.elem[:2]]),
        np.concatenate([mesh.boundaries['inner'].face, walls.face[:2]]))
    w = LEGOLAS(mesh).census.watertight()
    assert w['unassigned_faces'] == len(walls) - 2
    assert w['mixed_boundaries'] == ['inner']
    assert any('no boundary' in m for m in _levels(w['issues'], 'error'))


def test_watertight_flags_cells_touching_along_an_edge():
    # two unit cubes sharing only the edge x = 1, y = 1 (nodes 6 and 7 of
    # the first are 0 and 3 of... the second, shifted by (1, 1, 0))
    p = []
    for dx, dy in ((0, 0), (1, 1)):
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    p.append((x + dx, y + dy, z))
    p = np.array(p, float)
    a = np.arange(8)
    b = np.arange(8, 16)
    b[[0, 4]] = [3, 7]                 # (1,1,0) and (1,1,1) are shared
    mesh = Mesh(p, [CellBlock('hex', 1, np.array([a, b]))])
    w = LEGOLAS(mesh).census.watertight()
    assert w['nonmanifold_edges'] == 1 and w['open_edges'] == 0


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

def test_duplicates_none_on_a_clean_box():
    d = LEGOLAS(box_mesh(order=2)).census.duplicates()
    assert d['groups'] == d['crack_faces'] == d['collapsed_cells'] == 0
    assert d['issues'] == []


def test_duplicates_high_order_only():
    mesh = box_mesh(NX, NY, NZ, order=2)
    conn = mesh.blocks[0].conn.copy()
    # cell 1 gets its own copy of the centre node of its face shared with
    # cell 0 (face 4, i = 0): coincident, but not a corner
    from FotR.characters.legolas.ordering import hex_face_grid
    node = conn[1, hex_face_grid(2, 4)[1, 1]]
    pts = np.vstack([mesh.points, mesh.points[node]])
    conn[1, hex_face_grid(2, 4)[1, 1]] = len(pts) - 1
    lg = LEGOLAS(Mesh(pts, [CellBlock('hex', 2, conn)], dict(mesh.boundaries)))
    d = lg.census.duplicates()
    assert (d['groups'], d['extra_nodes'], d['highorder_groups'],
            d['corner_groups'], d['crack_faces']) == (1, 1, 1, 0, 0)
    assert _levels(d['issues'], 'info')


def test_duplicates_crack_and_collapsed():
    mesh = box_mesh(NX, NY, NZ, order=1)
    conn = mesh.blocks[0].conn.copy()
    pts = mesh.points.copy()
    # cut the box along z = 2: cells above get their own copy of the nodes
    upper = np.arange(len(conn)) >= NX * NY * 2
    plane = np.unique(conn[upper][np.isclose(pts[conn[upper], 2], 2.0)])
    new = {int(n): len(pts) + k for k, n in enumerate(plane)}
    pts = np.vstack([pts, pts[plane]])
    for c in np.nonzero(upper)[0]:
        conn[c] = [new.get(int(n), n) for n in conn[c]]
    lg = LEGOLAS(Mesh(pts, [CellBlock('hex', 1, conn)]))
    d = lg.census.duplicates()
    assert d['crack_faces'] == 2 * NX * NY
    assert d['corner_groups'] == len(plane)
    assert any('on top of another' in m for m in _levels(d['issues'], 'error'))
    # collapse one cell: its corner 1 moved onto its corner 0
    pts2 = mesh.points.copy()
    c0 = mesh.blocks[0].conn[0]
    pts2[c0[1]] = pts2[c0[0]]
    d = LEGOLAS(Mesh(pts2, [CellBlock('hex', 1, mesh.blocks[0].conn)])
                ).census.duplicates()
    assert d['collapsed_cells'] >= 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_renumbered_and_other_order():
    a = box_mesh(NX, NY, NZ, order=1)
    # same mesh, cells in reverse order and nodes permuted
    rng = np.random.default_rng(0)
    perm = rng.permutation(a.n_points)
    inv = np.argsort(perm)
    b = Mesh(a.points[perm], [CellBlock('hex', 1,
                                        inv[a.blocks[0].conn[::-1]])],
             {k: Boundary(k, a.n_cells - 1 - v.elem, v.face)
              for k, v in a.boundaries.items()})
    c = LEGOLAS(a).census.compare(b)
    assert c['same_corners'] and c['issues'] == []
    # the same box at order 2: corners equal, orders differ
    c2 = LEGOLAS(a).census.compare(box_mesh(NX, NY, NZ, order=2))
    assert c2['same_corners'] and c2['orders'] == (1, 2)
    assert all(r['area_rel_diff'] < 1e-12 for r in c2['boundaries'])


def test_compare_flags_differences():
    a = box_mesh(NX, NY, NZ, order=1)
    b = box_mesh(NX, NY, NZ + 1, order=1)
    b.boundaries['extra'] = Boundary('extra', [], [])
    c = LEGOLAS(a).census.compare(b)
    assert not c['same_corners']
    errors = _levels(c['issues'], 'error')
    assert any('number of cells' in m for m in errors)
    assert c['only_there'] == ['extra']


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def test_check_collects_and_sorts(files):
    lg = LEGOLAS(box_with_inner())
    r = lg.census.check(control=files[0], ini=files[1])
    assert not r['ok']
    levels = [i['level'] for i in r['issues']]
    assert levels == sorted(levels, key=['error', 'warning', 'info'].index)
    assert set(r) >= {'watertight', 'bcs', 'duplicates', 'issues', 'ok'}
    text = lg.census.check_report(control=files[0], ini=files[1])
    assert 'ERRORS' in text and 'ghost' in text
    assert LEGOLAS(box_mesh()).census.check()['ok']


# ---------------------------------------------------------------------------
# real meshes (skipped when absent)
# ---------------------------------------------------------------------------

def _need(*paths):
    for p in paths:
        if not os.path.isfile(p):
            pytest.skip(f'{p} not available')


def test_turbine_files_belong_to_the_empty_mesh():
    fluid = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
    empty = os.path.join(TURBINE, 'Wind_turbine_disk_empty_cgns_g2_mesh.h5')
    control = os.path.join(TURBINE, 'wind_turbine.control')
    ini = os.path.join(TURBINE, 'parameter.ini')
    _need(fluid, empty, control, ini)
    r = LEGOLAS(fluid).census.check_bcs(control=control, ini=ini)
    assert any("'disk_side'" in m and 'wants 100' in m
               for m in _levels(r['issues'], 'error'))
    r = LEGOLAS(empty).census.check_bcs(control=control, ini=ini)
    assert _levels(r['issues'], 'error') == []
    assert _levels(r['issues'], 'warning') == []


def test_turbine_cgns_duplicates_and_orders():
    g2 = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_g2.cgns')
    h5 = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
    g4 = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_g4.cgns')
    _need(g2, h5, g4)
    d = LEGOLAS(g2).census.duplicates()
    assert d['extra_nodes'] == 2980 and d['crack_faces'] == 0
    assert d['corner_groups'] == 0
    lg = LEGOLAS(h5)
    assert lg.mesh.n_points + 2980 == LEGOLAS(g2).mesh.n_points
    c = lg.census.compare(g4)
    assert c['same_corners'] and c['orders'] == (2, 4)
