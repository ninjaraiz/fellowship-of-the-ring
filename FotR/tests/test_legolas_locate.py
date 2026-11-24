"""LEGOLAS L6: point location and HORSES3D's actuator-disk search."""

from __future__ import annotations

import os

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS, CellBlock, Mesh
from FotR.characters.legolas import ordering
from FotR.characters.legolas.locate import read_actuator_points
from FotR.characters.legolas.quality import lagrange

from _legolas_meshes import box_mesh

TURBINE = os.path.expanduser('~/Documentos/COMAC/wind_turbine_ansa/sources')


def mapping(m, cell, xi):
    blk = m.blocks[0]
    n = blk.order + 1
    X = m.points[blk.conn[cell]].reshape(n, n, n, 3)
    nodes = np.linspace(-1, 1, n)
    L = [lagrange(nodes, [v])[0][0] for v in xi]
    return np.einsum('k,j,i,kjid->d', L[2], L[1], L[0], X)


@pytest.mark.parametrize('order', [1, 2, 4])
def test_points_in_a_curved_box(order):
    m = box_mesh(3, 2, 2, order=order)
    x, y, z = m.points.T
    m.points = np.stack([x + 0.05 * np.sin(np.pi * y), y, z], 1)
    m._cache.clear()
    lg = LEGOLAS(m)
    rng = np.random.default_rng(0)
    cells = rng.integers(0, m.n_cells, 40)
    xi = rng.uniform(-0.95, 0.95, (40, 3))
    pts = np.array([mapping(m, c, r) for c, r in zip(cells, xi)])
    r = lg.locate.points(pts)
    np.testing.assert_array_equal(r['cell'], cells)
    np.testing.assert_allclose(r['xi'], xi, atol=1e-9)
    out = lg.locate.points([[10.0, 0, 0], [-1, -1, -1]])
    assert (out['cell'] == -1).all() and (out['distance'] > 0).all()
    assert np.isnan(out['xi']).all()


def test_points_in_tetrahedra():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                 float)
    m = Mesh(p, [CellBlock('tet', 1, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))])
    r = LEGOLAS(m).locate.points([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6],
                                  [2, 2, 2]])
    assert r['cell'].tolist() == [0, 1, -1]


def bulged_pair():
    """Two order-2 cubes along x whose shared face bulges to x = 1.2."""
    m = box_mesh(2, 1, 1, order=2, with_bcs=False)
    centre = m.blocks[0].conn[0][ordering.tensor_index(2, 1, 1, 2)]
    m.points[centre, 0] = 1.2
    m._cache.clear()
    return m


def test_horses3d_ad_loses_a_point_between_chord_and_arc():
    m = bulged_pair()
    lg = LEGOLAS(m)
    p = [[1.1, 0.5, 0.5], [0.5, 0.5, 0.5], [1.8, 0.5, 0.5]]
    r = lg.locate.horses3d_ad(p, nranks=1)
    assert r['honest'].tolist() == [0, 0, 1]
    assert r['found'].tolist() == [False, True, True]
    assert r['lost'].tolist() == [0]
    assert r['max_xi'][0] > 1
    # with 'mesh inner curves = .false.' the trilinear answer stands, and
    # these cells have no boundary face (no BCs), so the point is taken
    r = lg.locate.horses3d_ad(p, nranks=2, inner_curves=False)
    assert r['found'].all() and r['cell'][0] == 1


def test_read_actuator_points(tmp_path):
    head = ['#'] * 6 + ['1.0 2.0 3.0', '#', '1 0 0', '#', '0.5', '#', '0.05',
                        '#', '3 4']
    a = head + ['#', '# MODEL A', '#', '#', '0.1 -50 0', '0.2 -50 0',
                '0.3 -50 0', '#', '#', '#', '#', '2.5']
    path = tmp_path / 'a.dat'
    path.write_text('\n'.join(a) + '\n')
    r = read_actuator_points(path)
    assert r['model'] == 'A' and r['radii'].tolist() == [0.1, 0.2, 0.3]
    assert r['points'].shape == (12, 3) and r['ij'][4].tolist() == [2, 2]
    np.testing.assert_allclose(r['points'][0], [1.0, 2.1, 3.0])
    np.testing.assert_allclose(r['points'][4], [1.0, 2.0, 3.2])   # θ = 90°
    r = read_actuator_points(path, x=5.0, n_azimuthal=2)
    assert r['points'].shape == (6, 3) and (r['points'][:, 0] == 5.0).all()
    # model B: number of blades, 'omega pitch' and comments before the table
    b = head + ['#', '# MODEL B', '#', '3', '#', '10 0', '#', '#', '#',
                '0.1 0.05 10 2', 'p1.dat', 'p2.dat',
                '0.2 0.04 8 1', 'p1.dat', '0.3 0.03 6 1', 'p1.dat']
    path = tmp_path / 'b.dat'
    path.write_text('\n'.join(b) + '\n')
    r = read_actuator_points(path)
    assert r['model'] == 'B' and r['radii'].tolist() == [0.1, 0.2, 0.3]


def test_turbine_actuator_stations():
    mesh = os.path.join(TURBINE, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
    dat = os.path.join(TURBINE, 'ActuatorDef', 'Act_ActuatorDiskDef.dat')
    for f in (mesh, dat):
        if not os.path.isfile(f):
            pytest.skip(f'{f} not available')
    lg = LEGOLAS(mesh)
    st = read_actuator_points(dat)
    r = lg.locate.horses3d_ad(st['points'], nranks=4)
    assert r['found'].all() and len(r['lost']) == 0
    # last station moved to r = 0.446, beyond the safe 0.444: 48 lost,
    # what check_ad_points.py reports for the same case
    radii = st['radii'].copy()
    radii[-1] = 0.446
    th = np.arange(st['n_azimuthal']) * 2 * np.pi / st['n_azimuthal']
    h = st['hub']
    pts = np.array([[h[0], h[1] + rr * np.cos(t), h[2] + rr * np.sin(t)]
                    for t in th for rr in radii])
    r = lg.locate.horses3d_ad(pts, nranks=4)
    assert len(r['lost']) == 48
    assert np.nanmax(r['max_xi']) == pytest.approx(1.048558, abs=1e-6)
