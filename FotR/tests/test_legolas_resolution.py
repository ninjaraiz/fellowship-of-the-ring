"""LEGOLAS L5: cell sizes, first cells at a wall, estimated wall units."""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS, Boundary, CellBlock, Mesh
from FotR.characters.legolas.resolution import (WRLES, Resolution,
                                                first_node_fraction)

from _legolas_meshes import box_mesh

SPHERE = os.path.expanduser('~/Documentos/COMAC/my_ESPHERE_1/outputs/'
                            'M_0.3_re_400/MESH/my_esphere_v2g2_mesh.h5')


def wall_box(order=1, nz=5, ratio=1.2, y1=0.01, nx=4, ny=3):
    """Box [0,nx]x[0,ny]x[0,H] with layers growing by *ratio* from z = 0,
    and the bottom face as BC 'wall'."""
    m = box_mesh(nx, ny, nz, order=order, with_bcs=False)
    heights = y1 * ratio ** np.arange(nz)
    edges = np.concatenate([[0.0], np.cumsum(heights)])
    z = m.points[:, 2]
    k = np.minimum(np.floor(z).astype(int), nz - 1)
    frac = z - k
    m.points[:, 2] = edges[k] + frac * heights[k]
    m._cache.clear()
    bottom = np.arange(nx * ny)                      # layer k = 0
    m.boundaries['wall'] = Boundary('wall', bottom, np.zeros(nx * ny, int))
    return m, heights


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('p,frac', [(1, 1.0), (2, 0.5),
                                    (3, (1 - 1 / np.sqrt(5)) / 2),
                                    (4, (1 - np.sqrt(3 / 7)) / 2)])
def test_first_node_fraction(p, frac):
    assert first_node_fraction(p) == pytest.approx(frac)


def test_skin_friction():
    cf = Resolution.skin_friction([1e6], 'turbulent')[0]
    assert cf == pytest.approx(0.026 * 1e6 ** (-1 / 7))
    assert Resolution.skin_friction([1e4], 'laminar')[0] == pytest.approx(
        0.664e-2)
    with pytest.raises(ValueError):
        Resolution.skin_friction([1.0], 'magic')


# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3])
def test_sizes_of_a_cube_and_a_stretched_box(order):
    s = LEGOLAS(box_mesh(2, 2, 2, order=order)).resolution.sizes(p=3)
    np.testing.assert_allclose(s['h'], 1.0)
    np.testing.assert_allclose(s['anisotropy'], 1.0)
    np.testing.assert_allclose(s['delta'], 0.25)
    m = box_mesh(1, 1, 1, order=order, with_bcs=False)
    m.points[:, 2] *= 8.0
    s = LEGOLAS(m).resolution.sizes()
    assert s['h'][0] == pytest.approx(2.0)
    assert s['h_min'][0] == pytest.approx(1.0)
    assert s['h_max'][0] == pytest.approx(8.0)
    assert 'delta' not in s


def test_directional_size_follows_curved_edges():
    # bend the order-2 cube: the mid-edge nodes at x = 0.5 move up by 0.2
    m = box_mesh(1, 1, 1, order=2, with_bcs=False)
    x, z = m.points[:, 0], m.points[:, 2]
    m.points[:, 2] = z + 0.2 * (x == 0.5) * (z == 0)
    s = LEGOLAS(m).resolution.sizes()
    arc = 2 * np.hypot(0.5, 0.2)
    # two of the four edges along x are bent
    assert s['h_max'][0] == pytest.approx((2 * arc + 2) / 4)


# ---------------------------------------------------------------------------
# wall
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2])
def test_wall_first_cell_growth_and_tangential(order):
    m, heights = wall_box(order=order)
    w = LEGOLAS(m).resolution.wall('wall', layers=5)
    np.testing.assert_allclose(w['y1'], heights[0])
    np.testing.assert_allclose(w['growth'], 1.2)
    np.testing.assert_allclose(w['dt1'], 1.0)
    np.testing.assert_allclose(w['dt2'], 1.0)
    np.testing.assert_allclose(w['normal'], [[0, 0, 1]] * len(w['y1']),
                               atol=1e-12)
    w = LEGOLAS(m).resolution.wall('wall', p=3, layers=7)
    np.testing.assert_allclose(w['y1_dof'], heights[0] * first_node_fraction(3))
    np.testing.assert_allclose(w['dt1'], 0.25)
    # the stack is 5 cells: the last steps leave the mesh
    assert np.isnan(w['growth'][:, 4:]).all()


def test_wall_on_tetrahedra_has_height_but_no_stack():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.3, 0.5]], float)
    m = Mesh(p, [CellBlock('tet', 1, np.array([[0, 1, 2, 3]]))],
             {'wall': Boundary('wall', [0], [0])})
    w = LEGOLAS(m).resolution.wall('wall', layers=3)
    assert w['y1'][0] == pytest.approx(0.5)
    assert np.isnan(w['growth']).all()


def test_prism_stack_is_followed():
    # two triangles extruded in z, three layers: wall on the bottom triangles
    tri = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    zs = [0.0, 0.1, 0.25, 0.475]                     # growth 1.5
    pts = np.array([[x, y, z] for z in zs for x, y in tri])
    cells = []
    for k in range(3):
        for a, b, c in ((0, 1, 2), (1, 3, 2)):
            lo, hi = 4 * k, 4 * (k + 1)
            # VTK prism: (0, 1, 2) clockwise seen from (3, 4, 5)
            cells.append([lo + a, lo + c, lo + b, hi + a, hi + c, hi + b])
    m = Mesh(pts, [CellBlock('prism', 1, np.array(cells))],
             {'wall': Boundary('wall', [0, 1], [0, 0])})
    w = LEGOLAS(m).resolution.wall('wall', layers=3)
    np.testing.assert_allclose(w['y1'], 0.1)
    np.testing.assert_allclose(w['growth'], 1.5)


# ---------------------------------------------------------------------------
# wall units
# ---------------------------------------------------------------------------

def test_yplus_matches_the_formula():
    m, heights = wall_box()
    Re = 1e6
    r = LEGOLAS(m).resolution.yplus('wall', Re=Re, L=2.0)
    cf = 0.026 * Re ** (-1 / 7)
    unit = Re * np.sqrt(cf / 2) / 2.0
    np.testing.assert_allclose(r['y1_plus'], heights[0] * unit)
    np.testing.assert_allclose(r['dt1_plus'], unit)
    assert r['summary']['y1_plus']['in_wrles'] in (0.0, 1.0)
    assert set(WRLES) <= set(r['summary']) | {'y1_plus'}


def test_yplus_local_reynolds_along_the_flow():
    m, _ = wall_box(nx=4)
    r = LEGOLAS(m).resolution.yplus('wall', Re=1e5, flow_dir=[1, 0, 0],
                                    correlation='laminar')
    x = r['centroid'][:, 0]
    np.testing.assert_allclose(r['re_x'], 1e5 * x)      # leading edge x = 0
    # thinner wall shear downstream: y+ falls along x
    order = np.argsort(x)
    assert (np.diff(r['y1_plus'][order]) <= 1e-12).all()


def test_report_and_fields():
    m, _ = wall_box()
    lg = LEGOLAS(m)
    text = lg.resolution.report(walls='wall', p=2, Re=1e6)
    assert 'y1_dof_plus' in text and 'WRLES' in text and 'growth' in text
    f = lg.resolution.fields(p=2, walls='wall', Re=1e6)
    assert set(f) == {'h', 'anisotropy', 'delta', 'y1', 'y1_plus'}
    y1 = f['y1']['values']
    assert np.isfinite(y1[:12]).all() and np.isnan(y1[12:]).all()


def test_fields_reach_the_viewer():
    m, _ = wall_box()
    lg = LEGOLAS(m)
    fields = lg.resolution.fields(p=2, walls='wall', Re=1e6)
    srv = lg.viewer.server(port=0, fields=fields).start()
    get = lambda p: urllib.request.urlopen(srv.url.rstrip('/') + p,
                                           timeout=20).read()
    try:
        names = [f['name'] for f in json.loads(get('/api/info'))['fields']]
        assert {'y1_plus', 'delta'} <= set(names)
        blob = get('/api/field?name=y1_plus')
        n = int.from_bytes(blob[:4], 'little')
        head = json.loads(blob[4:4 + n])
        assert head['better'] == 'low' and head['max'] > 0
        worst = json.loads(get('/api/worst?name=y1_plus&n=3'))
        assert len(worst) == 3 and worst[0]['cell'] < 12
        assert 'y1_plus' in json.loads(get('/api/cell?id=0'))['quality']
    finally:
        srv.stop()
    with pytest.raises(ValueError, match='values'):
        lg.viewer.server(port=0, fields={'bad': np.zeros(3)})
    with pytest.raises(ValueError, match='hide'):
        lg.viewer.server(port=0, fields={'volume': np.zeros(m.n_cells)})


def test_show_ships_fields_to_the_server_process():
    m, _ = wall_box()
    lg = LEGOLAS(m)
    h = lg.viewer.show(port=0, idle_timeout=60, timeout=60,
                       fields={'mine': {'values': np.arange(m.n_cells),
                                        'label': 'mine', 'better': 'high'}})
    try:
        with urllib.request.urlopen(h.url + 'api/info', timeout=10) as r:
            names = [f['name'] for f in json.loads(r.read())['fields']]
        assert 'mine' in names
    finally:
        h.stop()


def test_sphere_first_node():
    if not os.path.isfile(SPHERE):
        pytest.skip('sphere mesh not available')
    w = LEGOLAS(SPHERE).resolution.wall('esphere_surface', p=3)
    assert len(w['y1']) == 252
    assert 0.05 < np.median(w['y1']) < 0.2
    np.testing.assert_allclose(w['y1_dof'], w['y1'] * first_node_fraction(3))
