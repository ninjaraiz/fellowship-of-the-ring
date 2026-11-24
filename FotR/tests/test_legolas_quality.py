"""LEGOLAS L4: cell quality metrics and their route in the viewer."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS, CellBlock, Mesh
from FotR.characters.legolas.quality import METRICS, gauss_lobatto, lagrange

from _legolas_meshes import box_mesh

TURBINE = os.path.expanduser('~/Documentos/COMAC/wind_turbine_ansa/sources')
S3 = np.sqrt(3.0)


def one(points, kind, order=1):
    conn = np.arange(len(points))[None]
    return LEGOLAS(Mesh(np.asarray(points, float),
                        [CellBlock(kind, order, conn)])).quality


def mapped_box(order, f, nx=1, ny=1, nz=1):
    """``box_mesh`` with every node moved by ``f(x, y, z) -> (x, y, z)``."""
    m = box_mesh(nx, ny, nz, order=order, with_bcs=False)
    m.points = np.stack(f(*m.points.T), 1)
    return m


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_gauss_lobatto_integrates_degree_2n_minus_1(n):
    x, w = gauss_lobatto(n)
    assert x[0] == -1 and x[-1] == 1 and np.allclose(x, -x[::-1])
    for k in range(2 * n):
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert np.dot(w, x ** k) == pytest.approx(exact, abs=1e-12)


def test_lagrange_reproduces_polynomials_and_derivatives():
    nodes = np.linspace(-1, 1, 4)
    x = np.linspace(-1, 1, 7)
    L, D = lagrange(nodes, x)
    p = lambda t: 2 * t ** 3 - t + 0.5
    dp = lambda t: 6 * t ** 2 - 1
    np.testing.assert_allclose(L @ p(nodes), p(x), atol=1e-12)
    np.testing.assert_allclose(D @ p(nodes), dp(x), atol=1e-12)


# ---------------------------------------------------------------------------
# ideal and known cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_cube_is_ideal_at_every_order(order):
    q = LEGOLAS(box_mesh(2, 2, 2, order=order)).quality.metrics()
    for name, ideal in (('scaled_jacobian', 1), ('jacobian_ratio', 1),
                        ('aspect_ratio', 1), ('skewness', 0),
                        ('non_orthogonality', 0), ('volume_ratio', 1),
                        ('volume', 1)):
        np.testing.assert_allclose(q[name], ideal, atol=1e-10, err_msg=name)


@pytest.mark.parametrize('kind,points,volume', [
    ('tet', [[0, 0, 0], [1, 0, 0], [.5, S3 / 2, 0],
             [.5, S3 / 6, np.sqrt(2 / 3)]], np.sqrt(2) / 12),
    # VTK prism: the normal of (0, 1, 2) points away from (3, 4, 5)
    ('prism', [[0, 0, 0], [.5, S3 / 2, 0], [1, 0, 0], [0, 0, 1],
               [.5, S3 / 2, 1], [1, 0, 1]], S3 / 4),
    ('pyramid', [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [.5, .5, np.sqrt(.5)]], np.sqrt(.5) / 3),
])
def test_regular_linear_cells_score_one(kind, points, volume):
    q = one(points, kind).metrics()
    assert q['scaled_jacobian'][0] == pytest.approx(1.0)
    assert q['skewness'][0] == pytest.approx(0.0, abs=1e-12)
    assert q['volume'][0] == pytest.approx(volume)
    assert np.isnan(q['jacobian_ratio'][0])        # hexahedra only


def test_sheared_hex_is_affine():
    m = mapped_box(1, lambda x, y, z: (x + 0.5 * y, y, z))
    q = LEGOLAS(m).quality.metrics()
    assert q['jacobian_ratio'][0] == pytest.approx(1.0)
    assert q['scaled_jacobian'][0] == pytest.approx(1 / np.sqrt(1.25))
    assert q['volume'][0] == pytest.approx(1.0)


@pytest.mark.parametrize('order', [1, 2, 3])
def test_non_affine_volume_is_exact(order):
    # z' = z (1 + 0.2 x): det J = 1 + 0.2 x, volume 1.1, ratio 1/1.2
    m = mapped_box(order, lambda x, y, z: (x, y, z * (1 + 0.2 * x)))
    q = LEGOLAS(m).quality.metrics()
    assert q['volume'][0] == pytest.approx(1.1)
    assert q['jacobian_ratio'][0] == pytest.approx(1 / 1.2)


def test_hexahedron_cut_from_a_regular_tetrahedron():
    # what the my_ESPHERE_1 mesh is made of: valid, but far from affine
    T = np.array([[0, 0, 0], [1, 0, 0], [.5, S3 / 2, 0],
                  [.5, S3 / 6, np.sqrt(2 / 3)]])
    v0, v1, v2, v3 = T
    m = lambda *p: np.mean(p, 0)
    vtk = np.array([v0, m(v0, v1), m(v0, v1, v2), m(v0, v2), m(v0, v3),
                    m(v0, v1, v3), m(v0, v1, v2, v3), m(v0, v2, v3)])
    q = one(vtk[[0, 1, 3, 2, 4, 5, 7, 6]], 'hex').metrics()
    assert q['jacobian_ratio'][0] == pytest.approx(0.0741, abs=1e-4)
    assert q['scaled_jacobian'][0] == pytest.approx(np.sqrt(.5))
    assert q['skewness'][0] == pytest.approx(1 / 3)
    assert METRICS['jacobian_ratio']['warn'] < 0.074    # no warning for it


def test_inverted_cell_is_negative_and_an_error():
    m = box_mesh(3, 1, 1, order=1, with_bcs=False)
    conn = m.blocks[0].conn.copy()
    conn[1] = conn[1][[1, 0, 3, 2, 5, 4, 7, 6]]       # mirror cell 1 in x
    lg = LEGOLAS(Mesh(m.points, [CellBlock('hex', 1, conn)]))
    q = lg.quality.metrics()
    assert q['scaled_jacobian'][1] < 0 and q['jacobian_ratio'][1] < 0
    assert (q['scaled_jacobian'][[0, 2]] > 0).all()
    errors = [i for i in lg.quality.issues() if i['level'] == 'error']
    assert errors and 'inverted' in errors[0]['message']
    assert lg.quality.worst('scaled_jacobian', 1)[0]['cell'] == 1


def test_left_handed_blocks_are_turned_round():
    m = box_mesh(2, 1, 1, order=1, with_bcs=False)
    conn = m.blocks[0].conn[:, [1, 0, 3, 2, 5, 4, 7, 6]]   # all mirrored
    lg = LEGOLAS(Mesh(m.points, [CellBlock('hex', 1, conn)]))
    q = lg.quality.metrics()
    np.testing.assert_allclose(q['scaled_jacobian'], 1.0)
    assert lg.quality.meta['flipped'] == ['hex1']
    assert any(i['level'] == 'info' for i in lg.quality.issues())
    # gmsh-ordered prisms (normal of (0,1,2) towards (3,4,5)) score 1 too
    pr = [[0, 0, 0], [1, 0, 0], [.5, S3 / 2, 0], [0, 0, 1], [1, 0, 1],
          [.5, S3 / 2, 1]]
    q = one(pr, 'prism')
    assert q.metrics()['scaled_jacobian'][0] == pytest.approx(1.0)
    assert q.meta['flipped'] == ['prism1']


def test_aspect_ratio_skewness_and_neighbours():
    m = mapped_box(1, lambda x, y, z: (x, y, 4 * z))
    assert LEGOLAS(m).quality.metrics()['aspect_ratio'][0] == pytest.approx(4)
    # a rhombus face: 60/120 degrees -> skewness 30/90
    m = mapped_box(1, lambda x, y, z: (x + y * np.cos(np.pi / 3),
                                       y * np.sin(np.pi / 3), z))
    assert LEGOLAS(m).quality.metrics()['skewness'][0] == pytest.approx(1 / 3)
    # two cells stacked in z and sheared: centroids 26.57° off the normal
    m = mapped_box(1, lambda x, y, z: (x + 0.5 * z, y, z), nz=2)
    q = LEGOLAS(m).quality.metrics()
    np.testing.assert_allclose(q['non_orthogonality'],
                               np.degrees(np.arctan(0.5)))
    # heights 1 and 3: volume jump 3 on both sides
    m = mapped_box(1, lambda x, y, z: (x, y, np.where(z > 1.5, 3 * z - 2, z)),
                   nz=2)
    np.testing.assert_allclose(LEGOLAS(m).quality.metrics()['volume_ratio'],
                               3.0)


def test_summary_worst_report():
    q = LEGOLAS(mapped_box(2, lambda x, y, z: (x + 0.3 * y * y, y, z),
                           nx=3, ny=3, nz=2)).quality
    s = q.summary()
    assert set(s) == set(METRICS)
    assert s['scaled_jacobian']['min'] <= s['scaled_jacobian']['p50']
    w = q.worst('scaled_jacobian', 3)
    vals = [x['value'] for x in w]
    assert vals == sorted(vals) and len(w) == 3
    w = q.worst('aspect_ratio', 2)
    assert w[0]['value'] >= w[1]['value']
    text = q.report()
    assert 'scaled_jacobian' in text and '#' in text
    with pytest.raises(KeyError):
        q.field('beauty')


# ---------------------------------------------------------------------------
# viewer route
# ---------------------------------------------------------------------------

def _unpack(payload: bytes):
    n = int.from_bytes(payload[:4], 'little')
    head = json.loads(payload[4:4 + n])
    arr = {k: np.frombuffer(payload, a['dtype'], a['count'], a['offset'])
           for k, a in head['arrays'].items()}
    return head, arr


def test_viewer_field_and_worst_routes():
    lg = LEGOLAS(mapped_box(2, lambda x, y, z: (x + 0.3 * y * y, y, z),
                            nx=3, ny=3, nz=2))
    srv = lg.viewer.server(port=0).start()
    get = lambda p: urllib.request.urlopen(srv.url.rstrip('/') + p,
                                           timeout=20).read()
    try:
        info = json.loads(get('/api/info'))
        assert [f['name'] for f in info['fields']] == list(METRICS)
        assert 'quality' not in json.loads(get('/api/cell?id=0'))
        head, arr = _unpack(get('/api/field?name=scaled_jacobian'))
        assert head['better'] == 'high' and len(arr['values']) == 18
        np.testing.assert_allclose(arr['values'],
                                   lg.quality.field('scaled_jacobian'),
                                   rtol=1e-6)
        worst = json.loads(get('/api/worst?name=scaled_jacobian&n=4'))
        assert len(worst) == 4
        cell = json.loads(get('/api/cell?id=0'))
        assert set(cell['quality']) == set(METRICS)
        with pytest.raises(urllib.error.HTTPError) as exc:
            get('/api/field?name=beauty')
        assert exc.value.code == 404
    finally:
        srv.stop()


# ---------------------------------------------------------------------------
# real meshes
# ---------------------------------------------------------------------------

def test_turbine_matches_pyhope():
    """PyHOPE reported a minimum scaled Jacobian of 0.4-0.5 for the g2 and g4
    meshes of the turbine, with no negative cells (17/11/2026)."""
    files = [os.path.join(TURBINE, f'Wind_turbine_disk_fluid_cgns_g{g}_mesh.h5')
             for g in (2, 4)]
    for f in files:
        if not os.path.isfile(f):
            pytest.skip(f'{f} not available')
        s = LEGOLAS(f).quality.summary()
        assert 0.4 < s['jacobian_ratio']['min'] < 0.5
        assert s['scaled_jacobian']['error'] == 0
