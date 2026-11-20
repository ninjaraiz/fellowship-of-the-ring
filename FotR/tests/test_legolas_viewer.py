"""LEGOLAS L2: viewer geometry (scene), binary packing, HTTP server, launcher.

The browser side (WebGL) is not tested here: it needs a display. It was
checked by hand in a headless Chromium on an Xvnc display (see
``doc/PLAN_LEGOLAS.md``, L2). Everything the page asks the server for is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS, Mesh, CellBlock, Boundary
from FotR.characters.legolas import scene, topology
from FotR.characters.legolas import viewer as viewer_mod

STATIC = os.path.join(os.path.dirname(viewer_mod.__file__), 'static', 'three')


def box_mesh(nx=3, ny=2, nz=4, order=2, with_bcs=True):
    """Structured box [0,nx]x[0,ny]x[0,nz] of hexahedra of *order*."""
    n = order
    gx, gy, gz = (np.linspace(0, m, m * n + 1) for m in (nx, ny, nz))
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing='ij')
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    sx, sy = nx * n + 1, ny * n + 1
    conn = []
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                kk, jj, ii = np.meshgrid(*(np.arange(n + 1),) * 3,
                                         indexing='ij')
                conn.append(((ei * n + ii) + sx * (ej * n + jj)
                             + sx * sy * (ek * n + kk)).ravel())
    mesh = Mesh(pts, [CellBlock('hex', order, np.array(conn))],
                zones={0: 'fluid'}, meta={'format': 'test'})
    if with_bcs:
        elem, face = topology.exterior_faces(mesh)
        b, local = mesh.locate_cell(elem)
        x = mesh.points[mesh.blocks[0].conn[local][:, 0], 0]
        inlet = (face == 4) & (x == 0)
        mesh.boundaries['inlet'] = Boundary('inlet', elem[inlet], face[inlet],
                                            'BCInflow')
        mesh.boundaries['walls'] = Boundary('walls', elem[~inlet],
                                            face[~inlet], 'BCWall')
    return mesh


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 3])
def test_face_geometry_counts_and_outward_normals(order):
    mesh = box_mesh(order=order)
    elem, face = scene.zone_faces(mesh, scene.EXTERIOR)
    g = scene.face_geometry(mesh, elem, face)
    nf = len(elem)
    assert g['indices'].shape == (nf * 2 * order ** 2, 3)
    assert g['edges'].shape == (nf * 4 * order, 2, 3)
    assert g['positions'].dtype == np.float32
    assert g['tri_cell'].max() < mesh.n_cells
    # every triangle of the box's skin points out of the box
    p = g['positions'].astype(float) + scene.origin(mesh)
    a, b, c = (p[g['indices'][:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    centre = mesh.points.mean(0)
    assert (np.einsum('ij,ij->i', normal, (a + b + c) / 3 - centre) > 0).all()


def test_face_geometry_is_relative_to_origin():
    mesh = box_mesh()
    mesh.points += 1000.0
    mesh._cache.clear()
    g = scene.face_geometry(mesh, *scene.zone_faces(mesh, 'inlet'))
    np.testing.assert_allclose(scene.origin(mesh), [1001.5, 1001.0, 1002.0])
    assert np.abs(g['positions']).max() <= 2.0 + 1e-6


def test_decimation_respects_the_budget():
    mesh = box_mesh(order=2)
    elem, face = scene.zone_faces(mesh, scene.EXTERIOR)
    g = scene.face_geometry(mesh, elem, face, max_triangles=50)
    assert g['stride'] > 1 and len(g['indices']) <= 50 + 8


def test_zone_list_and_pseudo_zones():
    mesh = box_mesh()
    names = [z['name'] for z in scene.zone_list(mesh)]
    assert names == ['inlet', 'walls']                 # nothing unassigned
    del mesh.boundaries['walls']
    mesh._cache.clear()
    zl = {z['name']: z for z in scene.zone_list(mesh)}
    assert scene.UNASSIGNED in zl
    assert zl[scene.UNASSIGNED]['n_faces'] == 2 * (3 * 2 + 2 * 4 + 3 * 4) - 8
    bare = box_mesh(with_bcs=False)
    assert [z['name'] for z in scene.zone_list(bare)] == [scene.EXTERIOR]
    with pytest.raises(KeyError):
        scene.zone_faces(mesh, 'nope')


def test_section_slice_and_clip_on_a_box():
    nx, ny, nz = 3, 2, 4
    mesh = box_mesh(nx, ny, nz, order=1)
    # a plane inside layer k = 1 crosses one layer of cells
    elem, face = scene.section_faces(mesh, 2, 1.5, 'slice')
    assert len(np.unique(elem)) == nx * ny
    assert len(elem) == 2 * nx * ny + 2 * (nx + ny)
    # clip keeps z < 2.1: centroids 0.5 and 1.5; only the opened faces come
    elem, face = scene.section_faces(mesh, 2, 2.1, 'clip', side=-1)
    assert len(elem) == nx * ny and set(face) == {5}
    elem, face = scene.section_faces(mesh, 2, 2.1, 'clip', side=+1)
    assert len(elem) == nx * ny and set(face) == {0}
    with pytest.raises(ValueError):
        scene.section_faces(mesh, 2, 1.0, 'bogus')


def test_cell_info_neighbours_and_boundaries():
    mesh = box_mesh(3, 2, 4, order=2)
    info = scene.cell_info(mesh, 0)                     # corner cell
    by_face = {f['face']: f for f in info['faces']}
    assert by_face[4]['boundary'] == 'inlet' and by_face[4]['neighbour'] is None
    assert by_face[0]['boundary'] == 'walls'
    assert by_face[2]['neighbour'] == 1                  # +x neighbour
    assert by_face[3]['neighbour'] == 3                  # +y neighbour
    assert by_face[5]['neighbour'] == 6                  # +z neighbour
    assert info['kind'] == 'hex' and info['order'] == 2
    assert len(info['outline']) == 12 * 2               # 12 edges, 2 segments
    np.testing.assert_allclose(info['centroid'], [0.5, 0.5, 0.5])
    # neighbourhood is symmetric
    other = scene.cell_info(mesh, 1)
    assert {f['neighbour'] for f in other['faces']} >= {0}
    with pytest.raises(ValueError):
        scene.cell_info(mesh, mesh.n_cells)


def _unpack(payload: bytes):
    n = int.from_bytes(payload[:4], 'little')
    head = json.loads(payload[4:4 + n])
    arrays = {k: np.frombuffer(payload, a['dtype'], a['count'], a['offset'])
              .reshape(a['shape']) for k, a in head['arrays'].items()}
    return head, arrays


def test_pack_round_trip_and_alignment():
    arrays = {'a': np.arange(7, dtype=np.float32).reshape(7, 1),
              'b': np.arange(5, dtype=np.uint32)}
    blob = scene.pack({'zone': 'x' * 37}, arrays)
    head, back = _unpack(blob)
    assert head['zone'] == 'x' * 37
    for k in arrays:
        assert head['arrays'][k]['offset'] % 4 == 0
        np.testing.assert_array_equal(back[k], arrays[k])
    with pytest.raises(TypeError):
        scene.pack({}, {'c': np.zeros(3)})               # float64 is not sent


def test_mesh_save_load_round_trip(tmp_path):
    mesh = box_mesh()
    mesh.meta['file_cell_index'] = np.arange(mesh.n_cells)[::-1].copy()
    path = tmp_path / 'm.npz'
    mesh.save(path)
    back = Mesh.load(path)
    np.testing.assert_array_equal(back.points, mesh.points)
    np.testing.assert_array_equal(back.blocks[0].conn, mesh.blocks[0].conn)
    assert list(back.boundaries) == ['inlet', 'walls']
    assert back.boundaries['inlet'].bc_type == 'BCInflow'
    np.testing.assert_array_equal(back.meta['file_cell_index'],
                                  mesh.meta['file_cell_index'])
    assert back.zones == {0: 'fluid'}


# ---------------------------------------------------------------------------
# HTTP server, in this process
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    srv = LEGOLAS(box_mesh()).viewer.server(port=0).start()
    yield srv
    srv.stop()


def _get(srv, path):
    with urllib.request.urlopen(srv.url.rstrip('/') + path, timeout=10) as r:
        return r.status, r.headers.get('Content-Type'), r.read()


def test_server_routes(server):
    code, ctype, page = _get(server, '/')
    assert code == 200 and 'text/html' in ctype
    text = page.decode()
    assert '/static/three/three.module.min.js' in text
    # nothing is fetched from the internet
    assert not re.search(r'https?://', text)
    info = json.loads(_get(server, '/api/info')[2])
    assert info['n_cells'] == 24 and info['ngeo'] == 2
    assert [z['name'] for z in info['zones']] == ['inlet', 'walls']
    assert all(z['visible'] for z in info['zones'])
    head, arr = _unpack(_get(server, '/api/surface?zone=inlet')[2])
    assert head['zone'] == 'inlet' and head['n_faces'] == 8
    assert arr['indices'].shape == (8 * 8, 3)
    head, arr = _unpack(_get(server, '/api/section?axis=2&value=1.5'
                                     '&mode=slice&side=-1')[2])
    assert head['n_faces'] == 2 * 6 + 2 * 5
    cell = json.loads(_get(server, '/api/cell?id=0')[2])
    assert cell['cell'] == 0
    assert json.loads(_get(server, '/api/ping')[2]) == {'ok': True}
    code, ctype, js = _get(server, '/static/three/three.module.min.js')
    assert ctype == 'text/javascript' and len(js) > 500_000


@pytest.mark.parametrize('path,code', [
    ('/api/surface?zone=nope', 404),
    ('/api/cell?id=999', 400),
    ('/api/section?axis=7', 400),
    ('/api/section?axis=0&mode=bogus', 400),
    ('/static/../viewer.py', 404),
    ('/static/three/../../viewer.py', 404),
    ('/etc/passwd', 404),
])
def test_server_rejects(server, path, code):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, path)
    assert exc.value.code == code


def test_big_zones_start_hidden():
    srv = LEGOLAS(box_mesh()).viewer.server(port=0, max_triangles=100)
    try:
        vis = {z['name']: z['visible'] for z in srv.info['zones']}
        assert vis == {'inlet': True, 'walls': False}
    finally:
        srv.stop()


def test_idle_timeout_stops_the_server():
    srv = LEGOLAS(box_mesh()).viewer.server(port=0, idle_timeout=1.0).start()
    url = srv.url
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + 'api/ping', timeout=1)
        except Exception:
            break
        time.sleep(1.5)        # longer than the timeout: no keep-alive
    else:
        srv.stop()
        pytest.fail('the server did not stop by itself')
    srv.stop()


# ---------------------------------------------------------------------------
# launcher (separate process)
# ---------------------------------------------------------------------------

def test_show_from_file_and_from_memory(tmp_path):
    path = os.path.join(os.path.dirname(__file__), 'data', 'legolas',
                        'hex_p3_v22.msh')
    for lg in (LEGOLAS(path), LEGOLAS(box_mesh())):
        h = lg.viewer.show(port=0, idle_timeout=60, timeout=60)
        try:
            assert h.alive()
            with urllib.request.urlopen(h.url + 'api/info', timeout=10) as r:
                info = json.loads(r.read())
            assert info['n_cells'] == lg.mesh.n_cells
            assert 'LEGOLAS viewer' in repr(h) and h.url in h._repr_html_()
        finally:
            h.stop()
        assert not h.alive()
    # the temporary npz of the in-memory mesh is gone
    leftovers = [f for f in os.listdir(viewer_mod._CACHE_DIR)
                 if f.endswith('.npz') and f'-{os.getpid()}-' in f]
    assert leftovers == []


def test_launch_reports_a_failing_server(tmp_path):
    with pytest.raises(RuntimeError, match='exited'):
        viewer_mod.launch({'path': str(tmp_path / 'missing.msh')}, port=0,
                          timeout=60)


# ---------------------------------------------------------------------------
# vendored three.js
# ---------------------------------------------------------------------------

def test_vendored_three_matches_its_recorded_hashes():
    with open(os.path.join(STATIC, 'VERSION')) as fh:
        version = fh.read()
    assert '0.160.0' in version
    with open(os.path.join(STATIC, 'LICENSE')) as fh:
        assert 'MIT License' in fh.read()
    recorded = dict((name, sha) for sha, name in
                    re.findall(r'^\s+([0-9a-f]{64})\s+(\S+)$', version, re.M))
    assert set(recorded) == {'three.module.min.js',
                             'addons/controls/OrbitControls.js'}
    for name, sha in recorded.items():
        with open(os.path.join(STATIC, name), 'rb') as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha, name
