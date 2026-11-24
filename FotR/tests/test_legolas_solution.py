"""LEGOLAS: HORSES3D solutions on the mesh (lg.solution) and in the viewer."""

from __future__ import annotations

import glob
import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest

from FotR.characters.legolas import LEGOLAS
from FotR.characters.legolas.quality import lagrange
from FotR.characters.legolas.solution import (read_control_keys,
                                              solution_nodes)

from _legolas_meshes import box_mesh
from test_horses3d_hsol import REFS, write_horses_file, write_stats_file

MARINO = os.path.expanduser('~/Documentos/COMAC/wind_turbine_marino2024/'
                            'model_A/outputs/marino_A_tsr_6.0')
SPHERE = os.path.expanduser('~/Documentos/COMAC/my_ESPHERE_1/outputs/'
                            'M_0.3_re_200')
NX, NY, NZ = 3, 2, 2


def write_box_solution(path, p, node_type, fn, n_vars=3, nx=NX, ny=NY,
                       nz=NZ):
    """A .hsol of the unit-cell box: q[var, i, j, k] = fn(x, y, z)[var] at
    the solution nodes (the box is affine, so they are known exactly)."""
    xs = (solution_nodes(p, node_type) + 1) / 2
    q = []
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                X, Y, Z = np.meshgrid(ei + xs, ej + xs, ek + xs, indexing='ij')
                q.append(np.stack(fn(X, Y, Z))[:n_vars])
    return write_horses_file(path, 2, q, node_type=node_type)


def linear(X, Y, Z):
    # as conservative variables: rho, rho u, rho v, rho w, rho E (rho E
    # large enough for a positive pressure)
    return [1 + 0.1 * X - 0.2 * Y + 0.3 * Z, 2 * X + Y, Z - X,
            0.5 + 0 * X, 3000 + X * 0]


@pytest.mark.parametrize('node_type', [1, 2])
@pytest.mark.parametrize('p,ngeo', [(2, 1), (3, 2), (4, 3), (2, 4)])
def test_raw_fields_are_evaluated_at_the_mesh_nodes(tmp_path, node_type, p,
                                                    ngeo):
    path = write_box_solution(tmp_path / 's.hsol', p, node_type, linear)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=ngeo, with_bcs=False))
    sol = lg.solution.load(path)
    assert sol.variables == ['q0', 'q1', 'q2'] and sol.p == p
    m = lg.mesh
    x = m.points[m.blocks[0].conn]                     # (cells, nodes, 3)
    exp = linear(x[..., 0], x[..., 1], x[..., 2])
    for k in range(3):
        np.testing.assert_allclose(sol.node_field(f'q{k}'), exp[k],
                                   atol=1e-12)
    # a linear field has the value of the cell centre as its mean
    ctr = x.mean(1)
    np.testing.assert_allclose(sol.cell_field('q1'),
                               2 * ctr[:, 0] + ctr[:, 1], atol=1e-12)
    # shared nodes: one value per mesh node for exports
    np.testing.assert_allclose(sol.point_field('q0'),
                               linear(*m.points.T)[0], atol=1e-12)


def test_primitive_variables(tmp_path):
    # rho, rho u, rho v, rho w, rho E with rho linear: rho stays exact
    path = write_box_solution(tmp_path / 's.hsol', 3, 1, linear, n_vars=5)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2, with_bcs=False))
    sol = lg.solution.load(path)
    assert set(sol.variables) == {'rho', 'u', 'v', 'w', 'p', 'T', 'Mach',
                                  'Cp', 'Vmag'}
    x = lg.mesh.points[lg.mesh.blocks[0].conn]
    np.testing.assert_allclose(sol.node_field('rho'),
                               linear(x[..., 0], x[..., 1], x[..., 2])[0],
                               atol=1e-12)
    s = sol.summary()
    assert s['p'] == 3 and s['node_type'] == 'Gauss' and 'rho' in s['range']
    si = LEGOLAS(lg.mesh).solution.load(path, physical=True)
    np.testing.assert_allclose(si.node_field('rho'),
                               sol.node_field('rho') * REFS[3], rtol=1e-12)


def test_wrong_solution_is_refused(tmp_path):
    path = write_box_solution(tmp_path / 's.hsol', 2, 2, linear, nx=2)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2, with_bcs=False))
    with pytest.raises(ValueError, match='not a solution of this mesh'):
        lg.solution.load(path)
    with pytest.raises(RuntimeError):
        lg.solution.node_field('q0')
    assert 'none loaded' in repr(lg.solution)


def test_from_control(tmp_path):
    from test_legolas_io import write_hopr, _grid, _structured_faces
    import test_legolas_io as io_tests
    pts, conn = _grid(io_tests.NX, io_tests.NY, io_tests.NZ, 2)
    (tmp_path / 'MESH').mkdir()
    (tmp_path / 'RESULTS').mkdir()
    write_hopr(tmp_path / 'MESH' / 'box_g2_mesh.h5', pts, conn, 2,
               _structured_faces(io_tests.NX, io_tests.NY, io_tests.NZ))
    n = len(conn)
    q = [np.full((3, 3, 3, 3), float(e)) for e in range(n)]   # q = element
    write_horses_file(tmp_path / 'RESULTS' / 'box_g2p2.hsol', 2, q)
    control = tmp_path / 'case.control'
    control.write_text('Flow equations = "NS"\n'
                       'mesh file name = "MESH/box_g2_mesh.h5"  ! comment\n'
                       'Polynomial order = 2\n'
                       'solution file name = "RESULTS/box_g2p2.hsol"\n')
    assert read_control_keys(control)['polynomial order'] == '2'
    lg = LEGOLAS.from_control(control)
    assert lg.solution.p == 2
    vals = lg.solution.node_field('q0')
    np.testing.assert_allclose(vals, np.arange(n)[:, None] * np.ones((1, 27)))
    assert LEGOLAS.from_control(control, solution='').solution.path is None
    with pytest.raises(FileNotFoundError):
        LEGOLAS.from_control(control, solution='RESULTS/none.hsol')


def _unpack(payload: bytes):
    k = int.from_bytes(payload[:4], 'little')
    head = json.loads(payload[4:4 + k])
    arr = {name: np.frombuffer(payload, a['dtype'], a['count'], a['offset'])
           for name, a in head['arrays'].items()}
    return head, arr


def test_viewer_serves_node_fields(tmp_path):
    path = write_box_solution(tmp_path / 's.hsol', 3, 1, linear)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2))
    lg.solution.load(path)
    srv = lg.viewer.server(port=0).start()
    get = lambda p: urllib.request.urlopen(srv.url.rstrip('/') + p,
                                           timeout=20).read()
    try:
        info = json.loads(get('/api/info'))
        assert [f['name'] for f in info['node_fields']] == ['q0', 'q1', 'q2']
        assert info['solution']['p'] == 3 and info['solution']['nodes'] == 'Gauss'
        head, arr = _unpack(get('/api/nodefield?name=q1'))
        np.testing.assert_allclose(arr['values'],
                                   lg.solution.node_field('q1').ravel(),
                                   rtol=1e-6)
        _, surf = _unpack(get('/api/surface?zone=inlet'))
        vi = surf['vert_index']
        assert len(vi) == len(surf['positions']) // 3
        # each vertex picks the value of its own node: q1 = 2x + y there
        pos = surf['positions'].reshape(-1, 3) + np.array(info['origin'])
        np.testing.assert_allclose(arr['values'][vi],
                                   2 * pos[:, 0] + pos[:, 1], atol=1e-5)
        cell = json.loads(get('/api/cell?id=0'))
        assert cell['quality']['q1'] == pytest.approx(
            lg.solution.node_field('q1')[0].mean(), rel=1e-6)
    finally:
        srv.stop()


def test_show_reads_the_solution_in_the_server_process(tmp_path):
    path = write_box_solution(tmp_path / 's.hsol', 2, 2, linear)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2))
    h = lg.viewer.show(port=0, idle_timeout=60, timeout=60, solution=path)
    try:
        with urllib.request.urlopen(h.url + 'api/info', timeout=10) as r:
            info = json.loads(r.read())
        assert len(info['node_fields']) == 3
        assert info['solution']['file'] == 's.hsol'
    finally:
        h.stop()


@pytest.mark.parametrize('case,mesh', [
    (MARINO, 'MESH/Wind_turbine_disk_fluid_cgns_g2_mesh.h5'),
    (SPHERE, 'MESH/my_esphere_v2g2_mesh.h5'),
])
def test_geometry_at_the_solution_nodes_is_the_solvers(case, mesh):
    """The LEGOLAS map at the Gauss/GL nodes equals HORSES3D's .hmesh:
    element order, node order and node positions are the solver's."""
    from FotR.solver_tools import horses3d as h3d
    hmeshes = [f for f in glob.glob(os.path.join(case, 'MESH', '*.hmesh'))
               if 'surf' not in f]
    if not os.path.isfile(os.path.join(case, mesh)) or not hmeshes:
        pytest.skip('case not available')
    lg = LEGOLAS(os.path.join(case, mesh))
    blk = lg.mesh.blocks[0]
    n = blk.order + 1
    G = lg.mesh.points[blk.conn].reshape(-1, n, n, n, 3)
    fidx = lg.mesh.meta['file_cell_index']
    for hm in hmeshes:
        d = h3d.read_mesh(hm)
        p = d['x'].shape[1] - 1
        L = lagrange(np.linspace(-1, 1, n),
                     solution_nodes(p, d['header'].node_type))[0]
        X = np.einsum('ck,bj,ai,ekjid->ecbad', L, L, L, G, optimize=True)
        assert np.abs(X - d['x'][fidx]).max() < 1e-10


# ---------------------------------------------------------------------------
# time series (autosaves) and the slider's route
# ---------------------------------------------------------------------------

def write_series(folder, n=3):
    """box_g2p2.hsol plus autosaves box_g2p2_<iter>.hsol with q0 = frame,
    and one autosave still being written (no end marker)."""
    paths = []
    for k in range(n):
        name = 'box_g2p2.hsol' if k == n - 1 else f'box_g2p2_{1000 * k:010d}.hsol'
        q = [np.full((3, 3, 3, 3), float(k)) for _ in range(NX * NY * NZ)]
        paths.append(write_horses_file(folder / name, 2, q, iteration=1000 * k,
                                       time=0.5 * k))
    q = [np.zeros((3, 3, 3, 3)) for _ in range(NX * NY * NZ)]
    write_horses_file(folder / 'box_g2p2_0000099999.hsol', 2, q,
                      iteration=99999, seal=False)
    # another order of the same case: not part of this series
    write_horses_file(folder / 'box_g2p3.hsol', 2,
                      [np.zeros((3, 4, 4, 4)) for _ in range(NX * NY * NZ)])
    return paths


def test_series_lists_complete_autosaves_of_one_order(tmp_path):
    paths = write_series(tmp_path)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2, with_bcs=False))
    lg.solution.load(paths[0])
    s = lg.solution.series()
    assert [r['iteration'] for r in s] == [0, 1000, 2000]
    assert [r['time'] for r in s] == [0.0, 0.5, 1.0]
    assert len(lg.solution.series(complete_only=False)) == 4
    assert lg.solution.series(paths[-1]) == s


def test_viewer_serves_every_frame(tmp_path):
    paths = write_series(tmp_path)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2))
    lg.solution.load(paths[-1])
    srv = lg.viewer.server(port=0).start()
    get = lambda p: urllib.request.urlopen(srv.url.rstrip('/') + p,
                                           timeout=20).read()
    try:
        info = json.loads(get('/api/info'))
        assert [f['iteration'] for f in info['series']] == [0, 1000, 2000]
        assert info['frame'] == 2
        for k in (0, 1, 2):
            _, arr = _unpack(get(f'/api/nodefield?name=q0&frame={k}'))
            np.testing.assert_allclose(arr['values'], float(k))
        with pytest.raises(urllib.error.HTTPError) as exc:
            get('/api/nodefield?name=q0&frame=9')
        assert exc.value.code == 400
    finally:
        srv.stop()
    srv = lg.viewer.server(port=0, series=False)
    try:
        assert srv.info['series'] == []
    finally:
        srv.stop()


# ---------------------------------------------------------------------------
# Statistics and sampling at points
# ---------------------------------------------------------------------------

def write_box_stats(path, p, node_type, nx=NX, ny=NY, nz=NZ):
    """Statistics of the unit-cell box with a linear mean velocity
    U = (x, 2 y, z), <uu> = U**2 + 0.01, <vv> = V**2 + 0.04, <ww> = W**2
    + 0.09, <uv> = U V + 0.005 (the other products without fluctuation)
    and a mean density 1 + 0.1 x."""
    xs = (solution_nodes(p, node_type) + 1) / 2
    moments, means = [], []
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                X, Y, Z = np.meshgrid(ei + xs, ej + xs, ek + xs, indexing='ij')
                U, V, W = X, 2 * Y, Z
                moments.append(np.stack([U, V, W, U ** 2 + 0.01, V ** 2 + 0.04,
                                         W ** 2 + 0.09, U * V + 0.005, U * W, V * W]))
                means.append(np.stack([1 + 0.1 * X, U, V, W, 3000 + 0 * X]))
    return write_stats_file(path, moments, means, node_type=node_type)


@pytest.mark.parametrize('node_type', [1, 2])
def test_statistics_variables(tmp_path, node_type):
    path = write_box_stats(tmp_path / 'b.stats.hsol', 3, node_type)
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=2, with_bcs=False))
    st = lg.solution.load(path)
    assert st.variables == ['u', 'v', 'w', 'Vmag', 'uu', 'vv', 'ww', 'uv',
                            'uw', 'vw', 'tke', 'rho']
    x = lg.mesh.points[lg.mesh.blocks[0].conn]
    np.testing.assert_allclose(st.node_field('v'), 2 * x[..., 1], atol=1e-11)
    np.testing.assert_allclose(st.node_field('uu'), 0.01, atol=1e-10)
    np.testing.assert_allclose(st.node_field('uv'), 0.005, atol=1e-10)
    np.testing.assert_allclose(st.node_field('uw'), 0.0, atol=1e-10)
    np.testing.assert_allclose(st.node_field('tke'), 0.07, atol=1e-10)
    np.testing.assert_allclose(st.node_field('rho'), 1 + 0.1 * x[..., 0], atol=1e-11)
    phys = lg.solution.load(path, physical=True)
    V, R = REFS[2], REFS[3]
    np.testing.assert_allclose(phys.node_field('u'), x[..., 0] * V, rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(phys.node_field('tke'), 0.07 * V ** 2, rtol=1e-9)
    np.testing.assert_allclose(phys.node_field('rho'), (1 + 0.1 * x[..., 0]) * R, rtol=1e-11)


@pytest.mark.parametrize('ngeo', [1, 3])
def test_sample_evaluates_the_element_polynomial(tmp_path, ngeo):
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=ngeo, with_bcs=False))
    rng = np.random.default_rng(3)
    pts = rng.uniform([0, 0, 0], [NX, NY, NZ], size=(50, 3))
    pts = np.vstack([pts, [[NX + 1.0, 0.5, 0.5]]])        # one point outside
    sol = lg.solution.load(write_box_solution(tmp_path / 's.hsol', 3, 1, linear))
    r = sol.sample(pts)
    exp = linear(*pts.T)
    for k in range(3):
        np.testing.assert_allclose(r[f'q{k}'][:-1], exp[k][:-1], atol=1e-11)
    assert r['cell'][-1] == -1 and np.isnan(r['q0'][-1])
    st = lg.solution.load(write_box_stats(tmp_path / 'b.stats.hsol', 2, 2))
    r = st.sample(pts[:-1], ['u', 'w', 'tke'])
    np.testing.assert_allclose(r['u'], pts[:-1, 0], atol=1e-11)
    np.testing.assert_allclose(r['w'], pts[:-1, 2], atol=1e-11)
    np.testing.assert_allclose(r['tke'], 0.07, atol=1e-10)
    assert set(r) == {'cell', 'xi', 'u', 'w', 'tke'}
    with pytest.raises(KeyError):
        st.sample(pts[:2], ['p'])


def test_sample_needs_a_solution():
    lg = LEGOLAS(box_mesh(NX, NY, NZ, order=1, with_bcs=False))
    with pytest.raises(RuntimeError, match='load'):
        lg.solution.sample([[0.5, 0.5, 0.5]])


def test_statistics_of_the_turbine_match_its_probes():
    """Marino validation, model A, p = 2: the mean velocity sampled from the
    statistics at the probes equals the probes, whose signal is steady."""
    stats = os.path.join(MARINO, 'RESULTS', 'adA_g2p2.stats.hsol')
    mesh = os.path.join(MARINO, 'MESH', 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
    if not (os.path.isfile(stats) and os.path.isfile(mesh)):
        pytest.skip('case not available')
    lg = LEGOLAS(mesh)
    st = lg.solution.load(stats)
    names = ['u_xd1_axis', 'u_xd1_r08', 'u_xd5_axis', 'u_xd5_r08']
    pts, last = [], []
    for n in names:
        f = os.path.join(MARINO, 'RESULTS', f'adA_g2p2.{n}.probe')
        with open(f) as fh:
            lines = fh.readlines()
        pts.append([float(lines[k].split()[-1]) for k in (2, 3, 4)])
        last.append(float(lines[-1].split()[-1]))
    r = st.sample(np.array(pts), ['u'])
    np.testing.assert_allclose(r['u'], last, atol=2e-4)
