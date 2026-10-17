"""Tests for ``FotR.solver_tools.horses3d`` (HORSES3D .hsol/.hmesh reader).

The synthetic files are written here with ``struct``, following
``HexMesh_SaveSolution``/``SolutionFile.f90`` (record positions included),
and independently of the reader: a field written as Fortran ``q(var,i,j,k)``
must come back as ``Q[k, j, i, var]``.

The module only needs numpy, so no stubs are involved. The two smoke tests
on real files are skipped where those files do not exist.
"""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from FotR.solver_tools import horses3d as h3d

REFS = (1.4, 287.15, 68.157, 1.2215, 288.889, 0.2)   # gamma, R, V_ref, rho_ref, T_ref, Mach

SPHERE_HSOL = ('/home/m.jaraiz/Documentos/COMAC/my_ESPHERE_1/outputs/M_0.3_re_200/'
               'RESULTS/my_esphere_v2g2p2.hsol')
SPHERE_HMESH = ('/home/m.jaraiz/Documentos/COMAC/my_ESPHERE_1/outputs/M_0.3_re_400/'
                'MESH/my_esphere_v2g2p2_0000000000.hmesh')
FAN_HSOL = ('/home/m.jaraiz/Documentos/COMAC/fan_duct_AD_pressure/outputs/M_0.001_dpp0_-0.05/'
            'RESULTS/dp_0000554885.hsol')


def write_horses_file(path, file_type, q, *, node_type=2, iteration=0, time=0.0, refs=REFS,
                      grads=None, sensor=None, les=None, seal=True, name=None):
    """Write a HORSES3D binary file the way the solver does.

    q: list of per-element arrays indexed ``[var, i, j, k]``.
    grads: list of ``(3, n_grad, nx, ny, nz)``; sensor: list of floats;
    les: list of ``(2, nx, ny, nz)``.
    """
    n_vars = q[0].shape[0]
    n_grad = grads[0].shape[1] if grads is not None else 0
    padding = n_vars + 3 * n_grad + (2 if les is not None else 0)
    header = bytearray(204)
    stored = (name if name is not None else os.path.basename(path)).encode()[:128]
    header[:len(stored)] = stored
    struct.pack_into('<4i', header, 128, file_type, node_type, len(q), iteration)
    struct.pack_into('<d', header, 144, time)
    struct.pack_into('<6d', header, 152, *refs)
    struct.pack_into('<i', header, 200, 88)
    with open(path, 'wb') as f:
        f.write(header)
        offset_io = 0
        for e, qe in enumerate(q):
            pos = 204 + e * 20 + padding * offset_io * 8
            if sensor is not None:
                pos += e * 8
            f.seek(pos)
            f.write(struct.pack('<5i', 4, *qe.shape))
            f.write(np.asarray(qe, '<f8').tobytes(order='F'))
            if grads is not None:
                for d in range(3):
                    f.write(np.asarray(grads[e][d], '<f8').tobytes(order='F'))
            if sensor is not None:
                f.write(struct.pack('<d', sensor[e]))
            if les is not None:
                f.write(np.asarray(les[e][0], '<f8').tobytes(order='F'))
                f.write(np.asarray(les[e][1], '<f8').tobytes(order='F'))
            offset_io += int(np.prod(qe.shape[1:]))
        if seal:
            f.seek(0, 2)
            f.write(struct.pack('<i', 99))
    return path


def gl_nodes(n):
    """Gauss-Lobatto nodes on [0, 1] (only n = 1, 2, 3 needed here)."""
    return {1: np.array([0.0, 1.0]),
            2: np.array([0.0, 0.5, 1.0]),
            3: np.array([0.0, 0.5 - 0.5 / np.sqrt(5), 0.5 + 0.5 / np.sqrt(5), 1.0])}[n]


def element_coords(e, order):
    """Fortran-indexed x[3, i, j, k] of element e, the unit cube shifted by e in x."""
    xi, eta, zeta = (gl_nodes(n) for n in order)
    X, Y, Z = np.meshgrid(xi + e, eta, zeta, indexing='ij')
    return np.stack([X, Y, Z])


A = np.array([1.0, 0.3, -0.2, 0.1, 2.5])
B = np.array([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5], [0.0, 2.0, 1.0], [-1.0, 0.5, 0.0], [0.2, 0.2, 0.2]])


def linear_q(x):
    """q[var, i, j, k] = A_var + B_var . x, a field the reader must return exactly."""
    return A[:, None, None, None] + np.einsum('vd,dijk->vijk', B, x)


def fortran_to_reader(arr):
    """[var, i, j, k] -> [k, j, i, var], what the reader returns."""
    return np.transpose(arr, (3, 2, 1, 0))


@pytest.fixture
def uniform_case(tmp_path):
    orders = [(2, 2, 2)] * 4
    x = [element_coords(e, o) for e, o in enumerate(orders)]
    q = [linear_q(xe) for xe in x]
    mesh = write_horses_file(tmp_path / 'case_0000000000.hmesh', 1, x, refs=(0.0,) * 6)
    sol = write_horses_file(tmp_path / 'case_0000001000.hsol', 2, q, iteration=1000, time=0.25)
    return mesh, sol, x, q


# ── header ──────────────────────────────────────────────────────────────────


def test_header_fields(uniform_case):
    _, sol, _, _ = uniform_case
    h = h3d.read_header(sol)
    assert (h.file_type, h.file_type_name) == (2, 'SOLUTION')
    assert (h.node_type, h.node_type_name) == (2, 'Gauss-Lobatto')
    assert (h.n_elements, h.iteration, h.time) == (4, 1000, 0.25)
    assert h.name == 'case_0000001000.hsol'
    assert h.refs == dict(zip(('gamma', 'R', 'V_ref', 'rho_ref', 'T_ref', 'Mach'), REFS))
    assert h.complete and not h.is_mesh
    assert h.physical_time() == pytest.approx(0.25 / 68.157)


def test_header_name_padded_with_spaces(tmp_path):
    q = [linear_q(element_coords(0, (1, 1, 1)))]
    path = write_horses_file(tmp_path / 'a.hsol', 2, q, name='RESULTS/a.hsol' + ' ' * 20)
    assert h3d.read_header(path).name == 'RESULTS/a.hsol'


def test_not_a_horses_file(tmp_path):
    path = tmp_path / 'junk.hsol'
    path.write_bytes(b'\x00' * 400)
    with pytest.raises(ValueError, match='not a HORSES3D'):
        h3d.read_header(path)


def test_incomplete_file(tmp_path):
    q = [linear_q(element_coords(e, (2, 2, 2))) for e in range(3)]
    path = write_horses_file(tmp_path / 'run_0000000500.hsol', 2, q, seal=False)
    assert not h3d.read_header(path).complete
    with pytest.raises(h3d.IncompleteFileError):
        h3d.read_solution(path)


def test_surface_files_not_supported(tmp_path):
    q = [linear_q(element_coords(0, (1, 1, 1)))]
    path = write_horses_file(tmp_path / 'x_bc001_0000000000.surf.hsol', 6, q)
    with pytest.raises(NotImplementedError, match='ZONE_SOLUTION'):
        h3d.scan(path)


# ── uniform orders ──────────────────────────────────────────────────────────


def test_solution_round_trip(uniform_case):
    _, sol, _, q = uniform_case
    out = h3d.read_solution(sol)
    assert out['layout'].uniform
    assert out['Q'].shape == (4, 3, 3, 3, 5)
    for e in range(4):
        np.testing.assert_array_equal(out['Q'][e], fortran_to_reader(q[e]))
    assert set(out) == {'header', 'layout', 'elements', 'Q'}


def test_mesh_and_solution_agree_node_by_node(uniform_case):
    mesh, sol, _, _ = uniform_case
    x = h3d.to_nodes(h3d.read_mesh(mesh)['x'])
    Q = h3d.to_nodes(h3d.read_solution(sol)['Q'])
    np.testing.assert_allclose(Q, A + x @ B.T, rtol=0, atol=1e-13)


def test_selection_reads_requested_elements_in_order(uniform_case):
    _, sol, _, q = uniform_case
    out = h3d.read_solution(sol, elements=[3, 0, 2])
    np.testing.assert_array_equal(out['elements'], [3, 0, 2])
    for got, e in zip(out['Q'], [3, 0, 2]):
        np.testing.assert_array_equal(got, fortran_to_reader(q[e]))
    with pytest.raises(IndexError):
        h3d.read_solution(sol, elements=[4])


def test_layout_reused_between_selections(uniform_case):
    _, sol, _, q = uniform_case
    layout = h3d.scan(sol)
    np.testing.assert_array_equal(layout.offsets, 204 + np.arange(4) * (20 + 5 * 27 * 8))
    a = h3d.read_solution(sol, elements=[1], layout=layout)['Q'][0]
    np.testing.assert_array_equal(a, fortran_to_reader(q[1]))


def test_to_nodes_local_order(uniform_case):
    _, sol, _, q = uniform_case
    nodes = h3d.to_nodes(h3d.read_solution(sol, elements=[2])['Q'])
    for i, j, k in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 1, 2)]:
        np.testing.assert_array_equal(nodes[i + 3 * j + 9 * k], q[2][:, i, j, k])


def test_wrong_reader_for_file_type(uniform_case):
    mesh, sol, _, _ = uniform_case
    with pytest.raises(ValueError, match='mesh file'):
        h3d.read_solution(mesh)
    with pytest.raises(ValueError, match='not a mesh'):
        h3d.read_mesh(sol)


@pytest.mark.parametrize('file_type, with_sensor', [(3, False), (9, True)])
def test_gradients_and_sensor(tmp_path, file_type, with_sensor):
    rng = np.random.default_rng(0)
    q = [rng.normal(size=(5, 3, 3, 3)) for _ in range(3)]
    grads = [rng.normal(size=(3, 5, 3, 3, 3)) for _ in range(3)]
    sensor = [0.1, 0.2, 0.3] if with_sensor else None
    path = write_horses_file(tmp_path / 'g.hsol', file_type, q, grads=grads, sensor=sensor)
    out = h3d.read_solution(path)
    assert out['layout'].n_grad == 5
    for e in range(3):
        np.testing.assert_array_equal(out['Q'][e], fortran_to_reader(q[e]))
        for d, name in enumerate(('U_x', 'U_y', 'U_z')):
            np.testing.assert_array_equal(out[name][e], fortran_to_reader(grads[e][d]))
    if with_sensor:
        np.testing.assert_array_equal(out['sensor'], sensor)
    else:
        assert 'sensor' not in out


def test_sensor_without_gradients(tmp_path):
    q = [linear_q(element_coords(e, (2, 2, 2))) for e in range(2)]
    path = write_horses_file(tmp_path / 's.hsol', 8, q, sensor=[5.0, -1.0])
    out = h3d.read_solution(path)
    np.testing.assert_array_equal(out['sensor'], [5.0, -1.0])
    np.testing.assert_array_equal(out['Q'][1], fortran_to_reader(q[1]))


@pytest.mark.parametrize('file_type', [2, 3])
def test_les_viscosities_deduced_from_size(tmp_path, file_type):
    rng = np.random.default_rng(1)
    q = [rng.normal(size=(5, 3, 3, 3)) for _ in range(2)]
    grads = [rng.normal(size=(3, 5, 3, 3, 3)) for _ in range(2)] if file_type == 3 else None
    les = [rng.uniform(size=(2, 3, 3, 3)) for _ in range(2)]
    path = write_horses_file(tmp_path / 'les.hsol', file_type, q, grads=grads, les=les)
    out = h3d.read_solution(path)
    assert out['layout'].has_les
    for e in range(2):
        np.testing.assert_array_equal(out['Q'][e], fortran_to_reader(q[e]))
        np.testing.assert_array_equal(out['mu_total'][e], np.transpose(les[e][0], (2, 1, 0)))
        np.testing.assert_array_equal(out['mu_sgs'][e], np.transpose(les[e][1], (2, 1, 0)))
    assert h3d.to_nodes(out['mu_sgs']).shape == (2 * 27,)


def test_gradient_count_deduced_when_not_five(tmp_path):
    rng = np.random.default_rng(2)
    q = [rng.normal(size=(1, 2, 2, 2)) for _ in range(3)]        # e.g. Cahn-Hilliard
    grads = [rng.normal(size=(3, 1, 2, 2, 2)) for _ in range(3)]
    path = write_horses_file(tmp_path / 'ch.hsol', 3, q, grads=grads)
    out = h3d.read_solution(path)
    assert (out['layout'].n_vars, out['layout'].n_grad, out['layout'].has_les) == (1, 1, False)
    np.testing.assert_array_equal(out['U_z'][2], fortran_to_reader(grads[2][2]))


# ── different orders per element (p-adaptation) ─────────────────────────────


def test_variable_orders_are_walked(tmp_path):
    orders = [(1, 1, 1), (2, 2, 2), (3, 2, 1), (2, 2, 2)]
    x = [element_coords(e, o) for e, o in enumerate(orders)]
    q = [linear_q(xe) for xe in x]
    rng = np.random.default_rng(3)
    grads = [rng.normal(size=(3, 5) + tuple(n + 1 for n in o)) for o in orders]
    mesh = write_horses_file(tmp_path / 'p_0000000000.hmesh', 1, x, refs=(0.0,) * 6)
    sol = write_horses_file(tmp_path / 'p.hsol', 9, q, grads=grads, sensor=[1.0, 2.0, 3.0, 4.0])

    layout = h3d.scan(sol)
    assert not layout.uniform
    np.testing.assert_array_equal(layout.orders, orders)
    out = h3d.read_solution(sol, layout=layout)
    assert isinstance(out['Q'], list) and out['Q'][2].shape == (2, 3, 4, 5)
    for e in range(4):
        np.testing.assert_array_equal(out['Q'][e], fortran_to_reader(q[e]))
        np.testing.assert_array_equal(out['U_y'][e], fortran_to_reader(grads[e][1]))
    np.testing.assert_array_equal(out['sensor'], [1.0, 2.0, 3.0, 4.0])

    xs = h3d.to_nodes(h3d.read_mesh(mesh)['x'])
    np.testing.assert_allclose(h3d.to_nodes(out['Q']), A + xs @ B.T, rtol=0, atol=1e-13)
    sub = h3d.read_solution(sol, elements=[2, 0], layout=layout)
    np.testing.assert_array_equal(sub['Q'][0], fortran_to_reader(q[2]))


# ── derived variables and solution lists ────────────────────────────────────


def test_primitive_variables_free_stream():
    gamma, R, T, M = 1.4, 287.15, 288.889, 0.2
    refs = dict(gamma=gamma, R=R, V_ref=M * np.sqrt(gamma * R * T), rho_ref=101325.0 / (R * T),
                T_ref=T, Mach=M)
    theta = np.radians(180.0)
    p = 1.0 / (gamma * M ** 2)
    Q = np.array([1.0, np.cos(theta), np.sin(theta), 0.0, p / (gamma - 1) + 0.5])
    nd = h3d.primitive_variables(Q, refs)
    assert nd['p'] == pytest.approx(p)
    assert nd['T'] == pytest.approx(1.0)
    assert nd['Mach'] == pytest.approx(M)
    assert nd['Cp'] == pytest.approx(0.0, abs=1e-12)
    si = h3d.primitive_variables(Q, refs, physical=True)
    assert si['p'] == pytest.approx(101325.0)
    assert si['T'] == pytest.approx(T)
    assert si['u'] == pytest.approx(-refs['V_ref'])
    assert si['rho'] == pytest.approx(refs['rho_ref'])


def test_primitive_variables_incompressible():
    refs = dict(gamma=0.0, R=0.0, V_ref=2.0, rho_ref=1.2, T_ref=0.0, Mach=0.0)
    out = h3d.primitive_variables(np.array([[1.0, 2.0, 0.0, -1.0, 3.0]]), refs)
    assert set(out) == {'rho', 'u', 'v', 'w', 'p', 'Cp'}
    assert (out['u'][0], out['w'][0], out['p'][0]) == (2.0, -1.0, 3.0)


def test_find_solutions(tmp_path):
    q = [linear_q(element_coords(0, (1, 1, 1)))]
    write_horses_file(tmp_path / 'run_0000000100.hsol', 2, q, iteration=100, seal=False)
    write_horses_file(tmp_path / 'run.hsol', 2, q, iteration=200)
    write_horses_file(tmp_path / 'run_0000000000.hsol', 2, q, iteration=0)
    write_horses_file(tmp_path / 'other_0000000005.hsol', 2, q, iteration=5)
    write_horses_file(tmp_path / 'run_bc001_0000000000.surf.hsol', 6, q)
    (tmp_path / 'run.residuals').write_text('#\n')

    found = h3d.find_solutions(tmp_path, stem='run')
    assert [h.iteration for h in found] == [0, 100, 200]
    assert [h.complete for h in found] == [True, False, True]
    assert len(h3d.find_solutions(tmp_path)) == 4


def test_import_needs_numpy_only():
    """The reader must import on a cluster with numpy alone (pvpython)."""
    code = ("import sys, FotR.solver_tools.horses3d; "
            "heavy = {'torch', 'pandas', 'pyvista', 'matplotlib', 'h5py', 'scipy', 'seaborn', 'sklearn'}; "
            "print(sorted(heavy & set(sys.modules)))")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, check=True)
    assert res.stdout.strip() == '[]'


# ── real files (skipped when absent) ────────────────────────────────────────


@pytest.mark.skipif(not (os.path.isfile(SPHERE_HSOL) and os.path.isfile(SPHERE_HMESH)),
                    reason='sphere case not available')
def test_real_sphere():
    out = h3d.read_solution(SPHERE_HSOL)
    h = out['header']
    assert (h.file_type, h.n_elements, h.iteration) == (3, 13328, 1000000)
    assert h.time == pytest.approx(107.83, abs=0.01)
    assert out['layout'].uniform and np.all(out['layout'].orders == 2)
    assert out['Q'].shape == (13328, 3, 3, 3, 5) and out['U_x'].shape == (13328, 3, 3, 3, 5)
    prim = h3d.primitive_variables(out['Q'], h.refs)
    assert np.all(np.isfinite(out['Q']))
    assert 0.9 < prim['rho'].min() and prim['rho'].max() < 1.1
    assert prim['Mach'].max() < 0.5

    x = h3d.to_nodes(h3d.read_mesh(SPHERE_HMESH)['x'])
    assert x.shape == (13328 * 27, 3)
    assert np.allclose(x.min(0), -10.0) and np.allclose(x.max(0), 10.0)
    assert np.linalg.norm(x, axis=1).min() == pytest.approx(0.5, abs=1e-6)


@pytest.mark.skipif(not os.path.isfile(FAN_HSOL), reason='fan case not available')
def test_real_fan_selection_only():
    layout = h3d.scan(FAN_HSOL)
    assert (layout.header.file_type, layout.header.n_elements) == (2, 704736)
    assert layout.uniform and layout.n_grad == 0
    out = h3d.read_solution(FAN_HSOL, elements=[0, 352368, 704735], layout=layout)
    assert out['Q'].shape == (3, 3, 3, 3, 5) and np.all(np.isfinite(out['Q']))
