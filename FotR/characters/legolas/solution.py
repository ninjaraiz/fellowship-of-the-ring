"""``LEGOLAS.Solution``: a HORSES3D solution on the mesh, node by node.

A ``.hsol`` stores, per element, the solution at ``(p+1)**3`` Gauss or
Gauss-Lobatto nodes, in the element order of the mesh ``.h5`` and with ``i``
running fastest along xi. The mesh in LEGOLAS has its own nodes: the
geometric ones, equispaced, of order Ngeo. :meth:`Solution.load` evaluates
the solution's polynomial of each element at those nodes, so the viewer can
colour every face vertex with the value of *its* element: the jumps of a DG
solution between elements stay visible, nothing is averaged.

Checked on 24/11/2026 against the solver itself: the LEGOLAS geometry
evaluated at the solution nodes reproduces the coordinates of HORSES3D's
``.hmesh`` to 1e-14 (turbine, Gauss nodes, p = 2, 3, 4; sphere,
Gauss-Lobatto, p = 2). Element order, node order and node positions are
therefore the solver's.

Variables: with the 5 conservative variables, ``rho``, ``u``, ``v``, ``w``,
``p``, ``T``, ``Mach``, ``Cp`` (:func:`FotR.solver_tools.horses3d.
primitive_variables`; incompressible runs have no ``T`` nor ``Mach``) and
``Vmag``. They are computed at the solution nodes and then interpolated.
Other files keep their raw components ``q0``, ``q1``…

Statistics files (``.stats.hsol``, ``#define statistics``) load the time
averages: ``u``, ``v``, ``w`` and ``Vmag`` are the mean velocity, ``uu``,
``vv``, ``ww``, ``uv``, ``uw``, ``vw`` the Reynolds stresses (fluctuations,
``<u'u'> = <uu> - <u>**2``), ``tke`` the turbulent kinetic energy and ``rho``
the mean density.

:meth:`Solution.sample` evaluates the loaded variables at any points: each
point is located in its element (:meth:`LEGOLAS.Locate.points`, curved
shapes included) and the polynomial of that element is evaluated at its
reference coordinates, so profiles along lines or planes keep the order of
the solution.

Not yet: p-adaptive solutions, gradients (vorticity, Q criterion) and
surfaces (``.surf.hsol``).
"""

from __future__ import annotations

import os

import numpy as np

from .quality import gauss_lobatto, lagrange

__all__ = ['Solution', 'solution_nodes', 'read_control_keys']


def solution_nodes(p: int, node_type: int) -> np.ndarray:
    """Reference positions in ``[-1, 1]`` of the ``p + 1`` solution nodes.

    ``node_type`` is the header code: 1 Gauss, 2 Gauss-Lobatto.
    """
    if node_type == 1:
        return np.polynomial.legendre.leggauss(p + 1)[0]
    if node_type == 2:
        return gauss_lobatto(p)[0]
    raise ValueError(f'unknown node type {node_type}')


def read_control_keys(path, keys=('mesh file name', 'solution file name',
                                  'polynomial order')) -> dict:
    """Values of some keys of a HORSES3D ``.control`` (comments dropped).

    Keys are matched without case; values lose their quotes.
    """
    want = {k.lower(): k for k in keys}
    out = {}
    with open(path, errors='replace') as fh:
        for line in fh:
            line = line.split('!', 1)[0]
            if '=' not in line or line.lstrip().startswith('#'):
                continue
            k, v = line.split('=', 1)
            k = ' '.join(k.split()).lower()
            if k in want:
                out[want[k]] = v.strip().strip('"\'')
    return out


def _stats_variables(M, refs: dict, physical: bool) -> dict:
    """Mean velocity, Reynolds stresses, TKE and mean density from the 9
    moments of a statistics file and its mean density (last component)."""
    V = refs['V_ref'] if physical else 1.0
    R = refs['rho_ref'] if physical else 1.0
    u, v, w = M[..., 0], M[..., 1], M[..., 2]
    out = {'u': u * V, 'v': v * V, 'w': w * V,
           'Vmag': np.sqrt(u ** 2 + v ** 2 + w ** 2) * V}
    for k, (name, a, b) in enumerate((('uu', u, u), ('vv', v, v), ('ww', w, w),
                                      ('uv', u, v), ('uw', u, w), ('vw', v, w))):
        out[name] = (M[..., 3 + k] - a * b) * V ** 2
    out['tke'] = 0.5 * (out['uu'] + out['vv'] + out['ww'])
    out['rho'] = M[..., 9] * R
    return out


def _finite_range(v) -> tuple:
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if len(v) else (np.nan, np.nan)


class Solution:
    """A HORSES3D solution read onto the nodes of the mesh.

    Reached as ``lg.solution``; :meth:`load` fills it. Values live per
    (cell, local node): ``node_field(name)`` is ``(n_cells, (Ngeo+1)**3)``
    for a hexahedral mesh, in the canonical node order of each cell.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.

    Attributes
    ----------
    path : str or None
        The ``.hsol`` loaded.
    header : HorsesHeader or None
        Its header: time, iteration, node type, references.
    p : int or None
        Polynomial order of the solution.
    """

    def __init__(self, legolas):
        self._lg = legolas
        self.path = None
        self.header = None
        self.p = None
        self.physical = False
        self._node = {}
        self._cell = {}
        self._raw = {}
        self._xs = None

    @property
    def mesh(self):
        """The mesh the solution lives on."""
        return self._lg.mesh

    @property
    def variables(self) -> list:
        """Names of the loaded variables."""
        return list(self._node)

    def load(self, path, physical: bool = False) -> 'Solution':
        """Read a ``.hsol`` and evaluate it at the nodes of the mesh.

        Parameters
        ----------
        path : str or path-like
            A volume solution (final or autosave) of this very mesh, or a
            statistics file (``.stats.hsol``) of it.
        physical : bool
            SI units (kg/m3, m/s, Pa, K) instead of the solver's
            non-dimensional ones.

        Returns
        -------
        Solution
            ``self``, so that ``lg.solution.load(f).variables`` works.

        Raises
        ------
        ValueError
            If the element count does not match the mesh, or the mesh has
            other cells than hexahedra.

        Examples
        --------
        ::

            lg = LEGOLAS('MESH/Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
            sol = lg.solution.load('RESULTS/adA_g2p3.hsol')
            sol.variables, sol.node_field('Mach').shape
            lg.viewer.show(solution='RESULTS/adA_g2p3.hsol')
            st = lg.solution.load('RESULTS/adA_g2p3.stats.hsol')
            st.variables          # u v w Vmag uu vv ww uv uw vw tke rho
        """
        from FotR.solver_tools import horses3d as h3d

        m = self.mesh
        if any(b.kind != 'hex' for b in m.blocks):
            raise ValueError('solutions need a mesh of hexahedra')
        stats = h3d.read_header(path).file_type == 4
        if stats:
            data = h3d.read_stats(path)
            Q = np.concatenate([data['moments'], data['Q'][..., :1]], axis=-1)
        else:
            data = h3d.read_solution(path)
            Q = data['Q']
        header = data['header']
        if isinstance(Q, list):
            raise NotImplementedError(f'{path}: elements of different orders '
                                      '(p-adaptation) are not supported yet')
        fidx = m.meta.get('file_cell_index')
        if fidx is None:
            fidx = np.arange(m.n_cells)
        if len(Q) != int(np.max(fidx)) + 1 or len(Q) != m.n_cells:
            raise ValueError(f'{path} has {len(Q)} elements, the mesh '
                             f'{m.n_cells} cells: not a solution of this mesh')
        nz, ny, nx = Q.shape[1:4]
        if not nx == ny == nz:
            raise NotImplementedError(f'{path}: anisotropic orders')
        p = nx - 1
        Q = Q[fidx]                                   # file order -> cells
        if stats:
            raw = _stats_variables(Q, header.refs, physical)
        elif Q.shape[-1] == 5:
            raw = h3d.primitive_variables(Q, header.refs, physical=physical)
            raw['Vmag'] = np.sqrt(raw['u'] ** 2 + raw['v'] ** 2
                                  + raw['w'] ** 2)
        else:
            raw = {f'q{k}': Q[..., k] for k in range(Q.shape[-1])}
        xs = solution_nodes(p, header.node_type)
        _, w1 = (np.polynomial.legendre.leggauss(p + 1) if header.node_type == 1
                 else gauss_lobatto(p))
        w3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
        node, cell = {}, {}
        for name, v in raw.items():
            parts = []
            for blk, start in zip(m.blocks, m.offsets[:-1]):
                n = blk.order + 1
                L = lagrange(xs, np.linspace(-1, 1, n))[0]     # (n, p+1)
                vb = v[start:start + len(blk)]
                out = np.einsum('ck,bj,ai,ekji->ecba', L, L, L, vb,
                                optimize=True)
                parts.append(out.reshape(len(blk), -1))
            node[name] = np.vstack(parts) if len(parts) > 1 else parts[0]
            cell[name] = (v * w3).sum((1, 2, 3)) / w3.sum()
        self.path = os.fspath(path)
        self.header = header
        self.p = p
        self.physical = physical
        self._node, self._cell = node, cell
        self._raw, self._xs = raw, xs
        return self

    def sample(self, xyz, variables=None) -> dict:
        """Evaluate the loaded solution at any points.

        Each point is located in its element (:meth:`LEGOLAS.Locate.points`,
        with the curved shape of the element) and the solution polynomial of
        that element is evaluated at the point's reference coordinates: the
        value is that of the solver's own discretization, not an
        interpolation between nodes. Points outside the mesh get NaN.

        Parameters
        ----------
        xyz : array_like
            ``(n, 3)`` points, in the units of the mesh.
        variables : list of str, optional
            Default: every loaded variable.

        Returns
        -------
        dict
            One ``(n,)`` array per variable, plus ``cell`` (``-1`` outside)
            and ``xi`` (reference coordinates).

        Examples
        --------
        ::

            st = lg.solution.load('RESULTS/adA_g2p3.stats.hsol')
            y = np.linspace(1.355 - 0.894, 1.355 + 0.894, 201)
            line = np.column_stack([np.full_like(y, 4.554), y,
                                    np.full_like(y, 0.817)])
            a = 1 - st.sample(line, ['u'])['u']     # wake deficit at x/D = 1
        """
        if not self._raw:
            raise RuntimeError('no solution loaded: call load(path) first')
        names = list(variables or self.variables)
        for n in names:
            self._check(n)
        loc = self._lg.locate.points(xyz)
        cell, xi = loc['cell'], loc['xi']
        ok = cell >= 0
        out = {'cell': cell, 'xi': xi}
        Lx = lagrange(self._xs, xi[ok, 0])[0]
        Ly = lagrange(self._xs, xi[ok, 1])[0]
        Lz = lagrange(self._xs, xi[ok, 2])[0]
        for n in names:
            v = np.full(len(cell), np.nan)
            v[ok] = np.einsum('ak,aj,ai,akji->a', Lz, Ly, Lx,
                              self._raw[n][cell[ok]], optimize=True)
            out[n] = v
        return out

    # -- reading the values -------------------------------------------------
    def _check(self, name):
        if not self._node:
            raise RuntimeError('no solution loaded: call load(path) first')
        if name not in self._node:
            raise KeyError(f'no variable {name!r}; loaded: {self.variables}')

    def node_field(self, name: str) -> np.ndarray:
        """Values at the nodes of every cell, ``(n_cells, (Ngeo+1)**3)``."""
        self._check(name)
        return self._node[name]

    def cell_field(self, name: str) -> np.ndarray:
        """Mean over each element (quadrature at the solution nodes, in the
        reference cube), ``(n_cells,)``."""
        self._check(name)
        return self._cell[name]

    def point_field(self, name: str) -> np.ndarray:
        """Values per mesh node, averaged over the cells that share it, for
        exports that need one value per node (``to_vtu(point_fields=…)``)."""
        self._check(name)
        m = self.mesh
        acc = np.zeros(m.n_points)
        cnt = np.zeros(m.n_points)
        v = self._node[name]
        for blk, start in zip(m.blocks, m.offsets[:-1]):
            vb = v[start:start + len(blk)]
            np.add.at(acc, blk.conn.ravel(), vb.ravel())
            np.add.at(cnt, blk.conn.ravel(), 1)
        with np.errstate(invalid='ignore'):
            return acc / cnt

    def series(self, path=None, complete_only: bool = True) -> list:
        """The autosaves of a solution, as a time series.

        Parameters
        ----------
        path : str or path-like, optional
            A ``.hsol``; default the one loaded. Its series is every
            ``<stem>.hsol`` and ``<stem>_<iteration>.hsol`` in its folder,
            with ``<stem>`` its name without the iteration (so one order p).
        complete_only : bool
            Leave out files the solver is still writing. Default True.

        Returns
        -------
        list[dict]
            ``path``, ``iteration``, ``time`` and ``complete``, by iteration.
        """
        import re
        from FotR.solver_tools import horses3d as h3d

        path = os.fspath(path or self.path or '')
        if not path:
            raise RuntimeError('no solution loaded and no path given')
        name = os.path.basename(path)
        m = re.match(r'^(?P<stem>.+?)(?:_\d{10})?\.hsol$', name)
        stem = m.group('stem') if m else name[:-5]
        out = [{'path': h.path, 'iteration': int(h.iteration),
                'time': float(h.time), 'complete': bool(h.complete)}
               for h in h3d.find_solutions(os.path.dirname(os.path.abspath(
                   path)), stem)]
        return [r for r in out if r['complete'] or not complete_only]

    def viewer_fields(self, variables=None) -> dict:
        """Node fields ready for :meth:`LEGOLAS.Viewer.show` (``node_fields=``)."""
        names = variables or self.variables
        unit = ' (SI)' if self.physical else ''
        return {n: {'values': self.node_field(n), 'better': None,
                    'label': {'es': n + unit, 'en': n + unit}} for n in names}

    def summary(self) -> dict:
        """Where it comes from, and min/max of every variable at the nodes."""
        if not self._node:
            return {}
        h = self.header
        return {'path': self.path, 'p': self.p,
                'node_type': h.node_type_name, 'time': h.time,
                'iteration': h.iteration, 'physical': self.physical,
                'range': {k: _finite_range(v) for k, v in self._node.items()}}

    def __repr__(self) -> str:
        if not self._node:
            return '<LEGOLAS.Solution: none loaded>'
        h = self.header
        return (f'<LEGOLAS.Solution {os.path.basename(self.path)}: p={self.p}, '
                f'{h.node_type_name}, t={h.time:g}, iter={h.iteration}, '
                f'{len(self._node)} variables>')
