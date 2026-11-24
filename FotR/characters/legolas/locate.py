"""``LEGOLAS.Locate``: which cell holds a point, and where inside it.

* :meth:`Locate.points` -- the honest search: candidate cells by their
  centroids (k-d tree) and bounding boxes, then Newton on the curved map of
  each hexahedron (Lagrange through its equispaced geometric nodes, any
  order) or barycentric coordinates in a tetrahedron. Reference coordinates
  come back in ``[-1, 1]^3``.
* :meth:`Locate.horses3d_ad` -- what HORSES3D's actuator disk does instead
  (``FindActuatorPointElementAD`` in ``ActuatorDisk.f90``): each MPI rank
  takes the **first** of its elements whose *trilinear* shape (8 corners)
  holds the point, and only that one is tested with the curved shape. A
  point between the chord and the arc of a curved face is then lost, and
  the solver stops with "Actuator disk point not found in mesh". This
  generalises ``check_ad_points.py`` (which only knew Ngeo = 2).
* :func:`read_actuator_points` -- the stations of an
  ``ActuatorDef/Act_ActuatorDiskDef.dat`` (models A and B), as HORSES3D
  places them: plane ``x = x_hub``, angle from +y towards +z.

Prisms and pyramids are not located (their cells are skipped); a point that
only they could hold comes back as outside.
"""

from __future__ import annotations

import numpy as np

from .quality import lagrange

__all__ = ['Locate', 'read_actuator_points']

_SGN = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def read_actuator_points(path, x: float | None = None,
                         n_azimuthal: int | None = None) -> dict:
    """Stations of an actuator-disk definition file, as HORSES3D builds them.

    The file is read by position, like the solver does (one disk): centre on
    line 7, radius on 11, hub radius on 13, ``n_radial n_azimuthal`` on 15,
    then the model block (``# MODEL A`` lists ``r delta_p0 swirl`` from line
    20; model B lists ``r chord twist n_polars`` from line 25, each followed
    by its polar files).

    Parameters
    ----------
    path : str or path-like
        ``Act_ActuatorDiskDef.dat``.
    x : float, optional
        Replace the x of the disk plane.
    n_azimuthal : int, optional
        Replace the number of azimuthal points.

    Returns
    -------
    dict
        ``hub`` (3,), ``radius``, ``hub_radius``, ``radii``, ``n_azimuthal``,
        ``model`` ('A' or 'B'), ``points`` ``(n_az * n_r, 3)`` (radius fastest)
        and ``ij`` (1-based radial and azimuthal index of each point).
    """
    lines = open(path).read().splitlines()
    hub = np.array([float(v) for v in lines[6].split()[:3]])
    radius = float(lines[10].split()[0])
    hub_radius = float(lines[12].split()[0])
    nr, na = (int(v) for v in lines[14].split()[:2])
    if lines[16].strip().upper().startswith('# MODEL A'):
        model = 'A'
        radii = [float(lines[19 + i].split()[0]) for i in range(nr)]
    else:
        model = 'B'
        radii, k = [], 24
        for _ in range(nr):
            t = lines[k].split()
            radii.append(float(t[0]))
            k += 1 + int(t[3])
    if x is not None:
        hub[0] = x
    if n_azimuthal is not None:
        na = int(n_azimuthal)
    theta = np.arange(na) * 2 * np.pi / na
    pts = np.array([[hub[0], hub[1] + r * np.cos(t), hub[2] + r * np.sin(t)]
                    for t in theta for r in radii])
    ij = np.array([(i + 1, j + 1) for j in range(na) for i in range(nr)])
    return {'hub': hub, 'radius': radius, 'hub_radius': hub_radius,
            'radii': np.array(radii), 'n_azimuthal': na, 'model': model,
            'points': pts, 'ij': ij}


def _newton_hex(X, x, order, iters=50):
    """Reference coordinates of points *x* ``(m, 3)`` in hexahedra with
    canonical nodes *X* ``(m, n^3, 3)``; returns ``(xi (m, 3), residual)``."""
    n = order + 1
    nodes = np.linspace(-1, 1, n)
    Xg = X.reshape(len(X), n, n, n, 3)                        # [m, k, j, i]
    xi = np.zeros((len(X), 3))
    res = np.full(len(X), np.inf)
    active = np.ones(len(X), bool)
    for _ in range(iters):
        idx = np.nonzero(active)[0]
        if not len(idx):
            break
        L0, D0 = lagrange(nodes, xi[idx, 0])                  # (a, n)
        L1, D1 = lagrange(nodes, xi[idx, 1])
        L2, D2 = lagrange(nodes, xi[idx, 2])
        Xa = Xg[idx]
        F = np.einsum('ak,aj,ai,akjid->ad', L2, L1, L0, Xa) - x[idx]
        J = np.stack([np.einsum('ak,aj,ai,akjid->ad', L2, L1, D0, Xa),
                      np.einsum('ak,aj,ai,akjid->ad', L2, D1, L0, Xa),
                      np.einsum('ak,aj,ai,akjid->ad', D2, L1, L0, Xa)], -1)
        res[idx] = np.abs(F).max(1)
        done = res[idx] < 1e-12
        try:
            step = np.linalg.solve(J, -F[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(F)
        xi[idx] = np.where(done[:, None], xi[idx], xi[idx] + step)
        xi[idx] = np.clip(xi[idx], -3, 3)
        active[idx] = ~done & (np.abs(xi[idx]).max(1) < 2.5)
    return xi, res


def _newton_trilinear(C, x, iters=50):
    """Reference coordinates in the trilinear cells of VTK corners *C*."""
    xi = np.zeros((len(C), 3))
    for _ in range(iters):
        w = 1 + _SGN[None] * xi[:, None, :]                     # (m, 8, 3)
        N = w.prod(-1) / 8
        F = np.einsum('mc,mcd->md', N, C) - x
        dN = np.stack([_SGN[None, :, d] * np.delete(w, d, -1).prod(-1) / 8
                       for d in range(3)], -1)
        J = np.einsum('mcd,mce->med', dN, C)
        try:
            step = np.linalg.solve(J, -F[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        xi = np.clip(xi + step, -3, 3)
        if np.abs(F).max() < 1e-12:
            break
    return xi


class Locate:
    """Find the cell that holds each point.

    Reached as ``lg.locate``. See the module documentation for the two
    searches.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.
    """

    def __init__(self, legolas):
        self._lg = legolas

    @property
    def mesh(self):
        """The mesh searched."""
        return self._lg.mesh

    def _index(self):
        m = self.mesh
        if 'locate_index' not in m._cache:
            from scipy.spatial import cKDTree
            ctr = np.empty((m.n_cells, 3))
            lo = np.empty((m.n_cells, 3))
            hi = np.empty((m.n_cells, 3))
            for blk, s in zip(m.blocks, m.offsets[:-1]):
                x = m.points[blk.conn]
                ctr[s:s + len(blk)] = m.points[blk.corners()].mean(1)
                lo[s:s + len(blk)] = x.min(1)
                hi[s:s + len(blk)] = x.max(1)
            m._cache['locate_index'] = (cKDTree(ctr), lo, hi)
        return m._cache['locate_index']

    def _inside(self, cells, x, tol):
        """Reference coordinates of point i in cell i (NaN if unsupported)."""
        m = self.mesh
        xi = np.full((len(cells), 3), np.nan)
        b_of, local = m.locate_cell(cells)
        for b, blk in enumerate(m.blocks):
            sel = np.nonzero(b_of == b)[0]
            if not len(sel):
                continue
            if blk.kind == 'hex':
                X = m.points[blk.conn[local[sel]]]
                xi[sel], _ = _newton_hex(X, x[sel], blk.order)
            elif blk.kind == 'tet':
                X = m.points[blk.conn[local[sel]]]
                T = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0],
                              X[:, 3] - X[:, 0]], -1)
                lam = np.linalg.solve(T, (x[sel] - X[:, 0])[..., None])[..., 0]
                # barycentric (l1, l2, l3) -> [-1, 1] style: inside when all
                # >= 0 and their sum <= 1; mapped so that the test is |.|<=1
                l0 = 1 - lam.sum(1)
                xi[sel] = np.stack([2 * lam[:, 0] - 1, 2 * lam[:, 1] - 1,
                                    2 * lam[:, 2] - 1], 1)
                bad = (l0 < -tol)
                xi[sel[bad]] = 9.0
        return xi

    def points(self, xyz, k: int = 16, tol: float = 1e-8) -> dict:
        """Locate points, honestly (curved shapes, any order).

        Parameters
        ----------
        xyz : array_like
            ``(n, 3)`` points.
        k : int
            Nearest cell centroids tried first; when none holds a point,
            ``4 k`` and then ``16 k`` are tried.
        tol : float
            Tolerance on the reference coordinates (inside when every
            ``|xi| <= 1 + tol``).

        Returns
        -------
        dict
            ``cell`` (``-1`` when outside every hexahedron and tetrahedron),
            ``xi`` ``(n, 3)`` reference coordinates (NaN when outside),
            ``file_cell`` (the file's element, when the reader keeps it),
            ``distance`` (to the nearest cell centroid, for the points
            outside).

        Examples
        --------
        ::

            r = lg.locate.points([[3.66, 1.355, 0.817]])
            r['cell'][0], r['xi'][0]
        """
        x = np.atleast_2d(np.asarray(xyz, float))
        tree, lo, hi = self._index()
        n = len(x)
        cell = np.full(n, -1, np.int64)
        xi = np.full((n, 3), np.nan)
        todo = np.arange(n)
        kk = min(k, self.mesh.n_cells)
        while len(todo):
            _, cand = tree.query(x[todo], kk)
            cand = np.atleast_2d(cand.reshape(len(todo), -1))
            pt = np.repeat(todo, cand.shape[1])
            cc = cand.ravel()
            box = ((lo[cc] <= x[pt] + tol) & (hi[cc] >= x[pt] - tol)).all(1)
            pt, cc = pt[box], cc[box]
            if len(cc):
                r = self._inside(cc, x[pt], tol)
                ok = np.isfinite(r).all(1) & (np.abs(r).max(1) <= 1 + tol)
                for p_, c_, r_ in zip(pt[ok], cc[ok], r[ok]):
                    if cell[p_] < 0:
                        cell[p_] = c_
                        xi[p_] = r_
            todo = todo[cell[todo] < 0]
            if kk >= min(16 * k, self.mesh.n_cells):
                break
            kk = min(4 * kk, self.mesh.n_cells)
        dist, _ = tree.query(x)
        fidx = self.mesh.meta.get('file_cell_index')
        return {'cell': cell, 'xi': xi,
                'file_cell': (np.where(cell >= 0, fidx[np.maximum(cell, 0)], -1)
                              if fidx is not None else None),
                'distance': np.where(cell < 0, dist, 0.0)}

    def horses3d_ad(self, points, nranks: int = 1, inner_curves: bool = True,
                    tol: float = 1e-8) -> dict:
        """Would HORSES3D's actuator disk find these points?

        Replicates ``FindActuatorPointElementAD``: the elements are split
        among ``nranks`` ranks in contiguous blocks of the file's element
        order; a point is found when some rank's *first* element (lowest
        index) whose trilinear shape holds it also holds it with the curved
        shape. With ``inner_curves=False`` (``mesh inner curves = .false.``)
        only elements with a boundary face are curved.

        Parameters
        ----------
        points : array_like or str
            ``(n, 3)`` points, or the path of an ``Act_ActuatorDiskDef.dat``
            (read with :func:`read_actuator_points`).
        nranks : int
            MPI ranks of the run.
        inner_curves : bool
            The solver's ``mesh inner curves``. Default True (its default).
        tol : float
            Tolerance on the reference coordinates.

        Returns
        -------
        dict
            ``found`` (bool per point), ``cell`` (the element that would take
            it, -1 if none), ``honest`` (:meth:`points`'s answer, to compare),
            ``lost`` (indices of points inside the fluid that HORSES3D would
            not find) and ``max_xi`` (largest curved ``|xi|`` among the
            elements whose trilinear shape holds each point: values just
            above 1 flag points at risk for other numbers of ranks).
        """
        if isinstance(points, (str, bytes)) or hasattr(points, '__fspath__'):
            points = read_actuator_points(points)['points']
        x = np.atleast_2d(np.asarray(points, float))
        m = self.mesh
        if any(b.kind != 'hex' for b in m.blocks):
            raise NotImplementedError('the HORSES3D search is for hexahedra')
        fidx = m.meta.get('file_cell_index')
        if fidx is None:
            fidx = np.arange(m.n_cells)
        file_order = np.empty(m.n_cells, np.int64)
        file_order[:] = fidx
        n_file = int(file_order.max()) + 1
        block = int(np.ceil(n_file / nranks))
        boundary_cell = np.zeros(m.n_cells, bool)
        for bc in m.boundaries.values():
            boundary_cell[bc.elem] = True
        _, lo, hi = self._index()
        found = np.zeros(len(x), bool)
        took = np.full(len(x), -1, np.int64)
        max_xi = np.full(len(x), np.nan)
        b_of_all, local_all = m.locate_cell(np.arange(m.n_cells))
        for i, p in enumerate(x):
            cand = np.nonzero(((lo <= p + tol) & (hi >= p - tol)).all(1))[0]
            if not len(cand):
                continue
            C = np.stack([m.points[m.blocks[b_of_all[c]].corners()[local_all[c]]]
                          for c in cand])
            lin = np.abs(_newton_trilinear(C, np.repeat(p[None], len(cand), 0))
                         ).max(1)
            hold = cand[lin <= 1 + tol]
            if not len(hold):
                continue
            curved = np.empty(len(hold))
            for j, c in enumerate(hold):
                blk = m.blocks[b_of_all[c]]
                if inner_curves or boundary_cell[c]:
                    X = m.points[blk.conn[local_all[c]]][None]
                    r, _ = _newton_hex(X, p[None], blk.order)
                    curved[j] = np.abs(r).max()
                else:
                    curved[j] = lin[lin <= 1 + tol][j]
            max_xi[i] = curved.max()
            rank = file_order[hold] // block
            for rk in np.unique(rank):
                mine = hold[rank == rk]
                first = mine[np.argmin(file_order[mine])]
                if curved[np.nonzero(hold == first)[0][0]] <= 1 + tol:
                    found[i] = True
                    took[i] = first
                    break
        honest = self.points(x, tol=tol)['cell']
        lost = np.nonzero(~found & (honest >= 0))[0]
        return {'found': found, 'cell': took, 'honest': honest,
                'lost': lost, 'max_xi': max_xi}

    def __repr__(self) -> str:
        return f"<LEGOLAS.Locate of {self.mesh!r}>"
