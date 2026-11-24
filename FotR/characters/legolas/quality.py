"""``LEGOLAS.Quality``: cell quality metrics, one value per cell.

Metrics (all per cell, in the global cell order of the mesh):

``scaled_jacobian``
    Minimum over the sample points of ``det J / (|x_xi| |x_eta| |x_zeta|)``
    (the Verdict measure): 1 for a perfect cube, 0 for a flat cell,
    negative for an inverted one. Higher is better.
``jacobian_ratio``
    ``min det J / max det J`` over the sample points: what PyHOPE prints as
    the "scaled Jacobian". 1 for an affine cell, <= 0 for an invalid one.
    Higher is better. Hexahedra only (NaN for the linear kinds, whose
    Jacobian is constant or piecewise constant).
``aspect_ratio``
    Longest over shortest corner edge. Lower is better; boundary layers are
    expected to be large.
``skewness``
    Equiangle skewness of the worst face, 0 (ideal angles) to 1
    (degenerate). Lower is better.
``non_orthogonality``
    Worst angle, in degrees, between a face normal and the line joining the
    centroids of the two cells it separates (inner faces only). Lower is
    better; it matters most to finite-volume solvers.
``volume_ratio``
    Largest ratio of volumes with a neighbour (>= 1). Lower is better.
``volume``
    Cell volume.

Where the Jacobian is evaluated: hexahedra of order N on the
``(N+1)**3`` Gauss-Lobatto points of the reference cube, from the Lagrange
interpolant through the geometric nodes, which every supported format places
at **equispaced** reference positions; for N = 1 those points are the
corners. Linear tetrahedra, prisms and pyramids use the three edges of each
corner (the apex of a pyramid has four and is skipped), normalised so that
the regular element scores 1.

Orientation: a format's convention may make every cell of a kind
"left-handed" (VTK and gmsh order the prism differently). Each block keeps
the orientation of its majority of cells; only the minority comes out
negative, and ``meta['flipped']`` says when a block was turned round.

Only numpy. SAM's finite-difference helpers are not used on purpose: they
would import SAM (and torch) into a module that must stay light.
"""

from __future__ import annotations

import numpy as np

from . import ordering, topology

__all__ = ['Quality', 'METRICS', 'gauss_lobatto', 'lagrange', 'worst_cells']

#: Metric name -> what is good, a short description and a warning threshold
#: (None: no threshold).
METRICS = {
    'scaled_jacobian': {'better': 'high', 'warn': 0.2, 'error': 0.0,
                        'label': {'es': 'jacobiano escalado',
                                  'en': 'scaled Jacobian'}},
    # 0.05 and not more: a hexahedron cut from a regular tetrahedron scores
    # 0.074 and is a valid cell (the my_ESPHERE_1 mesh is made of them)
    'jacobian_ratio': {'better': 'high', 'warn': 0.05, 'error': 0.0,
                       'label': {'es': 'jacobiano mín/máx (PyHOPE)',
                                 'en': 'Jacobian min/max (PyHOPE)'}},
    'aspect_ratio': {'better': 'low', 'warn': None, 'error': None,
                     'label': {'es': 'relación de aspecto',
                               'en': 'aspect ratio'}},
    'skewness': {'better': 'low', 'warn': 0.95, 'error': None,
                 'label': {'es': 'oblicuidad', 'en': 'skewness'}},
    'non_orthogonality': {'better': 'low', 'warn': 80.0, 'error': None,
                          'label': {'es': 'no ortogonalidad (°)',
                                    'en': 'non-orthogonality (°)'}},
    'volume_ratio': {'better': 'low', 'warn': 20.0, 'error': None,
                     'label': {'es': 'salto de volumen',
                               'en': 'volume jump'}},
    'volume': {'better': None, 'warn': None, 'error': None,
               'label': {'es': 'volumen', 'en': 'volume'}},
}

# reference corners of the linear kinds, in the VTK order LEGOLAS keeps
_REF = {
    'tet': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    'pyramid': np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [.5, .5, 1]], float),
    # VTK: the normal of (0, 1, 2) points away from (3, 4, 5)
    'prism': np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1],
                       [0, 1, 1], [1, 0, 1]], float),
}
_IDEAL = {'tet': np.sqrt(2.0), 'pyramid': np.sqrt(2.0),
          'prism': 2.0 / np.sqrt(3.0)}


def gauss_lobatto(n: int) -> tuple:
    """Gauss-Lobatto points and weights on [-1, 1] for order *n* (n+1 points)."""
    if n == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    from numpy.polynomial import legendre as L
    inner = L.legroots(L.legder([0] * n + [1]))
    x = np.concatenate([[-1.0], np.sort(inner.real), [1.0]])
    pn = L.legval(x, [0] * n + [1])
    w = 2.0 / (n * (n + 1) * pn ** 2)
    return x, w


def lagrange(nodes: np.ndarray, x: np.ndarray) -> tuple:
    """Lagrange basis through *nodes* and its derivative, evaluated at *x*.

    Returns
    -------
    L, D : numpy.ndarray
        ``(len(x), len(nodes))``: ``L[a, i] = l_i(x_a)``, ``D[a, i] =
        l_i'(x_a)``.
    """
    nodes = np.asarray(nodes, float)
    x = np.asarray(x, float)
    n = len(nodes)
    L = np.ones((len(x), n))
    D = np.zeros((len(x), n))
    for i in range(n):
        others = [j for j in range(n) if j != i]
        den = np.prod(nodes[i] - nodes[others])
        for j in others:
            L[:, i] *= (x - nodes[j])
        L[:, i] /= den
        for k in others:
            term = np.ones(len(x))
            for j in others:
                if j != k:
                    term *= (x - nodes[j])
            D[:, i] += term
        D[:, i] /= den
    return L, D


def _corner_triples(kind: str) -> list:
    """For each corner, three neighbours ordered for a positive determinant
    on the reference element (corners with four neighbours are skipped)."""
    ref = _REF[kind]
    faces = ordering.LINEAR_FACES[kind]
    nb = {v: set() for v in range(len(ref))}
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            nb[a].add(b)
            nb[b].add(a)
    out = []
    for v in range(len(ref)):
        if len(nb[v]) != 3:
            continue
        a, b, c = sorted(nb[v])
        e = ref[[a, b, c]] - ref[v]
        if np.linalg.det(e) < 0:
            a, b = b, a
        out.append((v, a, b, c))
    return out


def _hex_jacobian(points, conn, order, chunk=None):
    """Jacobian extremes at the Gauss-Lobatto points, per cell (chunked).

    Returns
    -------
    dict
        ``smin``, ``smax`` (scaled Jacobian), ``dmin``, ``dmax`` (det J) and
        ``vol`` (sum of det J times the GL weights), each ``(n_cells,)``.
    """
    n = order + 1
    nodes = np.linspace(-1.0, 1.0, n)
    xg, wg = gauss_lobatto(order)
    L, D = lagrange(nodes, xg)
    w3 = (wg[:, None, None] * wg[None, :, None] * wg[None, None, :])
    m = len(conn)
    chunk = chunk or max(1, int(4e6 // (n ** 3 * 9)))
    out = {k: np.empty(m) for k in ('smin', 'smax', 'dmin', 'dmax', 'vol')}
    for s in range(0, m, chunk):
        x = points[conn[s:s + chunk]].reshape(-1, n, n, n, 3)   # [e,k,j,i,:]

        def along(Mk, Mj, Mi):
            return np.einsum('ck,bj,ai,ekjid->ecbad', Mk, Mj, Mi, x,
                             optimize=True)
        a = along(L, L, D)          # d/dxi
        b = along(L, D, L)          # d/deta
        c = along(D, L, L)          # d/dzeta
        det = np.einsum('...i,...i->...', a, np.cross(b, c))
        norm = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
                * np.linalg.norm(c, axis=-1))
        sj = np.where(norm > 0, det / np.where(norm > 0, norm, 1), 0.0)
        e = len(x)
        det = det.reshape(e, -1)
        sj = sj.reshape(e, -1)
        out['smin'][s:s + e] = sj.min(1)
        out['smax'][s:s + e] = sj.max(1)
        out['dmin'][s:s + e] = det.min(1)
        out['dmax'][s:s + e] = det.max(1)
        out['vol'][s:s + e] = (det.reshape(e, n, n, n) * w3).sum((1, 2, 3))
    return out


def _linear_jacobian(points, conn, kind):
    """Corner scaled Jacobians ``(n_cells, n_corners)``, normalised, and
    volume. The minimum is taken by the caller, once the block's
    orientation is known (taking it before turning the signs round would
    keep the best corner instead of the worst)."""
    x = points[conn]
    vals = []
    for v, a, b, c in _corner_triples(kind):
        e = np.stack([x[:, a] - x[:, v], x[:, b] - x[:, v],
                      x[:, c] - x[:, v]], 1)
        det = np.linalg.det(e)
        norm = np.prod(np.linalg.norm(e, axis=2), axis=1)
        vals.append(np.where(norm > 0, det / np.where(norm > 0, norm, 1), 0))
    sj = np.stack(vals, 1) * _IDEAL[kind]
    # volume: tetrahedra from the corner centroid to every face triangle
    ctr = x.mean(1)
    vol = np.zeros(len(conn))
    for f in ordering.LINEAR_FACES[kind]:
        tris = [(f[0], f[1], f[2])] + ([(f[0], f[2], f[3])]
                                       if len(f) == 4 else [])
        for p, q, r in tris:
            vol += np.abs(np.einsum('ij,ij->i', x[:, p] - ctr, np.cross(
                x[:, q] - ctr, x[:, r] - ctr))) / 6.0
    return sj, vol


def _edges(kind):
    faces = (ordering.HEX_FACES_VTK if kind == 'hex'
             else ordering.LINEAR_FACES[kind])
    return sorted({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                   for f in faces for i in range(len(f))})


def _angles(c):
    """Interior angles (degrees) of polygons ``c`` (m, k, 3)."""
    k = c.shape[1]
    out = []
    for i in range(k):
        u = c[:, (i - 1) % k] - c[:, i]
        v = c[:, (i + 1) % k] - c[:, i]
        cos = np.einsum('ij,ij->i', u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.stack(out, 1)


def worst_cells(mesh, values, better, n: int = 20) -> list:
    """The *n* worst cells of any per-cell array.

    Parameters
    ----------
    mesh : Mesh
    values : numpy.ndarray
        ``(n_cells,)``; NaN cells are skipped.
    better : str or None
        ``'high'`` (lowest first), ``'low'`` or None (highest first).
    n : int
        How many.

    Returns
    -------
    list[dict]
        As :meth:`Quality.worst`.
    """
    v = np.asarray(values, float)
    ok = np.nonzero(np.isfinite(v))[0]
    key = v[ok] if better == 'high' else -v[ok]
    pick = ok[np.argsort(key, kind='stable')[:n]]
    b_of, local = mesh.locate_cell(pick)
    fidx = mesh.meta.get('file_cell_index')
    out = []
    for c, b, l in zip(pick, b_of, local):
        blk = mesh.blocks[b]
        out.append({'cell': int(c), 'value': float(v[c]),
                    'kind': blk.kind, 'order': blk.order,
                    'centroid': mesh.points[blk.corners()[l]].mean(0)
                    .tolist(),
                    'file_cell': int(fidx[c]) if fidx is not None else None})
    return out


class Quality:
    """Cell quality: Jacobians, aspect ratio, skewness, non-orthogonality.

    Reached as ``lg.quality``. Metrics are computed once, on first use, and
    cached (:meth:`metrics`); see the module documentation for what each one
    means and where it is evaluated.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.
    """

    def __init__(self, legolas):
        self._lg = legolas
        self._metrics = None
        self.meta = {}

    @property
    def mesh(self):
        """The mesh being measured."""
        return self._lg.mesh

    # -- computing ------------------------------------------------------------
    def metrics(self) -> dict:
        """Every metric, as ``{name: (n_cells,) float array}``.

        Examples
        --------
        ::

            q = lg.quality.metrics()
            q['scaled_jacobian'].min()
        """
        if self._metrics is not None:
            return self._metrics
        mesh = self.mesh
        nc = mesh.n_cells
        out = {k: np.full(nc, np.nan) for k in METRICS}
        flipped = []
        for blk, start in zip(mesh.blocks, mesh.offsets[:-1]):
            sl = slice(start, start + len(blk))
            if blk.kind == 'hex':
                h = _hex_jacobian(mesh.points, blk.conn, blk.order)
                if len(h['vol']) and (h['vol'] < 0).mean() > 0.5:
                    # left-handed convention: turn the whole block round
                    h = {'smin': -h['smax'], 'smax': -h['smin'],
                         'dmin': -h['dmax'], 'dmax': -h['dmin'],
                         'vol': -h['vol']}
                    flipped.append(f'hex{blk.order}')
                sj, vol = h['smin'], h['vol']
                pos = h['dmax'] > 0
                out['jacobian_ratio'][sl] = np.where(
                    pos, h['dmin'] / np.where(pos, h['dmax'], 1), -1.0)
            else:
                corners, vol = _linear_jacobian(mesh.points, blk.conn,
                                                blk.kind)
                if len(corners) and (corners.sum(1) < 0).mean() > 0.5:
                    corners = -corners
                    flipped.append(f'{blk.kind}{blk.order}')
                sj = corners.min(1)
            out['scaled_jacobian'][sl] = sj
            out['volume'][sl] = vol
            c = mesh.points[blk.corners()]
            e = np.array(_edges(blk.kind))
            lens = np.linalg.norm(c[:, e[:, 0]] - c[:, e[:, 1]], axis=2)
            out['aspect_ratio'][sl] = lens.max(1) / np.maximum(lens.min(1),
                                                               1e-300)
            skew = np.zeros(len(blk))
            for f in range(blk.n_faces):
                fc = blk.face_corners(f)
                ang = _angles(mesh.points[fc])
                ideal = 90.0 if fc.shape[1] == 4 else 60.0
                s = np.maximum((ang.max(1) - ideal) / (180.0 - ideal),
                               (ideal - ang.min(1)) / ideal)
                skew = np.maximum(skew, s)
            out['skewness'][sl] = skew
        self.meta['flipped'] = flipped
        self._neighbour_metrics(out)
        self._metrics = out
        return out

    def _neighbour_metrics(self, out) -> None:
        """Non-orthogonality and volume jump, over inner faces."""
        mesh = self.mesh
        faces, idx = topology.all_faces(mesh), topology.face_index(mesh)
        inner = idx['count'][idx['inverse']] == 2
        rows = np.nonzero(inner)[0]
        order = rows[np.argsort(idx['inverse'][rows], kind='stable')]
        ra, rb = order[0::2], order[1::2]
        ea, eb = faces['elem'][ra], faces['elem'][rb]
        ctr = np.empty((mesh.n_cells, 3))
        for blk, start in zip(mesh.blocks, mesh.offsets[:-1]):
            ctr[start:start + len(blk)] = mesh.points[blk.corners()].mean(1)
        # face normals from the corners of side a
        b_of, local = mesh.locate_cell(ea)
        fa = faces['face'][ra]
        normal = np.zeros((len(ra), 3))
        for b, blk in enumerate(mesh.blocks):
            for f in range(blk.n_faces):
                sel = (b_of == b) & (fa == f)
                if not sel.any():
                    continue
                c = mesh.points[blk.face_corners(f)[local[sel]]]
                if c.shape[1] == 4:
                    normal[sel] = np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])
                else:
                    normal[sel] = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
        d = ctr[eb] - ctr[ea]
        cos = np.abs(np.einsum('ij,ij->i', normal, d)) / np.maximum(
            np.linalg.norm(normal, axis=1) * np.linalg.norm(d, axis=1),
            1e-300)
        ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
        no = np.zeros(mesh.n_cells)
        np.maximum.at(no, ea, ang)
        np.maximum.at(no, eb, ang)
        out['non_orthogonality'][:] = no
        v = np.abs(out['volume'])
        r = np.maximum(v[ea], v[eb]) / np.maximum(np.minimum(v[ea], v[eb]),
                                                  1e-300)
        vr = np.ones(mesh.n_cells)
        np.maximum.at(vr, ea, r)
        np.maximum.at(vr, eb, r)
        out['volume_ratio'][:] = vr

    # -- reading the results --------------------------------------------------
    def field(self, name: str) -> np.ndarray:
        """One metric, ``(n_cells,)``."""
        if name not in METRICS:
            raise KeyError(f"no metric {name!r}; choose one of "
                           f"{sorted(METRICS)}")
        return self.metrics()[name]

    def summary(self) -> dict:
        """Statistics of every metric.

        Returns
        -------
        dict
            ``{name: {'min', 'p01', 'p50', 'p99', 'max', 'n', 'warn',
            'error'}}``: percentiles over the cells where the metric is
            defined, and how many cells pass the warning and error
            thresholds of :data:`METRICS` (on the bad side).
        """
        out = {}
        for name, spec in METRICS.items():
            v = self.metrics()[name]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            p = np.percentile(v, [1, 50, 99])
            row = {'min': float(v.min()), 'p01': float(p[0]),
                   'p50': float(p[1]), 'p99': float(p[2]),
                   'max': float(v.max()), 'n': int(len(v)),
                   'warn': 0, 'error': 0}
            for key in ('warn', 'error'):
                t = spec[key]
                if t is None:
                    continue
                bad = v <= t if spec['better'] == 'high' else v >= t
                row[key] = int(bad.sum())
            out[name] = row
        return out

    def worst(self, metric: str = 'scaled_jacobian', n: int = 20) -> list:
        """The *n* worst cells for *metric*.

        Returns
        -------
        list[dict]
            ``cell``, ``value``, ``kind``, ``order``, ``centroid`` and
            ``file_cell`` (index in the file, when the reader keeps it; add
            one for HORSES3D's numbering), worst first.
        """
        return worst_cells(self.mesh, self.field(metric),
                           METRICS[metric]['better'], n)

    def issues(self) -> list:
        """Findings in the format of :meth:`Census.check`.

        Inverted or flat cells (scaled Jacobian or Jacobian ratio <= 0) are
        errors; the other thresholds of :data:`METRICS` give warnings.
        """
        s = self.summary()
        out = []
        add = lambda level, msg: out.append(
            {'level': level, 'check': 'quality', 'message': msg})
        for name, row in s.items():
            spec = METRICS[name]
            word = spec['label']['en']
            if row['error']:
                add('error', f"{row['error']} cells with {word} <= "
                             f"{spec['error']:g} (inverted or flat)")
            elif row['warn']:
                op = '<=' if spec['better'] == 'high' else '>='
                add('warning', f"{row['warn']} cells with {word} {op} "
                               f"{spec['warn']:g}")
        for f in self.meta.get('flipped', []):
            add('info', f"{f} cells are left-handed in this file's "
                        "convention; signs were turned round")
        return out

    def report(self, bins: int = 10, width: int = 30) -> str:
        """Plain-text table and a histogram per metric."""
        s = self.summary()
        lines = [f"{'metric':<20}{'min':>10}{'p01':>10}{'p50':>10}"
                 f"{'p99':>10}{'max':>11}{'warn':>7}{'err':>6}"]
        for name, r in s.items():
            lines.append(f"{name:<20}{r['min']:>10.4g}{r['p01']:>10.4g}"
                         f"{r['p50']:>10.4g}{r['p99']:>10.4g}{r['max']:>11.4g}"
                         f"{r['warn']:>7}{r['error']:>6}")
        for name in ('scaled_jacobian', 'jacobian_ratio', 'skewness',
                     'non_orthogonality'):
            if name not in s:
                continue
            v = self.field(name)
            v = v[np.isfinite(v)]
            lo, hi = s[name]['min'], s[name]['max']
            if hi <= lo:
                continue
            h, edges = np.histogram(v, bins=bins, range=(lo, hi))
            lines += ['', name]
            top = max(h.max(), 1)
            for c, a, b in zip(h, edges[:-1], edges[1:]):
                bar = '#' * int(round(width * c / top))
                lines.append(f"  {a:>9.3g} - {b:<9.3g} {bar:<{width}} {c}")
        lines += [''] + [f"{i['level']:<8}{i['message']}"
                         for i in self.issues()]
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        state = 'computed' if self._metrics is not None else 'not computed'
        return f"<LEGOLAS.Quality of {self.mesh!r}: {state}>"
