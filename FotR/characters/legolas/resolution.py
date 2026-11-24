"""``LEGOLAS.Resolution``: is the mesh fine enough for an LES? (a priori)

Mesh design, not post-processing: sizes come from the geometry and wall
units from a skin-friction correlation, before any solution exists. The
real y+ of a run needs the wall shear of a ``.hsol`` and belongs to the
HORSES3D tools.

Per cell (:meth:`Resolution.sizes`):

``h``
    ``V**(1/3)``.
``h_min``, ``h_max``
    For hexahedra, the mean length of the four edges along each reference
    direction, measured along the high-order nodes; the smallest and the
    largest of the three. For the linear kinds, the shortest and longest
    corner edge.
``anisotropy``
    ``h_max / h_min``.
``delta``, ``delta_max``
    With the polynomial order ``p`` of the solution: ``h / (p + 1)`` and
    ``h_max / (p + 1)``, the usual effective resolution of a DG element
    with ``(p + 1)**3`` degrees of freedom.

At a wall (:meth:`Resolution.wall`), per wall face:

``y1``
    Thickness of the wall cell along the wall normal (extent of its corners
    measured from the plane of the wall face).
``y1_dof``
    With ``p``: height of the first solution node off the wall, the first
    inner Gauss-Lobatto point of order p (``0.5 y1`` for p = 2, ``0.276``
    for 3, ``0.173`` for 4). Meaningful for cells stacked on the wall.
``dt1``, ``dt2``
    Lengths of the wall face along its two directions (mean of opposite
    edges; for a triangle, the longest and shortest edge), divided by
    ``p + 1`` when ``p`` is given.
``growth``
    Thickness ratio between consecutive cells of the stack on that face,
    for ``layers - 1`` steps. The stack is followed through opposite faces
    (hexahedra and prisms), so it stops (NaN) at any other kind.

Wall units (:meth:`Resolution.yplus`): with ``U = 1``, ``nu = L / Re`` and
``u_tau = sqrt(Cf / 2)``, a length ``y`` is ``y Re sqrt(Cf / 2) / L`` in
wall units. ``Cf`` comes from a flat-plate correlation, turbulent
``0.026 Re_x**(-1/7)`` or laminar (Blasius) ``0.664 Re_x**(-1/2)``, with
``Re_x = Re x / L``: ``x`` is the distance from the leading edge along the
flow when a direction is given, or ``L`` otherwise. It is an estimate for
sizing the first cell, not a prediction of the flow.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import topology
from .quality import Quality, gauss_lobatto

__all__ = ['Resolution', 'first_node_fraction', 'WRLES']

#: Usual targets of a wall-resolved LES (Piomelli & Chasnov 1996;
#: Georgiadis, Rizzetta & Fureby 2010), in wall units: (low, high).
WRLES = {'y1_plus': (0.0, 1.0), 'dt1_plus': (50.0, 150.0),
         'dt2_plus': (15.0, 40.0)}

#: Opposite local face, for the kinds where a stack of cells has a
#: direction.
_OPPOSITE = {'hex': {0: 5, 5: 0, 1: 3, 3: 1, 2: 4, 4: 2},
             'prism': {0: 1, 1: 0}}


def first_node_fraction(p: int) -> float:
    """Height of the first inner Gauss-Lobatto node of order *p*, as a
    fraction of the element thickness."""
    if p < 1:
        raise ValueError('p must be >= 1')
    x, _ = gauss_lobatto(p)
    return float((x[1] + 1.0) / 2.0)


def _arc(line):
    """Arc length of polylines ``(m, n, 3)``."""
    return np.linalg.norm(np.diff(line, axis=1), axis=2).sum(1)


def _hex_directional(points, conn, order):
    """Mean arc length of the four edges along i, j and k: ``(m, 3)``."""
    n = order + 1
    x = points[conn].reshape(-1, n, n, n, 3)                 # [e, k, j, i]
    ends = (0, -1)
    along_i = np.mean([_arc(x[:, k, j, :, :]) for k in ends for j in ends], 0)
    along_j = np.mean([_arc(x[:, k, :, i, :]) for k in ends for i in ends], 0)
    along_k = np.mean([_arc(x[:, :, j, i, :]) for j in ends for i in ends], 0)
    return np.stack([along_i, along_j, along_k], 1)


def _edge_lengths(points, block):
    from .quality import _edges
    c = points[block.corners()]
    e = np.array(_edges(block.kind))
    return np.linalg.norm(c[:, e[:, 0]] - c[:, e[:, 1]], axis=2)


class Resolution:
    """Sizes of the cells and of the first cells at a wall, for an LES.

    Reached as ``lg.resolution``. See the module documentation for every
    quantity and for how wall units are estimated.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.
    """

    def __init__(self, legolas):
        self._lg = legolas
        self._sizes = None

    @property
    def mesh(self):
        """The mesh being measured."""
        return self._lg.mesh

    def _quality(self):
        q = getattr(self._lg, 'quality', None)
        return q if q is not None else Quality(SimpleNamespace(mesh=self.mesh))

    # -- sizes ----------------------------------------------------------------
    def sizes(self, p: int | None = None) -> dict:
        """Size of every cell.

        Parameters
        ----------
        p : int, optional
            Polynomial order of the solution; adds ``delta`` and
            ``delta_max``.

        Returns
        -------
        dict
            ``h``, ``h_min``, ``h_max``, ``anisotropy`` (and ``delta``,
            ``delta_max``), each ``(n_cells,)``.

        Examples
        --------
        ::

            s = lg.resolution.sizes(p=3)
            s['delta'].min(), s['anisotropy'].max()
        """
        if self._sizes is None:
            mesh = self.mesh
            vol = np.abs(self._quality().field('volume'))
            hmin = np.empty(mesh.n_cells)
            hmax = np.empty(mesh.n_cells)
            for blk, start in zip(mesh.blocks, mesh.offsets[:-1]):
                sl = slice(start, start + len(blk))
                if blk.kind == 'hex':
                    d = _hex_directional(mesh.points, blk.conn, blk.order)
                else:
                    d = _edge_lengths(mesh.points, blk)
                hmin[sl], hmax[sl] = d.min(1), d.max(1)
            self._sizes = {'h': np.cbrt(vol), 'h_min': hmin, 'h_max': hmax,
                           'anisotropy': hmax / np.maximum(hmin, 1e-300)}
        out = dict(self._sizes)
        if p is not None:
            out['delta'] = out['h'] / (p + 1)
            out['delta_max'] = out['h_max'] / (p + 1)
        return out

    # -- walls ----------------------------------------------------------------
    def _partner(self):
        """Arrays to find, for a row of all_faces, the other row of the face."""
        mesh = self.mesh
        idx = topology.face_index(mesh)
        if 'rows_by_face' not in mesh._cache:
            o = np.argsort(idx['inverse'], kind='stable')
            mesh._cache['rows_by_face'] = (o, idx['inverse'][o])
        return idx, mesh._cache['rows_by_face']

    def wall(self, zones, p: int | None = None, layers: int = 4) -> dict:
        """First cells on one or more wall boundaries.

        Parameters
        ----------
        zones : str or list[str]
            Boundary names (as in ``mesh.boundaries``).
        p : int, optional
            Polynomial order of the solution: adds ``y1_dof`` and divides
            the tangential sizes by ``p + 1``.
        layers : int
            Cells followed away from the wall for ``growth``. Default 4.

        Returns
        -------
        dict
            Per wall face: ``zone``, ``elem``, ``face``, ``centroid``,
            ``normal`` (unit, into the domain), ``y1``, ``y1_dof`` (if
            ``p``), ``dt1``, ``dt2``, ``growth`` ``(n_faces, layers - 1)``;
            and ``p``.
        """
        mesh = self.mesh
        if isinstance(zones, str):
            zones = [zones]
        elem, face, zone = [], [], []
        for z in zones:
            bc = mesh.boundaries[z]
            elem.append(bc.elem)
            face.append(bc.face)
            zone.append(np.full(len(bc), z, dtype=object))
        elem = np.concatenate(elem) if elem else np.empty(0, np.int64)
        face = np.concatenate(face) if face else np.empty(0, np.int64)
        zone = np.concatenate(zone) if zone else np.empty(0, object)
        nf = len(elem)
        ctr = np.empty((nf, 3))
        nrm = np.empty((nf, 3))
        dt = np.empty((nf, 2))
        b_of, local = mesh.locate_cell(elem)
        for b, blk in enumerate(mesh.blocks):
            for f in range(blk.n_faces):
                sel = np.nonzero((b_of == b) & (face == f))[0]
                if not len(sel):
                    continue
                c = mesh.points[blk.face_corners(f)[local[sel]]]
                ctr[sel] = c.mean(1)
                if c.shape[1] == 4:
                    n = np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])
                    l1 = 0.5 * (np.linalg.norm(c[:, 1] - c[:, 0], axis=1)
                                + np.linalg.norm(c[:, 2] - c[:, 3], axis=1))
                    l2 = 0.5 * (np.linalg.norm(c[:, 3] - c[:, 0], axis=1)
                                + np.linalg.norm(c[:, 2] - c[:, 1], axis=1))
                    dt[sel] = np.stack([np.maximum(l1, l2),
                                        np.minimum(l1, l2)], 1)
                else:
                    n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
                    e = np.stack([np.linalg.norm(c[:, (k + 1) % 3] - c[:, k],
                                                 axis=1) for k in range(3)], 1)
                    dt[sel] = np.stack([e.max(1), e.min(1)], 1)
                nrm[sel] = n / np.maximum(np.linalg.norm(n, axis=1,
                                                         keepdims=True),
                                          1e-300)
        # orient the normal into the domain: towards the cell's corners
        cell_ctr = self._cell_centroids()
        flip = np.einsum('ij,ij->i', cell_ctr[elem] - ctr, nrm) < 0
        nrm[flip] *= -1

        def thickness(cells):
            out = np.full(nf, np.nan)
            ok = cells >= 0
            if not ok.any():
                return out
            bb, ll = mesh.locate_cell(cells[ok])
            proj = np.empty(ok.sum())
            lo = np.empty(ok.sum())
            for b, blk in enumerate(mesh.blocks):
                s = bb == b
                if s.any():
                    c = mesh.points[blk.corners()[ll[s]]]
                    d = np.einsum('ijk,ik->ij', c - ctr[ok][s][:, None],
                                  nrm[ok][s])
                    proj[s], lo[s] = d.max(1), d.min(1)
            out[ok] = proj - lo
            return out

        y1 = thickness(elem)
        growth = np.full((nf, max(layers - 1, 0)), np.nan)
        cur_e, cur_f = elem.copy(), face.copy()
        prev = y1
        idx, (order, sorted_uid) = self._partner()
        for k in range(max(layers - 1, 0)):
            # opposite face of the current cell, then the cell behind it
            nxt_e = np.full(nf, -1, np.int64)
            nxt_f = np.full(nf, -1, np.int64)
            ok = cur_e >= 0
            bb, ll = mesh.locate_cell(np.where(ok, cur_e, 0))
            opp = np.full(nf, -1, np.int64)
            for b, blk in enumerate(mesh.blocks):
                table = _OPPOSITE.get(blk.kind)
                if table is None:
                    continue
                s = ok & (bb == b)
                for f_in, f_out in table.items():
                    opp[s & (cur_f == f_in)] = f_out
            has = opp >= 0
            if has.any():
                rows = topology.face_rows(mesh, cur_e[has], opp[has])
                uid = idx['inverse'][rows]
                pos = np.searchsorted(sorted_uid, uid)
                a = order[pos]
                bpos = np.minimum(pos + 1, len(order) - 1)
                b2 = order[bpos]
                two = (bpos != pos) & (sorted_uid[bpos] == uid)
                other = np.where(a == rows, b2, a)
                other = np.where(two, other, -1)
                faces = topology.all_faces(mesh)
                hi = np.nonzero(has)[0]
                good = other >= 0
                nxt_e[hi[good]] = faces['elem'][other[good]]
                nxt_f[hi[good]] = faces['face'][other[good]]
            t = thickness(nxt_e)
            growth[:, k] = t / prev
            prev, cur_e, cur_f = t, nxt_e, nxt_f
        out = {'zone': zone, 'elem': elem, 'face': face, 'centroid': ctr,
               'normal': nrm, 'y1': y1, 'dt1': dt[:, 0], 'dt2': dt[:, 1],
               'growth': growth, 'p': p}
        if p is not None:
            out['y1_dof'] = y1 * first_node_fraction(p)
            out['dt1'] = out['dt1'] / (p + 1)
            out['dt2'] = out['dt2'] / (p + 1)
        return out

    def _corner_points(self, elem, face):
        """All the corner points of the given faces, ``(k, 3)``."""
        mesh = self.mesh
        b_of, local = mesh.locate_cell(elem)
        pts = []
        for b, blk in enumerate(mesh.blocks):
            for f in range(blk.n_faces):
                sel = (b_of == b) & (face == f)
                if sel.any():
                    pts.append(mesh.points[blk.face_corners(f)[local[sel]]]
                               .reshape(-1, 3))
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def _cell_centroids(self):
        mesh = self.mesh
        if 'cell_centroids' not in mesh._cache:
            c = np.empty((mesh.n_cells, 3))
            for blk, start in zip(mesh.blocks, mesh.offsets[:-1]):
                c[start:start + len(blk)] = mesh.points[blk.corners()].mean(1)
            mesh._cache['cell_centroids'] = c
        return mesh._cache['cell_centroids']

    @staticmethod
    def skin_friction(re_x, correlation: str = 'turbulent'):
        """Flat-plate ``Cf`` at ``Re_x``: ``'turbulent'`` (0.026 Re_x^-1/7)
        or ``'laminar'`` (Blasius, 0.664 Re_x^-1/2)."""
        re_x = np.asarray(re_x, float)
        if correlation == 'turbulent':
            return 0.026 * re_x ** (-1.0 / 7.0)
        if correlation == 'laminar':
            return 0.664 * re_x ** -0.5
        raise ValueError("correlation must be 'turbulent' or 'laminar'")

    def yplus(self, zones, Re: float, L: float = 1.0, p: int | None = None,
              correlation: str = 'turbulent', flow_dir=None,
              x0: float | None = None, layers: int = 4) -> dict:
        """Estimate y+ and the tangential spacings in wall units.

        Parameters
        ----------
        zones : str or list[str]
            Wall boundaries.
        Re : float
            Reynolds number based on ``L`` and the free-stream velocity.
        L : float
            Reference length of ``Re``, in mesh units. Default 1.
        p : int, optional
            Polynomial order of the solution (see :meth:`wall`).
        correlation : str
            ``'turbulent'`` (default) or ``'laminar'``.
        flow_dir : array_like, optional
            Free-stream direction. With it, ``Re_x`` uses each face's
            distance from the leading edge (the most upstream point of the
            zones, or ``x0``); without it, ``Re_x = Re``.
        x0 : float, optional
            Position of the leading edge along ``flow_dir``.
        layers : int
            Passed to :meth:`wall`.

        Returns
        -------
        dict
            Everything :meth:`wall` returns, plus ``cf``, ``re_x``,
            ``y1_plus``, ``y1_dof_plus`` (with ``p``), ``dt1_plus``,
            ``dt2_plus``, and ``summary`` (min, median and max of each,
            and the share of faces inside the :data:`WRLES` ranges).

        Examples
        --------
        ::

            r = lg.resolution.yplus('esphere_surface', Re=400, p=3,
                                    correlation='laminar')
            r['summary']['y1_dof_plus']
        """
        w = self.wall(zones, p=p, layers=layers)
        if flow_dir is not None:
            d = np.asarray(flow_dir, float)
            d = d / np.linalg.norm(d)
            s = w['centroid'] @ d
            if x0 is None:
                # the leading edge: the most upstream corner of the zones
                start = (self._corner_points(w['elem'], w['face']) @ d).min()
            else:
                start = float(x0)
            x = np.maximum(s - start, 1e-3 * L)
        else:
            x = np.full(len(w['y1']), float(L))
        re_x = Re * x / L
        cf = self.skin_friction(re_x, correlation)
        unit = Re * np.sqrt(cf / 2.0) / L          # 1 / (nu / u_tau)
        w.update({'cf': cf, 're_x': re_x, 'y1_plus': w['y1'] * unit,
                  'dt1_plus': w['dt1'] * unit, 'dt2_plus': w['dt2'] * unit,
                  'Re': Re, 'L': L, 'correlation': correlation})
        if p is not None:
            w['y1_dof_plus'] = w['y1_dof'] * unit
        summ = {}
        for key in ('y1_plus', 'y1_dof_plus', 'dt1_plus', 'dt2_plus'):
            if key not in w:
                continue
            v = w[key][np.isfinite(w[key])]
            if not len(v):
                continue
            row = {'min': float(v.min()), 'p50': float(np.median(v)),
                   'max': float(v.max())}
            target = WRLES.get(key if key != 'y1_dof_plus' else 'y1_plus')
            if target:
                row['in_wrles'] = float(((v >= target[0])
                                         & (v <= target[1])).mean())
            summ[key] = row
        w['summary'] = summ
        return w

    # -- for the viewer and for text ------------------------------------------
    def fields(self, p: int | None = None, walls=(), Re: float | None = None,
               L: float = 1.0, correlation: str = 'turbulent',
               flow_dir=None) -> dict:
        """Per-cell arrays for :meth:`LEGOLAS.Viewer.show` (``fields=``).

        Always ``h``, ``anisotropy`` (and ``delta`` with ``p``); with
        ``walls``, ``y1`` of the wall cells (NaN elsewhere); with ``Re``
        too, ``y1_plus``. A cell on two wall faces keeps the smaller value.

        Returns
        -------
        dict
            ``{name: {'values', 'label', 'better'}}``.
        """
        s = self.sizes(p)
        out = {
            'h': {'values': s['h'], 'better': None,
                  'label': {'es': 'tamaño V^(1/3)', 'en': 'size V^(1/3)'}},
            'anisotropy': {'values': s['anisotropy'], 'better': 'low',
                           'label': {'es': 'anisotropía', 'en': 'anisotropy'}},
        }
        if p is not None:
            out['delta'] = {'values': s['delta'], 'better': 'low',
                            'label': {'es': f'Δ DG (p={p})',
                                      'en': f'DG Δ (p={p})'}}
        if walls:
            if Re is not None:
                w = self.yplus(walls, Re, L=L, p=p, correlation=correlation,
                               flow_dir=flow_dir, layers=1)
            else:
                w = self.wall(walls, p=p, layers=1)
            n = self.mesh.n_cells

            def per_cell(v):
                arr = np.full(n, np.inf)
                np.minimum.at(arr, w['elem'], v)
                arr[np.isinf(arr)] = np.nan
                return arr
            out['y1'] = {'values': per_cell(w['y1']), 'better': 'low',
                         'label': {'es': 'altura 1.ª celda',
                                   'en': 'first cell height'}}
            if Re is not None:
                key = 'y1_dof_plus' if p is not None else 'y1_plus'
                out['y1_plus'] = {'values': per_cell(w[key]), 'better': 'low',
                                  'label': {'es': 'y⁺ estimado',
                                            'en': 'estimated y⁺'}}
        return out

    def report(self, walls=(), p: int | None = None, Re: float | None = None,
               L: float = 1.0, correlation: str = 'turbulent',
               flow_dir=None) -> str:
        """Plain-text summary: sizes, and wall units if ``walls`` and ``Re``."""
        s = self.sizes(p)
        lines = [f"{'size':<12}{'min':>11}{'p50':>11}{'max':>11}"]
        for k in ('h', 'h_min', 'h_max', 'anisotropy', 'delta'):
            if k in s:
                v = s[k]
                lines.append(f"{k:<12}{v.min():>11.4g}{np.median(v):>11.4g}"
                             f"{v.max():>11.4g}")
        if walls:
            w = (self.yplus(walls, Re, L=L, p=p, correlation=correlation,
                            flow_dir=flow_dir) if Re is not None
                 else self.wall(walls, p=p))
            names = ', '.join([walls] if isinstance(walls, str) else walls)
            lines += ['', f"wall: {names} ({len(w['y1'])} faces"
                          + (f", p = {p}" if p else '') + ')']
            for k in ('y1', 'y1_dof', 'dt1', 'dt2'):
                if k in w:
                    v = w[k]
                    lines.append(f"{k:<12}{np.nanmin(v):>11.4g}"
                                 f"{np.nanmedian(v):>11.4g}"
                                 f"{np.nanmax(v):>11.4g}")
            g = w['growth']
            if g.size and np.isfinite(g).any():
                lines.append(f"{'growth':<12}{np.nanmin(g):>11.4g}"
                             f"{np.nanmedian(g):>11.4g}{np.nanmax(g):>11.4g}")
            if Re is not None:
                lines += ['', f"wall units ({correlation}, Re = {Re:g}, "
                              f"L = {L:g}); WRLES targets: y+ <= 1, "
                              "dx+ 50-150, dz+ 15-40"]
                for k, r in w['summary'].items():
                    share = (f"   {100 * r['in_wrles']:.0f} % in range"
                             if 'in_wrles' in r else '')
                    lines.append(f"{k:<12}{r['min']:>11.4g}{r['p50']:>11.4g}"
                                 f"{r['max']:>11.4g}{share}")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"<LEGOLAS.Resolution of {self.mesh!r}>"
