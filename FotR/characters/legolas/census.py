"""``LEGOLAS.Census``: what a mesh contains, counted and checked.

* :meth:`Census.summary` / :meth:`Census.report` -- the inventory (L1).
* :meth:`Census.check_bcs` -- zones of the mesh against the BCs of a
  HORSES3D ``.control`` and a PyHOPE ``parameter.ini`` (L3).
* :meth:`Census.watertight` -- the skin of the domain: faces with no BC,
  non-manifold faces and edges, holes (L3).
* :meth:`Census.duplicates` -- coincident nodes, cracks, collapsed cells
  (L3).
* :meth:`Census.compare` -- two meshes cell by cell and BC by BC (L3).
* :meth:`Census.check` -- all of the above as one list of findings.

Every check returns a dict with its numbers and an ``issues`` list of
``{'level', 'check', 'message'}``, ``level`` being ``'error'`` (the solver
or PyHOPE will stop, or the mesh is broken), ``'warning'`` (probably a
mistake) or ``'info'``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from . import topology

__all__ = ['Census']


class Census:
    """Inventory of a mesh: sizes, element kinds, zones and boundaries.

    Reached as ``lg.census`` on a :class:`~FotR.characters.legolas.LEGOLAS`
    instance. Results are computed once and cached.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.
    """

    def __init__(self, legolas):
        self._lg = legolas
        self._summary = None

    @property
    def mesh(self):
        """The mesh being counted."""
        return self._lg.mesh

    def summary(self) -> dict:
        """Count everything in the mesh.

        Returns
        -------
        dict
            ``path``, ``format``, ``ngeo``, ``n_points``, ``n_cells``,
            ``cells`` (one entry per kind and order), ``bbox`` (``min``,
            ``max``, ``size``), ``zones`` (name → cells), ``boundaries``
            (one entry per boundary: ``name``, ``bc_type``, ``n_faces``,
            ``area``, ``interior_faces``, the faces that have a cell on each
            side), ``exterior_faces``, ``unassigned_faces`` (exterior faces in
            no boundary), ``nonmanifold_faces`` (shared by more than two
            cells), ``orphan_nodes`` (used by no cell) and the reader's
            ``warnings``.

        Examples
        --------
        ::

            lg = LEGOLAS('MESH/my_esphere_v1g2_mesh.h5')
            s = lg.census.summary()
            s['boundaries'][0]['n_faces']
            print(lg.census)               # the same, as a table
        """
        if self._summary is not None:
            return self._summary
        mesh = self.mesh
        idx = topology.face_index(mesh)
        count_of_row = idx['count'][idx['inverse']]
        exterior = count_of_row == 1
        in_bc = np.zeros(len(count_of_row), dtype=bool)

        boundaries = []
        for name, bc in mesh.boundaries.items():
            rows = topology.face_rows(mesh, bc.elem, bc.face)
            in_bc[rows] = True
            boundaries.append({
                'name': name,
                'bc_type': bc.bc_type,
                'n_faces': int(len(bc)),
                'area': float(topology.face_area(mesh, bc.elem, bc.face).sum()),
                'interior_faces': int((count_of_row[rows] >= 2).sum()),
            })

        zone_cells = {}
        for block in mesh.blocks:
            z, n = np.unique(block.zone, return_counts=True)
            for zi, ni in zip(z, n):
                name = mesh.zones.get(int(zi), f'zone_{int(zi)}')
                zone_cells[name] = zone_cells.get(name, 0) + int(ni)

        pmin, pmax = mesh.points.min(0), mesh.points.max(0)
        self._summary = {
            'path': mesh.meta.get('path'),
            'format': mesh.meta.get('format'),
            'ngeo': mesh.ngeo,
            'n_points': mesh.n_points,
            'n_cells': mesh.n_cells,
            'cells': [{'kind': b.kind, 'order': b.order, 'n': len(b)}
                      for b in mesh.blocks],
            'bbox': {'min': pmin.tolist(), 'max': pmax.tolist(),
                     'size': (pmax - pmin).tolist()},
            'zones': zone_cells,
            'boundaries': boundaries,
            'exterior_faces': int(exterior.sum()),
            'unassigned_faces': int((exterior & ~in_bc).sum()),
            'nonmanifold_faces': int((idx['count'] > 2).sum()),
            'orphan_nodes': int((~topology.used_nodes(mesh)).sum()),
            'warnings': list(mesh.meta.get('warnings', [])),
        }
        return self._summary

    # -- L3: checks -----------------------------------------------------------
    def zone_status(self) -> dict:
        """Where each boundary's faces lie.

        Returns
        -------
        dict
            ``{name: {'n_faces', 'interior_faces', 'status'}}`` with
            ``status`` one of ``'empty'`` (no faces), ``'boundary'`` (all on
            the skin of the domain), ``'interior'`` (all with a cell on each
            side: an inner zone) or ``'mixed'``.
        """
        out = {}
        for b in self.summary()['boundaries']:
            n, i = b['n_faces'], b['interior_faces']
            status = ('empty' if n == 0 else 'interior' if i == n
                      else 'boundary' if i == 0 else 'mixed')
            out[b['name']] = {'n_faces': n, 'interior_faces': i,
                              'status': status}
        return out

    def check_bcs(self, control=None, ini=None) -> dict:
        """Compare the zones of the mesh with the BCs declared in files.

        Names are compared without case: HOPR writes them in lower case
        (``sidea``) while ANSA and the files keep ``sideA``.

        Rules, from what HORSES3D and PyHOPE actually do:

        * HORSES3D needs a ``#define boundary`` with a ``type`` for every
          zone that has faces, **inner zones included** (their type is never
          applied, but the solver stops without it). A zone with no faces
          needs nothing.
        * PyHOPE needs a ``BoundaryName`` for every surface group of a
          ``.cgns``/``.msh``, and an inner zone must have type 100: any other
          type stops it with "internal face with multiple BCs". Type 100 on
          a zone that lies on the skin of the domain is a mistake too.
        * A name declared in a file but absent from the mesh is reported as
          a warning: harmless for the solver, but usually a sign that the
          file belongs to another mesh.

        Parameters
        ----------
        control : str or path-like, optional
            HORSES3D ``.control`` file.
        ini : str or path-like, optional
            PyHOPE ``parameter.ini``.

        Returns
        -------
        dict
            ``rows``: one per name (union of mesh and files), with
            ``name``, ``mesh`` (status, None if absent), ``n_faces``,
            ``control`` (type, None if absent), ``ini`` (type tuple);
            and ``issues``.

        Examples
        --------
        ::

            lg = LEGOLAS('MESH/Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
            r = lg.census.check_bcs(control='wind_turbine.control',
                                    ini='parameter.ini')
            print(lg.census.bcs_report(r))
        """
        from .bcfiles import read_control_bcs, read_ini_bcs

        issues = []
        add = lambda level, msg: issues.append(
            {'level': level, 'check': 'bcs', 'message': msg})
        zones = self.zone_status()
        fmt = self.mesh.meta.get('format')

        def index(entries, what):
            out = {}
            for e in entries:
                key = e['name'].lower()
                if key in out:
                    add('warning', f"{what}: '{e['name']}' is declared twice "
                                   f"(lines {out[key]['line']} and {e['line']})")
                    continue
                out[key] = e
            return out

        ctrl = (index(read_control_bcs(control), os.path.basename(control))
                if control else None)
        hopr = (index(read_ini_bcs(ini), os.path.basename(ini))
                if ini else None)
        if ctrl is not None:
            for e in ctrl.values():
                if e['type'] is None:
                    add('warning', f"control: block of '{e['name']}' (line "
                                   f"{e['line']}) has no 'type'")

        rows, seen = [], set()
        for name, z in zones.items():
            key = name.lower()
            seen.add(key)
            c = ctrl.get(key) if ctrl is not None else None
            h = hopr.get(key) if hopr is not None else None
            rows.append({'name': name, 'mesh': z['status'],
                         'n_faces': z['n_faces'],
                         'control': c['type'] if c else None,
                         'ini': h['type'] if h else None})
            where = (f"'{name}' ({z['n_faces']} faces, {z['status']})")
            if ctrl is not None and c is None:
                if z['status'] == 'empty':
                    add('info', f"{where}: not in the control; fine, it has "
                                "no faces")
                elif z['status'] == 'interior':
                    add('error', f"{where}: not in the control. HORSES3D "
                                 "wants a type even for inner zones (it is "
                                 "never applied; FreeSlipWall will do)")
                else:
                    add('error', f"{where}: not in the control: HORSES3D "
                                 "has no condition for these faces")
            if c is not None and c['name'] != name:
                add('info', f"'{name}' is '{c['name']}' in the control "
                            "(case does not matter)")
            if hopr is not None:
                if h is None:
                    if z['status'] == 'empty':
                        add('info', f"{where}: not in parameter.ini")
                    else:
                        level = 'error' if fmt in ('cgns', 'gmsh') else 'warning'
                        add(level, f"{where}: not in parameter.ini; PyHOPE "
                                   "needs a BoundaryName for it")
                elif h['type']:
                    kind = h['type'][0]
                    if z['status'] in ('interior', 'mixed') and kind != 100:
                        add('error', f"{where}: type {kind} in parameter.ini "
                                     "(line {}), but its faces are inner: "
                                     "PyHOPE wants 100 ('internal face with "
                                     "multiple BCs')".format(h['line']))
                    if z['status'] == 'boundary' and kind == 100:
                        add('error', f"{where}: type 100 (inner) in "
                                     f"parameter.ini (line {h['line']}), but "
                                     "all its faces are on the skin")
        for entries, label in ((ctrl, 'the control'),
                               (hopr, 'parameter.ini')):
            if entries is None:
                continue
            for key, e in entries.items():
                if key in seen:
                    continue
                seen.add(key)
                c = (ctrl or {}).get(key)
                h = (hopr or {}).get(key)
                rows.append({'name': e['name'], 'mesh': None, 'n_faces': 0,
                             'control': c['type'] if c else None,
                             'ini': h['type'] if h else None})
                where = ' and '.join(
                    f"{lab} (line {x['line']})"
                    for lab, x in (('the control', c), ('parameter.ini', h))
                    if x)
                add('warning', f"'{e['name']}': declared in {where} but not "
                               "in the mesh")
        return {'rows': rows, 'issues': issues,
                'control': os.fspath(control) if control else None,
                'ini': os.fspath(ini) if ini else None}

    def watertight(self) -> dict:
        """Check the skin of the domain.

        The exterior faces of a sound volume mesh form closed surfaces: every
        edge of the skin belongs to exactly two exterior faces. An edge with
        one face is a hole (typically a non-conforming interface or two zones
        whose nodes were not merged); with three or more, surfaces touch
        along a line.

        Returns
        -------
        dict
            ``exterior_faces``, ``unassigned_faces`` (in no boundary),
            ``nonmanifold_faces`` (more than two cells), ``open_edges``,
            ``nonmanifold_edges``, ``faces_in_two_boundaries``,
            ``mixed_boundaries`` (names of BCs partly inner, partly on the
            skin), ``closed`` (no open edges) and ``issues``.
        """
        mesh = self.mesh
        s = self.summary()
        idx = topology.face_index(mesh)
        elem, face = topology.exterior_faces(mesh)
        b_of, local = mesh.locate_cell(elem)
        edges = []
        for b, block in enumerate(mesh.blocks):
            for f in range(block.n_faces):
                sel = (b_of == b) & (face == f)
                if sel.any():
                    c = block.face_corners(f)[local[sel]]
                    k = c.shape[1]
                    for i in range(k):
                        edges.append(np.sort(np.stack(
                            [c[:, i], c[:, (i + 1) % k]], 1), axis=1))
        if edges:
            e = np.vstack(edges)
            code = e[:, 0] * np.int64(mesh.n_points) + e[:, 1]
            _, count = np.unique(code, return_counts=True)
        else:
            count = np.zeros(0, np.int64)
        rows_all = np.concatenate([
            topology.face_rows(mesh, bc.elem, bc.face)
            for bc in mesh.boundaries.values()] or [np.zeros(0, np.int64)])
        _, per_row = np.unique(rows_all, return_counts=True)
        zs = self.zone_status()
        out = {
            'exterior_faces': s['exterior_faces'],
            'unassigned_faces': s['unassigned_faces'],
            'nonmanifold_faces': int((idx['count'] > 2).sum()),
            'open_edges': int((count == 1).sum()),
            'nonmanifold_edges': int((count > 2).sum()),
            'faces_in_two_boundaries': int((per_row > 1).sum()),
            'mixed_boundaries': [n for n, z in zs.items()
                                 if z['status'] == 'mixed'],
        }
        out['closed'] = out['open_edges'] == 0
        issues = []
        add = lambda level, msg: issues.append(
            {'level': level, 'check': 'watertight', 'message': msg})
        if out['unassigned_faces']:
            add('error', f"{out['unassigned_faces']} exterior faces belong to "
                         "no boundary (zone __unassigned__ in the viewer)")
        if out['nonmanifold_faces']:
            add('error', f"{out['nonmanifold_faces']} faces shared by more "
                         "than two cells")
        if out['open_edges']:
            add('error', f"{out['open_edges']} edges of the skin belong to a "
                         "single exterior face: the skin has holes "
                         "(non-conforming or unmerged interfaces?)")
        if out['nonmanifold_edges']:
            add('warning', f"{out['nonmanifold_edges']} edges of the skin "
                           "belong to more than two exterior faces")
        if out['faces_in_two_boundaries']:
            add('warning', f"{out['faces_in_two_boundaries']} faces are in "
                           "two boundaries at once")
        for name in out['mixed_boundaries']:
            z = zs[name]
            add('warning', f"'{name}': {z['interior_faces']} of "
                           f"{z['n_faces']} faces are inner, the rest on the "
                           "skin")
        out['issues'] = issues
        return out

    def duplicates(self, rtol: float = 1e-9) -> dict:
        """Find nodes that coincide in space but are different nodes.

        Three things come out of the same search, and they mean different
        things:

        * duplicates among **high-order nodes only** (edge or face nodes
          written once per element): harmless for the topology, which only
          looks at corners; PyHOPE merges them when it writes the ``.h5``;
        * coincident **exterior faces** (cracks): two faces in the same
          place that the topology sees as different, so the domain is cut
          along them and both count as skin;
        * **collapsed cells**: two nodes of one cell in the same place.

        Parameters
        ----------
        rtol : float
            Tolerance relative to the bounding-box diagonal.

        Returns
        -------
        dict
            ``tol``, ``groups`` (sets of coincident nodes), ``extra_nodes``
            (nodes that merging would remove), ``corner_groups`` (groups
            with two or more corner nodes), ``highorder_groups``,
            ``crack_faces``, ``collapsed_cells`` and ``issues``.
        """
        mesh = self.mesh
        label, first, tol = topology.coincident_labels(mesh.points, rtol)
        size = np.bincount(label)
        dup = size > 1
        corner = np.zeros(mesh.n_points, dtype=bool)
        for block in mesh.blocks:
            corner[block.corners().ravel()] = True
        corners_per_group = np.bincount(label[corner],
                                        minlength=len(size))
        corner_groups = int((corners_per_group > 1).sum())
        elem, face = topology.exterior_faces(mesh)
        b_of, local = mesh.locate_cell(elem)
        keys = np.full((len(elem), 4), -1, dtype=np.int64)
        for b, block in enumerate(mesh.blocks):
            for f in range(block.n_faces):
                sel = (b_of == b) & (face == f)
                if sel.any():
                    keys[sel] = topology.face_keys(
                        label[block.face_corners(f)[local[sel]]])
        inv, cnt = topology.group_keys(keys)
        crack = int((cnt[inv] > 1).sum()) if len(inv) else 0
        collapsed = 0
        for block in mesh.blocks:
            lab = np.sort(label[block.corners()], axis=1)
            collapsed += int((np.diff(lab, axis=1) == 0).any(1).sum())
        out = {'tol': tol, 'groups': int(dup.sum()),
               'extra_nodes': int(mesh.n_points - len(size)),
               'corner_groups': corner_groups,
               'highorder_groups': int(dup.sum()) - corner_groups,
               'crack_faces': crack, 'collapsed_cells': collapsed}
        issues = []
        add = lambda level, msg: issues.append(
            {'level': level, 'check': 'duplicates', 'message': msg})
        if crack:
            add('error', f"{crack} exterior faces lie on top of another one: "
                         "the mesh is cut along them (unmerged nodes)")
        if collapsed:
            add('error', f"{collapsed} cells have two corners in the same "
                         "place (collapsed)")
        if corner_groups and not crack:
            add('warning', f"{corner_groups} groups of coincident corner "
                           "nodes that form no crack")
        if out['highorder_groups']:
            add('info', f"{out['highorder_groups']} groups of coincident "
                        f"high-order nodes ({out['extra_nodes']} extra nodes);"
                        " harmless, PyHOPE merges them")
        out['issues'] = issues
        return out

    def compare(self, other, rtol: float = 1e-6) -> dict:
        """Compare this mesh with another one, cell by cell and BC by BC.

        Meant for the same mesh in two formats (``.cgns`` and its ``.h5``)
        or at two geometric orders (g2 and g4 of one ANSA mesh): cells are
        paired by the centroid of their corners, so the comparison does not
        depend on how the files number them, and the corners must coincide.
        Areas of curved boundaries differ a little between orders; that is
        expected and reported, not flagged.

        Parameters
        ----------
        other : LEGOLAS, Mesh or path
            The mesh to compare with.
        rtol : float
            Tolerance relative to this mesh's bounding-box diagonal.

        Returns
        -------
        dict
            ``cells`` (both counts), ``matched_cells``, ``max_corner_diff``,
            ``same_corners``, ``boundaries`` (rows ``name``, ``faces``,
            ``area`` as pairs and ``area_rel_diff``), ``only_here``,
            ``only_there``, ``orders`` and ``issues``.
        """
        from scipy.spatial import cKDTree
        from .mesh import Mesh

        if isinstance(other, Mesh):
            om = other
        elif hasattr(other, 'mesh'):
            om = other.mesh
        else:
            from .io import read
            om = read(other)
        a, b = self.mesh, om
        tol = rtol * float(np.linalg.norm(a.points.max(0) - a.points.min(0)))

        def cell_corners(m):
            """Corner coordinates of every cell, padded to 8 with NaN."""
            out = np.full((m.n_cells, 8, 3), np.nan)
            for blk, start in zip(m.blocks, m.offsets[:-1]):
                c = m.points[blk.corners()]
                out[start:start + len(blk), :c.shape[1]] = c
            return out

        xa, xb = cell_corners(a), cell_corners(b)
        matched, maxdiff = 0, 0.0
        if len(xa) and len(xb):
            ca, cb = np.nanmean(xa, 1), np.nanmean(xb, 1)
            dist, j = cKDTree(cb).query(ca)
            ok = dist <= tol
            na = (~np.isnan(xa[:, :, 0])).sum(1)
            nb = (~np.isnan(xb[:, :, 0])).sum(1)
            ok &= na == nb[j]
            matched = int(ok.sum())
            if matched:
                pa, pb = xa[ok], xb[j[ok]]
                # each corner against the nearest corner of its twin
                d = np.linalg.norm(pa[:, :, None, :] - pb[:, None, :, :],
                                   axis=-1)
                d = np.where(np.isnan(d), np.inf, d).min(2)
                d = np.where(np.isinf(d), 0.0, d)
                maxdiff = float(d.max())
        bsa = {x['name'].lower(): x for x in self.summary()['boundaries']}
        other_census = Census(SimpleNamespace(mesh=b))
        bsb = {x['name'].lower(): x
               for x in other_census.summary()['boundaries']}
        rows = []
        for key in [k for k in bsa if k in bsb]:
            x, y = bsa[key], bsb[key]
            ref = max(abs(x['area']), 1e-300)
            rows.append({'name': x['name'],
                         'faces': (x['n_faces'], y['n_faces']),
                         'area': (x['area'], y['area']),
                         'area_rel_diff': abs(x['area'] - y['area']) / ref})
        out = {'cells': (a.n_cells, b.n_cells), 'matched_cells': matched,
               'max_corner_diff': maxdiff, 'tol': tol,
               'same_corners': (matched == a.n_cells == b.n_cells
                                and maxdiff <= tol),
               'orders': (a.ngeo, b.ngeo), 'boundaries': rows,
               'only_here': [bsa[k]['name'] for k in bsa if k not in bsb],
               'only_there': [bsb[k]['name'] for k in bsb if k not in bsa]}
        issues = []
        add = lambda level, msg: issues.append(
            {'level': level, 'check': 'compare', 'message': msg})
        if a.n_cells != b.n_cells:
            add('error', f"different number of cells: {a.n_cells} vs "
                         f"{b.n_cells}")
        if matched < min(a.n_cells, b.n_cells):
            add('error', f"only {matched} of {a.n_cells} cells have a twin "
                         "with the same corners in the other mesh")
        elif maxdiff > tol:
            add('warning', f"corners differ by up to {maxdiff:.3g} "
                           f"(tolerance {tol:.3g})")
        for r in rows:
            if r['faces'][0] != r['faces'][1]:
                add('error', f"'{r['name']}': {r['faces'][0]} vs "
                             f"{r['faces'][1]} faces")
            elif r['area_rel_diff'] > 1e-3 and a.ngeo == b.ngeo:
                add('warning', f"'{r['name']}': areas differ by "
                               f"{100 * r['area_rel_diff']:.2g} %")
        for n in out['only_here']:
            add('warning', f"'{n}' only in this mesh")
        for n in out['only_there']:
            add('warning', f"'{n}' only in the other mesh")
        out['issues'] = issues
        return out

    def check(self, control=None, ini=None, duplicates: bool = True) -> dict:
        """Run every check and collect the findings.

        Parameters
        ----------
        control, ini : str or path-like, optional
            Passed to :meth:`check_bcs` (skipped when both are None).
        duplicates : bool
            Run :meth:`duplicates` too (a k-d tree over all nodes; a few
            seconds for millions of nodes). Default True.

        Returns
        -------
        dict
            ``ok`` (no errors), ``issues`` (all of them, errors first) and
            the result of each check under its name.
        """
        out = {'watertight': self.watertight()}
        if control or ini:
            out['bcs'] = self.check_bcs(control=control, ini=ini)
        if duplicates:
            out['duplicates'] = self.duplicates()
        order = {'error': 0, 'warning': 1, 'info': 2}
        issues = [i for r in out.values() for i in r['issues']]
        issues.sort(key=lambda i: order[i['level']])
        out['issues'] = issues
        out['ok'] = not any(i['level'] == 'error' for i in issues)
        return out

    def check_report(self, control=None, ini=None,
                     duplicates: bool = True) -> str:
        """:meth:`check` as plain text, with the BC table if files given."""
        r = self.check(control=control, ini=ini, duplicates=duplicates)
        lines = [f"{self._lg.name if hasattr(self._lg, 'name') else ''}"
                 f" -- {'OK' if r['ok'] else 'ERRORS'}"]
        if 'bcs' in r:
            lines += ['', self.bcs_report(r['bcs'], issues=False)]
        lines.append('')
        if not r['issues']:
            lines.append('no findings')
        lines += [f"{i['level']:<8}[{i['check']}] {i['message']}"
                  for i in r['issues']]
        return '\n'.join(lines)

    @staticmethod
    def bcs_report(result: dict, issues: bool = True) -> str:
        """Table of a :meth:`check_bcs` result."""
        lines = [f"{'zone':<22}{'mesh':<10}{'faces':>8}  {'control':<16}"
                 f"{'ini':<14}"]
        for r in result['rows']:
            ini = ('-' if r['ini'] is None
                   else '(' + ','.join(str(v) for v in r['ini']) + ')')
            lines.append(f"{r['name']:<22}{r['mesh'] or '-':<10}"
                         f"{r['n_faces']:>8}  {str(r['control'] or '-'):<16}"
                         f"{ini:<14}")
        if issues:
            lines.append('')
            lines += [f"{i['level']:<8}{i['message']}"
                      for i in result['issues']] or ['no findings']
        return '\n'.join(lines)

    def report(self) -> str:
        """:meth:`summary` as plain text."""
        s = self.summary()
        cells = ', '.join(f"{c['n']} {c['kind']} (order {c['order']})"
                          for c in s['cells'])
        lines = [
            f"{s['path']}",
            f"format {s['format']} · Ngeo {s['ngeo']} · {s['n_points']} nodes"
            f" · {s['n_cells']} cells: {cells}",
            "bbox  min " + _vec(s['bbox']['min']) + "  max "
            + _vec(s['bbox']['max']) + "  size " + _vec(s['bbox']['size']),
            "zones " + ', '.join(f"{k}: {v}" for k, v in s['zones'].items()),
            "",
            f"{'boundary':<24}{'type':<14}{'faces':>10}{'area':>14}"
            f"{'interior':>10}",
        ]
        for b in s['boundaries']:
            lines.append(f"{b['name']:<24}{str(b['bc_type'] or '-'):<14}"
                         f"{b['n_faces']:>10}{b['area']:>14.6g}"
                         f"{b['interior_faces']:>10}")
        lines += [
            "",
            f"exterior faces {s['exterior_faces']} · in no boundary "
            f"{s['unassigned_faces']} · non-manifold {s['nonmanifold_faces']}"
            f" · orphan nodes {s['orphan_nodes']}",
        ]
        lines += [f"warning: {w}" for w in s['warnings']]
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        return f"<LEGOLAS.Census of {self.mesh!r}>"


def _vec(v) -> str:
    return '(' + ', '.join(f"{x:.6g}" for x in v) + ')'
