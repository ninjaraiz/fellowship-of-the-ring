"""``LEGOLAS.Census``: what a mesh contains, counted and checked.

L1 covers :meth:`Census.summary`. Validation against ``.control`` /
``parameter.ini``, watertightness and mesh comparison come in L3
(``doc/PLAN_LEGOLAS.md``).
"""

from __future__ import annotations

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
