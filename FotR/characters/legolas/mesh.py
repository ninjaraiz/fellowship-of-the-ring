"""The neutral mesh every LEGOLAS reader returns.

Plain numpy containers, so that everything downstream (census, quality,
viewer) is written once and not once per file format. See
:mod:`.ordering` for the node layout of each element kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ordering

__all__ = ['CellBlock', 'Boundary', 'Mesh', 'VOLUME_KINDS']

#: Volume element kinds LEGOLAS understands.
VOLUME_KINDS = ('hex', 'tet', 'pyramid', 'prism')


@dataclass
class CellBlock:
    """Volume elements of one kind and order.

    Attributes
    ----------
    kind : str
        ``'hex'``, ``'tet'``, ``'pyramid'`` or ``'prism'``.
    order : int
        Geometric order. Only hexahedra may be above 1.
    conn : numpy.ndarray
        ``(n_cells, n_nodes)`` int64 indices into :attr:`Mesh.points`, in
        the canonical layout (tensor for ``'hex'``, VTK for the rest).
    zone : numpy.ndarray
        ``(n_cells,)`` int zone id of each cell (see :attr:`Mesh.zones`).
    """

    kind: str
    order: int
    conn: np.ndarray
    zone: np.ndarray | None = None

    def __post_init__(self):
        if self.kind not in VOLUME_KINDS:
            raise ValueError(f"unknown element kind {self.kind!r}")
        if self.kind != 'hex' and self.order != 1:
            raise ValueError(f"{self.kind} elements are linear only")
        self.conn = np.ascontiguousarray(self.conn, dtype=np.int64)
        expected = ((self.order + 1) ** 3 if self.kind == 'hex'
                    else ordering.NODES_PER_KIND[self.kind])
        if self.conn.ndim != 2 or self.conn.shape[1] != expected:
            raise ValueError(
                f"{self.kind} of order {self.order} needs {expected} nodes "
                f"per cell, got shape {self.conn.shape}")
        if self.zone is None:
            self.zone = np.zeros(len(self.conn), dtype=np.int64)

    def __len__(self) -> int:
        """Number of cells in the block."""
        return len(self.conn)

    @property
    def n_faces(self) -> int:
        """Faces per cell."""
        return 6 if self.kind == 'hex' else len(ordering.LINEAR_FACES[self.kind])

    def corners(self) -> np.ndarray:
        """``(n_cells, n_corners)`` corner nodes, in VTK corner order."""
        if self.kind == 'hex':
            return self.conn[:, ordering.hex_corners(self.order)]
        return self.conn

    def face_corners(self, face: int) -> np.ndarray:
        """Corner nodes of local *face* of every cell (3 or 4 columns)."""
        if self.kind == 'hex':
            return self.conn[:, ordering.hex_face_corners(self.order)[face]]
        return self.conn[:, list(ordering.LINEAR_FACES[self.kind][face])]

    def face_nodes(self, face: int) -> np.ndarray:
        """All nodes of local *face* of every cell.

        Returns ``(n_cells, N+1, N+1)`` for hexahedra (see
        :func:`.ordering.hex_face_grid`) and ``(n_cells, 3 or 4)`` for the
        linear kinds.
        """
        if self.kind == 'hex':
            return self.conn[:, ordering.hex_face_grid(self.order, face)]
        return self.face_corners(face)


@dataclass
class Boundary:
    """A named set of element faces: a boundary condition or a surface zone.

    Faces are stored as (element, local face) pairs, never as separate
    surface cells, so their nodes always come from the volume element and
    every format ends up with the same face layout.

    Attributes
    ----------
    name : str
        Name as written in the file.
    elem : numpy.ndarray
        ``(n_faces,)`` global element index (see :meth:`Mesh.locate_cell`).
    face : numpy.ndarray
        ``(n_faces,)`` local face index of each.
    bc_type : str or None
        Type as declared in the file (``'BCWall'``, HOPR code ``'4'``…).
    """

    name: str
    elem: np.ndarray
    face: np.ndarray
    bc_type: str | None = None

    def __post_init__(self):
        self.elem = np.asarray(self.elem, dtype=np.int64)
        self.face = np.asarray(self.face, dtype=np.int64)

    def __len__(self) -> int:
        """Number of faces in the boundary."""
        return len(self.elem)


@dataclass
class Mesh:
    """Unstructured volume mesh in the LEGOLAS canonical layout.

    Attributes
    ----------
    points : numpy.ndarray
        ``(n_points, 3)`` float64 unique node coordinates.
    blocks : list[CellBlock]
        Volume elements, one block per (kind, order). The global index of a
        cell runs through the blocks in order.
    boundaries : dict[str, Boundary]
        Named face sets.
    zones : dict[int, str]
        Volume zone id → name.
    meta : dict
        ``format``, ``path``, ``ngeo`` and whatever the reader keeps; the
        list ``meta['warnings']`` collects what the reader could not do
        cleanly.
    """

    points: np.ndarray
    blocks: list = field(default_factory=list)
    boundaries: dict = field(default_factory=dict)
    zones: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.meta.setdefault('warnings', [])
        self._cache = {}

    # -- sizes -----------------------------------------------------------
    @property
    def n_points(self) -> int:
        """Number of nodes."""
        return len(self.points)

    @property
    def n_cells(self) -> int:
        """Number of volume elements."""
        return sum(len(b) for b in self.blocks)

    @property
    def offsets(self) -> np.ndarray:
        """Global index of the first cell of each block, plus the total."""
        return np.concatenate([[0], np.cumsum([len(b) for b in self.blocks])]
                              ).astype(np.int64)

    @property
    def ngeo(self) -> int:
        """Highest geometric order among the blocks."""
        return max((b.order for b in self.blocks), default=1)

    def cell_node_offsets(self) -> np.ndarray:
        """Where the nodes of each cell start in a flat per-(cell, node)
        array: ``(n_cells + 1,)``, the last entry being the total."""
        if 'cell_node_offsets' not in self._cache:
            sizes = np.concatenate([np.full(len(b), b.conn.shape[1], np.int64)
                                    for b in self.blocks]) if self.blocks \
                else np.zeros(0, np.int64)
            self._cache['cell_node_offsets'] = np.concatenate(
                [[0], np.cumsum(sizes)]).astype(np.int64)
        return self._cache['cell_node_offsets']

    def locate_cell(self, cells) -> tuple:
        """Split global cell indices into ``(block index, local index)``."""
        cells = np.asarray(cells, dtype=np.int64)
        off = self.offsets
        b = np.searchsorted(off, cells, side='right') - 1
        return b, cells - off[b]

    def warn(self, message: str) -> None:
        """Record a reader warning in ``meta['warnings']``."""
        self.meta['warnings'].append(message)

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        """Write the mesh to a ``.npz`` file (see :meth:`load`).

        ``meta`` entries that are numpy arrays are stored as arrays; the
        rest must be JSON-serialisable and are stored as JSON.
        """
        import json
        arrays = {'points': self.points}
        for i, b in enumerate(self.blocks):
            arrays[f'block{i}_conn'] = b.conn
            arrays[f'block{i}_zone'] = b.zone
        for i, bc in enumerate(self.boundaries.values()):
            arrays[f'bc{i}_elem'] = bc.elem
            arrays[f'bc{i}_face'] = bc.face
        meta, array_meta = {}, []
        for k, v in self.meta.items():
            if isinstance(v, np.ndarray):
                arrays[f'meta_{k}'] = v
                array_meta.append(k)
            else:
                meta[k] = v
        head = {
            'blocks': [{'kind': b.kind, 'order': b.order} for b in self.blocks],
            'boundaries': [{'name': bc.name, 'bc_type': bc.bc_type}
                           for bc in self.boundaries.values()],
            'zones': {str(k): v for k, v in self.zones.items()},
            'meta': meta, 'array_meta': array_meta,
        }
        arrays['header'] = np.frombuffer(
            json.dumps(head, default=str).encode(), dtype=np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> 'Mesh':
        """Read a mesh written by :meth:`save`."""
        import json
        with np.load(path, allow_pickle=False) as z:
            head = json.loads(z['header'].tobytes().decode())
            blocks = [CellBlock(b['kind'], b['order'], z[f'block{i}_conn'],
                                z[f'block{i}_zone'])
                      for i, b in enumerate(head['blocks'])]
            bcs = {b['name']: Boundary(b['name'], z[f'bc{i}_elem'],
                                       z[f'bc{i}_face'], b['bc_type'])
                   for i, b in enumerate(head['boundaries'])}
            meta = dict(head['meta'])
            for k in head['array_meta']:
                meta[k] = z[f'meta_{k}']
            points = z['points']
        return cls(points, blocks, bcs,
                   {int(k): v for k, v in head['zones'].items()}, meta)

    def __repr__(self) -> str:
        kinds = ', '.join(f"{len(b)} {b.kind}{b.order}" for b in self.blocks)
        return (f"Mesh({self.meta.get('format', '?')}: {self.n_points} nodes, "
                f"{kinds or 'no cells'}, {len(self.boundaries)} boundaries)")
