"""LEGOLAS -- Lightweight Exploratory Graphics Of Loaded Aerodynamic Simulations.

Since 20/11/2026, the character for CFD **meshes**: read a mesh file into a
neutral numpy mesh (:class:`~.mesh.Mesh`) and look at it from several
angles, each one a nested class of :class:`LEGOLAS`. Plan and milestones in
``doc/PLAN_LEGOLAS.md``.

Only numpy and h5py are needed to import and read; meshio (gmsh files) and
scipy (``.hmesh``) are imported by the readers that use them.
"""

from __future__ import annotations

import os

from .census import Census
from .convert import Convert
from .io import read
from .locate import Locate
from .mesh import Boundary, CellBlock, Mesh
from .quality import Quality
from .resolution import Resolution
from .solution import Solution, read_control_keys
from .viewer import Viewer

__all__ = ['LEGOLAS', 'Mesh', 'CellBlock', 'Boundary', 'read']


class LEGOLAS:
    """A CFD mesh and the tools to inspect it.

    Parameters
    ----------
    source : str, path-like or Mesh
        Mesh file (HOPR/PyHOPE ``*_mesh.h5``, CGNS/HDF5 ``.cgns``, gmsh
        ``.msh``, HORSES3D ``.hmesh``) or an already built :class:`Mesh`.
    fmt : str, optional
        ``'hopr'``, ``'cgns'``, ``'gmsh'`` or ``'hmesh'``; guessed from the
        file when omitted.
    **read_kwargs
        Passed to the format reader (e.g. ``rtol`` for ``.hmesh``).

    Attributes
    ----------
    mesh : Mesh
        The mesh, in the canonical layout (see
        :mod:`FotR.characters.legolas.ordering`).
    census : LEGOLAS.Census
        Sizes, element kinds, zones and boundaries.
    quality : LEGOLAS.Quality
        Cell quality metrics (Jacobians, aspect ratio, skewness…).
    resolution : LEGOLAS.Resolution
        Cell sizes, first cells at a wall and estimated y+ (LES).
    convert : LEGOLAS.Convert
        Node-order repair, VTK (``.vtu``/``.vtm``) and gmsh writers.
    locate : LEGOLAS.Locate
        Cell holding each point; HORSES3D's actuator-disk search.
    solution : LEGOLAS.Solution
        A HORSES3D ``.hsol`` evaluated at the nodes of the mesh
        (``lg.solution.load(path)``), for the viewer and for exports.
    viewer : LEGOLAS.Viewer
        Web viewer: ``lg.viewer.show()`` returns the address to open.

    Examples
    --------
    ::

        from FotR import LEGOLAS
        lg = LEGOLAS('MESH/Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
        print(lg.census)
        lg.mesh.boundaries['inlet']
        h = lg.viewer.show()      # open h.url in the browser
    """

    #: Inventory and checks of the mesh (:class:`.census.Census`).
    Census = Census
    #: Cell quality metrics (:class:`.quality.Quality`).
    Quality = Quality
    #: Sizes and wall resolution for an LES (:class:`.resolution.Resolution`).
    Resolution = Resolution
    #: Node order, VTK and gmsh writers (:class:`.convert.Convert`).
    Convert = Convert
    #: Point location (:class:`.locate.Locate`).
    Locate = Locate
    #: A HORSES3D solution on the mesh (:class:`.solution.Solution`).
    Solution = Solution
    #: Web viewer of the mesh (:class:`.viewer.Viewer`).
    Viewer = Viewer

    def __init__(self, source, fmt: str | None = None, **read_kwargs):
        if isinstance(source, Mesh):
            self.mesh = source
            self._source = None
        else:
            self.mesh = read(source, fmt=fmt, **read_kwargs)
            # how to read it again (the viewer's server process does)
            self._source = (self.mesh.meta['path'], self.mesh.meta['format'],
                            dict(read_kwargs))
        self.census = LEGOLAS.Census(self)
        self.quality = LEGOLAS.Quality(self)
        self.resolution = LEGOLAS.Resolution(self)
        self.convert = LEGOLAS.Convert(self)
        self.locate = LEGOLAS.Locate(self)
        self.solution = LEGOLAS.Solution(self)
        self.viewer = LEGOLAS.Viewer(self)

    @classmethod
    def from_control(cls, control, solution: str | None = None,
                     physical: bool = False) -> 'LEGOLAS':
        """Open a HORSES3D case the way the solver does: from its control.

        Parameters
        ----------
        control : str or path-like
            The case's ``.control``; ``mesh file name`` and ``solution file
            name`` are read from it, relative to the control's folder.
        solution : str, optional
            Another ``.hsol`` to load instead (an autosave, say); relative
            paths are taken from the control's folder. ``''`` loads none.
        physical : bool
            SI units for the solution.

        Returns
        -------
        LEGOLAS
            With the mesh read and, when the file exists, the solution
            loaded in ``lg.solution``.

        Examples
        --------
        ::

            lg = LEGOLAS.from_control('outputs/marino_A_tsr_6.0/file_control_p3.control')
            lg.solution.variables
            lg.viewer.show()          # the solution is in the colour selector
        """
        folder = os.path.dirname(os.path.abspath(control))
        keys = read_control_keys(control)
        if 'mesh file name' not in keys:
            raise ValueError(f"{control}: no 'mesh file name'")
        lg = cls(os.path.join(folder, keys['mesh file name']))
        sol = keys.get('solution file name') if solution is None else solution
        if sol:
            path = sol if os.path.isabs(sol) else os.path.join(folder, sol)
            if not path.endswith('.hsol'):
                path += '.hsol'
            if os.path.isfile(path):
                lg.solution.load(path, physical=physical)
            elif solution:
                raise FileNotFoundError(path)
        return lg

    def _reset(self) -> None:
        """Drop every cached result after the mesh has been changed."""
        self.mesh._cache.clear()
        self.census._summary = None
        self.quality._metrics = None
        self.quality.meta = {}
        self.resolution._sizes = None

    @property
    def name(self) -> str:
        """File name of the mesh, or ``'<memory>'``."""
        path = self.mesh.meta.get('path')
        return os.path.basename(path) if path else '<memory>'

    def __repr__(self) -> str:
        return f"LEGOLAS({self.name}: {self.mesh!r})"

    def __str__(self) -> str:
        return (f"Legolas has his eye on {self.name}: "
                f"{self.mesh.n_cells} cells, {self.mesh.n_points} nodes, "
                f"{len(self.mesh.boundaries)} boundaries.")
