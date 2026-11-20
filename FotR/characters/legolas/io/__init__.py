"""Mesh readers: one module per format, one entry point, :func:`read`.

Every reader returns a :class:`~FotR.characters.legolas.mesh.Mesh` in the
canonical layout. Format modules import their libraries inside the
function, so ``import`` of this package costs nothing beyond numpy.
"""

from __future__ import annotations

import os

__all__ = ['read', 'detect_format', 'FORMATS']

#: Format name → module (relative) implementing ``read(path, **kw)``.
FORMATS = {
    'hopr': '.hopr',
    'cgns': '.cgns',
    'gmsh': '.gmsh',
    'hmesh': '.hmesh',
}


def _hdf5_flavour(path: str) -> str | None:
    """'cgns' or 'hopr' for an HDF5 file, None if it is neither."""
    import h5py
    if not h5py.is_hdf5(path):
        return None
    with h5py.File(path, 'r') as fh:
        if 'CGNSLibraryVersion' in fh:
            return 'cgns'
        if 'ElemInfo' in fh and 'NodeCoords' in fh:
            return 'hopr'
    return None


def detect_format(path: str) -> str:
    """Guess the format of *path* from its extension and, for HDF5, content.

    Raises
    ------
    ValueError
        If the format cannot be told, or a ``.cgns`` file is not HDF5
        (the old ADF flavour, which needs the CGNS library to read).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.msh':
        return 'gmsh'
    if ext == '.hmesh':
        return 'hmesh'
    if ext in ('.cgns', '.h5', '.hdf5', '.hdf'):
        flavour = _hdf5_flavour(path)
        if flavour:
            return flavour
        if ext == '.cgns':
            raise ValueError(
                f"{path} is not an HDF5 file: probably a CGNS/ADF file. "
                "LEGOLAS reads CGNS/HDF5 only; convert it first, e.g. with "
                "`adf2hdf` from the CGNS tools.")
        raise ValueError(f"{path}: HDF5 file that is neither CGNS nor a "
                         "HOPR/PyHOPE mesh")
    raise ValueError(f"cannot tell the mesh format of {path}; pass fmt= "
                     f"one of {sorted(FORMATS)}")


def read(path, fmt: str | None = None, **kwargs):
    """Read a mesh file into a canonical :class:`Mesh`.

    Parameters
    ----------
    path : str or path-like
        Mesh file: HOPR/PyHOPE ``*_mesh.h5``, CGNS/HDF5 ``.cgns``, gmsh
        ``.msh`` (v2.2 or v4) or HORSES3D ``.hmesh``.
    fmt : str, optional
        ``'hopr'``, ``'cgns'``, ``'gmsh'`` or ``'hmesh'``. Guessed when
        omitted (:func:`detect_format`).
    **kwargs
        Passed to the format reader.

    Returns
    -------
    Mesh
    """
    import importlib
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    fmt = (fmt or detect_format(path)).lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown mesh format {fmt!r}; use one of "
                         f"{sorted(FORMATS)}")
    module = importlib.import_module(FORMATS[fmt], __name__)
    mesh = module.read(path, **kwargs)
    mesh.meta.setdefault('format', fmt)
    mesh.meta.setdefault('path', os.path.abspath(path))
    _check_order(mesh)
    return mesh


def _check_order(mesh, sample: int = 2000) -> None:
    """Warn when high-order hexahedra fail the tensor-order screen."""
    from ..ordering import tensor_monotonic
    for block in mesh.blocks:
        if block.kind == 'hex' and block.order >= 2:
            bad = tensor_monotonic(mesh.points, block.conn, block.order,
                                   sample=sample)
            if len(bad):
                checked = min(sample, len(block))
                mesh.warn(
                    f"{len(bad)} of {checked} sampled order-{block.order} "
                    "hexahedra fail the tensor-order check: the file's node "
                    "ordering is probably not the one its format declares")
