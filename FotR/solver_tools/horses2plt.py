"""Prepare ``horses2plt`` conversions of HORSES3D solutions to Tecplot.

``horses2plt`` (``Solver/src/addons/horses2tecplot`` of the CPU repository)
turns a ``.hsol`` plus the ``.hmesh`` of the same order into an ASCII
``.tec``. This module decides *what* to convert and writes everything a run
needs; running it is left to ``utils_H3d/POST/horses2plt_batch.slurm``
(see :meth:`Horses3DSets.to_tecplot`). Standard library only.

What the binary imposes (read from its source, commit ``ab3b02326``):

* It is serial: one process per file, no ``mpirun``.
* The ``.tec`` is written next to the ``.hsol`` it is given (``x.hsol`` ->
  ``x.tec``). To keep ``RESULTS/`` untouched, every conversion runs in its
  own folder under ``POST/tec/`` on symbolic links (:func:`stage`).
* Control-file lines are cut at 132 characters, so the control names the
  links, never long paths (:func:`control_text`).
* The mesh must be the ``.hmesh`` written for the same order:
  ``MESH/<stem>_<iteration>.hmesh`` (:func:`mesh_for`).
* With Gauss nodes the FE sub-cells built on the file's own nodes stop short
  of the element faces and leave gaps; the output is interpolated to
  ``Homogeneous`` points of the same order instead.
* Statistics files (type 4) hold 9 statistics + 5 mean conservative
  variables per node, plus gradients when the run saves them: the header
  does not say so, and only the control-file key ``has gradients`` tells
  ``horses2plt``.
* Surface files (``.surf.hsol``) fail with this binary; they are left out.
"""

from __future__ import annotations

import os
import re
import struct
from typing import Iterable, Optional

__all__ = [
    'KINDS', 'select_files', 'mesh_for', 'stage', 'control_text',
    'default_variables', 'count_columns', 'is_converted', 'estimate_bytes',
    'read_header_info', 'control_flag',
]

#: Kinds of solution file a conversion can take.
KINDS = ('final', 'stats', 'snapshots')

_GAUSS, _GAUSS_LOBATTO = 1, 2
_MAX_LINE = 132
_SNAPSHOT_RE = re.compile(r"^(?P<stem>.+)_(?P<iter>\d+)\.hsol$")
_STATS_RE = re.compile(r"^(?P<stem>.+)\.stats(?:\.(?P<iter>\d+))?\.hsol$")
_HMESH_RE = re.compile(r"^(?P<stem>.+)_(?P<iter>\d+)\.hmesh$")

#: Columns written by each ``horses2plt`` output variable that expands.
_EXPANDS = {'Q': 5, 'V': 3, 'Vmean': 3, 'Sij': 6, 'Vfrms': 3, 'gradV': 9,
            'omega': 3, 'Axes': 9, 'ThreeAxes': 9}


def read_header_info(path: str) -> Optional[dict]:
    """``file_type``, ``node_type``, ``n_elements``, ``iteration`` and
    ``complete`` (end marker 99) of a HORSES3D binary file, or None if it is
    not one (too short)."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            head = fh.read(144)
            if size < 208 or len(head) < 144:
                return None
            fh.seek(-4, os.SEEK_END)
            tail = struct.unpack('<i', fh.read(4))[0]
    except OSError:
        return None
    file_type, node_type, n_elem, iteration = struct.unpack('<4i', head[128:144])
    return {'file_type': file_type, 'node_type': node_type,
            'n_elements': n_elem, 'iteration': iteration,
            'complete': tail == 99}


def select_files(
    case_dir: str,
    solution_file: str,
    kinds: Iterable[str] = KINDS,
    iterations: Optional[tuple] = None,
) -> list:
    """Solution files of one control, as ``(kind, path, complete)``.

    Parameters
    ----------
    case_dir : str
        Case folder.
    solution_file : str
        ``solution file name`` of the control, relative to ``case_dir``
        (``RESULTS/<stem>.hsol``).
    kinds : iterable of str
        Any of :data:`KINDS`: the final solution ``<stem>.hsol``, the
        statistics ``<stem>.stats[.<iter>].hsol`` and the autosaves
        ``<stem>_<iter>.hsol``.
    iterations : (int, int), optional
        Keep autosaves and dated statistics with ``first <= iteration <=
        last``. The undated final and statistics files are always kept.

    Returns
    -------
    list of tuple
        Sorted by kind and iteration. ``complete`` is False for a file the
        solver has not finished writing.
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown kinds {sorted(unknown)}; use {KINDS}.")
    results = os.path.join(case_dir, os.path.dirname(solution_file))
    stem = os.path.splitext(os.path.basename(solution_file))[0]
    lo, hi = iterations if iterations else (None, None)

    def in_range(it):
        return it is None or ((lo is None or it >= lo) and (hi is None or it <= hi))

    found = []
    try:
        names = os.listdir(results)
    except OSError:
        return []
    for name in names:
        if name.endswith('.surf.hsol'):
            continue
        path = os.path.join(results, name)
        if name == stem + '.hsol':
            kind, it = 'final', None
        else:
            m = _STATS_RE.match(name)
            if m and m.group('stem') == stem:
                kind, it = 'stats', int(m.group('iter')) if m.group('iter') else None
            else:
                m = _SNAPSHOT_RE.match(name)
                if not (m and m.group('stem') == stem):
                    continue
                kind, it = 'snapshots', int(m.group('iter'))
        if kind not in kinds or not in_range(it):
            continue
        info = read_header_info(path)
        found.append((KINDS.index(kind), -1 if it is None else it, kind, path,
                      bool(info and info['complete'])))
    found.sort(key=lambda t: (t[0], t[1] if t[1] >= 0 else float('inf')))
    return [(kind, path, complete) for _, _, kind, path, complete in found]


def mesh_for(case_dir: str, stem: str) -> Optional[str]:
    """``MESH/<stem>_<iteration>.hmesh`` (the earliest), or None.

    The solver writes it when a run with that solution name starts; its
    order is the run's polynomial order, the one ``horses2plt`` needs.
    """
    mesh_dir = os.path.join(case_dir, 'MESH')
    try:
        names = os.listdir(mesh_dir)
    except OSError:
        return None
    hits = sorted(
        (int(m.group('iter')), name) for name in names
        if not name.endswith('.surf.hmesh')
        for m in [_HMESH_RE.match(name)] if m and m.group('stem') == stem
    )
    return os.path.join(mesh_dir, hits[0][1]) if hits else None


def stage(target_dir: str, hsol: str, hmesh: str) -> tuple:
    """Link ``hsol`` and ``hmesh`` into ``target_dir``; return their names.

    ``horses2plt`` writes the ``.tec`` next to the file it reads, so it
    lands in ``target_dir``. Links are relative when both paths share a
    root, and replaced when they point elsewhere.
    """
    os.makedirs(target_dir, exist_ok=True)
    names = []
    for src in (hsol, hmesh):
        name = os.path.basename(src)
        link = os.path.join(target_dir, name)
        rel = os.path.relpath(os.path.abspath(src), os.path.abspath(target_dir))
        if os.path.islink(link):
            if os.readlink(link) == rel:
                names.append(name)
                continue
            os.remove(link)
        elif os.path.exists(link):
            raise FileExistsError(f"{link} exists and is not a link.")
        os.symlink(rel, link)
        names.append(name)
    return tuple(names)


def default_variables(kind: str) -> str:
    """``horses2plt`` output variables used when none are given.

    ``Vfrms`` is left out of the statistics: ``horses2plt`` takes
    sqrt(<uu> - <u>^2), which is round-off and often negative where the
    flow does not fluctuate (a steady wake), so it writes ``NaN`` into the
    ``.tec``. The variances are the diagonal of ``Sij``.
    """
    return 'Vmean,Sij,Q' if kind == 'stats' else 'Q,p,T,Mach'


def count_columns(variables: str) -> int:
    """Columns of the ``.tec`` besides x, y, z (for size estimates)."""
    return sum(_EXPANDS.get(v.strip(), 1) for v in variables.split(',') if v.strip())


def control_text(
    hsol: str,
    hmesh: str,
    variables: str,
    node_type: int,
    order: int,
    dimensionless: bool = False,
    has_gradients: bool = False,
) -> str:
    """Control file of one conversion, for the names given by :func:`stage`.

    FE output (one zone of bricks). With Gauss nodes (``node_type`` 1) the
    output is interpolated to ``Homogeneous`` points of order ``order`` so
    that the sub-cells reach the element faces; with Gauss-Lobatto nodes
    the file's own nodes are used. Output is dimensional unless
    ``dimensionless``.

    Raises
    ------
    ValueError
        If a line would exceed the 132 characters ``horses2plt`` reads.
    """
    lines = [
        f"hmesh file = {hmesh}",
        f"hsol file = {hsol}",
        "output mode = FE",
        f"output variables = {variables}",
    ]
    if node_type == _GAUSS:
        lines += ["output basis = Homogeneous", f"output order = {int(order)}"]
    if dimensionless:
        lines.append("dimensionless = .true.")
    if has_gradients:
        lines.append("has gradients = .true.")
    too_long = [line for line in lines if len(line) > _MAX_LINE]
    if too_long:
        raise ValueError(
            f"horses2plt reads {_MAX_LINE} characters per line: {too_long}"
        )
    return '\n'.join(lines) + '\n'


def is_converted(tec: str, hsol: str) -> bool:
    """True if ``tec`` exists and is newer than ``hsol``."""
    try:
        return os.path.getmtime(tec) >= os.path.getmtime(hsol)
    except OSError:
        return False


def estimate_bytes(n_elements: int, order: int, n_columns: int,
                   gradients: bool = False) -> int:
    """Rough peak memory of one ``horses2plt`` run (bytes).

    The whole mesh and solution stay in memory: coordinates, state,
    optional gradients and the output variables, in doubles, per node;
    doubled for the interpolated copy and the output buffers.
    """
    nodes = int(n_elements) * (int(order) + 1) ** 3
    per_node = 3 + 5 + (24 if gradients else 0) + int(n_columns)
    return 2 * 8 * nodes * per_node


def control_flag(control_path: str, key: str) -> Optional[bool]:
    """Boolean value of ``key`` in a ``.control`` (outside ``#define``
    blocks), or None if absent."""
    want = ' '.join(key.lower().split())
    in_block = False
    try:
        with open(control_path, errors='replace') as fh:
            for raw in fh:
                line = raw.split('!')[0].strip()
                low = line.lower()
                if low.startswith('#define'):
                    in_block = True
                if in_block:
                    in_block = not low.startswith('#end')
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if ' '.join(k.lower().split()) == want:
                    return v.strip().strip('"\'').lower() in ('.true.', 'true', 't')
    except OSError:
        return None
    return None
