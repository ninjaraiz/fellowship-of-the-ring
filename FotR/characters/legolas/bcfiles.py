"""Boundary conditions declared outside the mesh: HORSES3D ``.control`` and
PyHOPE ``parameter.ini``.

Only what :meth:`LEGOLAS.Census.check_bcs <.census.Census.check_bcs>` needs
is read: which zone names each file declares and with which type. Comments
start with ``!`` in both formats and are dropped; key names are matched
without case.

* ``.control``: blocks ``#define boundary a__b__c`` … ``type = X`` …
  ``#end``. Several zones share a block by joining their names with
  ``__`` (that is how HORSES3D splits them).
* ``parameter.ini``: pairs ``BoundaryName = x`` / ``BoundaryType =
  (/t,c,s,a/)``; the first integer of the type is the kind, and 100 marks
  an inner zone (faces with a cell on each side).

The GANDALF ring ``horses3d`` writes these blocks on its own; the two do not
share code on purpose (``doc/PLAN_HORSES3D.md``), so a change of syntax has
to be made in both.
"""

from __future__ import annotations

import re

__all__ = ['read_control_bcs', 'read_ini_bcs']

_DEFINE = re.compile(r'^\s*#define\s+boundary\b\s*(.*)$', re.I)
_END = re.compile(r'^\s*#end\b', re.I)
_KEYVAL = re.compile(r'^\s*([^=]+?)\s*=\s*(.*?)\s*$')


def _strip(line: str) -> str:
    return line.split('!', 1)[0].rstrip('\n')


def read_control_bcs(path) -> list:
    """Zones declared in the ``#define boundary`` blocks of a ``.control``.

    Parameters
    ----------
    path : str or path-like
        HORSES3D control file.

    Returns
    -------
    list[dict]
        One entry per zone name: ``name`` (as written), ``type`` (value of
        the block's ``type`` key, or None), ``line`` (1-based line of the
        ``#define``) and ``group`` (all the names of that block).
        Commented-out blocks are ignored.
    """
    out = []
    block = None
    with open(path, errors='replace') as fh:
        for n, raw in enumerate(fh, 1):
            line = _strip(raw)
            m = _DEFINE.match(line)
            if m:
                names = [x for x in re.split(r'__|\s+', m.group(1).strip())
                         if x]
                block = {'names': names, 'type': None, 'line': n}
                continue
            if block is None:
                continue
            if _END.match(line):
                for name in block['names']:
                    out.append({'name': name, 'type': block['type'],
                                'line': block['line'],
                                'group': list(block['names'])})
                block = None
                continue
            kv = _KEYVAL.match(line)
            if kv and kv.group(1).strip().lower() == 'type':
                block['type'] = kv.group(2).strip().strip('"\'')
    if block is not None:
        raise ValueError(f"{path}: '#define boundary' at line "
                         f"{block['line']} has no '#end'")
    return out


def read_ini_bcs(path) -> list:
    """Boundaries declared in a PyHOPE/HOPR ``parameter.ini``.

    Parameters
    ----------
    path : str or path-like
        PyHOPE parameter file.

    Returns
    -------
    list[dict]
        ``name``, ``type`` (tuple of ints from ``(/t,c,s,a/)``, or None if
        no ``BoundaryType`` follows) and ``line``, in file order.
    """
    out = []
    with open(path, errors='replace') as fh:
        for n, raw in enumerate(fh, 1):
            kv = _KEYVAL.match(_strip(raw))
            if not kv:
                continue
            key, val = kv.group(1).strip().lower(), kv.group(2).strip()
            if key == 'boundaryname':
                out.append({'name': val, 'type': None, 'line': n})
            elif key == 'boundarytype' and out and out[-1]['type'] is None:
                nums = re.findall(r'-?\d+', val)
                out[-1]['type'] = tuple(int(x) for x in nums) or None
    return out
