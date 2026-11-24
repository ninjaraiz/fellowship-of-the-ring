"""
rings/horses3d.py
=================
HORSES3D ring: one shared mesh, one ``.control`` per polynomial order.

Every case folder gets a copy of the mesh in ``MESH/``, one
``file_control_p<p>.control`` per requested ``p`` (each ``p`` restarts
from the previous one) and, optionally, a free-stream preservation test
``fsp_p<p>.control``. The controls are written from a working template
by rewriting ``key = value`` lines, never by placeholders, so the
template can be any control that already runs.

Naming contract, shared with ``Horses3DReader`` (the reader re-derives it
on its own; ``cases_metadata.json`` is the bridge, as in CODA):

* the mesh name carries its geometric order as a ``g<n>`` token, equal to
  the ``Ngeo`` attribute of the ``.h5``;
* the solution of order ``p`` is ``RESULTS/<tag>_g<g>p<p>.hsol``.
"""

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import BaseRing
from .coda import _resolve_update_value

log = logging.getLogger(__name__)

#: Same rule as ``Horses3DReader._MESH_G_RE``.
_MESH_G_RE = re.compile(r"(?<![A-Za-z])g(?P<g>\d+)")

_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^=!#\s][^=!]*?)\s*=\s*(?P<value>.*)$")

#: Keys the ring owns: they cannot be set through ``control_values``.
_RING_KEYS = ('mesh file name', 'polynomial order', 'solution file name')

#: Keys HORSES3D refuses to start without (``Setup.f90``), even when unused:
#: ``restart file name`` is read with ``restart = .false.`` too.
_MANDATORY_KEYS = (
    'mesh file name', 'restart', 'restart file name', 'solution file name',
    'number of time steps', 'output interval', 'convergence tolerance',
)
_MANDATORY_NS_KEYS = ('mach number', 'reynolds number')

#: Keys removed from the FSP control (anything that is not plain flow).
_FSP_DROP_PREFIXES = (
    'restart polorder', 'stl file name', 'number of stl',
    'boundaries to save', 'surface save', 'actuator ',
)
_FSP_OFF_KEYS = ('use actuatorline', 'use actuatordisk')

#: Type written for inner zones the template leaves out. Any type is
#: accepted and none is ever applied: inner zones have no boundary faces.
_INNER_ZONE_TYPE = 'FreeSlipWall'


def _norm_key(key: str) -> str:
    return ' '.join(key.lower().split())


def _strip_comment(line: str) -> str:
    """Drop a ``!`` comment that is not inside quotes."""
    quote = None
    for i, ch in enumerate(line):
        if ch in '"\'':
            quote = None if quote == ch else (quote or ch)
        elif ch == '!' and quote is None:
            return line[:i].rstrip()
    return line.rstrip()


def _fmt(value) -> str:
    """A Python value as a control value."""
    if isinstance(value, (bool, np.bool_)):
        return '.true.' if value else '.false.'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _quoted(path: str) -> str:
    return f'"{path}"'


def _block_key(key: str):
    """``'#define statistics: starting time'`` -> (``'#define statistics'``,
    ``'starting time'``); ``None`` for a key outside the blocks."""
    norm = _norm_key(key)
    if not norm.startswith('#define') or ':' not in norm:
        return None
    header, inner = norm.split(':', 1)
    return _norm_key(header), _norm_key(inner)


def _rewrite_control(
    text: str,
    values: dict,
    strip_comments: bool = True,
    drop_defines: bool = False,
) -> str:
    """Set, add or remove ``key = value`` lines of a control.

    ``values`` maps keys (case and spacing insensitive) to their new value;
    ``None`` removes the line. Missing keys are added before the first
    block. With ``drop_defines`` every block is removed.

    Keys inside ``#define … #end`` blocks are only touched when named with
    their block, ``'<block header>: <key>'``, e.g.
    ``'#define statistics: starting time'``: the line is replaced, or added
    before the block's ``#end`` if missing. Every block with that header is
    changed; a header the control does not have is an error.
    """
    wanted, in_blocks = {}, {}
    for k, v in values.items():
        bk = _block_key(k)
        if bk is None:
            wanted[_norm_key(k)] = v
        else:
            in_blocks.setdefault(bk[0], {})[bk[1]] = (k.split(':', 1)[1].strip(), v)
    done = set()
    seen_blocks = set()
    out = []
    first_define = None
    block = None
    block_done = set()
    for raw in text.splitlines():
        line = _strip_comment(raw) if strip_comments else raw.rstrip()
        stripped = line.strip().lower()
        if strip_comments and not line.strip() and raw.strip():
            continue
        if stripped.startswith('#define'):
            block = _norm_key(line.strip())
            block_done = set()
            seen_blocks.add(block)
            if drop_defines:
                continue
            if first_define is None:
                first_define = len(out)
        if block is not None:
            changes = in_blocks.get(block, {})
            if stripped.startswith('#end'):
                if not drop_defines:
                    out.extend(f"   {name} = {_fmt(v)}" for key, (name, v) in changes.items()
                               if key not in block_done and v is not None)
                    out.append(line)
                block = None
                continue
            m = _KEY_RE.match(line)
            if m and _norm_key(m.group('key')) in changes:
                key = _norm_key(m.group('key'))
                block_done.add(key)
                if not drop_defines and changes[key][1] is not None:
                    out.append(f"{m.group('indent')}{m.group('key')} = {_fmt(changes[key][1])}")
                continue
            if not drop_defines:
                out.append(line)
            continue
        m = _KEY_RE.match(line)
        if m and _norm_key(m.group('key')) in wanted:
            key = _norm_key(m.group('key'))
            done.add(key)
            if wanted[key] is not None:
                out.append(f"{m.group('indent')}{m.group('key')} = {_fmt(wanted[key])}")
            continue
        out.append(line)

    missing = sorted(set(in_blocks) - seen_blocks)
    if missing:
        raise ValueError(f"The control has no block {missing}: block keys can only "
                         "change a block the template already has.")
    extra = [f"{k} = {_fmt(v)}" for k, v in values.items()
             if _block_key(k) is None and _norm_key(k) not in done and v is not None]
    if extra:
        at = first_define if first_define is not None else len(out)
        while at and not out[at - 1].strip():
            at -= 1
        out[at:at] = extra

    cleaned = []
    for line in out:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip('\n') + '\n'


def _control_keys(text: str) -> dict:
    """``{normalised key: raw value}`` of the lines outside ``#define`` blocks."""
    found, in_block = {}, False
    for raw in text.splitlines():
        line = _strip_comment(raw)
        low = line.strip().lower()
        if low.startswith('#define'):
            in_block = True
        if in_block:
            in_block = not low.startswith('#end')
            continue
        m = _KEY_RE.match(line)
        if m:
            found[_norm_key(m.group('key'))] = m.group('value').strip().strip('"\'')
    return found


def _defined_zones(text: str) -> set:
    """Zone names with a ``#define boundary a__b…`` block (lowercase)."""
    names = set()
    for raw in text.splitlines():
        line = _strip_comment(raw).strip().lower()
        if line.startswith('#define boundary'):
            names.update(n for n in line[len('#define boundary'):].strip().split('__') if n)
    return names


def _referenced_zones(text: str) -> tuple:
    """Zones named by surface-monitor ``marker`` lines and by ``boundaries to
    save``, as two sets. Unlike ``#define boundary``, HORSES3D does not split
    these names on ``__``: each one must be a single zone."""
    markers, saved = set(), set()
    for raw in text.splitlines():
        line = _strip_comment(raw).strip().lower()
        m = re.match(r"marker\s*=\s*(\S+)", line)
        if m:
            markers.add(m.group(1))
        m = re.match(r"boundaries to save\s*=\s*\[(.*)\]", line)
        if m:
            saved.update(n.strip() for n in m.group(1).split(',') if n.strip())
    return markers, saved


def _read_mesh_info(mesh_path: str) -> dict:
    """``Ngeo`` and the boundary names (inner ones, BCType 100, apart)."""
    import h5py   # deferred: only this ring needs it

    with h5py.File(mesh_path, 'r') as fh:
        ngeo = fh.attrs.get('Ngeo')
        names = [n.decode().strip() if isinstance(n, bytes) else str(n).strip()
                 for n in fh['BCNames'][()]] if 'BCNames' in fh else []
        types = fh['BCType'][()] if 'BCType' in fh else None
    boundaries, inner = [], []
    for i, name in enumerate(names):
        is_inner = types is not None and int(types[i][0]) == 100
        (inner if is_inner else boundaries).append(name.lower())
    return {
        'ngeo': int(ngeo) if ngeo is not None else None,
        'boundaries': boundaries,
        'inner': inner,
    }


class Horses3DRing(BaseRing):
    """
    HORSES3D case generator with one shared mesh and a ``p`` ladder.

    Cases come from :meth:`define_cases` (and :meth:`add_param` /
    :meth:`compute_param`) as in every ring; :meth:`generate_folders`
    writes them, :meth:`submit_cases` launches them through
    ``utils_H3d/run.sh`` and :meth:`fsp_report` checks the free-stream
    preservation tests.

    Parameters
    ----------
    root_dir : str
        Campaign root (``metadata/`` and ``outputs/``).
    eq_type, num_stages
        Accepted for the GANDALF facade and ignored: the flow equations
        come from the control template, and the stages of a HORSES3D case
        are its polynomial orders (``p_levels``).

    Examples
    --------
    ::

        gdf = GANDALF(root_dir='~/Documentos/COMAC/wind_turbine_ansa',
                      ring='horses3d')
        gdf.define_cases(method='external', external_dataframe=df)
    """

    def __init__(
        self, root_dir: str,
        eq_type: Optional[str] = None,
        num_stages: Optional[int] = None,
        **kwargs
    ) -> None:
        """Create the ring and ``root_dir/metadata``.

        ``outputs/`` is created by :meth:`generate_folders`.
        """
        self.root_dir = root_dir
        self.eq_type = None
        self.num_stages = None
        self.version = kwargs.get('version', None)
        self.case_tensor = None
        self.design_vars = None
        self.df_cases = None
        self.df_geom = None
        self.tag = None
        self.mesh_g = None
        self.p_levels = None
        os.makedirs(os.path.join(self.root_dir, 'metadata'), exist_ok=True)

    # ── naming ────────────────────────────────────────────────────────────

    def solution_name(self, p: int, fsp: bool = False) -> str:
        """Path of the solution of order ``p``, relative to the case folder.

        ``RESULTS/<tag>_g<g>p<p>.hsol``, or
        ``RESULTS/FSP/<tag>_g<g>p<p>_fsp.hsol`` for the FSP test.
        """
        if self.tag is None:
            try:
                self.load_campaign()
            except FileNotFoundError:
                raise RuntimeError(
                    "No campaign yet: call generate_folders() first."
                ) from None
        stem = f"{self.tag}_g{self.mesh_g}p{int(p)}"
        return f"RESULTS/FSP/{stem}_fsp.hsol" if fsp else f"RESULTS/{stem}.hsol"

    @staticmethod
    def control_name(p: int, fsp: bool = False) -> str:
        """``file_control_p<p>.control``, or ``fsp_p<p>.control``."""
        return f"fsp_p{int(p)}.control" if fsp else f"file_control_p{int(p)}.control"

    # ── folders ───────────────────────────────────────────────────────────

    def generate_folders(
        self,
        control_template: str,
        mesh_path: str,
        tag: str,
        folder_fmt: str,
        p_levels: Union[list, tuple] = (2,),
        control_values: Optional[dict] = None,
        p_values: Optional[dict] = None,
        extra_dirs: Optional[list] = None,
        data_to_update: Optional[dict] = None,
        setup_dir: Optional[str] = None,
        run_script: Optional[str] = None,
        fsp: bool = True,
        fsp_steps: int = 200,
        strip_comments: bool = True,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Write one folder per case under ``outputs/`` and the metadata.

        Per case: ``MESH/<mesh>`` (copy), ``RESULTS/``, ``LOG/``, one
        ``file_control_p<p>.control`` per entry of ``p_levels``, the FSP
        controls (``fsp=True``), ``extra_dirs``, ``SETUP/`` and a
        ``run.sh`` link. Existing folders are skipped (``exist = True``)
        unless ``overwrite``.

        Every control starts from ``control_template`` and is changed key
        by key, in this order: ``control_values``, the restart ladder,
        ``p_values[p]``, and the keys the ring owns (``mesh file name``,
        ``Polynomial order``, ``solution file name``).

        Parameters
        ----------
        control_template : str
            A working ``.control``. Its ``#define`` blocks (boundaries,
            monitors…) are copied untouched. It must declare every boundary
            of the mesh; the inner zones (``BCType`` 100, e.g. refinement
            surfaces) may be left out: HORSES3D still wants a ``type`` for
            them, so the ring adds a ``#define boundary`` block with
            ``type = FreeSlipWall`` and says so. That type is never applied:
            faces with an element on each side stay interior (``Number of
            faces: 0`` for those zones in the solver log).
        mesh_path : str
            HOPR/PyHOPE ``.h5`` mesh. Its name must carry ``g<Ngeo>``.
        tag : str
            Short campaign label for the solution names
            (``RESULTS/<tag>_g<g>p<p>.hsol``). Letters, digits, ``-`` and
            ``_``; it must not end in ``_<digits>`` (autosave pattern).
        folder_fmt : str
            Case folder name, formatted with the design variables, e.g.
            ``'tsr_{tsr:.1f}_M_{M:.4f}'``.
        p_levels : list of int
            Polynomial orders, ascending, each ``>= g``. The first starts
            from scratch; each next one restarts from the previous
            solution (``restart polorder`` = previous ``p``).
        control_values : dict, optional
            ``{control key: df_cases column or value}`` for every ``p``,
            e.g. ``{'mach number': 'M', 'final time': 20.0}``. A string
            naming a column takes the case's value; ``None`` removes the
            key.
        p_values : dict, optional
            ``{p: {control key: value}}`` for one level, e.g. a later
            absolute ``final time`` for ``p=3``. Overrides the ladder.
            A key inside a ``#define`` block is named with its block, e.g.
            ``'#define statistics: starting time'`` (also in
            ``control_values``); the template must have that block.
        extra_dirs : list of str, optional
            Folders copied into each case (e.g. ``ActuatorDef``). Text files
            in them get ``data_to_update`` substituted.
        data_to_update : dict, optional
            ``{PLACEHOLDER: df_cases column or literal}``, as in the CODA
            ring, for the files of ``extra_dirs``.
        setup_dir : str, optional
            ``SETUP`` folder (``ProblemFile.f90``, ``Makefile``) copied as
            ``SETUP/``. Compile it afterwards with ``build_setup.sh``.
        run_script : str, optional
            Launcher linked as ``run.sh`` (``utils_H3d/run.sh``). Needed by
            :meth:`submit_cases`.
        fsp : bool
            Also write ``fsp_p<p>.control`` (see :meth:`fsp_control`).
            Default True.
        fsp_steps : int
            Time steps of the FSP test. Default 200.
        strip_comments : bool
            Remove ``!`` comments from the controls. Default True.
        overwrite : bool
            Rewrite the controls, mesh, ``extra_dirs``, ``SETUP`` and
            ``run.sh`` of existing case folders. ``RESULTS/`` and ``LOG/``
            are never deleted. Default False (existing folders are
            skipped).

        Returns
        -------
        pd.DataFrame
            ``df_cases`` with ``folder`` and ``exist``.

        Raises
        ------
        RuntimeError
            Without :meth:`define_cases`.
        ValueError
            Bad ``tag``, ``p_levels`` not ascending or below ``g``, a mesh
            name without ``g<Ngeo>``, a ring-owned key in
            ``control_values`` or ``p_values``, a control that would miss a key HORSES3D
            requires, or a mesh zone without a ``#define boundary`` block
            (all checked before writing anything).
        FileNotFoundError
            Missing template, mesh, extra folder, SETUP or run script.

        Examples
        --------
        ::

            gdf.generate_folders(
                control_template='sources/turbine.control',
                mesh_path='sources/Wind_turbine_disk_fluid_cgns_g2_mesh.h5',
                tag='wt', folder_fmt='tsr_{tsr:.1f}',
                p_levels=[2, 3],
                control_values={'mach number': 0.02934, 'final time': 20.0},
                p_values={3: {'final time': 25.0}},
                extra_dirs=['sources/ActuatorDef'],
                data_to_update={'OMEGA_PLACEHOLDER': 'omega'},
                run_script='/home/m.jaraiz/Documentos/utils_H3d/run.sh',
            )
        """
        if self.case_tensor is None or self.design_vars is None:
            raise RuntimeError(
                "Cases must be defined before generating folders. "
                "Use define_cases()."
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag or '') \
                or re.search(r"_\d+$", tag):
            raise ValueError(
                f"Invalid tag {tag!r}: letters, digits, '-' and '_', not "
                "ending in '_<digits>'."
            )
        p_levels = [int(p) for p in p_levels]
        if not p_levels or p_levels != sorted(set(p_levels)):
            raise ValueError(f"p_levels must be ascending and unique: {p_levels}.")
        for path, what in ((control_template, 'Control template'),
                           (mesh_path, 'Mesh')):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{what} not found: {path}")
        for path in (extra_dirs or []) + ([setup_dir] if setup_dir else []):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Folder not found: {path}")
        if run_script and not os.path.isfile(run_script):
            raise FileNotFoundError(f"Run script not found: {run_script}")
        owned = [k for k in (control_values or {}) if _norm_key(k) in _RING_KEYS]
        owned += [f"p_values[{p}]['{k}']" for p, vals in (p_values or {}).items()
                  for k in vals if _norm_key(k) in _RING_KEYS]
        if owned:
            raise ValueError(
                f"{owned} are set by the ring (mesh, p, solution name)."
            )

        mesh_info = _read_mesh_info(mesh_path)
        g = mesh_info['ngeo']
        mesh_name = os.path.basename(mesh_path)
        g_names = {int(x) for x in _MESH_G_RE.findall(mesh_name)}
        if g is None or g_names != {g}:
            raise ValueError(
                f"Mesh '{mesh_name}' must carry its geometric order as "
                f"'g{g}' (Ngeo = {g}) in its name."
            )
        low = [p for p in p_levels if p < g]
        if low:
            raise ValueError(f"p_levels {low} are below the mesh order g={g}.")

        with open(control_template) as fh:
            template = fh.read()
        self.tag, self.mesh_g, self.p_levels = tag, g, p_levels
        self.eq_type = _control_keys(template).get('flow equations')
        self.folder_fmt = folder_fmt
        self.folders_name = []
        self.mesh_path = mesh_path

        zones = mesh_info['boundaries'] + mesh_info['inner']
        first_values = {
            k: self._case_value(v, dict(zip(self.design_vars,
                                            map(float, self.case_tensor[0]))), 0)
            for k, v in (control_values or {}).items()
        }
        defined = _defined_zones(template)
        auto_inner = [z for z in mesh_info['inner'] if z not in defined]
        for k, p in enumerate(p_levels):
            text = self._control_text(
                template, first_values, p, p_levels[k - 1] if k else None,
                (p_values or {}).get(p, {}), mesh_name, strip_comments,
                auto_inner,
            )
            self._check_mandatory(text, self.control_name(p))
            self._check_zones(text, self.control_name(p), zones)
            if fsp:
                self._check_mandatory(
                    self.fsp_control(text, p, zones, fsp_steps),
                    self.control_name(p, fsp=True),
                )

        if auto_inner:
            print(
                f"[horses3d ring] Inner zone(s) {auto_inner} have no "
                "'#define boundary' block in the template: added one with "
                f"type = {_INNER_ZONE_TYPE} to every control. HORSES3D wants a "
                "type for every zone of the mesh, but these surfaces have no "
                "boundary faces (they stay interior), so the type is never "
                "applied."
            )

        df_cases = self.df_cases.copy()
        df_cases['exist'] = None
        df_cases['folder'] = None
        out_dir = os.path.join(self.root_dir, 'outputs')
        log.info("Creating %s HORSES3D case folders in %s",
                 len(self.case_tensor), out_dir)

        for i, row in enumerate(self.case_tensor):
            case_dict = {v: float(row[j]) for j, v in enumerate(self.design_vars)}
            try:
                folder = folder_fmt.format(**case_dict)
            except KeyError as e:
                raise KeyError(f"Variable {e} not found in design_vars ({self.design_vars})")
            self.folders_name.append(folder)
            df_cases.loc[i, 'folder'] = folder
            case_dir = os.path.join(out_dir, folder)
            existed = os.path.exists(case_dir)
            df_cases.loc[i, 'exist'] = existed
            if existed and not overwrite:
                log.info("Folder %s already exists. Skipping...", case_dir)
                continue

            for sub in ('MESH', 'RESULTS', 'LOG'):
                os.makedirs(os.path.join(case_dir, sub), exist_ok=True)
            shutil.copyfile(mesh_path, os.path.join(case_dir, 'MESH', mesh_name))

            case_values = {
                k: self._case_value(v, case_dict, i)
                for k, v in (control_values or {}).items()
            }
            for k, p in enumerate(p_levels):
                text = self._control_text(
                    template, case_values, p,
                    p_levels[k - 1] if k else None,
                    (p_values or {}).get(p, {}), mesh_name, strip_comments,
                    auto_inner,
                )
                with open(os.path.join(case_dir, self.control_name(p)), 'w') as fh:
                    fh.write(text)
                if fsp:
                    os.makedirs(os.path.join(case_dir, 'RESULTS', 'FSP'), exist_ok=True)
                    with open(os.path.join(case_dir, self.control_name(p, fsp=True)), 'w') as fh:
                        fh.write(self.fsp_control(text, p, zones, fsp_steps))

            for src in extra_dirs or []:
                dst = os.path.join(case_dir, os.path.basename(os.path.normpath(src)))
                shutil.copytree(src, dst, dirs_exist_ok=True)
                if data_to_update:
                    self._update_tree(dst, data_to_update, case_dict, i,
                                      self.df_cases)
            if setup_dir:
                shutil.copytree(setup_dir, os.path.join(case_dir, 'SETUP'),
                                dirs_exist_ok=True)
            if run_script:
                link = os.path.join(case_dir, 'run.sh')
                if os.path.lexists(link):
                    os.remove(link)
                os.symlink(os.path.abspath(run_script), link)

        self._write_cases_metadata(df_cases, extra={
            'solver': 'HORSES3D',
            'tag': tag,
            'mesh': {'file': mesh_name, 'g': g,
                     'source': os.path.abspath(mesh_path),
                     'boundaries': mesh_info['boundaries'],
                     'inner': mesh_info['inner']},
            'p_levels': p_levels,
            'control_template': os.path.abspath(control_template),
            'fsp': bool(fsp),
        })
        return df_cases

    def _case_value(self, value, case_dict: dict, i: int):
        """A ``control_values`` entry for case ``i``: a column, or itself."""
        if isinstance(value, str) and value in getattr(self.df_cases, 'columns', []):
            if value in case_dict:
                return case_dict[value]
            return self.df_cases[value].iloc[i]
        return value

    def _control_text(
        self, template: str, case_values: dict, p: int,
        p_prev: Optional[int], level_values: dict, mesh_name: str,
        strip_comments: bool, auto_inner: Optional[list] = None,
    ) -> str:
        """The ``file_control_p<p>.control`` of one case; ``auto_inner``
        zones get a ``#define boundary`` block of their own."""
        values = dict(case_values)
        if p_prev is None:
            values.update({'restart': False,
                           'restart file name': _quoted(self.solution_name(p)),
                           'restart polorder': None})
        else:
            values.update({'restart': True,
                           'restart file name': _quoted(self.solution_name(p_prev)),
                           'restart polorder': p_prev})
        values.update(level_values)
        values.update({
            'mesh file name': _quoted(f"MESH/{mesh_name}"),
            'Polynomial order': p,
            'solution file name': _quoted(self.solution_name(p)),
        })
        text = _rewrite_control(template, values, strip_comments)
        if auto_inner:
            text += (f"\n#define boundary {'__'.join(auto_inner)}\n"
                     f"   type = {_INNER_ZONE_TYPE}\n#end\n")
        return text

    def fsp_control(
        self, control_text: str, p: int, zones: list, steps: int = 200,
    ) -> str:
        """
        Free-stream preservation test derived from a case control.

        The same flow, the same mesh and order, but ``Inflow`` on every
        boundary of the mesh and nothing else: no restart, actuator line or
        disk, IBM, monitors, probes or saved surfaces. The uniform initial
        state is then an exact solution, and a valid mesh (consistent
        metric terms, no inverted elements) keeps it to round-off (judge
        it with :meth:`fsp_report`). It checks the mesh, not any
        physics.

        Parameters
        ----------
        control_text : str
            The ``file_control_p<p>.control`` of the case.
        p : int
            Polynomial order.
        zones : list of str
            Every zone name of the mesh (``BCNames``). HORSES3D wants a
            ``type`` for each, inner surfaces included; those have no
            boundary faces, so ``Inflow`` there changes nothing.
        steps : int
            Time steps. Default 200.

        Returns
        -------
        str
            Text of ``fsp_p<p>.control``.
        """
        keys = _control_keys(control_text)
        values = {k: None for k in keys if k.startswith(_FSP_DROP_PREFIXES)}
        values.update({k: False for k in _FSP_OFF_KEYS if k in keys})
        values.update({
            'restart': False,
            'restart file name': _quoted(self.solution_name(p, fsp=True)),
            'number of time steps': int(steps),
            'output interval': max(1, int(steps) // 10),
            'autosave mode': '"iteration"',
            'autosave interval': int(steps) + 1,
            'solution file name': _quoted(self.solution_name(p, fsp=True)),
        })
        text = _rewrite_control(control_text, values, drop_defines=True)
        block = (f"\n#define boundary {'__'.join(zones)}\n"
                 "   type = Inflow\n#end\n")
        return text + block

    @staticmethod
    def _check_zones(text: str, name: str, zones: list) -> None:
        """Every mesh zone needs a ``#define boundary`` block (HORSES3D stops
        with "Missing boundary condition type" otherwise); by now the inner
        ones were added by the ring, so a missing zone is a real boundary.

        A surface monitor whose ``marker`` is not a zone is an error: HORSES3D
        disables it but still updates it with zone -1 and crashes
        (SIGSEGV in ``VectorSurfaceIntegral``). An unknown name in
        ``boundaries to save`` is skipped by HORSES3D, so it only warns."""
        missing = [z for z in zones if z not in _defined_zones(text)]
        if missing:
            raise ValueError(
                f"{name}: no '#define boundary' block for mesh boundary "
                f"zone(s) {missing}. Declare their condition in the template."
            )
        markers, saved = _referenced_zones(text)
        bad = sorted(markers - set(zones))
        if bad:
            raise ValueError(
                f"{name}: surface monitor marker(s) {bad} are not zones of the "
                f"mesh {sorted(zones)}. A marker names one zone ('a__b' is not "
                "split): write one monitor per zone. HORSES3D would crash."
            )
        unknown = sorted(saved - set(zones))
        if unknown:
            warnings.warn(
                f"{name}: 'boundaries to save' names {unknown}, which are not "
                "zones of the mesh: HORSES3D skips them. Write the zones "
                "separated by commas ('a__b' is not split).",
                UserWarning,
            )

    def _check_mandatory(self, text: str, name: str) -> None:
        """Raise if a control misses a key HORSES3D refuses to start without."""
        keys = _control_keys(text)
        required = list(_MANDATORY_KEYS)
        if keys.get('flow equations', 'NS').upper() == 'NS':
            required += list(_MANDATORY_NS_KEYS)
        missing = [k for k in required if k not in keys]
        if missing:
            raise ValueError(
                f"{name} would miss keys HORSES3D requires: {missing}. Add them "
                "to the template or to control_values."
            )

    @staticmethod
    def _update_tree(root: str, data_to_update: dict, case_dict: dict, i: int,
                     df_cases=None) -> None:
        """Substitute placeholders in every text file under ``root``."""
        for base, _, files in os.walk(root):
            for name in files:
                path = os.path.join(base, name)
                try:
                    with open(path) as fh:
                        content = fh.read()
                except UnicodeDecodeError:
                    continue
                new = content
                for placeholder, var in data_to_update.items():
                    new = new.replace(
                        placeholder,
                        _resolve_update_value(df_cases, var, case_dict, i),
                    )
                if new != content:
                    with open(path, 'w') as fh:
                        fh.write(new)

    # ── launching ─────────────────────────────────────────────────────────

    def load_campaign(self) -> None:
        """Recover a campaign written earlier from ``cases_metadata.json``.

        Restores ``tag``, the mesh ``g``, ``p_levels``, ``folder_fmt``,
        ``design_vars``, ``df_cases`` and the folder names, so that
        :meth:`submit_cases` and :meth:`fsp_report` work in a new session
        without calling :meth:`generate_folders` again.

        Raises
        ------
        FileNotFoundError
            Without ``metadata/cases_metadata.json``.
        ValueError
            If the metadata was not written by this ring.
        """
        path = os.path.join(self.root_dir, 'metadata', 'cases_metadata.json')
        with open(path) as fh:
            meta = json.load(fh)
        if meta.get('solver') != 'HORSES3D':
            raise ValueError(f"{path} was not written by the horses3d ring.")
        self.tag = meta['tag']
        self.mesh_g = meta['mesh']['g']
        self.p_levels = meta['p_levels']
        self.folder_fmt = meta['folder_fmt']
        self.design_vars = meta['design_vars']
        self.df_cases = pd.DataFrame(meta['df_cases'])
        self.folders_name = list(self.df_cases['folder'])

    def _case_dirs(self) -> list:
        if not getattr(self, 'folders_name', None):
            self.load_campaign()
        return [(f, os.path.join(self.root_dir, 'outputs', f))
                for f in self.folders_name]

    def assign_jobs(self, *args, **kwargs):
        """Not used by HORSES3D: ``run.sh`` takes node and resources as
        arguments. Use :meth:`submit_cases`."""
        raise NotImplementedError(
            "The horses3d ring launches through run.sh: use "
            "submit_cases(p, repo, arch=..., node=..., ...)."
        )

    def submit_cases(
        self,
        p: Union[int, list, tuple],
        repo: str,
        arch: str = 'GPU',
        node: str = 'n001',
        ngpu: Union[int, dict] = 1,
        cpus_per_gpu: int = 16,
        ncpus: Union[int, dict] = 48,
        branch: Optional[str] = None,
        solver: str = 'ns',
        fsp: bool = False,
        cases: Optional[list] = None,
        dry_run: bool = True,
    ) -> pd.DataFrame:
        """
        Launch the control of order ``p`` of every case through ``run.sh``.

        Runs ``HORSES3D_REPO=<repo> SOLVER=<solver> ./run.sh <control>
        <branch> GPU <node> <ngpu> <cpus_per_gpu>`` (or ``… CPU <node>
        <ncpus>``) inside each case folder; ``run.sh`` checks the control,
        the solver and ``SETUP`` before calling ``sbatch``. A case whose
        solution already has a ``.hsol`` or a ``.residuals`` is skipped.

        ``p`` may be a list of ascending orders, e.g. ``[2, 3, 4, 5]``: the
        restart ladder is then chained. In each case, every order is
        submitted with ``DEPENDENCY=afterok:<job of the previous order>``,
        so Slurm starts it only when the previous one ends well and cancels
        it if that one fails (``run.sh`` adds ``--kill-on-invalid-dep=yes``).
        An order already started (skipped) leaves no job to wait for: the
        next one is submitted without a dependency and ``run.sh`` then needs
        its restart file. If an order fails or has no control, the rest of
        that case's chain is not submitted.

        The solver repository is always explicit: ``HORSES3D_REPO`` is set
        from ``repo`` for every case and any value already in the
        environment is ignored. Before planning anything, the repository
        must have ``Solver/bin/horses3d.<solver>`` and be on ``branch``.
        ``SETUP`` must be compiled against the same repository
        (``HORSES3D_REPO=<repo> build_setup.sh``); ``run.sh`` checks it.

        Parameters
        ----------
        p : int or list of int
            Polynomial order to launch, or ascending orders to chain.
        repo : str
            Solver repository (``~`` is expanded), e.g.
            ``'~/repos/horses3d-gpu'`` or a clone of a development branch
            such as ``'~/repos/horses3d-gpu-ad'``.
        arch : {'GPU', 'CPU'}
        node : str
            Slurm node list, e.g. ``'n001'`` or ``'n[001,003-007]'``.
        ngpu : int or dict
            GPUs of a GPU run. Default 1. A dict ``{p: n}`` sets them per
            order, e.g. ``{2: 1, 3: 1, 4: 2, 5: 4}``; it must have every
            order of ``p``.
        cpus_per_gpu : int
            CPUs of each MPI task of a GPU run. Default 16.
        ncpus : int or dict
            MPI tasks of a CPU run. Default 48. A dict ``{p: n}`` sets them
            per order, like ``ngpu``.
        branch : str, optional
            Branch ``repo`` must be on. Default: the branch it is on.
            Never checked out, only compared.
        solver : str
            ``ns``, ``ins``, ``mu`` or ``nssa``. Default ``'ns'``.
        fsp : bool
            Launch ``fsp_p<p>.control`` instead (a single order only).
            Default False.
        cases : list of str, optional
            Folder names. Default: every case.
        dry_run : bool
            Only report the commands. Default True.

        Returns
        -------
        pd.DataFrame
            One row per case and order: ``folder``, ``p``, ``control``,
            ``status`` (``'planned'``, ``'submitted'``, ``'skipped'``,
            ``'failed'``), ``command``, ``job_id``, ``depends_on`` (job it
            waits for; ``'<p2>'`` in a dry run) and ``message``.

        Examples
        --------
        ::

            repo = '~/repos/horses3d-gpu'
            plan = gdf.submit_cases(2, repo, fsp=True)          # what would run
            gdf.submit_cases(2, repo, fsp=True, dry_run=False)  # from nibbler
            gdf.submit_cases([2, 3, 4, 5], repo, ngpu={2: 1, 3: 1, 4: 2, 5: 4},
                             dry_run=False)                       # chained
        """
        arch = arch.upper()
        if arch not in ('GPU', 'CPU'):
            raise ValueError("arch must be 'GPU' or 'CPU'.")
        levels = [p] if isinstance(p, (int, np.integer)) else list(p)
        levels = [int(x) for x in levels]
        if not levels or levels != sorted(set(levels)):
            raise ValueError(f"p must be an order or ascending unique orders: {p}.")
        if fsp and len(levels) > 1:
            raise ValueError("FSP tests are not chained: pass a single p with fsp=True.")
        repo, branch = self._check_solver_repo(repo, solver, branch)
        self.solution_name(levels[0], fsp=fsp)   # raises if no campaign is loaded
        def per_order(value, name):
            if not isinstance(value, dict):
                return {level: int(value) for level in levels}
            missing = [level for level in levels if level not in value]
            if missing:
                raise ValueError(f"{name} has no value for p = {missing}.")
            return {level: int(value[level]) for level in levels}

        if arch == 'GPU':
            gpus = per_order(ngpu, 'ngpu')
            tails = {level: [str(gpus[level]), str(cpus_per_gpu)] for level in levels}
        else:
            cpus = per_order(ncpus, 'ncpus')
            tails = {level: [str(cpus[level])] for level in levels}
        base_env = dict(os.environ, SOLVER=solver, HORSES3D_REPO=repo)
        base_env.pop('DEPENDENCY', None)
        if not dry_run and shutil.which('sbatch') is None:
            raise RuntimeError(
                f"'sbatch' not found on {socket.gethostname()}: launch from "
                "a Slurm login node."
            )

        rows = []
        for folder, case_dir in self._case_dirs():
            if cases is not None and folder not in cases:
                continue
            prev_job, stopped = None, None
            for level in levels:
                control = self.control_name(level, fsp=fsp)
                solution = self.solution_name(level, fsp=fsp)
                command = ['./run.sh', control, branch, arch, node] + tails[level]
                dep = f"afterok:{prev_job}" if prev_job else None
                prefix = f"HORSES3D_REPO={repo} SOLVER={solver} "
                if dep:
                    prefix += f"DEPENDENCY={dep} "
                row = {'folder': folder, 'p': level, 'control': control,
                       'status': 'planned', 'command': prefix + ' '.join(command),
                       'job_id': None, 'depends_on': prev_job, 'message': ''}
                stem = os.path.join(case_dir, os.path.splitext(solution)[0])
                if stopped is not None:
                    row.update(status='skipped', depends_on=None,
                               message=f"p{stopped} was not submitted: chain stopped.")
                elif not os.path.isfile(os.path.join(case_dir, control)):
                    row.update(status='skipped', message=f"{control} not found.")
                    stopped = level
                elif not os.path.exists(os.path.join(case_dir, 'run.sh')):
                    row.update(status='skipped', message="run.sh not found.")
                    stopped = level
                elif any(os.path.exists(stem + ext) for ext in ('.hsol', '.residuals')):
                    row.update(status='skipped', message="already started.")
                    prev_job = None
                elif dry_run:
                    prev_job = f"<p{level}>"
                else:
                    env = dict(base_env, DEPENDENCY=dep) if dep else base_env
                    res = subprocess.run(
                        command, cwd=case_dir, capture_output=True, text=True,
                        env=env,
                    )
                    m = re.search(r"Submitted batch job (\d+)", res.stdout)
                    if res.returncode == 0 and m:
                        row.update(status='submitted', job_id=m.group(1))
                        prev_job = m.group(1)
                    else:
                        row.update(status='failed',
                                   message=(res.stdout + res.stderr).strip()[-500:])
                        stopped = level
                rows.append(row)
        return pd.DataFrame(rows, columns=['folder', 'p', 'control', 'status',
                                           'command', 'job_id', 'depends_on',
                                           'message'])

    @staticmethod
    def _check_solver_repo(repo: str, solver: str,
                           branch: Optional[str]) -> tuple:
        """
        Resolve and check the solver repository of :meth:`submit_cases`.

        Returns the absolute path of ``repo`` and the branch to pass to
        ``run.sh``. Raises if the repository or ``Solver/bin/horses3d.<solver>``
        is missing, if it is not on a branch (detached ``HEAD``) or if it is
        not on ``branch``.
        """
        repo = os.path.abspath(os.path.expanduser(str(repo)))
        if not os.path.isdir(os.path.join(repo, 'Solver')):
            raise FileNotFoundError(f"{repo} is not a HORSES3D repository "
                                    "(no Solver/ folder).")
        exe = os.path.join(repo, 'Solver', 'bin', f'horses3d.{solver}')
        if not os.access(exe, os.X_OK):
            raise FileNotFoundError(f"{exe} not found: compile the solver "
                                    "in that repository first.")
        # cwd instead of 'git -C' and no '--short': flexo has an old git
        res = subprocess.run(['git', 'symbolic-ref', 'HEAD'], cwd=repo,
                             capture_output=True, text=True)
        current = res.stdout.strip()
        if current.startswith('refs/heads/'):
            current = current[len('refs/heads/'):]
        if res.returncode != 0 or not current:
            raise RuntimeError(f"{repo} is not on a branch (detached HEAD or "
                               "not a git repository); run.sh needs one.")
        if branch is None:
            branch = current
        elif branch != current:
            raise ValueError(f"{repo} is on branch '{current}', not "
                             f"'{branch}'. Pass branch='{current}' or use "
                             "another repository.")
        return repo, branch

    # ── checks ────────────────────────────────────────────────────────────

    def fsp_report(self, p: int, tol: float = 1e-6) -> pd.DataFrame:
        """
        Read the FSP tests of order ``p``: does each mesh keep free stream?

        The verdict comes from the solution, not from the residuals: the
        final ``.hsol`` is compared node by node with the uniform state the
        solver wrote at iteration 0 (``<stem>_0000000000.hsol``). The
        residuals are reported for information only: at low Mach they are
        far from zero even for an exact uniform state (round-off of large
        energy fluxes summed over every degree of freedom: up to 1.6e-4 on
        the NTNU turbine mesh, whose solution moved by 5e-9 relative).

        Parameters
        ----------
        p : int
            Polynomial order of the test.
        tol : float
            Largest relative change of the solution accepted as round-off.
            Default 1e-6 (a sound mesh gives ~1e-9 after 200 steps; an
            inverted or badly stitched element gives orders of magnitude
            more).

        Returns
        -------
        pd.DataFrame
            One row per case: ``folder``, ``status`` (``'pending'``,
            ``'running'`` while the final ``.hsol`` is missing or
            incomplete, ``'done'``), ``iterations``, ``max_residual``,
            ``last_residual``, ``max_deviation`` (largest change of any
            variable, relative to its free-stream scale: density, momentum
            magnitude, energy), ``worst_element`` (0-based) and
            ``preserved`` (``max_deviation < tol``).

        Examples
        --------
        ::

            gdf.fsp_report(2)
        """
        from ..residuals.horses3d import Horses3DResiduals   # deferred

        rows = []
        for folder, case_dir in self._case_dirs():
            stem = os.path.join(
                case_dir, os.path.splitext(self.solution_name(p, fsp=True))[0])
            row = {'folder': folder, 'status': 'pending', 'iterations': 0,
                   'max_residual': None, 'last_residual': None,
                   'max_deviation': None, 'worst_element': None,
                   'preserved': None}
            if os.path.isfile(stem + '.residuals'):
                row['status'] = 'running'
                try:
                    df = Horses3DResiduals._read_residuals_file(stem + '.residuals')
                except ValueError:
                    df = None
                if df is not None and len(df):
                    cols = (['Max-Residual'] if 'Max-Residual' in df.columns else
                            [c for c in df.columns if c not in (
                                'Iteration', 'Time', 'Total_elapsed_Time(s)',
                                'Solver_elapsed_Time(s)')])
                    res = df[cols].astype(float).abs().max(axis=1)
                    row.update(iterations=len(df), max_residual=float(res.max()),
                               last_residual=float(res.iloc[-1]))
            deviation = self._fsp_deviation(stem)
            if deviation is not None:
                row.update(status='done', max_deviation=deviation[0],
                           worst_element=deviation[1],
                           preserved=bool(deviation[0] < tol))
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _fsp_deviation(stem: str):
        """``(max relative change, worst element)`` between the iteration-0
        and the final solution of an FSP test, or None if either is missing
        or the final one is incomplete."""
        from ...solver_tools import horses3d as h3d   # deferred: numpy reader

        final, first = stem + '.hsol', stem + '_0000000000.hsol'
        if not (os.path.isfile(final) and os.path.isfile(first)):
            return None
        try:
            if not h3d.read_header(final).complete:
                return None
            q0 = h3d.read_solution(first)['Q']
            q1 = h3d.read_solution(final)['Q']
        except ValueError:
            return None
        ref = h3d.to_nodes(q0)
        scale = np.array([
            np.abs(ref[:, 0]).max(),
            *[np.linalg.norm(ref[:, 1:4], axis=1).max()] * 3,
            np.abs(ref[:, 4]).max(),
        ])
        scale[scale == 0] = 1.0
        per_elem = [float((np.abs(np.asarray(b) - np.asarray(a)).reshape(-1, 5)
                           / scale).max()) for a, b in zip(q0, q1)]
        worst = int(np.argmax(per_elem))
        return per_elem[worst], worst
