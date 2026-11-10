"""Follow a CODA run through its Slurm log (``<job>-<id>-out.txt``).

While CODA runs, its log is the only file that grows: the
``output_<s>__monitors_*.dat`` monitors are exported once, at the end of
each stage. This module reads that log incrementally, so that a run can be
watched while it happens (see :mod:`FotR.solver_tools.coda_monitor`).

It only needs the standard library and must not import anything else from
the package: the monitor server runs it in a process of its own.

What the log contains
---------------------
Measured on CODA 2024.04 logs written by ``run_sst_v5.py`` (TIFON), one
line per iteration and no buffering delay:

* ``date`` before and after ``mpiexec`` (``run.sh``). The second one means
  that the job ended on its own (not killed by Slurm).
* `` STAGE <s>`` when a stage begins.
* The stop criteria of the stage, between ``Using the following stop
  criteria parameters`` and a line ``],``: relative reduction
  (``variables``, ``threshold``), Cauchy (``variables``, ``thresholds``,
  ``number of samples``) and iteration limit (``NumberOfIterations``).
* ``Monitor variables: ...`` names the values of the next
  ``Iteration N: v1 v2 ...`` lines: first the initial residual (one line,
  no ``CFL``), then one line per iteration with ``CFL`` and the residual
  reductions.
* After each iteration line, ``Iteration N: BoundaryIntegralValues:
  <category>`` and one ``<name> = <value>`` line per integral
  (``CoefDrag``, ``CoefLift``, ``CoefMomentY``).
* ``Iteration N: Cauchy convergence achieved over <n> samples.`` at every
  iteration where CODA's Cauchy criterion holds, and ``Iteration N:
  residual stagnation achieved over <n> samples (tol <t>, ref <r>).``,
  printed by the ``StopResidualStagnation`` callback of ``run_sst_v5.py``.

Cauchy criterion
----------------
:meth:`CodaLog.checks` computes it as ``max |x_i - x_k|`` over the current
value ``x_k`` and the ``number of samples`` values before it, in the same
stage, below the threshold of each variable. This was deduced from the
logs alone, not from CODA's code: it gives CODA's ``Cauchy convergence
achieved`` message at every one of the 2.1 million iterations of the 329
rans5 logs (27/10/2026), where ``max - min`` over the window disagrees on
about 2 % of them.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

__all__ = ['STATUSES', 'find_logs', 'CodaLog', 'CodaCase']

#: Status codes of :meth:`CodaCase.summary`, in the order a run goes through.
STATUSES = (
    'loading',    # not read yet
    'pending',    # no log in the folder
    'queued',     # last log ended and its stage output became ci.h5 (restart)
    'starting',   # log growing, no iteration yet
    'running',
    'quiet',      # no write for longer than ``stale_after``
    'finished',   # job ended before the iteration limit
    'limit',      # job ended at the iteration limit of its last stage
    'diverged',   # NaN in the residuals or the integrals
    'cancelled',  # Slurm cancelled the job (time limit, scancel)
    'error',      # job ended and wrote to its -err.txt
)

_RE_ITERATION = re.compile(r'Iteration (\d+):\s+(.*?)\s*$')
_RE_EVENT = re.compile(
    r'^(Cauchy convergence|residual stagnation) achieved over (\d+) samples'
    r'(?: \(tol ([^,]+), ref ([^)]+)\))?'
)
_EVENT_KIND = {'Cauchy convergence': 'cauchy',
               'residual stagnation': 'stagnation'}
_RE_MONITOR = re.compile(r'Monitor variables:\s+(.*?)\s*$')
_RE_STAGE = re.compile(r'^\s*STAGE (\d+)\s*$')
_RE_INTEGRAL = re.compile(r'^\[\d+\]\s+(\w+) = (\S+)\s*$')
_RE_DATE = re.compile(
    r'^\s*(\w+)\.? (\w+)\.? +(\d{1,2}) (\d{2}):(\d{2}):(\d{2}) \S+ (\d{4})\s*$'
)
_RE_JOB_ID = re.compile(r'-(\d+)-out\.txt$')
_RE_SBATCH = re.compile(r'^#SBATCH\s+--([\w-]+)=(\S+)', re.MULTILINE)

_CRITERIA_START = 'Using the following stop criteria parameters'
_CRITERION_KEYS = {
    'threshold': re.compile(r"'threshold':\s*([^,\s]+)"),
    'thresholds': re.compile(r"'thresholds':\s*\[([^\]]*)\]"),
    'samples': re.compile(r"'number of samples':\s*(\d+)"),
    'max_iterations': re.compile(r"'NumberOfIterations':\s*(\d+)"),
    'category': re.compile(r"'data log category':\s*'([^']*)'"),
    'combine': re.compile(r"'combine with':\s*'(\w+)'"),
}
_CRITERION_KIND = {
    'stop relative reduction': 'residual',
    'stop Cauchy criteria': 'cauchy',
    'stop number of iterations': 'iterations',
}

# Month abbreviations printed by ``date`` in Spanish and English locales.
_MONTHS = {
    'ene': 1, 'jan': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'apr': 4, 'may': 5,
    'jun': 6, 'jul': 7, 'ago': 8, 'aug': 8, 'sep': 9, 'set': 9, 'oct': 10,
    'nov': 11, 'dic': 12, 'dec': 12,
}


def _finite(value):
    """``value`` if it is a finite number, else None (JSON has no NaN)."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _parse_date(match) -> Optional[float]:
    """Epoch seconds (local time) of a ``date`` line, or None."""
    month = _MONTHS.get(match.group(2).lower()[:3])
    if month is None:
        return None
    try:
        stamp = datetime(int(match.group(7)), month, int(match.group(3)),
                         int(match.group(4)), int(match.group(5)),
                         int(match.group(6)))
    except ValueError:
        return None
    return time.mktime(stamp.timetuple())


def find_logs(case_dir: str, suffix: str = '-out.txt') -> list:
    """Slurm logs of a case folder, oldest job first.

    Parameters
    ----------
    case_dir : str
        Case folder.
    suffix : str
        End of the log file names. Default ``'-out.txt'`` (``run.sh``
        writes ``%x-%j-out.txt``).

    Returns
    -------
    list[str]
        Full paths, sorted by Slurm job id (by modification time for
        names without one). Empty if the folder does not exist.

    Examples
    --------
    ::

        newest = find_logs('outputs/aoa_4.3363_mach_0.7247')[-1]
    """
    try:
        names = [n for n in os.listdir(case_dir) if n.endswith(suffix)]
    except (FileNotFoundError, NotADirectoryError):
        return []

    def key(name):
        m = _RE_JOB_ID.search(name)
        if m:
            return (0, int(m.group(1)), 0.0)
        try:
            mtime = os.path.getmtime(os.path.join(case_dir, name))
        except OSError:
            mtime = 0.0
        return (1, 0, mtime)

    return [os.path.join(case_dir, n) for n in sorted(names, key=key)]


class CodaLog:
    """Incremental parser of one CODA Slurm log.

    Every :meth:`update` reads only the bytes written since the previous
    call, up to the last complete line, so it can be called every few
    seconds on a log that is still growing (a full 3.5 MB log takes about
    0.2 s, an update well under 1 ms). A file that shrinks is read again
    from the start.

    Parameters
    ----------
    path : str
        Log file (``<job>-<id>-out.txt``).

    Attributes
    ----------
    columns : dict[str, list]
        One list per quantity, all of length :attr:`n_rows`, one entry per
        iteration line: ``stage``, ``iteration``, the monitor variables of
        the log (``CFL``, ``DensityResidualReduction``...) and the
        integrals (``CoefDrag``...). Missing or non-finite values are None.
    n_rows : int
        Iteration lines read so far.
    criteria : dict[int, dict]
        Stop criteria per stage: ``'residual'`` (``variables``,
        ``threshold``), ``'cauchy'`` (``variables``, ``thresholds``,
        ``samples``, ``category``), ``'iterations'`` (``max_iterations``),
        each with its ``combine`` flag.
    initial_residual : dict[int, dict]
        Absolute initial residual per stage.
    events : dict[int, dict]
        Convergence messages per stage: ``'cauchy'`` (CODA) and
        ``'stagnation'`` (``run_sst_v5.py``), each ``{'first_iteration',
        'last_iteration', 'samples', 'tolerance'}``.
    started, ended : float or None
        Epoch seconds of the first and the second ``date`` line.
    mtime, size : float, int
        Of the log at the last update.
    has_nan : bool
        A residual or an integral was NaN or infinite.

    Examples
    --------
    ::

        log = CodaLog('outputs/aoa_4.3363_mach_0.7247/a4.34m0.72-51685-out.txt')
        log.update()
        log.columns['CoefLift'][-1], log.stage
    """

    def __init__(self, path: str) -> None:
        """Bind to ``path``; nothing is read until :meth:`update`."""
        self.path = path
        self._reset()

    def _reset(self) -> None:
        """Forget everything read so far."""
        self.columns: dict = {'stage': [], 'iteration': []}
        self.n_rows = 0
        self.criteria: dict = {}
        self.initial_residual: dict = {}
        self.events: dict = {}
        self.stages: list = []
        self.stage: Optional[int] = None
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.has_nan = False
        self.mtime: Optional[float] = None
        self.size = 0
        self._offset = 0
        self._monitor: list = []
        self._stage_start: dict = {}
        self._in_integrals = False
        self._in_criteria = False
        self._criterion: Optional[dict] = None
        self._want_names = False

    # ── reading ───────────────────────────────────────────────────────────

    def update(self) -> int:
        """Read what was appended since the last call.

        Returns
        -------
        int
            Number of new iteration lines. 0 also when the file does not
            exist (yet).
        """
        try:
            st = os.stat(self.path)
        except OSError:
            return 0
        if st.st_size < self._offset:
            self._reset()
        self.mtime = st.st_mtime
        self.size = st.st_size
        if st.st_size == self._offset:
            return 0
        with open(self.path, 'rb') as fh:
            fh.seek(self._offset)
            data = fh.read(st.st_size - self._offset)
        end = data.rfind(b'\n') + 1
        if end == 0:
            return 0
        self._offset += end
        before = self.n_rows
        for line in data[:end].decode('utf-8', errors='replace').splitlines():
            self._parse_line(line)
        return self.n_rows - before

    def _parse_line(self, line: str) -> None:
        """Update the state with one complete line."""
        if self._in_integrals:
            m = _RE_INTEGRAL.match(line)
            if m:
                self._set_integral(m.group(1), m.group(2))
                return
            self._in_integrals = False

        if self._in_criteria:
            self._parse_criteria_line(line)
            return

        m = _RE_ITERATION.search(line)
        if m:
            rest = m.group(2)
            event = _RE_EVENT.match(rest)
            if event:
                self._event(int(m.group(1)), event)
            elif rest.startswith('BoundaryIntegralValues'):
                # The integrals belong to the iteration line just read.
                self._in_integrals = (
                    self.n_rows > 0
                    and self.columns['iteration'][-1] == int(m.group(1))
                    and self.columns['stage'][-1] == self._current_stage()
                )
            else:
                self._iteration_line(int(m.group(1)), rest)
            return

        m = _RE_MONITOR.search(line)
        if m:
            self._monitor = m.group(1).split()
            return

        m = _RE_STAGE.match(line)
        if m:
            self.stage = int(m.group(1))
            if self.stage not in self.stages:
                self.stages.append(self.stage)
            return

        if _CRITERIA_START in line:
            self._in_criteria = True
            self._criterion = None
            self._want_names = False
            return

        m = _RE_DATE.match(line)
        if m:
            stamp = _parse_date(m)
            if stamp is not None:
                if self.started is None:
                    self.started = stamp
                else:
                    self.ended = stamp

    def _parse_criteria_line(self, line: str) -> None:
        """One line of the stop-criteria block, a dump with comments."""
        if line.strip() == '],':
            self._in_criteria = False
            return
        m = re.search(r"'type':\s*'([^']+)'", line)
        if m and m.group(1) in _CRITERION_KIND:
            self._criterion = {'kind': _CRITERION_KIND[m.group(1)]}
            stage = self._current_stage()
            self.criteria.setdefault(stage, {})[self._criterion['kind']] = \
                self._criterion
            return
        crit = self._criterion
        if crit is None:
            return
        if self._want_names:
            crit['variables'] = re.findall(r"'([^']+)'", line.split('#')[0])
            self._want_names = False
            return
        if re.search(r"'variables':\s*\[", line):
            self._want_names = True
            return
        for key, rx in _CRITERION_KEYS.items():
            m = rx.search(line)
            if not m:
                continue
            raw = m.group(1)
            if key in ('samples', 'max_iterations'):
                crit[key] = int(raw)
            elif key == 'threshold':
                crit[key] = float(raw)
            elif key == 'thresholds':
                crit[key] = [float(x) for x in raw.split(',') if x.strip()]
            else:
                crit[key] = raw
            return

    def _current_stage(self) -> int:
        """Stage of the lines being read (0 before any ``STAGE`` line)."""
        return 0 if self.stage is None else self.stage

    def _iteration_line(self, iteration: int, rest: str) -> None:
        """``Iteration N: v1 v2 ...`` with the names of ``Monitor variables``."""
        try:
            values = [float(tok) for tok in rest.split()]
        except ValueError:
            return
        names = self._monitor
        if len(names) != len(values):
            return
        stage = self._current_stage()
        if not any(n == 'CFL' or n.endswith('Reduction') for n in names):
            self.initial_residual[stage] = {
                n: _finite(v) for n, v in zip(names, values)
            }
            return
        if any(not math.isfinite(v) for v in values):
            self.has_nan = True
        if stage not in self._stage_start:
            self._stage_start[stage] = self.n_rows
            if stage not in self.stages:
                self.stages.append(stage)
        row = {'stage': stage, 'iteration': iteration}
        row.update((n, _finite(v)) for n, v in zip(names, values))
        n = self.n_rows
        for name in row:
            if name not in self.columns:
                self.columns[name] = [None] * n
        for name, col in self.columns.items():
            col.append(row.get(name))
        self.n_rows += 1

    def _event(self, iteration: int, m) -> None:
        """``Iteration N: <criterion> achieved over <n> samples ...``."""
        kind = _EVENT_KIND[m.group(1)]
        ev = self.events.setdefault(self._current_stage(), {}).setdefault(
            kind, {'first_iteration': iteration, 'last_iteration': iteration,
                   'samples': int(m.group(2)), 'tolerance': None})
        ev['last_iteration'] = iteration
        if m.group(3):
            try:
                ev['tolerance'] = float(m.group(3))
            except ValueError:
                pass

    def _set_integral(self, name: str, raw: str) -> None:
        """``<name> = <value>`` after ``BoundaryIntegralValues``."""
        try:
            value = float(raw)
        except ValueError:
            return
        if not math.isfinite(value):
            self.has_nan = True
        col = self.columns.setdefault(name, [None] * self.n_rows)
        col[-1] = _finite(value)

    # ── queries ───────────────────────────────────────────────────────────

    def stage_span(self, stage: int) -> tuple:
        """``(first, end)`` row indices of ``stage``; ``(0, 0)`` if absent."""
        if stage not in self._stage_start:
            return (0, 0)
        first = self._stage_start[stage]
        later = [s for s in self._stage_start.values() if s > first]
        return (first, min(later) if later else self.n_rows)

    def residual_column(self, variable: str) -> Optional[str]:
        """Column holding the monitored value of a stop-criterion variable.

        The criterion names ``MomentumResidual`` and the log prints
        ``MomentumResidualReduction``.
        """
        for name in (variable + 'Reduction', variable):
            if name in self.columns:
                return name
        return None

    def checks(self, stage: int) -> dict:
        """Where each stop criterion of ``stage`` stands.

        Returns
        -------
        dict
            ``'residuals'``: per variable of the relative-reduction
            criterion, ``{'variable', 'column', 'value', 'threshold',
            'ok'}`` with the last value of the stage. ``'cauchy'``: per
            variable, ``{'variable', 'threshold', 'current', 'delta',
            'min', 'max', 'samples', 'required', 'from_iteration',
            'to_iteration', 'ok'}``, where ``delta = max |x_i - current|``
            over the window of ``required`` previous samples plus the
            current one (CODA's criterion, see the module docstring) and
            ``ok`` needs the full window. ``'max_iterations'``: int or
            None. ``'events'``: the stage's entry of :attr:`events`, with
            ``'now'`` added to each (the message was printed at the last
            iteration read).
        """
        crit = self.criteria.get(stage, {})
        first, end = self.stage_span(stage)
        out = {'residuals': [], 'cauchy': [], 'max_iterations':
               crit.get('iterations', {}).get('max_iterations'), 'events': {}}
        last_it = self.columns['iteration'][end - 1] if end > first else None
        for kind, ev in self.events.get(stage, {}).items():
            out['events'][kind] = dict(ev, now=ev['last_iteration'] == last_it)

        res = crit.get('residual')
        if res:
            thr = res.get('threshold')
            for var in res.get('variables', []):
                col_name = self.residual_column(var)
                value = None
                if col_name is not None:
                    col = self.columns[col_name]
                    value = next((col[i] for i in range(end - 1, first - 1, -1)
                                  if col[i] is not None), None)
                out['residuals'].append({
                    'variable': var, 'column': col_name, 'value': value,
                    'threshold': thr,
                    'ok': value is not None and thr is not None and value < thr,
                })

        cau = crit.get('cauchy')
        if cau:
            required = cau.get('samples') or 0
            lo = max(first, end - 1 - required)
            its = self.columns['iteration']
            for var, thr in zip(cau.get('variables', []),
                                cau.get('thresholds', [])):
                col = self.columns.get(var)
                vals = [v for v in col[lo:end] if v is not None] if col else []
                current = vals[-1] if vals else None
                q = {
                    'variable': var, 'threshold': thr, 'current': current,
                    'delta': (max(abs(v - current) for v in vals)
                              if vals else None),
                    'min': min(vals) if vals else None,
                    'max': max(vals) if vals else None,
                    'samples': max(0, len(vals) - 1), 'required': required,
                    'from_iteration': its[lo] if end > lo else None,
                    'to_iteration': its[end - 1] if end > lo else None,
                }
                q['ok'] = (q['delta'] is not None and q['samples'] >= required
                           and q['delta'] < thr)
                out['cauchy'].append(q)
        return out

    def series(self, since: int = 0) -> dict:
        """Columns from row ``since`` on, ready for JSON.

        Returns
        -------
        dict[str, list]
            ``{name: values[since:]}`` for every column.
        """
        return {name: col[since:] for name, col in self.columns.items()}


class CodaCase:
    """One case folder followed through its newest Slurm log.

    When a newer log appears (the case was restarted), the case switches
    to it and starts reading from scratch; :attr:`log_id` changes so that
    clients can tell.

    Parameters
    ----------
    path : str
        Case folder (``outputs/<folder>``).
    label : str or None
        Name shown for the case. Default: the folder name.
    info : dict or None
        Free JSON-compatible data passed through to :meth:`summary`
        (``case_idx``, design variables...).

    Examples
    --------
    ::

        case = CodaCase('outputs/aoa_4.3363_mach_0.7247')
        case.update()
        case.summary()['status']
    """

    #: Seconds of observations used for the live iteration rate.
    RATE_WINDOW = 600.0

    def __init__(self, path: str, label: Optional[str] = None,
                 info: Optional[dict] = None) -> None:
        """Bind to a case folder; nothing is read until :meth:`update`."""
        self.path = path
        self.label = label or os.path.basename(os.path.normpath(path))
        self.info = dict(info or {})
        self.log: Optional[CodaLog] = None
        self.loaded = False
        self.lock = threading.Lock()
        self._rows_seen: deque = deque()
        self._err_size = 0
        self._err_tail = ''
        self._sbatch: dict = {}
        self._run_sh_mtime: Optional[float] = None
        self._ci = False
        self._restart_prepared = False

    @property
    def log_id(self) -> Optional[str]:
        """File name of the log being followed, or None."""
        return os.path.basename(self.log.path) if self.log else None

    def update(self, now: Optional[float] = None) -> int:
        """Look for a newer log and read what was appended.

        Returns
        -------
        int
            New iteration lines (all of them after a switch of log).
        """
        now = time.time() if now is None else now
        logs = find_logs(self.path)
        newest = logs[-1] if logs else None
        if newest is None:
            self.log = None
        elif self.log is None or self.log.path != newest:
            self.log = CodaLog(newest)
            self._rows_seen.clear()
        new = self.log.update() if self.log else 0

        if self.log is not None:
            err = self.log.path[:-len('-out.txt')] + '-err.txt'
            try:
                size = os.path.getsize(err)
            except OSError:
                size = 0
            if size != self._err_size:
                self._err_size = size
                self._err_tail = ''
                if size:
                    with open(err, 'rb') as fh:
                        fh.seek(max(0, size - 2048))
                        self._err_tail = fh.read().decode('utf-8', 'replace')
            if new or not self._rows_seen:
                self._rows_seen.append((now, self.log.n_rows))
            while (len(self._rows_seen) > 2
                   and now - self._rows_seen[1][0] > self.RATE_WINDOW):
                self._rows_seen.popleft()

        run_sh = os.path.join(self.path, 'run.sh')
        try:
            mtime = os.path.getmtime(run_sh)
        except OSError:
            mtime = None
        if mtime != self._run_sh_mtime:
            self._run_sh_mtime = mtime
            self._sbatch = {}
            if mtime is not None:
                with open(run_sh, errors='replace') as fh:
                    self._sbatch = dict(_RE_SBATCH.findall(fh.read()))
        self._ci = os.path.isfile(os.path.join(self.path, 'ci.h5'))
        # restart_stalled_cases (or the manual procedure) renames the
        # output_<s>.h5 of the stage that stalled to ci.h5: the last log is
        # over and the next job has not written its own yet.
        self._restart_prepared = bool(
            self._ci and self.log is not None and self.log.ended is not None
            and self.log.n_rows and not os.path.isfile(os.path.join(
                self.path, f"output_{self.log.columns['stage'][-1]}.h5"))
        )
        self.loaded = True
        return new

    # ── derived state ─────────────────────────────────────────────────────

    def _status(self, now: float, stale_after: float) -> str:
        """One of :data:`STATUSES`."""
        log = self.log
        if not self.loaded:
            return 'loading'
        if log is None:
            return 'pending'
        if log.ended is not None:
            if self._restart_prepared:
                return 'queued'
            if log.has_nan:
                return 'diverged'
            if self._err_size:
                return 'error'
            if log.n_rows == 0:
                return 'error'
            last_stage = log.columns['stage'][-1]
            limit = log.checks(last_stage)['max_iterations']
            if limit is not None and log.columns['iteration'][-1] >= limit:
                return 'limit'
            return 'finished'
        if 'CANCELLED' in self._err_tail:
            return 'cancelled'
        if log.has_nan:
            return 'diverged'
        if log.mtime is not None and now - log.mtime > stale_after:
            return 'quiet'
        return 'running' if log.n_rows else 'starting'

    def _rate(self) -> tuple:
        """Seconds per iteration and where it comes from.

        ``'live'``: iterations seen arriving during the last
        :attr:`RATE_WINDOW` seconds. ``'average'``: since the first
        ``date`` of the log, startup included.
        """
        seen = self._rows_seen
        if len(seen) >= 2 and seen[-1][1] > seen[0][1]:
            return ((seen[-1][0] - seen[0][0]) / (seen[-1][1] - seen[0][1]),
                    'live')
        log = self.log
        if log and log.started and log.n_rows and log.mtime:
            last = log.ended or log.mtime
            if last > log.started:
                return ((last - log.started) / log.n_rows, 'average')
        return (None, None)

    def summary(self, now: Optional[float] = None,
                stale_after: float = 900.0) -> dict:
        """State of the case, JSON-compatible.

        Parameters
        ----------
        now : float or None
            Epoch seconds. Default: the current time.
        stale_after : float
            Seconds without writes after which a running log counts as
            ``'quiet'``. Default 900.

        Returns
        -------
        dict
            ``label``, ``path``, ``info``, ``log``, ``status`` (one of
            :data:`STATUSES`), ``stage``, ``stages``, ``iteration``,
            ``max_iterations``, ``rows``, ``age`` (seconds since the last
            write), ``elapsed``, ``sec_per_iter`` and ``rate_source``,
            ``eta`` (seconds to the iteration limit, an upper bound),
            ``checks`` (:meth:`CodaLog.checks` per stage, keys as str),
            ``initial_residual``, ``ci``, ``job_id``, ``node``, ``ntasks``,
            ``err_bytes`` and ``has_nan``.
        """
        now = time.time() if now is None else now
        log = self.log
        out = {
            'label': self.label, 'path': self.path, 'info': self.info,
            'log': self.log_id, 'status': self._status(now, stale_after),
            'stage': None, 'stages': [], 'iteration': None,
            'max_iterations': None, 'rows': 0, 'age': None, 'elapsed': None,
            'sec_per_iter': None, 'rate_source': None, 'eta': None,
            'checks': {}, 'initial_residual': {}, 'ci': self._ci,
            'job_id': None, 'node': self._sbatch.get('nodelist'),
            'ntasks': self._sbatch.get('ntasks'), 'err_bytes': self._err_size,
            'has_nan': False,
        }
        if log is None:
            return out
        m = _RE_JOB_ID.search(log.path)
        out['job_id'] = m.group(1) if m else None
        out['has_nan'] = log.has_nan
        out['rows'] = log.n_rows
        out['stages'] = list(log.stages)
        out['stage'] = (log.columns['stage'][-1] if log.n_rows
                        else log.stage)
        out['checks'] = {str(s): log.checks(s) for s in log.stages}
        out['initial_residual'] = {str(s): v for s, v
                                   in log.initial_residual.items()}
        if log.n_rows:
            out['iteration'] = log.columns['iteration'][-1]
        if out['stage'] is not None:
            out['max_iterations'] = log.checks(out['stage'])['max_iterations']
        if log.mtime is not None:
            out['age'] = max(0.0, now - log.mtime)
        if log.started is not None:
            out['elapsed'] = (log.ended or now) - log.started
        rate, source = self._rate()
        out['sec_per_iter'], out['rate_source'] = rate, source
        if (rate and out['status'] in ('running', 'starting')
                and out['max_iterations'] and out['iteration'] is not None):
            out['eta'] = max(0, out['max_iterations'] - out['iteration']) * rate
        return out

    def series(self, since: int = 0) -> dict:
        """Columns of the current log from row ``since`` on (see
        :meth:`CodaLog.series`); empty without a log."""
        return self.log.series(since) if self.log else {}
