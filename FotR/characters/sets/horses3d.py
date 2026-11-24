"""
sets/horses3d.py
================
Sets class for the HORSES3D format.

For now it carries one operation: exporting solutions to Tecplot with
``horses2plt`` (:meth:`Horses3DSets.to_tecplot`). It works from the case
discovery of ``Horses3DReader`` (``db.sim_metadata``), so it needs neither
``extract_inputs`` nor ``extract_outputs``. Which files, which mesh and
which control each conversion takes is decided by
``FotR/solver_tools/horses2plt.py``; the runs themselves are done by
``utils_H3d/POST/horses2plt_batch.slurm``, one serial ``horses2plt``
process per file.

Layout written per case::

    outputs/<case>/POST/tec/
        horses2plt_<date>.list                 # one 'folder<TAB>control' per line
        p<p>/
            <stem>.hsol -> ../../../RESULTS/…  # links: RESULTS/ is never written
            <stem>_<iter>.hmesh -> ../../../MESH/…
            <stem>.horses2plt                  # control of one conversion
            <stem>.tec                         # written by horses2plt
    outputs/<case>/LOG/horses2plt_<jobid>.log  # Slurm runs
"""

import os
import re
import shutil
import subprocess
import time
from typing import Optional, Union

import pandas as pd

from .base import BaseSets
from ...solver_tools import horses2plt as h2p

_SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
_RESULT_RE = re.compile(r"^(OK|ERROR)\s+(\S+)")
_COLUMNS = ['folder', 'p', 'kind', 'hsol', 'hmesh', 'tec', 'tec_mb', 'status',
            'job_id', 'message']


def _slurm_cluster() -> Optional[str]:
    """``ClusterName`` of this machine's Slurm, from ``slurm.conf``."""
    for conf in ('/etc/slurm/slurm.conf', '/etc/slurm-llnl/slurm.conf'):
        try:
            with open(conf) as fh:
                for line in fh:
                    m = re.match(r"\s*ClusterName\s*=\s*(\S+)", line)
                    if m:
                        return m.group(1)
        except OSError:
            continue
    return None


class Horses3DSets(BaseSets):
    """
    Sets for HORSES3D databases: export of solutions with ``horses2plt``.

    Parameters
    ----------
    db : FRODO
        Parent database opened with ``format='HORSES3D'``.
    """

    def create_jset(self, *args, **kwargs):
        """Not implemented for HORSES3D: which tensor to build from the
        solutions is still to be decided (the priority is looking at the
        data). Use :meth:`to_tecplot` to export them."""
        raise NotImplementedError(
            "HORSES3D has no create_jset yet; export the solutions with "
            "to_tecplot()."
        )

    def to_tecplot(
        self,
        binary: str,
        script: str,
        p: Union[int, list, str] = 'all',
        kinds: Union[str, list, tuple] = h2p.KINDS,
        cases: Optional[list] = None,
        variables: Optional[str] = None,
        stats_variables: Optional[str] = None,
        dimensionless: bool = False,
        iterations: Optional[tuple] = None,
        run: str = 'sbatch',
        node: Optional[str] = None,
        cpus: int = 8,
        mem_gb: Optional[float] = None,
        overwrite: bool = False,
        dry_run: bool = True,
    ) -> pd.DataFrame:
        """
        Convert HORSES3D solutions to Tecplot ``.tec`` files with ``horses2plt``.

        For every selected case and polynomial order, takes the final
        solution, the statistics and the autosaves of that order (only the
        files the solver finished writing), links each one with the
        ``.hmesh`` of the same order into ``POST/tec/p<p>/``, writes the
        control of the conversion there and runs ``horses2plt`` on it, so
        the ``.tec`` lands next to the link and ``RESULTS/`` is not
        touched. Files whose ``.tec`` is newer than the ``.hsol`` are
        skipped unless ``overwrite``.

        Output: one FE zone of bricks, dimensional (SI with the references
        of the file) unless ``dimensionless``. With Gauss nodes the output
        is interpolated to equispaced points of the same order, so the
        sub-cells reach the element faces.

        Parameters
        ----------
        binary : str
            ``horses2plt`` executable (``<CPU repo>/Solver/bin/horses2plt``).
        script : str
            Batch runner, ``utils_H3d/POST/horses2plt_batch.slurm``.
        p : int, list of int or 'all'
            Polynomial orders. Default every order of each case.
        kinds : str or list
            Any of ``'final'``, ``'stats'``, ``'snapshots'``. Default all.
        cases : list, optional
            Case folder names or ``case_idx``. Default every case.
        variables : str, optional
            ``horses2plt`` output variables for solutions (final and
            autosaves), comma separated, exact names of
            ``OutputVariables.f90``. Default ``'Q,p,T,Mach'``.
        stats_variables : str, optional
            Output variables for statistics. Default ``'Vmean,Sij,Q'``
            (mean velocity, Reynolds stresses, mean conservative
            variables). ``Vfrms`` (RMS fluctuations) can be added, but
            ``horses2plt`` writes ``NaN`` where the flow does not fluctuate.
        dimensionless : bool
            Keep the solver's non-dimensional values. Default False.
        iterations : (int, int), optional
            Only autosaves and dated statistics in that iteration range.
        run : {'sbatch', 'local'}
            ``'sbatch'`` submits one Slurm job per case (from a login node
            of nibbler or flexo); ``'local'`` runs the batch script here and
            waits for it.
        node : str, optional
            ``--nodelist`` for the job. Default: ``n001`` on nibbler (with
            ``--partition=gpu``), Slurm's choice elsewhere.
        cpus : int
            Conversions in parallel (``--cpus-per-task``). Default 8.
        mem_gb : float, optional
            Memory of the job. Default: twice the estimated peak of the
            largest file times the conversions in parallel, at least 4 GB.
        overwrite : bool
            Convert again files already converted. Default False.
        dry_run : bool
            Only report what would be done. Default True: nothing is
            written and nothing runs.

        Returns
        -------
        pd.DataFrame
            One row per file: ``folder``, ``p``, ``kind``, ``hsol``,
            ``hmesh``, ``tec`` (paths relative to the case folder),
            ``tec_mb`` (estimated size), ``status`` (``'planned'``,
            ``'skipped'`` already converted, ``'incomplete'`` still being
            written, ``'no-mesh'``, ``'submitted'``, ``'converted'``,
            ``'failed'``), ``job_id`` and ``message``.

        Raises
        ------
        FileNotFoundError
            If ``binary`` or ``script`` do not exist.
        ValueError
            Bad ``run`` or ``kinds``.

        Examples
        --------
        ::

            db = FRODO(root_dir='~/Documentos/COMAC/wind_turbine_marino2024/model_A',
                       format='HORSES3D')
            plan = db.to_tecplot(BIN, SCRIPT, p=[2, 3], kinds=['final', 'stats'])
            done = db.to_tecplot(BIN, SCRIPT, p=[2, 3], kinds=['final', 'stats'],
                                 dry_run=False)
        """
        if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            raise FileNotFoundError(f"horses2plt not found or not executable: {binary}")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"Batch script not found: {script}")
        if run not in ('sbatch', 'local'):
            raise ValueError("run must be 'sbatch' or 'local'.")
        kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
        if set(kinds) - set(h2p.KINDS):
            raise ValueError(f"kinds must be among {h2p.KINDS}.")
        if not dry_run and run == 'sbatch' and shutil.which('sbatch') is None:
            raise RuntimeError(
                "'sbatch' not found on this machine: run from a login node of "
                "nibbler or flexo, or use run='local'."
            )

        rows = []
        stamp = time.strftime('%Y%m%d-%H%M%S')
        for folder in self._select_cases(cases):
            sim = self.db.sim_metadata[folder]
            case_dir = sim['path']
            case_rows, todo = [], []
            for order in self._select_orders(sim['solutions'], p):
                sol = sim['solutions'][order]
                if not sol.get('solution_file'):
                    continue
                stem = os.path.splitext(os.path.basename(sol['solution_file']))[0]
                hmesh = h2p.mesh_for(case_dir, stem)
                save_grad = bool(h2p.control_flag(
                    os.path.join(case_dir, sol['control_file']),
                    'save gradients with solution'))
                target = os.path.join(case_dir, 'POST', 'tec', f'p{order}')
                for kind, path, complete in h2p.select_files(
                        case_dir, sol['solution_file'], kinds, iterations):
                    base = os.path.basename(path)
                    tec = os.path.join(target, os.path.splitext(base)[0] + '.tec')
                    row = {
                        'folder': folder, 'p': order, 'kind': kind,
                        'hsol': os.path.relpath(path, case_dir),
                        'hmesh': os.path.relpath(hmesh, case_dir) if hmesh else None,
                        'tec': os.path.relpath(tec, case_dir), 'tec_mb': None,
                        'status': 'planned', 'job_id': None, 'message': '',
                    }
                    case_rows.append(row)
                    if not complete:
                        row.update(status='incomplete',
                                   message='the solver is still writing it')
                        continue
                    if hmesh is None:
                        row.update(status='no-mesh',
                                   message=f"no MESH/{stem}_<iteration>.hmesh")
                        continue
                    if not overwrite and h2p.is_converted(tec, path):
                        row.update(status='skipped', message='already converted')
                        continue
                    info = h2p.read_header_info(path)
                    var = (stats_variables or h2p.default_variables('stats')) \
                        if kind == 'stats' else \
                        (variables or h2p.default_variables(kind))
                    n_col = h2p.count_columns(var)
                    nodes = info['n_elements'] * (order + 1) ** 3
                    row['tec_mb'] = round(nodes * (3 + n_col) * 19 / 1e6, 1)
                    peak = h2p.estimate_bytes(
                        info['n_elements'], order, n_col,
                        gradients=save_grad or info['file_type'] in (3, 9))
                    control = os.path.splitext(base)[0] + '.horses2plt'
                    text = h2p.control_text(
                        os.path.basename(path), os.path.basename(hmesh), var,
                        info['node_type'], order, dimensionless=dimensionless,
                        has_gradients=kind == 'stats' and save_grad)
                    todo.append((row, target, control, peak, path, hmesh, text))
            rows += case_rows
            if todo and not dry_run:
                for _, target, control, _, path, hmesh, text in todo:
                    h2p.stage(target, path, hmesh)
                    with open(os.path.join(target, control), 'w') as fh:
                        fh.write(text)
                self._launch(folder, case_dir, todo, binary, script, run, node,
                             cpus, mem_gb, stamp)
        return pd.DataFrame(rows, columns=_COLUMNS)

    # ── helpers ───────────────────────────────────────────────────────────

    def _select_cases(self, cases) -> list:
        """Folder names from names or ``case_idx`` (default: all)."""
        sim = self.db.sim_metadata
        if cases is None:
            return list(sim)
        by_idx = {m['case_idx']: name for name, m in sim.items()}
        out = []
        for c in ([cases] if isinstance(cases, (str, int)) else cases):
            name = by_idx.get(c) if isinstance(c, int) else c
            if name not in sim:
                raise ValueError(f"Unknown case {c!r}.")
            out.append(name)
        return out

    @staticmethod
    def _select_orders(solutions: dict, p) -> list:
        """Orders of a case to convert: ``'all'``, one or several."""
        if isinstance(p, str):
            if p != 'all':
                raise ValueError("p must be an int, a list of int or 'all'.")
            return sorted(solutions)
        wanted = [p] if isinstance(p, int) else list(p)
        return [q for q in sorted(solutions) if q in wanted]

    def _launch(self, folder, case_dir, todo, binary, script, run, node,
                cpus, mem_gb, stamp) -> None:
        """Write the list of one case and run or submit it; update the rows."""
        list_path = os.path.join(case_dir, 'POST', 'tec', f'horses2plt_{stamp}.list')
        with open(list_path, 'w') as fh:
            for _, target, control, *_ in todo:
                fh.write(f"{target}\t{control}\n")
        parallel = max(1, min(int(cpus), len(todo)))
        cluster = _slurm_cluster()
        env = dict(os.environ, H2P_CLUSTER=cluster or '', H2P_PARALLEL=str(parallel))

        if run == 'local':
            res = subprocess.run(['bash', script, binary, list_path],
                                 capture_output=True, text=True, env=env)
            outcome = {}
            for line in res.stdout.splitlines():
                m = _RESULT_RE.match(line)
                if m:
                    outcome[m.group(2)] = m.group(1)
            for row, target, control, *_ in todo:
                tec = os.path.join(case_dir, row['tec'])
                if outcome.get(tec) == 'OK':
                    row['status'] = 'converted'
                else:
                    row.update(status='failed', message=(
                        f"see {os.path.join(target, os.path.splitext(control)[0] + '.log')}"))
            return

        peak = max(t[3] for t in todo)
        mem = mem_gb if mem_gb is not None else max(4.0, 2 * peak * parallel / 1e9)
        cmd = ['sbatch', f'--job-name=h2p_{folder}'[:60],
               f"--output={os.path.join(case_dir, 'LOG', 'horses2plt_%j.log')}",
               f'--cpus-per-task={parallel}', f'--mem={int(-(-mem // 1))}G']
        if cluster == 'nibbler':
            cmd += ['--partition=gpu', f'--nodelist={node or "n001"}']
        elif node:
            cmd.append(f'--nodelist={node}')
        cmd += [script, binary, list_path]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        m = _SUBMITTED_RE.search(res.stdout)
        for row, *_ in todo:
            if res.returncode == 0 and m:
                row.update(status='submitted', job_id=m.group(1))
            else:
                row.update(status='failed',
                           message=(res.stdout + res.stderr).strip()[-300:])
