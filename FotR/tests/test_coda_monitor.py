"""
tests/test_coda_monitor.py
==========================
Tests for the live monitor of CODA runs: the log parser
(``FotR.solver_tools.coda_log``), the HTTP server
(``FotR.solver_tools.coda_monitor``) and
``CODAResiduals.monitor_cases``.

The synthetic logs reproduce, line by line, what ``run.sh`` +
``run_sst_v5.py`` + CODA 2024.04 write to ``<job>-<id>-out.txt`` (copied
from rans5 logs, comments shortened). The smoke test on a real rans5 case
is skipped where the dataset does not exist.
"""

import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd
import pytest

from FotR.solver_tools import coda_log, coda_monitor
from FotR.solver_tools.coda_log import CodaCase, CodaLog, find_logs
from FotR.solver_tools.coda_monitor import MonitorServer
from FotR.characters.residuals.coda import CODAResiduals


REAL_CASE = ('/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans5/outputs/'
             'aoa_0.0222_mach_1.0715')

BANNER = """--------------------------------------------------------------------------------
CODA LIBRARIES
--------------------------------------------------------------------------------
 - SPLISS_ROOT                      : /opt/spliss-3.3
--------------------------------------------------------------------------------
"""


def _criteria(res_vars, threshold, max_iterations, cauchy=None):
    """Stop-criteria block as CODA prints it (``cauchy`` = samples)."""
    lines = [
        "[0]\t Using the following stop criteria parameters (con posibilidades):",
        "[0]\t [",
        "    { # Discriminator key: 'type', Description: Callbacks for stopping CODA.",
        "        'type': 'stop relative reduction', # Available options: ['stop relative reduction', 'stop number of iterations', 'stop Cauchy criteria'], Description: The type of callback",
        "        'variables': [ # Description: The list of selected variables. For example, '%{State}Residual' will be expanded into 'D",
        "".join(f"'{v}'" for v in res_vars) + "        ],",
        f"        'threshold': {threshold!r}, # min = 0.0, max = 1.0, Default: 0.0, Description: The reduction threshold.",
        "        'combine with': 'OR', # Available options: ['AND', 'OR'], Default: 'OR', Description: The operation used.",
        "    },",
    ]
    if cauchy:
        lines += [
            "    { # Discriminator key: 'type', Description: Callbacks for stopping CODA.",
            "        'type': 'stop Cauchy criteria', # Available options: ['stop relative reduction', 'stop number of iterations', 'stop Cauchy criteria'], Description: The type of callback",
            "        'variables': [ # Description: A list of variables to consider for the Cauchy convergence.",
            "'CoefLift''CoefDrag''CoefMomentY'        ],",
            "        'thresholds': [0.001, 0.0001, 0.001], # Description: A list of absolute thresholds for the Cauchy convergence.",
            "        'data log category': 'wall_boundary_integrals', # Description: The datalog category.",
            "        'reset sample counter': True, # Default: True, Description: If True, samples will be considered from the first iteration.",
            "        'sample size': { # Discriminator key: 'type', Description: Parameters to determine the number of samples.",
            "            'type': 'fixed', # Available options: ['fixed', 'CFL-weighted'], Default: 'fixed', Description: The method",
            f"            'number of samples': {cauchy}, # Description: The fixed number of samples to consider.",
            "        },",
            "        'combine with': 'AND', # Available options: ['AND', 'OR'], Default: 'OR', Description: The operation used.",
            "    },",
        ]
    lines += [
        "    { # Discriminator key: 'type', Description: Callbacks for stopping CODA.",
        "        'type': 'stop number of iterations', # Available options: ['stop relative reduction', 'stop number of iterations', 'stop Cauchy criteria'], Description: The type of callback",
        f"        'NumberOfIterations': {max_iterations}, # min = 0, Description: The maximum number of iterations.",
        "        'combine with': 'OR', # Available options: ['AND', 'OR'], Default: 'OR', Description: The operation used.",
        "    },",
        "],",
        "",
    ]
    return lines


def _default_res(it):
    return (min(100.0, 0.1 * (it + 1)), 0.9 ** it, 0.8 ** it, 0.85 ** it)


def _default_coefs(it):
    return (0.05 + 1.0 / (it + 1), 0.5 + 0.001 * it, 0.01)


def stage_text(stage, iterations, *, max_iterations=10000, cauchy=None,
               res=_default_res, coefs=_default_coefs, events=None):
    """Lines of one stage: header, criteria, initial residual, iterations.

    ``events``: ``{iteration: [message, ...]}`` printed before that
    iteration's values, as CODA does.
    """
    res_vars = (['DensityResidual', 'MomentumResidual',
                 'EnergyStagnationDensityResidual'] if stage == 0
                else ['MomentumResidual', 'EnergyStagnationDensityResidual'])
    threshold = 1e-4 if stage == 0 else 1e-5
    lines = ["[0]\t ", "", f" STAGE {stage}", "", ""]
    lines += _criteria(res_vars, threshold, max_iterations, cauchy)
    lines += [
        "[0]\t Computing the initial residual...",
        "Info: Using signal number 2 to externally stop iterating.",
        "[0]\t Monitor variables: DensityResidual MomentumResidual EnergyStagnationDensityResidual",
        "[0]\t Iteration 0: 261869 890834 1.12731e+06",
        "[0]\t Computing the initial residual... done",
        "Info: Using signal number 2 to externally stop iterating.",
        "[0]\t Monitor variables: CFL DensityResidualReduction MomentumResidualReduction EnergyStagnationDensityResidualReduction",
    ]
    for it in iterations:
        for msg in (events or {}).get(it, []):
            lines.append(f"[0]\t Iteration {it}: {msg}")
        lines.append("[0]\t Iteration {}: {} {} {} {}".format(it, *res(it)))
        lines.append(f"[0]\t Iteration {it}: BoundaryIntegralValues: wall_boundary_integrals")
        cd, cl, cm = coefs(it)
        lines += [f"[0]\t   CoefDrag = {cd!r}", f"[0]\t   CoefLift = {cl!r}",
                  f"[0]\t   CoefMomentY = {cm!r}"]
    return lines


START = "mar oct 27 10:00:00 CET 2026"
END = "Tue Oct 27 10:00:50 CET 2026"


def log_text(stages, *, start=START, end=None):
    """Full log: banner, date, stages, exports and (optionally) end date."""
    lines = BANNER.splitlines() + [start,
                                   "[0]\t Starting run with CODA version: 2024.04-unknown-commit",
                                   "[0]\t Re = 2628928.5157668204"]
    for st in stages:
        lines += st
    if end:
        lines += ["[0]\t exporting all logging data from category 'wall_boundary_integrals' "
                  "to TECPLOT file 'output_1__monitors_wall_boundary_integrals.dat' ...",
                  end]
    return "\n".join(lines) + "\n"


def two_stages(n0=10, n1=15, max1=10000, cauchy=5, **kw):
    return [stage_text(0, range(n0), max_iterations=5000),
            stage_text(1, range(n1), max_iterations=max1, cauchy=cauchy, **kw)]


RUN_SH = """#!/bin/bash
#SBATCH --job-name=a4.34m0.72           # job name
#SBATCH --output=%x-%j-out.txt      # output file
#SBATCH --nodelist=n002               # node name
#SBATCH --ntasks=40                 # number of MPI processes
date
mpiexec python run_sst_v5.py
date
"""


def make_case(root, name='aoa_4.3363_mach_0.7247', text=None, job=51685,
              err='', run_sh=RUN_SH, files=()):
    case = os.path.join(str(root), name)
    os.makedirs(case, exist_ok=True)
    with open(os.path.join(case, 'run.sh'), 'w') as fh:
        fh.write(run_sh)
    if text is not None:
        with open(os.path.join(case, f'a4.34m0.72-{job}-out.txt'), 'w') as fh:
            fh.write(text)
        with open(os.path.join(case, f'a4.34m0.72-{job}-err.txt'), 'w') as fh:
            fh.write(err)
    for fname in files:
        with open(os.path.join(case, fname), 'w') as fh:
            fh.write('x')
    return case


def read_log(text, tmp_path):
    path = tmp_path / 'a-1-out.txt'
    path.write_text(text)
    log = CodaLog(str(path))
    log.update()
    return log


# =========================================================================
# 1. Log parser
# =========================================================================

def test_full_log_is_parsed(tmp_path):
    log = read_log(log_text(two_stages(), end=END), tmp_path)

    assert log.stages == [0, 1]
    assert log.n_rows == 25
    assert list(log.columns) == [
        'stage', 'iteration', 'CFL', 'DensityResidualReduction',
        'MomentumResidualReduction', 'EnergyStagnationDensityResidualReduction',
        'CoefDrag', 'CoefLift', 'CoefMomentY',
    ]
    assert all(len(c) == 25 for c in log.columns.values())
    assert log.columns['stage'] == [0] * 10 + [1] * 15
    assert log.columns['iteration'] == list(range(10)) + list(range(15))
    assert 'Re' not in log.columns

    assert log.criteria[0]['residual'] == {
        'kind': 'residual', 'threshold': 1e-4, 'combine': 'OR',
        'variables': ['DensityResidual', 'MomentumResidual',
                      'EnergyStagnationDensityResidual']}
    assert 'cauchy' not in log.criteria[0]
    assert log.criteria[0]['iterations']['max_iterations'] == 5000
    cau = log.criteria[1]['cauchy']
    assert cau['variables'] == ['CoefLift', 'CoefDrag', 'CoefMomentY']
    assert cau['thresholds'] == [1e-3, 1e-4, 1e-3]
    assert cau['samples'] == 5
    assert cau['category'] == 'wall_boundary_integrals'
    assert log.criteria[1]['residual']['threshold'] == 1e-5
    assert log.criteria[1]['iterations']['max_iterations'] == 10000

    assert log.initial_residual[0] == {
        'DensityResidual': 261869.0, 'MomentumResidual': 890834.0,
        'EnergyStagnationDensityResidual': 1127310.0}
    assert log.ended - log.started == 50.0


def test_integrals_belong_to_their_iteration(tmp_path):
    coefs = lambda it: (it / 100, it / 10, -float(it))
    log = read_log(log_text([stage_text(0, range(6), coefs=coefs)]), tmp_path)
    assert log.columns['CoefDrag'] == [i / 100 for i in range(6)]
    assert log.columns['CoefLift'] == [i / 10 for i in range(6)]
    assert log.columns['CoefMomentY'] == [-float(i) for i in range(6)]


def test_incremental_reads_match_a_full_read(tmp_path):
    text = log_text(two_stages(n1=40), end=END).encode()
    path = tmp_path / 'a-1-out.txt'
    path.write_bytes(b'')
    log = CodaLog(str(path))
    seen = 0
    for k in range(0, len(text), 997):          # chunks cut lines in half
        with open(path, 'ab') as fh:
            fh.write(text[k:k + 997])
        log.update()
        assert log.n_rows >= seen
        seen = log.n_rows

    (tmp_path / 'full').mkdir()
    full = read_log(text.decode(), tmp_path / 'full')
    assert log.columns == full.columns
    assert log.criteria == full.criteria
    assert (log.started, log.ended) == (full.started, full.ended)


def test_partial_line_waits_for_its_newline(tmp_path):
    text = log_text([stage_text(0, range(3))])
    path = tmp_path / 'a-1-out.txt'
    path.write_text(text + "[0]\t Iteration 3: 0.4 0.7")
    log = CodaLog(str(path))
    log.update()
    assert log.n_rows == 3
    with open(path, 'a') as fh:
        fh.write("29 0.512 0.614125\n[0]\t Iteration 3: BoundaryIntegralValues: x\n"
                 "[0]\t   CoefLift = 0.7\n")
    assert log.update() == 1
    assert log.columns['DensityResidualReduction'][-1] == 0.729
    assert log.columns['MomentumResidualReduction'][-1] == 0.512
    assert log.columns['CoefLift'][-1] == 0.7


def test_a_file_that_shrinks_is_read_again(tmp_path):
    path = tmp_path / 'a-1-out.txt'
    path.write_text(log_text(two_stages()))
    log = CodaLog(str(path))
    log.update()
    assert log.n_rows == 25
    path.write_text(log_text([stage_text(0, range(4))]))
    log.update()
    assert log.n_rows == 4
    assert log.stages == [0]


def test_nan_becomes_none_and_is_flagged(tmp_path):
    res = lambda it: (1.0, float('nan') if it == 2 else 0.5, 0.5, 0.5)
    log = read_log(log_text([stage_text(0, range(4), res=res)]), tmp_path)
    assert log.has_nan
    assert log.columns['DensityResidualReduction'][2] is None
    json.dumps(log.series(), allow_nan=False)


def test_residual_check_reads_the_reduction_columns(tmp_path):
    res = lambda it: (100.0, 1e-3, 5e-6 if it == 9 else 1e-3, 2e-5)
    log = read_log(log_text(two_stages(n1=10, res=res)), tmp_path)
    checks = log.checks(1)['residuals']
    assert [(r['variable'], r['column'], r['ok']) for r in checks] == [
        ('MomentumResidual', 'MomentumResidualReduction', True),
        ('EnergyStagnationDensityResidual',
         'EnergyStagnationDensityResidualReduction', False),
    ]
    assert checks[0]['value'] == 5e-6 and checks[0]['threshold'] == 1e-5


def test_cauchy_check_is_max_deviation_from_the_current_value(tmp_path):
    # CL ends in a window whose max - min (1.1e-3) exceeds the threshold
    # (1e-3) while every value stays within 1e-3 of the current one: CODA
    # counts that as converged, and so must the check.
    cl = [0.9, 0.8, 0.4994, 0.5002, 0.5005, 0.4999, 0.5000]
    coefs = lambda it: (0.05, cl[it], 0.01)
    log = read_log(log_text([stage_text(1, range(7), cauchy=4, coefs=coefs)]),
                   tmp_path)
    q = log.checks(1)['cauchy'][0]
    assert q['variable'] == 'CoefLift'
    assert q['current'] == 0.5 and q['samples'] == 4 and q['required'] == 4
    assert q['delta'] == pytest.approx(6e-4)
    assert q['max'] - q['min'] > 1e-3
    assert q['ok']
    assert (q['from_iteration'], q['to_iteration']) == (2, 6)
    assert all(c['ok'] for c in log.checks(1)['cauchy'])


def test_cauchy_needs_a_full_window(tmp_path):
    log = read_log(log_text([stage_text(1, range(4), cauchy=5,
                                        coefs=lambda it: (0.05, 0.5, 0.01))]),
                   tmp_path)
    q = log.checks(1)['cauchy'][0]
    assert q['delta'] == 0.0 and q['samples'] == 3 and not q['ok']


def test_cauchy_window_does_not_cross_into_the_previous_stage(tmp_path):
    log = read_log(log_text(two_stages(n0=10, n1=3, cauchy=2)), tmp_path)
    q = log.checks(1)['cauchy'][0]
    assert q['from_iteration'] == 0 and q['samples'] == 2


def test_convergence_messages_are_events(tmp_path):
    events = {
        3: ['residual stagnation achieved over 500 samples (tol 0.05, ref stage1InitialResidual).'],
        5: ['Cauchy convergence achieved over 5 samples.'] * 2,
        6: ['Cauchy convergence achieved over 5 samples.'] * 2,
    }
    log = read_log(log_text([stage_text(1, range(7), cauchy=5, events=events)]),
                   tmp_path)
    assert log.n_rows == 7
    ev = log.checks(1)['events']
    assert ev['cauchy'] == {'first_iteration': 5, 'last_iteration': 6,
                            'samples': 5, 'tolerance': None, 'now': True}
    assert ev['stagnation']['tolerance'] == 0.05
    assert ev['stagnation']['now'] is False


def test_find_logs_sorts_by_job_id(tmp_path):
    for name in ('a-99-out.txt', 'a-100-out.txt', 'a-100-err.txt', 'notes.txt'):
        (tmp_path / name).write_text('')
    assert [os.path.basename(p) for p in find_logs(str(tmp_path))] == [
        'a-99-out.txt', 'a-100-out.txt']
    assert find_logs(str(tmp_path / 'missing')) == []


def test_import_needs_the_standard_library_only():
    """The server runs these in a process of its own, without the stack."""
    code = ("import sys, FotR.solver_tools.coda_log, FotR.solver_tools.coda_monitor; "
            "heavy = {'numpy', 'torch', 'pandas', 'pyvista', 'matplotlib', 'h5py', 'scipy'}; "
            "print(sorted(heavy & set(sys.modules)))")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run([sys.executable, '-c', code], cwd=root,
                         capture_output=True, text=True, check=True)
    assert res.stdout.strip() == '[]'


# =========================================================================
# 2. Case status
# =========================================================================

def _status(case, now=None, stale_after=900.0):
    case.update(now=now)
    return case.summary(now=now, stale_after=stale_after)['status']


def test_loading_then_pending_without_log(tmp_path):
    case = CodaCase(make_case(tmp_path))
    assert case.summary()['status'] == 'loading'
    assert _status(case) == 'pending'


def test_starting_running_and_quiet(tmp_path):
    path = make_case(tmp_path, text=log_text([stage_text(0, [])]))
    case = CodaCase(path)
    assert _status(case) == 'starting'

    make_case(tmp_path, text=log_text([stage_text(0, range(3))]))
    case = CodaCase(path)
    mtime = os.path.getmtime(find_logs(path)[-1])
    assert _status(case, now=mtime + 10) == 'running'
    assert case.summary(now=mtime + 2000, stale_after=900)['status'] == 'quiet'


def test_finished_and_limit(tmp_path):
    path = make_case(tmp_path, text=log_text(two_stages(n1=15, max1=10000), end=END),
                     files=['output_1.h5'])
    assert _status(CodaCase(path)) == 'finished'
    path = make_case(tmp_path, 'b', text=log_text(two_stages(n1=15, max1=14), end=END),
                     files=['output_1.h5'])
    assert _status(CodaCase(path)) == 'limit'


def test_diverged_cancelled_and_error(tmp_path):
    res = lambda it: (1.0, float('nan'), 0.5, 0.5)
    path = make_case(tmp_path, 'nan', text=log_text([stage_text(0, range(3), res=res)]))
    assert _status(CodaCase(path)) == 'diverged'

    path = make_case(tmp_path, 'cancel', text=log_text(two_stages()),
                     err='slurmstepd: error: *** JOB 51685 ON n002 CANCELLED AT '
                         '2026-10-27T10:00:00 DUE TO TIME LIMIT ***\n')
    assert _status(CodaCase(path)) == 'cancelled'

    path = make_case(tmp_path, 'err', text=log_text(two_stages(), end=END),
                     err='Traceback (most recent call last):\n')
    assert _status(CodaCase(path)) == 'error'


def test_queued_after_a_prepared_restart(tmp_path):
    # restart_stalled_cases turned output_1.h5 into ci.h5; the new job has
    # not written its log yet, so the newest log is the stalled one.
    text = log_text(two_stages(n1=15, max1=14), end=END)
    path = make_case(tmp_path, text=text, files=['ci.h5', 'output_0.h5'])
    case = CodaCase(path)
    assert _status(case) == 'queued'
    assert case.summary()['ci'] is True

    # The new job starts: its log is followed from scratch.
    with open(os.path.join(path, 'a4.34m0.72-51700-out.txt'), 'w') as fh:
        fh.write(log_text([stage_text(1, range(4), cauchy=5)]))
    case.update()
    s = case.summary()
    assert s['log'] == 'a4.34m0.72-51700-out.txt' and s['job_id'] == '51700'
    assert s['status'] == 'running' and s['rows'] == 4 and s['stages'] == [1]


def test_summary_reads_run_sh_and_is_json_safe(tmp_path):
    path = make_case(tmp_path, text=log_text(two_stages()))
    case = CodaCase(path, info={'case_idx': 3, 'aoa': 4.3})
    case.update()
    s = case.summary()
    assert (s['node'], s['ntasks'], s['job_id']) == ('n002', '40', '51685')
    assert s['label'] == 'aoa_4.3363_mach_0.7247' and s['info']['case_idx'] == 3
    assert (s['stage'], s['iteration'], s['max_iterations']) == (1, 14, 10000)
    assert set(s['checks']) == {'0', '1'}
    json.dumps(s, allow_nan=False)


def test_live_rate_and_eta(tmp_path):
    path = make_case(tmp_path, text=log_text(two_stages(n1=10)))
    case = CodaCase(path)
    case.update(now=1000.0)
    with open(find_logs(path)[-1], 'w') as fh:
        fh.write(log_text(two_stages(n1=15)))
    case.update(now=1010.0)
    s = case.summary(now=1010.0, stale_after=1e12)
    assert s['rate_source'] == 'live'
    assert s['sec_per_iter'] == pytest.approx(2.0)
    assert s['eta'] == pytest.approx((10000 - 14) * 2.0)


def test_average_rate_from_the_date_lines(tmp_path):
    # 25 iterations between 10:00:00 and 10:00:50.
    path = make_case(tmp_path, text=log_text(two_stages(), end=END),
                     files=['output_1.h5'])
    case = CodaCase(path)
    case.update()
    s = case.summary()
    assert s['rate_source'] == 'average'
    assert s['sec_per_iter'] == pytest.approx(2.0)
    assert s['elapsed'] == 50.0 and s['eta'] is None


# =========================================================================
# 3. HTTP server
# =========================================================================

def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


@pytest.fixture
def server(tmp_path):
    running = make_case(tmp_path, 'running', text=log_text(two_stages(n1=5)))
    pending = make_case(tmp_path, 'pending')
    srv = MonitorServer(
        [{'path': running, 'info': {'case_idx': 7}}, {'path': pending}],
        title='test', port=0, interval=0.05, idle_timeout=0,
    ).start()
    deadline = time.time() + 5
    while not all(c.loaded for c in srv.cases) and time.time() < deadline:
        time.sleep(0.02)
    yield srv, running
    srv.stop()


def test_page_config_and_summary(server):
    srv, _ = server
    with urllib.request.urlopen(srv.url, timeout=10) as r:
        page = r.read().decode()
    assert 'plotly-basic' in page and 'api/summary' in page

    cfg = _get(srv.url + 'api/config')
    assert cfg['title'] == 'test' and cfg['interval'] == 0.05
    assert [c['label'] for c in cfg['cases']] == ['running', 'pending']
    assert cfg['cases'][0]['info'] == {'case_idx': 7}

    summary = _get(srv.url + 'api/summary')
    assert [(c['i'], c['status']) for c in summary['cases']] == [
        (0, 'running'), (1, 'pending')]


def test_case_rows_are_sent_once_and_reset_on_a_new_log(server):
    srv, path = server
    first = _get(srv.url + 'api/case?i=0&since=0&log=')
    assert first['reset'] and first['since'] == 0 and first['n'] == 15
    assert len(first['columns']['iteration']) == 15

    again = _get(srv.url + f"api/case?i=0&since=15&log={first['log']}")
    assert not again['reset'] and again['columns']['iteration'] == []

    with open(find_logs(path)[-1], 'w') as fh:
        fh.write(log_text(two_stages(n1=8)))
    time.sleep(0.3)
    more = _get(srv.url + f"api/case?i=0&since=15&log={first['log']}")
    assert not more['reset'] and more['n'] == 18
    assert more['columns']['iteration'] == [5, 6, 7]

    with open(os.path.join(path, 'a4.34m0.72-99999-out.txt'), 'w') as fh:
        fh.write(log_text([stage_text(1, range(2))]))
    time.sleep(0.3)
    new = _get(srv.url + f"api/case?i=0&since=18&log={first['log']}")
    assert new['reset'] and new['log'] == 'a4.34m0.72-99999-out.txt'
    assert new['columns']['iteration'] == [0, 1]


def test_unknown_routes_and_bad_requests(server):
    srv, _ = server
    for path, code in (('nothing', 404), ('api/case?i=5', 404),
                       ('api/case?i=x', 400), ('../etc/passwd', 404)):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(srv.url + path, timeout=10)
        assert err.value.code == code


def test_a_taken_port_moves_to_the_next(tmp_path):
    blocker = socket.socket()
    blocker.bind(('127.0.0.1', 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        srv = MonitorServer([{'path': make_case(tmp_path)}], port=port)
        try:
            assert srv.httpd.server_address[1] != port
        finally:
            srv.stop()
    finally:
        blocker.close()


def test_idle_server_stops_by_itself(tmp_path):
    srv = MonitorServer([{'path': make_case(tmp_path)}], port=0,
                        interval=0.05, idle_timeout=0.2).start()
    deadline = time.time() + 5
    while not srv._stop.is_set() and time.time() < deadline:
        time.sleep(0.05)
    assert srv._stop.is_set()


def test_no_case_is_an_error():
    with pytest.raises(ValueError):
        MonitorServer([], port=0)


def test_launch_runs_the_server_in_its_own_process(tmp_path, monkeypatch):
    monkeypatch.setattr(coda_monitor, '_CACHE_DIR', str(tmp_path / 'cache'))
    path = make_case(tmp_path, text=log_text(two_stages()))
    handle = coda_monitor.launch([{'path': path}], title='proc', port=0)
    try:
        assert handle.alive() and handle.pid != os.getpid()
        assert _get(handle.url + 'api/config')['title'] == 'proc'
        assert os.path.isfile(handle.manifest_path)
    finally:
        handle.stop()
    assert not handle.alive()


# =========================================================================
# 4. CODAResiduals.monitor_cases
# =========================================================================

FOLDERS = ['aoa_1.0000_mach_0.3000', 'aoa_2.0000_mach_0.5000',
           'aoa_3.0000_mach_0.7000']


class _FakeReader:
    def __init__(self, folders):
        self.folders = folders
        self.subsets = {'fast': [2, 0]}

    def case_per_idx(self, idx):
        return self.folders[idx]

    def get_subset(self, name):
        return list(self.subsets[name])


class _FakeDB:
    def __init__(self, root):
        self.root_dir = str(root)
        self.format = 'CODA'
        self.sim_metadata = {}
        for k, f in enumerate(FOLDERS):
            self.sim_metadata[f] = {'path': make_case(root, f),
                                    'aoa': float(k + 1), 'mach': 0.3 + 0.2 * k}
        df = pd.DataFrame({'case_idx': [0, 1, 2, 3],
                           'aoa': [1.0, 2.0, 3.0, 4.0],
                           'mach': [0.3, 0.5, 0.7, 0.9],
                           'folder': FOLDERS + [float('nan')]})
        self.metadata = {'df_cases': df, 'design_vars': ['aoa', 'mach'],
                         'num_stages': 2}
        self.reader = _FakeReader(FOLDERS)


@pytest.fixture
def residuals(tmp_path):
    return CODAResiduals(_FakeDB(tmp_path / 'outputs'))


def test_cases_by_index_folder_and_dataframe(residuals):
    pick = residuals._monitor_folders
    assert pick([2, FOLDERS[0], 2], None, None) == [FOLDERS[2], FOLDERS[0]]
    assert pick(1, None, None) == [FOLDERS[1]]
    assert pick(FOLDERS[1], None, None) == [FOLDERS[1]]
    df = residuals.db.metadata['df_cases']
    assert pick(df[df['mach'] > 0.4].iloc[:2], None, None) == FOLDERS[1:]
    assert pick(df[['case_idx']].iloc[[2]], None, None) == [FOLDERS[2]]
    assert pick('all', None, None) == FOLDERS


def test_cases_by_subset_and_query(residuals):
    assert residuals._monitor_folders(None, 'fast', None) == [FOLDERS[2], FOLDERS[0]]
    assert residuals._monitor_folders(None, None, 'aoa >= 2 and mach < 0.8') == FOLDERS[1:]


def test_case_without_folder_is_skipped_with_a_warning(residuals):
    with pytest.warns(UserWarning, match='without a folder'):
        assert residuals._monitor_folders(None, None, 'aoa > 2') == [FOLDERS[2]]


def test_case_selection_errors(residuals):
    with pytest.raises(ValueError, match='exactly one'):
        residuals._monitor_folders(None, None, None)
    with pytest.raises(ValueError, match='exactly one'):
        residuals._monitor_folders([0], 'fast', None)
    with pytest.raises(ValueError, match='Not in db.sim_metadata'):
        residuals._monitor_folders(['aoa_9.0000_mach_0.9000'], None, None)
    with pytest.raises(ValueError, match='No case'):
        residuals._monitor_folders([], None, None)
    with pytest.raises(ValueError, match="'folder' or a 'case_idx'"):
        residuals._monitor_folders(pd.DataFrame({'aoa': [1.0]}), None, None)


def test_monitor_cases_starts_a_background_server(residuals, tmp_path, monkeypatch):
    monkeypatch.setattr(coda_monitor, '_CACHE_DIR', str(tmp_path / 'cache'))
    handle = residuals.monitor_cases([2, 0], port=0)
    try:
        cfg = _get(handle.url + 'api/config')
    finally:
        handle.stop()
    assert cfg['title'] == 'outputs'
    assert [c['label'] for c in cfg['cases']] == [FOLDERS[2], FOLDERS[0]]
    assert cfg['cases'][0]['info'] == {'case_idx': 2, 'aoa': 3.0, 'mach': 0.7}


# =========================================================================
# 5. Real rans5 case (skipped when absent)
# =========================================================================

def _read_dat(path):
    """TECPLOT monitor written by CODA at the end of a stage."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    names = lines[1].split('=', 1)[1].replace('"', ' ').split()
    rows = [[float(x) for x in ln.split()] for ln in lines[2:] if ln.strip()]
    return {n: [r[k] for r in rows] for k, n in enumerate(names)}


@pytest.mark.skipif(not os.path.isdir(REAL_CASE), reason='rans5 not available')
def test_real_log_agrees_with_the_monitors_coda_writes():
    case = CodaCase(REAL_CASE)
    case.update()
    log = case.log
    s = case.summary()
    assert s['stages'] == [0, 1]
    assert s['status'] in ('finished', 'limit', 'queued')

    stage = log.columns['stage'][-1]
    first, end = log.stage_span(stage)
    dat = _read_dat(os.path.join(REAL_CASE, f'output_{stage}__monitors_TimeIntegration.dat'))
    coefs = _read_dat(os.path.join(REAL_CASE,
                                   f'output_{stage}__monitors_wall_boundary_integrals.dat'))
    assert log.columns['iteration'][first:end] == [int(i) for i in dat['Iteration']]
    r0 = log.initial_residual[stage]['MomentumResidual']
    for k in (0, (end - first) // 2, end - first - 1):
        red = log.columns['MomentumResidualReduction'][first + k]
        assert red == pytest.approx(dat['MomentumResidual'][k] / r0, rel=1e-4)
        # The log prints the integrals with 10 decimals.
        assert log.columns['CoefLift'][first + k] == pytest.approx(coefs['CoefLift'][k], abs=5e-11)

    # A run that stopped before its limit met Cauchy at its last iteration,
    # and CODA said so.
    if s['status'] == 'finished':
        checks = log.checks(stage)
        assert all(q['ok'] for q in checks['cauchy'])
        assert checks['events']['cauchy']['now']
