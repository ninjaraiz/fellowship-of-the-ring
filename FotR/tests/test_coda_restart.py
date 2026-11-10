"""
tests/test_coda_restart.py
==========================
Tests for ``CODAResiduals.restart_stalled_cases`` and its module-level
helpers.

Each test builds a CODA case folder in ``tmp_path`` with a realistic
``run.sh`` and a minimal run script containing the exact lines the restart
patches, plus ``output_0*`` / ``output_1*`` files. Slurm is never called:
the cluster name, ``sinfo`` and ``squeue`` are monkeypatched.
"""

import os
import subprocess

import pandas as pd
import pytest

from FotR.characters.residuals import coda as coda_mod
from FotR.characters.residuals.coda import CODAResiduals


RUN_SH = """#!/bin/bash
#SBATCH --job-name=a0.33m0.88           # job name
#SBATCH --output=%x-%j-out.txt      # output file
#SBATCH --nodelist=n005               # node name
##SBATCH --partition=gpu           # Partition with the GPU nodes
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks=48                 # number of MPI processes
#SBATCH --cpus-per-task=1           # number of CPUs per task
#SBATCH --tasks-per-node=48         # number of tasks per node
##SBATCH --ntasks=2
mpiexec python run_sst_v5.py
# mpiexec python howToQueryMeshInfo.py<
"""

RUN_PY = """custom_ci = False
ci_name = "ci"
n=2
stages = [f"stage{i}" for i in range(n)]
for cont, stage in enumerate(stages):
    i=int(stage[-1])
    if i==0 and ~custom_ci:
        pass
    isNotFirstStage = stages.index(stage) != 0
    if not isNotFirstStage:
        pass
"""

OUTPUT_1 = [
    "output_1.h5",
    "output_1__monitors_CFLRamp.dat",
    "output_1__monitors_stage1InitialResidual.dat",
    "output_1_surf.dat",
    "output_1_volume.vtu",
]
KEPT = [
    "output_0.h5",
    "output_0__monitors_CFLRamp.dat",
    "output_0_volume.vtu",
    "output_10.h5",            # stage 10 must never be taken for stage 1
    "a0.33m0.88-111-out.txt",
    "a0.33m0.88-111-err.txt",
    "mesh.msh",
]
FOLDER = "aoa_0.3347_mach_0.8863"


def _make_case(root, folder=FOLDER, run_py=RUN_PY, run_sh=RUN_SH):
    case = os.path.join(root, folder)
    os.makedirs(case)
    with open(os.path.join(case, "run.sh"), "w") as fh:
        fh.write(run_sh)
    with open(os.path.join(case, "run_sst_v5.py"), "w") as fh:
        fh.write(run_py)
    for fname in OUTPUT_1 + KEPT:
        with open(os.path.join(case, fname), "w") as fh:
            fh.write(f"content of {fname}")
    return case


class _FakeReader:
    def __init__(self, folders):
        self.folders = folders

    def case_per_idx(self, idx):
        return self.folders[idx]


class _FakeDB:
    def __init__(self, root, folders):
        self.root_dir = root
        self.format = "CODA"
        self.sim_metadata = {
            f: {"path": os.path.join(root, f), "stages": {0: {}, 1: {}}}
            for f in folders
        }
        self.metadata = {"num_stages": 2, "design_vars": ["aoa", "mach"]}
        self.reader = _FakeReader(folders)


def _snapshot(case):
    """{file name: content} of a case folder."""
    out = {}
    for fname in sorted(os.listdir(case)):
        with open(os.path.join(case, fname)) as fh:
            out[fname] = fh.read()
    return out


@pytest.fixture
def on_flexo(monkeypatch):
    """Pretend to run on flexo with nodes n002/n005 and an empty queue."""
    monkeypatch.setattr(coda_mod, "_current_slurm_cluster", lambda: "flexo")
    monkeypatch.setattr(coda_mod, "_slurm_node_cpus",
                        lambda: {"n002": 40, "n005": 48})
    monkeypatch.setattr(coda_mod, "_slurm_active_workdirs", lambda: {})


@pytest.fixture
def case(tmp_path):
    path = _make_case(str(tmp_path))
    return path, CODAResiduals(_FakeDB(str(tmp_path), [FOLDER]))


# =========================================================================
# 1. Dry run and apply
# =========================================================================

def test_dry_run_changes_nothing(case, on_flexo):
    path, res = case
    before = _snapshot(path)

    df = res.restart_stalled_cases(FOLDER, cpus=40, node="n002")

    assert _snapshot(path) == before
    row = df.iloc[0]
    assert row["status"] == "planned"
    changes = "\n".join(row["changes"])
    assert "custom_ci = False -> custom_ci = True" in changes
    assert "--nodelist: n005 -> n002" in changes
    assert "output_1.h5 -> ci.h5" in changes
    assert "output_10.h5" not in changes
    # The free-stream re-initialisation of the run script is left alone.
    assert "isNotFirstStage" not in changes and "~custom_ci" not in changes


def test_apply_restarts_the_case(case, on_flexo):
    path, res = case
    old_h5 = open(os.path.join(path, "output_1.h5")).read()

    df = res.restart_stalled_cases(FOLDER, cpus=40, node="n002",
                                   dry_run=False)

    assert df.iloc[0]["status"] == "restarted"
    files = set(os.listdir(path))
    assert not [f for f in files if coda_mod._output_stage(f) == 1]
    assert set(KEPT) <= files
    assert open(os.path.join(path, "ci.h5")).read() == old_h5

    # Only custom_ci and stages change: the stage restarts from free stream
    # exactly as with the manual procedure.
    py = open(os.path.join(path, "run_sst_v5.py")).read()
    assert py == (RUN_PY
                  .replace("custom_ci = False", "custom_ci = True")
                  .replace('stages = [f"stage{i}" for i in range(n)]',
                           "stages = ['stage1']"))

    sh = open(os.path.join(path, "run.sh")).read()
    assert "#SBATCH --nodelist=n002               # node name" in sh
    assert "#SBATCH --ntasks=40                 # number" in sh
    assert "#SBATCH --tasks-per-node=40         # number" in sh
    assert "##SBATCH --ntasks=2" in sh            # commented line untouched
    assert not [f for f in files if f.endswith(".restart_tmp")]

    assert 1 not in res.db.sim_metadata[FOLDER]["stages"]


def test_second_restart_overwrites_ci(case, on_flexo):
    path, res = case
    res.restart_stalled_cases(FOLDER, cpus=40, node="n002", dry_run=False)
    for fname in OUTPUT_1:
        with open(os.path.join(path, fname), "w") as fh:
            fh.write(f"second run {fname}")

    df = res.restart_stalled_cases(FOLDER, cpus=40, node="n002",
                                   dry_run=False)

    row = df.iloc[0]
    assert row["status"] == "restarted"
    assert any("overwrites" in c for c in row["changes"])
    # Script and run.sh were already in their target form.
    assert not [c for c in row["changes"]
                if c.startswith(("run_sst_v5.py", "run.sh"))]
    assert (open(os.path.join(path, "ci.h5")).read()
            == "second run output_1.h5")


def test_case_index_is_resolved(case, on_flexo):
    _, res = case
    df = res.restart_stalled_cases(0, cpus=40, node="n002")
    assert df.iloc[0]["folder"] == FOLDER


# =========================================================================
# 2. Skipped cases leave everything untouched
# =========================================================================

def _assert_skipped_untouched(path, res, match, **kwargs):
    before = _snapshot(path)
    df = res.restart_stalled_cases(FOLDER, cpus=40, node="n002",
                                   dry_run=False, **kwargs)
    assert df.iloc[0]["status"] == "skipped"
    assert match in df.iloc[0]["reason"]
    assert _snapshot(path) == before


def test_skips_without_stage_output(case, on_flexo):
    path, res = case
    os.remove(os.path.join(path, "output_1.h5"))
    _assert_skipped_untouched(path, res, "output_1.h5 not found")


@pytest.mark.parametrize("run_py", [
    RUN_PY.replace('stages = [f"stage{i}" for i in range(n)]\n', ""),
    RUN_PY.replace('stages = [f"stage{i}" for i in range(n)]\n',
                   'stages = ["stage0"]\nstages = ["stage1"]\n'),
])
def test_skips_on_missing_or_duplicate_stages_line(tmp_path, on_flexo,
                                                  run_py):
    path = _make_case(str(tmp_path), run_py=run_py)
    res = CODAResiduals(_FakeDB(str(tmp_path), [FOLDER]))
    _assert_skipped_untouched(path, res, "stages")


def test_skips_on_missing_nodelist(tmp_path, on_flexo):
    run_sh = RUN_SH.replace("#SBATCH --nodelist=n005", "##SBATCH --nodelist=n005")
    path = _make_case(str(tmp_path), run_sh=run_sh)
    res = CODAResiduals(_FakeDB(str(tmp_path), [FOLDER]))
    _assert_skipped_untouched(path, res, "--nodelist")


def test_skips_case_with_active_job(case, on_flexo, monkeypatch):
    path, res = case
    monkeypatch.setattr(
        coda_mod, "_slurm_active_workdirs",
        lambda: {os.path.realpath(path): [("51581", "RUNNING")]},
    )
    _assert_skipped_untouched(path, res, "51581 (RUNNING)")


# =========================================================================
# 3. Cluster checks
# =========================================================================

def _forbid_slurm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Slurm must not be queried off-cluster")
    monkeypatch.setattr(coda_mod, "_slurm_active_workdirs", boom)
    monkeypatch.setattr(coda_mod, "_slurm_node_cpus", boom)
    monkeypatch.setattr(coda_mod, "_sbatch", boom)


def test_off_cluster_dry_run_warns(case, monkeypatch):
    _, res = case
    monkeypatch.setattr(coda_mod, "_current_slurm_cluster", lambda: "nibbler")
    _forbid_slurm(monkeypatch)

    with pytest.warns(UserWarning, match="NOT checked"):
        df = res.restart_stalled_cases(FOLDER, cpus=40, node="n002")
    assert df.iloc[0]["status"] == "planned"


@pytest.mark.parametrize("submit", [False, True])
def test_off_cluster_apply_raises_and_touches_nothing(case, monkeypatch,
                                                      submit):
    path, res = case
    monkeypatch.setattr(coda_mod, "_current_slurm_cluster", lambda: "nibbler")
    _forbid_slurm(monkeypatch)
    before = _snapshot(path)

    with pytest.raises(RuntimeError, match="'nibbler'.*'flexo'"):
        res.restart_stalled_cases(FOLDER, cpus=40, node="n002",
                                  dry_run=False, submit=submit)
    assert _snapshot(path) == before


def test_squeue_failure_on_cluster_raises(case, monkeypatch):
    path, res = case
    monkeypatch.setattr(coda_mod, "_current_slurm_cluster", lambda: "flexo")
    monkeypatch.setattr(coda_mod, "_slurm_node_cpus", lambda: {"n002": 40})

    def no_squeue(cmd, **kwargs):
        raise FileNotFoundError("squeue")
    monkeypatch.setattr(coda_mod.subprocess, "run", no_squeue)
    before = _snapshot(path)

    with pytest.raises(RuntimeError, match="squeue failed"):
        res.restart_stalled_cases(FOLDER, cpus=40, node="n002",
                                  dry_run=False)
    assert _snapshot(path) == before


@pytest.mark.parametrize("node, cpus, match", [
    ("n0O2", 40, "does not exist"),
    ("n002", 64, "has 40 CPUs"),
])
def test_invalid_node_or_cpus_raise(case, on_flexo, node, cpus, match):
    _, res = case
    with pytest.raises(ValueError, match=match):
        res.restart_stalled_cases(FOLDER, cpus=cpus, node=node)


# =========================================================================
# 4. Submission
# =========================================================================

def test_submit_requires_apply(case, on_flexo):
    _, res = case
    with pytest.raises(ValueError, match="dry_run=False"):
        res.restart_stalled_cases(FOLDER, cpus=40, node="n002", submit=True)


def test_submit_calls_sbatch_in_case_dir(case, on_flexo, monkeypatch):
    path, res = case
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("cwd")))
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 12345\n", stderr="")
    monkeypatch.setattr(coda_mod.subprocess, "run", fake_run)

    df = res.restart_stalled_cases(FOLDER, cpus=40, node="n002",
                                   dry_run=False, submit=True)

    assert calls == [(["sbatch", "run.sh"], path)]
    assert df.iloc[0]["status"] == "submitted"
    assert df.iloc[0]["job_id"] == "12345"


# =========================================================================
# 5. Helpers
# =========================================================================

def test_set_sbatch_option_ignores_comments_and_longer_options():
    text = ("##SBATCH --ntasks=2\n"
            "#SBATCH --ntasks-per-node=8\n"
            "#SBATCH --ntasks=4   # n\n")
    new, change = coda_mod._set_sbatch_option(text, "ntasks", 6)
    assert new == ("##SBATCH --ntasks=2\n"
                   "#SBATCH --ntasks-per-node=8\n"
                   "#SBATCH --ntasks=6   # n\n")
    assert change == "--ntasks: 4 -> 6"


def test_script_from_sh_ignores_commented_lines():
    assert coda_mod._script_from_sh(RUN_SH, "run.sh") == "run_sst_v5.py"
    with pytest.raises(ValueError, match="script=..."):
        coda_mod._script_from_sh("# mpiexec python a.py\n", "run.sh")


def test_output_stage_matches_reader_rule():
    assert coda_mod._output_stage("output_1.h5") == 1
    assert coda_mod._output_stage("output_1__monitors_CFLRamp.dat") == 1
    assert coda_mod._output_stage("output_10.h5") == 10
    assert coda_mod._output_stage("ci.h5") is None


def test_patch_line_ignores_indented_lines():
    text = "custom_ci = False\ndef f():\n    custom_ci = False\n"
    new, change = coda_mod._patch_line(
        text, r"custom_ci\s*=\s*(?:True|False)", "custom_ci = True", "x")
    assert new == "custom_ci = True\ndef f():\n    custom_ci = False\n"
    assert change == "custom_ci = False -> custom_ci = True"


def test_restart_script_text_reads_ci_name():
    text = RUN_PY.replace('ci_name = "ci"', "ci_name = 'restart_state'")
    _, _, ci_name = coda_mod._restart_script_text(text, 1)
    assert ci_name == "restart_state"


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
