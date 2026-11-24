"""
tests/test_horses2plt.py
========================
Tests for the ``horses2plt`` export of HORSES3D solutions:
``FotR/solver_tools/horses2plt.py`` (what to convert, controls, links) and
``Horses3DSets.to_tecplot`` (planning, local runs, Slurm submission).

The solution files are written with ``write_horses_file`` from
``test_horses3d_hsol.py`` (same binary layout as the solver). ``horses2plt``
itself is replaced by a small script that writes the ``.tec`` its control
asks for; the local run goes through the real
``utils_H3d/POST/horses2plt_batch.slurm`` when it exists. Slurm is never
called: ``sbatch`` is monkeypatched.
"""

import os
import stat
import subprocess
import sys

import numpy as np
import pytest

from FotR.solver_tools import horses2plt as h2p
from FotR.characters.readers.horses3d import Horses3DReader
from FotR.characters.sets import horses3d as sets_mod
from FotR.characters.sets.horses3d import Horses3DSets
from test_horses3d_hsol import write_horses_file

BATCH = '/home/m.jaraiz/Documentos/utils_H3d/POST/horses2plt_batch.slurm'
N = 2

CONTROL = """Flow equations = "NS"
mesh file name = "MESH/wt_g2_mesh.h5"
Polynomial order = 2
Number of time steps = 100
Output Interval = 10
Convergence tolerance = 1.d-10
mach number = 0.1
Reynolds number = 1000.0
solution file name = "RESULTS/wt_g2p2.hsol"
save gradients with solution = {grad}
restart = .false.
restart file name = "RESULTS/wt_g2p2.hsol"
"""

FAKE_H2P = """#!/bin/bash
hsol=$(awk -F= 'tolower($1) ~ /^ *hsol file *$/ {gsub(/ /, "", $2); print $2}' "$1")
cp "$1" "${hsol%.*}.tec"
"""


def _q(n_elem=2, n_var=5, value=1.0):
    return [np.full((n_var, N + 1, N + 1, N + 1), value) for _ in range(n_elem)]


@pytest.fixture
def case(tmp_path):
    """outputs/c1 with p=2: final, two autosaves (one being written),
    statistics, a surface file, and the .hmesh of p=2 plus a foreign one."""
    root = tmp_path / "camp"
    case = root / "outputs" / "c1"
    for sub in ("RESULTS/surfaces", "MESH", "LOG"):
        (case / sub).mkdir(parents=True)
    (case / "file_control_p2.control").write_text(CONTROL.format(grad=".false."))
    h5py = pytest.importorskip("h5py")
    with h5py.File(case / "MESH" / "wt_g2_mesh.h5", "w") as fh:
        fh.create_dataset("NodeCoords", data=np.zeros((4, 3)))
        fh.attrs["Ngeo"] = 2
    R = case / "RESULTS"
    write_horses_file(R / "wt_g2p2.hsol", 2, _q(), node_type=1)
    write_horses_file(R / "wt_g2p2_0000000100.hsol", 2, _q(), node_type=1, iteration=100)
    write_horses_file(R / "wt_g2p2_0000000200.hsol", 2, _q(), node_type=1,
                      iteration=200, seal=False)
    write_horses_file(R / "wt_g2p2.stats.0000000150.hsol", 4, _q(n_var=9), node_type=1)
    write_horses_file(R / "wt_g2p2.stats.hsol", 4, _q(n_var=9), node_type=1)
    (R / "surfaces" / "wt_g2p2_bc001_0000000100.surf.hsol").write_bytes(b"x" * 300)
    write_horses_file(case / "MESH" / "wt_g2p2_0000000000.hmesh", 1, _q(n_var=3), node_type=1)
    write_horses_file(case / "MESH" / "other_0000000000.hmesh", 1, _q(n_var=3), node_type=1)
    return str(root), str(case)


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "horses2plt"
    path.write_text(FAKE_H2P)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _sets(root):
    reader = Horses3DReader(root_dir=root, strict=False)
    reader.parse_simulation_dirs()
    return Horses3DSets(reader), reader


# =========================================================================
# 1. Core: which files, which mesh, which control
# =========================================================================

def test_select_files_by_kind_and_completeness(case):
    _, cdir = case
    got = h2p.select_files(cdir, "RESULTS/wt_g2p2.hsol")
    assert [(k, os.path.basename(p), c) for k, p, c in got] == [
        ("final", "wt_g2p2.hsol", True),
        ("stats", "wt_g2p2.stats.0000000150.hsol", True),
        ("stats", "wt_g2p2.stats.hsol", True),
        ("snapshots", "wt_g2p2_0000000100.hsol", True),
        ("snapshots", "wt_g2p2_0000000200.hsol", False),
    ]
    only = h2p.select_files(cdir, "RESULTS/wt_g2p2.hsol", kinds=["snapshots"],
                            iterations=(150, 300))
    assert [os.path.basename(p) for _, p, _ in only] == ["wt_g2p2_0000000200.hsol"]
    with pytest.raises(ValueError, match="kinds"):
        h2p.select_files(cdir, "RESULTS/wt_g2p2.hsol", kinds=["surface"])


def test_mesh_for_takes_the_mesh_of_the_same_run(case):
    _, cdir = case
    assert os.path.basename(h2p.mesh_for(cdir, "wt_g2p2")) == "wt_g2p2_0000000000.hmesh"
    assert h2p.mesh_for(cdir, "wt_g2p3") is None


def test_control_text_gauss_gradients_and_units():
    gauss = h2p.control_text("a.hsol", "a_0.hmesh", "Q,p", node_type=1, order=3)
    assert "output basis = Homogeneous\noutput order = 3\n" in gauss
    assert "output mode = FE" in gauss and "dimensionless" not in gauss
    gl = h2p.control_text("a.stats.hsol", "a_0.hmesh", "Vmean", node_type=2, order=3,
                          dimensionless=True, has_gradients=True)
    assert "output basis" not in gl
    assert gl.endswith("dimensionless = .true.\nhas gradients = .true.\n")
    with pytest.raises(ValueError, match="132"):
        h2p.control_text("x" * 130 + ".hsol", "m.hmesh", "Q", 2, 2)


def test_stage_links_relative_and_idempotent(case, tmp_path):
    _, cdir = case
    target = os.path.join(cdir, "POST", "tec", "p2")
    hsol = os.path.join(cdir, "RESULTS", "wt_g2p2.hsol")
    hmesh = os.path.join(cdir, "MESH", "wt_g2p2_0000000000.hmesh")
    assert h2p.stage(target, hsol, hmesh) == ("wt_g2p2.hsol", "wt_g2p2_0000000000.hmesh")
    assert os.readlink(os.path.join(target, "wt_g2p2.hsol")) == "../../../RESULTS/wt_g2p2.hsol"
    h2p.stage(target, hsol, hmesh)                           # again: no error
    open(os.path.join(target, "x.hsol"), "w").close()
    with pytest.raises(FileExistsError):
        h2p.stage(target, os.path.join(cdir, "RESULTS", "x.hsol"), hmesh)


def test_small_helpers(case, tmp_path):
    _, cdir = case
    assert h2p.default_variables("stats") == "Vmean,Sij,Q"
    assert h2p.count_columns("Q,p,T,Mach") == 8
    assert h2p.count_columns("Vmean,Sij,Q") == 14
    assert h2p.estimate_bytes(10, 2, 8) == 2 * 8 * 270 * 16
    control = os.path.join(cdir, "file_control_p2.control")
    assert h2p.control_flag(control, "save gradients with solution") is False
    assert h2p.control_flag(control, "nothing here") is None
    hsol = os.path.join(cdir, "RESULTS", "wt_g2p2.hsol")
    tec = tmp_path / "x.tec"
    assert not h2p.is_converted(str(tec), hsol)
    tec.write_text("")
    os.utime(tec, (os.path.getmtime(hsol) + 10,) * 2)
    assert h2p.is_converted(str(tec), hsol)


def test_core_needs_the_standard_library_only():
    code = ("import sys, FotR.solver_tools.horses2plt; "
            "print(sorted({'numpy','pandas','h5py','torch'} & set(sys.modules)))")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run([sys.executable, "-c", code], cwd=root,
                         capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "[]"


# =========================================================================
# 2. Horses3DSets.to_tecplot
# =========================================================================

def test_dry_run_plans_without_writing(case, fake_binary):
    root, cdir = case
    sets, _ = _sets(root)
    plan = sets.to_tecplot(fake_binary, BATCH if os.path.isfile(BATCH) else fake_binary)
    assert list(plan["kind"]) == ["final", "stats", "stats", "snapshots", "snapshots"]
    assert list(plan["status"]) == ["planned"] * 4 + ["incomplete"]
    assert plan.loc[0, "tec"] == "POST/tec/p2/wt_g2p2.tec"
    assert plan.loc[0, "hmesh"] == "MESH/wt_g2p2_0000000000.hmesh"
    assert not os.path.exists(os.path.join(cdir, "POST"))


@pytest.mark.skipif(not os.path.isfile(BATCH), reason="utils_H3d batch script not available")
def test_local_run_converts_into_post(case, fake_binary):
    root, cdir = case
    before = sorted(os.listdir(os.path.join(cdir, "RESULTS")))
    sets, reader = _sets(root)
    done = sets.to_tecplot(fake_binary, BATCH, kinds=["final", "stats"],
                           run="local", cpus=2, dry_run=False)
    assert list(done["status"]) == ["converted"] * 3
    tec_dir = os.path.join(cdir, "POST", "tec", "p2")
    assert sorted(f for f in os.listdir(tec_dir) if f.endswith(".tec")) == [
        "wt_g2p2.stats.0000000150.tec", "wt_g2p2.stats.tec", "wt_g2p2.tec"]
    control = open(os.path.join(tec_dir, "wt_g2p2.stats.horses2plt")).read()
    assert "output variables = Vmean,Sij,Q" in control
    assert "output basis = Homogeneous" in control
    assert sorted(os.listdir(os.path.join(cdir, "RESULTS"))) == before

    again = sets.to_tecplot(fake_binary, BATCH, kinds=["final"], run="local",
                            dry_run=False)
    assert list(again["status"]) == ["skipped"]
    reader.parse_simulation_dirs()
    assert reader.sim_metadata["c1"]["solutions"][2]["tec"] == [
        "POST/tec/p2/wt_g2p2.stats.0000000150.tec",
        "POST/tec/p2/wt_g2p2.stats.tec", "POST/tec/p2/wt_g2p2.tec"]


def test_sbatch_submission(case, fake_binary, monkeypatch):
    root, cdir = case
    sets, _ = _sets(root)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 77\n", "")

    monkeypatch.setattr(sets_mod.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setattr(sets_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(sets_mod, "_slurm_cluster", lambda: "nibbler")
    done = sets.to_tecplot(fake_binary, fake_binary, kinds=["final", "stats"],
                           cpus=8, dry_run=False)
    assert list(done["status"]) == ["submitted"] * 3
    assert set(done["job_id"]) == {"77"}
    cmd = calls[0]
    assert cmd[0] == "sbatch" and "--cpus-per-task=3" in cmd
    assert "--partition=gpu" in cmd and "--nodelist=n001" in cmd
    assert any(c.startswith("--mem=") for c in cmd)
    listing = cmd[-1]
    assert listing.startswith(os.path.join(cdir, "POST", "tec", "horses2plt_"))
    lines = open(listing).read().splitlines()
    assert lines[0] == f"{os.path.join(cdir, 'POST', 'tec', 'p2')}\twt_g2p2.horses2plt"


def test_errors(case, fake_binary, tmp_path):
    root, _ = case
    sets, _ = _sets(root)
    with pytest.raises(FileNotFoundError, match="horses2plt"):
        sets.to_tecplot(str(tmp_path / "nope"), fake_binary)
    with pytest.raises(ValueError, match="run"):
        sets.to_tecplot(fake_binary, fake_binary, run="remote")
    with pytest.raises(ValueError, match="kinds"):
        sets.to_tecplot(fake_binary, fake_binary, kinds=["surface"])
    with pytest.raises(ValueError, match="Unknown case"):
        sets.to_tecplot(fake_binary, fake_binary, cases=["zz"])
    with pytest.raises(NotImplementedError, match="to_tecplot"):
        sets.create_jset()


def test_registered_for_horses3d():
    # Read the declaration: other test modules replace SETS_REGISTRY with stubs.
    import FotR.characters.sets as sets_pkg
    with open(sets_pkg.__file__) as fh:
        assert "'HORSES3D': '.horses3d:Horses3DSets'" in fh.read()
