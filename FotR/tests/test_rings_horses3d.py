"""
tests/test_rings_horses3d.py
============================
Tests for the GANDALF ``horses3d`` ring (``FotR/characters/rings/horses3d.py``)
and its round trip through ``Horses3DReader``.

The mesh is a tiny HOPR-like ``.h5`` with the attributes and datasets the
ring reads (``Ngeo``, ``BCNames``, ``BCType``); the template is a short
control in the style of the real cases. Slurm is never called: ``run.sh``
is a fake script and ``shutil.which`` is monkeypatched.
"""

import json
import os
import stat
import subprocess

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from FotR import GANDALF
from FotR.characters.readers.horses3d import Horses3DReader
from FotR.characters.rings import horses3d as ring_mod
from FotR.characters.rings.horses3d import Horses3DRing, _rewrite_control


TEMPLATE = """Flow equations        = "NS"
mesh file name        = "MESH/old_mesh.h5"
Polynomial order      = 1          ! >= NGeo
Number of time steps  = 50000
Output Interval       = 2000
Convergence tolerance = 1.d-10
mach number           = 0.1
Reynolds number       = 1000.0
solution file name    = "RESULTS/old.hsol"
restart               = .true.
restart file name     = "RESULTS/somewhere.hsol"
use actuatorline      = .true.
actuator epsilon type = 0
stl file name         = "tower"
final time            = 10.0
boundaries to save    = [inlet]
! a comment line

#define IBM
    active = .true.   ! kept inside the block
#end

#define boundary inlet
      type = Inflow
#end

#define boundary outlet
      type = Outflow
#end

#define boundary walls__rotor_cylinder
      type = FreeSlipWall
#end
"""


def make_mesh(path, ngeo=2):
    with h5py.File(path, "w") as fh:
        fh.create_dataset("NodeCoords", data=np.zeros((4, 3)))
        fh.create_dataset("BCNames", data=np.array(
            [b"inlet", b"outlet", b"walls", b"rotor_cylinder"], dtype="S255"))
        fh.create_dataset("BCType", data=np.array(
            [[4, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0], [100, 0, 0, 0]],
            dtype=np.int32))
        fh.attrs["Ngeo"] = ngeo
    return str(path)


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    (d / "ActuatorDef").mkdir(parents=True)
    (d / "ActuatorDef" / "Act_ActuatorDef.dat").write_text("Omega\nOMEGA_PLACEHOLDER\n")
    (d / "template.control").write_text(TEMPLATE)
    run = d / "run.sh"
    run.write_text("#!/bin/bash\necho \"Submitted batch job 4242\"\n")
    run.chmod(run.stat().st_mode | stat.S_IEXEC)
    return {
        "template": str(d / "template.control"),
        "mesh": make_mesh(d / "wt_cgns_g2_mesh.h5"),
        "actuator": str(d / "ActuatorDef"),
        "run": str(run),
    }


def _ring(root):
    ring = Horses3DRing(root_dir=str(root))
    ring.define_cases(method="external",
                      external_dataframe=pd.DataFrame({"tsr": [5.0, 6.0]}))
    ring.compute_param(name="omega", formula="tsr*10")
    return ring


def _generate(ring, src, **kw):
    args = dict(
        control_template=src["template"], mesh_path=src["mesh"], tag="wt",
        folder_fmt="tsr_{tsr:.1f}", p_levels=[2, 3],
        control_values={"mach number": 0.02934, "final time": 20.0,
                        "Reynolds number": "omega"},
        p_values={3: {"final time": 25.0}},
        extra_dirs=[src["actuator"]],
        data_to_update={"OMEGA_PLACEHOLDER": "omega"},
        run_script=src["run"],
    )
    args.update(kw)
    return ring.generate_folders(**args)


def _keys(path):
    return ring_mod._control_keys(open(path).read())


# =========================================================================
# 1. Folders and controls
# =========================================================================

def test_case_folders_and_controls(tmp_path, src):
    ring = _ring(tmp_path / "camp")
    df = _generate(ring, src)
    assert list(df["folder"]) == ["tsr_5.0", "tsr_6.0"]
    case = tmp_path / "camp" / "outputs" / "tsr_6.0"

    assert (case / "MESH" / "wt_cgns_g2_mesh.h5").is_file()
    assert (case / "RESULTS" / "FSP").is_dir() and (case / "LOG").is_dir()
    assert os.readlink(case / "run.sh") == os.path.abspath(src["run"])
    assert (case / "ActuatorDef" / "Act_ActuatorDef.dat").read_text() == "Omega\n60.0\n"

    p2 = _keys(case / "file_control_p2.control")
    assert p2["mesh file name"] == "MESH/wt_cgns_g2_mesh.h5"
    assert p2["polynomial order"] == "2"
    assert p2["solution file name"] == "RESULTS/wt_g2p2.hsol"
    assert p2["restart"] == ".false."
    assert p2["restart file name"] == "RESULTS/wt_g2p2.hsol"   # HORSES3D requires it
    assert "restart polorder" not in p2
    assert p2["mach number"] == "0.02934" and p2["final time"] == "20.0"
    assert p2["reynolds number"] == "60.0"          # column, per case

    p3 = _keys(case / "file_control_p3.control")
    assert p3["restart"] == ".true."
    assert p3["restart file name"] == "RESULTS/wt_g2p2.hsol"
    assert p3["restart polorder"] == "2"
    assert p3["final time"] == "25.0"


def test_controls_lose_comments_but_keep_blocks(tmp_path, src):
    _generate(_ring(tmp_path / "camp"), src)
    text = (tmp_path / "camp" / "outputs" / "tsr_5.0" / "file_control_p2.control").read_text()
    assert "!" not in text
    assert "#define IBM" in text and "#define boundary outlet" in text
    assert text.count("\n\n\n") == 0


def test_fsp_control(tmp_path, src):
    _generate(_ring(tmp_path / "camp"), src, fsp_steps=50)
    case = tmp_path / "camp" / "outputs" / "tsr_5.0"
    text = (case / "fsp_p2.control").read_text()
    keys = ring_mod._control_keys(text)
    assert ("#define boundary inlet__outlet__walls__rotor_cylinder\n"
            "   type = Inflow\n#end") in text
    assert "#define IBM" not in text and "Outflow" not in text
    assert keys["use actuatorline"] == ".false."
    assert not any(k.startswith(("actuator ", "stl file", "boundaries to save"))
                   for k in keys)
    assert keys["restart"] == ".false."
    assert keys["restart file name"] == "RESULTS/FSP/wt_g2p2_fsp.hsol"
    assert "restart polorder" not in keys
    assert keys["number of time steps"] == "50"
    assert keys["solution file name"] == "RESULTS/FSP/wt_g2p2_fsp.hsol"
    assert keys["polynomial order"] == "2"


def test_metadata(tmp_path, src):
    _generate(_ring(tmp_path / "camp"), src)
    meta = json.load(open(tmp_path / "camp" / "metadata" / "cases_metadata.json"))
    assert meta["solver"] == "HORSES3D" and meta["tag"] == "wt"
    assert meta["mesh"]["g"] == 2
    assert meta["mesh"]["boundaries"] == ["inlet", "outlet", "walls"]
    assert meta["mesh"]["inner"] == ["rotor_cylinder"]
    assert meta["p_levels"] == [2, 3] and meta["eq_type"] == "NS"
    assert meta["df_cases"]["folder"] == ["tsr_5.0", "tsr_6.0"]


def test_existing_folders_are_skipped(tmp_path, src):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    df = _generate(ring, src)
    assert list(df["exist"]) == [True, True]


def test_overwrite_rewrites_controls_and_keeps_results(tmp_path, src):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    case = tmp_path / "camp" / "outputs" / "tsr_5.0"
    (case / "LOG" / "out1.log").write_text("old run")
    (case / "RESULTS" / "wt_g2p2_0000000100.hsol").write_text("autosave")
    _generate(ring, src, overwrite=True, control_values={"final time": 99.0})
    assert _keys(case / "file_control_p2.control")["final time"] == "99.0"
    assert (case / "LOG" / "out1.log").read_text() == "old run"
    assert (case / "RESULTS" / "wt_g2p2_0000000100.hsol").is_file()
    assert os.path.islink(case / "run.sh")


def test_inner_zones_left_out_are_added_with_a_message(tmp_path, src, capsys):
    with open(src["template"]) as fh:
        text = fh.read().replace("walls__rotor_cylinder", "walls")
    with open(src["template"], "w") as fh:
        fh.write(text)
    _generate(_ring(tmp_path / "camp"), src)
    out = capsys.readouterr().out
    assert "['rotor_cylinder']" in out and "never" in out
    control = (tmp_path / "camp" / "outputs" / "tsr_5.0" / "file_control_p3.control").read_text()
    assert control.endswith("#define boundary rotor_cylinder\n   type = FreeSlipWall\n#end\n")


def test_inner_zones_declared_in_the_template_are_left_alone(tmp_path, src, capsys):
    _generate(_ring(tmp_path / "camp"), src)
    assert "Inner zone" not in capsys.readouterr().out
    control = (tmp_path / "camp" / "outputs" / "tsr_5.0" / "file_control_p2.control").read_text()
    assert control.count("rotor_cylinder") == 1


def test_a_real_boundary_left_out_is_an_error(tmp_path, src):
    with open(src["template"]) as fh:
        text = fh.read().replace("walls__rotor_cylinder", "rotor_cylinder")
    with open(src["template"], "w") as fh:
        fh.write(text)
    with pytest.raises(ValueError, match="walls"):
        _generate(_ring(tmp_path / "camp"), src)
    assert not (tmp_path / "camp" / "outputs").exists()


def test_monitor_on_an_unknown_zone_is_an_error(tmp_path, src):
    with open(src["template"], "a") as fh:
        fh.write("#define surface monitor 1\n   Marker = nacelle\n#end\n")
    with pytest.raises(ValueError, match="nacelle"):
        _generate(_ring(tmp_path / "camp"), src)
    assert not (tmp_path / "camp" / "outputs").exists()


def test_monitor_marker_is_not_split_on_double_underscore(tmp_path, src):
    # 'inlet__outlet' is fine in '#define boundary' but not as a marker:
    # HORSES3D compares it whole with each zone name (job 52370, SIGSEGV)
    with open(src["template"], "a") as fh:
        fh.write("#define surface monitor 1\n   Marker = inlet__outlet\n#end\n")
    with pytest.raises(ValueError, match="inlet__outlet"):
        _generate(_ring(tmp_path / "camp"), src)


def test_unknown_boundary_to_save_warns(tmp_path, src):
    with open(src["template"]) as fh:
        text = fh.read().replace("boundaries to save    = [inlet]",
                                 "boundaries to save    = [inlet__outlet]")
    with open(src["template"], "w") as fh:
        fh.write(text)
    with pytest.warns(UserWarning, match="inlet__outlet"):
        _generate(_ring(tmp_path / "camp"), src)


def test_keys_inside_a_define_block(tmp_path, src):
    with open(src["template"], "a") as fh:
        fh.write("#define statistics\n   sampling interval = 1\n   starting time = 0.0\n#end\n")
    _generate(_ring(tmp_path / "camp"), src,
              control_values={"#define statistics: sampling interval": 10},
              p_values={3: {"final time": 25.0, "#define statistics: starting time": 30.0,
                            "#define statistics: dump interval": 500}})
    case = tmp_path / "camp" / "outputs" / "tsr_5.0"
    p2 = (case / "file_control_p2.control").read_text()
    p3 = (case / "file_control_p3.control").read_text()
    assert "sampling interval = 10" in p2 and "starting time = 0.0" in p2
    block = p3[p3.index("#define statistics"):]
    block = block[:block.index("#end")]
    assert "starting time = 30.0" in block and "dump interval = 500" in block
    assert "starting time" not in p3[:p3.index("#define statistics")]
    with pytest.raises(ValueError, match="no block"):
        _generate(_ring(tmp_path / "other"), src,
                  control_values={"#define probe 1: name": "x"})


def test_missing_mandatory_key_is_caught_before_writing(tmp_path, src):
    with open(src["template"]) as fh:
        text = fh.read().replace("Convergence tolerance = 1.d-10\n", "")
    with open(src["template"], "w") as fh:
        fh.write(text)
    with pytest.raises(ValueError, match="convergence tolerance"):
        _generate(_ring(tmp_path / "camp"), src)
    assert not (tmp_path / "camp" / "outputs").exists()


@pytest.mark.parametrize("kw,match", [
    ({"tag": "wt_01"}, "Invalid tag"),
    ({"tag": "w t"}, "Invalid tag"),
    ({"p_levels": [3, 2]}, "ascending"),
    ({"p_levels": [1, 2]}, "below the mesh order"),
    ({"control_values": {"Polynomial order": 4}}, "set by the ring"),
    ({"p_values": {3: {"solution file name": '"x.hsol"'}}}, "set by the ring"),
])
def test_generate_errors(tmp_path, src, kw, match):
    with pytest.raises(ValueError, match=match):
        _generate(_ring(tmp_path / "camp"), src, **kw)


@pytest.mark.parametrize("name,ngeo", [("wt_mesh.h5", 2), ("wt_g3_mesh.h5", 2)])
def test_mesh_name_must_carry_ngeo(tmp_path, src, name, ngeo):
    mesh = make_mesh(tmp_path / name, ngeo=ngeo)
    with pytest.raises(ValueError, match="geometric order"):
        _generate(_ring(tmp_path / "camp"), src, mesh_path=mesh)


def test_cases_must_be_defined(tmp_path, src):
    with pytest.raises(RuntimeError, match="define_cases"):
        _generate(Horses3DRing(root_dir=str(tmp_path / "camp")), src)


def test_gandalf_facade_needs_no_stages(tmp_path):
    gdf = GANDALF(root_dir=str(tmp_path / "camp"), ring="horses3d")
    assert str(gdf) == "GANDALF(ring=Horses3DRing)"
    with pytest.raises(NotImplementedError, match="submit_cases"):
        gdf.assign_jobs("run.sh", ["n001"], 4)


def test_rewrite_control_keeps_quoted_bang_and_adds_missing_keys():
    text = 'name = "a!b"   ! comment\n\n#define boundary x\n   name = inner\n#end\n'
    out = _rewrite_control(text, {"name": '"c"', "new key": 3})
    assert out == 'name = "c"\nnew key = 3\n\n#define boundary x\n   name = inner\n#end\n'
    assert _rewrite_control(text, {}) .startswith('name = "a!b"\n')


# =========================================================================
# 2. Round trip through the reader
# =========================================================================

def test_round_trip_pending_then_finished(tmp_path, src):
    _generate(_ring(tmp_path / "camp"), src)
    reader = Horses3DReader(root_dir=str(tmp_path / "camp"), strict=True)
    reader.parse_simulation_dirs()
    st = reader.df_state.set_index("case")
    assert list(st["state"]) == ["pending", "pending"]
    assert list(st["g"]) == [2, 2]
    assert st.loc["tsr_6.0", "p_available"] == [2, 3]
    assert st.loc["tsr_6.0", "tsr"] == 6.0

    sol = tmp_path / "camp" / "outputs" / "tsr_6.0" / "RESULTS" / "wt_g2p2.hsol"
    sol.write_bytes(b"FAKEHSOL" + (99).to_bytes(4, "little"))
    reader.parse_simulation_dirs()
    st = reader.df_state.set_index("case")
    assert st.loc["tsr_6.0", "p_finished"] == [2]
    assert st.loc["tsr_6.0", "state"] == "started"


# =========================================================================
# 3. Launching and FSP report
# =========================================================================

def make_solver_repo(path, branch="main", solvers=("ns",)):
    """Solver repository stub: ``Solver/bin/horses3d.<solver>`` and a branch."""
    exe_dir = path / "Solver" / "bin"
    exe_dir.mkdir(parents=True)
    for solver in solvers:
        exe = exe_dir / f"horses3d.{solver}"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "symbolic-ref", "HEAD",
                    f"refs/heads/{branch}"], check=True)
    return str(path)


@pytest.fixture
def repo(tmp_path):
    return make_solver_repo(tmp_path / "horses3d-gpu")


def test_submit_cases_dry_run_and_skip(tmp_path, src, repo):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    (tmp_path / "camp" / "outputs" / "tsr_5.0" / "RESULTS" / "wt_g2p2.residuals").write_text("")
    plan = ring.submit_cases(2, repo)
    assert list(plan["status"]) == ["skipped", "planned"]
    assert plan.loc[1, "command"] == (
        f"HORSES3D_REPO={repo} SOLVER=ns "
        "./run.sh file_control_p2.control main GPU n001 1 16")
    cpu = ring.submit_cases(2, repo, arch="CPU", node="n002", ncpus=48, fsp=True)
    assert cpu.loc[0, "command"].endswith("fsp_p2.control main CPU n002 48")


def test_submit_cases_in_a_new_session(tmp_path, src, repo):
    _generate(_ring(tmp_path / "camp"), src)
    fresh = Horses3DRing(root_dir=str(tmp_path / "camp"))
    plan = fresh.submit_cases(2, repo, fsp=True)
    assert list(plan["status"]) == ["planned", "planned"]
    with pytest.raises(RuntimeError, match="generate_folders"):
        Horses3DRing(root_dir=str(tmp_path / "empty")).submit_cases(2, repo)


def test_submit_cases_runs_run_sh(tmp_path, src, repo, monkeypatch):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    monkeypatch.setattr(ring_mod.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setenv("HORSES3D_REPO", "/somewhere/else")
    seen = {}
    real_run = ring_mod.subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] == "./run.sh":
            seen.update(kw["env"])
        return real_run(cmd, **kw)

    monkeypatch.setattr(ring_mod.subprocess, "run", fake_run)
    done = ring.submit_cases(2, repo, fsp=True, dry_run=False, cases=["tsr_6.0"])
    assert list(done["status"]) == ["submitted"]
    assert list(done["job_id"]) == ["4242"]
    assert seen["HORSES3D_REPO"] == repo           # the argument, not the environment
    assert seen["SOLVER"] == "ns"


def _fake_sbatch(monkeypatch, fail_on=None):
    """Replace run.sh: numbered job ids, and the DEPENDENCY each call saw."""
    monkeypatch.setattr(ring_mod.shutil, "which", lambda name: "/usr/bin/sbatch")
    calls = []
    real_run = ring_mod.subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "./run.sh":
            return real_run(cmd, **kw)
        calls.append((os.path.basename(kw["cwd"]), cmd[1], kw["env"].get("DEPENDENCY")))
        if cmd[1] == fail_on:
            return subprocess.CompletedProcess(cmd, 1, "ERROR: Restart file not found", "")
        return subprocess.CompletedProcess(cmd, 0, f"Submitted batch job {100 + len(calls)}\n", "")

    monkeypatch.setattr(ring_mod.subprocess, "run", fake_run)
    return calls


def test_submit_cases_chains_the_orders(tmp_path, src, repo, monkeypatch):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    plan = ring.submit_cases([2, 3], repo, ngpu=4)
    assert plan[["folder", "p", "status"]].values.tolist() == [
        ["tsr_5.0", 2, "planned"], ["tsr_5.0", 3, "planned"],
        ["tsr_6.0", 2, "planned"], ["tsr_6.0", 3, "planned"]]
    assert plan.loc[1, "depends_on"] == "<p2>"
    assert plan.loc[1, "command"].startswith(
        f"HORSES3D_REPO={repo} SOLVER=ns DEPENDENCY=afterok:<p2> ./run.sh file_control_p3.control")
    monkeypatch.setenv("DEPENDENCY", "afterok:1")       # never inherited
    calls = _fake_sbatch(monkeypatch)
    done = ring.submit_cases([2, 3], repo, dry_run=False)
    assert done["status"].tolist() == ["submitted"] * 4
    assert calls == [("tsr_5.0", "file_control_p2.control", None),
                     ("tsr_5.0", "file_control_p3.control", "afterok:101"),
                     ("tsr_6.0", "file_control_p2.control", None),
                     ("tsr_6.0", "file_control_p3.control", "afterok:103")]
    assert done["depends_on"].tolist() == [None, "101", None, "103"]


def test_submit_cases_resources_per_order(tmp_path, src, repo):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    plan = ring.submit_cases([2, 3], repo, ngpu={2: 1, 3: 4}, cpus_per_gpu=8, cases=["tsr_5.0"])
    assert plan["command"].str.split("GPU n001 ").str[1].tolist() == ["1 8", "4 8"]
    cpu = ring.submit_cases([2, 3], repo, arch="CPU", node="n[001-002]", ncpus={2: 48, 3: 96},
                            cases=["tsr_5.0"])
    assert cpu["command"].str.split("CPU n[001-002] ", regex=False).str[1].tolist() == ["48", "96"]
    with pytest.raises(ValueError, match=r"ngpu has no value for p = \[3\]"):
        ring.submit_cases([2, 3], repo, ngpu={2: 1})


def test_submit_cases_chain_after_a_started_or_failed_order(tmp_path, src, repo, monkeypatch):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    (tmp_path / "camp" / "outputs" / "tsr_5.0" / "RESULTS" / "wt_g2p2.hsol").write_text("")
    calls = _fake_sbatch(monkeypatch, fail_on="file_control_p2.control")
    done = ring.submit_cases([2, 3], repo, dry_run=False)
    # tsr_5.0: p2 already done, p3 goes without a dependency (needs the restart file)
    # tsr_6.0: p2 fails, so p3 is not submitted
    assert done["status"].tolist() == ["skipped", "submitted", "failed", "skipped"]
    assert calls == [("tsr_5.0", "file_control_p3.control", None),
                     ("tsr_6.0", "file_control_p2.control", None)]
    assert "chain stopped" in done.loc[3, "message"]
    with pytest.raises(ValueError, match="ascending"):
        ring.submit_cases([3, 2], repo)
    with pytest.raises(ValueError, match="FSP"):
        ring.submit_cases([2, 3], repo, fsp=True)


def test_submit_cases_checks_the_repository(tmp_path, src):
    ring = _ring(tmp_path / "camp")
    _generate(ring, src)
    ad = make_solver_repo(tmp_path / "horses3d-gpu-ad", branch="arr/actuator_disk")
    plan = ring.submit_cases(2, ad)
    assert plan.loc[0, "command"].startswith(f"HORSES3D_REPO={ad} SOLVER=ns ")
    assert " arr/actuator_disk GPU " in plan.loc[0, "command"]
    assert ring.submit_cases(2, ad, branch="arr/actuator_disk")["status"].tolist() \
        == ["planned", "planned"]
    with pytest.raises(ValueError, match="arr/actuator_disk"):
        ring.submit_cases(2, ad, branch="main")
    with pytest.raises(FileNotFoundError, match="horses3d.ins"):
        ring.submit_cases(2, ad, solver="ins")
    with pytest.raises(FileNotFoundError, match="no Solver"):
        ring.submit_cases(2, str(tmp_path / "nowhere"))
    loose = tmp_path / "loose"
    (loose / "Solver" / "bin").mkdir(parents=True)
    exe = loose / "Solver" / "bin" / "horses3d.ns"
    exe.write_text("")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(RuntimeError, match="not on a branch"):
        ring.submit_cases(2, str(loose))


def _write_hsol(path, q_elems):
    """HORSES3D solution file (type 2) from per-element ``[var, i, j, k]``."""
    from test_horses3d_hsol import write_horses_file
    write_horses_file(path, 2, q_elems)


def _uniform(n_elem=3, N=2, value=(1.0, 1.0, 0.0, 0.0, 2000.0)):
    return [np.ones((5, N + 1, N + 1, N + 1)) * np.array(value)[:, None, None, None]
            for _ in range(n_elem)]


def test_fsp_report_and_load_campaign(tmp_path, src):
    _generate(_ring(tmp_path / "camp"), src)
    out = tmp_path / "camp" / "outputs"
    fsp = "RESULTS/FSP/wt_g2p2_fsp"
    (out / "tsr_5.0" / f"{fsp}.residuals").write_text(
        "#Residuals file\n"
        "#Iteration Time Total elapsed Time (s) Solver elapsed Time (s) "
        "continuity x-momentum y-momentum z-momentum energy Max-Residual\n"
        "1 0.0 1.0 0.1 1e-9 2e-6 1e-6 1e-6 3e-5 3e-5\n"
        "2 0.1 2.0 0.2 1e-9 2e-6 1e-6 1e-6 1e-4 1e-4\n"
    )
    _write_hsol(out / "tsr_5.0" / f"{fsp}_0000000000.hsol", _uniform())
    moved = _uniform()
    moved[1][1] += 3e-9                      # round-off: preserved
    _write_hsol(out / "tsr_5.0" / f"{fsp}.hsol", moved)

    (out / "tsr_6.0" / f"{fsp}.residuals").write_text(
        "#Iteration Time continuity\n1 0.0 1e-9\n")
    _write_hsol(out / "tsr_6.0" / f"{fsp}_0000000000.hsol", _uniform())
    broken = _uniform()
    broken[2][4] += 50.0                     # energy off by 2.5 %: not preserved
    _write_hsol(out / "tsr_6.0" / f"{fsp}.hsol", broken)

    fresh = Horses3DRing(root_dir=str(tmp_path / "camp"))   # new session
    rep = fresh.fsp_report(2).set_index("folder")
    ok, bad = rep.loc["tsr_5.0"], rep.loc["tsr_6.0"]
    assert ok["status"] == "done" and ok["iterations"] == 2
    assert ok["max_residual"] == pytest.approx(1e-4)   # large, and irrelevant
    assert ok["max_deviation"] == pytest.approx(3e-9)
    assert ok["worst_element"] == 1 and bool(ok["preserved"])
    assert bad["max_deviation"] == pytest.approx(50.0 / 2000.0)
    assert bad["worst_element"] == 2 and not bool(bad["preserved"])
    assert fresh.tag == "wt" and fresh.mesh_g == 2


def test_fsp_report_running_and_pending(tmp_path, src):
    _generate(_ring(tmp_path / "camp"), src)
    out = tmp_path / "camp" / "outputs"
    (out / "tsr_5.0" / "RESULTS/FSP/wt_g2p2_fsp.residuals").write_text(
        "#Iteration Time continuity\n1 0.0 1e-9\n")
    rep = Horses3DRing(root_dir=str(tmp_path / "camp")).fsp_report(2).set_index("folder")
    assert rep.loc["tsr_5.0", "status"] == "running"
    assert rep.loc["tsr_5.0", "preserved"] is None
    assert rep.loc["tsr_6.0", "status"] == "pending"
