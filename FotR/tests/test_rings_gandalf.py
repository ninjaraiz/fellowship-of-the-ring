"""
tests/test_rings_gandalf.py
===========================
Tests for the GANDALF rings (``BaseRing`` shared logic, ``CODARing``,
``CODASingleRing``) and the ``GANDALF`` facade.

No SLURM is touched: ``assign_jobs``/``submit_cases``/``recover`` are
only exercised through their no-backend guards.
"""

import copy
import json
import pickle
import os

import numpy as np
import pandas as pd
import pytest

from FotR import GANDALF
from FotR.characters.rings import (
    RING_REGISTRY, BaseRing, CODARing, CODASingleRing,
)
from FotR.characters.rings.base import BaseRing as BaseRingMod


def _ring(root, **kwargs):
    return CODARing(root_dir=str(root), eq_type="rans", num_stages=2, **kwargs)


# =========================================================================
# define_cases
# =========================================================================

def test_define_cases_single_element_tuple_is_constant(tmp_path):
    g = _ring(tmp_path)
    g.define_cases(
        method="halton", bounds={"AoA": (0.0, 5.0), "h": (11000,)},
        n_samples=4, seed=0,
    )
    assert list(g.design_vars) == ["AoA", "h"]
    assert (g.df_cases["h"].values == 11000).all()


def test_define_cases_accepts_numpy_scalars(tmp_path):
    g = _ring(tmp_path)
    g.define_cases(
        method="halton",
        bounds={"AoA": (0.0, 5.0), "Mach": np.float64(0.8)},
        n_samples=4, seed=0,
    )
    assert (g.df_cases["Mach"].values == 0.8).all()


def test_define_cases_rejects_bad_bounds(tmp_path):
    g = _ring(tmp_path)
    with pytest.raises(ValueError, match="min < max"):
        g.define_cases(
            method="halton", bounds={"AoA": (5.0, 0.0)}, n_samples=4,
        )
    with pytest.raises(ValueError, match="positive int"):
        g.define_cases(
            method="halton", bounds={"AoA": (0.0, 5.0)}, n_samples=0,
        )


def test_define_cases_peak_validation(tmp_path):
    g = _ring(tmp_path)
    with pytest.raises(ValueError, match="min < max"):
        g.define_cases(
            method="halton", bounds={"AoA": (0.0, 5.0)},
            n_samples=4, peak_ranges={"AoA": (2.0, 2.0)},
        )
    with pytest.raises(ValueError, match="outside"):
        g.define_cases(
            method="halton", bounds={"AoA": (0.0, 5.0)},
            n_samples=4, peak_ranges={"AoA": (4.0, 9.0)},
        )
    with pytest.raises(ValueError, match="not sampled"):
        g.define_cases(
            method="halton", bounds={"AoA": (0.0, 5.0), "h": 11000},
            n_samples=4, peak_ranges={"h": (0.0, 1.0)},
        )


def test_define_cases_external_copies_dataframe(tmp_path):
    g = _ring(tmp_path)
    ext = pd.DataFrame({"AoA": [1.0, 2.0], "Mach": [0.5, 0.6]})
    g.define_cases(method="external", external_dataframe=ext)
    g.add_param(name="extra", data=np.array([7.0, 8.0]))
    assert list(ext.columns) == ["AoA", "Mach"]


def test_define_cases_float64_precision(tmp_path):
    g = _ring(tmp_path)
    g.define_cases(
        method="halton", bounds={"AoA": (0.0, 5.0)}, n_samples=3, seed=1,
    )
    assert g.df_cases["AoA"].dtype == np.float64


# =========================================================================
# add_param / compute_param
# =========================================================================

def test_add_param_guards(tmp_path):
    g = _ring(tmp_path)
    with pytest.raises(RuntimeError, match="define_cases"):
        g.add_param(name="x", data=np.array([1.0]))
    g.define_cases(
        method="halton", bounds={"AoA": (0.0, 5.0)}, n_samples=2, seed=0,
    )
    with pytest.raises(ValueError, match="rows"):
        g.add_param(name="x", data=np.array([1.0, 2.0, 3.0]))


def test_compute_param_formula_and_rejection(tmp_path):
    g = _ring(tmp_path)
    g.define_cases(
        method="halton", bounds={"AoA": (0.0, 4.0)}, n_samples=3, seed=0,
    )
    g.compute_param(name="double", formula="2 * AoA + 1")
    np.testing.assert_allclose(
        g.df_cases["double"].values, 2 * g.df_cases["AoA"].values + 1
    )
    with pytest.raises(ValueError, match="Unknown name"):
        g.compute_param(name="bad", formula="AoA + Nope")
    with pytest.raises(ValueError, match="not allowed"):
        g.compute_param(name="evil", formula="().__class__.__bases__[0]")
    with pytest.raises(ValueError, match="not allowed"):
        g.compute_param(name="evil2", formula="np.__class__")


def test_compute_param_requires_define_first(tmp_path):
    g = _ring(tmp_path)
    with pytest.raises(RuntimeError, match="define_cases"):
        g.compute_param(name="x", formula="1 + 1")


# =========================================================================
# generate_folders: coda (shared mesh)
# =========================================================================

def _sources(tmp_path):
    src = tmp_path / "sources"
    src.mkdir()
    (src / "run.py").write_text(
        "mesh=MESH_PLACEHOLDER\nquoted=\"MESH_PLACEHOLDER\"\n"
        "aoa=AOA_PLACEHOLDER\nchord=CHORD_PLACEHOLDER\n"
    )
    (src / "mesh.msh").write_text("mesh")
    return src


def _cases(g):
    g.define_cases(
        method="external",
        external_dataframe=pd.DataFrame(
            {"AoA": [1.0, 2.0], "Mach": [0.5, 0.6]}
        ),
    )


def test_coda_generate_folders_placeholders(tmp_path):
    src = _sources(tmp_path)
    g = _ring(tmp_path / "db")
    _cases(g)
    g.generate_folders(
        base_files=["run.py"], mesh_path=str(src / "mesh.msh"),
        script_dir=str(src), folder_fmt="aoa_{AoA:.1f}_mach_{Mach:.1f}",
        update_base_files=True,
        data_to_update={
            "AOA_PLACEHOLDER": "AoA",
            "CHORD_PLACEHOLDER": "0.5",
        },
    )
    assert sorted(g.folders_name) == ["aoa_1.0_mach_0.5", "aoa_2.0_mach_0.6"]
    txt = (tmp_path / "db" / "outputs" / "aoa_2.0_mach_0.6" / "run.py").read_text()
    assert 'mesh="mesh.msh"' in txt
    assert txt.count('"mesh.msh"') == 2
    assert "aoa=2.0" in txt and "chord=0.5" in txt
    assert os.path.isfile(
        tmp_path / "db" / "outputs" / "aoa_1.0_mach_0.5" / "mesh.msh"
    )


def test_coda_skip_keeps_folder_for_assign(tmp_path):
    src = _sources(tmp_path)
    g = _ring(tmp_path / "db")
    _cases(g)
    kw = dict(
        base_files=["run.py"], mesh_path=str(src / "mesh.msh"),
        script_dir=str(src), folder_fmt="aoa_{AoA:.1f}_mach_{Mach:.1f}",
    )
    g.generate_folders(**kw)
    g.generate_folders(**kw)
    row = g.df_cases.loc[g.df_cases["folder"] == "aoa_1.0_mach_0.5"].iloc[0]
    assert bool(row["exist"]) is True


def test_coda_missing_mesh_raises(tmp_path):
    src = _sources(tmp_path)
    g = _ring(tmp_path / "db")
    _cases(g)
    with pytest.raises(FileNotFoundError):
        g.generate_folders(
            base_files=["run.py"],
            mesh_path=str(src / "nope.msh"),
            script_dir=str(src),
        )


# =========================================================================
# generate_folders: coda_single (mesh per case)
# =========================================================================

def test_single_per_case_mesh_and_map(tmp_path):
    src = tmp_path / "sources"
    src.mkdir()
    (src / "run.py").write_text("mesh=MESH_PLACEHOLDER\n")
    (src / "m_a.msh").write_text("a")
    (src / "m_b.msh").write_text("b")
    g = CODASingleRing(
        root_dir=str(tmp_path / "db"), eq_type="rans", num_stages=1,
    )
    _cases(g)
    g.generate_folders(
        base_files=["run.py"],
        mesh_paths=[str(src / "m_a.msh"), str(src / "m_b.msh")],
        script_dir=str(src), folder_fmt="aoa_{AoA:.1f}_mach_{Mach:.1f}",
    )
    folders = sorted(g.folders_name)
    first = (tmp_path / "db" / "outputs" / folders[0] / "run.py").read_text()
    second = (tmp_path / "db" / "outputs" / folders[1] / "run.py").read_text()
    assert '"m_a.msh"' in first and '"m_b.msh"' in second

    meta = json.load(
        open(tmp_path / "db" / "metadata" / "cases_metadata.json")
    )
    assert sorted(meta["mesh_map"]) == folders
    assert meta["mesh_map"][folders[0]].endswith("m_a.msh")


def test_single_mesh_validation(tmp_path):
    g = CODASingleRing(
        root_dir=str(tmp_path / "db"), eq_type="rans", num_stages=1,
    )
    _cases(g)
    with pytest.raises(ValueError, match="one mesh per case"):
        g.generate_folders(
            base_files=[], mesh_paths=None, script_dir=str(tmp_path),
        )
    with pytest.raises(ValueError, match="one mesh per case"):
        g.generate_folders(
            base_files=[], mesh_paths=["only_one.msh"],
            script_dir=str(tmp_path),
        )


def test_single_duplicate_basenames_rejected(tmp_path):
    d1 = tmp_path / "s1"
    d2 = tmp_path / "s2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "same.msh").write_text("a")
    (d2 / "same.msh").write_text("b")
    g = CODASingleRing(
        root_dir=str(tmp_path / "db"), eq_type="rans", num_stages=1,
    )
    _cases(g)
    with pytest.raises(ValueError, match="unique"):
        g.generate_folders(
            base_files=[],
            mesh_paths=[str(d1 / "same.msh"), str(d2 / "same.msh")],
            script_dir=str(tmp_path),
        )


# =========================================================================
# Helpers: _job_name, _expand_nodelist
# =========================================================================

def test_job_name_pairs_and_fallback():
    assert BaseRing._job_name("aoa_3.50_mach_0.75") == "a3.50m0.75"
    assert BaseRing._job_name("weird folder!") == "weirdfolder"


def test_expand_nodelist():
    assert BaseRingMod._expand_nodelist("n003") == ["n003"]
    assert BaseRingMod._expand_nodelist("n003,n004") == ["n003", "n004"]
    assert BaseRingMod._expand_nodelist("n[003-005]") == ["n003", "n004", "n005"]
    assert BaseRingMod._expand_nodelist("n[001,003]") == ["n001", "n003"]
    assert BaseRingMod._expand_nodelist("(null)") == []


# =========================================================================
# Facade
# =========================================================================

def test_facade_delegates_and_blocks_backpack(tmp_path):
    g = GANDALF(str(tmp_path / "db"), eq_type="rans", num_stages=2)
    g.define_cases(
        method="halton", bounds={"AoA": (0.0, 5.0)}, n_samples=2, seed=0,
    )
    assert g.design_vars == ["AoA"]
    assert "define_cases" in dir(g)
    assert "Backpack" not in dir(g)
    assert not hasattr(g, "Backpack")
    with pytest.raises(AttributeError):
        GANDALF.Backpack
    with pytest.raises(ValueError, match="not supported"):
        GANDALF(str(tmp_path / "db2"), num_stages=1, ring="nope")


def test_facade_setattr_copy_pickle(tmp_path):
    import copy as _copy

    g = GANDALF(str(tmp_path / "db"), eq_type="rans", num_stages=2)
    g.custom_flag = 123
    assert g._ring.custom_flag == 123
    assert "custom_flag" not in g.__dict__
    g2 = _copy.deepcopy(g)
    g2.custom_flag = 999
    assert g._ring.custom_flag == 123
    data = pickle.dumps(g)
    g3 = pickle.loads(data)
    assert g3.num_stages == 2
    assert str(g3).startswith("GANDALF(ring=")


def test_facade_str_without_ring():
    g = GANDALF.__new__(GANDALF)
    assert str(g) == "GANDALF(no active ring)"


def test_submit_guards_without_slurm(tmp_path):
    g = _ring(tmp_path / "db")
    with pytest.raises(RuntimeError, match="generate_folders"):
        g.submit_cases()
    with pytest.raises(RuntimeError, match="generate_folders"):
        g.assign_jobs(file_sh="run.sh", nodes=["n001"], cpus_per_job=4)


import os


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
