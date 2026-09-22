"""
tests/test_coda_single_sets.py
==============================
Tests for ``CODASingleSets`` and ``CODASingleStats``.

The synthetic dataset mirrors what a mesh-refinement study looks like:
cases with **different point counts**, one of them unfinished, so every
test exercises the ragged layout rather than a lucky rectangular one.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

pyvista = pytest.importorskip("pyvista")

from FotR import FRODO


def _write_monitor(path, columns, n_rows=4, start=10.0):
    with open(path, "w") as fh:
        fh.write("# CODA monitor file\n")
        fh.write(" ".join(f'"{c}"' for c in columns) + "\n")
        for i in range(n_rows):
            vals = [start / (i + 1)] * (len(columns) - 1)
            fh.write(" ".join([str(i)] + [str(v) for v in vals]) + "\n")


def _grid(n_cells, value):
    """A row of ``n_cells`` quads carrying a constant-ish Pressure."""
    x = np.arange(n_cells + 1, dtype=float)
    xx, yy = np.meshgrid(x, np.array([0.0, 1.0]))
    points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]
    )
    cells, celltypes = [], []
    for i in range(n_cells):
        cells.extend([4, i, i + 1, i + 1 + n_cells + 1, i + n_cells + 1])
        celltypes.append(pyvista.CellType.QUAD)
    grid = pyvista.UnstructuredGrid(
        np.array(cells), np.array(celltypes), points
    )
    grid.cell_data["CADGroupID"] = np.full(grid.n_cells, 3, dtype=int)
    grid.cell_data["Pressure"] = np.full(grid.n_cells, value, dtype=float)
    grid.cell_data["Density"] = np.full(grid.n_cells, 2.0, dtype=float)
    return grid


@pytest.fixture
def gci_root(tmp_path):
    """Four meshes of decreasing size; the coarsest is still running."""
    root = tmp_path / "gci"
    out = root / "outputs"
    out.mkdir(parents=True)
    (root / "metadata").mkdir()
    mesh_dir = root / "meshes"
    mesh_dir.mkdir()

    # n_cells halves each step -> h doubles -> r = 2 in 1-D refinement.
    spec = {
        "f1":  (32, 1.0),
        "f2":  (16, 1.5),
        "f5":  (8,  2.25),
        "f10": (4,  3.375),
    }
    stages = {"f1": [0, 1], "f2": [0, 1], "f5": [0, 1], "f10": [0]}

    mesh_map = {}
    for folder, (n_cells, value) in spec.items():
        case = out / folder
        case.mkdir()
        grid = _grid(n_cells, value)
        for stage in stages[folder]:
            grid.save(str(case / f"output_{stage}_surface.vtu"))
            cols = ["Time", "DensityResidual"]
            _write_monitor(
                str(case / f"output_{stage}__monitors_TimeIntegration.dat"),
                ["Iteration"] + cols,
            )
            _write_monitor(
                str(case /
                    f"output_{stage}__monitors_stage{stage}InitialResidual.dat"),
                ["Iteration"] + cols,
            )
            with open(
                case / f"output_{stage}__monitors_CFLRamp.dat", "w"
            ) as fh:
                fh.write("# CODA monitor file\n")
                fh.write('"Iteration" "SERReferenceDensityResidual"\n')
                fh.write("0 1.0\n")
        mesh = mesh_dir / f"mesh_{folder}.msh"
        mesh.write_text("")
        mesh_map[folder] = str(mesh)

    cm = {
        "eq_type": "rans",
        "folder_fmt": "f{mesh:g}",
        "design_vars": ["aoa", "mach", "mesh"],
        "num_stages": 2,
        "df_cases": {
            "aoa": [1.0] * 4,
            "mach": [0.8] * 4,
            "mesh": [1.0, 2.0, 5.0, 10.0],
            "folder": list(spec),
        },
        "mesh_map": mesh_map,
    }
    with open(root / "metadata" / "cases_metadata.json", "w") as fh:
        json.dump(cm, fh)
    return str(root)


@pytest.fixture
def db(gci_root):
    database = FRODO(root_dir=gci_root, format="CODA_SINGLE",
                     initial_parse=True)
    database.extract_inputs(id_groups=(3,), cases_idx="all")
    database.extract_outputs(
        stage=1, id_groups=(3,), cases_idx="all",
        var_name_excluded=["CADGroupID"],
    )
    return database


# ── Registration ──────────────────────────────────────────────────────

def test_sets_and_stats_are_registered(db):
    from FotR.characters.sets.coda_single import CODASingleSets
    from FotR.characters.stats.coda_single import CODASingleStats

    assert isinstance(db.sets, CODASingleSets)
    assert isinstance(db.stats, CODASingleStats)


def test_sets_imports_without_pylom(monkeypatch):
    """CODASets vanishes from the registry without pyLOM; this must not.

    ``sets/coda.py`` imports pyLOM at module level, so the registry's
    ``except ModuleNotFoundError`` silently drops CODA. CODASingleSets
    imports it inside ``to_pylom`` precisely to avoid that.
    """
    import importlib
    import FotR.characters.sets.coda_single as mod

    importlib.reload(mod)
    assert mod.CODASingleSets is not None


# ── CaseRef ───────────────────────────────────────────────────────────

def test_case_refs_expose_per_case_identity(db):
    refs = db.sets.case_refs("3")
    assert [r.name for r in refs] == ["f1", "f2", "f5", "f10"]
    assert [r.npts for r in refs] == [32, 16, 8, 4]
    assert [r.mesh_file for r in refs] == [
        "mesh_f1.msh", "mesh_f2.msh", "mesh_f5.msh", "mesh_f10.msh"
    ]
    assert refs[-1].stages == [0]          # unfinished case


def test_case_ref_rereads_when_nothing_is_materialised(db):
    """The lazy path must reproduce the eager arrays exactly."""
    ref = db.sets.case_refs("3")[1]
    eager = np.array(ref.coord(), copy=True)
    eager_cp = np.array(ref.var("Pressure", stage=1), copy=True)

    group = db.data_dict["CADGroup_3"]
    group["Coord"] = []                    # force the re-read path
    group["Vars"]["1"]["Pressure"] = []

    assert np.allclose(ref.coord(), eager)
    assert np.allclose(ref.var("Pressure", stage=1), eager_cp)


def test_case_ref_returns_none_for_a_missing_stage(db):
    ref = db.sets.case_refs("3")[-1]        # f10, no stage 1
    assert ref.var("Pressure", stage=1) is None


# ── M5 · variable algebra ─────────────────────────────────────────────

def test_compute_var_evaluates_per_case(db):
    db.sets.compute_var("p2", "Pressure**2", stage=1, id_group="3")
    stage_vars = db.data_dict["CADGroup_3"]["Vars"]["1"]
    for pos, value in enumerate(stage_vars["p2"]):
        source = stage_vars["Pressure"][pos]
        if source is None:
            assert value is None
        else:
            assert np.allclose(value, source ** 2)


def test_compute_var_can_use_design_variables(db):
    db.sets.compute_var("scaled", "Pressure / mach", stage=1, id_group="3")
    stage_vars = db.data_dict["CADGroup_3"]["Vars"]["1"]
    assert np.allclose(stage_vars["scaled"][0], stage_vars["Pressure"][0] / 0.8)


def test_compute_var_broadcasts_a_scalar_expression(db):
    """A formula of design vars only is still a field over the points."""
    db.sets.compute_var("q", "0.5 * mach**2", stage=1, id_group="3")
    values = db.data_dict["CADGroup_3"]["Vars"]["1"]["q"]
    assert values[0].shape == (32,)
    assert np.allclose(values[0], 0.5 * 0.8 ** 2)


def test_compute_var_refuses_to_overwrite_silently(db):
    db.sets.compute_var("p2", "Pressure**2", stage=1, id_group="3")
    with pytest.raises(ValueError, match="already exists"):
        db.sets.compute_var("p2", "Pressure", stage=1, id_group="3")
    db.sets.compute_var("p2", "Pressure", stage=1, id_group="3",
                        overwrite=True)


def test_compute_var_rejects_arbitrary_code(db):
    with pytest.warns(UserWarning, match="could not evaluate"):
        db.sets.compute_var(
            "bad", "__import__('os').system('true')",
            stage=1, id_group="3",
        )
    assert db.data_dict["CADGroup_3"]["Vars"]["1"]["bad"] == [None] * 4


# ── M1 · reduction across meshes ──────────────────────────────────────

def test_reduce_cases_builds_one_row_per_case(db):
    table = db.sets.reduce_cases("Pressure", stage=1, id_group="3")
    assert table["case"].tolist() == ["f1", "f2", "f5", "f10"]
    assert table["n_points"].tolist() == [32, 16, 8, 4]
    assert np.allclose(table["Pressure"].tolist()[:3], [1.0, 1.5, 2.25])
    # f10 has no stage 1: information, not noise.
    assert np.isnan(table["Pressure"].iloc[-1])
    assert table["mesh"].tolist() == [1.0, 2.0, 5.0, 10.0]


def test_reduce_cases_dropna_removes_unfinished_cases(db):
    table = db.sets.reduce_cases(
        "Pressure", stage=1, id_group="3", dropna=True
    )
    assert table["case"].tolist() == ["f1", "f2", "f5"]


def test_reduce_cases_accepts_a_callable(db):
    table = db.sets.reduce_cases(
        "Pressure", stage=1, id_group="3", how=lambda a: float(a.max() * 2),
    )
    assert np.isclose(table["Pressure"].iloc[0], 2.0)


def test_reduce_cases_rejects_bookkeeping_variables(db):
    with pytest.raises(ValueError, match="bookkeeping"):
        db.sets.reduce_cases("GlobalNumber", stage=1, id_group="3")


# ── M2 · long joint tensor ────────────────────────────────────────────

def test_create_jset_concatenates_blocks_of_different_height(db):
    jset = db.sets.create_jset(
        stage=1, id_group="3", variables=["Pressure"], cases_idx=[0, 1, 2],
    )
    npts = [32, 16, 8]
    assert jset["tensor"].shape[0] == sum(npts)
    assert jset["case_offsets"].tolist() == [0, 32, 48, 56]
    assert jset["case_order"] == ["f1", "f2", "f5"]

    group = db.data_dict["CADGroup_3"]
    for i in range(3):
        a, b = jset["case_offsets"][i], jset["case_offsets"][i + 1]
        block = jset["tensor"][a:b].numpy()
        assert np.allclose(block[:, :3], group["Coord"][i])
        assert np.allclose(block[:, 3:6], group["FlCc"][i])
        assert np.allclose(block[:, -1], group["Vars"]["1"]["Pressure"][i])


def test_create_jset_normalises_over_the_whole_tensor(db):
    jset = db.sets.create_jset(
        stage=1, id_group="3", variables=["Pressure"], cases_idx=[0, 1, 2],
    )
    scaled = jset["scaled"].numpy()
    assert scaled.min() >= -1e-12 and scaled.max() <= 1 + 1e-12
    # The Pressure column spans the three cases, so both ends are reached.
    assert np.isclose(scaled[:, -1].min(), 0.0)
    assert np.isclose(scaled[:, -1].max(), 1.0)


def test_create_jset_drops_variables_missing_in_some_case(db):
    """A variable absent from one case would shift the columns."""
    db.data_dict["CADGroup_3"]["Vars"]["1"]["Density"][1] = None
    with pytest.warns(UserWarning, match="missing in at least one"):
        jset = db.sets.create_jset(stage=1, id_group="3", cases_idx=[0, 1, 2])
    assert jset["info"]["noutputs"] == 1       # only Pressure survives


# ── M4 · round-trip ───────────────────────────────────────────────────

def test_save_to_npy_round_trips_through_the_numpy_reader(db, tmp_path):
    """One top-level key per case is what makes NUMPYReader accept this."""
    path = db.sets.save_to_npy(
        str(tmp_path / "gci"), stage=1, id_group="3", cases_idx=[0, 1, 2],
    )
    raw = np.load(path, allow_pickle=True).item()
    assert sorted(raw) == [
        "CADGroup_3__f1", "CADGroup_3__f2", "CADGroup_3__f5",
    ]

    back = FRODO(root_dir=str(tmp_path), format="NUMPY",
                 file=os.path.basename(path), initial_parse=True)
    group = db.data_dict["CADGroup_3"]
    for pos, case in enumerate(["f1", "f2", "f5"]):
        key = f"CADGroup_3__{case}"
        back.extract_inputs(id_groups=key)
        back.extract_outputs(stage=1, id_groups=key)
        got = back.data_dict[key]
        assert np.allclose(got["Coord"], group["Coord"][pos])
        assert np.allclose(got["FlCc"], group["FlCc"][pos:pos + 1])
        assert np.allclose(
            got["Vars"]["1"]["Pressure"][:, 0],
            group["Vars"]["1"]["Pressure"][pos],
        )


def test_save_to_h5_writes_one_subgroup_per_case(db, tmp_path):
    h5py = pytest.importorskip("h5py")
    path = db.sets.save_to_h5(
        str(tmp_path / "gci"), stage=1, id_group="3", cases_idx=[0, 1],
    )
    with h5py.File(path) as fh:
        cases = fh["CADGroup_3/cases"]
        assert sorted(cases) == ["f1", "f2"]
        assert cases["f1/Coord"].shape == (32, 3)
        assert cases["f1/Vars/1/Pressure"].shape == (32,)
        assert cases["f2/Coord"].shape == (16, 3)


# ── Bridge · sample_on ────────────────────────────────────────────────

def test_sample_on_recovers_the_dense_layout(db):
    dense = db.sets.sample_on(
        "case:f5", stage=1, id_group="3", variables=["Pressure"],
        cases_idx=[0, 1, 2],
    )
    assert dense["Coord"].shape == (8, 3)
    assert dense["Vars"]["1"]["Pressure"].shape == (8, 3)
    assert dense["case_order"] == ["f1", "f2", "f5"]


def test_sample_on_is_the_identity_for_the_reference_case(db):
    dense = db.sets.sample_on(
        "case:f5", stage=1, id_group="3", variables=["Pressure"],
        cases_idx=[0, 1, 2],
    )
    column = dense["case_order"].index("f5")
    original = db.data_dict["CADGroup_3"]["Vars"]["1"]["Pressure"][2]
    assert np.allclose(dense["Vars"]["1"]["Pressure"][:, column], original)


def test_sample_on_can_publish_a_dense_group(db):
    db.sets.sample_on(
        "case:f5", stage=1, id_group="3", variables=["Pressure"],
        cases_idx=[0, 1, 2], new_group_id="3_ref",
    )
    group = db.data_dict["CADGroup_3_ref"]
    assert group["Coord"].shape == (8, 3)
    assert group["Vars"]["1"]["Pressure"].shape == (8, 3)


def test_sample_on_rejects_an_unknown_target(db):
    with pytest.raises(ValueError, match="not in case_order"):
        db.sets.sample_on("case:nope", stage=1, id_group="3")


# ── Inherited, mesh-independent ───────────────────────────────────────

def test_plot_wall_integrals_is_available(db):
    """It reads monitor files, never a mesh, so this format gets it too."""
    assert hasattr(db.sets, "plot_wall_integrals")
    assert callable(db.sets.plot_wall_integrals)


# ── Stats · mesh convergence ──────────────────────────────────────────

def test_compute_stats_tabulates_reductions_per_case(db):
    table = db.stats.compute_stats(
        id_group="3", stage=1, variables=["Pressure"],
        reductions=("mean", "max"),
    )
    assert "Pressure_mean" in table.columns
    assert "Pressure_max" in table.columns
    assert table["case"].tolist() == ["f1", "f2", "f5", "f10"]


@pytest.mark.parametrize("p_true", [1.0, 2.0, 2.5])
def test_richardson_recovers_a_known_order(db, p_true):
    """f(h) = f_exact + C*h**p must give back p and f_exact."""
    f_exact, c = 3.14159, 0.7
    n_points = np.array([10000.0, 2500.0, 625.0])       # r = 2 in 2-D
    h = n_points ** (-1 / 2)
    table = pd.DataFrame({
        "case": ["fine", "medium", "coarse"],
        "n_points": n_points,
        "value": f_exact + c * h ** p_true,
    })
    row = db.stats.richardson(table, "value", dim=2).iloc[0]
    assert row["converged"]
    assert np.isclose(row["p"], p_true, atol=1e-6)
    assert np.isclose(row["f_extrapolated"], f_exact, atol=1e-9)


def test_richardson_reports_nan_when_two_meshes_agree(db):
    """An ill-posed triplet must not yield an authoritative-looking p."""
    table = pd.DataFrame({
        "case": ["a", "b", "c"],
        "n_points": [10000.0, 2500.0, 625.0],
        "value": [1.0, 1.0, 2.0],
    })
    row = db.stats.richardson(table, "value").iloc[0]
    assert np.isnan(row["p"]) and not row["converged"]


def test_richardson_rejects_meshes_that_do_not_refine(db):
    """Equal cell counts give r == 1, which has no observed order."""
    table = pd.DataFrame({
        "case": ["a", "b", "c"],
        "n_points": [1000.0, 1000.0, 250.0],
        "value": [1.0, 1.2, 1.5],
    })
    row = db.stats.richardson(table, "value").iloc[0]
    assert np.isclose(row["r21"], 1.0)
    assert np.isnan(row["p"]) and not row["converged"]


def test_richardson_needs_three_meshes(db):
    table = pd.DataFrame({
        "case": ["a", "b"], "n_points": [1000.0, 250.0], "value": [1.0, 1.2],
    })
    with pytest.raises(ValueError, match="at least 3 meshes"):
        db.stats.richardson(table, "value")


def test_representative_size_matches_the_refinement_ratio(db):
    h = db.stats.representative_size([10000, 2500], dim=2)
    assert np.isclose(h[1] / h[0], 2.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
