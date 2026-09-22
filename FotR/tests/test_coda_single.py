"""
tests/test_coda_single.py
=========================
Tests for ``CODASingleReader`` (CODA cases with one mesh per case).

Synthetic GCI-like dataset: two finished cases with *different* point
counts (the shared-mesh invariant of ``CODAReader`` must NOT apply),
one pending case, and a ``mesh_map`` in ``cases_metadata.json``.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

pyvista = pytest.importorskip("pyvista")

from FotR import FRODO


def _write_monitor(path, columns, n_rows=4, start=10.0):
    header = " ".join(f'"{c}"' for c in columns)
    with open(path, "w") as fh:
        fh.write("# CODA monitor file\n")
        fh.write(header + "\n")
        for i in range(n_rows):
            vals = [start / (i + 1)] * (len(columns) - 1)
            fh.write(" ".join([str(i)] + [str(v) for v in vals]) + "\n")


def _write_case(root, folder, n_cells, with_outputs=True):
    case_path = os.path.join(root, folder)
    os.makedirs(case_path, exist_ok=True)
    # 1-D row of quad cells with n_cells cells -> distinct point counts.
    nx = n_cells
    x = np.arange(nx + 1, dtype=float)
    y = np.array([0.0, 1.0])
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]
    )
    cells = []
    celltypes = []
    for i in range(nx):
        a, b = i, i + 1
        cells.extend([4, a, b, b + nx + 1, a + nx + 1])
        celltypes.append(pyvista.CellType.QUAD)
    grid = pyvista.UnstructuredGrid(
        np.array(cells), np.array(celltypes), points
    )
    npts = grid.n_cells
    grid.cell_data["CADGroupID"] = np.full(npts, 3, dtype=int)
    grid.cell_data["GlobalNumber"] = np.arange(npts)
    grid.cell_data["Pressure"] = np.linspace(0.0, 1.0, npts)
    if with_outputs:
        grid.save(os.path.join(case_path, "output_0_surface.vtu"))
        cols = ["Time", "DensityResidual"]
        _write_monitor(
            os.path.join(
                case_path, "output_0__monitors_TimeIntegration.dat"
            ),
            ["Iteration"] + cols,
        )
        _write_monitor(
            os.path.join(
                case_path,
                "output_0__monitors_stage0InitialResidual.dat",
            ),
            ["Iteration"] + cols,
        )
        with open(
            os.path.join(case_path, "output_0__monitors_CFLRamp.dat"), "w"
        ) as fh:
            fh.write("# CODA monitor file\n")
            fh.write('"Iteration" "SERReferenceDensityResidual"\n')
            fh.write("0 1.0\n")
    mesh_name = f"{folder}.msh"
    open(os.path.join(case_path, mesh_name), "w").close()
    return mesh_name


@pytest.fixture
def gci_root(tmp_path):
    root = tmp_path / "gci"
    out = root / "outputs"
    out.mkdir(parents=True)
    (root / "metadata").mkdir()
    folders = ["f0.8", "f1", "f2"]
    meshes = {}
    cells = {"f0.8": 2, "f1": 4, "f2": 3}
    for folder in folders:
        mesh_name = _write_case(
            str(out), folder, cells[folder],
            with_outputs=(folder != "f0.8"),
        )
        meshes[folder] = str(out / folder / mesh_name)
    cm = {
        "root_dir": str(root),
        "eq_type": "rans",
        "folder_fmt": "f{mesh:g}",
        "design_vars": ["aoa", "mach", "mesh"],
        "num_stages": 1,
        "df_cases": {
            "aoa": [1.0, 1.0, 1.0],
            "mach": [0.8, 0.8, 0.8],
            "mesh": [0.8, 1.0, 2.0],
            "folder": ["f0.8", "f1", "f2"],
        },
        "mesh_map": meshes,
    }
    with open(root / "metadata" / "cases_metadata.json", "w") as fh:
        json.dump(cm, fh)
    return str(root)


def test_parse_joins_by_folder_and_records_mesh(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    assert set(db.sim_metadata) == {"f0.8", "f1", "f2"}
    # Mesh identity lives under 'mesh_info': 'mesh' itself is a design var.
    assert db.sim_metadata["f1"]["mesh_info"]["file"] == "f1.msh"
    assert db.sim_metadata["f1"]["mesh"] == 1.0
    row = db.df_state.loc[db.df_state["folder"] == "f2"].iloc[0]
    assert row["mesh"] == 2.0 and row["stage"] == 1
    row_pending = db.df_state.loc[db.df_state["folder"] == "f0.8"].iloc[0]
    assert row_pending["stage"] == 0
    row_done = db.df_state.loc[db.df_state["folder"] == "f1"].iloc[0]
    assert row_done["stage"] == 1


def test_parse_requires_df_cases(tmp_path):
    root = tmp_path / "empty"
    (root / "outputs" / "f1").mkdir(parents=True)
    from FotR.characters.readers.coda_single import CODASingleReader

    reader = CODASingleReader.__new__(CODASingleReader)
    from FotR.characters.readers.base import BaseReader

    BaseReader.__init__(reader, root_dir=str(root))
    reader.metadata = {}
    reader.output_dir = str(root / "outputs")
    reader.subsets = {}
    with pytest.raises(RuntimeError, match="df_cases"):
        reader.parse_simulation_dirs()


def test_extract_inputs_keeps_per_case_geometry(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1, 2])
    grp = db.data_dict["CADGroup_3"]
    assert grp["case_order"] == ["f1", "f2"]
    assert [c.shape[0] for c in grp["Coord"]] == [4, 3]
    assert grp["FlCc"].tolist() == [[1.0, 0.8, 1.0], [1.0, 0.8, 2.0]]
    assert grp["mesh_files"] == ["f1.msh", "f2.msh"]
    # No shared-mesh invariant: distinct point counts coexist.
    assert len(grp["idx_sort"]) == 2


def test_extract_outputs_vars_are_per_case_lists(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1, 2])
    db.extract_outputs(
        stage=0, id_groups=(3,), cases_idx=[1, 2],
        var_name_excluded=["GlobalNumber", "CADGroupID"],
    )
    var = db.data_dict["CADGroup_3"]["Vars"]["0"]["Pressure"]
    assert [a.shape for a in var] == [(4,), (3,)]


def test_extract_outputs_accepts_a_subset_of_the_extracted_cases(gci_root):
    """A narrower selection is allowed and stays aligned with case_order.

    ``extract_outputs`` used to demand that the selection match
    ``extract_inputs`` exactly. That is unworkable while a study is still
    running (cases get skipped for lack of stages), so a subset is now
    accepted: the stored lists keep one entry per ``case_order`` case and
    the ones left out hold ``None``.
    """
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1, 2])
    db.extract_outputs(
        stage=0, id_groups=(3,), cases_idx=[1],
        var_name_excluded=["GlobalNumber", "CADGroupID"],
    )
    grp = db.data_dict["CADGroup_3"]
    var = grp["Vars"]["0"]["Pressure"]
    assert len(var) == len(grp["case_order"]) == 2
    assert var[0].shape == (4,)     # f1, the one requested
    assert var[1] is None           # f2, left out of this call


def test_extract_outputs_accumulates_across_calls(gci_root):
    """Reading one case at a time must not discard the previous ones."""
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1, 2])
    for case in ([1], [2]):
        db.extract_outputs(
            stage=0, id_groups=(3,), cases_idx=case,
            var_name_excluded=["GlobalNumber", "CADGroupID"],
        )
    var = db.data_dict["CADGroup_3"]["Vars"]["0"]["Pressure"]
    assert [None if a is None else a.shape for a in var] == [(4,), (3,)]


def test_extract_outputs_rejects_a_selection_outside_the_group(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1])
    with pytest.raises(ValueError, match="none of the requested cases"):
        db.extract_outputs(stage=0, id_groups=(3,), cases_idx=[2])


def test_extract_outputs_rejects_an_unavailable_stage(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1, 2])
    with pytest.raises(ValueError, match="stage 7 is not available"):
        db.extract_outputs(stage=7, id_groups=(3,), cases_idx=[1, 2])


def test_residuals_reuse_coda_class(gci_root):
    from FotR.characters.residuals.coda import CODAResiduals

    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    assert isinstance(db.residuals, CODAResiduals)
    df = db.residuals.get_all_final_residuals(
        stage=[0], load_in_metadata=False
    )
    assert len(df) == 2


def test_subsets_over_mesh_cases(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.define_subset(name="fine", cases_idx=[1, 2])
    db.extract_inputs(id_groups=(3,), subset="fine")
    assert db.data_dict["CADGroup_3"]["case_order"] == ["f1", "f2"]


# =========================================================================
# Regression fixtures/tests for the CODA_SINGLE audit
#
# The fixture above deliberately keeps the historical layout (mesh inside
# the case folder, one stage, folder names that sort the same numerically
# and lexicographically). The ones below reproduce what CODASingleRing
# actually writes, which is where the bugs lived.
# =========================================================================


def _write_stage(case_path, stage, grid):
    """Write one stage's surface .vtu plus its residual monitors."""
    grid.save(os.path.join(case_path, f"output_{stage}_surface.vtu"))
    cols = ["Time", "DensityResidual"]
    _write_monitor(
        os.path.join(
            case_path, f"output_{stage}__monitors_TimeIntegration.dat"
        ),
        ["Iteration"] + cols,
    )
    _write_monitor(
        os.path.join(
            case_path,
            f"output_{stage}__monitors_stage{stage}InitialResidual.dat",
        ),
        ["Iteration"] + cols,
    )
    with open(
        os.path.join(case_path, f"output_{stage}__monitors_CFLRamp.dat"), "w"
    ) as fh:
        fh.write("# CODA monitor file\n")
        fh.write('"Iteration" "SERReferenceDensityResidual"\n')
        fh.write("0 1.0\n")


def _quad_row_grid(n_cells):
    """A 1-D row of *n_cells* quads, tagged as CADGroupID 3."""
    x = np.arange(n_cells + 1, dtype=float)
    y = np.array([0.0, 1.0])
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]
    )
    cells, celltypes = [], []
    for i in range(n_cells):
        a, b = i, i + 1
        cells.extend([4, a, b, b + n_cells + 1, a + n_cells + 1])
        celltypes.append(pyvista.CellType.QUAD)
    grid = pyvista.UnstructuredGrid(
        np.array(cells), np.array(celltypes), points
    )
    grid.cell_data["CADGroupID"] = np.full(grid.n_cells, 3, dtype=int)
    grid.cell_data["Pressure"] = np.linspace(0.0, 1.0, grid.n_cells)
    return grid


@pytest.fixture
def gci_running(tmp_path):
    """GCI-like root as CODASingleRing writes it, with a study in progress.

    * meshes live in a shared ``meshes/`` directory, **outside** the case
      folders, and ``mesh_map`` points there (so the fallback that scans
      the case folder cannot silently stand in for it);
    * ``f10`` sorts before ``f2`` lexicographically but after it
      numerically, which is what desynchronised ``df_state``;
    * ``num_stages=2`` and ``f10`` has only stage 0 written.
    """
    root = tmp_path / "gci_running"
    out = root / "outputs"
    out.mkdir(parents=True)
    (root / "metadata").mkdir()
    mesh_dir = root / "meshes"
    mesh_dir.mkdir()

    folders = ["f0.8", "f1", "f2", "f10"]
    cells = {"f0.8": 6, "f1": 5, "f2": 4, "f10": 2}
    stages = {"f0.8": [0, 1], "f1": [0, 1], "f2": [0, 1], "f10": [0]}

    mesh_map = {}
    for folder in folders:
        case_path = out / folder
        case_path.mkdir()
        grid = _quad_row_grid(cells[folder])
        for stage in stages[folder]:
            _write_stage(str(case_path), stage, grid)
        mesh_file = mesh_dir / f"mesh_{folder}.msh"
        mesh_file.write_text("")
        mesh_map[folder] = str(mesh_file)

    cm = {
        "root_dir": str(root),
        "eq_type": "rans",
        "folder_fmt": "f{mesh:g}",
        "design_vars": ["aoa", "mach", "mesh"],
        "num_stages": 2,
        "df_cases": {
            "aoa": [1.0, 1.0, 1.0, 1.0],
            "mach": [0.8, 0.8, 0.8, 0.8],
            "mesh": [0.8, 1.0, 2.0, 10.0],
            "folder": folders,
        },
        "mesh_map": mesh_map,
    }
    with open(root / "metadata" / "cases_metadata.json", "w") as fh:
        json.dump(cm, fh)
    return str(root)


def test_mesh_map_is_loaded_and_used(gci_running):
    """The declared mesh_map must reach the reader and win over scanning.

    ``CODAReader.__init__`` keeps only four keys of the JSON, and
    ``CODASingleReader`` did not extend it, so ``mesh_map`` was always
    empty and every case fell back to scanning its own folder.
    """
    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    assert db.reader.metadata["mesh_map"], "mesh_map never reached the reader"

    info = db.sim_metadata["f2"]["mesh_info"]
    assert info["source"] == "mesh_map"
    assert info["file"] == "mesh_f2.msh"
    # Resolved outside outputs/, where no scan of the case folder could
    # have found it.
    assert "meshes" in info["abspath"]
    assert os.path.isfile(info["abspath"])


def test_mesh_map_falls_back_to_the_local_copy(gci_running):
    """A declared mesh that moved is recovered from the case folder."""
    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    declared = db.reader.metadata["mesh_map"]["f2"]
    os.remove(declared)
    local = os.path.join(
        gci_running, "outputs", "f2", os.path.basename(declared)
    )
    open(local, "w").close()

    db.reader.parse_simulation_dirs()
    info = db.reader.sim_metadata["f2"]["mesh_info"]
    assert info["source"] == "mesh_map_local_copy"
    assert os.path.samefile(info["abspath"], local)


def test_df_state_is_ordered_by_case_idx(gci_running):
    """Row position and case_idx must agree.

    Folders are discovered lexicographically ('f10' < 'f2'), which does
    not match df_cases order. Leaving df_state in that order made every
    consumer that indexes it positionally read the wrong case.
    """
    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    assert db.df_state["case_idx"].tolist() == [0, 1, 2, 3]
    assert db.df_state["folder"].tolist() == ["f0.8", "f1", "f2", "f10"]
    for pos, case_idx in enumerate(db.df_state["case_idx"]):
        assert db.reader.case_per_idx(int(case_idx)) == (
            db.df_state["folder"].iloc[pos]
        )


def test_unfinished_case_is_extracted_from_the_stages_it_has(gci_running):
    """A case with fewer stages than num_stages still yields its geometry."""
    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx="all")
    grp = db.data_dict["CADGroup_3"]

    assert grp["case_order"] == ["f0.8", "f1", "f2", "f10"]
    assert grp["stages_available"] == [[0, 1], [0, 1], [0, 1], [0]]
    assert [c.shape[0] for c in grp["Coord"]] == [6, 5, 4, 2]
    assert db.reader.skipped_cases["CADGroup_3"] == []


def test_extract_outputs_marks_cases_without_the_stage(gci_running):
    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx="all")
    db.extract_outputs(
        stage=1, id_groups=(3,), cases_idx="all",
        var_name_excluded=["CADGroupID"],
    )
    var = db.data_dict["CADGroup_3"]["Vars"]["1"]["Pressure"]
    # f10 has no stage 1 -> None, and the list still tracks case_order.
    assert [None if a is None else a.shape[0] for a in var] == [6, 5, 4, None]


def test_extract_inputs_is_atomic_per_case(gci_running, monkeypatch):
    """A case failing mid-way must leave no partial geometry behind.

    Geometry used to be appended at stage 0 while case_order/FlCc were
    only appended after every stage had been read, so a failure at
    stage > 0 shifted every geometry list by one case.
    """
    from FotR.characters.readers.coda_single import CODASingleReader

    original = CODASingleReader.load_vtu_from_stage

    def flaky(self, case_name, stage, vtu_type="surface", verbose=False):
        if case_name == "f1" and stage == 1:
            raise RuntimeError("simulated mid-case failure")
        return original(self, case_name, stage, vtu_type, verbose)

    monkeypatch.setattr(CODASingleReader, "load_vtu_from_stage", flaky)

    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    with pytest.warns(UserWarning, match="simulated mid-case failure"):
        db.extract_inputs(id_groups=(3,), cases_idx="all")
    grp = db.data_dict["CADGroup_3"]

    assert grp["case_order"] == ["f0.8", "f2", "f10"]
    assert grp["FlCc"][:, 2].tolist() == [0.8, 2.0, 10.0]
    for key in ("Coord", "NodeCoord", "Conec", "eltype", "cellOrder",
                "pointOrder", "idx_sort", "mesh_files", "stages_available"):
        assert len(grp[key]) == 3, key
    # Geometry belongs to the surviving cases, not shifted by one.
    assert [c.shape[0] for c in grp["Coord"]] == [6, 4, 2]
    assert [c for c, _ in db.reader.skipped_cases["CADGroup_3"]] == ["f1"]


def test_extract_inputs_does_not_wipe_previously_read_vars(gci_running):
    db = FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx="all")
    db.extract_outputs(stage=0, id_groups=(3,), cases_idx="all")
    assert list(db.data_dict["CADGroup_3"]["Vars"]) == ["0"]

    db.extract_inputs(id_groups=(3,), cases_idx="all")
    assert list(db.data_dict["CADGroup_3"]["Vars"]) == ["0"]


def _rewrite_metadata(root, mutate):
    path = os.path.join(root, "metadata", "cases_metadata.json")
    with open(path) as fh:
        cm = json.load(fh)
    mutate(cm)
    with open(path, "w") as fh:
        json.dump(cm, fh)


def test_ragged_bookkeeping_column_is_dropped_with_a_warning(gci_running):
    """Adding a case can leave 'exist' one entry short; that must not block.

    ``pd.DataFrame.from_dict`` raises an opaque ``ValueError: All arrays
    must be of the same length`` from inside pandas, which the JSON guard
    does not catch. Bookkeeping columns are dropped instead.
    """
    def mutate(cm):
        cm["df_cases"]["exist"] = [False, False, False]   # 3 vs 4 cases
    _rewrite_metadata(gci_running, mutate)

    with pytest.warns(UserWarning, match="ragged column"):
        db = FRODO(
            root_dir=gci_running, format="CODA_SINGLE", initial_parse=True
        )
    assert "exist" not in db.reader.metadata["df_cases"].columns
    assert db.df_state["folder"].tolist() == ["f0.8", "f1", "f2", "f10"]


def test_ragged_identity_column_is_rejected(gci_running):
    """A short design-var column compromises identity, so it must raise."""
    def mutate(cm):
        cm["df_cases"]["mesh"] = [0.8, 1.0, 2.0]          # 3 vs 4 cases
    _rewrite_metadata(gci_running, mutate)

    with pytest.raises(ValueError, match="case-identity column"):
        FRODO(root_dir=gci_running, format="CODA_SINGLE", initial_parse=True)


@pytest.fixture
def mixed_cell_root(tmp_path):
    """One case whose CAD group mixes triangles and quads.

    ``SAM.Backpack.get_unified_connectivity`` groups cells by element
    type, so masking its output with a mask built in original cell order
    picked the wrong rows on any mixed-type mesh.
    """
    root = tmp_path / "mixed"
    out = root / "outputs"
    (out / "f1").mkdir(parents=True)
    (root / "metadata").mkdir()

    # 3x3 node grid.
    xx, yy = np.meshgrid(np.arange(3.0), np.arange(3.0))
    points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(9)]
    )
    cells = [
        4, 0, 1, 4, 3,      # quad
        4, 1, 2, 5, 4,      # quad
        3, 3, 4, 6,         # tri
        3, 4, 7, 6,         # tri
        3, 4, 5, 7,         # tri
        3, 5, 8, 7,         # tri
    ]
    celltypes = [
        pyvista.CellType.QUAD, pyvista.CellType.QUAD,
        pyvista.CellType.TRIANGLE, pyvista.CellType.TRIANGLE,
        pyvista.CellType.TRIANGLE, pyvista.CellType.TRIANGLE,
    ]
    grid = pyvista.UnstructuredGrid(
        np.array(cells), np.array(celltypes), points
    )
    # Interleave the groups so the mask is neither contiguous nor aligned
    # with the type grouping.
    grid.cell_data["CADGroupID"] = np.array([3, 7, 3, 3, 7, 3], dtype=int)
    grid.cell_data["Pressure"] = np.arange(6, dtype=float)
    _write_stage(str(out / "f1"), 0, grid)

    cm = {
        "eq_type": "rans",
        "folder_fmt": "f{mesh:g}",
        "design_vars": ["mesh"],
        "num_stages": 1,
        "df_cases": {"mesh": [1.0], "folder": ["f1"]},
        "mesh_map": {},
    }
    with open(root / "metadata" / "cases_metadata.json", "w") as fh:
        json.dump(cm, fh)
    return str(root)


def test_connectivity_is_consistent_with_the_stored_geometry(mixed_cell_root):
    """Conec must index NodeCoord and describe the right cells.

    Reconstructing each cell's centroid from Conec + NodeCoord has to
    reproduce Coord exactly. That catches all three historical defects at
    once: connectivity taken from the full mesh instead of the extracted
    subset, rows reordered by element type, and node ids not remapped
    through the sort permutation.
    """
    db = FRODO(
        root_dir=mixed_cell_root, format="CODA_SINGLE", initial_parse=True
    )
    db.extract_inputs(id_groups=(3,), cases_idx="all")
    grp = db.data_dict["CADGroup_3"]

    conec = grp["Conec"][0]
    nodes = grp["NodeCoord"][0]
    coord = grp["Coord"][0]

    assert conec.shape[0] == coord.shape[0] == 4     # the 4 group-3 cells
    assert conec.max() < nodes.shape[0]

    padding = conec < 0
    gathered = nodes[np.where(padding, 0, conec)]
    weights = (~padding).astype(float)[:, :, None]
    centroids = (gathered * weights).sum(axis=1) / weights.sum(axis=1)
    assert np.allclose(centroids, coord, atol=1e-12)

    # eltype follows the same permutation as Coord/Conec: the group holds
    # one quad and three triangles, and the padding says which is which.
    n_nodes_per_cell = (~padding).sum(axis=1)
    expected = np.where(
        grp["eltype"][0] == int(pyvista.CellType.QUAD), 4, 3
    )
    assert np.array_equal(n_nodes_per_cell, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
