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


def test_extract_outputs_requires_matching_selection(gci_root):
    db = FRODO(root_dir=gci_root, format="CODA_SINGLE", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[1, 2])
    with pytest.raises(ValueError, match="Selection mismatch"):
        db.extract_outputs(stage=0, id_groups=(3,), cases_idx=[1])


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
