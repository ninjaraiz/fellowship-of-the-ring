"""
tests/test_horses3d.py
================================
Tests for ``Horses3DReader``.

Heavy optional dependencies pulled in transitively via
``FotR.characters.sam.SAM`` (pyvista, torch, h5py, seaborn, sklearn, …)
are stubbed out with ``types.ModuleType`` placeholders before import, so
this suite runs without those packages installed — mirroring the
isolation approach already used by the other reader test modules in this
project.
"""

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ── Stub heavy optional dependencies so importing FotR.characters.sam.SAM
#    (transitively imported by Horses3DReader) does not require them to be
#    actually installed. Only stub what is not already importable. ────────
def _stub_module(name: str, **attrs):
    try:
        if importlib.util.find_spec(name) is not None:
            return
    except ValueError:
        # Otro modulo de tests ya inserto un stub sin __spec__ en
        # sys.modules durante la recoleccion conjunta; reutilizarlo.
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_stub_module("torch")
_stub_module("pyvista", CellType=types.SimpleNamespace(TRIANGLE=5, TETRA=10))
_stub_module("h5py")
_stub_module("seaborn")
_stub_module("sklearn")
_stub_module("sklearn.preprocessing", StandardScaler=object)
_stub_module("sklearn.decomposition", PCA=object)
_stub_module("sklearn.mixture", GaussianMixture=object)
_stub_module("scipy")
_stub_module("scipy.spatial", Delaunay=object, cKDTree=object, KDTree=object)
_stub_module("tqdm")
_stub_module("tqdm.auto", tqdm=lambda x, **k: x)
_stub_module("plotly")
_stub_module("plotly.graph_objects")
_stub_module("networkx")

# ── Joint-collection robustness: sibling test modules install a minimal
#    FotR.characters.sam stub (find_files -> []) for isolation. This suite
#    exercises the real SAM mesh/hsol helpers, so evict the stand-in when
#    it is not the real implementation before importing the reader. ──────
_incumbent = sys.modules.get("FotR.characters.sam")
_incumbent_bp = getattr(getattr(_incumbent, "SAM", None), "Backpack", None)
if _incumbent is not None and not hasattr(
    _incumbent_bp, "read_horses_mesh_h5"
):
    for _dotted in (
        "FotR.characters.readers.horses3d",
        "FotR.characters.sam",
    ):
        sys.modules.pop(_dotted, None)

from FotR.characters.readers.horses3d import Horses3DReader  # noqa: E402


# =============================================================================
# Synthetic dataset builder
# =============================================================================

_CONTROL_TEMPLATE = """\
Flow equations        = "NS"
mesh file name        = "MESH/{mesh_name}"
Polynomial order      = {p}
Number of time steps  = 1.d6
Output Interval       = 1.d4

Convergence tolerance = 1.d-10
cfl                   = 0.4
dcfl                  = 0.4
mach number           = {mach}
Reynolds number       = 200.0

AOA theta             = {aoa}
AOA phi               = 0.0

solution file name    = "RESULTS/{sol_name}"
save gradients with solution = .true.

restart               = {restart_flag}
{restart_lines}
simulation type       = time-accurate
final time            = 16.d2
"""


def _write_control(
    case_path, p, mesh_name, mach=0.3, aoa=0.0,
    restart=False, restart_sol=None, restart_p=None,
    p_in_filename=None,
    declared_p=None,
):
    """Write one file_control_p<p>.control plus its .hsol/.residuals."""
    p_name = p if p_in_filename is None else p_in_filename
    sol_name = f"case_v2g2p{p}.hsol"

    restart_lines = ""
    if restart:
        restart_lines = (
            f'restart file name     = "RESULTS/{restart_sol}"\n'
            f"restart polorder      = {restart_p}\n"
        )

    content = _CONTROL_TEMPLATE.format(
        mesh_name=mesh_name,
        p=(declared_p if declared_p is not None else p),
        mach=mach,
        aoa=aoa,
        sol_name=sol_name,
        restart_flag=".true." if restart else ".false.",
        restart_lines=restart_lines,
    )

    control_path = os.path.join(case_path, f"file_control_p{p_name}.control")
    with open(control_path, "w") as fh:
        fh.write(content)

    results_dir = os.path.join(case_path, "RESULTS")
    os.makedirs(results_dir, exist_ok=True)
    open(os.path.join(results_dir, sol_name), "w").close()
    open(
        os.path.join(results_dir, sol_name.replace(".hsol", ".residuals")),
        "w",
    ).close()

    return control_path, sol_name


def _make_mesh(case_path, mesh_name="case_v2g2_mesh.h5"):
    mesh_dir = os.path.join(case_path, "MESH")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, mesh_name)
    # Minimal valid Hopr-like HDF5 mesh so extract_inputs can read
    # NodeCoords verbatim (no .bmesh/.hmesh needed).
    try:
        import h5py
        import numpy as _np

        coords = _np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=_np.float64,
        )
        with h5py.File(mesh_path, "w") as hf:
            hf.create_dataset("NodeCoords", data=coords)
            hf.create_dataset(
                "GlobalNodeIDs",
                data=_np.arange(1, 5, dtype=_np.int32),
            )
            hf.create_dataset(
                "ElemInfo", data=_np.zeros((1, 6), dtype=_np.int32)
            )
            hf.create_dataset(
                "ElemCounter", data=_np.zeros((1, 2), dtype=_np.int32)
            )
            hf.attrs["Ngeo"] = 2
            hf.attrs["nElems"] = 1
            hf.attrs["nNodes"] = 4
    except ImportError:
        open(mesh_path, "w").close()
    return mesh_name


def _make_basic_dataset(root_dir):
    """
    case_1: p=[2, 3], g=2 (inferred from 'v2g2'), p=3 restarts from p=2.
    case_2: p=[2], g=2.
    design_vars come from metadata/cases_metadata.json (FlCc uses
    user design_vars only, never .control params).
    """
    outputs = os.path.join(root_dir, "outputs")
    os.makedirs(outputs, exist_ok=True)

    case_1 = os.path.join(outputs, "case_1")
    os.makedirs(case_1, exist_ok=True)
    mesh_1 = _make_mesh(case_1)
    _write_control(case_1, p=2, mesh_name=mesh_1, mach=0.3, aoa=0.0)
    _write_control(
        case_1, p=3, mesh_name=mesh_1, mach=0.3, aoa=0.0,
        restart=True, restart_sol="case_v2g2p2.hsol", restart_p=2,
    )

    case_2 = os.path.join(outputs, "case_2")
    os.makedirs(case_2, exist_ok=True)
    mesh_2 = _make_mesh(case_2)
    _write_control(case_2, p=2, mesh_name=mesh_2, mach=0.5, aoa=2.0)

    meta_dir = os.path.join(root_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    cm = {
        "eq_type": "NS",
        "design_vars": ["mach", "aoa"],
        "df_cases": {
            "folder": ["case_1", "case_2"],
            "mach":   [0.3, 0.5],
            "aoa":    [0.0, 2.0],
        },
    }
    with open(os.path.join(meta_dir, "cases_metadata.json"), "w") as fh:
        json.dump(cm, fh)

    return root_dir


@pytest.fixture
def basic_dataset(tmp_path):
    return _make_basic_dataset(str(tmp_path))


# =============================================================================
# Discovery
# =============================================================================

def test_discovers_all_cases(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    assert set(reader.sim_metadata.keys()) == {"case_1", "case_2"}


def test_case_idx_assigned_per_case_not_per_p(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    # 2 physical cases -> exactly 2 rows, regardless of case_1 having 2 p's.
    assert len(reader.df_state) == 2
    idxs = {m["case_idx"] for m in reader.sim_metadata.values()}
    assert idxs == {0, 1}


def test_p_discovery_per_case(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    assert sorted(reader.sim_metadata["case_1"]["solutions"].keys()) == [2, 3]
    assert sorted(reader.sim_metadata["case_2"]["solutions"].keys()) == [2]


def test_df_state_has_list_valued_p_available_not_fixed_columns(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    row1 = reader.df_state.loc[reader.df_state["case"] == "case_1"].iloc[0]
    row2 = reader.df_state.loc[reader.df_state["case"] == "case_2"].iloc[0]
    assert row1["p_available"] == [2, 3]
    assert row2["p_available"] == [2]
    # No p1..p7-style fixed boolean columns.
    assert not any(c.startswith("p") and c[1:].isdigit()
                   for c in reader.df_state.columns)


# =============================================================================
# Control -> solution association
# =============================================================================

def test_control_associated_with_correct_solution(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    sol = reader.sim_metadata["case_1"]["solutions"][3]
    assert sol["control_file"] == "file_control_p3.control"
    assert sol["solution_file"] == "RESULTS/case_v2g2p3.hsol"
    assert sol["residuals_found"] is True
    assert sol["residuals_file"] == "RESULTS/case_v2g2p3.residuals"


def test_restart_information_preserved(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    sol3 = reader.sim_metadata["case_1"]["solutions"][3]
    assert sol3["restart"] is True
    assert sol3["restart_file"] == "RESULTS/case_v2g2p2.hsol"
    assert sol3["restart_polorder"] == 2

    sol2 = reader.sim_metadata["case_1"]["solutions"][2]
    assert sol2["restart"] is False
    assert sol2["restart_file"] is None


# =============================================================================
# Mesh / g
# =============================================================================

def test_mesh_shared_and_associated_per_case(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    assert (
        reader.sim_metadata["case_1"]["mesh"]["file"]
        == "MESH/case_v2g2_mesh.h5"
    )


def test_geometric_g_inferred_from_filename(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    assert reader.sim_metadata["case_1"]["mesh"]["g"] == 2
    assert reader.df_state.loc[
        reader.df_state["case"] == "case_1", "g"
    ].iloc[0] == 2


def test_geometric_g_left_none_when_pattern_absent(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case, mesh_name="unrelated_mesh_name.h5")
    _write_control(case, p=2, mesh_name=mesh)

    reader = Horses3DReader(root_dir=root)
    reader.parse_simulation_dirs()
    assert reader.sim_metadata["case_1"]["mesh"]["g"] is None


# =============================================================================
# Flight conditions
# =============================================================================

def test_flight_conditions_parsed_per_solution(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    params = reader.sim_metadata["case_2"]["solutions"][2]["params"]
    assert params["mach_number"] == pytest.approx(0.5)
    assert params["aoa_theta"] == pytest.approx(2.0)
    assert params["flow_equations"] == "NS"


def test_df_state_pools_consistent_params_across_p(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    row1 = reader.df_state.loc[reader.df_state["case"] == "case_1"].iloc[0]
    # case_1 p=2 and p=3 share mach=0.3 -> pooled into df_state.
    assert row1["mach_number"] == pytest.approx(0.3)


def test_df_state_drops_param_that_disagrees_across_p(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case)
    _write_control(case, p=2, mesh_name=mesh, mach=0.3)
    _write_control(case, p=3, mesh_name=mesh, mach=0.6)  # disagreeing mach

    reader = Horses3DReader(root_dir=root)
    reader.parse_simulation_dirs()
    assert "mach_number" not in reader.df_state.columns
    # still accessible per-solution:
    assert reader.sim_metadata["case_1"]["solutions"][2]["params"]["mach_number"] == 0.3
    assert reader.sim_metadata["case_1"]["solutions"][3]["params"]["mach_number"] == 0.6


# =============================================================================
# Subsets
# =============================================================================

def test_define_and_resolve_subset(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.define_subset("training", case_idx=[0])
    resolved = reader._resolve_cases_idx("training", "all")
    assert resolved == [0]


def test_subset_and_explicit_cases_idx_conflict(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.define_subset("training", case_idx=[0])
    with pytest.raises(ValueError):
        reader._resolve_cases_idx("training", [1])


def test_subset_with_unknown_case_idx_raises(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    with pytest.raises(KeyError):
        reader.define_subset("bad", case_idx=[99])


def test_subset_refers_to_physical_cases_not_p(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.define_subset("training", case_idx=[0])
    # case_idx 0 == case_1, which has p in [2, 3]; subset carries both.
    assert reader.sim_metadata["case_1"]["case_idx"] == 0
    assert sorted(reader.sim_metadata["case_1"]["solutions"]) == [2, 3]


# =============================================================================
# extract_inputs / extract_outputs — interface contract
# =============================================================================

def test_extract_inputs_builds_codalike_group(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.extract_inputs(subset=None, cases_idx='all')
    # One group per (mesh, g); FlCc uses user design_vars only.
    group_keys = [k for k in reader.data_dict if k.startswith("CADGroup_")]
    assert len(group_keys) == 1
    grp = reader.data_dict[group_keys[0]]
    assert grp["FlCc"].shape == (2, 2)
    assert grp["case_order"] == ["case_1", "case_2"]
    assert grp["design_vars"] == ["mach", "aoa"]
    assert grp["Coord"].shape == (4, 3)
    assert grp["g"] == 2
    # Backward-compatible top-level view:
    assert reader.data_dict["FlCc"].shape[0] == 2


def test_extract_inputs_flcc_uses_design_vars_only(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.extract_inputs()
    grp = reader.data_dict[
        next(k for k in reader.data_dict if k.startswith("CADGroup_"))
    ]
    # mach/aoa from cases_metadata.json — never cfl/dcfl/.control params.
    assert grp["FlCc"].tolist() == [[0.3, 0.0], [0.5, 2.0]]


def test_extract_outputs_requires_extract_inputs_first(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    with pytest.raises(RuntimeError):
        reader.extract_outputs()


def test_extract_outputs_header_lazy_after_inputs(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.extract_inputs(p=2)
    reader.extract_outputs(p=2)
    grp = reader.data_dict[
        next(k for k in reader.data_dict if k.startswith("CADGroup_"))
    ]
    assert set(grp["Vars"]["2"].keys()) == {
        "rho", "rhou", "rhov", "rhow", "rhoE"
    }
    lazy = grp["Vars"]["2"]["rho"][0]
    assert lazy.shape == (4,)
    with pytest.raises(NotImplementedError):
        lazy.load()


def test_extract_outputs_rejects_mixed_p(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    reader.extract_inputs(p="max")
    # case_1 max p=3, case_2 max p=2 -> mixed -> ValueError.
    with pytest.raises(ValueError):
        reader.extract_outputs(p="max")


# =============================================================================
# Structural validations
# =============================================================================

def test_case_without_control_raises_in_strict_mode(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    os.makedirs(os.path.join(outputs, "empty_case"))

    reader = Horses3DReader(root_dir=root, strict=True)
    with pytest.raises(ValueError):
        reader.parse_simulation_dirs()


def test_case_without_control_warns_in_non_strict_mode(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    os.makedirs(os.path.join(outputs, "empty_case"))

    reader = Horses3DReader(root_dir=root, strict=False)
    with pytest.warns(UserWarning):
        reader.parse_simulation_dirs()
    assert reader.sim_metadata["empty_case"]["solutions"] == {}


def test_p_mismatch_between_filename_and_content_raises_strict(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case)
    # filename says p2 but control declares Polynomial order = 3
    _write_control(case, p=2, mesh_name=mesh, declared_p=3)

    reader = Horses3DReader(root_dir=root, strict=True)
    with pytest.raises(ValueError):
        reader.parse_simulation_dirs()


def test_p_less_than_g_raises_strict(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case, mesh_name="case_v2g3_mesh.h5")  # g=3
    _write_control(case, p=2, mesh_name=mesh)  # p=2 < g=3

    reader = Horses3DReader(root_dir=root, strict=True)
    with pytest.raises(ValueError):
        reader.parse_simulation_dirs()


def test_missing_declared_solution_file_raises_strict(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case)

    control_path = os.path.join(case, "file_control_p2.control")
    with open(control_path, "w") as fh:
        fh.write(_CONTROL_TEMPLATE.format(
            mesh_name=mesh, p=2, mach=0.3, aoa=0.0,
            sol_name="does_not_exist.hsol",
            restart_flag=".false.", restart_lines="",
        ))
    # Note: solution file intentionally not created on disk.

    reader = Horses3DReader(root_dir=root, strict=True)
    with pytest.raises(ValueError):
        reader.parse_simulation_dirs()


def test_restart_true_without_existing_restart_file_raises_strict(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case)
    _write_control(
        case, p=3, mesh_name=mesh,
        restart=True, restart_sol="nonexistent_p2.hsol", restart_p=2,
    )

    reader = Horses3DReader(root_dir=root, strict=True)
    with pytest.raises(ValueError):
        reader.parse_simulation_dirs()


def test_incompatible_mesh_between_controls_raises_strict(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh_a = _make_mesh(case, mesh_name="mesh_a_v2g2.h5")
    mesh_b = _make_mesh(case, mesh_name="mesh_b_v2g2.h5")
    _write_control(case, p=2, mesh_name=mesh_a)
    _write_control(case, p=3, mesh_name=mesh_b)

    reader = Horses3DReader(root_dir=root, strict=True)
    with pytest.raises(ValueError):
        reader.parse_simulation_dirs()


def test_single_control_yields_single_solution(tmp_path):
    # A duplicate p via filename is unreachable on a real filesystem
    # (the second file would overwrite the first), so this test pins the
    # attainable behaviour: one control file -> exactly one solution.
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case)
    _write_control(case, p=2, mesh_name=mesh)
    reader = Horses3DReader(root_dir=root, strict=True)
    reader.parse_simulation_dirs()
    assert list(reader.sim_metadata["case_1"]["solutions"].keys()) == [2]


# =============================================================================
# Different p sets per case
# =============================================================================

def test_cases_with_different_p_sets_are_independent(basic_dataset):
    reader = Horses3DReader(root_dir=basic_dataset)
    reader.parse_simulation_dirs()
    assert sorted(reader.sim_metadata["case_1"]["solutions"]) == [2, 3]
    assert sorted(reader.sim_metadata["case_2"]["solutions"]) == [2]
    assert reader.df_state.set_index("case").loc["case_1", "n_solutions"] == 2
    assert reader.df_state.set_index("case").loc["case_2", "n_solutions"] == 1


# =============================================================================
# Optional cases_metadata.json (additive design_vars join)
# =============================================================================

def test_optional_cases_metadata_joins_design_vars(tmp_path):
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    case = os.path.join(outputs, "case_1")
    os.makedirs(case)
    mesh = _make_mesh(case)
    _write_control(case, p=2, mesh_name=mesh)

    meta_dir = os.path.join(root, "metadata")
    os.makedirs(meta_dir)
    cm = {
        "eq_type": "NS",
        "design_vars": ["reynolds"],
        "df_cases": {
            "folder":    ["case_1"],
            "reynolds":  [200.0],
        },
    }
    with open(os.path.join(meta_dir, "cases_metadata.json"), "w") as fh:
        json.dump(cm, fh)

    reader = Horses3DReader(root_dir=root)
    reader.parse_simulation_dirs()
    assert reader.sim_metadata["case_1"]["design_vars"]["reynolds"] == 200.0
    assert reader.df_state.loc[
        reader.df_state["case"] == "case_1", "reynolds"
    ].iloc[0] == 200.0


def test_missing_cases_metadata_does_not_block_discovery(tmp_path):
    # Dataset without metadata/cases_metadata.json and without an
    # inferable '<name>_<value>' folder pattern: discovery proceeds
    # with folder names as the sole case identity.
    root = str(tmp_path)
    outputs = os.path.join(root, "outputs")
    os.makedirs(outputs)
    for case_name in ("runA", "runB"):
        case = os.path.join(outputs, case_name)
        os.makedirs(case)
        mesh = _make_mesh(case)
        _write_control(case, p=2, mesh_name=mesh)
    reader = Horses3DReader(root_dir=root)
    reader.parse_simulation_dirs()
    assert len(reader.df_state) == 2
    assert reader.metadata["design_vars"] is None