"""
tests/test_coda_residuals_plot.py
==================================
Regression tests for ``CODAResiduals.plot_all_final_residuals``.

Covers the single-design-variable (1-D) branch, which used to fail with
``KeyError: 'total_iterations'`` (colour column) and ``IndexError`` on
the ``(p[0], p[1])`` annotation, plus the empty-columns guard.
"""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from FotR.characters.residuals.coda import CODAResiduals


def _write_monitor(path, columns, rows):
    header = " ".join(f'"{c}"' for c in columns)
    with open(path, "w") as fh:
        fh.write("# CODA monitor file\n")
        fh.write(header + "\n")
        for i, row in enumerate(rows):
            fh.write(" ".join([str(i)] + [str(v) for v in row]) + "\n")


def _make_case(root, folder, m_value):
    case_path = os.path.join(root, folder)
    os.makedirs(case_path, exist_ok=True)
    cols = ["Time", "DensityResidual"]
    rows = [[0.0 + i * 0.1, 10.0 / (i + 1)] for i in range(5)]
    _write_monitor(
        os.path.join(case_path, "output_0__monitors_TimeIntegration.dat"),
        ["Iteration"] + cols, [[r[0], r[1]] for r in rows],
    )
    _write_monitor(
        os.path.join(case_path, "output_0__monitors_stage0InitialResidual.dat"),
        ["Iteration"] + cols, [[10.0, 10.0]],
    )
    _write_monitor(
        os.path.join(case_path, "output_0__monitors_CFLRamp.dat"),
        ["Iteration", "SERReferenceDensityResidual"], [[1.0, 1.0]],
    )
    return case_path


class _FakeDB:
    def __init__(self, root, folders, m_values):
        self.root_dir = root
        self.format = "CODA"
        self.sim_metadata = {}
        for folder, m in zip(folders, m_values):
            path = _make_case(root, folder, m)
            self.sim_metadata[folder] = {"path": path, "stages": {0: {}}}
        self.metadata = {
            "folder_fmt": "M_{M:.4f}",
            "design_vars": ["M"],
            "num_stages": 1,
            "df_cases": pd.DataFrame(
                {"M": m_values, "folder": folders}
            ),
        }
        self.df_state = pd.DataFrame(
            {"M": m_values, "stage": [1, 1]}
        )


def test_plot_all_final_residuals_1d_saves_figure(tmp_path):
    db = _FakeDB(str(tmp_path), ["M_0.3000", "M_0.5000"], [0.3, 0.5])
    res = CODAResiduals(db)
    out = str(tmp_path / "plots")
    res.plot_all_final_residuals(
        save_dir=out, mode="norm", stage=[0], only_finished=True
    )
    assert os.path.isfile(os.path.join(out, "residuals_all_cases.png"))


def test_plot_all_final_residuals_1d_single_axes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = _FakeDB(str(tmp_path), ["M_0.3000", "M_0.5000"], [0.3, 0.5])
    res = CODAResiduals(db)
    res.plot_all_final_residuals(
        save_dir=str(tmp_path / "plots"), mode="norm", stage=[0],
        only_finished=True,
    )
    fig = plt.gcf()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "M"
    labels = [
        t.get_text()
        for legend in fig.legends
        for t in legend.get_texts()
    ]
    assert any("converged" in label for label in labels)
    assert any("lim" in label or "1e-" in label for label in labels)
    plt.close("all")


def test_plot_all_final_residuals_1d_legend_has_no_duplicates(tmp_path):
    """Each residual appears once; the marker carries the state.

    Labelling both the ``*`` and the ``o`` artists of every column listed
    each residual twice ("<name> (converged)" / "<name> (non-converged)"),
    so a run with five residuals produced an eleven-entry legend that was
    mostly repetition.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = _FakeDB(str(tmp_path), ["M_0.3000", "M_0.5000"], [0.3, 0.5])
    res = CODAResiduals(db)
    plt.close("all")
    res.plot_all_final_residuals(
        save_dir=str(tmp_path / "plots"), mode="norm", stage=[0],
        only_finished=True,
    )
    fig = plt.gcf()
    labels = [
        t.get_text()
        for legend in fig.legends
        for t in legend.get_texts()
    ]
    assert labels, "the legend should not be empty"
    assert len(labels) == len(set(labels)), f"duplicated entries: {labels}"
    # No entry pairs a residual name with its state any more.
    assert not any("(converged)" in label for label in labels)
    assert not any("(non-converged)" in label for label in labels)
    # The state is its own, neutral entry — and only the states actually
    # present are listed, so this fixture (nothing converged) shows one.
    assert {"converged", "not converged"} & set(labels)
    plt.close("all")


def test_plot_all_final_residuals_1d_one_plot_per_stage(tmp_path):
    db = _FakeDB(str(tmp_path), ["M_0.3000", "M_0.5000"], [0.3, 0.5])
    res = CODAResiduals(db)
    out = str(tmp_path / "plots")
    # Fake a mixed-stage table: same cases at two stages.
    df = res.get_all_final_residuals(
        stage=[0], load_in_metadata=False
    )
    df_mixed = pd.concat(
        [df.assign(stage=0), df.assign(stage=1)], ignore_index=True
    )
    real = res.get_all_final_residuals
    res.get_all_final_residuals = lambda **k: df_mixed
    try:
        res.plot_all_final_residuals(save_dir=out, mode="norm")
    finally:
        res.get_all_final_residuals = real
    assert os.path.isfile(os.path.join(out, "residuals_all_cases_stage0.png"))
    assert os.path.isfile(os.path.join(out, "residuals_all_cases_stage1.png"))
    assert not os.path.isfile(os.path.join(out, "residuals_all_cases.png"))


def test_plot_all_final_residuals_no_columns_warns(tmp_path):
    db = _FakeDB(str(tmp_path), ["M_0.3000", "M_0.5000"], [0.3, 0.5])
    res = CODAResiduals(db)
    with pytest.warns(UserWarning, match="No residual columns"):
        res.plot_all_final_residuals(
            save_dir=str(tmp_path / "plots"), mode="nonexistent-mode",
            stage=[0],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
