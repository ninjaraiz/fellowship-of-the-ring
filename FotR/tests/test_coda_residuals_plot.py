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
