"""
tests/test_coda_wall_integrals.py
==================================
Tests for ``CODASets.plot_wall_integrals`` with synthetic monitor files.
"""

import os
import sys
import types

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# plot_wall_integrals never touches pyLOM, but sets/coda.py imports it
# at module level: provide a bare stand-in when it is not installed
# (other test modules may have installed a fuller one already).
if "pyLOM" not in sys.modules:
    try:
        import pyLOM  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["pyLOM"] = types.ModuleType("pyLOM")

from FotR.characters.readers.base import BaseReader
from FotR.characters.sets.coda import CODASets


def _write_wall(path, n_rows=50, start=0.1):
    with open(path, "w") as fh:
        fh.write("# CODA wall monitors\n")
        fh.write('"Iteration" "Time" "CoefDrag" "CoefLift"\n')
        for i in range(n_rows):
            fh.write(
                f"{i} {i * 0.01} {start / (i + 1):.6f} "
                f"{2 * start / (i + 1):.6f}\n"
            )


class _StubReader:
    def __init__(self, df_cases):
        self._df_cases = df_cases
        self.subsets = {}

    def _resolve_cases_idx(self, cases_idx, subset):
        if subset is not None:
            raise KeyError(subset)
        return BaseReader._normalise_cases_idx(cases_idx, self._df_cases)


class _FakeDB:
    def __init__(self, root, df_cases):
        self.root_dir = root
        self.metadata = {
            "design_vars": ["M"],
            "num_stages": 1,
            "df_cases": df_cases,
        }
        self.reader = _StubReader(df_cases)


@pytest.fixture
def db(tmp_path):
    folders = ["M_0.3000", "M_0.5000"]
    for folder, n in zip(folders, (50, 30)):
        case_path = tmp_path / "outputs" / folder
        case_path.mkdir(parents=True)
        _write_wall(
            str(case_path / "output_0__monitors_wall_boundary_integrals.dat"),
            n_rows=n,
        )
    df_cases = pd.DataFrame(
        {"M": [0.3, 0.5], "folder": folders}
    )
    return _FakeDB(str(tmp_path), df_cases)


def test_plot_wall_integrals_saves_per_variable(db, tmp_path):
    sets = CODASets(db)
    out = str(tmp_path / "plots")
    sets.plot_wall_integrals(
        cases_idx=[0, 1],
        var_metrics=["CoefDrag", "CoefLift"],
        save_dir=out,
    )
    assert os.path.isfile(os.path.join(out, "CoefDrag.png"))
    assert os.path.isfile(os.path.join(out, "CoefLift.png"))


def test_plot_wall_integrals_all_vars_and_band(db, tmp_path):
    sets = CODASets(db)
    out = str(tmp_path / "plots")
    sets.plot_wall_integrals(cases_idx=[0, 1], band=True, save_dir=out)
    assert os.path.isfile(os.path.join(out, "CoefDrag.png"))


def test_plot_wall_integrals_rejects_all_and_empty(db):
    sets = CODASets(db)
    with pytest.raises(ValueError, match="non-empty list"):
        sets.plot_wall_integrals(cases_idx="all")
    with pytest.raises(ValueError, match="non-empty list"):
        sets.plot_wall_integrals(cases_idx=[])
    with pytest.raises(ValueError, match="stride"):
        sets.plot_wall_integrals(cases_idx=[0], stride=0)
    with pytest.raises(ValueError, match="x_axis"):
        sets.plot_wall_integrals(cases_idx=[0], x_axis="foo")


def test_plot_wall_integrals_missing_files_warn(tmp_path):
    df_cases = pd.DataFrame({"M": [0.3], "folder": ["M_0.3000"]})
    (tmp_path / "outputs" / "M_0.3000").mkdir(parents=True)
    sets = CODASets(_FakeDB(str(tmp_path), df_cases))
    with pytest.warns(UserWarning, match="No wall-integral data"):
        sets.plot_wall_integrals(cases_idx=[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
