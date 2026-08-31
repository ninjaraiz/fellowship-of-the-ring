"""
tests/test_coda_subsets.py
===========================
Standalone tests for the CODAReader case-subset system
(``define_subset`` / ``get_subset`` / ``list_subsets`` / ``remove_subset``
and the central ``_resolve_cases_idx`` resolver), plus the shared
``BaseReader._normalise_cases_idx`` helper.

These tests deliberately avoid importing the full ``FotR`` package (which
pulls in heavy optional dependencies such as ``pyvista``, ``torch``,
``h5py`` and ``pyLOM`` via ``characters/sam.py``). Instead, they load
``readers/base.py`` and ``readers/coda.py`` directly as standalone
modules against a minimal stub of ``FotR.characters.sam.SAM`` and
``pyvista``, and construct a ``CODAReader`` instance by hand
(bypassing ``__init__``'s filesystem / JSON requirements) with a
synthetic ``df_cases``.

Run with::

    python -m pytest FotR/tests/test_coda_subsets.py -v

or directly::

    python FotR/tests/test_coda_subsets.py
"""

import sys
import types
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest


# =========================================================================
# Fixture: a CODAReader instance with a synthetic df_cases, no filesystem
# or heavy-dependency imports required.
# =========================================================================

def _install_stub_modules():
    """Install minimal stand-ins for pyvista and FotR.characters.sam.SAM
    so that readers/base.py and readers/coda.py can be imported in
    isolation, without touching the rest of the FotR package."""

    if 'pyvista' not in sys.modules:
        m = types.ModuleType('pyvista')

        class _Grid:
            pass

        m.UnstructuredGrid = _Grid
        m.read = lambda *a, **k: None
        m.CellType = types.SimpleNamespace(TRIANGLE=5, TETRA=10)
        sys.modules['pyvista'] = m

    if 'FotR' not in sys.modules:
        fotr_pkg = types.ModuleType('FotR')
        fotr_pkg.__path__ = [os.path.join(os.path.dirname(__file__), '..')]
        sys.modules['FotR'] = fotr_pkg

    if 'FotR.characters' not in sys.modules:
        chars_pkg = types.ModuleType('FotR.characters')
        chars_pkg.__path__ = [
            os.path.join(os.path.dirname(__file__), '..', 'characters')
        ]
        sys.modules['FotR.characters'] = chars_pkg

    if 'FotR.characters.sam' not in sys.modules:
        sam_mod = types.ModuleType('FotR.characters.sam')

        class _SAMStub:
            class Backpack:
                class pattern_pocket:
                    @staticmethod
                    def find_files(*a, **k):
                        return []

                    class FilenamePattern:
                        @classmethod
                        def from_template(cls, *a, **k):
                            import re
                            return types.SimpleNamespace(compiled=re.compile('.*'))

                @staticmethod
                def get_unified_connectivity(*a, **k):
                    return None

                @staticmethod
                def ensure_cell_data(m):
                    return m

                @staticmethod
                def same_columns(*a, **k):
                    return True

                @staticmethod
                def get_df_from_csv(*a, **k):
                    return pd.DataFrame()

            class Weapons:
                @staticmethod
                def sort_lexsort(points):
                    return points, np.arange(len(points))

                @staticmethod
                def sort_by_centroid(points):
                    return points, np.arange(len(points))

                @staticmethod
                def sort_closed_curve_by_kdtree(points, **k):
                    return points, np.arange(len(points))

                @staticmethod
                def sort_points_by_hull_projection(points, **k):
                    return points, np.arange(len(points))

        sam_mod.SAM = _SAMStub
        sys.modules['FotR.characters.sam'] = sam_mod

    if 'FotR.characters.readers' not in sys.modules:
        readers_pkg = types.ModuleType('FotR.characters.readers')
        readers_pkg.__path__ = [os.path.dirname(__file__) + '/../characters/readers']
        sys.modules['FotR.characters.readers'] = readers_pkg


def _load_module(dotted_name: str, path: str):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_install_stub_modules()

_HERE = os.path.dirname(__file__)
_base_mod = _load_module(
    'FotR.characters.readers.base',
    os.path.join(_HERE, '..', 'characters', 'readers', 'base.py'),
)
_coda_mod = _load_module(
    'FotR.characters.readers.coda',
    os.path.join(_HERE, '..', 'characters', 'readers', 'coda.py'),
)

BaseReader = _base_mod.BaseReader
CODAReader = _coda_mod.CODAReader


@pytest.fixture
def reader():
    """A bare CODAReader with a synthetic 6-case df_cases (2 Mach x 3 AoA)."""
    r = CODAReader.__new__(CODAReader)
    r.root_dir      = '/tmp/does_not_matter'
    r.sim_metadata  = {}
    r.df_state      = pd.DataFrame()
    r.data_dict     = {}
    r.subsets       = {}
    r.active_cases_idx = {}

    df_cases = pd.DataFrame({
        'case_idx': [0, 1, 2, 3, 4, 5],
        'AoA':  [0.0, 2.0, 4.0, 0.0, 2.0, 4.0],
        'Mach': [0.7, 0.7, 0.7, 0.8, 0.8, 0.8],
        'folder': [f'f{i}' for i in range(6)],
    })
    r.metadata = {
        'design_vars': ['AoA', 'Mach'],
        'df_cases':    df_cases,
        'num_stages':  1,
    }
    r._sync_subsets_column()
    return r


# =========================================================================
# define_subset
# =========================================================================

def test_define_subset_basic(reader):
    idx = reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    assert idx == [0, 1, 2]
    assert reader.subsets['mach_07'] == [0, 1, 2]


def test_define_subset_mirrors_to_df_cases_column(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    assert reader.metadata['df_cases']['subsets'].tolist()[0] == ['mach_07']
    assert reader.metadata['df_cases']['subsets'].tolist()[5] == []


def test_define_subset_duplicate_name_raises(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    with pytest.raises(ValueError):
        reader.define_subset(name='mach_07', cases_idx=[3, 4, 5])


def test_define_subset_overwrite(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    reader.define_subset(name='mach_07', cases_idx=[0, 1], overwrite=True)
    assert reader.subsets['mach_07'] == [0, 1]


def test_define_subset_multi_membership(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    reader.define_subset(name='low_aoa', cases_idx=[0, 3])
    assert sorted(reader.metadata['df_cases'].loc[0, 'subsets']) == [
        'low_aoa', 'mach_07',
    ]


def test_define_subset_out_of_range_raises(reader):
    with pytest.raises(IndexError):
        reader.define_subset(name='bad', cases_idx=[0, 99])


def test_define_subset_empty_name_raises(reader):
    with pytest.raises(ValueError):
        reader.define_subset(name='   ', cases_idx=[0])


def test_define_subset_empty_selection_raises(reader):
    with pytest.raises(ValueError):
        reader.define_subset(name='mach_09', cases_idx=[])


def test_define_subset_accepts_all(reader):
    idx = reader.define_subset(name='every_case', cases_idx='all')
    assert idx == list(range(6))


# =========================================================================
# get_subset / list_subsets / remove_subset
# =========================================================================

def test_get_subset_missing_raises(reader):
    with pytest.raises(KeyError):
        reader.get_subset('nope')


def test_get_subset_returns_copy(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    got = reader.get_subset('mach_07')
    got.append(999)
    assert reader.subsets['mach_07'] == [0, 1, 2]


def test_list_subsets_returns_independent_copies(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    snap = reader.list_subsets()
    snap['mach_07'].append(999)
    assert reader.subsets['mach_07'] == [0, 1, 2]


def test_remove_subset(reader):
    reader.define_subset(name='low_aoa', cases_idx=[0, 3])
    reader.remove_subset('low_aoa')
    assert 'low_aoa' not in reader.subsets
    assert reader.metadata['df_cases'].loc[3, 'subsets'] == []


def test_remove_subset_missing_raises(reader):
    with pytest.raises(KeyError):
        reader.remove_subset('ghost')


# =========================================================================
# _resolve_cases_idx (central resolution mechanism)
# =========================================================================

def test_resolve_cases_idx_passthrough_when_no_subset(reader):
    assert reader._resolve_cases_idx(cases_idx=[3, 4], subset=None) == [3, 4]
    assert reader._resolve_cases_idx(cases_idx='all', subset=None) == list(range(6))


def test_resolve_cases_idx_uses_subset_when_cases_idx_default(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    assert reader._resolve_cases_idx(cases_idx='all', subset='mach_07') == [0, 1, 2]


def test_resolve_cases_idx_conflict_raises(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    with pytest.raises(ValueError):
        reader._resolve_cases_idx(cases_idx=[5, 6], subset='mach_07')


def test_resolve_cases_idx_unknown_subset_raises(reader):
    with pytest.raises(KeyError):
        reader._resolve_cases_idx(cases_idx='all', subset='ghost')


# =========================================================================
# Subsets survive re-parsing (df_cases mutated in place, never replaced)
# =========================================================================

def test_subsets_survive_inplace_df_cases_mutation(reader):
    reader.define_subset(name='mach_07', cases_idx=[0, 1, 2])
    df_cases_id_before = id(reader.metadata['df_cases'])

    # Mirrors exactly what CODAReader.parse_simulation_dirs() does to
    # df_cases on every call: mutate existing rows in place, never
    # reassign self.metadata['df_cases'] to a new object.
    reader.metadata['df_cases'].at[0, 'folder'] = 'f0_updated'

    assert id(reader.metadata['df_cases']) == df_cases_id_before
    assert reader.subsets.get('mach_07') == [0, 1, 2]


# =========================================================================
# BaseReader._normalise_cases_idx (shared helper)
# =========================================================================

def test_base_normalise_cases_idx_all(reader):
    df_cases = reader.metadata['df_cases']
    assert BaseReader._normalise_cases_idx('all', df_cases) == list(range(6))


def test_base_normalise_cases_idx_invalid_string_raises(reader):
    df_cases = reader.metadata['df_cases']
    with pytest.raises(ValueError):
        BaseReader._normalise_cases_idx('bogus', df_cases)


def test_base_normalise_cases_idx_out_of_range_raises(reader):
    df_cases = reader.metadata['df_cases']
    with pytest.raises(IndexError):
        BaseReader._normalise_cases_idx([0, 99], df_cases)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))