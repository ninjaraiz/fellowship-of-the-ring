"""
tests/test_merge_datasets.py
==============================
Tests for the restructured ``FRODO.merge_datasets()``.

These tests load ``characters/frodo.py`` in isolation (stubbing out
``EarendilsLight`` and the ``readers`` / ``sets`` / ``residuals`` / ``stats``
registries, exactly like ``test_coda_subsets.py`` does for
``readers/coda.py``) so that the merge logic can be exercised without any of
FRODO's heavy optional dependencies (pyvista, torch, h5py, pyLOM, …).

``merge_datasets`` is a ``@staticmethod`` whose real collaborators are only
ever accessed through two attributes of each source object:

* ``db.sets.interpolate_msh2msh(...)`` — mesh homogenisation.
* ``db.residuals.get_df_metrics(...)`` — CODA-only post-processing metrics.

Both are therefore replaced with small fakes that are correct with respect
to the *case bookkeeping* contract merge_datasets relies on (they preserve
``FlCc`` and design-variable identity exactly), without needing real
geometry, mesh files or on-disk residual monitors. This keeps the tests
focused on what this refactor is actually about: making sure
``FlCc`` / ``df_cases`` / ``df_state`` / ``df_post`` stay aligned and that
no case is silently dropped, duplicated, or fabricated.

Run with::

    python -m pytest FotR/tests/test_merge_datasets.py -v
"""

import os
import sys
import types
import importlib.util

import numpy as np
import pandas as pd
import pytest


# =========================================================================
# Isolated import of FRODO (see module docstring)
# =========================================================================

def _install_stub_modules():
    root = os.path.join(os.path.dirname(__file__), '..')

    def mk(name):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    fotr = mk('FotR')
    fotr.__path__ = [root]
    chars = mk('FotR.characters')
    chars.__path__ = [os.path.join(root, 'characters')]

    el = mk('FotR.EarendilsLight')
    if not hasattr(el, 'EarendilsLight'):
        class _EL:
            def __init__(self, *a, **k):
                pass

            def help(self, *a, **k):
                pass
        el.EarendilsLight = _EL

    sam_mod = mk('FotR.characters.sam')
    if not hasattr(sam_mod, 'SAM'):
        class _SAM:
            light = sys.modules['FotR.EarendilsLight'].EarendilsLight('x')

            class DictVisualizer:
                @staticmethod
                def rich_tree(d):
                    pass
        sam_mod.SAM = _SAM

    # A trivial reader/sets/residuals class, registered for every format
    # this test suite exercises ('CODA', 'NUMPY'). FRODO._set_subclasses()
    # is called at the very end of merge_datasets() (to wire up the new
    # merged instance), so the registries need *something* resolvable —
    # its behaviour beyond construction is irrelevant to these tests, since
    # merge_datasets() only ever talks to sources' *own* db.sets /
    # db.residuals (the FakeSets / FakeResiduals below), never to the
    # merged instance's freshly (re)built ones.
    class _TrivialReader:
        def __init__(self, root_dir, **kwargs):
            self.root_dir = root_dir

    class _TrivialSets:
        def __init__(self, db):
            self.db = db

    class _TrivialResiduals:
        def __init__(self, db):
            self.db = db

    readers_mod = mk('FotR.characters.readers')
    readers_mod.READER_REGISTRY = getattr(readers_mod, 'READER_REGISTRY', {})
    readers_mod.BaseReader = getattr(readers_mod, 'BaseReader', object)
    for fmt in ('CODA', 'NUMPY'):
        readers_mod.READER_REGISTRY.setdefault(fmt, _TrivialReader)

    sets_mod = mk('FotR.characters.sets')
    sets_mod.SETS_REGISTRY = getattr(sets_mod, 'SETS_REGISTRY', {})
    sets_mod.BaseSets = getattr(sets_mod, 'BaseSets', object)
    for fmt in ('CODA', 'NUMPY'):
        sets_mod.SETS_REGISTRY.setdefault(fmt, _TrivialSets)

    residuals_mod = mk('FotR.characters.residuals')
    residuals_mod.RESIDUALS_REGISTRY = getattr(residuals_mod, 'RESIDUALS_REGISTRY', {})
    residuals_mod.BaseResiduals = getattr(residuals_mod, 'BaseResiduals', object)
    for fmt in ('CODA', 'NUMPY'):
        residuals_mod.RESIDUALS_REGISTRY.setdefault(fmt, _TrivialResiduals)

    stats_mod = mk('FotR.characters.stats')
    stats_mod.STATS_REGISTRY = getattr(stats_mod, 'STATS_REGISTRY', {})
    stats_mod.BaseStats = getattr(stats_mod, 'BaseStats', object)


def _load_frodo():
    _install_stub_modules()
    dotted = 'FotR.characters.frodo'
    if dotted in sys.modules:
        return sys.modules[dotted].FRODO

    root = os.path.join(os.path.dirname(__file__), '..')
    spec = importlib.util.spec_from_file_location(
        dotted, os.path.join(root, 'characters', 'frodo.py')
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod.FRODO


FRODO = _load_frodo()


# =========================================================================
# Fake source database
# =========================================================================

class _FakeSets:
    """Fake ``db.sets`` exposing only ``interpolate_msh2msh``.

    Unlike the real CODASets/NUMPYSets implementation, this does not
    perform any actual geometric interpolation — it simply copies the
    source group (mesh, FlCc and Vars) verbatim under the new group id.
    This is sufficient and correct for testing the *case bookkeeping*
    contract of merge_datasets, which never inspects mesh coordinates.
    """

    def __init__(self, db):
        self.db = db

    def interpolate_msh2msh(self, id_group_src, new_group_id, new_mesh,
                             vars='all', method='idw', k=4):
        src = self.db.data_dict[f'CADGroup_{id_group_src}']
        new_key = f'CADGroup_{new_group_id}'
        self.db.data_dict[new_key] = {
            'Coord': new_mesh['Coord'].copy(),
            'FlCc':  src['FlCc'].copy(),
            'Vars':  {
                stage: {k: v.copy() for k, v in stage_vars.items()}
                for stage, stage_vars in src.get('Vars', {}).items()
            },
        }


class _FakeResiduals:
    """Fake ``db.residuals`` exposing only ``get_df_metrics``.

    Returns a DataFrame keyed on the (lower-cased) design variables plus a
    dummy metric column, built directly from ``db.df_state`` — mirroring
    the real ``CODAResiduals.get_df_metrics`` contract of returning one row
    per case *defined* in ``df_state`` (i.e. potentially more cases than
    were actually extracted), which is exactly the behaviour
    ``merge_datasets`` must now filter down via FlCc-identity matching.
    """

    def __init__(self, db):
        self.db = db

    def get_df_metrics(self, var_metrics=None, **kwargs):
        df = self.db.df_state.copy()
        df.columns = [c.lower() for c in df.columns]
        df['dummy_metric_mean'] = np.arange(len(df), dtype=float)
        return df


class FakeFRODO:
    """Minimal duck-typed stand-in for a real FRODO instance, exposing
    exactly the attributes/methods ``FRODO.merge_datasets`` touches."""

    def __init__(self, format, name, design_vars, data_dict, df_state=None,
                 metadata_extra=None):
        self.format  = format
        self.name    = name
        self.data_dict    = data_dict
        self.df_state     = df_state if df_state is not None else pd.DataFrame()
        self.sim_metadata = {}
        self.metadata   = {'design_vars': design_vars}
        if metadata_extra:
            self.metadata.update(metadata_extra)
        self.sets      = _FakeSets(self)
        self.residuals = _FakeResiduals(self)


def _make_group(flcc, npoints=5, stages=('0',), var_names=('Pressure',)):
    """Build a synthetic CADGroup dict: Coord + FlCc + Vars."""
    n_cases = flcc.shape[0]
    coord = np.random.default_rng(0).random((npoints, 3))
    group = {'Coord': coord, 'FlCc': flcc, 'Vars': {}}
    for s in stages:
        group['Vars'][s] = {
            vn: np.random.default_rng(1).random((npoints, n_cases))
            for vn in var_names
        }
    return group


def _make_df_state(flcc, design_vars, extra_case_rows=0, stage_value=1):
    """Build a df_state DataFrame that may contain *more* rows than FlCc
    (to simulate cases defined but never extracted)."""
    n = flcc.shape[0]
    all_flcc = flcc
    if extra_case_rows:
        rng = np.random.default_rng(42)
        # Extra rows with design-var values guaranteed not to collide with
        # the real ones (offset well outside the used range).
        extra = 1000.0 + rng.random((extra_case_rows, flcc.shape[1]))
        all_flcc = np.vstack([flcc, extra])
    df = pd.DataFrame(all_flcc, columns=design_vars)
    df['stage'] = stage_value
    df['case_idx'] = np.arange(len(df), dtype=np.int32)
    df['folder'] = [f'folder_{i}' for i in range(len(df))]
    return df


# =========================================================================
# 1. Normal case: two disjoint, fully-extracted sources
# =========================================================================

def test_merge_normal_case_flcc_df_cases_df_state_aligned(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7], [2.0, 0.7]])
    flcc_b = np.array([[4.0, 0.7], [0.0, 0.8]])

    db_a = FakeFRODO(
        'NUMPY', 'db_a', design_vars,
        {'CADGroup_3': _make_group(flcc_a)},
        df_state=_make_df_state(flcc_a, design_vars),
    )
    db_b = FakeFRODO(
        'NUMPY', 'db_b', design_vars,
        {'CADGroup_3': _make_group(flcc_b)},
        df_state=_make_df_state(flcc_b, design_vars),
    )

    merged = FRODO.merge_datasets(
        root_dir=str(tmp_path / 'merged'),
        name='merged',
        sources=[(db_a, '3'), (db_b, '3')],
        new_group_id='3_merged',
        mesh_ref=0,
    )

    flcc_merged = merged.data_dict['CADGroup_3_merged']['FlCc']
    df_cases    = merged.metadata['df_cases']
    df_state    = merged.df_state

    assert flcc_merged.shape[0] == 4
    assert len(df_cases) == 4
    assert len(df_state) == 4

    np.testing.assert_allclose(df_cases[design_vars].to_numpy(), flcc_merged)
    np.testing.assert_allclose(df_state[design_vars].to_numpy(), flcc_merged)
    assert list(df_cases['case_idx']) == [0, 1, 2, 3]


# =========================================================================
# 2. Sources with cases defined but never extracted (reproduces the bug)
# =========================================================================

def test_merge_excludes_cases_not_present_in_flcc(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7], [2.0, 0.7]])   # only 2 cases extracted
    flcc_b = np.array([[4.0, 0.7]])                # only 1 case extracted

    # df_state of each source describes MORE cases than were extracted.
    df_state_a = _make_df_state(flcc_a, design_vars, extra_case_rows=3)
    df_state_b = _make_df_state(flcc_b, design_vars, extra_case_rows=5)

    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)}, df_state=df_state_a)
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)}, df_state=df_state_b)

    assert len(df_state_a) == 5   # 2 extracted + 3 extra
    assert len(df_state_b) == 6   # 1 extracted + 5 extra

    merged = FRODO.merge_datasets(
        root_dir=str(tmp_path / 'merged'),
        name='merged',
        sources=[(db_a, '3'), (db_b, '3')],
        new_group_id='3_merged',
        mesh_ref=0,
    )

    # The bug this replaces: previously df_state/df_cases could end up with
    # 5 + 6 = 11 rows (every defined case) instead of 2 + 1 = 3 (only the
    # extracted ones).
    assert merged.data_dict['CADGroup_3_merged']['FlCc'].shape[0] == 3
    assert len(merged.metadata['df_cases']) == 3
    assert len(merged.df_state) == 3

    np.testing.assert_allclose(
        merged.df_state[design_vars].to_numpy(),
        merged.data_dict['CADGroup_3_merged']['FlCc'],
    )


# =========================================================================
# 3. Different, disjoint case subsets extracted per source
# =========================================================================

def test_merge_different_subsets_extracted(tmp_path):
    design_vars = ['AoA', 'Mach']
    # DB1 conceptually "owns" positions [0, 2, 4] of some larger grid,
    # DB2 "owns" positions [1, 3] — represented directly by their FlCc.
    flcc_a = np.array([[0.0, 0.7], [4.0, 0.7], [8.0, 0.7]])   # 3 cases
    flcc_b = np.array([[2.0, 0.7], [6.0, 0.7]])                # 2 cases

    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_1': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_1': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    merged = FRODO.merge_datasets(
        root_dir=str(tmp_path / 'merged'),
        name='merged',
        sources=[(db_a, '1'), (db_b, '1')],
        new_group_id='1_merged',
        mesh_ref=0,
    )

    flcc_merged = merged.data_dict['CADGroup_1_merged']['FlCc']
    assert flcc_merged.shape[0] == 5

    expected_aoa = sorted([0.0, 4.0, 8.0, 2.0, 6.0])
    assert sorted(flcc_merged[:, 0].tolist()) == expected_aoa
    assert len(merged.metadata['df_cases']) == 5
    assert len(merged.df_state) == 5


# =========================================================================
# 4. Duplicate case across sources -> error, not silent dedup
# =========================================================================

def test_merge_duplicate_case_raises(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7], [2.0, 0.7]])
    flcc_b = np.array([[2.0, 0.7], [4.0, 0.7]])   # (2.0, 0.7) collides with A

    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_1': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_1': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    with pytest.raises(ValueError, match='Duplicate case'):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'),
            name='merged',
            sources=[(db_a, '1'), (db_b, '1')],
            new_group_id='1_merged',
            mesh_ref=0,
        )


# =========================================================================
# 5. Row order consistency across FlCc / df_cases / df_state / df_post
# =========================================================================

def test_merge_row_order_consistent_across_structures(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[9.0, 0.6], [1.0, 0.6], [5.0, 0.6]])
    flcc_b = np.array([[3.0, 0.9]])

    db_a = FakeFRODO('CODA', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('CODA', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    merged = FRODO.merge_datasets(
        root_dir=str(tmp_path / 'merged'),
        name='merged',
        sources=[(db_a, '3'), (db_b, '3')],
        new_group_id='3_merged',
        mesh_ref=0,
        get_df_metrics_attr={'var_metrics': ['dummy']},
    )

    flcc_merged = merged.data_dict['CADGroup_3_merged']['FlCc']
    df_cases = merged.metadata['df_cases']
    df_state = merged.df_state
    df_post  = pd.read_csv(
        os.path.join(str(tmp_path / 'merged'), 'metadata', 'df_post.csv')
    )

    np.testing.assert_allclose(df_cases[design_vars].to_numpy(), flcc_merged)
    np.testing.assert_allclose(df_state[design_vars].to_numpy(), flcc_merged)
    np.testing.assert_allclose(
        df_post[[v.lower() for v in design_vars]].to_numpy(), flcc_merged
    )


# =========================================================================
# 6. CODA: df_post matches exactly the extracted cases
# =========================================================================

def test_merge_coda_df_post_matches_extracted_cases(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7], [2.0, 0.7]])          # 2 extracted
    df_state_a = _make_df_state(flcc_a, design_vars, extra_case_rows=4)  # 6 defined

    flcc_b = np.array([[4.0, 0.7]])
    df_state_b = _make_df_state(flcc_b, design_vars, extra_case_rows=2)  # 3 defined

    db_a = FakeFRODO('CODA', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)}, df_state=df_state_a)
    db_b = FakeFRODO('CODA', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)}, df_state=df_state_b)

    merged = FRODO.merge_datasets(
        root_dir=str(tmp_path / 'merged'),
        name='merged',
        sources=[(db_a, '3'), (db_b, '3')],
        new_group_id='3_merged',
        mesh_ref=0,
        get_df_metrics_attr={'var_metrics': ['dummy']},
    )

    n_extracted = 3  # 2 + 1, NOT 6 + 3
    flcc_merged = merged.data_dict['CADGroup_3_merged']['FlCc']
    df_post = pd.read_csv(
        os.path.join(str(tmp_path / 'merged'), 'metadata', 'df_post.csv')
    )

    assert flcc_merged.shape[0] == n_extracted
    assert len(df_post) == n_extracted
    assert len(merged.metadata['df_cases']) == n_extracted
    assert len(merged.df_state) == n_extracted


def test_merge_coda_requires_get_df_metrics_attr(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7]])
    flcc_b = np.array([[2.0, 0.7]])
    db_a = FakeFRODO('CODA', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('CODA', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    with pytest.raises(ValueError, match='get_df_metrics_attr'):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'),
            name='merged',
            sources=[(db_a, '3'), (db_b, '3')],
            new_group_id='3_merged',
            mesh_ref=0,
            # get_df_metrics_attr intentionally omitted
        )


# =========================================================================
# 7. Non-CODA format: works without get_df_metrics_attr, no df_post forced
# =========================================================================

def test_merge_non_coda_format_no_df_post_required(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7]])
    flcc_b = np.array([[2.0, 0.7]])
    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    merged = FRODO.merge_datasets(
        root_dir=str(tmp_path / 'merged'),
        name='merged',
        sources=[(db_a, '3'), (db_b, '3')],
        new_group_id='3_merged',
        mesh_ref=0,
    )

    assert merged.df_state is not None
    assert len(merged.df_state) == 2
    assert not os.path.exists(
        os.path.join(str(tmp_path / 'merged'), 'metadata', 'df_post.csv')
    )


def test_merge_non_coda_rejects_get_df_metrics_attr(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7]])
    flcc_b = np.array([[2.0, 0.7]])
    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    with pytest.raises(ValueError, match='only supported for format'):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'),
            name='merged',
            sources=[(db_a, '3'), (db_b, '3')],
            new_group_id='3_merged',
            mesh_ref=0,
            get_df_metrics_attr={'var_metrics': ['dummy']},
        )


# =========================================================================
# 8. Basic input validation
# =========================================================================

def test_merge_requires_at_least_two_sources(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7]])
    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))

    with pytest.raises(ValueError):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'), name='merged',
            sources=[(db_a, '3')], new_group_id='3_merged',
        )


def test_merge_requires_same_format(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc = np.array([[0.0, 0.7]])
    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc)},
                      df_state=_make_df_state(flcc, design_vars))
    db_b = FakeFRODO('CODA', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc + 1)},
                      df_state=_make_df_state(flcc + 1, design_vars))

    with pytest.raises(ValueError, match='same format'):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'), name='merged',
            sources=[(db_a, '3'), (db_b, '3')], new_group_id='3_merged',
        )


def test_merge_invalid_mesh_ref(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7]])
    flcc_b = np.array([[2.0, 0.7]])
    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    with pytest.raises(ValueError, match='mesh_ref'):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'), name='merged',
            sources=[(db_a, '3'), (db_b, '3')], new_group_id='3_merged',
            mesh_ref=5,
        )


def test_merge_missing_group_raises_keyerror(tmp_path):
    design_vars = ['AoA', 'Mach']
    flcc_a = np.array([[0.0, 0.7]])
    flcc_b = np.array([[2.0, 0.7]])
    db_a = FakeFRODO('NUMPY', 'db_a', design_vars,
                      {'CADGroup_3': _make_group(flcc_a)},
                      df_state=_make_df_state(flcc_a, design_vars))
    db_b = FakeFRODO('NUMPY', 'db_b', design_vars,
                      {'CADGroup_3': _make_group(flcc_b)},
                      df_state=_make_df_state(flcc_b, design_vars))

    with pytest.raises(KeyError):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'), name='merged',
            sources=[(db_a, '3'), (db_b, '999')], new_group_id='3_merged',
            mesh_ref=0,
        )


def test_merge_incompatible_flcc_columns_raises(tmp_path):
    db_a = FakeFRODO('NUMPY', 'db_a', ['AoA', 'Mach'],
                      {'CADGroup_3': _make_group(np.array([[0.0, 0.7]]))},
                      df_state=_make_df_state(np.array([[0.0, 0.7]]), ['AoA', 'Mach']))
    db_b = FakeFRODO('NUMPY', 'db_b', ['AoA', 'Mach', 'Beta'],
                      {'CADGroup_3': _make_group(np.array([[2.0, 0.7, 1.0]]))},
                      df_state=_make_df_state(
                          np.array([[2.0, 0.7, 1.0]]), ['AoA', 'Mach', 'Beta']
                      ))

    with pytest.raises(ValueError, match='incompatible column counts'):
        FRODO.merge_datasets(
            root_dir=str(tmp_path / 'merged'), name='merged',
            sources=[(db_a, '3'), (db_b, '3')], new_group_id='3_merged',
            mesh_ref=0,
        )


# =========================================================================
# 9. Direct unit tests of the pure helper functions
# =========================================================================

def test_check_no_duplicate_cases_passes_for_disjoint_sets():
    FRODO._check_no_duplicate_cases(
        [np.array([[0.0, 0.7]]), np.array([[2.0, 0.7]])],
        decimals=8, source_labels=['s0', 's1'],
    )  # should not raise


def test_check_no_duplicate_cases_raises_for_overlap():
    with pytest.raises(ValueError):
        FRODO._check_no_duplicate_cases(
            [np.array([[2.0, 0.7]]), np.array([[2.0, 0.7]])],
            decimals=8, source_labels=['s0', 's1'],
        )


def test_match_rows_by_identity_ambiguous_raises():
    flcc = np.array([[2.0, 0.7]])
    ref  = pd.DataFrame({
        'AoA':  [2.0, 2.0],
        'Mach': [0.7, 0.7],
    })
    with pytest.raises(RuntimeError, match='cannot be unambiguously identified'):
        FRODO._match_rows_by_identity(
            flcc, ['AoA', 'Mach'], ref, decimals=8, context='test',
        )


def test_match_rows_by_identity_missing_case_raises():
    flcc = np.array([[9.0, 0.7]])
    ref  = pd.DataFrame({'AoA': [2.0], 'Mach': [0.7]})
    with pytest.raises(RuntimeError, match='no matching row'):
        FRODO._match_rows_by_identity(
            flcc, ['AoA', 'Mach'], ref, decimals=8, context='test',
        )


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))