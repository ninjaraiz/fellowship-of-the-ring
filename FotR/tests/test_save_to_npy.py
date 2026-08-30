"""
tests/test_save_to_npy.py
===========================
Tests for the fixed ``CODASets.save_to_npy()``.

``sets/coda.py`` is loaded in isolation (stubbing ``FotR.EarendilsLight``,
``FotR.characters.sam.SAM`` and ``FotR.characters.sets.base.BaseSets``)
exactly like the other test modules in this suite do for ``readers/coda.py``
and ``characters/frodo.py``, so the fix can be exercised without any of
FRODO's optional heavy dependencies beyond ``numpy``/``torch``/``h5py``
(which ``sets/coda.py`` itself imports at module level) and a trivial
``pyLOM`` stand-in (only imported, never actually used by ``save_to_npy``).

Run with::

    python -m pytest FotR/tests/test_save_to_npy.py -v
"""

import os
import sys
import types
import importlib.util

import numpy as np
import pytest


# =========================================================================
# Isolated import of CODASets
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
            """Minimal SAM stand-in: save_to_npy never touches SAM.Weapons
            or SAM.Gardener, so an (almost) empty stub is sufficient."""
            class Weapons:
                pass

            class Gardener:
                pass
        sam_mod.SAM = _SAM

    sets_pkg = mk('FotR.characters.sets')
    sets_pkg.__path__ = [os.path.join(root, 'characters', 'sets')]

    base_mod = mk('FotR.characters.sets.base')
    if not hasattr(base_mod, 'BaseSets'):
        class _BaseSets:
            def __init__(self, db):
                self.db = db
        base_mod.BaseSets = _BaseSets

    # Trivial pyLOM stand-in: sets/coda.py does `import pyLOM as SMEAGOL`
    # at module level (used by create_pylom_mesh / create_NN_pylom, never
    # by save_to_npy), so it only needs to exist, not do anything.
    if 'pyLOM' not in sys.modules:
        pylom_mod = types.ModuleType('pyLOM')

        class _Dataset:
            @staticmethod
            def load(*a, **k):
                return _Dataset()

        class _PartitionTable:
            @staticmethod
            def new(*a, **k):
                return _PartitionTable()

        class _Mesh:
            def __init__(self, *a, **k):
                pass

        pylom_mod.Dataset = _Dataset
        pylom_mod.PartitionTable = _PartitionTable
        pylom_mod.Mesh = _Mesh
        sys.modules['pyLOM'] = pylom_mod


def _load_codasets():
    _install_stub_modules()
    dotted = 'FotR.characters.sets.coda'
    if dotted in sys.modules:
        return sys.modules[dotted].CODASets

    root = os.path.join(os.path.dirname(__file__), '..')
    spec = importlib.util.spec_from_file_location(
        dotted, os.path.join(root, 'characters', 'sets', 'coda.py')
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod.CODASets


CODASets = _load_codasets()


# =========================================================================
# Fixture: a fake FRODO-like db with one synthetic CADGroup
# =========================================================================

class _FakeDB:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.format = 'CODA'
        self.name = 'fake_db'


def _make_group(n_cases=5, npoints=4, n_stages=1, with_aux=True,
                aux_has_case_axis=True, n_aux_vars=1):
    rng = np.random.default_rng(0)

    flcc = np.column_stack([
        np.arange(n_cases, dtype=float),
        0.7 * np.ones(n_cases),
    ])

    idx_sort = np.zeros((n_stages, n_cases, npoints), dtype=np.int32)
    for s in range(n_stages):
        for c in range(n_cases):
            idx_sort[s, c] = rng.permutation(npoints)

    group = {
        'Coord':     rng.random((npoints, 3)),
        'FlCc':      flcc,
        'Conec':     np.zeros((npoints, 3), dtype=np.int32),
        'idx_sort':  idx_sort,
        'eltype':    np.full(npoints, 5, dtype=np.int32),
        'cellOrder': np.arange(npoints, dtype=np.int32),
        'Vars': {
            str(s): {
                'Pressure': rng.random((npoints, n_cases)),
                'Velocity': rng.random((3, npoints, n_cases)),
            }
            for s in range(n_stages)
        },
    }

    if with_aux:
        aux = {}
        for i in range(n_aux_vars):
            name = f'aux_var_{i}'
            if aux_has_case_axis:
                aux[name] = rng.random((npoints, n_cases))
            else:
                aux[name] = rng.random((npoints,))
        group['Aux'] = aux

    return group


@pytest.fixture
def sets_obj():
    group = _make_group()
    db = _FakeDB({'CADGroup_3': group})
    return CODASets(db)


# =========================================================================
# 1-3. case_idx handling: 'all', list, single int
# =========================================================================

def test_save_to_npy_case_idx_all(sets_obj, tmp_path):
    out_path = str(tmp_path / 'out_all')
    sets_obj.save_to_npy(stage=0, id_group='3', filepath=out_path)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_3']
    n_cases_total = sets_obj.db.data_dict['CADGroup_3']['FlCc'].shape[0]

    assert g['FlCc'].shape[0] == n_cases_total
    assert g['idx_sort'].shape == (1, n_cases_total, 4)
    assert g['Vars']['0']['Pressure'].shape[0] == n_cases_total


def test_save_to_npy_case_idx_list(sets_obj, tmp_path):
    out_path = str(tmp_path / 'out_list')
    sets_obj.save_to_npy(stage=0, id_group='3', filepath=out_path,
                          case_idx=[1, 3])

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_3']

    assert g['FlCc'].shape[0] == 2
    np.testing.assert_allclose(
        g['FlCc'], sets_obj.db.data_dict['CADGroup_3']['FlCc'][[1, 3]]
    )
    assert g['idx_sort'].shape == (1, 2, 4)
    assert g['Vars']['0']['Pressure'].shape[0] == 2


def test_save_to_npy_case_idx_single_int(sets_obj, tmp_path):
    out_path = str(tmp_path / 'out_single')
    sets_obj.save_to_npy(stage=0, id_group='3', filepath=out_path,
                          case_idx=2)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_3']

    assert g['FlCc'].shape[0] == 1
    np.testing.assert_allclose(
        g['FlCc'][0], sets_obj.db.data_dict['CADGroup_3']['FlCc'][2]
    )


# =========================================================================
# 4-6. Aux export
# =========================================================================

def test_save_to_npy_exports_aux_when_present(sets_obj, tmp_path):
    out_path = str(tmp_path / 'out_aux')
    sets_obj.save_to_npy(stage=0, id_group='3', filepath=out_path)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_3']

    assert 'Aux' in g
    assert 'aux_var_0' in g['Aux']
    # Aux was polluting the top level before the fix; make sure it no
    # longer does.
    assert 'aux_var_0' not in g


def test_save_to_npy_exports_multiple_aux_vars(sets_obj, tmp_path):
    group = _make_group(n_aux_vars=3)
    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)

    out_path = str(tmp_path / 'out_aux_multi')
    sets.save_to_npy(stage=0, id_group='3', filepath=out_path)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    aux = loaded['CADGroup_3']['Aux']
    assert set(aux.keys()) == {'aux_var_0', 'aux_var_1', 'aux_var_2'}


def test_save_to_npy_no_aux_is_valid(tmp_path):
    group = _make_group(with_aux=False)
    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)

    out_path = str(tmp_path / 'out_no_aux')
    sets.save_to_npy(stage=0, id_group='3', filepath=out_path)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    assert 'Aux' not in loaded['CADGroup_3']


def test_save_to_npy_aux_without_case_axis_kept_unfiltered(tmp_path):
    group = _make_group(n_cases=5, aux_has_case_axis=False)
    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)

    out_path = str(tmp_path / 'out_aux_geom')
    sets.save_to_npy(stage=0, id_group='3', filepath=out_path, case_idx=[0, 1])

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    aux_arr = loaded['CADGroup_3']['Aux']['aux_var_0']
    # No case axis (shape (n_points,)) -> shared across cases, unfiltered.
    assert aux_arr.shape == (4,)


def test_save_to_npy_aux_with_case_axis_matches_case_idx(tmp_path):
    group = _make_group(n_cases=6, aux_has_case_axis=True)
    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)

    out_path = str(tmp_path / 'out_aux_cases')
    case_idx = [0, 2, 5]
    sets.save_to_npy(stage=0, id_group='3', filepath=out_path,
                      case_idx=case_idx)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_3']
    aux_arr = g['Aux']['aux_var_0']

    assert aux_arr.shape[-1] == len(case_idx)
    np.testing.assert_allclose(
        aux_arr, group['Aux']['aux_var_0'][:, case_idx]
    )
    # Same case set/order as everything else in the file.
    assert g['FlCc'].shape[0] == len(case_idx)
    assert g['Vars']['0']['Pressure'].shape[0] == len(case_idx)


# =========================================================================
# 7-8. Vars export and ignore_vars
# =========================================================================

def test_save_to_npy_exports_vars(sets_obj, tmp_path):
    out_path = str(tmp_path / 'out_vars')
    sets_obj.save_to_npy(stage=0, id_group='3', filepath=out_path)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    vars0 = loaded['CADGroup_3']['Vars']['0']

    assert set(vars0.keys()) == {'Pressure', 'Velocity'}
    # Scalar var: (n_points, n_cases) -> transposed to (n_cases, n_points).
    assert vars0['Pressure'].shape == (5, 4)
    # Vector var: (n_dim, n_points, n_cases), sliced on the case axis only.
    assert vars0['Velocity'].shape == (3, 4, 5)


def test_save_to_npy_ignore_vars(sets_obj, tmp_path):
    out_path = str(tmp_path / 'out_ignore')
    sets_obj.save_to_npy(stage=0, id_group='3', filepath=out_path,
                          ignore_vars=['Velocity'])

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    vars0 = loaded['CADGroup_3']['Vars']['0']
    assert 'Velocity' not in vars0
    assert 'Pressure' in vars0


# =========================================================================
# 9. Different stages, including the int/str equivalence
# =========================================================================

def test_save_to_npy_multiple_stages(tmp_path):
    group = _make_group(n_stages=2)
    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)

    for stage in (0, 1):
        out_path = str(tmp_path / f'out_stage_{stage}')
        sets.save_to_npy(stage=stage, id_group='3', filepath=out_path)
        loaded = np.load(out_path + '.npy', allow_pickle=True).item()
        assert str(stage) in loaded['CADGroup_3']['Vars']


def test_save_to_npy_invalid_stage_out_of_bounds_raises(sets_obj, tmp_path):
    with pytest.raises(IndexError):
        sets_obj.save_to_npy(
            stage=99, id_group='3', filepath=str(tmp_path / 'bad_stage'),
        )


def test_save_to_npy_non_numeric_stage_raises_clear_error(sets_obj, tmp_path):
    with pytest.raises(ValueError):
        sets_obj.save_to_npy(
            stage='not_a_number', id_group='3',
            filepath=str(tmp_path / 'bad_stage2'),
        )


# =========================================================================
# 10. Cross-structure case count/order consistency
# =========================================================================

def test_save_to_npy_case_order_consistent_across_structures(tmp_path):
    group = _make_group(n_cases=6, aux_has_case_axis=True, n_aux_vars=2)
    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)

    case_idx = [4, 0, 3]  # deliberately out of order
    out_path = str(tmp_path / 'out_order')
    sets.save_to_npy(stage=0, id_group='3', filepath=out_path,
                      case_idx=case_idx)

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_3']

    assert g['FlCc'].shape[0] == 3
    assert g['idx_sort'].shape[1] == 3
    assert g['Vars']['0']['Pressure'].shape[0] == 3
    assert g['Vars']['0']['Velocity'].shape[-1] == 3
    for aux_arr in g['Aux'].values():
        assert aux_arr.shape[-1] == 3

    np.testing.assert_allclose(g['FlCc'], group['FlCc'][case_idx])
    np.testing.assert_allclose(
        g['Vars']['0']['Pressure'], group['Vars']['0']['Pressure'][:, case_idx].T
    )
    np.testing.assert_allclose(
        g['Aux']['aux_var_0'], group['Aux']['aux_var_0'][:, case_idx]
    )

    # idx_sort/eltype/cellOrder use the exact same case order as FlCc/Vars.
    for i, c in enumerate(case_idx):
        np.testing.assert_array_equal(
            g['idx_sort'][0, i], group['idx_sort'][0, c]
        )


# =========================================================================
# Reproduction of the reported IndexError (stage passed as a string,
# case_idx='all')
# =========================================================================

def test_save_to_npy_reproduces_and_fixes_reported_indexerror(tmp_path):
    """
    This is a direct reproduction of the bug report:

        db.sets.save_to_npy(
            stage='0', id_group='4', filepath='./CAD_3_stage_1.npy',
            case_idx='all', ignore_vars=None, verbose=True,
        )

    which used to raise:

        IndexError: only integers, slices (`:`), ellipsis (`...`),
        numpy.newaxis (`None`) and integer or boolean arrays are valid
        indices

    because `stage='0'` (a string) was used directly as a positional
    NumPy index into `idx_sort`. It must now complete successfully.
    """
    group = _make_group(n_cases=4)
    db = _FakeDB({'CADGroup_4': group})
    sets = CODASets(db)

    out_path = str(tmp_path / 'CAD_3_stage_1')
    sets.save_to_npy(
        stage='0', id_group='4', filepath=out_path,
        case_idx='all', ignore_vars=None, verbose=True,
    )

    loaded = np.load(out_path + '.npy', allow_pickle=True).item()
    g = loaded['CADGroup_4']
    assert g['FlCc'].shape[0] == 4
    assert '0' in g['Vars']


# =========================================================================
# Explicit, clear errors instead of a bare IndexError further downstream
# =========================================================================

def test_save_to_npy_out_of_range_case_idx_raises_clear_indexerror(sets_obj, tmp_path):
    with pytest.raises(IndexError, match='out-of-range'):
        sets_obj.save_to_npy(
            stage=0, id_group='3', filepath=str(tmp_path / 'bad_case'),
            case_idx=[0, 999],
        )


def test_save_to_npy_invalid_case_idx_string_raises(sets_obj, tmp_path):
    with pytest.raises(ValueError):
        sets_obj.save_to_npy(
            stage=0, id_group='3', filepath=str(tmp_path / 'bad_str'),
            case_idx='everything',
        )


def test_save_to_npy_missing_group_raises_keyerror(sets_obj, tmp_path):
    with pytest.raises(KeyError):
        sets_obj.save_to_npy(
            stage=0, id_group='999', filepath=str(tmp_path / 'missing'),
        )


def test_save_to_npy_missing_stage_in_vars_only_raises_keyerror():
    group = _make_group(n_stages=1)
    # Artificially widen idx_sort's stage axis without adding the matching
    # Vars entry, to isolate the "valid idx_sort stage, but Vars lacks it"
    # case from the "idx_sort itself is out of bounds" case.
    extra_stage = np.zeros_like(group['idx_sort'])
    group['idx_sort'] = np.concatenate([group['idx_sort'], extra_stage], axis=0)

    db = _FakeDB({'CADGroup_3': group})
    sets = CODASets(db)
    with pytest.raises(KeyError):
        sets.save_to_npy(stage=1, id_group='3', filepath='/tmp/whatever_unused')


# =========================================================================
# Direct unit tests of the shared case-selection helper
# =========================================================================

def test_normalise_cases_idx_all():
    assert CODASets._normalise_cases_idx('all', n_cases=5) == [0, 1, 2, 3, 4]


def test_normalise_cases_idx_list():
    assert CODASets._normalise_cases_idx([1, 3], n_cases=5) == [1, 3]


def test_normalise_cases_idx_int():
    assert CODASets._normalise_cases_idx(2, n_cases=5) == [2]


def test_normalise_cases_idx_range():
    assert CODASets._normalise_cases_idx(range(0, 3), n_cases=5) == [0, 1, 2]


def test_normalise_cases_idx_out_of_range_raises():
    with pytest.raises(IndexError):
        CODASets._normalise_cases_idx([0, 99], n_cases=5)


def test_normalise_cases_idx_bad_string_raises():
    with pytest.raises(ValueError):
        CODASets._normalise_cases_idx('everything', n_cases=5)


def test_normalise_cases_idx_bad_type_raises():
    with pytest.raises(ValueError):
        CODASets._normalise_cases_idx(3.5, n_cases=5)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))