"""
tests/test_numpy_reader.py
============================
Tests for the fixed ``NUMPYReader`` (``FotR/characters/readers/numpy.py``):

* the ``'Aux'`` bug in ``extract_outputs()`` (previously silently ignored);
* the new per-CADGroup ``subset`` mechanism, mirroring ``CODAReader``'s;
* continued compatibility with the existing ``cases_idx``-based API.

``readers/numpy.py`` is loaded in isolation (stubbing only
``FotR.characters.readers.base.BaseReader`` — the real one is pure-Python
and dependency-free, so it is loaded for real) exactly like the other test
modules in this suite, so no heavy optional dependency is required at all.

Run with::

    python -m pytest FotR/tests/test_numpy_reader.py -v
"""

import os
import sys
import types
import importlib.util

import numpy as np
import pytest


# =========================================================================
# Isolated import of NUMPYReader
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

    readers_pkg = mk('FotR.characters.readers')
    readers_pkg.__path__ = [os.path.join(root, 'characters', 'readers')]


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
_numpy_mod = _load_module(
    'FotR.characters.readers.numpy',
    os.path.join(_HERE, '..', 'characters', 'readers', 'numpy.py'),
)

NUMPYReader = _numpy_mod.NUMPYReader


# =========================================================================
# Fixture: a synthetic .npy file with one CADGroup, with/without Aux
# =========================================================================

def _make_group_dict(n_cases=6, npoints=4, n_stages=1, with_aux=True,
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

    gd = {
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
            aux[name] = (
                rng.random((npoints, n_cases)) if aux_has_case_axis
                else rng.random((npoints,))
            )
        gd['Aux'] = aux

    return gd


def _make_reader(tmp_path, group_dicts: dict, filename='data.npy', **kwargs):
    """group_dicts: {'CADGroup_3': gd, ...}"""
    path = tmp_path / filename
    np.save(path, group_dicts, allow_pickle=True)
    reader = NUMPYReader(root_dir=str(tmp_path), file=filename, **kwargs)
    reader.parse_simulation_dirs()
    return reader


# =========================================================================
# 1. Aux is read when present
# =========================================================================

def test_extract_outputs_reads_aux_when_present(tmp_path):
    gd = _make_group_dict(n_cases=5, with_aux=True)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})

    reader.extract_inputs(id_groups='3')
    reader.extract_outputs(stage=0, id_groups='3')

    assert 'Aux' in reader.data_dict['CADGroup_3']
    assert 'aux_var_0' in reader.data_dict['CADGroup_3']['Aux']
    np.testing.assert_allclose(
        reader.data_dict['CADGroup_3']['Aux']['aux_var_0'],
        gd['Aux']['aux_var_0'],
    )


def test_extract_outputs_no_aux_key_when_absent(tmp_path):
    gd = _make_group_dict(n_cases=5, with_aux=False)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})

    reader.extract_inputs(id_groups='3')
    reader.extract_outputs(stage=0, id_groups='3')

    assert 'Aux' not in reader.data_dict['CADGroup_3']


def test_extract_outputs_aux_multiple_vars(tmp_path):
    gd = _make_group_dict(n_cases=5, with_aux=True, n_aux_vars=3)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})

    reader.extract_inputs(id_groups='3')
    reader.extract_outputs(stage=0, id_groups='3')

    aux = reader.data_dict['CADGroup_3']['Aux']
    assert set(aux.keys()) == {'aux_var_0', 'aux_var_1', 'aux_var_2'}


def test_extract_outputs_aux_without_case_axis_kept_unfiltered(tmp_path):
    gd = _make_group_dict(n_cases=6, with_aux=True, aux_has_case_axis=False)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})

    reader.extract_inputs(id_groups='3', cases_idx=[0, 1])
    reader.extract_outputs(stage=0, id_groups='3')

    aux_arr = reader.data_dict['CADGroup_3']['Aux']['aux_var_0']
    assert aux_arr.shape == (4,)   # (n_points,), shared across cases


# =========================================================================
# 2. cases_idx propagation and consistency Vars <-> Aux
# =========================================================================

def test_extract_outputs_aux_matches_cases_idx_selection(tmp_path):
    gd = _make_group_dict(n_cases=8, with_aux=True, aux_has_case_axis=True)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})

    case_idx = [5, 1, 3]
    reader.extract_inputs(id_groups='3', cases_idx=case_idx)
    reader.extract_outputs(stage=0, id_groups='3')

    group_out = reader.data_dict['CADGroup_3']
    aux_arr = group_out['Aux']['aux_var_0']

    assert aux_arr.shape[-1] == 3
    np.testing.assert_allclose(aux_arr, gd['Aux']['aux_var_0'][:, case_idx])

    # Same case set/order as Vars and FlCc.
    assert group_out['FlCc'].shape[0] == 3
    assert group_out['Vars']['0']['Pressure'].shape[1] == 3
    np.testing.assert_allclose(group_out['FlCc'], gd['FlCc'][case_idx])


def test_extract_inputs_cases_idx_all(tmp_path):
    gd = _make_group_dict(n_cases=4)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.extract_inputs(id_groups='3', cases_idx='all')
    assert reader.data_dict['CADGroup_3']['FlCc'].shape[0] == 4


def test_extract_inputs_cases_idx_list(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.extract_inputs(id_groups='3', cases_idx=[0, 2, 4])
    np.testing.assert_allclose(
        reader.data_dict['CADGroup_3']['FlCc'], gd['FlCc'][[0, 2, 4]]
    )


def test_extract_inputs_cases_idx_int(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.extract_inputs(id_groups='3', cases_idx=3)
    assert reader.data_dict['CADGroup_3']['FlCc'].shape[0] == 1
    np.testing.assert_allclose(
        reader.data_dict['CADGroup_3']['FlCc'][0], gd['FlCc'][3]
    )


def test_extract_outputs_requires_extract_inputs_first(tmp_path):
    gd = _make_group_dict(n_cases=4)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    with pytest.raises(RuntimeError):
        reader.extract_outputs(stage=0, id_groups='3')


# =========================================================================
# 3. Subsets (new mechanism, mirroring CODAReader)
# =========================================================================

def test_define_subset_basic(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    idx = reader.define_subset(id_group='3', name='mach_07', cases_idx=[0, 1, 2])
    assert idx == [0, 1, 2]
    assert reader.get_subset('3', 'mach_07') == [0, 1, 2]


def test_define_subset_duplicate_raises(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[0, 1])
    with pytest.raises(ValueError):
        reader.define_subset(id_group='3', name='mach_07', cases_idx=[2, 3])


def test_define_subset_overwrite(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[0, 1])
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[2, 3],
                          overwrite=True)
    assert reader.get_subset('3', 'mach_07') == [2, 3]


def test_define_subset_out_of_range_raises(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    with pytest.raises(IndexError):
        reader.define_subset(id_group='3', name='bad', cases_idx=[0, 99])


def test_define_subset_empty_selection_raises(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    with pytest.raises(ValueError):
        reader.define_subset(id_group='3', name='empty', cases_idx=[])


def test_list_subsets_per_group_and_global(tmp_path):
    gd3 = _make_group_dict(n_cases=6)
    gd5 = _make_group_dict(n_cases=4)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd3, 'CADGroup_5': gd5})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[0, 1])
    reader.define_subset(id_group='5', name='low_aoa', cases_idx=[0])

    assert reader.list_subsets('3') == {'mach_07': [0, 1]}
    assert reader.list_subsets('5') == {'low_aoa': [0]}
    all_subs = reader.list_subsets()
    assert all_subs == {
        'CADGroup_3': {'mach_07': [0, 1]},
        'CADGroup_5': {'low_aoa': [0]},
    }


def test_subsets_are_independent_per_group(tmp_path):
    gd3 = _make_group_dict(n_cases=6)
    gd5 = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd3, 'CADGroup_5': gd5})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[0, 1])
    # Same subset name is free to mean something different in another group.
    reader.define_subset(id_group='5', name='mach_07', cases_idx=[4, 5])
    assert reader.get_subset('3', 'mach_07') == [0, 1]
    assert reader.get_subset('5', 'mach_07') == [4, 5]


def test_remove_subset(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[0, 1])
    reader.remove_subset('3', 'mach_07')
    with pytest.raises(KeyError):
        reader.get_subset('3', 'mach_07')


def test_remove_subset_missing_raises(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    with pytest.raises(KeyError):
        reader.remove_subset('3', 'ghost')


def test_get_subset_unknown_group_raises_keyerror(tmp_path):
    gd = _make_group_dict(n_cases=6)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    with pytest.raises(KeyError):
        reader.get_subset('does_not_exist', 'mach_07')


# =========================================================================
# 4. extract_inputs(subset=...) end-to-end, including Vars/Aux consistency
# =========================================================================

def test_extract_inputs_with_subset(tmp_path):
    gd = _make_group_dict(n_cases=8, with_aux=True, aux_has_case_axis=True)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})

    reader.define_subset(id_group='3', name='mach_07', cases_idx=[1, 4, 6])
    reader.extract_inputs(id_groups='3', subset='mach_07')
    reader.extract_outputs(stage=0, id_groups='3')

    group_out = reader.data_dict['CADGroup_3']
    assert group_out['FlCc'].shape[0] == 3
    np.testing.assert_allclose(group_out['FlCc'], gd['FlCc'][[1, 4, 6]])
    assert group_out['Vars']['0']['Pressure'].shape[1] == 3
    np.testing.assert_allclose(
        group_out['Aux']['aux_var_0'], gd['Aux']['aux_var_0'][:, [1, 4, 6]]
    )


def test_extract_inputs_subset_and_explicit_cases_idx_conflict_raises(tmp_path):
    gd = _make_group_dict(n_cases=8)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[1, 4, 6])
    with pytest.raises(ValueError):
        reader.extract_inputs(id_groups='3', cases_idx=[0, 1], subset='mach_07')


def test_extract_inputs_unknown_subset_raises_keyerror(tmp_path):
    gd = _make_group_dict(n_cases=8)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    with pytest.raises(KeyError):
        reader.extract_inputs(id_groups='3', subset='ghost')


def test_extract_inputs_cases_idx_still_works_after_subset_added(tmp_path):
    """Backward compatibility: introducing subsets must not change the
    default cases_idx-based behaviour."""
    gd = _make_group_dict(n_cases=8)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    reader.define_subset(id_group='3', name='mach_07', cases_idx=[1, 4, 6])
    reader.extract_inputs(id_groups='3', cases_idx=[0, 2])
    np.testing.assert_allclose(
        reader.data_dict['CADGroup_3']['FlCc'], gd['FlCc'][[0, 2]]
    )


# =========================================================================
# Direct unit tests of the shared _normalise_cases_idx / _resolve_cases_idx
# =========================================================================

def test_normalise_cases_idx_reused_by_define_subset_and_extract_inputs(tmp_path):
    gd = _make_group_dict(n_cases=5)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    assert NUMPYReader._normalise_cases_idx('all', n_cases=5) == [0, 1, 2, 3, 4]
    assert NUMPYReader._normalise_cases_idx([1, 2], n_cases=5) == [1, 2]
    with pytest.raises(IndexError):
        NUMPYReader._normalise_cases_idx([0, 99], n_cases=5)


def test_resolve_cases_idx_passthrough_without_subset(tmp_path):
    gd = _make_group_dict(n_cases=5)
    reader = _make_reader(tmp_path, {'CADGroup_3': gd})
    assert reader._resolve_cases_idx('3', cases_idx=[1, 2]) == [1, 2]
    assert reader._resolve_cases_idx('3', cases_idx='all') == [0, 1, 2, 3, 4]


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))