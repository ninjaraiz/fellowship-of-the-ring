"""Tests for the rewritten Eärendil's Light.

The previous implementation had none, and crashed with ``TypeError`` on
every query that found a docstring. These cover the two things it now
exists for — resolving a live delegation chain, and finding a name
without importing the package — plus the promise that importing it costs
nothing.
"""

import importlib
import inspect
import subprocess
import sys

import pytest

from FotR.EarendilsLight import EarendilsLight, _is_opaque, _name_index, _signature


# ── fakes: a delegation chain without any of FRODO's dependencies ───────


class _Reader:
    def extract_inputs(self, id_groups, vtu_type='surface'):
        """Read the mesh for one CAD group."""

    def only_on_reader(self):
        """Lives on the reader alone."""


class _Sets:
    def shared(self):
        """The sets version, which must win."""


class _ReaderWithShared(_Reader):
    def shared(self):
        """The reader version, which must lose."""


class _Fake:
    """Stands in for FRODO: same delegation contract, no imports."""

    _LIGHT_DELEGATES = ('sets', 'reader', 'residuals', 'stats')
    light = EarendilsLight(__name__)
    some_light = light.shortcut()

    def __init__(self, sets=None, reader=None):
        self.format = 'CODA'
        self.sets = sets
        self.reader = reader
        self.residuals = None
        self.stats = None

    def extract_inputs(self, *args, **kwargs):
        """Forwards to the reader."""


@pytest.fixture
def db():
    return _Fake(sets=_Sets(), reader=_ReaderWithShared())


# ── live resolution ─────────────────────────────────────────────────────


def test_live_names_the_class_actually_reached(db):
    out = db.some_light('extract_inputs', show=False)
    assert '_ReaderWithShared.extract_inputs' in out
    assert 'reached through .reader' in out
    assert "format 'CODA'" in out


def test_live_shows_the_real_signature_not_the_forwarding_one(db):
    out = db.some_light('extract_inputs', show=False)
    signature = next(l for l in out.splitlines() if 'extract_inputs(' in l)
    assert 'id_groups' in signature and 'vtu_type' in signature
    assert '*args' not in signature


def test_live_warns_that_the_owner_only_forwards(db):
    out = db.some_light('extract_inputs', show=False)
    assert 'only forwards' in out


def test_live_follows_the_declared_order(db):
    """sets comes before reader, exactly as FRODO.__getattr__ does."""
    out = db.some_light('shared', show=False)
    assert '_Sets.shared' in out
    assert 'reached through .sets' in out


def test_live_skips_empty_slots():
    out = _Fake(sets=None, reader=_Reader()).some_light('only_on_reader', show=False)
    assert '_Reader.only_on_reader' in out


def test_live_falls_back_to_the_index_when_the_chain_lacks_it(db):
    """A name no sub-object has must still be looked up statically."""
    out = db.some_light('merge_datasets', show=False)
    assert 'FRODO.merge_datasets' in out


def test_class_access_does_not_attempt_live_resolution():
    out = _Fake.some_light('extract_inputs', show=False)
    assert 'reached through' not in out


# ── static lookup through the ast index ─────────────────────────────────


def test_index_finds_names_across_the_package():
    index = _name_index()
    for name in ('merge_datasets', 'compute_param', 'isa_atmosphere', 'GMM'):
        assert name in index, name


def test_index_records_the_qualified_name():
    assert any(q == 'BaseRing.Backpack.isa_atmosphere'
               for _, q, _ in _name_index()['isa_atmosphere'])


def test_index_skips_the_test_suite():
    assert 'test_index_skips_the_test_suite' not in _name_index()


def test_static_lookup_renders_signature_and_docstring():
    out = EarendilsLight().help('compute_param', show=False)
    assert 'BaseRing.compute_param' in out
    assert 'formula' in out
    assert 'restricted expression parser' in out


def test_several_matches_are_all_listed():
    out = EarendilsLight().help('extract_inputs', show=False)
    assert 'matches for' in out
    assert 'CODAReader.extract_inputs' in out
    assert 'NUMPYReader.extract_inputs' in out


def test_unknown_name_says_so_without_raising():
    out = EarendilsLight().help('no_existe_este_metodo', show=False)
    assert 'Nothing called' in out


def test_overview_when_asked_for_nothing():
    assert "Eärendil's Light" in EarendilsLight().help(show=False)


# ── rendering details ───────────────────────────────────────────────────


def test_signature_drops_self():
    assert _signature(_Reader.extract_inputs).startswith('extract_inputs(id_groups')


def test_opaque_detection_ignores_self():
    assert _is_opaque(_Fake.extract_inputs)
    assert not _is_opaque(_Reader.extract_inputs)


def test_help_returns_the_text_it_prints(capsys):
    out = EarendilsLight().help('compute_param')
    assert capsys.readouterr().out.strip() == out.strip()


# ── the point of the rewrite: importing it must be free ─────────────────


def test_importing_costs_no_third_party_module():
    """rich used to be imported here, and cost ~49 ms on every FRODO import."""
    code = (
        'import sys, FotR.EarendilsLight;'
        "heavy = [m for m in ('rich', 'torch', 'pyvista', 'seaborn', 'sklearn',"
        " 'matplotlib', 'pandas') if m in sys.modules];"
        'print(heavy)'
    )
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]', out.stdout


def test_every_character_exposes_the_shortcut():
    for module, name in [('FotR.characters.frodo', 'FRODO'),
                         ('FotR.characters.gandalf', 'GANDALF'),
                         ('FotR.characters.rings.base', 'BaseRing')]:
        cls = getattr(importlib.import_module(module), name)
        assert callable(cls.some_light)
        assert inspect.isfunction(cls.some_light)
