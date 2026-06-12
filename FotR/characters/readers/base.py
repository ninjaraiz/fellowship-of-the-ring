"""
readers/base.py
===============
Abstract base class that every FRODO reader must implement.

Contract
--------
A concrete reader is responsible for three things:

1. ``parse_simulation_dirs()``
   Walk ``self.root_dir``, build ``self.sim_metadata`` (a dict keyed by
   folder / file name) and ``self.df_state`` (a DataFrame that summarises
   the design-variable space and completion state).

2. ``extract_inputs(*args, **kwargs)``
   Read mesh coordinates, connectivity and flight-condition arrays into
   ``self.data_dict``.  The exact signature depends on the format.

3. ``extract_outputs(*args, **kwargs)``
   Read field variables (Cp, velocity, …) into ``self.data_dict``.
   Requires ``extract_inputs`` to have been called first.

After each call FRODO syncs ``sim_metadata``, ``df_state`` and
``data_dict`` from the reader via ``FRODO._sync_reader()``.

How to implement a new reader
------------------------------
::

    # readers/my_format.py
    from .base import BaseReader

    class MyFormatReader(BaseReader):

        def __init__(self, root_dir: str, **kwargs):
            super().__init__(root_dir, **kwargs)
            # format-specific initialisation
            self.my_option = kwargs.get('my_option', 'default')

        def parse_simulation_dirs(self):
            # walk self.root_dir, fill self.sim_metadata and self.df_state
            ...

        def extract_inputs(self, keys_inputs, **kwargs):
            # fill self.data_dict['inputs'] / self.data_dict[key]
            ...

        def extract_outputs(self, keys_outputs, **kwargs):
            # fill self.data_dict['outputs'] / self.data_dict[key]['Vars']
            ...

Then register it in ``readers/__init__.py``::

    from .my_format import MyFormatReader
    READER_REGISTRY['MY_FORMAT'] = MyFormatReader
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseReader(ABC):
    """
    Abstract base class for all FRODO format readers.

    Subclasses must implement:
    - ``parse_simulation_dirs``
    - ``extract_inputs``
    - ``extract_outputs``

    Attributes
    ----------
    root_dir : str
        Absolute path to the dataset root directory.
    sim_metadata : dict
        Populated by ``parse_simulation_dirs``.  Maps each simulation
        identifier (folder name, file name, …) to a dict of metadata.
    df_state : pd.DataFrame or None
        Populated by ``parse_simulation_dirs``.  One row per simulation,
        columns are design variables plus status / stage information.
    data_dict : dict
        Populated by ``extract_inputs`` and ``extract_outputs``.
        Structure depends on the format; FRODO reads it after syncing.
    """

    def __init__(self, root_dir: str, **kwargs):
        """
        Parameters
        ----------
        root_dir : str
            Path to the dataset root directory.  Stored as-is (FRODO
            passes the already-resolved absolute path).
        **kwargs :
            Format-specific keyword arguments forwarded from
            ``FRODO.__init__``.
        """
        self.root_dir     = root_dir
        self.sim_metadata: dict             = {}
        self.df_state:     pd.DataFrame     = pd.DataFrame()
        self.data_dict:    dict             = {}

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def parse_simulation_dirs(self) -> None:
        """
        Walk ``self.root_dir``, discover simulations and populate
        ``self.sim_metadata`` and ``self.df_state``.

        Called automatically by ``FRODO._parse()`` during ``__init__``
        when ``initial_parse=True``.
        """

    @abstractmethod
    def extract_inputs(self, *args, **kwargs) -> None:
        """
        Extract mesh geometry and flight-condition data into
        ``self.data_dict``.

        The exact signature (positional and keyword arguments) is defined
        by each concrete reader.  FRODO forwards ``*args`` and ``**kwargs``
        from ``db.extract_inputs(...)`` directly to this method.

        After this call, at minimum the following keys must be present in
        ``self.data_dict`` (or a sub-dict thereof, depending on the format):

        - coordinates / point cloud
        - flight-condition / parametric array (FlCc)
        """

    @abstractmethod
    def extract_outputs(self, *args, **kwargs) -> None:
        """
        Extract field variables into ``self.data_dict``.

        Must be called after ``extract_inputs``.  The exact signature is
        defined by each concrete reader.

        After this call, field arrays (Cp, velocity, …) must be accessible
        inside ``self.data_dict`` so that FRODO can sync them.
        """

    # ── Optional hooks (override as needed) ──────────────────────────────────

    def summary(self) -> str:
        """Return a short human-readable summary of the parsed dataset."""
        n_sims = len(self.sim_metadata)
        df_ok  = not self.df_state.empty
        return (
            f"{type(self).__name__}  |  root: {self.root_dir}\n"
            f"  simulations: {n_sims}  |  df_state: {'yes' if df_ok else 'empty'}"
        )

    def __repr__(self) -> str:
        return self.summary()