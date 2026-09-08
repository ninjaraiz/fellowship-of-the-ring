"""
readers/horses3d.py
====================
Reader for HORSES3D CFD solver output.

Expected directory layout
--------------------------
::

    root_dir/
    ├── metadata/
    │   └── cases_metadata.json          (optional)
    └── outputs/
        ├── case_001/
        │   ├── file_control_p2.control
        │   ├── file_control_p3.control
        │   ├── MESH/
        │   │   └── mesh_v2g2.h5
        │   ├── RESULTS/
        │   │   ├── solution_p2.hsol
        │   │   ├── solution_p2.residuals
        │   │   ├── solution_p3.hsol
        │   │   └── solution_p3.residuals
        │   └── LOG/
        │       └── ...
        └── ...

Core concept
------------
Unlike CODA (one ``outputs/<folder>`` = one fully resolved simulation) or
NUMPY (one ``.npy`` file = one pre-assembled CADGroup), a HORSES3D case
folder can hold **several polynomial-order solutions** (``p``) of the
*same* physical case, all sharing a single mesh:

    case folder  (== case_idx, the physical-case identity)
     ├── mesh                     (shared across every p)
     │    ├── file
     │    └── geometric degree g  (a.k.a. NGeo)
     └── solutions
          ├── p=2  ← file_control_p2.control, .hsol, .residuals, restart
          ├── p=3  ← file_control_p3.control, .hsol, .residuals, restart
          └── ...

``p`` is **never** a case-identity axis: two solutions with different
``p`` inside the same folder are the same physical case at a different
resolution, exactly the same way FotR already keeps polynomial order (or
any other resolution dimension) out of ``df_cases`` identity elsewhere in
the framework. HORSES3D additionally requires ``p >= g`` (mesh geometric
order), which is validated whenever both are known.

Design decisions
-----------------
* **Case discovery is direct.** Every subdirectory of ``outputs/`` *is*
  a case — there is no folder-name -> numeric-distance matching against
  ``df_cases`` (that CODA-specific trick assumes folder names encode the
  design variables numerically and unambiguously, which HORSES3D does
  not guarantee).
* **The ``.control`` files are the source of truth for solutions.** A
  case's available ``p`` values, mesh, and solution/restart files are
  read from ``Polynomial order`` / ``mesh file name`` / ``solution file
  name`` / ``restart`` / ``restart file name`` / ``restart polorder``
  inside each ``file_control_p<p>.control`` — never guessed by globbing
  ``*_p<p>.hsol`` and assuming the numbers line up.
* **No fixed ``p1..p7`` columns.** ``df_state`` carries a
  variable-length ``'p_available'`` list column (one row per physical
  case), and the exhaustive per-``p`` detail lives in ``sim_metadata``.
* **``design_vars`` / ``df_cases`` are inferred whenever possible**,
  even without ``metadata/cases_metadata.json``. When the file is
  present it is the preferred, authoritative source (and is joined onto
  the discovered cases purely additively, by folder name). When it is
  absent or unusable, the case-folder names themselves are inspected
  for a consistent ``<name>_<value>_<name>_<value>...`` pattern (the
  same convention already used by ``CODAReader._infer_metadata_from_folders``
  for its own format) to build ``design_vars`` and ``df_cases``
  automatically. If no consistent pattern can be found, discovery still
  succeeds — it simply proceeds without design-variable information
  (``self.metadata['design_vars']`` / ``['df_cases']`` stay ``None``),
  since a case's physical identity in HORSES3D is the folder name
  itself, not the design variables.

What this implementation does and does not do
-------------------------------------------------
This iteration builds correct **discovery and metadata**:
``parse_simulation_dirs`` (case discovery, control parsing, mesh/``g``
association, ``design_vars``/``df_cases`` inference, validations) and a
**subset mechanism** over physical cases, mirroring the one used by
``CODAReader``. ``extract_inputs`` / ``extract_outputs`` are defined
with the same responsibility split used by every other FotR reader (mesh
+ FlCc vs. field variables), but binary ``.hsol`` / mesh reading is
intentionally left unimplemented: no HORSES3D binary-format utility
exists yet in ``FotR.characters.sam.SAM`` to reuse, and guessing one
would be speculative.
"""

import os
import re
import json
import warnings
from typing import Union

import numpy as np
import pandas as pd

from ..sam import SAM
from .base import BaseReader


class Horses3DReader(BaseReader):
    """
    Reader for HORSES3D CFD solver output.

    A *case* is a single subdirectory of ``outputs/``. A case may hold
    one or more polynomial-order *solutions* (``p``), each declared by
    its own ``file_control_p<p>.control`` file and backed by a
    ``RESULTS/*.hsol`` (+ optional ``.residuals``) pair. All solutions
    inside a case share the same mesh.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root directory (passed by FRODO). Must
        contain an ``outputs/`` subdirectory.
    strict : bool
        If True (default), structural inconsistencies (missing declared
        files, ``p < g``, mismatched ``p`` between control filename and
        content, duplicate ``p`` within a case, incompatible mesh
        declarations, …) raise ``ValueError``. If False, they are
        emitted as ``UserWarning`` and the offending solution is kept
        but flagged via ``sim_metadata[case]['solutions'][p]['valid'] =
        False`` with the reason(s) recorded in ``'issues'``.

    Attributes set after ``__init__`` / ``parse_simulation_dirs``
    -----------------------------------------------------------------
    metadata : dict
        Keys: ``'eq_type'``, ``'design_vars'``, ``'df_cases'``.
        Populated from ``metadata/cases_metadata.json`` when present and
        usable (see :meth:`_load_optional_cases_metadata`); otherwise
        inferred directly from the case-folder names when they follow a
        consistent ``<name>_<value>_...`` pattern (see
        :meth:`_infer_metadata_from_folders`); if neither source yields
        usable information, both stay ``None`` and case discovery still
        proceeds using folder names as the sole case identity.
    sim_metadata : dict
        Keyed by case folder name. See :meth:`parse_simulation_dirs` for
        the full per-case hierarchical structure.
    df_state : pd.DataFrame
        One row per **physical case** (never per ``p``). See
        :meth:`parse_simulation_dirs`.
    subsets : dict[str, list[int]]
        Named collections of ``case_idx`` values (physical cases, not
        solutions), mirroring the subset mechanism used elsewhere in
        FotR.

    Examples
    --------
    ::

        from FotR.characters.frodo import FRODO

        db = FRODO(root_dir='/data/horses_run', format='HORSES3D')
        print(db.df_state[['case', 'case_idx', 'p_available', 'g']])

        db.reader.define_subset('training', case_idx=[0, 2, 5])
    """

    #: Matches a control filename and captures the p encoded in it.
    _CONTROL_RE = re.compile(r"^file_control_p(?P<p>\d+)\.control$")

    #: Opportunistic hint for the mesh geometric order g from filenames
    #: following a "...v<version>g<g>..." convention. Only ever used as
    #: a hint — never treated as ground truth on its own, and never
    #: guessed when the pattern does not match unambiguously.
    _MESH_G_RE = re.compile(r"v\d+g(?P<g>\d+)", re.IGNORECASE)

    #: Separators tried when inferring design_vars from case-folder names.
    _FOLDER_NAME_SEPARATORS = ['_', '-']

    def __init__(self, root_dir: str, strict: bool = True, **kwargs):
        super().__init__(root_dir, **kwargs)
        self.output_dir = os.path.join(self.root_dir, "outputs")
        print(f'\n NEW HORSES3D SIMULATION WILL BE LOADED FROM {root_dir}')

        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(
                f"Expected an 'outputs' directory at '{self.output_dir}'."
            )

        self.strict  = strict
        self.subsets: dict = {}
        self._active_cases_idx: dict = {}

        self.metadata = {
            'eq_type':     None,
            'design_vars': None,
            'df_cases':    None,
        }

        loaded_from_json = self._load_optional_cases_metadata(root_dir)
        if not loaded_from_json:
            self._infer_metadata_from_folders()

    # ── Optional cases_metadata.json ─────────────────────────────────────────

    def _load_optional_cases_metadata(self, root_dir: str) -> bool:
        """
        Load ``metadata/cases_metadata.json`` when present, as the
        preferred (but still purely additive) source of design-variable
        information, later joined onto the discovered cases by folder
        name.

        Unlike ``CODAReader``, this file never drives case *discovery*
        (every ``outputs/`` subdirectory is a case regardless of whether
        this metadata exists or matches it) and no folder-name -> numeric
        distance matching is performed here.

        Parameters
        ----------
        root_dir : str
            Dataset root directory.

        Returns
        -------
        bool
            True if ``metadata/cases_metadata.json`` was found and
            successfully parsed into at least ``design_vars`` or
            ``df_cases`` (in which case folder-name inference is
            skipped); False otherwise (the caller should fall back to
            :meth:`_infer_metadata_from_folders`).

        Side-effects
        ------------
        Populates ``self.metadata['eq_type']``, ``['design_vars']`` and
        ``['df_cases']`` when the file is usable.

        Examples
        --------
        ::

            ok = reader._load_optional_cases_metadata('/data/horses_run')
            if not ok:
                reader._infer_metadata_from_folders()
        """
        meta_path = os.path.join(root_dir, 'metadata', 'cases_metadata.json')
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, 'r') as fh:
                cm = json.load(fh)

            design_vars = cm.get('design_vars', None)
            df_cases_raw = cm.get('df_cases', None)

            if design_vars is None and df_cases_raw is None:
                return False

            self.metadata['eq_type']     = cm.get('eq_type', None)
            self.metadata['design_vars'] = design_vars

            if df_cases_raw is not None:
                df_cases = pd.DataFrame.from_dict(df_cases_raw)
                if "case_idx" not in df_cases.columns:
                    df_cases.insert(
                        0, "case_idx", df_cases.index.astype(np.int32)
                    )
                self.metadata['df_cases'] = df_cases

            return True
        except Exception as exc:
            warnings.warn(
                f"Could not parse '{meta_path}': {exc}. Falling back to "
                "design_vars/df_cases inference from case-folder names.",
                UserWarning,
            )
            return False

    # ── Metadata inference fallback (folder-name based) ───────────────────────

    def _infer_metadata_from_folders(self) -> None:
        """
        Infer ``design_vars`` and ``df_cases`` directly from the
        case-folder names inside ``outputs/``, when no usable
        ``cases_metadata.json`` is available.

        This reuses the folder-naming convention already relied upon
        elsewhere in FotR (``CODAReader._infer_metadata_from_folders``):
        a case folder name is expected to alternate
        ``<name>_<value>_<name>_<value>...`` (or with ``-`` as
        separator), e.g. a folder made of tokens
        ``['alpha', '3.0', 'reynolds', '200']`` encodes design variables
        ``'alpha'`` and ``'reynolds'`` with values ``3.0`` and ``200``.

        Unlike ``CODAReader``, this method does **not** need to match a
        folder to the "closest" row of a numeric table by Euclidean
        distance — since every case folder already *is* the case here,
        each folder directly produces exactly one ``df_cases`` row with
        an explicit ``'folder'`` column, so :meth:`parse_simulation_dirs`
        can join back onto it unambiguously by name instead of by
        numeric proximity.

        Behaviour when inference is not possible
        ------------------------------------------
        If the case folders do not share a consistent separator, token
        count or naming pattern, this method does **not** raise: unlike
        CODA (where the folder name is the *only* way to recover the
        design-variable identity of a simulation), a HORSES3D case's
        identity is its folder name regardless of whether that name
        happens to encode numeric parameters. In that situation a
        warning is emitted and ``self.metadata['design_vars']`` /
        ``['df_cases']`` are left as ``None`` — case discovery in
        :meth:`parse_simulation_dirs` proceeds unaffected.

        Side-effects
        ------------
        Populates ``self.metadata['design_vars']`` (``list[str]`` or
        ``None``) and ``self.metadata['df_cases']`` (``pd.DataFrame``
        with a ``'folder'`` column, or ``None``).

        Examples
        --------
        ::

            reader._infer_metadata_from_folders()
            print(reader.metadata['design_vars'])
            print(reader.metadata['df_cases'][['folder', 'case_idx']])
        """
        case_names = sorted(
            d for d in os.listdir(self.output_dir)
            if os.path.isdir(os.path.join(self.output_dir, d))
        )

        if not case_names:
            warnings.warn(
                "No case folders found under 'outputs/'. Skipping "
                "design_vars/df_cases inference.",
                UserWarning,
            )
            return

        for sep in self._FOLDER_NAME_SEPARATORS:
            if not all(sep in c for c in case_names):
                continue

            tokenised = [c.split(sep) for c in case_names]
            n_tokens  = len(tokenised[0])
            if any(len(t) != n_tokens for t in tokenised):
                continue

            # Expect alternating <name>_<value>_<name>_<value>...
            name_positions  = range(0, n_tokens, 2)
            value_positions = range(1, n_tokens, 2)
            if len(list(value_positions)) == 0:
                continue

            names_consistent = all(
                all(not re.fullmatch(r"-?\d+\.?\d*", t[i]) for t in tokenised)
                for i in name_positions
            )
            values_numeric = all(
                all(re.fullmatch(r"-?\d+\.?\d*", t[i]) for t in tokenised)
                for i in value_positions
            )
            if not (names_consistent and values_numeric):
                continue

            design_vars = [tokenised[0][i] for i in name_positions]
            df_arr = np.array(
                [[float(t[i]) for i in value_positions] for t in tokenised],
                dtype=float,
            )

            df_cases = pd.DataFrame(df_arr, columns=design_vars)
            df_cases.insert(0, "folder", case_names)
            df_cases.insert(0, "case_idx", np.arange(len(case_names), dtype=np.int32))

            self.metadata['design_vars'] = design_vars
            self.metadata['df_cases']    = df_cases
            return

        warnings.warn(
            "Could not infer a consistent '<name>_<value>_...' pattern "
            "from case-folder names under 'outputs/'. Proceeding without "
            "design_vars/df_cases; cases remain identified solely by "
            "folder name.",
            UserWarning,
        )

    # =========================================================================
    # BaseReader interface
    # =========================================================================

    def parse_simulation_dirs(self) -> None:
        """
        Walk ``outputs/``, treating every subdirectory as one physical
        case, and build ``self.sim_metadata`` / ``self.df_state``.

        For every case folder:

        1. Find every ``file_control_p<p>.control`` file directly inside
           it (via ``SAM.Backpack.pattern_pocket.find_files``). A case
           with none is a structural error under ``self.strict``.
        2. Parse each control (:meth:`_parse_control_file`) and check the
           ``p`` encoded in its filename against the ``Polynomial
           order`` declared inside it.
        3. Reject (under ``self.strict``) two controls declaring the
           same ``p``, or declaring mutually incompatible mesh files.
        4. Resolve and check existence of the declared mesh, solution
           (``.hsol``) and, when ``restart=.true.``, restart files. The
           ``.residuals`` sibling of the solution file (same stem) is
           looked up and flagged present/absent — it is optional.
        5. Infer the mesh geometric order ``g`` from the mesh filename
           via ``_MESH_G_RE`` (only when unambiguous — see
           :meth:`_infer_geometric_g`) and validate ``p >= g`` for every
           solution once ``g`` is known.
        6. If ``self.metadata['df_cases']`` is available (from
           ``cases_metadata.json`` or from folder-name inference — see
           :meth:`_load_optional_cases_metadata` /
           :meth:`_infer_metadata_from_folders``), join the
           design-variable row that matches this folder name (via a
           ``'folder'`` column) onto the case, purely additively.

        Populates
        ---------
        self.sim_metadata : dict
            Keyed by case folder name::

                {
                    'case_001': {
                        'case_idx':    0,
                        'path':        '/root/outputs/case_001',
                        'design_vars': {'alpha': 3.0, ...} or {},
                        'mesh': {
                            'file': 'MESH/mesh_v2g2.h5',
                            'g':    2,          # or None
                        },
                        'solutions': {
                            2: {
                                'control_file':     'file_control_p2.control',
                                'polynomial_order':  2,
                                'solution_file':     'RESULTS/solution_p2.hsol',
                                'residuals_file':    'RESULTS/solution_p2.residuals',
                                'residuals_found':   True,
                                'surface_monitors': {
                                    'lift': {
                                        'monitor_name':      'lift',
                                        'surface_marker':    2,
                                        'selected_variable': 'lift',
                                        'dynamic_pressure':  6383.475,
                                        'columns':  ['Iteration', 'Time', 'lift'],
                                        'file':     'RESULTS/solution_p2.lift.surface',
                                    },
                                },
                                'stopwatch': {
                                    'events': {
                                        'TotalTime': {'elapsed_s': 18650.6, 'cpu_s': 18650.6},
                                        ...
                                    },
                                    'file': 'RESULTS/solution_p2.Stopwatch.info',
                                },
                                'other_files':       [],
                                'restart':           False,
                                'restart_file':      None,
                                'restart_polorder':  None,
                                'params': {
                                    'flow_equations':  'NS',
                                    'mach_number':     0.3,
                                    'reynolds_number': 200.0,
                                    'aoa_theta':       0.0,
                                    'aoa_phi':         0.0,
                                    ...
                                },
                                'valid':  True,
                                'issues': [],
                            },
                            3: {
                                ...,
                                'restart':          True,
                                'restart_file':     'RESULTS/solution_p2.hsol',
                                'restart_polorder': 2,
                            },
                        },
                    },
                    ...
                }

        self.df_state : pd.DataFrame
            One row per **physical case**. Columns: ``'case'``,
            ``'case_idx'``, ``'p_available'`` (``list[int]``, one entry
            per resolved ``p``), ``'n_solutions'``, ``'g'`` (nullable
            int), ``'mesh_file'``, ``'all_valid'`` (bool), every
            design-variable column (from ``cases_metadata.json`` or
            inferred from folder names) when available, and every
            flight/solver parameter that is present and numerically
            identical across *all* of the case's solutions (parameters
            that disagree across ``p`` for the same case are dropped
            from ``df_state`` and remain accessible per-solution in
            ``sim_metadata``).

        Raises
        ------
        ValueError
            (only when ``self.strict=True``) for the structural
            inconsistencies listed above.

        Examples
        --------
        ::

            reader.parse_simulation_dirs()
            print(reader.df_state[['case', 'p_available', 'g']])
            print(reader.sim_metadata['case_001']['solutions'][3])
        """
        self.sim_metadata = {}
        case_names = sorted(
            d for d in os.listdir(self.output_dir)
            if os.path.isdir(os.path.join(self.output_dir, d))
        )

        df_cases = self.metadata.get('df_cases')
        has_design_vars_table = (
            df_cases is not None and 'folder' in df_cases.columns
        )

        state_rows: list = []

        for case_idx, case_name in enumerate(case_names):
            case_path = os.path.join(self.output_dir, case_name)

            control_files = sorted(
                os.path.basename(f) for f in
                SAM.Backpack.pattern_pocket.find_files(
                    case_path, endswith='.control', verbose=False,
                )
                if self._CONTROL_RE.match(os.path.basename(f))
            )

            if not control_files:
                msg = (
                    f"Case '{case_name}' has no "
                    "'file_control_p<p>.control' file."
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning)
                self.sim_metadata[case_name] = {
                    'case_idx':    case_idx,
                    'path':        case_path,
                    'design_vars': {},
                    'mesh':        {'file': None, 'g': None},
                    'solutions':   {},
                }
                state_rows.append(self._build_state_row(
                    case_name, case_idx, None, None, {}, {}
                ))
                continue

            solutions: dict = {}
            mesh_files_seen: set = set()
            g_candidates: set = set()

            for cf in control_files:
                m = self._CONTROL_RE.match(cf)
                p_from_name = int(m.group('p'))

                if p_from_name in solutions:
                    msg = (
                        f"Case '{case_name}': duplicate solution declared "
                        f"for p={p_from_name} (control '{cf}' conflicts "
                        f"with '{solutions[p_from_name]['control_file']}')."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    warnings.warn(msg, UserWarning)
                    continue

                parsed = self._parse_control_file(os.path.join(case_path, cf))
                issues: list = []

                p_declared = parsed['polynomial_order']
                if p_declared is not None and p_declared != p_from_name:
                    msg = (
                        f"Case '{case_name}', control '{cf}': filename "
                        f"declares p={p_from_name} but 'Polynomial order' "
                        f"inside the file is {p_declared}."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    issues.append(msg)
                    warnings.warn(msg, UserWarning)

                p = p_from_name

                # ── mesh file ────────────────────────────────────────────
                mesh_file = parsed['mesh_file']
                if mesh_file is None:
                    msg = (
                        f"Case '{case_name}', control '{cf}': no "
                        "'mesh file name' declared."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    issues.append(msg)
                    warnings.warn(msg, UserWarning)
                else:
                    mesh_files_seen.add(mesh_file)
                    if not os.path.exists(os.path.join(case_path, mesh_file)):
                        msg = (
                            f"Case '{case_name}', control '{cf}': declared "
                            f"mesh file '{mesh_file}' not found on disk."
                        )
                        if self.strict:
                            raise ValueError(msg)
                        issues.append(msg)
                        warnings.warn(msg, UserWarning)

                    g_hint = self._infer_geometric_g(mesh_file)
                    if g_hint is not None:
                        g_candidates.add(g_hint)

                # ── solution / residuals files ──────────────────────────
                solution_file = parsed['solution_file']
                if solution_file is None:
                    msg = (
                        f"Case '{case_name}', control '{cf}': no "
                        "'solution file name' declared."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    issues.append(msg)
                    warnings.warn(msg, UserWarning)
                    residuals_file, residuals_found = None, False
                else:
                    if not os.path.exists(
                        os.path.join(case_path, solution_file)
                    ):
                        msg = (
                            f"Case '{case_name}', control '{cf}': declared "
                            f"solution file '{solution_file}' not found "
                            "on disk."
                        )
                        if self.strict:
                            raise ValueError(msg)
                        issues.append(msg)
                        warnings.warn(msg, UserWarning)

                    stem, _ = os.path.splitext(solution_file)
                    residuals_file  = f"{stem}.residuals"
                    residuals_found = os.path.exists(
                        os.path.join(case_path, residuals_file)
                    )
                    if not residuals_found:
                        residuals_file = None

                # ── related result files (surface monitors, stopwatch, …) ──
                related = (
                    self._discover_related_result_files(case_path, solution_file)
                    if solution_file is not None
                    else {'surface_monitors': {}, 'stopwatch': None, 'other_files': []}
                )

                # ── restart ──────────────────────────────────────────────
                restart          = bool(parsed['restart'])
                restart_file     = parsed['restart_file']
                restart_polorder = parsed['restart_polorder']
                if restart:
                    if restart_file is None:
                        msg = (
                            f"Case '{case_name}', control '{cf}': "
                            "restart=.true. but no 'restart file name' "
                            "declared."
                        )
                        if self.strict:
                            raise ValueError(msg)
                        issues.append(msg)
                        warnings.warn(msg, UserWarning)
                    elif not os.path.exists(
                        os.path.join(case_path, restart_file)
                    ):
                        msg = (
                            f"Case '{case_name}', control '{cf}': declared "
                            f"restart file '{restart_file}' not found on "
                            "disk."
                        )
                        if self.strict:
                            raise ValueError(msg)
                        issues.append(msg)
                        warnings.warn(msg, UserWarning)

                    if (
                        restart_polorder is not None
                        and restart_polorder >= p
                    ):
                        msg = (
                            f"Case '{case_name}', control '{cf}': "
                            f"restart polorder ({restart_polorder}) should "
                            f"be strictly lower than this solution's "
                            f"p ({p})."
                        )
                        if self.strict:
                            raise ValueError(msg)
                        issues.append(msg)
                        warnings.warn(msg, UserWarning)

                solutions[p] = {
                    'control_file':      cf,
                    'polynomial_order':  p,
                    'solution_file':     solution_file,
                    'residuals_file':    residuals_file,
                    'residuals_found':   residuals_found,
                    'surface_monitors':  related['surface_monitors'],
                    'stopwatch':         related['stopwatch'],
                    'other_files':       related['other_files'],
                    'restart':           restart,
                    'restart_file':      restart_file,
                    'restart_polorder':  restart_polorder,
                    'params':            parsed['params'],
                    'valid':             len(issues) == 0,
                    'issues':            issues,
                }

            # ── mesh consistency across this case's solutions ────────────
            if len(mesh_files_seen) > 1:
                msg = (
                    f"Case '{case_name}': solutions reference different "
                    f"mesh files {sorted(mesh_files_seen)}. Expected a "
                    "single shared mesh per physical case."
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning)
                mesh_file_final = sorted(mesh_files_seen)[0]
            else:
                mesh_file_final = (
                    next(iter(mesh_files_seen)) if mesh_files_seen else None
                )

            if len(g_candidates) > 1:
                warnings.warn(
                    f"Case '{case_name}': ambiguous geometric order g "
                    f"candidates {sorted(g_candidates)} inferred from mesh "
                    "filename(s). Leaving g=None rather than guessing.",
                    UserWarning,
                )
                g_final = None
            else:
                g_final = next(iter(g_candidates), None)

            # ── p >= g validation ─────────────────────────────────────────
            if g_final is not None:
                for p, sol in solutions.items():
                    if p < g_final:
                        msg = (
                            f"Case '{case_name}': solution p={p} violates "
                            f"p >= g (g={g_final})."
                        )
                        if self.strict:
                            raise ValueError(msg)
                        sol['valid'] = False
                        sol['issues'].append(msg)
                        warnings.warn(msg, UserWarning)

            # ── design_vars join (from cases_metadata.json or inferred) ────
            design_vars_row: dict = {}
            if has_design_vars_table:
                match = df_cases.loc[df_cases['folder'] == case_name]
                if not match.empty:
                    dv_cols = [
                        c for c in (self.metadata.get('design_vars') or [])
                        if c in match.columns
                    ]
                    design_vars_row = (
                        match.iloc[0][dv_cols].to_dict() if dv_cols else {}
                    )

            self.sim_metadata[case_name] = {
                'case_idx':    case_idx,
                'path':        case_path,
                'design_vars': design_vars_row,
                'mesh':        {'file': mesh_file_final, 'g': g_final},
                'solutions':   solutions,
            }

            state_rows.append(self._build_state_row(
                case_name, case_idx, mesh_file_final, g_final,
                solutions, design_vars_row,
            ))

        self.df_state = (
            pd.DataFrame.from_records(state_rows)
            if state_rows else pd.DataFrame()
        )

        n_sims = len(self.sim_metadata)
        print(
            f"{n_sims} simulation(s) found."
            if n_sims != 1 else "1 simulation found."
        )

    def extract_inputs(
        self,
        p: Union[int, str] = 'max',
        subset: Union[str, None] = None,
        cases_idx: Union[list, tuple, int, str] = 'all',
        verbose: bool = False,
    ) -> None:
        """
        Extract mesh geometry and flight/solver-condition data for the
        selected physical cases.

        Responsibility split (mirrors ``CODAReader`` / ``NUMPYReader``)
        -------------------------------------------------------------
        Responsible for the mesh (read from the ``MESH/`` file declared
        by each case's control) and for ``FlCc`` (flight/solver
        parameters, assembled from the selected representative
        solution's parsed ``params``). Field variables inside ``.hsol``
        files are the responsibility of :meth:`extract_outputs`.

        Parameters
        ----------
        p : int or 'max'
            Which polynomial-order solution's control/mesh reference to
            use as representative for each case (the mesh is shared
            across ``p`` within a case; ``params`` are re-read from this
            specific solution's control since they could in principle be
            overridden between restarts). ``'max'`` (default) uses the
            highest available ``p`` per case.
        subset : str or None
            Name of a subset registered via :meth:`define_subset`.
            Mutually exclusive with an explicit (non-``'all'``)
            ``cases_idx``.
        cases_idx : list, tuple, int or 'all'
            Physical ``case_idx`` selection. Default ``'all'``.
        verbose : bool
            Print per-case progress.

        Raises
        ------
        ValueError
            If both ``subset`` and a non-default ``cases_idx`` are given.
        NotImplementedError
            Mesh binary reading (``.h5`` / ``.bmesh`` / ``.pmesh`` /
            ``.hmesh``) is not implemented: no HORSES3D mesh-I/O utility
            currently exists in ``FotR.characters.sam.SAM`` to reuse.
            Case/subset resolution and ``FlCc`` assembly from
            already-parsed control parameters are performed first (and
            can be exercised independently of mesh I/O), then this
            method raises with a clear message at the point mesh reading
            would begin.

        Examples
        --------
        Once mesh I/O is available::

            reader.extract_inputs(p='max', subset='training')
            print(reader.data_dict['FlCc'])
        """
        resolved_idx = self._resolve_cases_idx(subset, cases_idx)

        design_vars = self._infer_param_names()
        flcc_rows, case_order = [], []

        for case_idx in resolved_idx:
            case_name = self._case_name_from_idx(case_idx)
            sols      = self.sim_metadata[case_name]['solutions']
            if not sols:
                raise KeyError(
                    f"Case '{case_name}' (idx={case_idx}) has no solutions."
                )
            p_use = max(sols) if p == 'max' else int(p)
            if p_use not in sols:
                raise KeyError(
                    f"p={p_use} not available for case '{case_name}'. "
                    f"Available: {sorted(sols)}."
                )

            params = sols[p_use]['params']
            flcc_rows.append([params.get(dv, np.nan) for dv in design_vars])
            case_order.append(case_name)

            if verbose:
                print(
                    f"[Horses3DReader] extract_inputs — case '{case_name}' "
                    f"(idx={case_idx}), p={p_use}: mesh="
                    f"{self.sim_metadata[case_name]['mesh']['file']}"
                )

        self._active_cases_idx['inputs'] = resolved_idx
        self.data_dict['FlCc']           = np.asarray(flcc_rows, dtype=np.float64)
        self.data_dict['case_order']     = case_order
        self.metadata['design_vars']     = design_vars

        raise NotImplementedError(
            "Horses3DReader.extract_inputs: case/subset resolution and "
            "FlCc assembly from parsed .control parameters succeeded "
            "(see self.data_dict['FlCc'] / ['case_order']), but mesh "
            "geometry reading ('MESH/*.h5' / '.bmesh' / '.pmesh' / "
            "'.hmesh') is not yet implemented. No HORSES3D mesh I/O "
            "utility is available in FotR.characters.sam.SAM to reuse, "
            "and a speculative binary parser was intentionally not "
            "written here."
        )

    def extract_outputs(
        self,
        p: Union[int, str] = 'max',
        subset: Union[str, None] = None,
        cases_idx: Union[list, tuple, int, str] = 'all',
        var_name_excluded: Union[list, tuple, None] = None,
        verbose: bool = False,
    ) -> None:
        """
        Extract field variables from the ``.hsol`` solution file(s) of
        the selected physical cases, for a given polynomial order ``p``.

        Responsibility split
        ---------------------
        Mirrors :meth:`extract_inputs`. Requires ``extract_inputs`` to
        have been called first (mirroring ``CODAReader.extract_outputs``
        depending on ``idx_sort`` from ``extract_inputs``).

        Parameters
        ----------
        p : int or 'max'
            Polynomial order whose solution file should be read for each
            case. Default ``'max'``.
        subset : str or None
            Named subset of ``case_idx``. See :meth:`extract_inputs`.
        cases_idx : list, tuple, int or 'all'
            Physical ``case_idx`` selection. Default ``'all'``.
        var_name_excluded : list, tuple or None
            Variable names to skip, once ``.hsol`` reading exists.
        verbose : bool
            Print per-case progress.

        Raises
        ------
        RuntimeError
            If ``extract_inputs`` has not been called first.
        NotImplementedError
            ``.hsol`` binary field reading is not implemented, for the
            reason documented in :meth:`extract_inputs`. The intended
            output layout is documented here so a future implementation
            slots directly into existing FRODO conventions: a
            ``(n_points, n_cases)`` (or ``(n_dim, n_points, n_cases)`` for
            vector fields) array per variable, stored under
            ``self.data_dict[case_name]['Vars'][str(p)][var_name]``.

        Examples
        --------
        Once ``.hsol`` reading is available::

            reader.extract_inputs(p=2, subset='training')
            reader.extract_outputs(p=2, subset='training')
            print(reader.data_dict['case_001']['Vars']['2'].keys())
        """
        if 'inputs' not in self._active_cases_idx:
            raise RuntimeError(
                "No case selection found. Run extract_inputs() first."
            )
        resolved_idx = self._resolve_cases_idx(subset, cases_idx)

        for case_idx in resolved_idx:
            case_name = self._case_name_from_idx(case_idx)
            sols      = self.sim_metadata[case_name]['solutions']
            p_use     = max(sols) if p == 'max' else int(p)
            if p_use not in sols:
                raise KeyError(
                    f"p={p_use} not available for case '{case_name}'. "
                    f"Available: {sorted(sols)}."
                )
            if verbose:
                print(
                    f"[Horses3DReader] extract_outputs — case '{case_name}' "
                    f"(idx={case_idx}), p={p_use}: solution="
                    f"{sols[p_use]['solution_file']}"
                )

        raise NotImplementedError(
            "Horses3DReader.extract_outputs: '.hsol' binary field reading "
            "is not yet implemented — no HORSES3D solution-file I/O "
            "utility is available in FotR.characters.sam.SAM to reuse. "
            "See the docstring for the intended data_dict layout."
        )

    # =========================================================================
    # Subsets
    # =========================================================================

    def define_subset(self, name: str, case_idx: Union[list, tuple]) -> None:
        """
        Register a named subset of physical cases.

        A subset is a named collection of ``case_idx`` values (physical
        cases). It never refers to individual ``p`` solutions: selecting
        a case via a subset implicitly makes every ``p`` available for
        that case eligible for a later, separate ``p`` selection at
        extraction time.

        Parameters
        ----------
        name : str
            Subset name, usable later as ``subset=name`` in
            ``extract_inputs`` / ``extract_outputs``.
        case_idx : list or tuple of int
            Physical-case indices belonging to this subset.

        Raises
        ------
        KeyError
            If any index in ``case_idx`` does not correspond to a known
            case.

        Examples
        --------
        ::

            reader.define_subset('training', case_idx=[0, 2, 5])
            reader.extract_inputs(subset='training')
        """
        valid_idx = {v['case_idx'] for v in self.sim_metadata.values()}
        missing   = [i for i in case_idx if i not in valid_idx]
        if missing:
            raise KeyError(
                f"case_idx {missing} not found among known cases "
                f"({sorted(valid_idx)})."
            )
        self.subsets[name] = list(case_idx)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _resolve_cases_idx(
        self,
        subset: Union[str, None],
        cases_idx: Union[list, tuple, int, str],
    ) -> list:
        """
        Resolve a ``(subset, cases_idx)`` pair to a concrete, sorted list
        of ``case_idx`` values, mirroring the centralised
        ``_resolve_cases_idx`` helper used by ``CODAReader`` /
        ``BaseReader``-derived readers and its explicit
        conflict-detection rule (a named subset and an explicit
        ``cases_idx`` cannot both be given).

        Parameters
        ----------
        subset : str or None
        cases_idx : list, tuple, int or 'all'

        Returns
        -------
        list[int]

        Raises
        ------
        ValueError
            If both ``subset`` and a non-``'all'`` ``cases_idx`` are
            given.
        KeyError
            If ``subset`` is not a registered subset name.
        IndexError
            If ``cases_idx`` contains out-of-range values.

        Examples
        --------
        ::

            idx = reader._resolve_cases_idx('training', 'all')
            idx = reader._resolve_cases_idx(None, [0, 1, 2])
        """
        n_cases = len(self.sim_metadata)

        if subset is not None:
            if not (isinstance(cases_idx, str) and cases_idx.lower() == 'all'):
                raise ValueError(
                    "Provide either 'subset' or an explicit 'cases_idx', "
                    "not both."
                )
            if subset not in self.subsets:
                raise KeyError(
                    f"Subset '{subset}' not found. "
                    f"Available: {list(self.subsets)}."
                )
            return sorted(self.subsets[subset])

        return self._normalise_cases_idx(cases_idx, n_cases)

    def _case_name_from_idx(self, case_idx: int) -> str:
        """Return the case folder name for a given ``case_idx``."""
        for name, meta in self.sim_metadata.items():
            if meta['case_idx'] == case_idx:
                return name
        raise KeyError(f"No case found for case_idx={case_idx}.")

    def _infer_param_names(self) -> list:
        """
        Union of every flight/solver parameter key seen across every
        parsed solution's ``params`` dict, used as the ``FlCc`` column
        order.

        Returns
        -------
        list[str]
            Sorted parameter names.

        Examples
        --------
        ::

            names = reader._infer_param_names()
            # → ['aoa_phi', 'aoa_theta', 'flow_equations', 'mach_number', ...]
        """
        keys: set = set()
        for meta in self.sim_metadata.values():
            for sol in meta['solutions'].values():
                keys.update(sol['params'].keys())
        return sorted(keys)

    @staticmethod
    def _build_state_row(
        case_name: str,
        case_idx: int,
        mesh_file: Union[str, None],
        g: Union[int, None],
        solutions: dict,
        design_vars_row: dict,
    ) -> dict:
        """
        Build a single ``df_state`` row for a physical case.

        ``p_available`` is stored as a variable-length ``list[int]`` —
        never expanded into fixed ``p1..pN`` boolean columns, since the
        set of resolved polynomial orders is arbitrary and
        case-dependent in HORSES3D. Flight/solver parameters are pooled
        into shared columns only where every solution of the case agrees
        on their value; a parameter that disagrees across ``p`` for the
        same case is left out of ``df_state`` (it remains accessible
        per-solution in ``sim_metadata``) rather than silently picking
        one value.

        Parameters
        ----------
        case_name : str
        case_idx : int
        mesh_file : str or None
        g : int or None
        solutions : dict
            ``sim_metadata[case_name]['solutions']``.
        design_vars_row : dict
            Design-variable values, either joined from an optional
            ``cases_metadata.json`` table or inferred from the case
            folder name (see :meth:`_infer_metadata_from_folders`), or
            ``{}`` when neither source is available.

        Returns
        -------
        dict

        Examples
        --------
        ::

            row = Horses3DReader._build_state_row(
                'case_001', 0, 'MESH/mesh_v2g2.h5', 2, solutions, {},
            )
        """
        row = {
            'case':        case_name,
            'case_idx':    case_idx,
            'p_available': sorted(solutions.keys()),
            'n_solutions': len(solutions),
            'g':           g,
            'mesh_file':   mesh_file,
            'all_valid':   all(s['valid'] for s in solutions.values())
                           if solutions else False,
        }
        row.update(design_vars_row)

        param_sets = [s['params'] for s in solutions.values()]
        if param_sets:
            common_keys = set.intersection(*(set(p) for p in param_sets))
            for k in sorted(common_keys):
                values = {p[k] for p in param_sets}
                if len(values) == 1:
                    row[k] = next(iter(values))

        return row

    def _discover_related_result_files(
        self,
        case_path: str,
        solution_file: str,
    ) -> dict:
        """
        Catalogue every file inside the solution's result directory that
        shares the solution's basename (i.e. the ``.hsol`` stem), beyond
        the ``.hsol`` itself and its ``.residuals`` sibling (both handled
        separately by the caller).

        This is a **generic, closed-list-free** classification: no
        specific monitor name (e.g. ``'lift'``, ``'drag'``) is ever
        hardcoded. A file is recognised as belonging to this solution
        purely because its name starts with ``'<stem>.'`` — the same
        basename convention already used for the ``.residuals`` sibling.
        From there, only the *file-type* suffix is pattern-matched:

        * ``<stem>.<monitor_name>.surface`` → a surface/force monitor.
          ``<monitor_name>`` is whatever text precedes ``.surface`` —
          never assumed to be ``'lift'``, ``'drag'`` or any other fixed
          name.
        * ``<stem>.Stopwatch.info`` → the run's timing breakdown.
        * anything else sharing the stem → recorded verbatim in
          ``'other_files'`` so it is not silently dropped, without
          guessing its meaning.

        Parameters
        ----------
        case_path : str
            Absolute path to the case folder.
        solution_file : str
            The ``solution file name`` declared in the control (relative
            to ``case_path``), e.g. ``'RESULTS/solution_p2.hsol'``.

        Returns
        -------
        dict
            ``{'surface_monitors': dict[str, dict], 'stopwatch': dict or
            None, 'other_files': list[str]}``.

            ``surface_monitors`` maps each monitor name to the dict
            returned by :meth:`_parse_surface_monitor_header` (plus a
            ``'file'`` key with the path relative to ``case_path``).
            ``stopwatch`` is the dict returned by
            :meth:`_parse_stopwatch_file` (plus a ``'file'`` key), or
            ``None`` if no ``.Stopwatch.info`` file was found next to the
            solution. ``other_files`` lists paths (relative to
            ``case_path``) of any further file sharing the stem that did
            not match a recognised suffix.

        Examples
        --------
        ::

            related = reader._discover_related_result_files(
                case_path, 'RESULTS/solution_p2.hsol',
            )
            print(list(related['surface_monitors']))
            print(related['stopwatch'])
        """
        results_dir_rel = os.path.dirname(solution_file)
        results_dir_abs = os.path.join(case_path, results_dir_rel)
        stem = os.path.splitext(os.path.basename(solution_file))[0]

        surface_monitors: dict = {}
        stopwatch = None
        other_files: list = []

        if not os.path.isdir(results_dir_abs):
            return {
                'surface_monitors': surface_monitors,
                'stopwatch':        stopwatch,
                'other_files':      other_files,
            }

        for fname in sorted(os.listdir(results_dir_abs)):
            if not fname.startswith(stem + "."):
                continue

            suffix = fname[len(stem) + 1:]
            rel_path = os.path.join(results_dir_rel, fname) if results_dir_rel else fname

            if suffix == "residuals" or fname == os.path.basename(solution_file):
                # Already handled by the caller (.hsol / .residuals).
                continue

            if suffix.endswith(".surface"):
                monitor_name = suffix[: -len(".surface")]
                try:
                    header = self._parse_surface_monitor_header(
                        os.path.join(results_dir_abs, fname)
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Could not parse surface monitor file '{rel_path}': "
                        f"{exc}. Cataloguing it without metadata.",
                        UserWarning,
                    )
                    header = {
                        'monitor_name':      monitor_name,
                        'surface_marker':    None,
                        'selected_variable': None,
                        'dynamic_pressure':  None,
                        'columns':           None,
                    }
                header['file'] = rel_path
                surface_monitors[monitor_name] = header
                continue

            if suffix == "Stopwatch.info":
                try:
                    stopwatch = self._parse_stopwatch_file(
                        os.path.join(results_dir_abs, fname)
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Could not parse stopwatch file '{rel_path}': "
                        f"{exc}. Cataloguing it without metadata.",
                        UserWarning,
                    )
                    stopwatch = {'events': {}}
                stopwatch['file'] = rel_path
                continue

            other_files.append(rel_path)

        return {
            'surface_monitors': surface_monitors,
            'stopwatch':        stopwatch,
            'other_files':      other_files,
        }

    @staticmethod
    def _parse_surface_monitor_header(path: str) -> dict:
        """
        Parse the metadata header of a HORSES3D ``*.surface`` monitor
        file (e.g. a force/moment monitor), without loading its full
        iteration history into memory.

        Expected layout::

             Monitor name:      <name>
             Surface marker:    <marker>
             Selected variable: <variable>
              Dynamic pressure:         <value>

             Iteration                      Time                      <name>
                     0    0.0000000000000000E+00   -6.4057841531848791E-03
                     ...

        Parameters
        ----------
        path : str
            Absolute path to the ``.surface`` file.

        Returns
        -------
        dict
            Keys: ``'monitor_name'`` (str or None), ``'surface_marker'``
            (int or None), ``'selected_variable'`` (str or None),
            ``'dynamic_pressure'`` (float or None), ``'columns'``
            (``list[str]`` — the history's column header, split on
            whitespace, or ``None`` if it could not be located).

        Examples
        --------
        ::

            header = Horses3DReader._parse_surface_monitor_header(
                '/data/case_001/RESULTS/solution_p2.lift.surface'
            )
            print(header['monitor_name'], header['columns'])
        """
        monitor_name = surface_marker = selected_variable = None
        dynamic_pressure = None
        columns = None

        with open(path, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()

                if lower.startswith("monitor name"):
                    monitor_name = stripped.split(":", 1)[1].strip()
                elif lower.startswith("surface marker"):
                    val = stripped.split(":", 1)[1].strip()
                    try:
                        surface_marker = int(val)
                    except ValueError:
                        surface_marker = val
                elif lower.startswith("selected variable"):
                    selected_variable = stripped.split(":", 1)[1].strip()
                elif lower.startswith("dynamic pressure"):
                    val = stripped.split(":", 1)[1].strip()
                    try:
                        dynamic_pressure = float(val)
                    except ValueError:
                        dynamic_pressure = None
                elif lower.startswith("iteration"):
                    columns = stripped.split()
                    break

        return {
            'monitor_name':      monitor_name,
            'surface_marker':    surface_marker,
            'selected_variable': selected_variable,
            'dynamic_pressure':  dynamic_pressure,
            'columns':           columns,
        }

    @staticmethod
    def _parse_stopwatch_file(path: str) -> dict:
        """
        Parse a HORSES3D ``*.Stopwatch.info`` timing-breakdown file.

        Expected layout::

            #Stopwatch information file
            # Event                                            Elapsed time(s)    CPU-Time (s)
            <event_name>                                       <elapsed>          <cpu>
            ...

        Parameters
        ----------
        path : str
            Absolute path to the ``.Stopwatch.info`` file.

        Returns
        -------
        dict
            ``{'events': {event_name: {'elapsed_s': float, 'cpu_s':
            float}}}``. Lines starting with ``'#'`` (comments/header) are
            skipped; every other non-blank line is expected to have the
            form ``<event_name> <elapsed> <cpu>``.

        Examples
        --------
        ::

            info = Horses3DReader._parse_stopwatch_file(
                '/data/case_001/RESULTS/solution_p2.Stopwatch.info'
            )
            print(info['events']['TotalTime'])
        """
        events: dict = {}
        with open(path, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                tokens = stripped.split()
                if len(tokens) < 3:
                    continue
                name = tokens[0]
                try:
                    elapsed_s = float(tokens[1])
                    cpu_s     = float(tokens[2])
                except ValueError:
                    continue
                events[name] = {'elapsed_s': elapsed_s, 'cpu_s': cpu_s}

        return {'events': events}

    def _parse_control_file(self, path: str) -> dict:
        """
        Parse a HORSES3D ``.control`` file into a flat dict of
        structurally-relevant fields.

        Only the keys documented in the module docstring are extracted;
        the full free-form control content (boundary conditions,
        surfaces, monitors, …) is intentionally not dumped wholesale
        into metadata.

        Parsing rules
        -------------
        * Lines are ``key = value`` (case-insensitive key, optionally
          quoted value, ``!`` starts a trailing comment).
        * Booleans are Fortran-style (``.true.`` / ``.false.``).
        * Numbers use Fortran ``d`` exponent notation (``1.d-10``) as
          well as standard notation; both are handled.
        * A key that is not present in the file is left absent from the
          returned dict (never set to a placeholder) — callers use
          ``dict.get``.

        Parameters
        ----------
        path : str
            Absolute path to the ``.control`` file.

        Returns
        -------
        dict
            Keys: ``'polynomial_order'`` (int or None), ``'mesh_file'``
            (str or None), ``'solution_file'`` (str or None),
            ``'restart'`` (bool), ``'restart_file'`` (str or None),
            ``'restart_polorder'`` (int or None), ``'params'`` (dict —
            ``'flow_equations'``, ``'mach_number'``,
            ``'reynolds_number'``, ``'aoa_theta'``, ``'aoa_phi'``,
            ``'number_of_time_steps'``, ``'output_interval'``,
            ``'convergence_tolerance'``, ``'simulation_type'``,
            ``'final_time'`` — only the keys actually present in the
            file).

        Examples
        --------
        ::

            parsed = reader._parse_control_file(
                '/data/case_001/file_control_p3.control'
            )
            print(parsed['polynomial_order'], parsed['params']['mach_number'])
        """
        raw: dict = {}
        with open(path, 'r') as fh:
            for line in fh:
                line = line.split('!', 1)[0].strip()
                if not line or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key   = key.strip().lower()
                value = value.strip().strip('"').strip("'").strip()
                if value:
                    raw[key] = value

        def _to_bool(v):
            return v is not None and v.strip().lower() in ('.true.', 'true', 'yes')

        def _to_float(v):
            if v is None:
                return None
            try:
                return float(v.lower().replace('d', 'e'))
            except ValueError:
                return None

        def _to_int(v):
            f = _to_float(v)
            return int(f) if f is not None else None

        polynomial_order = _to_int(raw.get('polynomial order'))
        mesh_file        = raw.get('mesh file name')
        solution_file    = raw.get('solution file name')
        restart          = _to_bool(raw.get('restart'))
        restart_file     = raw.get('restart file name')
        restart_polorder = _to_int(raw.get('restart polorder'))

        param_keys = {
            'flow equations':        ('flow_equations',         str),
            'mach number':           ('mach_number',             _to_float),
            'reynolds number':       ('reynolds_number',         _to_float),
            'aoa theta':             ('aoa_theta',                 _to_float),
            'aoa phi':               ('aoa_phi',                     _to_float),
            'number of time steps':  ('number_of_time_steps',   _to_float),
            'output interval':       ('output_interval',           _to_float),
            'convergence tolerance': ('convergence_tolerance', _to_float),
            'simulation type':       ('simulation_type',           str),
            'final time':            ('final_time',                   _to_float),
            'cfl':                   ('cfl',                        _to_float),
            'dcfl':                  ('dcfl',                   _to_float),
            'dt':                    ('dt',                      _to_float)
        }
        params = {}
        for raw_key, (out_key, caster) in param_keys.items():
            if raw_key in raw:
                val = raw[raw_key]
                params[out_key] = caster(val) if caster is not str else val

        return {
            'polynomial_order': polynomial_order,
            'mesh_file':        mesh_file,
            'solution_file':    solution_file,
            'restart':          restart,
            'restart_file':     restart_file,
            'restart_polorder': restart_polorder,
            'params':           params,
        }

    @classmethod
    def _infer_geometric_g(cls, mesh_filename: str) -> Union[int, None]:
        """
        Opportunistically infer the mesh geometric order ``g`` (NGeo)
        from a filename following the ``...v<version>g<g>...``
        convention (e.g. ``mesh_v2g2.h5`` → ``g=2``).

        Parameters
        ----------
        mesh_filename : str
            Mesh filename (basename or path).

        Returns
        -------
        int or None
            The inferred ``g``, or ``None`` if the filename does not
            match the expected pattern. Never guessed otherwise.

        Examples
        --------
        ::

            Horses3DReader._infer_geometric_g('mesh_v2g2.h5')
            # → 2
            Horses3DReader._infer_geometric_g('mesh.cgns')
            # → None
        """
        m = cls._MESH_G_RE.search(os.path.basename(mesh_filename))
        return int(m.group('g')) if m else None

    @staticmethod
    def _normalise_cases_idx(cases_idx, n_cases: int) -> list:
        """
        Normalise a ``cases_idx`` argument to a sorted list of valid
        integer case indices, mirroring the equivalent helpers in
        ``CODAReader`` / ``NUMPYReader`` / ``CODAStats``.

        Parameters
        ----------
        cases_idx : 'all', int, range, list[int] or tuple[int]
        n_cases : int
            Total number of physical cases (``len(sim_metadata)``).

        Returns
        -------
        list[int]

        Raises
        ------
        ValueError
            If ``cases_idx`` is a string other than ``'all'``, or has an
            unsupported type.
        IndexError
            If any requested index is out of range.

        Examples
        --------
        ::

            idx = Horses3DReader._normalise_cases_idx('all', n_cases=10)
            idx = Horses3DReader._normalise_cases_idx([0, 2], n_cases=10)
        """
        if isinstance(cases_idx, str):
            if cases_idx.lower() == 'all':
                cases_idx = list(range(n_cases))
            else:
                raise ValueError("Invalid string for cases_idx. Use 'all'.")
        elif isinstance(cases_idx, int):
            cases_idx = [cases_idx]
        elif isinstance(cases_idx, range):
            cases_idx = list(cases_idx)
        elif isinstance(cases_idx, (list, tuple)):
            cases_idx = list(cases_idx)
        else:
            raise ValueError(
                "cases_idx must be 'all', int, list[int], tuple[int] or range."
            )

        if any(i >= n_cases or i < 0 for i in cases_idx):
            raise IndexError("cases_idx contains out-of-range values.")

        return sorted(cases_idx)