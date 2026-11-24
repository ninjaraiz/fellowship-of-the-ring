# Graph Report - fellowship-of-the-ring  (2026-11-18)

## Corpus Check
- 73 files · ~164,107 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 44 file(s) not represented in the graph (top: (none) 28, .geo 11, .csv 4)
- WARNING: 44 of 129 files seen (34%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 1973 nodes · 3848 edges · 108 communities (35 shown, 73 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 188 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6126066a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_horses3d_hsol.py
- Horses3DReader
- gandalf/generate_docs.py
- numpy.py
- test_merge_datasets.py
- test_coda_restart.py
- os
- BaseSets
- ARAGORN
- FilenamePattern
- CODAStats
- CODASets
- test_numpy_reader.py
- test_coda_subsets.py
- CodaLog
- CodaCase
- test_coda_monitor.py
- json
- pytest
- coda_monitor.py
- CODASingleSets
- BaseReader
- FotR/__init__.py
- GIMLI
- NUMPYReader
- BaseRing
- test_coda_single.py
- test_rings_gandalf.py
- Weapons
- PYLOMSets
- LazyRegistry
- .extract_inputs
- XFoilCase
- GANDALF
- CODAResiduals
- MonitorServer
- test_earendils_light.py
- _install_stub_modules
- CaseRef
- Horses3DResiduals
- Backpack
- GANDALF
- _signature
- _Fake
- residuals/coda.py
- Backpack
- .restart_stalled_cases
- ndarray
- CODASingleReader
- BaseStats
- gimli.py
- _Handler
- ClusterPlotter
- .prepare
- PYLOMReader
- ._save_figures
- HDF5reader
- BaseClusterModel
- NUMPYFILEReader
- DifferentialOperators
- DictVisualizer
- BaseResiduals
- .create_final_tensor
- _select_model_worker
- MonitorHandle
- test_ragged_bookkeeping_column_is_dropped_with_a_warning
- server
- _Shortcut
- HorsesLazyField
- _LazyModule
- flow_coda.py
- _install_stub_modules
- CODAReader
- ._monitor_folders
- .surface_derivative
- flow_horses3d.py
- flow_numpy.py
- _name_index
- flow_pylom.py
- .sort_closed_curve_by_kdtree
- register_model
- reader
- _install_stub_modules

## God Nodes (most connected - your core abstractions)
1. `FRODO` - 70 edges
2. `Horses3DReader` - 56 edges
3. `CODAResiduals` - 36 edges
4. `SAM` - 36 edges
5. `CODASets` - 30 edges
6. `_make_reader()` - 30 edges
7. `EarendilsLight` - 28 edges
8. `_make_group_dict()` - 28 edges
9. `BaseRing` - 26 edges
10. `BaseReader` - 25 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_coda.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_horses3d.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_numpy.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_pylom.py → FotR/characters/frodo.py
- `test_expand_nodelist()` --uses--> `BaseRing`  [INFERRED]
  FotR/tests/test_rings_gandalf.py → FotR/characters/rings/base.py

## Import Cycles
- None detected.

## Communities (108 total, 73 thin omitted)

### Community 0 - "test_horses3d_hsol.py"
Cohesion: 0.05
Nodes (44): _field(), find_solutions(), HorsesHeader, HorsesLayout, IncompleteFileError, _layout_from_size(), primitive_variables(), read_header() (+36 more)

### Community 1 - "Horses3DReader"
Cohesion: 0.06
Nodes (39): Horses3DReader, _to_float(), _to_int(), basic_dataset(), _make_basic_dataset(), _make_mesh(), test_case_idx_assigned_per_case_not_per_p(), test_case_without_control_raises_in_strict_mode() (+31 more)

### Community 2 - "gandalf/generate_docs.py"
Cohesion: 0.05
Nodes (36): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _formats(), main() (+28 more)

### Community 3 - "numpy.py"
Cohesion: 0.05
Nodes (28): CODASingleStats, gci_root(), _grid(), test_case_ref_rereads_when_nothing_is_materialised(), test_compute_var_broadcasts_a_scalar_expression(), test_compute_var_can_use_design_variables(), test_compute_var_evaluates_per_case(), test_create_jset_concatenates_blocks_of_different_height() (+20 more)

### Community 4 - "test_merge_datasets.py"
Cohesion: 0.08
Nodes (30): fusionar_y_exportar(), FakeFRODO, _FakeResiduals, _FakeSets, _install_stub_modules(), _load_frodo(), _make_df_state(), _make_group() (+22 more)

### Community 5 - "test_coda_restart.py"
Cohesion: 0.10
Nodes (18): _assert_skipped_untouched(), case(), _FakeDB, _FakeReader, _forbid_slurm(), _make_case(), on_flexo(), _snapshot() (+10 more)

### Community 8 - "ARAGORN"
Cohesion: 0.06
Nodes (3): ARAGORN, preexec(), RunResult

### Community 9 - "FilenamePattern"
Cohesion: 0.05
Nodes (4): pattern_pocket, pattern_pocket_ant, FilenamePattern, FilenamePattern

### Community 11 - "CODASets"
Cohesion: 0.06
Nodes (9): CODASets, db(), _FakeDB, _StubReader, test_plot_wall_integrals_all_vars_and_band(), test_plot_wall_integrals_missing_files_warn(), test_plot_wall_integrals_rejects_all_and_empty(), test_plot_wall_integrals_saves_per_variable() (+1 more)

### Community 12 - "test_numpy_reader.py"
Cohesion: 0.15
Nodes (28): _make_group_dict(), _make_reader(), test_define_subset_basic(), test_define_subset_duplicate_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises(), test_define_subset_overwrite(), test_extract_inputs_cases_idx_all() (+20 more)

### Community 14 - "test_coda_subsets.py"
Cohesion: 0.06
Nodes (26): test_compute_var_refuses_to_overwrite_silently(), test_compute_var_rejects_arbitrary_code(), test_reduce_cases_rejects_bookkeeping_variables(), test_richardson_needs_three_meshes(), test_richardson_rejects_an_unknown_triplet_mode(), test_sample_on_rejects_an_unknown_target(), test_save_to_h5_writes_one_subgroup_per_case(), test_base_normalise_cases_idx_all() (+18 more)

### Community 16 - "CodaCase"
Cohesion: 0.11
Nodes (14): CodaCase, find_logs(), make_case(), _read_dat(), _status(), test_average_rate_from_the_date_lines(), test_find_logs_sorts_by_job_id(), test_finished_and_limit() (+6 more)

### Community 17 - "test_coda_monitor.py"
Cohesion: 0.14
Nodes (22): _criteria(), _get(), log_text(), read_log(), server(), stage_text(), test_a_file_that_shrinks_is_read_again(), test_case_rows_are_sent_once_and_reset_on_a_new_log() (+14 more)

### Community 18 - "json"
Cohesion: 0.11
Nodes (14): build(), build_cells(), code(), execute_notebook(), export_html(), main(), md(), pick() (+6 more)

### Community 19 - "pytest"
Cohesion: 0.19
Nodes (11): _FakeDB, _make_group(), sets_obj(), test_save_to_npy_aux_with_case_axis_matches_case_idx(), test_save_to_npy_aux_without_case_axis_kept_unfiltered(), test_save_to_npy_case_order_consistent_across_structures(), test_save_to_npy_exports_multiple_aux_vars(), test_save_to_npy_missing_stage_in_vars_only_raises_keyerror() (+3 more)

### Community 20 - "coda_monitor.py"
Cohesion: 0.15
Nodes (3): _finite(), _parse_date(), launch()

### Community 24 - "FotR/__init__.py"
Cohesion: 0.11
Nodes (5): juntar_db(), main(), symlog(), LEGOLAS, __getattr__()

### Community 27 - "BaseRing"
Cohesion: 0.06
Nodes (6): BaseRing, CODARing, _replace_mesh_placeholder(), _resolve_update_value(), __getattr__(), test_job_name_pairs_and_fallback()

### Community 28 - "test_coda_single.py"
Cohesion: 0.11
Nodes (14): gci_root(), gci_running(), mixed_cell_root(), _quad_row_grid(), test_df_state_is_ordered_by_case_idx(), test_mesh_map_falls_back_to_the_local_copy(), test_mesh_map_is_loaded_and_used(), test_parse_joins_by_folder_and_records_mesh() (+6 more)

### Community 29 - "test_rings_gandalf.py"
Cohesion: 0.15
Nodes (22): CODASingleRing, _cases(), _ring(), _sources(), test_add_param_guards(), test_coda_generate_folders_placeholders(), test_coda_missing_mesh_raises(), test_coda_skip_keeps_folder_for_assign() (+14 more)

### Community 33 - ".extract_inputs"
Cohesion: 0.14
Nodes (16): read_db_CODA(), read_db(), FRODO, db(), test_save_to_npy_round_trips_through_the_numpy_reader(), test_connectivity_is_consistent_with_the_stored_geometry(), test_extract_inputs_does_not_wipe_previously_read_vars(), test_extract_inputs_keeps_per_case_geometry() (+8 more)

### Community 34 - "XFoilCase"
Cohesion: 0.06
Nodes (4): GmshCase, __getattr__(), xfoil_parser(), XFoilCase

### Community 36 - "CODAResiduals"
Cohesion: 0.23
Nodes (9): CODAResiduals, _FakeDB, _make_case(), test_plot_all_final_residuals_1d_legend_has_no_duplicates(), test_plot_all_final_residuals_1d_one_plot_per_stage(), test_plot_all_final_residuals_1d_saves_figure(), test_plot_all_final_residuals_1d_single_axes(), test_plot_all_final_residuals_no_columns_warns() (+1 more)

### Community 37 - "MonitorServer"
Cohesion: 0.10
Nodes (5): main(), MonitorServer, test_a_taken_port_moves_to_the_next(), test_idle_server_stops_by_itself(), test_no_case_is_an_error()

### Community 38 - "test_earendils_light.py"
Cohesion: 0.13
Nodes (8): EarendilsLight, test_help_returns_the_text_it_prints(), test_live_falls_back_to_the_index_when_the_chain_lacks_it(), test_live_follows_the_declared_order(), test_overview_when_asked_for_nothing(), test_several_matches_are_all_listed(), test_static_lookup_renders_signature_and_docstring(), test_unknown_name_says_so_without_raising()

### Community 42 - "Backpack"
Cohesion: 0.12
Nodes (4): Backpack, _binop(), _resolve_callable(), _to_value()

### Community 44 - "_signature"
Cohesion: 0.11
Nodes (9): _doc(), _is_opaque(), _load(), _parameters(), _signature(), _where(), test_every_character_exposes_the_shortcut(), test_opaque_detection_ignores_self() (+1 more)

### Community 45 - "_Fake"
Cohesion: 0.11
Nodes (6): db(), _Fake, _Reader, _ReaderWithShared, _Sets, test_live_skips_empty_slots()

### Community 46 - "residuals/coda.py"
Cohesion: 0.21
Nodes (6): _current_slurm_cluster(), _patch_line(), _restart_script_text(), _sbatch(), test_patch_line_ignores_indented_lines(), test_restart_script_text_reads_ci_name()

### Community 49 - ".restart_stalled_cases"
Cohesion: 0.12
Nodes (7): _output_stage(), _script_from_sh(), _set_sbatch_option(), test_apply_restarts_the_case(), test_output_stage_matches_reader_rule(), test_script_from_sh_ignores_commented_lines(), test_set_sbatch_option_ignores_comments_and_longer_options()

### Community 66 - "DictVisualizer"
Cohesion: 0.17
Nodes (3): DictVisualizer, to_graph(), build()

### Community 75 - "test_ragged_bookkeeping_column_is_dropped_with_a_warning"
Cohesion: 0.27
Nodes (5): _rewrite_metadata(), test_ragged_bookkeeping_column_is_dropped_with_a_warning(), test_ragged_identity_column_is_rejected(), test_stale_folder_column_is_recovered_from_folder_fmt(), mutate()

### Community 77 - "server"
Cohesion: 0.25
Nodes (3): _FakeDB, _FakeReader, residuals()

### Community 89 - "._monitor_folders"
Cohesion: 0.40
Nodes (3): test_case_selection_errors(), test_case_without_folder_is_skipped_with_a_warning(), test_cases_by_subset_and_query()

### Community 94 - "_name_index"
Cohesion: 0.29
Nodes (5): _collect(), _name_index(), test_index_finds_names_across_the_package(), test_index_records_the_qualified_name(), test_index_skips_the_test_suite()

## Knowledge Gaps
- **73 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SAM` connect `os` to `Horses3DReader`, `BaseSets`, `CODASets`, `CODASingleSets`, `FotR/__init__.py`, `BaseRing`, `Weapons`, `PYLOMSets`, `.extract_inputs`, `CODAResiduals`, `test_earendils_light.py`, `Backpack`, `residuals/coda.py`, `CODASingleReader`, `HDF5reader`, `NUMPYFILEReader`, `DifferentialOperators`, `DictVisualizer`, `.create_final_tensor`, `CODAReader`?**
  _High betweenness centrality (0.267) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FRODO` (e.g. with `SAM` and `EarendilsLight`) actually correct?**
  _`FRODO` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_horses3d_hsol.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05209274314965372 - nodes in this community are weakly interconnected._
- **Why does `Backpack` connect `Backpack` to `os`, `FilenamePattern`, `UnstructuredGrid`, `.GMM`, `HorsesLazyField`, `Weapons`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Should `Horses3DReader` be split into smaller, more focused modules?**
  _Cohesion score 0.05708548479632817 - nodes in this community are weakly interconnected._
- **Why does `FRODO` connect `.extract_inputs` to `numpy.py`, `test_merge_datasets.py`, `os`, `.copy`, `.__getattr__`, `.summary_data`, `test_earendils_light.py`, `test_ragged_bookkeeping_column_is_dropped_with_a_warning`, `test_coda_single.py`, `residuals/coda.py`, `CODASingleReader`, `flow_coda.py`, `BaseStats`, `.__init__`, `FotR/__init__.py`, `flow_horses3d.py`, `flow_numpy.py`, `flow_pylom.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Should `gandalf/generate_docs.py` be split into smaller, more focused modules?**
  _Cohesion score 0.052429667519181586 - nodes in this community are weakly interconnected._