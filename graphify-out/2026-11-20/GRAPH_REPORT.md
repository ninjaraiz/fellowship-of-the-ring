# Graph Report - fellowship-of-the-ring  (2026-11-20)

## Corpus Check
- 97 files · ~199,006 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 57 file(s) not represented in the graph (top: (none) 28, .geo 12, .msh 8)
- WARNING: 57 of 168 files seen (34%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 2479 nodes · 5133 edges · 122 communities (55 shown, 67 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 248 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `22082267`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_horses3d_hsol.py
- Horses3DReader
- gandalf/generate_docs.py
- numpy.py
- test_merge_datasets.py
- test_coda_restart.py
- _Handler
- SAM
- ARAGORN
- FilenamePattern
- CODAStats
- CODASets
- test_numpy_reader.py
- test_legolas_viewer.py
- test_coda_subsets.py
- CodaLog
- CodaCase
- test_coda_monitor.py
- subprocess
- test_legolas_io.py
- viewer.py
- CODASingleSets
- BaseReader
- LEGOLAS
- GIMLI
- NUMPYReader
- os
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
- Horses3DRing
- _install_stub_modules
- CaseRef
- Horses3DResiduals
- Backpack
- GANDALF
- earendils_light.py
- _Fake
- Mesh
- Backpack
- test_rings_horses3d.py
- ViewerServer
- ndarray
- CODASingleReader
- launch
- legolas/generate_docs.py
- CellBlock
- ClusterPlotter
- gimli.py
- cgns.py
- ._prepare_restart
- test_save_to_npy.py
- HDF5reader
- BaseClusterModel
- types
- NUMPYFILEReader
- DifferentialOperators
- DictVisualizer
- Viewer
- .generate_folders
- .create_final_tensor
- .richardson
- _select_model_worker
- BaseStats
- BaseSets
- .fit
- test_ragged_bookkeeping_column_is_dropped_with_a_warning
- server
- frodo_sam/generate_docs.py
- _Shortcut
- HorsesLazyField
- _LazyModule
- flow_coda.py
- test_earendils_light.py
- gci_root
- CODAReader
- ._monitor_folders
- .surface_derivative
- ._job_name
- aragorn.py
- .sort_closed_curve_by_kdtree
- EarendilsLight
- rings/__init__.py
- Census
- BaseResiduals
- ._static
- GmshCase
- test_richardson_recovers_a_known_order
- is_loaded
- test_sets_imports_without_pylom
- _install_stub_modules
- _dedent
- _registry_entries
- _registry_entries
- make_solver_repo
- rings/horses3d.py

## God Nodes (most connected - your core abstractions)
1. `FRODO` - 70 edges
2. `Horses3DReader` - 65 edges
3. `Mesh` - 49 edges
4. `LEGOLAS` - 40 edges
5. `CODAResiduals` - 36 edges
6. `SAM` - 35 edges
7. `CODASets` - 30 edges
8. `_make_reader()` - 30 edges
9. `BaseRing` - 28 edges
10. `_make_group_dict()` - 28 edges

## Surprising Connections (you probably didn't know these)
- `test_expand_nodelist()` --uses--> `BaseRing`  [INFERRED]
  FotR/tests/test_rings_gandalf.py → FotR/characters/rings/base.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_coda.py → FotR/characters/frodo.py
- `main()` --calls--> `GANDALF`  [EXTRACTED]
  examples/gci_codasingle.py → FotR/characters/gandalf.py
- `LEGOLAS` --uses--> `Census`  [INFERRED]
  FotR/characters/legolas/__init__.py → FotR/characters/legolas/census.py
- `ViewerServer` --uses--> `Census`  [INFERRED]
  FotR/characters/legolas/viewer.py → FotR/characters/legolas/census.py

## Import Cycles
- 4-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/census.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`

## Communities (122 total, 67 thin omitted)

### Community 0 - "test_horses3d_hsol.py"
Cohesion: 0.05
Nodes (44): _field(), find_solutions(), HorsesHeader, HorsesLayout, IncompleteFileError, _layout_from_size(), primitive_variables(), read_header() (+36 more)

### Community 1 - "Horses3DReader"
Cohesion: 0.05
Nodes (46): Horses3DReader, _to_float(), _to_int(), basic_dataset(), _make_basic_dataset(), _make_mesh(), _one_case(), test_case_idx_assigned_per_case_not_per_p() (+38 more)

### Community 2 - "gandalf/generate_docs.py"
Cohesion: 0.11
Nodes (16): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _google_split(), main() (+8 more)

### Community 3 - "numpy.py"
Cohesion: 0.05
Nodes (23): test_case_ref_rereads_when_nothing_is_materialised(), test_compute_var_broadcasts_a_scalar_expression(), test_compute_var_can_use_design_variables(), test_compute_var_evaluates_per_case(), test_compute_var_refuses_to_overwrite_silently(), test_compute_var_rejects_arbitrary_code(), test_create_jset_concatenates_blocks_of_different_height(), test_create_jset_drops_variables_missing_in_some_case() (+15 more)

### Community 4 - "test_merge_datasets.py"
Cohesion: 0.08
Nodes (30): fusionar_y_exportar(), FakeFRODO, _FakeResiduals, _FakeSets, _install_stub_modules(), _load_frodo(), _make_df_state(), _make_group() (+22 more)

### Community 5 - "test_coda_restart.py"
Cohesion: 0.09
Nodes (21): _output_stage(), _assert_skipped_untouched(), case(), _FakeDB, _FakeReader, _forbid_slurm(), _make_case(), on_flexo() (+13 more)

### Community 8 - "ARAGORN"
Cohesion: 0.07
Nodes (3): ARAGORN, preexec(), RunResult

### Community 9 - "FilenamePattern"
Cohesion: 0.05
Nodes (4): pattern_pocket, pattern_pocket_ant, FilenamePattern, FilenamePattern

### Community 11 - "CODASets"
Cohesion: 0.06
Nodes (20): CODASets, db(), _FakeDB, _StubReader, test_plot_wall_integrals_all_vars_and_band(), test_plot_wall_integrals_missing_files_warn(), test_plot_wall_integrals_rejects_all_and_empty(), test_plot_wall_integrals_saves_per_variable() (+12 more)

### Community 12 - "test_numpy_reader.py"
Cohesion: 0.17
Nodes (24): _make_group_dict(), _make_reader(), test_define_subset_basic(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises(), test_define_subset_overwrite(), test_extract_inputs_cases_idx_all(), test_extract_inputs_cases_idx_int() (+16 more)

### Community 13 - "test_legolas_viewer.py"
Cohesion: 0.08
Nodes (22): _corner_range(), face_geometry(), origin(), pack(), section_faces(), _tris_per_face(), zone_faces(), zone_list() (+14 more)

### Community 14 - "test_coda_subsets.py"
Cohesion: 0.07
Nodes (10): reader(), test_base_normalise_cases_idx_all(), test_base_normalise_cases_idx_invalid_string_raises(), test_base_normalise_cases_idx_out_of_range_raises(), test_define_subset_empty_name_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises(), test_get_subset_missing_raises() (+2 more)

### Community 16 - "CodaCase"
Cohesion: 0.11
Nodes (14): CodaCase, find_logs(), make_case(), _read_dat(), _status(), test_average_rate_from_the_date_lines(), test_find_logs_sorts_by_job_id(), test_finished_and_limit() (+6 more)

### Community 17 - "test_coda_monitor.py"
Cohesion: 0.15
Nodes (23): _criteria(), _get(), log_text(), read_log(), server(), stage_text(), test_a_file_that_shrinks_is_read_again(), test_case_rows_are_sent_once_and_reset_on_a_new_log() (+15 more)

### Community 18 - "subprocess"
Cohesion: 0.10
Nodes (8): _current_slurm_cluster(), _sbatch(), _slurm_active_workdirs(), _slurm_node_cpus(), test_from_fotr_import_gives_the_class_whatever_was_imported_first(), test_importing_costs_no_third_party_module(), test_import_needs_numpy_only(), test_import_is_light()

### Community 19 - "test_legolas_io.py"
Cohesion: 0.07
Nodes (38): _check_order(), detect_format(), _hdf5_flavour(), read(), _corner_ijk(), hex_corners(), hex_face_corners(), hex_face_grid() (+30 more)

### Community 20 - "viewer.py"
Cohesion: 0.15
Nodes (3): _finite(), _parse_date(), test_idle_timeout_stops_the_server()

### Community 24 - "LEGOLAS"
Cohesion: 0.10
Nodes (26): LEGOLAS, Boundary, box_mesh(), box_with_inner(), files(), _levels(), _need(), test_check_bcs_clean_case() (+18 more)

### Community 27 - "os"
Cohesion: 0.12
Nodes (4): juntar_db(), main(), symlog(), __getattr__()

### Community 28 - "test_coda_single.py"
Cohesion: 0.11
Nodes (14): gci_root(), gci_running(), mixed_cell_root(), _quad_row_grid(), test_df_state_is_ordered_by_case_idx(), test_mesh_map_falls_back_to_the_local_copy(), test_mesh_map_is_loaded_and_used(), test_parse_joins_by_folder_and_records_mesh() (+6 more)

### Community 29 - "test_rings_gandalf.py"
Cohesion: 0.16
Nodes (21): CODASingleRing, _cases(), _ring(), _sources(), test_add_param_guards(), test_coda_generate_folders_placeholders(), test_coda_missing_mesh_raises(), test_coda_skip_keeps_folder_for_assign() (+13 more)

### Community 33 - ".extract_inputs"
Cohesion: 0.08
Nodes (22): log(), run(), log(), run(), log(), run(), read_db_CODA(), read_db() (+14 more)

### Community 35 - "GANDALF"
Cohesion: 0.10
Nodes (3): GANDALF, test_facade_delegates_and_blocks_backpack(), test_facade_setattr_copy_pickle()

### Community 36 - "CODAResiduals"
Cohesion: 0.23
Nodes (9): CODAResiduals, _FakeDB, _make_case(), test_plot_all_final_residuals_1d_legend_has_no_duplicates(), test_plot_all_final_residuals_1d_one_plot_per_stage(), test_plot_all_final_residuals_1d_saves_figure(), test_plot_all_final_residuals_1d_single_axes(), test_plot_all_final_residuals_no_columns_warns() (+1 more)

### Community 37 - "MonitorServer"
Cohesion: 0.11
Nodes (5): main(), MonitorServer, test_a_taken_port_moves_to_the_next(), test_idle_server_stops_by_itself(), test_no_case_is_an_error()

### Community 38 - "Horses3DRing"
Cohesion: 0.09
Nodes (4): Horses3DRing, test_fsp_report_and_load_campaign(), _uniform(), _write_hsol()

### Community 42 - "Backpack"
Cohesion: 0.12
Nodes (4): Backpack, _binop(), _resolve_callable(), _to_value()

### Community 44 - "earendils_light.py"
Cohesion: 0.13
Nodes (8): _collect(), _doc(), _is_opaque(), _parameters(), _signature(), _where(), test_opaque_detection_ignores_self(), test_signature_drops_self()

### Community 45 - "_Fake"
Cohesion: 0.11
Nodes (6): db(), _Fake, _Reader, _ReaderWithShared, _Sets, test_live_skips_empty_slots()

### Community 46 - "Mesh"
Cohesion: 0.07
Nodes (16): merge_points(), read(), Mesh, cell_info(), _row_boundary(), all_faces(), coincident_labels(), exterior_faces() (+8 more)

### Community 48 - "test_rings_horses3d.py"
Cohesion: 0.16
Nodes (26): _generate(), _keys(), make_mesh(), _ring(), test_a_real_boundary_left_out_is_an_error(), test_case_folders_and_controls(), test_cases_must_be_defined(), test_controls_lose_comments_but_keep_blocks() (+18 more)

### Community 49 - "ViewerServer"
Cohesion: 0.06
Nodes (12): build(), build_cells(), code(), execute_notebook(), export_html(), main(), md(), pick() (+4 more)

### Community 52 - "launch"
Cohesion: 0.20
Nodes (3): launch(), MonitorHandle, test_launch_runs_the_server_in_its_own_process()

### Community 53 - "legolas/generate_docs.py"
Cohesion: 0.11
Nodes (18): _arg(), _class_cards(), _collect_classes(), _decorators(), _descriptor_methods(), _doc_info(), _formats(), _function_card() (+10 more)

### Community 54 - "CellBlock"
Cohesion: 0.15
Nodes (3): _decode(), read(), CellBlock

### Community 56 - "gimli.py"
Cohesion: 0.15
Nodes (4): ClusterDataset, ClusterMetrics, register_model(), _resolve_scaler()

### Community 57 - "cgns.py"
Cohesion: 0.10
Nodes (17): _add_boundaries(), _children(), _face_corner_check(), _label(), _merge_block(), _name(), read(), _section() (+9 more)

### Community 58 - "._prepare_restart"
Cohesion: 0.14
Nodes (8): _patch_line(), _restart_script_text(), _script_from_sh(), _set_sbatch_option(), test_patch_line_ignores_indented_lines(), test_restart_script_text_reads_ci_name(), test_script_from_sh_ignores_commented_lines(), test_set_sbatch_option_ignores_comments_and_longer_options()

### Community 59 - "test_save_to_npy.py"
Cohesion: 0.09
Nodes (20): test_define_subset_duplicate_name_raises(), test_resolve_cases_idx_unknown_subset_raises(), test_define_subset_duplicate_raises(), test_get_subset_unknown_group_raises_keyerror(), test_remove_subset(), test_remove_subset_missing_raises(), test_normalise_cases_idx_bad_string_raises(), test_normalise_cases_idx_bad_type_raises() (+12 more)

### Community 66 - "DictVisualizer"
Cohesion: 0.17
Nodes (3): DictVisualizer, to_graph(), build()

### Community 67 - "Viewer"
Cohesion: 0.11
Nodes (4): launch(), Viewer, ViewerHandle, test_launch_reports_a_failing_server()

### Community 68 - ".generate_folders"
Cohesion: 0.11
Nodes (10): _control_keys(), _defined_zones(), _fmt(), _norm_key(), _quoted(), _read_mesh_info(), _referenced_zones(), _rewrite_control() (+2 more)

### Community 75 - "test_ragged_bookkeeping_column_is_dropped_with_a_warning"
Cohesion: 0.27
Nodes (5): _rewrite_metadata(), test_ragged_bookkeeping_column_is_dropped_with_a_warning(), test_ragged_identity_column_is_rejected(), test_stale_folder_column_is_recovered_from_folder_fmt(), mutate()

### Community 77 - "server"
Cohesion: 0.25
Nodes (3): _FakeDB, _FakeReader, residuals()

### Community 78 - "frodo_sam/generate_docs.py"
Cohesion: 0.12
Nodes (14): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _formats(), main() (+6 more)

### Community 85 - "test_earendils_light.py"
Cohesion: 0.14
Nodes (7): _name_index(), test_every_character_exposes_the_shortcut(), test_index_finds_names_across_the_package(), test_index_records_the_qualified_name(), test_index_skips_the_test_suite(), test_live_falls_back_to_the_index_when_the_chain_lacks_it(), test_live_follows_the_declared_order()

### Community 87 - "gci_root"
Cohesion: 0.33
Nodes (3): gci_root(), _grid(), _write_monitor()

### Community 89 - "._monitor_folders"
Cohesion: 0.40
Nodes (3): test_case_selection_errors(), test_case_without_folder_is_skipped_with_a_warning(), test_cases_by_subset_and_query()

### Community 97 - "EarendilsLight"
Cohesion: 0.14
Nodes (6): EarendilsLight, test_help_returns_the_text_it_prints(), test_overview_when_asked_for_nothing(), test_several_matches_are_all_listed(), test_static_lookup_renders_signature_and_docstring(), test_unknown_name_says_so_without_raising()

### Community 100 - "Census"
Cohesion: 0.08
Nodes (8): read_control_bcs(), read_ini_bcs(), _strip(), Census, _vec(), test_read_control_bcs(), test_read_control_bcs_unclosed(), test_read_ini_bcs()

### Community 111 - "_dedent"
Cohesion: 0.33
Nodes (3): _dedent(), _parse_entries(), _parse_google_entries()

### Community 113 - "_registry_entries"
Cohesion: 0.67
Nodes (3): _registry_entries(), entries_from_dict(), value_of()

### Community 114 - "_registry_entries"
Cohesion: 0.67
Nodes (3): _registry_entries(), entries_from_dict(), value_of()

### Community 116 - "make_solver_repo"
Cohesion: 0.40
Nodes (3): make_solver_repo(), repo(), src()

### Community 119 - "rings/horses3d.py"
Cohesion: 0.10
Nodes (4): BaseRing, CODARing, _replace_mesh_placeholder(), _resolve_update_value()

## Knowledge Gaps
- **67 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SAM` connect `SAM` to `NUMPYFILEReader`, `.extract_inputs`, `DictVisualizer`, `DifferentialOperators`, `CODAResiduals`, `.create_final_tensor`, `EarendilsLight`, `BaseSets`, `Backpack`, `CODASets`, `CODASingleReader`, `CODASingleSets`, `rings/horses3d.py`, `CODAReader`, `os`, `HDF5reader`, `Weapons`, `PYLOMSets`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FRODO` (e.g. with `SAM` and `EarendilsLight`) actually correct?**
  _`FRODO` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_horses3d_hsol.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05209274314965372 - nodes in this community are weakly interconnected._
- **Why does `Backpack` connect `Backpack` to `SAM`, `FilenamePattern`, `UnstructuredGrid`, `.GMM`, `HorsesLazyField`, `Weapons`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Should `Horses3DReader` be split into smaller, more focused modules?**
  _Cohesion score 0.05307017543859649 - nodes in this community are weakly interconnected._
- **Why does `FRODO` connect `.extract_inputs` to `EarendilsLight`, `numpy.py`, `test_merge_datasets.py`, `SAM`, `BaseStats`, `test_ragged_bookkeeping_column_is_dropped_with_a_warning`, `CODASingleReader`, `flow_coda.py`, `.__init__`, `os`, `test_coda_single.py`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Should `gandalf/generate_docs.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10795454545454546 - nodes in this community are weakly interconnected._