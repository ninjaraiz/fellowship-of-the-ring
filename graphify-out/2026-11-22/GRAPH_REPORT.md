# Graph Report - fellowship-of-the-ring  (2026-11-22)

## Corpus Check
- 100 files · ~203,690 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 57 file(s) not represented in the graph (top: (none) 28, .geo 12, .msh 8)
- WARNING: 57 of 171 files seen (33%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 2546 nodes · 5320 edges · 132 communities (57 shown, 75 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 253 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0fe65c6a`
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
- Mesh
- test_coda_subsets.py
- CodaLog
- CodaCase
- test_coda_monitor.py
- .restart_stalled_cases
- test_legolas_io.py
- coda_log.py
- BaseReader
- LEGOLAS
- GIMLI
- PYLOMReader
- os
- gci_running
- test_rings_gandalf.py
- Weapons
- PYLOMSets
- LazyRegistry
- .extract_inputs
- XFoilCase
- ViewerServer
- topology.py
- MonitorServer
- Horses3DRing
- Quality
- CaseRef
- Horses3DResiduals
- Backpack
- GANDALF
- earendils_light.py
- _Fake
- test_legolas_viewer.py
- Backpack
- test_rings_horses3d.py
- json
- ndarray
- CODASingleReader
- launch
- legolas/generate_docs.py
- hex_face_grid
- ClusterPlotter
- gimli.py
- cgns.py
- ._prepare_restart
- test_save_to_npy.py
- HDF5reader
- BaseClusterModel
- types
- NUMPYReader
- DifferentialOperators
- DictVisualizer
- Viewer
- _rewrite_control
- .create_final_tensor
- GANDALF
- _select_model_worker
- BaseStats
- build_gci_report.py
- test_ragged_bookkeeping_column_is_dropped_with_a_warning
- server
- frodo_sam/generate_docs.py
- _Shortcut
- HorsesLazyField
- _LazyModule
- flow_coda.py
- test_earendils_light.py
- .generate_folders
- CODAReader
- .do_GET
- .surface_derivative
- NUMPYFILEReader
- solver_tools/__init__.py
- ViewerHandle
- .sort_closed_curve_by_kdtree
- EarendilsLight
- flow_numpy.py
- Census
- .fit
- earendils_light.py
- GmshCase
- flow_horses3d.py
- DataFrame
- is_loaded
- flow_pylom.py
- _dedent
- test_legolas_quality.py
- _registry_entries
- _registry_entries
- make_solver_repo
- test_fsp_report_and_load_campaign
- rings/horses3d.py
- test_import_needs_the_standard_library_only
- _collect

## God Nodes (most connected - your core abstractions)
1. `FRODO` - 70 edges
2. `Horses3DReader` - 65 edges
3. `Mesh` - 51 edges
4. `LEGOLAS` - 50 edges
5. `CODAResiduals` - 36 edges
6. `SAM` - 35 edges
7. `box_mesh()` - 32 edges
8. `CODASets` - 30 edges
9. `CellBlock` - 30 edges
10. `_make_reader()` - 30 edges

## Surprising Connections (you probably didn't know these)
- `test_expand_nodelist()` --uses--> `BaseRing`  [INFERRED]
  FotR/tests/test_rings_gandalf.py → FotR/characters/rings/base.py
- `main()` --calls--> `GANDALF`  [EXTRACTED]
  examples/gci_codasingle.py → FotR/characters/gandalf.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_coda.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_horses3d.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_numpy.py → FotR/characters/frodo.py

## Import Cycles
- 4-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/census.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`

## Communities (132 total, 75 thin omitted)

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
Nodes (22): CODASingleStats, gci_root(), _grid(), test_case_ref_rereads_when_nothing_is_materialised(), test_compute_var_broadcasts_a_scalar_expression(), test_compute_var_can_use_design_variables(), test_compute_var_evaluates_per_case(), test_create_jset_concatenates_blocks_of_different_height() (+14 more)

### Community 4 - "test_merge_datasets.py"
Cohesion: 0.08
Nodes (30): fusionar_y_exportar(), FakeFRODO, _FakeResiduals, _FakeSets, _install_stub_modules(), _load_frodo(), _make_df_state(), _make_group() (+22 more)

### Community 5 - "test_coda_restart.py"
Cohesion: 0.10
Nodes (18): _assert_skipped_untouched(), case(), _FakeDB, _FakeReader, _forbid_slurm(), _make_case(), on_flexo(), _snapshot() (+10 more)

### Community 7 - "SAM"
Cohesion: 0.10
Nodes (5): SAM, BaseSets, CODASingleSets, NUMPYFILESets, lazy_module()

### Community 8 - "ARAGORN"
Cohesion: 0.07
Nodes (3): ARAGORN, preexec(), RunResult

### Community 9 - "FilenamePattern"
Cohesion: 0.05
Nodes (4): pattern_pocket, pattern_pocket_ant, FilenamePattern, FilenamePattern

### Community 11 - "CODASets"
Cohesion: 0.06
Nodes (9): CODASets, db(), _FakeDB, _StubReader, test_plot_wall_integrals_all_vars_and_band(), test_plot_wall_integrals_missing_files_warn(), test_plot_wall_integrals_rejects_all_and_empty(), test_plot_wall_integrals_saves_per_variable() (+1 more)

### Community 12 - "test_numpy_reader.py"
Cohesion: 0.14
Nodes (29): _install_stub_modules(), _make_group_dict(), _make_reader(), test_define_subset_basic(), test_define_subset_duplicate_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises(), test_define_subset_overwrite() (+21 more)

### Community 13 - "Mesh"
Cohesion: 0.08
Nodes (10): merge_points(), read(), _decode(), read(), CellBlock, Mesh, test_watertight_flags_cells_touching_along_an_edge(), test_mesh_from_memory_and_repr() (+2 more)

### Community 14 - "test_coda_subsets.py"
Cohesion: 0.06
Nodes (20): test_compute_var_refuses_to_overwrite_silently(), test_compute_var_rejects_arbitrary_code(), test_create_jset_drops_variables_missing_in_some_case(), test_reduce_cases_rejects_bookkeeping_variables(), test_richardson_needs_three_meshes(), test_richardson_rejects_an_unknown_triplet_mode(), test_sample_on_rejects_an_unknown_target(), test_save_to_h5_writes_one_subgroup_per_case() (+12 more)

### Community 16 - "CodaCase"
Cohesion: 0.13
Nodes (12): CodaCase, make_case(), _read_dat(), _status(), test_average_rate_from_the_date_lines(), test_finished_and_limit(), test_live_rate_and_eta(), test_loading_then_pending_without_log() (+4 more)

### Community 17 - "test_coda_monitor.py"
Cohesion: 0.16
Nodes (23): _criteria(), _get(), log_text(), read_log(), server(), stage_text(), test_a_file_that_shrinks_is_read_again(), test_case_rows_are_sent_once_and_reset_on_a_new_log() (+15 more)

### Community 18 - ".restart_stalled_cases"
Cohesion: 0.20
Nodes (3): _output_stage(), test_apply_restarts_the_case(), test_output_stage_matches_reader_rule()

### Community 19 - "test_legolas_io.py"
Cohesion: 0.08
Nodes (31): _check_order(), detect_format(), _hdf5_flavour(), read(), tensor_monotonic(), _bc(), _cgns_node(), _cgns_str() (+23 more)

### Community 20 - "coda_log.py"
Cohesion: 0.17
Nodes (4): find_logs(), _finite(), _parse_date(), test_find_logs_sorts_by_job_id()

### Community 24 - "LEGOLAS"
Cohesion: 0.08
Nodes (33): read_control_bcs(), read_ini_bcs(), _strip(), LEGOLAS, Boundary, hex_face_grid(), box_mesh(), box_with_inner() (+25 more)

### Community 27 - "os"
Cohesion: 0.10
Nodes (3): main(), symlog(), __getattr__()

### Community 28 - "gci_running"
Cohesion: 0.20
Nodes (7): gci_root(), gci_running(), mixed_cell_root(), _quad_row_grid(), _write_case(), _write_monitor(), _write_stage()

### Community 29 - "test_rings_gandalf.py"
Cohesion: 0.15
Nodes (22): CODASingleRing, _cases(), _ring(), _sources(), test_add_param_guards(), test_coda_generate_folders_placeholders(), test_coda_missing_mesh_raises(), test_coda_skip_keeps_folder_for_assign() (+14 more)

### Community 33 - ".extract_inputs"
Cohesion: 0.10
Nodes (23): juntar_db(), read_db_CODA(), read_db(), FRODO, db(), test_save_to_npy_round_trips_through_the_numpy_reader(), test_connectivity_is_consistent_with_the_stored_geometry(), test_df_state_is_ordered_by_case_idx() (+15 more)

### Community 36 - "topology.py"
Cohesion: 0.11
Nodes (13): _corner_range(), section_faces(), all_faces(), coincident_labels(), exterior_faces(), face_area(), face_index(), face_keys() (+5 more)

### Community 37 - "MonitorServer"
Cohesion: 0.09
Nodes (7): MonitorServer, test_a_taken_port_moves_to_the_next(), test_case_selection_errors(), test_case_without_folder_is_skipped_with_a_warning(), test_cases_by_subset_and_query(), test_idle_server_stops_by_itself(), test_no_case_is_an_error()

### Community 38 - "Horses3DRing"
Cohesion: 0.12
Nodes (4): Horses3DRing, test_cases_must_be_defined(), test_fsp_report_running_and_pending(), test_submit_cases_in_a_new_session()

### Community 39 - "Quality"
Cohesion: 0.12
Nodes (3): _angles(), _edges(), Quality

### Community 42 - "Backpack"
Cohesion: 0.12
Nodes (4): Backpack, _binop(), _resolve_callable(), _to_value()

### Community 44 - "earendils_light.py"
Cohesion: 0.13
Nodes (8): _doc(), _is_opaque(), _parameters(), _signature(), _where(), test_every_character_exposes_the_shortcut(), test_opaque_detection_ignores_self(), test_signature_drops_self()

### Community 45 - "_Fake"
Cohesion: 0.11
Nodes (6): db(), _Fake, _Reader, _ReaderWithShared, _Sets, test_live_skips_empty_slots()

### Community 46 - "test_legolas_viewer.py"
Cohesion: 0.07
Nodes (21): cell_info(), face_geometry(), _hex_edges(), origin(), pack(), _row_boundary(), _tris_per_face(), zone_faces() (+13 more)

### Community 48 - "test_rings_horses3d.py"
Cohesion: 0.16
Nodes (25): _fake_sbatch(), _generate(), _keys(), make_mesh(), _ring(), test_a_real_boundary_left_out_is_an_error(), test_case_folders_and_controls(), test_controls_lose_comments_but_keep_blocks() (+17 more)

### Community 49 - "json"
Cohesion: 0.14
Nodes (6): launch(), _slurm_node_cpus(), launch(), main(), test_import_needs_numpy_only(), test_launch_reports_a_failing_server()

### Community 53 - "legolas/generate_docs.py"
Cohesion: 0.11
Nodes (18): _arg(), _class_cards(), _collect_classes(), _decorators(), _descriptor_methods(), _doc_info(), _formats(), _function_card() (+10 more)

### Community 54 - "hex_face_grid"
Cohesion: 0.11
Nodes (6): _corner_ijk(), hex_corners(), hex_face_corners(), quad_corners(), tensor_index(), test_face_grids_point_outward()

### Community 56 - "gimli.py"
Cohesion: 0.15
Nodes (4): ClusterDataset, ClusterMetrics, register_model(), _resolve_scaler()

### Community 57 - "cgns.py"
Cohesion: 0.09
Nodes (17): _add_boundaries(), _children(), _face_corner_check(), _label(), _merge_block(), _name(), read(), _section() (+9 more)

### Community 58 - "._prepare_restart"
Cohesion: 0.12
Nodes (11): _current_slurm_cluster(), _patch_line(), _restart_script_text(), _sbatch(), _script_from_sh(), _set_sbatch_option(), _slurm_active_workdirs(), test_patch_line_ignores_indented_lines() (+3 more)

### Community 59 - "test_save_to_npy.py"
Cohesion: 0.09
Nodes (25): _FakeDB, _make_group(), sets_obj(), test_normalise_cases_idx_bad_string_raises(), test_normalise_cases_idx_bad_type_raises(), test_normalise_cases_idx_out_of_range_raises(), test_save_to_npy_aux_with_case_axis_matches_case_idx(), test_save_to_npy_aux_without_case_axis_kept_unfiltered() (+17 more)

### Community 66 - "DictVisualizer"
Cohesion: 0.17
Nodes (3): DictVisualizer, to_graph(), build()

### Community 68 - "_rewrite_control"
Cohesion: 0.29
Nodes (3): _defined_zones(), _referenced_zones(), _strip_comment()

### Community 74 - "build_gci_report.py"
Cohesion: 0.26
Nodes (9): build(), build_cells(), code(), execute_notebook(), export_html(), main(), md(), pick() (+1 more)

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
Cohesion: 0.12
Nodes (8): _name_index(), test_from_fotr_import_gives_the_class_whatever_was_imported_first(), test_importing_costs_no_third_party_module(), test_index_finds_names_across_the_package(), test_index_records_the_qualified_name(), test_index_skips_the_test_suite(), test_live_falls_back_to_the_index_when_the_chain_lacks_it(), test_live_follows_the_declared_order()

### Community 87 - ".generate_folders"
Cohesion: 0.11
Nodes (8): _control_keys(), _fmt(), _norm_key(), _quoted(), _read_mesh_info(), _rewrite_control(), test_fsp_control(), test_rewrite_control_keeps_quoted_bang_and_adds_missing_keys()

### Community 97 - "EarendilsLight"
Cohesion: 0.14
Nodes (6): EarendilsLight, test_help_returns_the_text_it_prints(), test_overview_when_asked_for_nothing(), test_several_matches_are_all_listed(), test_static_lookup_renders_signature_and_docstring(), test_unknown_name_says_so_without_raising()

### Community 100 - "Census"
Cohesion: 0.06
Nodes (3): Census, _vec(), _install_stub_modules()

### Community 105 - "DataFrame"
Cohesion: 0.12
Nodes (9): CODAResiduals, _FakeDB, _make_case(), test_plot_all_final_residuals_1d_legend_has_no_duplicates(), test_plot_all_final_residuals_1d_one_plot_per_stage(), test_plot_all_final_residuals_1d_saves_figure(), test_plot_all_final_residuals_1d_single_axes(), test_plot_all_final_residuals_no_columns_warns() (+1 more)

### Community 111 - "_dedent"
Cohesion: 0.33
Nodes (3): _dedent(), _parse_entries(), _parse_google_entries()

### Community 112 - "test_legolas_quality.py"
Cohesion: 0.10
Nodes (20): _corner_triples(), gauss_lobatto(), _hex_jacobian(), lagrange(), _linear_jacobian(), mapped_box(), one(), test_aspect_ratio_skewness_and_neighbours() (+12 more)

### Community 113 - "_registry_entries"
Cohesion: 0.67
Nodes (3): _registry_entries(), entries_from_dict(), value_of()

### Community 114 - "_registry_entries"
Cohesion: 0.67
Nodes (3): _registry_entries(), entries_from_dict(), value_of()

### Community 116 - "make_solver_repo"
Cohesion: 0.33
Nodes (4): make_solver_repo(), repo(), src(), test_submit_cases_checks_the_repository()

### Community 118 - "test_fsp_report_and_load_campaign"
Cohesion: 0.50
Nodes (3): test_fsp_report_and_load_campaign(), _uniform(), _write_hsol()

### Community 119 - "rings/horses3d.py"
Cohesion: 0.06
Nodes (6): BaseRing, CODARing, _replace_mesh_placeholder(), _resolve_update_value(), __getattr__(), test_job_name_pairs_and_fallback()

## Knowledge Gaps
- **75 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SAM` connect `SAM` to `.extract_inputs`, `DictVisualizer`, `DifferentialOperators`, `EarendilsLight`, `.create_final_tensor`, `DataFrame`, `Backpack`, `CODASets`, `HDF5reader`, `CODASingleReader`, `rings/horses3d.py`, `CODAReader`, `._prepare_restart`, `os`, `NUMPYFILEReader`, `Weapons`, `PYLOMSets`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FRODO` (e.g. with `SAM` and `EarendilsLight`) actually correct?**
  _`FRODO` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_horses3d_hsol.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05209274314965372 - nodes in this community are weakly interconnected._
- **Why does `Backpack` connect `Backpack` to `SAM`, `FilenamePattern`, `UnstructuredGrid`, `.GMM`, `HorsesLazyField`, `Weapons`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Should `Horses3DReader` be split into smaller, more focused modules?**
  _Cohesion score 0.05307017543859649 - nodes in this community are weakly interconnected._
- **Why does `FRODO` connect `.extract_inputs` to `.copy`, `EarendilsLight`, `.__getattr__`, `flow_numpy.py`, `test_merge_datasets.py`, `numpy.py`, `SAM`, `flow_horses3d.py`, `BaseStats`, `test_ragged_bookkeeping_column_is_dropped_with_a_warning`, `flow_pylom.py`, `CODASingleReader`, `flow_coda.py`, `.__init__`, `rings/horses3d.py`, `.summary_data`, `._prepare_restart`, `os`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Should `gandalf/generate_docs.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10795454545454546 - nodes in this community are weakly interconnected._