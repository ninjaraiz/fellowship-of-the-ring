# Graph Report - fellowship-of-the-ring  (2026-11-24)

## Corpus Check
- 115 files · ~225,922 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 57 file(s) not represented in the graph (top: (none) 28, .geo 12, .msh 8)
- WARNING: 57 of 186 files seen (31%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 2840 nodes · 6133 edges · 146 communities (64 shown, 82 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 267 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d3a4f18c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_horses3d_hsol.py
- Horses3DReader
- gandalf/generate_docs.py
- test_coda_single_sets.py
- test_merge_datasets.py
- test_coda_restart.py
- MonitorServer
- SAM
- ARAGORN
- FilenamePattern
- CODAStats
- CODASets
- test_numpy_reader.py
- Census
- test_coda_subsets.py
- CodaLog
- Resolution
- test_coda_monitor.py
- json
- test_legolas_io.py
- Convert
- viewer.py
- NUMPYFILEReader
- LEGOLAS
- GIMLI
- test_horses2plt.py
- census.py
- gci_running
- test_rings_gandalf.py
- Weapons
- PYLOMSets
- solver_tools/horses3d.py
- .extract_inputs
- XFoilCase
- ViewerServer
- topology.py
- Mesh
- Horses3DRing
- numpy.py
- CaseRef
- Horses3DResiduals
- Backpack
- GANDALF
- earendils_light.py
- _Fake
- CODAResiduals
- Backpack
- test_rings_horses3d.py
- numpy.py
- ndarray
- CODASingleReader
- MonitorHandle
- legolas/generate_docs.py
- BaseReader
- ClusterPlotter
- gimli.py
- cgns.py
- Solution
- test_save_to_npy.py
- HDF5reader
- BaseClusterModel
- os
- types
- LazyRegistry
- DifferentialOperators
- DictVisualizer
- test_legolas_locate.py
- .create_final_tensor
- .do_GET
- _select_model_worker
- BaseStats
- pattern_pocket_ant
- coda_monitor.py
- server
- frodo_sam/generate_docs.py
- _Shortcut
- HorsesLazyField
- _LazyModule
- element_coords
- test_earendils_light.py
- cell_info
- rings/horses3d.py
- CODAReader
- residuals/coda.py
- .surface_derivative
- read_header
- ._normalise_cases_idx
- solver_tools/__init__.py
- _Handler
- Locate
- EarendilsLight
- pack
- _install_stub_modules
- Viewer
- GmshCase
- test_coda_monitor.py
- test_ragged_bookkeeping_column_is_dropped_with_a_warning
- .fit
- read_stats
- flow_coda.py
- _fake_sbatch
- legolas/generate_docs.py
- pytest
- _registry_entries
- _registry_entries
- make_solver_repo
- to_nodes
- flow_horses3d.py
- flow_numpy.py
- hmesh.py
- _FakeReader
- is_loaded
- BaseResiduals
- test_import_needs_numpy_only

## God Nodes (most connected - your core abstractions)
1. `LEGOLAS` - 93 edges
2. `FRODO` - 70 edges
3. `Horses3DReader` - 68 edges
4. `Mesh` - 62 edges
5. `box_mesh()` - 53 edges
6. `CellBlock` - 41 edges
7. `CODAResiduals` - 36 edges
8. `SAM` - 35 edges
9. `CODASets` - 30 edges
10. `_make_reader()` - 30 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `GANDALF`  [EXTRACTED]
  examples/gci_codasingle.py → FotR/characters/gandalf.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_coda.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_horses3d.py → FotR/characters/frodo.py
- `run()` --calls--> `FRODO`  [EXTRACTED]
  doc/frodo_sam/flows/flow_numpy.py → FotR/characters/frodo.py
- `test_expand_nodelist()` --uses--> `BaseRing`  [INFERRED]
  FotR/tests/test_rings_gandalf.py → FotR/characters/rings/base.py

## Import Cycles
- 2-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/convert.py -> FotR/characters/legolas/__init__.py`
- 3-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/locate.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/__init__.py`
- 3-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/solution.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/__init__.py`
- 3-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/convert.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`
- 4-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/census.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`
- 5-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/locate.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`
- 5-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/solution.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`

## Communities (146 total, 82 thin omitted)

### Community 0 - "test_horses3d_hsol.py"
Cohesion: 0.23
Nodes (14): read_solution(), fortran_to_reader(), test_gradient_count_deduced_when_not_five(), test_gradients_and_sensor(), test_incomplete_file(), test_layout_reused_between_selections(), test_les_viscosities_deduced_from_size(), test_selection_reads_requested_elements_in_order() (+6 more)

### Community 1 - "Horses3DReader"
Cohesion: 0.05
Nodes (46): Horses3DReader, _to_float(), _to_int(), basic_dataset(), _make_basic_dataset(), _make_mesh(), _one_case(), test_case_idx_assigned_per_case_not_per_p() (+38 more)

### Community 2 - "gandalf/generate_docs.py"
Cohesion: 0.11
Nodes (16): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _google_split(), main() (+8 more)

### Community 3 - "test_coda_single_sets.py"
Cohesion: 0.03
Nodes (31): CODASingleSets, CODASingleStats, gci_root(), _grid(), test_case_ref_rereads_when_nothing_is_materialised(), test_compute_var_broadcasts_a_scalar_expression(), test_compute_var_can_use_design_variables(), test_compute_var_evaluates_per_case() (+23 more)

### Community 4 - "test_merge_datasets.py"
Cohesion: 0.08
Nodes (30): fusionar_y_exportar(), FakeFRODO, _FakeResiduals, _FakeSets, _install_stub_modules(), _load_frodo(), _make_df_state(), _make_group() (+22 more)

### Community 5 - "test_coda_restart.py"
Cohesion: 0.10
Nodes (20): _output_stage(), _assert_skipped_untouched(), case(), _FakeDB, _forbid_slurm(), _make_case(), on_flexo(), _snapshot() (+12 more)

### Community 6 - "MonitorServer"
Cohesion: 0.09
Nodes (8): main(), MonitorServer, test_a_taken_port_moves_to_the_next(), test_case_selection_errors(), test_case_without_folder_is_skipped_with_a_warning(), test_cases_by_subset_and_query(), test_idle_server_stops_by_itself(), test_no_case_is_an_error()

### Community 7 - "SAM"
Cohesion: 0.09
Nodes (4): SAM, BaseSets, NUMPYFILESets, lazy_module()

### Community 8 - "ARAGORN"
Cohesion: 0.07
Nodes (3): ARAGORN, preexec(), RunResult

### Community 11 - "CODASets"
Cohesion: 0.08
Nodes (8): CODASets, db(), _FakeDB, test_plot_wall_integrals_all_vars_and_band(), test_plot_wall_integrals_missing_files_warn(), test_plot_wall_integrals_rejects_all_and_empty(), test_plot_wall_integrals_saves_per_variable(), _write_wall()

### Community 12 - "test_numpy_reader.py"
Cohesion: 0.07
Nodes (30): NUMPYReader, _install_stub_modules(), _make_group_dict(), _make_reader(), test_define_subset_basic(), test_define_subset_duplicate_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises() (+22 more)

### Community 14 - "test_coda_subsets.py"
Cohesion: 0.08
Nodes (9): reader(), test_define_subset_duplicate_name_raises(), test_define_subset_empty_name_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises(), test_get_subset_missing_raises(), test_remove_subset_missing_raises(), test_resolve_cases_idx_conflict_raises() (+1 more)

### Community 17 - "test_coda_monitor.py"
Cohesion: 0.11
Nodes (14): CodaCase, find_logs(), make_case(), _read_dat(), _status(), test_average_rate_from_the_date_lines(), test_find_logs_sorts_by_job_id(), test_finished_and_limit() (+6 more)

### Community 18 - "json"
Cohesion: 0.08
Nodes (34): solution_nodes(), launch(), box_mesh(), linear(), test_geometry_at_the_solution_nodes_is_the_solvers(), test_primitive_variables(), test_raw_fields_are_evaluated_at_the_mesh_nodes(), test_sample_evaluates_the_element_polynomial() (+26 more)

### Community 19 - "test_legolas_io.py"
Cohesion: 0.07
Nodes (33): _check_order(), detect_format(), _hdf5_flavour(), read(), tensor_monotonic(), read_control_keys(), _bc(), _cgns_node() (+25 more)

### Community 20 - "Convert"
Cohesion: 0.10
Nodes (6): Convert, _orient_linear(), _sub_hexes(), _vtk_type(), write_vtu(), array()

### Community 22 - "viewer.py"
Cohesion: 0.08
Nodes (17): _corner_range(), face_geometry(), _hex_edges(), origin(), section_faces(), _tris_per_face(), zone_faces(), zone_list() (+9 more)

### Community 24 - "LEGOLAS"
Cohesion: 0.14
Nodes (21): LEGOLAS, hex_face_grid(), box_with_inner(), files(), _levels(), _need(), test_check_bcs_clean_case(), test_check_bcs_finds_each_problem() (+13 more)

### Community 26 - "test_horses2plt.py"
Cohesion: 0.06
Nodes (23): control_flag(), control_text(), count_columns(), default_variables(), estimate_bytes(), is_converted(), mesh_for(), read_header_info() (+15 more)

### Community 27 - "census.py"
Cohesion: 0.21
Nodes (5): read_control_bcs(), read_ini_bcs(), _strip(), test_read_control_bcs(), test_read_ini_bcs()

### Community 28 - "gci_running"
Cohesion: 0.20
Nodes (7): gci_root(), gci_running(), mixed_cell_root(), _quad_row_grid(), _write_case(), _write_monitor(), _write_stage()

### Community 29 - "test_rings_gandalf.py"
Cohesion: 0.06
Nodes (26): GANDALF, CODASingleRing, _cases(), _ring(), _sources(), test_add_param_guards(), test_coda_generate_folders_placeholders(), test_coda_missing_mesh_raises() (+18 more)

### Community 30 - "Weapons"
Cohesion: 0.07
Nodes (3): Weapons, cost(), unit()

### Community 32 - "solver_tools/horses3d.py"
Cohesion: 0.14
Nodes (10): _field(), HorsesLayout, IncompleteFileError, _layout_from_size(), read_mesh(), _read_records(), _record_bytes(), _sample_matches() (+2 more)

### Community 33 - ".extract_inputs"
Cohesion: 0.08
Nodes (25): log(), run(), juntar_db(), read_db_CODA(), read_db(), FRODO, db(), test_save_to_npy_round_trips_through_the_numpy_reader() (+17 more)

### Community 36 - "topology.py"
Cohesion: 0.15
Nodes (8): coincident_labels(), face_area(), face_keys(), face_rows(), group_keys(), match_faces(), _tri_area(), used_nodes()

### Community 37 - "Mesh"
Cohesion: 0.08
Nodes (8): _decode(), read(), Boundary, CellBlock, Mesh, test_compare_renumbered_and_other_order(), test_prism_stack_is_followed(), test_wall_on_tetrahedra_has_height_but_no_stack()

### Community 38 - "Horses3DRing"
Cohesion: 0.10
Nodes (5): Horses3DRing, _quoted(), test_fsp_report_and_load_campaign(), _uniform(), _write_hsol()

### Community 39 - "numpy.py"
Cohesion: 0.06
Nodes (15): _newton_hex(), _angles(), _corner_triples(), _edges(), gauss_lobatto(), _hex_jacobian(), lagrange(), _linear_jacobian() (+7 more)

### Community 42 - "Backpack"
Cohesion: 0.10
Nodes (4): Backpack, _binop(), _resolve_callable(), _to_value()

### Community 44 - "earendils_light.py"
Cohesion: 0.13
Nodes (9): _doc(), _is_opaque(), _load(), _parameters(), _signature(), _where(), test_every_character_exposes_the_shortcut(), test_opaque_detection_ignores_self() (+1 more)

### Community 45 - "_Fake"
Cohesion: 0.11
Nodes (6): db(), _Fake, _Reader, _ReaderWithShared, _Sets, test_live_skips_empty_slots()

### Community 46 - "CODAResiduals"
Cohesion: 0.17
Nodes (9): CODAResiduals, _FakeDB, _make_case(), test_plot_all_final_residuals_1d_legend_has_no_duplicates(), test_plot_all_final_residuals_1d_one_plot_per_stage(), test_plot_all_final_residuals_1d_saves_figure(), test_plot_all_final_residuals_1d_single_axes(), test_plot_all_final_residuals_no_columns_warns() (+1 more)

### Community 48 - "test_rings_horses3d.py"
Cohesion: 0.18
Nodes (29): _generate(), _keys(), _ring(), test_a_real_boundary_left_out_is_an_error(), test_case_folders_and_controls(), test_cases_must_be_defined(), test_controls_lose_comments_but_keep_blocks(), test_existing_folders_are_skipped() (+21 more)

### Community 49 - "numpy.py"
Cohesion: 0.14
Nodes (15): _newton_trilinear(), _arc(), first_node_fraction(), _hex_directional(), test_directional_size_follows_curved_edges(), test_fields_reach_the_viewer(), test_first_node_fraction(), test_report_and_fields() (+7 more)

### Community 52 - "MonitorHandle"
Cohesion: 0.27
Nodes (3): launch(), MonitorHandle, test_launch_runs_the_server_in_its_own_process()

### Community 53 - "legolas/generate_docs.py"
Cohesion: 0.13
Nodes (10): _arg(), _collect_classes(), _decorators(), _descriptor_methods(), _doc_info(), _google_split(), _methods(), _signature() (+2 more)

### Community 56 - "gimli.py"
Cohesion: 0.15
Nodes (4): ClusterDataset, ClusterMetrics, register_model(), _resolve_scaler()

### Community 57 - "cgns.py"
Cohesion: 0.06
Nodes (25): _add_boundaries(), _children(), _face_corner_check(), _label(), _merge_block(), _name(), read(), _section() (+17 more)

### Community 59 - "test_save_to_npy.py"
Cohesion: 0.09
Nodes (25): _FakeDB, _make_group(), sets_obj(), test_normalise_cases_idx_bad_string_raises(), test_normalise_cases_idx_bad_type_raises(), test_normalise_cases_idx_out_of_range_raises(), test_save_to_npy_aux_with_case_axis_matches_case_idx(), test_save_to_npy_aux_without_case_axis_kept_unfiltered() (+17 more)

### Community 62 - "os"
Cohesion: 0.04
Nodes (8): main(), symlog(), BaseRing, CODARing, _replace_mesh_placeholder(), _resolve_update_value(), __getattr__(), __getattr__()

### Community 66 - "DictVisualizer"
Cohesion: 0.17
Nodes (3): DictVisualizer, to_graph(), build()

### Community 67 - "test_legolas_locate.py"
Cohesion: 0.21
Nodes (8): read_actuator_points(), bulged_pair(), mapping(), test_horses3d_ad_loses_a_point_between_chord_and_arc(), test_points_in_a_curved_box(), test_points_in_tetrahedra(), test_read_actuator_points(), test_turbine_actuator_stations()

### Community 73 - "pattern_pocket_ant"
Cohesion: 0.11
Nodes (3): pattern_pocket, pattern_pocket_ant, FilenamePattern

### Community 74 - "coda_monitor.py"
Cohesion: 0.11
Nodes (11): build(), build_cells(), code(), execute_notebook(), export_html(), main(), md(), pick() (+3 more)

### Community 77 - "server"
Cohesion: 0.25
Nodes (3): _FakeDB, _FakeReader, residuals()

### Community 78 - "frodo_sam/generate_docs.py"
Cohesion: 0.13
Nodes (14): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _formats(), main() (+6 more)

### Community 84 - "element_coords"
Cohesion: 0.21
Nodes (8): find_solutions(), element_coords(), gl_nodes(), linear_q(), test_find_solutions(), test_header_name_padded_with_spaces(), test_surface_files_not_supported(), uniform_case()

### Community 85 - "test_earendils_light.py"
Cohesion: 0.11
Nodes (9): _collect(), _name_index(), test_from_fotr_import_gives_the_class_whatever_was_imported_first(), test_importing_costs_no_third_party_module(), test_index_finds_names_across_the_package(), test_index_records_the_qualified_name(), test_index_skips_the_test_suite(), test_live_falls_back_to_the_index_when_the_chain_lacks_it() (+1 more)

### Community 86 - "cell_info"
Cohesion: 0.15
Nodes (5): cell_info(), _row_boundary(), all_faces(), exterior_faces(), face_index()

### Community 87 - "rings/horses3d.py"
Cohesion: 0.10
Nodes (10): _block_key(), _control_keys(), _defined_zones(), _fmt(), _norm_key(), _read_mesh_info(), _referenced_zones(), _rewrite_control() (+2 more)

### Community 89 - "residuals/coda.py"
Cohesion: 0.08
Nodes (15): _current_slurm_cluster(), _patch_line(), _restart_script_text(), _sbatch(), _script_from_sh(), _set_sbatch_option(), _slurm_active_workdirs(), _slurm_node_cpus() (+7 more)

### Community 91 - "read_header"
Cohesion: 0.20
Nodes (4): HorsesHeader, read_header(), test_header_fields(), test_not_a_horses_file()

### Community 92 - "._normalise_cases_idx"
Cohesion: 0.18
Nodes (4): test_base_normalise_cases_idx_all(), test_base_normalise_cases_idx_invalid_string_raises(), test_base_normalise_cases_idx_out_of_range_raises(), _StubReader

### Community 97 - "EarendilsLight"
Cohesion: 0.14
Nodes (6): EarendilsLight, test_help_returns_the_text_it_prints(), test_overview_when_asked_for_nothing(), test_several_matches_are_all_listed(), test_static_lookup_renders_signature_and_docstring(), test_unknown_name_says_so_without_raising()

### Community 104 - "test_coda_monitor.py"
Cohesion: 0.15
Nodes (23): _criteria(), _get(), log_text(), read_log(), server(), stage_text(), test_a_file_that_shrinks_is_read_again(), test_case_rows_are_sent_once_and_reset_on_a_new_log() (+15 more)

### Community 105 - "test_ragged_bookkeeping_column_is_dropped_with_a_warning"
Cohesion: 0.27
Nodes (5): _rewrite_metadata(), test_ragged_bookkeeping_column_is_dropped_with_a_warning(), test_ragged_identity_column_is_rejected(), test_stale_folder_column_is_recovered_from_folder_fmt(), mutate()

### Community 108 - "read_stats"
Cohesion: 0.32
Nodes (6): read_stats(), _selection(), _stats_arrays(), test_read_stats_layout_and_values(), test_read_stats_refuses_other_files(), write_stats_file()

### Community 111 - "legolas/generate_docs.py"
Cohesion: 0.17
Nodes (11): _class_cards(), _dedent(), _formats(), _function_card(), _imports(), main(), _module_of(), _nested_aliases() (+3 more)

### Community 112 - "pytest"
Cohesion: 0.10
Nodes (29): test_primitive_variables_free_stream(), test_read_control_bcs_unclosed(), curved_box(), test_ansa_g4_msh_becomes_the_cgns_mesh(), test_fix_order_drops_cached_results(), test_fix_order_undoes_a_permutation(), test_gmsh_quad_table_matches_the_reference_cube(), test_linear_kinds_are_oriented_for_vtk() (+21 more)

### Community 113 - "_registry_entries"
Cohesion: 0.67
Nodes (3): _registry_entries(), entries_from_dict(), value_of()

### Community 114 - "_registry_entries"
Cohesion: 0.67
Nodes (3): _registry_entries(), entries_from_dict(), value_of()

### Community 116 - "make_solver_repo"
Cohesion: 0.33
Nodes (4): make_mesh(), make_solver_repo(), repo(), src()

### Community 119 - "to_nodes"
Cohesion: 0.17
Nodes (6): primitive_variables(), to_nodes(), test_mesh_and_solution_agree_node_by_node(), test_primitive_variables_incompressible(), test_real_fan_selection_only(), test_real_sphere()

## Knowledge Gaps
- **82 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SAM` connect `SAM` to `.extract_inputs`, `DictVisualizer`, `DifferentialOperators`, `EarendilsLight`, `.create_final_tensor`, `test_coda_single_sets.py`, `Backpack`, `CODASets`, `CODAResiduals`, `CODASingleReader`, `BaseReader`, `Weapons`, `CODAReader`, `residuals/coda.py`, `HDF5reader`, `os`, `PYLOMSets`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `LEGOLAS` (e.g. with `Mesh` and `Solution`) actually correct?**
  _`LEGOLAS` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `Horses3DReader` be split into smaller, more focused modules?**
  _Cohesion score 0.05240549828178694 - nodes in this community are weakly interconnected._
- **Why does `Horses3DReader` connect `Horses3DReader` to `test_rings_horses3d.py`, `test_horses2plt.py`, `BaseReader`, `SAM`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FRODO` (e.g. with `SAM` and `EarendilsLight`) actually correct?**
  _`FRODO` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `gandalf/generate_docs.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10887096774193548 - nodes in this community are weakly interconnected._
- **Why does `Backpack` connect `Backpack` to `SAM`, `pattern_pocket_ant`, `UnstructuredGrid`, `.GMM`, `HorsesLazyField`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._