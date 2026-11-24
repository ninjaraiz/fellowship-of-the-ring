# Graph Report - fellowship-of-the-ring  (2026-11-24)

## Corpus Check
- 115 files · ~225,767 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 57 file(s) not represented in the graph (top: (none) 28, .geo 12, .msh 8)
- WARNING: 57 of 186 files seen (31%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 2837 nodes · 6125 edges · 153 communities (62 shown, 91 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 266 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `00a1748c`
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
- CODASingleSets
- test_legolas_viewer.py
- NUMPYFILEReader
- LEGOLAS
- GIMLI
- test_horses2plt.py
- census.py
- gci_running
- test_rings_gandalf.py
- Weapons
- solver_tools/horses3d.py
- .extract_inputs
- XFoilCase
- ViewerServer
- topology.py
- CellBlock
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
- CODASingleStats
- ndarray
- CODASingleReader
- MonitorHandle
- legolas/generate_docs.py
- _FakeDB
- ClusterPlotter
- .prepare
- cgns.py
- Solution
- test_save_to_npy.py
- HDF5reader
- BaseClusterModel
- os
- types
- BaseReader
- DifferentialOperators
- DictVisualizer
- viewer.py
- .create_final_tensor
- .do_GET
- _select_model_worker
- BaseStats
- pattern_pocket_ant
- coda_monitor.py
- test_horses2plt.py
- server
- frodo_sam/generate_docs.py
- _Shortcut
- HorsesLazyField
- _LazyModule
- ._prepare_restart
- test_earendils_light.py
- cell_info
- rings/horses3d.py
- CODAReader
- residuals/coda.py
- .surface_derivative
- read_header
- Quality
- solver_tools/__init__.py
- _Handler
- Locate
- EarendilsLight
- pack
- _install_stub_modules
- ViewerHandle
- GmshCase
- test_coda_monitor.py
- gci_root
- _fake_sbatch
- legolas/generate_docs.py
- pytest
- _registry_entries
- _registry_entries
- make_solver_repo
- main
- to_nodes
- earendils_light.py
- test_richardson_recovers_a_known_order
- merge_points
- test_sets_imports_without_pylom
- ._save_figures
- _collect
- test_import_needs_numpy_only
- register_model

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
- `test_expand_nodelist()` --uses--> `BaseRing`  [INFERRED]
  FotR/tests/test_rings_gandalf.py → FotR/characters/rings/base.py
- `main()` --calls--> `GANDALF`  [EXTRACTED]
  examples/gci_codasingle.py → FotR/characters/gandalf.py
- `LEGOLAS` --uses--> `Solution`  [INFERRED]
  FotR/characters/legolas/__init__.py → FotR/characters/legolas/solution.py
- `ViewerServer` --uses--> `Solution`  [INFERRED]
  FotR/characters/legolas/viewer.py → FotR/characters/legolas/solution.py
- `_ring()` --calls--> `CODARing`  [INFERRED]
  FotR/tests/test_rings_gandalf.py → FotR/characters/rings/coda.py

## Import Cycles
- 2-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/convert.py -> FotR/characters/legolas/__init__.py`
- 3-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/solution.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/__init__.py`
- 3-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/locate.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/__init__.py`
- 3-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/convert.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`
- 4-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/census.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`
- 5-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/solution.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`
- 5-file cycle: `FotR/characters/legolas/__init__.py -> FotR/characters/legolas/locate.py -> FotR/characters/legolas/quality.py -> FotR/characters/legolas/topology.py -> FotR/characters/legolas/mesh.py -> FotR/characters/legolas/__init__.py`

## Communities (153 total, 91 thin omitted)

### Community 0 - "test_horses3d_hsol.py"
Cohesion: 0.13
Nodes (27): read_solution(), element_coords(), fortran_to_reader(), gl_nodes(), linear_q(), _stats_arrays(), test_find_solutions(), test_gradient_count_deduced_when_not_five() (+19 more)

### Community 1 - "Horses3DReader"
Cohesion: 0.05
Nodes (46): Horses3DReader, _to_float(), _to_int(), basic_dataset(), _make_basic_dataset(), _make_mesh(), _one_case(), test_case_idx_assigned_per_case_not_per_p() (+38 more)

### Community 2 - "gandalf/generate_docs.py"
Cohesion: 0.11
Nodes (16): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _google_split(), main() (+8 more)

### Community 3 - "test_coda_single_sets.py"
Cohesion: 0.05
Nodes (23): test_case_ref_rereads_when_nothing_is_materialised(), test_compute_var_broadcasts_a_scalar_expression(), test_compute_var_can_use_design_variables(), test_compute_var_evaluates_per_case(), test_compute_var_refuses_to_overwrite_silently(), test_compute_var_rejects_arbitrary_code(), test_create_jset_concatenates_blocks_of_different_height(), test_create_jset_drops_variables_missing_in_some_case() (+15 more)

### Community 4 - "test_merge_datasets.py"
Cohesion: 0.08
Nodes (30): fusionar_y_exportar(), FakeFRODO, _FakeResiduals, _FakeSets, _install_stub_modules(), _load_frodo(), _make_df_state(), _make_group() (+22 more)

### Community 5 - "test_coda_restart.py"
Cohesion: 0.10
Nodes (18): _assert_skipped_untouched(), case(), _FakeDB, _FakeReader, _forbid_slurm(), _make_case(), on_flexo(), _snapshot() (+10 more)

### Community 6 - "MonitorServer"
Cohesion: 0.09
Nodes (8): main(), MonitorServer, test_a_taken_port_moves_to_the_next(), test_case_selection_errors(), test_case_without_folder_is_skipped_with_a_warning(), test_cases_by_subset_and_query(), test_idle_server_stops_by_itself(), test_no_case_is_an_error()

### Community 7 - "SAM"
Cohesion: 0.06
Nodes (7): BaseResiduals, SAM, BaseSets, NUMPYFILESets, PYLOMSets, is_loaded(), lazy_module()

### Community 8 - "ARAGORN"
Cohesion: 0.07
Nodes (3): ARAGORN, preexec(), RunResult

### Community 11 - "CODASets"
Cohesion: 0.17
Nodes (8): db(), _FakeDB, _StubReader, test_plot_wall_integrals_all_vars_and_band(), test_plot_wall_integrals_missing_files_warn(), test_plot_wall_integrals_rejects_all_and_empty(), test_plot_wall_integrals_saves_per_variable(), _write_wall()

### Community 12 - "test_numpy_reader.py"
Cohesion: 0.07
Nodes (30): NUMPYReader, _install_stub_modules(), _make_group_dict(), _make_reader(), test_define_subset_basic(), test_define_subset_duplicate_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises() (+22 more)

### Community 14 - "test_coda_subsets.py"
Cohesion: 0.06
Nodes (12): reader(), test_base_normalise_cases_idx_all(), test_base_normalise_cases_idx_invalid_string_raises(), test_base_normalise_cases_idx_out_of_range_raises(), test_define_subset_duplicate_name_raises(), test_define_subset_empty_name_raises(), test_define_subset_empty_selection_raises(), test_define_subset_out_of_range_raises() (+4 more)

### Community 17 - "test_coda_monitor.py"
Cohesion: 0.11
Nodes (14): CodaCase, find_logs(), make_case(), _read_dat(), _status(), test_average_rate_from_the_date_lines(), test_find_logs_sorts_by_job_id(), test_finished_and_limit() (+6 more)

### Community 18 - "json"
Cohesion: 0.14
Nodes (20): solution_nodes(), box_mesh(), linear(), test_geometry_at_the_solution_nodes_is_the_solvers(), test_primitive_variables(), test_raw_fields_are_evaluated_at_the_mesh_nodes(), test_sample_evaluates_the_element_polynomial(), test_sample_needs_a_solution() (+12 more)

### Community 19 - "test_legolas_io.py"
Cohesion: 0.07
Nodes (34): _check_order(), detect_format(), _hdf5_flavour(), read(), tensor_monotonic(), read_control_keys(), _bc(), _cgns_node() (+26 more)

### Community 20 - "Convert"
Cohesion: 0.10
Nodes (6): Convert, _orient_linear(), _sub_hexes(), _vtk_type(), write_vtu(), array()

### Community 22 - "test_legolas_viewer.py"
Cohesion: 0.09
Nodes (22): _corner_range(), face_geometry(), origin(), section_faces(), _tris_per_face(), zone_faces(), _get(), server() (+14 more)

### Community 24 - "LEGOLAS"
Cohesion: 0.12
Nodes (24): LEGOLAS, Boundary, box_with_inner(), files(), _levels(), _need(), test_check_bcs_clean_case(), test_check_bcs_finds_each_problem() (+16 more)

### Community 26 - "test_horses2plt.py"
Cohesion: 0.10
Nodes (15): control_flag(), control_text(), count_columns(), default_variables(), estimate_bytes(), is_converted(), mesh_for(), read_header_info() (+7 more)

### Community 27 - "census.py"
Cohesion: 0.19
Nodes (6): read_control_bcs(), read_ini_bcs(), _strip(), _vec(), test_read_control_bcs(), test_read_ini_bcs()

### Community 28 - "gci_running"
Cohesion: 0.20
Nodes (7): gci_root(), gci_running(), mixed_cell_root(), _quad_row_grid(), _write_case(), _write_monitor(), _write_stage()

### Community 29 - "test_rings_gandalf.py"
Cohesion: 0.06
Nodes (25): GANDALF, CODASingleRing, _cases(), _ring(), _sources(), test_add_param_guards(), test_coda_generate_folders_placeholders(), test_coda_missing_mesh_raises() (+17 more)

### Community 30 - "Weapons"
Cohesion: 0.07
Nodes (3): Weapons, cost(), unit()

### Community 32 - "solver_tools/horses3d.py"
Cohesion: 0.11
Nodes (12): _field(), HorsesLayout, IncompleteFileError, _layout_from_size(), read_mesh(), _read_records(), read_stats(), _record_bytes() (+4 more)

### Community 33 - ".extract_inputs"
Cohesion: 0.04
Nodes (37): log(), run(), log(), run(), log(), run(), log(), run() (+29 more)

### Community 36 - "topology.py"
Cohesion: 0.15
Nodes (10): all_faces(), coincident_labels(), exterior_faces(), face_area(), face_index(), face_keys(), group_keys(), match_faces() (+2 more)

### Community 37 - "CellBlock"
Cohesion: 0.04
Nodes (21): read(), _decode(), read(), _newton_trilinear(), read_actuator_points(), CellBlock, _corner_ijk(), detect_hex_order() (+13 more)

### Community 38 - "Horses3DRing"
Cohesion: 0.09
Nodes (4): Horses3DRing, test_fsp_report_and_load_campaign(), _uniform(), _write_hsol()

### Community 39 - "numpy.py"
Cohesion: 0.06
Nodes (29): _newton_hex(), _angles(), _corner_triples(), _edges(), gauss_lobatto(), _hex_jacobian(), lagrange(), _linear_jacobian() (+21 more)

### Community 42 - "Backpack"
Cohesion: 0.10
Nodes (4): Backpack, _binop(), _resolve_callable(), _to_value()

### Community 44 - "earendils_light.py"
Cohesion: 0.13
Nodes (8): _doc(), _is_opaque(), _parameters(), _signature(), _where(), test_every_character_exposes_the_shortcut(), test_opaque_detection_ignores_self(), test_signature_drops_self()

### Community 45 - "_Fake"
Cohesion: 0.11
Nodes (6): db(), _Fake, _Reader, _ReaderWithShared, _Sets, test_live_skips_empty_slots()

### Community 46 - "CODAResiduals"
Cohesion: 0.11
Nodes (9): CODAResiduals, _FakeDB, _make_case(), test_plot_all_final_residuals_1d_legend_has_no_duplicates(), test_plot_all_final_residuals_1d_one_plot_per_stage(), test_plot_all_final_residuals_1d_saves_figure(), test_plot_all_final_residuals_1d_single_axes(), test_plot_all_final_residuals_no_columns_warns() (+1 more)

### Community 48 - "test_rings_horses3d.py"
Cohesion: 0.17
Nodes (28): _generate(), _keys(), _ring(), test_a_real_boundary_left_out_is_an_error(), test_case_folders_and_controls(), test_cases_must_be_defined(), test_controls_lose_comments_but_keep_blocks(), test_existing_folders_are_skipped() (+20 more)

### Community 52 - "MonitorHandle"
Cohesion: 0.27
Nodes (3): launch(), MonitorHandle, test_launch_runs_the_server_in_its_own_process()

### Community 53 - "legolas/generate_docs.py"
Cohesion: 0.13
Nodes (10): _arg(), _collect_classes(), _decorators(), _descriptor_methods(), _doc_info(), _google_split(), _methods(), _signature() (+2 more)

### Community 54 - "_FakeDB"
Cohesion: 0.26
Nodes (12): CODASets, _FakeDB, _make_group(), sets_obj(), test_save_to_npy_aux_with_case_axis_matches_case_idx(), test_save_to_npy_aux_without_case_axis_kept_unfiltered(), test_save_to_npy_case_order_consistent_across_structures(), test_save_to_npy_exports_multiple_aux_vars() (+4 more)

### Community 57 - "cgns.py"
Cohesion: 0.08
Nodes (18): _add_boundaries(), _children(), _face_corner_check(), _label(), _merge_block(), _name(), read(), _section() (+10 more)

### Community 59 - "test_save_to_npy.py"
Cohesion: 0.10
Nodes (14): test_normalise_cases_idx_bad_string_raises(), test_normalise_cases_idx_bad_type_raises(), test_normalise_cases_idx_out_of_range_raises(), test_save_to_npy_case_idx_all(), test_save_to_npy_case_idx_list(), test_save_to_npy_case_idx_single_int(), test_save_to_npy_exports_aux_when_present(), test_save_to_npy_exports_vars() (+6 more)

### Community 62 - "os"
Cohesion: 0.04
Nodes (8): main(), symlog(), BaseRing, CODARing, _replace_mesh_placeholder(), _resolve_update_value(), __getattr__(), __getattr__()

### Community 66 - "DictVisualizer"
Cohesion: 0.17
Nodes (3): DictVisualizer, to_graph(), build()

### Community 67 - "viewer.py"
Cohesion: 0.11
Nodes (7): launch(), _load_fields(), _normalise_fields(), _normalise_node_fields(), _save_fields(), Viewer, test_launch_reports_a_failing_server()

### Community 73 - "pattern_pocket_ant"
Cohesion: 0.11
Nodes (3): pattern_pocket, pattern_pocket_ant, FilenamePattern

### Community 74 - "coda_monitor.py"
Cohesion: 0.11
Nodes (11): build(), build_cells(), code(), execute_notebook(), export_html(), main(), md(), pick() (+3 more)

### Community 75 - "test_horses2plt.py"
Cohesion: 0.10
Nodes (9): Horses3DSets, case(), fake_binary(), _q(), _sets(), test_dry_run_plans_without_writing(), test_errors(), test_local_run_converts_into_post() (+1 more)

### Community 77 - "server"
Cohesion: 0.25
Nodes (3): _FakeDB, _FakeReader, residuals()

### Community 78 - "frodo_sam/generate_docs.py"
Cohesion: 0.13
Nodes (14): _arg(), _collect_classes(), _decorators(), _dedent(), _descriptor_methods(), _doc_info(), _formats(), main() (+6 more)

### Community 84 - "._prepare_restart"
Cohesion: 0.11
Nodes (11): _output_stage(), _patch_line(), _restart_script_text(), _script_from_sh(), _set_sbatch_option(), test_apply_restarts_the_case(), test_output_stage_matches_reader_rule(), test_patch_line_ignores_indented_lines() (+3 more)

### Community 85 - "test_earendils_light.py"
Cohesion: 0.12
Nodes (8): _name_index(), test_from_fotr_import_gives_the_class_whatever_was_imported_first(), test_importing_costs_no_third_party_module(), test_index_finds_names_across_the_package(), test_index_records_the_qualified_name(), test_index_skips_the_test_suite(), test_live_falls_back_to_the_index_when_the_chain_lacks_it(), test_live_follows_the_declared_order()

### Community 86 - "cell_info"
Cohesion: 0.16
Nodes (4): cell_info(), _row_boundary(), zone_list(), face_rows()

### Community 87 - "rings/horses3d.py"
Cohesion: 0.11
Nodes (10): _control_keys(), _defined_zones(), _fmt(), _norm_key(), _quoted(), _read_mesh_info(), _referenced_zones(), _rewrite_control() (+2 more)

### Community 89 - "residuals/coda.py"
Cohesion: 0.17
Nodes (5): _current_slurm_cluster(), _sbatch(), _slurm_active_workdirs(), _slurm_node_cpus(), _slurm_cluster()

### Community 91 - "read_header"
Cohesion: 0.17
Nodes (5): find_solutions(), HorsesHeader, read_header(), test_header_fields(), test_not_a_horses_file()

### Community 97 - "EarendilsLight"
Cohesion: 0.14
Nodes (6): EarendilsLight, test_help_returns_the_text_it_prints(), test_overview_when_asked_for_nothing(), test_several_matches_are_all_listed(), test_static_lookup_renders_signature_and_docstring(), test_unknown_name_says_so_without_raising()

### Community 104 - "test_coda_monitor.py"
Cohesion: 0.15
Nodes (23): _criteria(), _get(), log_text(), read_log(), server(), stage_text(), test_a_file_that_shrinks_is_read_again(), test_case_rows_are_sent_once_and_reset_on_a_new_log() (+15 more)

### Community 107 - "gci_root"
Cohesion: 0.33
Nodes (3): gci_root(), _grid(), _write_monitor()

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
Cohesion: 0.25
Nodes (4): primitive_variables(), to_nodes(), test_primitive_variables_incompressible(), test_real_sphere()

## Knowledge Gaps
- **91 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SAM` connect `SAM` to `.extract_inputs`, `DictVisualizer`, `DifferentialOperators`, `EarendilsLight`, `.create_final_tensor`, `Backpack`, `CODAResiduals`, `CODASingleReader`, `Weapons`, `CODASingleSets`, `_FakeDB`, `NUMPYFILEReader`, `CODAReader`, `residuals/coda.py`, `HDF5reader`, `os`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `LEGOLAS` (e.g. with `Mesh` and `Solution`) actually correct?**
  _`LEGOLAS` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_horses3d_hsol.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13213213213213212 - nodes in this community are weakly interconnected._
- **Why does `Backpack` connect `Backpack` to `SAM`, `pattern_pocket_ant`, `UnstructuredGrid`, `.GMM`, `HorsesLazyField`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FRODO` (e.g. with `SAM` and `EarendilsLight`) actually correct?**
  _`FRODO` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Horses3DReader` be split into smaller, more focused modules?**
  _Cohesion score 0.05240549828178694 - nodes in this community are weakly interconnected._
- **Why does `CODAStats` connect `CODAStats` to `BaseStats`, `SAM`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._