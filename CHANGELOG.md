# Changelog

All notable changes to the DOLFINx MCP Server are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.10] - 2026-02-06

### Changed

#### Design-by-Contract Phase 11: Registry Accessor Completion and Refinement Postcondition
- **Registry accessors (2 methods in session.py)**: `get_solution` and `get_form` complete the
  accessor family. `get_solution` verifies name matches key and `space_name` references an
  existing function space (debug-only). `get_form` verifies name matches key (debug-only) and
  accepts optional `suggestion` parameter for context-specific error messages
- **Solver form retrieval (3 call sites in solver.py)**: `solve()` and `solve_time_dependent()`
  replaced 24 lines of manual form existence checks with `session.get_form()` calls.
  `get_solver_diagnostics()` replaced direct dict access with `session.get_solution()`
- **Refine mesh postcondition (mesh.py)**: Unconditional check that refined mesh has more cells
  than original -- uniform refinement must always increase cell count
- **Import additions**: `DOLFINxAPIError` added to `session.py`, `PostconditionError` added to
  `mesh.py`

### Testing
- 10 new contract tests (78 contract tests total, 117 local + 13 Docker = 130 total)
- Phase 11: solution name mismatch (1), solution dangling space (1), solution not found (1),
  solution valid access (1), form name mismatch (1), form not found (1), form valid access (1),
  form custom suggestion (1), refine mesh cell count unit (1), refine mesh integration (1)

---

## [0.1.9] - 2026-02-06

### Changed

#### Design-by-Contract Phase 10: Accessor Postconditions and Finiteness Guards
- **Accessor postconditions (4 methods in session.py)**: `get_mesh`, `get_space`, `get_function`,
  `get_only_space` now verify registry consistency under `if __debug__:` -- name matches key,
  referenced meshes/spaces exist. Zero cost in production with `python -O`
- **Finiteness postconditions (2 tools in postprocess.py)**: `evaluate_solution` and
  `query_point_values` unconditionally check `np.isfinite()` on `uh.eval()` results.
  Non-finite values always indicate a real bug
- **UFL context postconditions (2 functions in ufl_context.py)**: `safe_evaluate` rejects
  None results with `InvalidUFLExpressionError`; `build_namespace` verifies required UFL
  operators (`dx`, `ds`, `inner`, `grad`, `x`) are present under `if __debug__:`
- **Import addition**: `PostconditionError` added to `ufl_context.py` imports

### Testing
- 9 new contract tests (68 contract tests total, 107 local + 13 Docker = 120 total)
- Phase 10: accessor name mismatch (3), dangling mesh/space references (3),
  finiteness NaN postcondition (2), safe_evaluate None rejection (1)

---

## [0.1.8] - 2026-02-06

### Changed

#### Design-by-Contract Phase 9: Defensive Exception Guard Sweep
- **Exception guard sweep (25 blocks across 7 files)**: Every `except Exception` block in
  tool files now has a preceding `except DOLFINxMCPError: raise` guard, ensuring all
  contract errors (`PreconditionError`, `PostconditionError`, `InvariantError`,
  `InvalidUFLExpressionError`) propagate correctly to `@handle_tool_errors` and produce
  structured error responses
- **Plain dict return elimination (2 blocks)**: `assemble()` form evaluation and assembly
  failure paths now raise `DOLFINxAPIError` instead of returning plain `{"error": "..."}` dicts.
  All error responses now include `error_code`, `message`, and `suggestion` fields
- **Narrow guard broadening (4 blocks in mesh.py)**: Changed `except DOLFINxAPIError: raise`
  to `except DOLFINxMCPError: raise` in `create_mesh`, `mark_boundaries`, `create_submesh`,
  `manage_mesh_tags` -- prevents `PreconditionError`/`PostconditionError`/`InvariantError`
  from being silently swallowed
- **Import additions (5 files)**: Added `DOLFINxMCPError` to import lines in `mesh.py`,
  `solver.py`, `problem.py`, `interpolation.py`, `spaces.py`

### Excluded
- `run_custom_code` L128 exception handler intentionally left unguarded (user code capture)

### Testing
- 3 new contract tests (59 contract tests total, 98 local + 13 Docker = 111 total)
- Phase 9: InvalidUFLExpressionError propagation through assemble (1), assembly structured
  error (1), PreconditionError propagation through create_unit_square (1)

---

## [0.1.7] - 2026-02-06

### Changed

#### Design-by-Contract Phase 8: Assert Hardening and Error Integrity
- **Assert-to-InvariantError conversion (32)**: All `assert` statements in 8 session dataclasses
  (`MeshInfo`, `FunctionSpaceInfo`, `FunctionInfo`, `BCInfo`, `FormInfo`, `SolutionInfo`,
  `MeshTagsInfo`, `EntityMapInfo`) converted to `raise InvariantError(...)`. Invariants now
  enforced unconditionally -- `python -O` cannot strip them
- **Error swallowing fix (2 bugs)**: `compute_functionals` no longer wraps `PostconditionError`
  in `DOLFINxAPIError`; `assemble` no longer converts `DOLFINxAPIError` to plain dict. Both
  fixed via `except DOLFINxMCPError: raise` before generic `except Exception`
- **Missing invariant check**: Added `if __debug__: session.check_invariants()` to
  `run_custom_code` (total debug invariant checks: 20)

### Testing
- 3 new contract tests (56 contract tests total, 95 local + 13 Docker = 108 total)
- Phase 8: PostconditionError preservation (1), DOLFINxAPIError preservation (1),
  run_custom_code invariant check (1)
- 10 existing tests updated: `InvariantError` -> `InvariantError`
- Correspondence table: 14 entries updated `InvariantError` -> `InvariantError`

### Infrastructure
- Dockerfile: added `pytest-asyncio` to pip install (eliminates ad-hoc install during Docker tests)

---

## [0.1.6] - 2026-02-06

### Added

#### Design-by-Contract Phase 7: Docker Integration Testing
- **Runtime postcondition tests (7)**: Positive-path verification of solve finiteness,
  L2 norms, error computation, interpolation, scalar assembly, functionals
- **Negative-path contract tests (2)**: NaN injection verifies interpolation and
  compute_error postconditions fire correctly
- **Session operation tests (3)**: cleanup(), remove_mesh() cascade, reset_session()
  with real DOLFINx objects

### Fixed
- **DOLFINx stable API compatibility**: Updated for current `dolfinx/dolfinx:stable` image
  - `ufl.atan_2` -> `getattr(ufl, "atan_2", ufl.atan2)` (symbol rename in newer UFL)
  - `dirichletbc(Constant, dofs)` -> `dirichletbc(Constant, dofs, V)` (API signature change)
  - `LinearProblem` now receives UFL forms (`.ufl_form`) and requires non-empty `petsc_options_prefix`

### Testing
- 12 new Docker integration tests (105 total: 92 local + 13 Docker-only)
- All 19 debug invariants implicitly exercised via full workflow test
- Runtime postcondition coverage: 10/10 postconditions verified

---

## [0.1.5] - 2026-02-06

### Changed

#### Design-by-Contract Phase 6: Final Hardening and Table Completeness
- **Eager preconditions (2 tools)**: `create_unit_square` (cell_type string-set before imports), `create_mesh` (name/nx/ny/nz moved before imports)
- **Error type correction**: `create_unit_square` cell_type changed from `DOLFINxAPIError` to `PreconditionError`
- **Dead code removal**: 1 unreachable clause in `create_unit_square`
- **Correspondence table completeness**: 14 new entries (46 -> 60 total)

### Testing
- 4 new contract tests (53 contract tests total, 92 total)
- Phase 6: linear empty (1), exact empty (1), tag_values type (1), cell_type precondition (1)

---

## [0.1.4] - 2026-02-06

### Changed

#### Design-by-Contract Phase 5: Eager Preconditions and Cleanup Completeness
- **Eager preconditions (7 tools)**: Moved input validations before lazy imports: `solve` (solver_type), `solve_time_dependent` (time_scheme), `define_variational_form` (bilinear/linear), `compute_error` (exact/norm_type), `query_point_values` (points/tolerance), `create_custom_mesh` (name/filename), `create_submesh` (name/tag_values)
- **Error type correction (2 tools)**: `solve` solver_type and `solve_time_dependent` time_scheme changed from `DOLFINxAPIError` to `PreconditionError`
- **Missing postconditions (2 tools)**: `get_solver_diagnostics` L2 norm >= 0; `compute_error` NaN/Inf via `math.isfinite()`
- **Cleanup completeness**: `cleanup()` now checks `solver_diagnostics` and `log_buffer` (12/12 fields)
- **Dead code removal**: 3 unreachable clauses removed in `solve`, `solve_time_dependent`, `compute_error`

### Testing
- 9 new contract tests (49 contract tests total, 88 total)
- Phase 5: solver_type/time_scheme/bilinear/norm_type/tolerance/filename/tag_values preconditions (7), mark_boundaries condition gap fill (1), cleanup completeness (1)

---

## [0.1.3] - 2026-02-06

### Changed

#### Design-by-Contract Phase 4: Remaining Gap Remediation
- **Last assert-as-postcondition**: Converted `solver.py` L2 norm assertion to `PostconditionError` (enforced under `python -O`). Zero `assert` postconditions remain in tool files
- **Eager preconditions (3 tools)**: Moved input validations before lazy `import dolfinx` lines so they fire without DOLFINx installed: `mark_boundaries` (markers/tag/condition), `manage_mesh_tags` (action enum), `create_mixed_space` (subspace count)
- **String enum early validation (2 tools)**: Added `PreconditionError` for `create_mesh` shape enum and `manage_mesh_tags` action enum before imports
- **Error type correction**: `create_mixed_space` subspace count changed from `DOLFINxAPIError` to `PreconditionError` (semantically correct for input validation)
- **Dead code removal**: Removed 2 unreachable `else` clauses in `create_mesh` (shape) and `manage_mesh_tags` (action) superseded by eager preconditions

### Testing
- 5 new contract tests (40 contract tests total, 79 total)
- Phase 4: shape precondition (1), action precondition (1), markers empty/negative-tag preconditions (2), subspace count precondition (1)

---

## [0.1.2] - 2026-02-06

### Changed

#### Design-by-Contract Phase 3: Contract Hardening
- **PostconditionError activated**: Converted 16 session `assert` postconditions to `PostconditionError` raises (enforced even under `python -O`). Covers `cleanup()` (10), `remove_mesh()` (5), `_remove_space_dependents()` (1)
- **`compute_error` postcondition**: Non-negative error norm now raises `PostconditionError` instead of `assert`
- **Missing invariant checks**: Added `if __debug__: session.check_invariants()` to `interpolate` (3 return paths) and `create_discrete_operator` (1 return path). Total: 19 debug invariant checks
- **Eager preconditions**: Moved 5 tool input validations before lazy `import dolfinx` lines so they fire without DOLFINx installed: `create_discrete_operator` (operator_type), `export_solution` (format), `compute_functionals` (expressions non-empty), `plot_solution` (plot_type), `assemble` (target)
- **`compute_functionals` NaN/Inf postcondition**: Each assembled functional value checked with `math.isfinite()`; raises `PostconditionError` on non-finite results
- **Dead code removal**: Removed 4 unreachable `else` clauses superseded by eager preconditions

### Testing
- 7 new contract tests (35 contract tests total, 74 total)
- Phase 3: operator_type/format/target/expressions preconditions (4), BC dangling space invariant (1), PostconditionError integration (1), plot_type precondition (1)

---

## [0.1.1] - 2026-02-06

### Changed

#### Design-by-Contract Gap Remediation
- **FormInfo validator**: Added `__post_init__` to the last unvalidated dataclass (all 8 now covered)
- **Complete cleanup postconditions**: Expanded from 3 to 10 assertions covering all registries
- **`_remove_space_dependents()` postcondition**: Verifies no dangling `space_name` references remain
- **`remove_mesh()` entity_maps postcondition**: Verifies no dangling entity map references
- **Debug invariant checks**: Added `if __debug__: session.check_invariants()` to 12 additional state-mutating tools (15 total). Covers: `create_mesh`, `mark_boundaries`, `refine_mesh`, `create_custom_mesh`, `create_submesh`, `manage_mesh_tags`, `create_mixed_space`, `define_variational_form`, `apply_boundary_condition`, `set_material_properties`, `solve_time_dependent`, `reset_session`
- **Result postconditions**: NaN/Inf detection on `solve_time_dependent`, `interpolate`, `assemble` (scalar); non-negative assertion on `compute_error`
- **New preconditions**: `cell_type` validation on `create_mesh`, `code` non-empty on `run_custom_code`, `sub_space >= 0` on `apply_boundary_condition`

### Removed
- `deal>=4.24.0` dependency (unused -- all contracts enforced via plain assertions and error raises)

### Testing
- 10 new contract tests (28 contract tests total, 67 total)
- FormInfo rejection (2), cleanup/cascade coverage (2), invariant coverage for solutions/mesh_tags/entity_maps (3), tool preconditions for cell_type/code/sub_space (3)

---

## [0.1.0] - 2026-02-06

### Added

#### MCP Server Core
- FastMCP-based server with lifespan-managed `SessionState`
- Docker container deployment (`Dockerfile` with `--network none`, non-root, `--rm`)
- Structured error hierarchy (`DOLFINxMCPError` base with 9 error types)
- Restricted-namespace UFL expression evaluation with token blocklist

#### Tools (28 total)
- **Mesh** (7): `create_unit_square`, `create_mesh`, `create_custom_mesh`, `mark_boundaries`, `refine_mesh`, `create_submesh`, `manage_mesh_tags`
- **Spaces** (2): `create_function_space`, `create_mixed_space`
- **Problem** (3): `define_variational_form`, `apply_boundary_condition`, `set_material_properties`
- **Solver** (3): `solve`, `solve_time_dependent`, `get_solver_diagnostics`
- **Postprocessing** (5): `compute_error`, `export_solution`, `compute_derived_quantity`, `evaluate_solution`, `query_point_values`
- **Interpolation** (2): `interpolate`, `project`
- **Session** (6): `get_session_status`, `list_objects`, `remove_object`, `reset_session`, `assemble`, `compute_mesh_quality`

#### Prompts (6)
- `poisson_guide`, `linear_elasticity_guide`, `stokes_flow_guide`, `mesh_generation_guide`, `debugging_guide`, `time_dependent_guide`

#### Resources (6)
- `session://status`, `session://mesh/{name}`, `session://space/{name}`, `session://solution/{name}`, `config://element-families`, `config://solver-options`

#### Design-by-Contract Enforcement
- **Dataclass validators** (`__post_init__`): Bounds/type checks on all 7 session dataclasses (MeshInfo, FunctionSpaceInfo, FunctionInfo, BCInfo, SolutionInfo, MeshTagsInfo, EntityMapInfo)
- **Session invariants** (`SessionState.check_invariants()`): 7 referential integrity checks (active_mesh validity, cross-registry reference consistency)
- **Postconditions on `remove_mesh()`**: No dangling references after cascade deletion
- **Postconditions on `cleanup()`**: All registries empty after reset
- **Tool preconditions**: Input validation on all tool entry points before lazy imports (mesh dimensions > 0, degree bounds 0-10, non-empty names, reserved UFL name rejection, finite coordinate checks, etc.)
- **Tool postconditions**: Solution NaN/Inf detection, L2 norm non-negativity assertion, gmsh `finally`-block cleanup
- **Debug-mode invariant checks**: `if __debug__: session.check_invariants()` after all state-mutating operations (zero cost with `python -O`)
- **Contract error types**: `PreconditionError`, `PostconditionError`, `InvariantError` (all subclasses of `DOLFINxMCPError`, caught by `handle_tool_errors`)

#### Testing (57 tests)
- 39 core tests: session management, error handling, UFL security, Poisson workflow
- 18 contract violation tests: dataclass rejection (8), session invariant detection (5), tool precondition enforcement (5)

### Dependencies
- `mcp[cli]>=1.2.0`
- `pydantic>=2.0`
- Runtime: `dolfinx`, `petsc4py`, `basix`, `ufl` (provided by Docker image)

---

## Contract Correspondence Table

```
CONTRACT                     LOCATION                     ENFORCED BY
---------------------------------------------------------------------------
INV: num_cells > 0           MeshInfo.__post_init__       InvariantError
INV: tdim <= gdim            MeshInfo.__post_init__       InvariantError
INV: degree >= 0             FunctionSpaceInfo.__post     InvariantError
INV: num_dofs > 0            FunctionSpaceInfo.__post     InvariantError
INV: iterations >= 0         SolutionInfo.__post_init__   InvariantError
INV: active_mesh valid       check_invariants()           InvariantError
INV: no dangling space ref   check_invariants()           InvariantError
INV: no dangling func ref    check_invariants()           InvariantError
INV: no dangling BC ref      check_invariants()           InvariantError
PRE: nx > 0, ny > 0         create_unit_square           PreconditionError
PRE: dt > 0, t_end > t0     solve_time_dependent         PreconditionError
PRE: name not reserved       set_material_properties      PreconditionError
PRE: points non-empty        evaluate_solution            PreconditionError
POST: solution is finite     solve                        SolverError
POST: gmsh finalized         create_custom_mesh           finally block
POST: no dangling refs       remove_mesh                  PostconditionError
POST: cleanup complete       cleanup                      PostconditionError
POST: space deps removed     _remove_space_dependents     PostconditionError
POST: error_val >= 0         compute_error                PostconditionError
POST: functional finite      compute_functionals          PostconditionError
INV: check after interpolate interpolate (3 paths)        if __debug__
INV: check after disc. op.   create_discrete_operator     if __debug__
PRE: operator_type valid     create_discrete_operator     PreconditionError
PRE: format valid            export_solution              PreconditionError
PRE: target valid            assemble                     PreconditionError
PRE: expressions non-empty   compute_functionals          PreconditionError
PRE: plot_type valid         plot_solution                PreconditionError
POST: l2_norm >= 0           solve                        PostconditionError
PRE: shape valid             create_mesh                  PreconditionError
PRE: action valid            manage_mesh_tags             PreconditionError
PRE: markers non-empty       mark_boundaries              PreconditionError
PRE: tag >= 0                mark_boundaries              PreconditionError
PRE: condition non-empty     mark_boundaries              PreconditionError
PRE: subspaces >= 2          create_mixed_space           PreconditionError
PRE: solver_type valid       solve                        PreconditionError
PRE: time_scheme valid       solve_time_dependent         PreconditionError
PRE: bilinear non-empty      define_variational_form      PreconditionError
PRE: linear non-empty        define_variational_form      PreconditionError
PRE: exact non-empty         compute_error                PreconditionError
PRE: norm_type valid         compute_error                PreconditionError
PRE: tolerance > 0           query_point_values           PreconditionError
PRE: filename non-empty      create_custom_mesh           PreconditionError
PRE: tag_values non-empty    create_submesh               PreconditionError
PRE: tag_values all int      create_submesh               PreconditionError
POST: l2_norm >= 0           get_solver_diagnostics       PostconditionError
POST: error_val finite       compute_error                PostconditionError
INV: 1 <= gdim <= 3          MeshInfo.__post_init__       InvariantError
INV: dofs_constrained > 0    BCInfo.__post_init__         InvariantError
INV: parent/child non-empty  EntityMapInfo.__post_init__  InvariantError
INV: name non-empty (Form)   FormInfo.__post_init__       InvariantError
INV: form is not None        FormInfo.__post_init__       InvariantError
INV: no dangling solution    check_invariants()           InvariantError
INV: no dangling mesh_tags   check_invariants()           InvariantError
INV: no dangling entity_map  check_invariants()           InvariantError
PRE: cell_type valid         create_unit_square           PreconditionError
PRE: cell_type valid         create_mesh                  PreconditionError
PRE: code non-empty          run_custom_code              PreconditionError
PRE: sub_space >= 0          apply_boundary_condition     PreconditionError
PRE: degree 0-10             create_function_space        PreconditionError
POST: entity_maps cleaned    remove_mesh                  PostconditionError
POST: name == registry key   get_mesh                     PostconditionError (debug)
POST: name == registry key   get_space                    PostconditionError (debug)
POST: mesh_name in meshes    get_space                    PostconditionError (debug)
POST: name == registry key   get_function                 PostconditionError (debug)
POST: space in fn_spaces     get_function                 PostconditionError (debug)
POST: mesh_name in meshes    get_only_space               PostconditionError (debug)
POST: value is finite        evaluate_solution            PostconditionError
POST: value is finite        query_point_values           PostconditionError
POST: result is not None     safe_evaluate                InvalidUFLExpressionError
POST: required keys present  build_namespace              PostconditionError (debug)
POST: name == registry key   get_solution                 PostconditionError (debug)
POST: space in fn_spaces     get_solution                 PostconditionError (debug)
POST: name == registry key   get_form                     PostconditionError (debug)
POST: refined cells > orig   refine_mesh                  PostconditionError
USE:  get_form("bilinear")   solve                        Replaces manual check
USE:  get_form("linear")     solve                        Replaces manual check
USE:  get_form(bi/lin)       solve_time_dependent         Replaces manual check
USE:  get_solution(name)     get_solver_diagnostics       Replaces direct access
```
