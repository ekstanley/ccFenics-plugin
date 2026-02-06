# Changelog

All notable changes to the DOLFINx MCP Server are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
INV: num_cells > 0           MeshInfo.__post_init__       AssertionError
INV: tdim <= gdim            MeshInfo.__post_init__       AssertionError
INV: degree >= 0             FunctionSpaceInfo.__post     AssertionError
INV: num_dofs > 0            FunctionSpaceInfo.__post     AssertionError
INV: iterations >= 0         SolutionInfo.__post_init__   AssertionError
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
```
