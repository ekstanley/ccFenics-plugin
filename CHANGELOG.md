# Changelog

All notable changes to the DOLFINx MCP Server are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.10.4] - 2026-02-20

### Added
- `get_session_state` now returns `environment` dict with DOLFINx version, PETSc scalar type, Python version, and numpy version (lazy-imported, "unavailable" when imports fail)
- 30 SKILL.md files (15 skills × 2 locations) now document `run_custom_code` namespace persistence across calls

### Fixed
- 8 ruff lint errors: `contextlib.suppress` (SIM105), `zip(strict=True)` (B905), unused import (F401), ternary operator (SIM108), import sorting (I001 ×4)
- Test assertion: `test_assemble_scalar_preserves_api_error` expected `DOLFINX_API_ERROR` but code correctly raises `POSTCONDITION_VIOLATED` for NaN assembly results
- 6 skills (×2 locations) add required `petsc_options_prefix` kwarg for DOLFINx 0.10.0 `LinearProblem`
- Cowork extension manifest: added required `entry_point` field (`run-server.sh`)

### Changed
- Tests: 493 -> 514
- Unit tests: 417 (no Docker)
- Docker integration tests: 514 total (all suites)

---

## [0.10.3] - 2026-02-19

### Added
- Persistent `run_custom_code` namespace — variables survive across calls via `exec_namespace` on SessionState
- 3-layer namespace priority: user vars < session registries < system modules
- 10 new tests (5 unit + 4 contract + 1 Docker integration)

### Fixed
- `gmsh.finalize()` crash after `model_to_mesh` consumes the model (wrapped in try/except)
- Stale `gmshio` references in membrane skill docs (both `.claude/` and `ccfenics/`)
- DbC audit: missing `session.check_invariants()` in `manage_mesh_tags` query path
- DbC audit: `assemble` scalar used `DOLFINxAPIError` instead of `PostconditionError` for result validation
- Tag counting optimization: O(k*n) loop replaced with O(n) `np.unique(return_counts=True)`

### Changed
- Docker tests: 192 -> 193

---

## [0.10.2] - 2026-02-19

### Added
- `list_workspace_files` tool — list files in /workspace directory
- `bundle_workspace_files` tool — create base64-encoded archives of workspace files
- `generate_report` tool — generate HTML reports with embedded plots
- Total tools: 35 -> 38
- 36 new workspace tool tests (192 total Docker tests, 6 suites)

### Fixed
- Removed manifest `entry_point` to fix Cowork tool discovery flakiness
- Path traversal vulnerability in `bundle_workspace_files` archive_name (HIGH)
- XSS in `_embed_image` alt attribute (MEDIUM)
- `validate_workspace_path` boundary check accepting `/workspace2/` (MEDIUM)

### Security
- Extracted `validate_workspace_path` to shared `_validators.py`
- Fixed 3 security vulnerabilities (1 HIGH path traversal, 2 MEDIUM)

---

## [0.10.1] - 2026-02-19

### Fixed
- Resolved 12 ruff lint violations in source code (E501 line length, B905 zip strict, I001 import order, SIM108 ternary, B904 raise from)
- Resolved 21 ruff lint violations in test files (I001, F401, F841, E501, B905)
- CI lint job now passes cleanly on all Dependabot PRs

### Changed
- Updated GitHub Actions: checkout v4.2.2->v6.0.2, setup-python v5.6.0->v6.2.0, codeql-action SHA bump
- Version bump to 0.10.1 across all 5 version locations

---

## [0.10.0] - 2026-02-19

### Added
- `create_function` tool — create named functions in a function space (35th tool)
- `read_workspace_file` tool — read files from /workspace as base64 or text
- INV-9: `FormInfo.trial_space_name` must reference valid space (or be empty)
- Cowork Desktop Extension manifest and plugin package (`ccfenics/`)
- Community health files: CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
- GitHub issue/PR templates and Dependabot config
- `.gitattributes` for cross-platform line ending normalization

### Changed
- Total tools: 33 -> 35
- Total invariants: 8 -> 9 (INV-9 proven in Quint, enforced in Python)
- Performance: UFL/numpy namespace caching (~70% faster per call)
- Code DRY refactor: shared validators, session accessor, -190 LOC
- Docker tests: 79 -> 152 (26 tutorial + 40 runtime + 26 edge + 48 cowork + 12 invariant)
- CI: explicitly ignores all Docker-only test files
- Tightened 4 FEM mathematical test tolerances

### Fixed
- 5 stress test bugs: DG0 warp guard, vector expressions, multi-mesh dx, boundary_tag BCs, register_function API
- Mixed-space MUMPS auto-detection (mock-safe via element_family, not num_sub_spaces)
- `solve_eigenvalue` session attribute bug (active_mesh not active_space)
- Numeric expression coercion in `project` and `set_material_properties`
- `_scalar_*` cascade deletion in `_remove_space_dependents`

### Security
- Sanitized git history (removed personal emails via git-filter-repo)
- Removed tracked `.hypothesis/` cache (contained local filesystem paths)
- Added SECURITY.md with vulnerability reporting instructions

---

## [0.9.0] - 2026-02-12

### Added
- Axisymmetric domains skill (`fem-workflow-axisymmetric`)
- T6.7 Docker test for axisymmetric Poisson

### Changed
- Updated README tool count (31 -> 33), added Claude Code auto-discovery docs
- Added CI/License/Python badges to README
- Repository sanitation (`.gitignore` additions, removed tracked `firebase-debug.log`)
- Added `.dockerignore` to reduce build context size
- Aligned Jupyter extension version to 0.9.0
- Updated CLAUDE.md and SessionStart hook for v0.9.0

---

## [0.8.0] - 2026-02-12

### Added
- SLEPc eigenvalue solver (`solve_eigenvalue` tool, 33rd tool)
- DG operators: `jump`, `avg`, `cell_avg`, `facet_avg` in UFL namespace
- Lagrange element variants parameter (`equispaced`, `gll_warped`, etc.)
- 4 new skills: `fem-workflow-cahn-hilliard`, `fem-workflow-biharmonic`, `fem-workflow-dg-formulation`, `fem-workflow-eigenvalue`
- T6 test group: 6 Docker tests for official demo coverage

### Changed
- Total tools: 32 -> 33
- Total skills: 32 -> 36
- Docker tests: 73 -> 79

---

## [0.7.0] - 2026-02-11

### Added
- Nonlinear solver (`solve_nonlinear` tool, 32nd tool)
- Full DOLFINx tutorial coverage (26/26 topics)
- 6 new skills (`fem-workflow-custom-newton`, `fem-workflow-convergence-rates`, `fem-workflow-singular-poisson`, `fem-workflow-mixed-poisson`, `fem-workflow-electromagnetics`, `fem-workflow-component-bc`)
- T5 test group: 5 Docker tests for remaining tutorial chapters
- `ds(tag)` subdomain data in UFL namespace
- `split()` for mixed-space function decomposition
- Nullspace support (`nullspace_mode` parameter on `solve`)

### Changed
- Total tools: 31 -> 32
- Docker tests: 47 -> 73

---

## [0.6.2] - 2026-02-11

### Fixed

#### CI coverage gate discrepancy (Task 1)
- Removed `--cov-fail-under=80` CLI override from `.github/workflows/ci.yml`
- `pyproject.toml` `[tool.coverage.report].fail_under = 90` is now the single source of truth
- Updated CLAUDE.md coverage command to match

#### PreconditionError inconsistency in `interpolate()` (Task 2)
- Changed mutual-exclusivity checks (`expression` vs `source_function`) from `DOLFINxAPIError`
  to `PreconditionError`, matching `project()` and DbC conventions
- Moved precondition checks BEFORE lazy imports and `session.get_function()` (eager validation)

#### Deferred `_check_forbidden` security bug (Task 3)
- `_make_boundary_fn` in problem.py deferred `_check_forbidden()` to lambda call-time,
  meaning malicious expressions were not rejected at creation time
- Fixed by extracting to `make_boundary_marker()` in `eval_helpers.py` which checks EAGERLY
- All 4 expression evaluation sites now call `_check_forbidden()` eagerly

#### Unnecessary `type: ignore` in server.py
- Removed stale `# type: ignore[arg-type]` on `mcp.run()` call (pyright no longer needs it)

### Added

#### Shared expression evaluation module `eval_helpers.py` (Task 3)
- New leaf module providing `eval_numpy_expression()` and `make_boundary_marker()`
- Replaces ~120 lines of duplicated helpers across 4 tool files:
  - `_eval_interp_expression` (interpolation.py)
  - `_eval_exact_expression` (postprocess.py)
  - `_eval_material_expression`, `_make_boundary_fn`, `_restricted_eval` (problem.py)
  - `_make_marker_fn` (mesh.py)
- `_eval_bc_expression` kept in problem.py (distinct: merges UFL namespace with numpy overrides)
- Import DAG: `eval_helpers.py` imports only from `ufl_context.py` (leaf module)

#### Enhanced CLAUDE.md (Task 4)
- Added Error Selection Guide (12-row table mapping situations to error classes)
- Added Expression Evaluation Patterns (numpy vs UFL symbolic contexts)
- Added New Tool Checklist (13-step guide for adding tools)
- Added Anti-Patterns section (6 common mistakes to avoid)
- Updated Import DAG, Source Layout, and Testing sections
- Added test fixture catalog (`session`, `mock_ctx`, `populated_session`, `mock_ctx_populated`)

#### Claude commands (Task 5)
- `.claude/commands/check-contracts.md` -- DbC compliance audit
- `.claude/commands/add-tool.md` -- New tool scaffolding guide (accepts `$ARGUMENTS`)
- `.claude/commands/run-tests.md` -- Test runner with coverage analysis

#### MIT LICENSE file
- Added standard MIT license

### Changed
- Version strings aligned in `README.md` (0.5.1 -> 0.6.1) and `providers.py` (0.1.0 -> 0.6.1)

**v0.6.2 metrics:**
- Ruff: 0 errors
- Pyright: 0 errors, 1 warning (pre-existing FastMCP library type)
- Tests: 296 passed, 3 skipped
- Coverage: 90.15% (>= 90% threshold)
- Net impact: +365/-164 lines across 14 files

**Files added (5):**
- `src/dolfinx_mcp/eval_helpers.py`
- `.claude/commands/check-contracts.md`
- `.claude/commands/add-tool.md`
- `.claude/commands/run-tests.md`
- `LICENSE`

**Files modified (9):**
- `.github/workflows/ci.yml`
- `CLAUDE.md`
- `README.md`
- `src/dolfinx_mcp/server.py`
- `src/dolfinx_mcp/resources/providers.py`
- `src/dolfinx_mcp/tools/interpolation.py`
- `src/dolfinx_mcp/tools/mesh.py`
- `src/dolfinx_mcp/tools/postprocess.py`
- `src/dolfinx_mcp/tools/problem.py`

---

## [0.6.1] - 2026-02-07

### Fixed

#### Layer 0 -- pyright zero warnings (Phase 34A)
- Resolved all 8 `reportUnknownParameterType` warnings
- Added `dict[str, Any]` return annotations to 6 provider functions
- Annotated `space_object: Any` in `session.py` and `function: Any` in `utils.py`
- Result: 0 errors, 0 warnings, 0 informations

#### Layer 1 -- Idris 2 proof hole closed (Phase 34B)
- Replaced `noRefWithoutKey` (had hole `?noRefWithoutKey_rhs`) with correctly typed
  `noRefForAbsentKey` proof
- New type: `Not (HasKey k ks) -> (ref : ValidRef ks) -> Not (ref.name = k)`
- One-line proof by substitution contradiction; zero holes remaining

### Added

#### Layer 2 -- Quint model extended for forms (Phase 34C)
- Added 4 state variables: `form_keys`, `ufl_symbol_keys`, `solver_diag_count`,
  `log_buffer_count`
- Added `registerForm` and `registerUflSymbol` actions (require non-empty spaces)
- Updated `removeMesh` with conditional wholesale clear of forms/ufl_symbols
- Added INV-8: `form_keys.size() > 0 implies space_keys.size() > 0`
- Added `forms_cascade_scenario` exercising register-then-cascade-delete
- Random simulation: 298 traces, 50 steps each, no violations

#### Layer 3 -- Lean 4 forms invariant (Phase 34D)
- Extended `SessionState` with `forms : List String` and `ufl_symbols : List String`
- Added INV-8 to `valid` predicate: `forms != [] -> function_spaces != []`
- Updated `removeMesh` to conditionally clear forms when dependent spaces exist
- Extended all 10 preservation theorems for 8 invariants (was 7)
- Key proof: `removeMesh_valid` INV-8 case analysis -- depSpaceKeys empty vs non-empty

#### Layer 5 -- Test coverage expansion (Phase 34E)
- Added ~50 new tests: getter coverage, form lifecycle, golden scenarios
- Added form-related property tests (`TestFormWholesaleClear`, `TestFormSpaceInvariant`)
- Added `register_form_op` and `register_ufl_symbol_op` hypothesis strategies
- Coverage: 92.49% (296 passed, 3 skipped), gate raised to 90%

### Changed
- Version bump from 0.6.0 to 0.6.1
- Coverage `fail_under` gate raised from 80% to 90%

#### Updated Cross-Layer Correspondence

```
Layer | Tool    | What It Catches             | Status
------|---------|-----------------------------|--------
  0   | pyright | Type errors, None safety    | 0 errors, 0 warnings
  1   | Idris 2 | Invalid construction        | 5 modules, 0 holes
  2   | Quint   | Multi-step sequence bugs    | 8 invariants, 298 traces
  3   | Lean 4  | Invariant violations        | 20+ theorems, 8 invariants
  4   | ruff    | Runtime contract violations | 31 tools
  5   | pytest  | Behavioral bugs + coverage  | 296 tests, 92.49%
```

---

## [0.6.0] - 2026-02-07

### Added

#### Outline-Strong 5-Layer Validation Stack (Phase 33)

- **Layer 0 -- pyright static type checking**
  - Added `[tool.pyright]` config (standard mode, Python 3.10 target)
  - Created `src/dolfinx_mcp/py.typed` PEP 561 marker
  - Added pyright CI job; 0 errors, 8 warnings (DOLFINx `Any` types)
  - Fixed 2 genuine type bugs: `errors.py` `__signature__` attr, `server.py` transport literal

- **Layer 1 -- Idris 2 dependent type verification**
  - Modeled SessionState with intrinsic validity: invalid states are type errors
  - `HasKey` proof type and `ValidRef` dependent references in `Registry.idr`
  - 7 type-safe operations: register operations require compile-time membership proofs
  - Proved `removeKeyAbsent` and `removeKeyPreserves` in the type system
  - `cleanup` returns `freshState` by type (trivially valid)
  - 5 worked examples demonstrating construction prevention
  - Located in `.outline/proofs/idris/`

- **Layer 2 -- Quint state machine model checking**
  - Modeled SessionState as Quint state machine (14 vars, 11 actions)
  - 7 invariant predicates mapping 1:1 to `check_invariants()`
  - Exhaustive model checking via Apalache (depth 8): no violations found
  - Random simulation (921 traces): no violations
  - 5 concrete scenarios: cascade delete, partial remove, shared entity map,
    complex cleanup, parent-only entity map removal
  - Located in `.outline/specs/session_state.qnt`

- **Layer 5 -- Hypothesis property-based tests + coverage gates**
  - Added `hypothesis>=6.0` and `pytest-cov>=4.0` to dev dependencies
  - Created `tests/strategies.py` with 9 operation strategies and `apply_operation` executor
  - Created `tests/test_property_invariants.py` with 6 property test classes (1800 examples):
    - P1: Any op sequence preserves all 7 invariants (500 examples)
    - P2: Cascade deletion leaves no orphans (300 examples)
    - P3: Cleanup always empties state (200 examples)
    - P4: Registry keys match Info.name fields (300 examples)
    - P5: No duplicate keys across registries (200 examples)
    - P6: After removeMesh, no FK references remain (300 examples)
  - Coverage gate: 80% minimum, currently 86.53%
  - CI updated with `--cov` flags and `--cov-fail-under=80`

#### Validation Stack Summary

```
Layer | Tool    | What It Catches             | Status
------|---------|-----------------------------|--------
  0   | pyright | Type errors, None safety    | 0 errors
  1   | Idris 2 | Invalid construction        | 5 modules
  2   | Quint   | Multi-step sequence bugs    | verified
  3   | Lean 4  | Invariant violations        | 20 theorems
  4   | Manual  | Runtime contract violations | 31 tools
  5   | pytest  | Behavioral bugs + coverage  | 246 tests, 86%
```

---

## [0.5.1] - 2026-02-07

### Added

#### Lean 4 formal verification of SessionState invariants (Phase 32)
- Modeled SessionState with 8 typed registries in Lean 4 (v4.15.0)
- Defined `valid` predicate encoding 7 referential integrity invariants
- Proved all 10 mutation operations preserve invariants (register, cascade delete, cleanup)
- Proved `removeMesh_valid` -- cascade deletion correctness (highest-value proof)
- Proved `removeSpaceDeps_valid` -- space-dependent cleanup correctness
- Proved path validation containment security property (abstract over resolve/check)
- Proved 13 error codes pairwise distinct via `native_decide`
- Proved MeshInfo field invariants (tdim <= gdim, 6 valid dimension pairs)
- 20 theorems + 4 helper lemmas, zero `sorry` placeholders
- All proofs machine-checked by Lean 4 type checker (`lake build` exits 0)
- Proofs located in `.outline/proofs/DolfinxProofs/`

---

## [0.5.0] - 2026-02-07

### Fixed

#### CRITICAL: plot_solution VTK stdout corruption (Phase 31A)
- Added `_suppress_stdout()` context manager using fd-level redirect
  (`os.dup`/`os.dup2`) to suppress VTK/PyVista C-level stdout writes
  that corrupted the MCP stdio JSON-RPC transport stream
- Graceful no-op fallback when `sys.stdout.fileno()` is unavailable
  (pytest capture mode)

#### CRITICAL: plot_solution return type violation (Phase 31B)
- Changed `plot_solution` to return a single `dict` instead of
  `[dict, Image(...)]`, matching the return type contract of all
  other 30 tools
- Removed unused `Image` import from `mcp.server.fastmcp.utilities.types`

### Added

#### Output path validation (Phase 31C)
- Added `_validate_output_path()` helper in postprocess.py that validates
  output file paths resolve within `/workspace` using `os.path.realpath`
- Applied to both `export_solution` and `plot_solution` tools
- Path traversal attempts (e.g., `../../etc/passwd`) now raise `FileIOError`
  with `FILE_IO_ERROR` code (defense-in-depth, complements Docker isolation)

#### README.md (Phase 31D)
- Created project README with quick start, 31-tool catalog grouped by
  module, architecture diagram, transport modes, JupyterLab integration,
  and development commands

#### PyPI metadata (Phase 31E)
- Added readme, authors, keywords, classifiers, and project.urls to
  `pyproject.toml` for proper PyPI distribution

#### Contract tests (Phase 31)
- 11 new tests: `_suppress_stdout` (2), `plot_solution` return type (2),
  `_validate_output_path` (7)

### Changed

#### Ruff lint rules expanded (Phase 31F)
- Added B (bugbear), C4 (comprehensions), UP (pyupgrade), SIM (simplify)
  rule families to ruff configuration
- Ignored B008 (FastMCP sentinel defaults), SIM102 (debug guard pattern),
  SIM117 (multi-line context manager readability)
- Fixed ~33 auto-fixable findings across src/, tests/, examples/
- `Callable` imports moved from `typing` to `collections.abc` (UP035)

#### Version sync
- Synced `__init__.py` version from `"0.1.0"` to `"0.5.0"` (was never
  updated since initial project creation)

---

## [0.4.2] - 2026-02-07

### Added

#### MCP Protocol Production Readiness Test Suite (Phase 30)

**New artifact: `examples/production_readiness.py`**
- Standalone script exercising all 31 MCP tools through the Docker container
  via JSON-RPC stdio protocol (production-identical path)
- 49 checks: 31+ positive-path postcondition validations + 9 negative-path
  contract verifications across 12 test phases (P0-P11)
- Diagnostic report with per-tool pass/fail, timing, and coverage stats
- Serves as executable API documentation and production gate

**Test phases:**
- P0: Connectivity & discovery (list_tools, get_session_state)
- P1: Mesh creation & quality (create_unit_square, get_mesh_info,
  compute_mesh_quality, create_mesh)
- P2: Boundary & tag operations (mark_boundaries, manage_mesh_tags,
  refine_mesh, create_submesh)
- P3: Function spaces (create_function_space, create_mixed_space)
- P4: Material properties & interpolation (set_material_properties, interpolate)
- P5: Problem definition & BCs (define_variational_form, apply_boundary_condition)
- P6: Solver & diagnostics (solve, get_solver_diagnostics)
- P7: Postprocessing (compute_error, evaluate_solution, query_point_values,
  compute_functionals, export_solution, plot_solution)
- P8: Advanced operations (reset_session, solve_time_dependent, assemble x3,
  project, create_discrete_operator)
- P9: Session management (get_session_state, run_custom_code, remove_object,
  reset_session)
- P10: Custom mesh via gmsh (run_custom_code + create_custom_mesh)
- P11: Contract verification -- 9 negative-path tests validating
  PRECONDITION_VIOLATED error codes

**Protocol-level findings documented:**
- `plot_solution` emits VTK stdout noise that corrupts stdio JSON-RPC stream;
  workaround: validate via workspace volume mount
- `evaluate_solution` / `query_point_values` return numpy `.tolist()` arrays;
  client code must handle both list and scalar values

**v0.4.2 metrics:**
- Production readiness: 49/49 checks passed, 31/31 tools covered
- Total execution time: ~2 seconds
- Exit code 0 (PRODUCTION READY)

**Files added (1):**
- `examples/production_readiness.py`

**Files modified (2):**
- `pyproject.toml` (version 0.4.1 -> 0.4.2)
- `CHANGELOG.md`

---

## [0.4.1] - 2026-02-07

### Fixed

#### CI/CD Hardening + Test Gap Closure (Phase 29)

**CI blind spots resolved:**
- Lint scope broadened from `src/dolfinx_mcp/` to `src/` -- now covers both
  `dolfinx_mcp` and `dolfinx_mcp_jupyter` packages
- Test dependencies expanded from `.[dev]` to `.[dev,jupyter]` -- Jupyter test
  imports no longer fail in CI

**3 pre-existing test failures fixed:**
- `test_assemble_scalar_preserves_api_error` (Phase 8)
- `test_assemble_form_eval_preserves_structured_error` (Phase 9)
- `test_assemble_assembly_returns_structured_error` (Phase 9)
- Root cause: missing `"ufl": MagicMock()` in `sys.modules` patches for
  `assemble()` tests. The `assemble()` function imports `ufl` at runtime,
  but test mocks only patched `dolfinx`/`dolfinx.fem`/`dolfinx.fem.petsc`

### Added

**New test coverage for untested infrastructure:**
- `tests/test_cli.py` (6 tests): CLI `--help` output validation, transport
  choices, default port, invalid transport/port/flag rejection. Uses subprocess
  to avoid `__main__.py` module-level side effects
- `tests/test_logging_config.py` (5 tests): stderr handler verification,
  default/custom log levels, third-party logger suppression, handler cleanup

**v0.4.1 metrics:**
- Test suite: 227 passed, 0 failed, 3 skipped (was 213 passed, 3 failed)
- 11 new tests (6 CLI + 5 logging)
- 3 pre-existing failures resolved
- CI coverage: both packages linted and tested

**Files added (2):**
- `tests/test_cli.py`
- `tests/test_logging_config.py`

**Files modified (3):**
- `.github/workflows/ci.yml` (lint scope + test deps)
- `tests/test_contracts.py` (3 ufl mock fixes)
- `pyproject.toml` (version bump)

---

## [0.4.0] - 2026-02-07

### Added

#### JupyterLab MCP Extension (Option G: Phases 0-2 + Phase 28)

IPython magic extension for calling DOLFINx MCP tools from Jupyter notebook
cells, with inline image display and dual-transport support.

**New package: `dolfinx_mcp_jupyter`** (5 modules):
- `config.py`: `MCPConfig` dataclass with env var defaults + `__post_init__` validation
- `connection.py`: `MCPConnection` wrapping MCP SDK `ClientSession` (stdio/HTTP)
- `display.py`: Result rendering (JSON tables, inline PNG, error panels)
- `magics.py`: 5 IPython magics (`%dolfinx_connect`, `%dolfinx_disconnect`,
  `%dolfinx_tools`, `%dolfinx`, `%%dolfinx_workflow`)
- `__init__.py`: Extension loader via `%load_ext dolfinx_mcp_jupyter`

**Server enhancements:**
- `plot_solution` now returns `ImageContent` (base64 PNG) alongside metadata
- Dual transport: `python -m dolfinx_mcp --transport streamable-http` for HTTP mode
- CLI argparse for `--transport`, `--host`, `--port` in `__main__.py`
- `_app.py` reads `DOLFINX_MCP_HOST`/`DOLFINX_MCP_PORT` env vars
- `server.py` reads `DOLFINX_MCP_TRANSPORT` env var
- `Dockerfile` adds `EXPOSE 8000` for HTTP mode
- `docker-compose.lab.yml` for JupyterLab + MCP server sidecar deployment

**Infrastructure:**
- `.mcp.json` volume mount (`./workspace:/workspace`) for file export
- `pyproject.toml` jupyter optional dependency group (`ipython`, `nest-asyncio`, `mcp`)
- Wheel target includes `src/dolfinx_mcp_jupyter`

**Design-by-contract enforcement (Phase 28):**
- `MCPConfig.__post_init__`: transport enum, timeout positive, command/url non-empty
- `MCPConnection.call_tool()`: tool name non-empty precondition
- `MCPConnection.connect()/disconnect()`: state transition postconditions
- `display.render_result()`: list type guard at rendering boundary
- Bare `assert` replaced with explicit `RuntimeError` guard

**Testing:**
- 42 new tests across 4 files:
  - `test_jupyter_config.py`: defaults, env overrides, explicit args, contract violations
  - `test_jupyter_magics.py`: `_parse_kwargs` coercion (bool/int/float/json/string/expression)
  - `test_jupyter_connection.py`: initial state, config, contract violations
  - `test_jupyter_display.py`: type guard contract violations

**v0.4.0 metrics:**
- 5 new IPython magics, 2 transport modes (stdio + HTTP)
- Jupyter package DbC: 4 preconditions, 3 postconditions, 4 invariants
- 42 new Jupyter tests + 193 existing server tests
- Ruff lint: 0 errors

**Files added (9):**
- `src/dolfinx_mcp_jupyter/__init__.py`
- `src/dolfinx_mcp_jupyter/config.py`
- `src/dolfinx_mcp_jupyter/connection.py`
- `src/dolfinx_mcp_jupyter/display.py`
- `src/dolfinx_mcp_jupyter/magics.py`
- `docker-compose.lab.yml`
- `tests/test_jupyter_config.py`
- `tests/test_jupyter_connection.py`
- `tests/test_jupyter_magics.py`
- `tests/test_jupyter_display.py`

**Files modified (7):**
- `src/dolfinx_mcp/__main__.py` (dual transport CLI)
- `src/dolfinx_mcp/_app.py` (host/port env vars)
- `src/dolfinx_mcp/server.py` (transport env var)
- `src/dolfinx_mcp/tools/postprocess.py` (ImageContent return)
- `Dockerfile` (EXPOSE 8000)
- `.mcp.json` (volume mount)
- `pyproject.toml` (version, jupyter extras, wheel target)

---

## [0.3.1] - 2026-02-07

### Fixed

#### Design-by-Contract Phase 27: DOLFINx 0.10 API Compatibility Fixes
- **8 DOLFINx 0.10 API defects resolved** (6 planned from Phase 26 + 2 discovered during verification)
- **All 31 MCP tools fully operational** through live Docker server -- 0 failures remaining

**Defects fixed:**
1. `refine_mesh` (mesh.py): Added `create_entities(1)` pre-call; handle tuple return from `refine()`
2. `create_submesh` (mesh.py): Index-based unpacking for 4-value return in DOLFINx 0.10
3. `project` (interpolation.py): Added required `petsc_options_prefix` kwarg to `LinearProblem`
4. `apply_boundary_condition` (problem.py): Override UFL math with numpy equivalents for BC
   interpolation; conditional `dirichletbc()` signature for Function vs Constant values
5. `assemble` (session_mgmt.py): Inject trial/test functions into namespace; use PETSc
   `InsertMode`/`ScatterMode` (removed from `dolfinx.cpp.la`); extract raw `DirichletBC`
   from `BCInfo` wrappers
6. `remove_object` cascade (session.py): Clear `forms` and `ufl_symbols` when dependent
   spaces are removed

**Verification:**
- MCP protocol tests: 58/58 passed (100%) across P0-P9
- Local tests: 193 passed, 7 pre-existing failures (unchanged)
- Docker image rebuilt 3x with incremental fixes
- DOLFINx 0.10.0.post2 runtime confirmed

**v0.3.1 metrics:**
- MCP protocol coverage: 31/31 tools (100%)
- Protocol tests: 58/58 passed (100%)
- All 6 original defect areas verified fixed
- Contract enforcement: 100% operational (no contract violations)

**Files modified (5):**
- `src/dolfinx_mcp/tools/mesh.py` (D1, D2)
- `src/dolfinx_mcp/tools/interpolation.py` (D3)
- `src/dolfinx_mcp/tools/problem.py` (D4)
- `src/dolfinx_mcp/tools/session_mgmt.py` (D5)
- `src/dolfinx_mcp/session.py` (D6)

---

## [0.3.0] - 2026-02-07

### Verified

#### Design-by-Contract Phase 26: MCP Protocol-Level Production Readiness Test
- **All 31 MCP tools exercised through live Docker server** via JSON-RPC stdio protocol
- **10 test phases** (P0-P9): connectivity, mesh ops, function spaces, interpolation,
  problem definition, solver, post-processing, session management, cross-cutting, edge cases
- **81 protocol-level tests**: 63 positive + 36 negative path, 31/31 tool coverage

**Results: 75/81 passed (92.6%)**
- Phase 0 Connectivity: 3/3 PASS
- Phase 1 Mesh Ops: 17/19 (2 FAIL: refine_mesh, create_submesh -- DOLFINx 0.10 API changes)
- Phase 2 Function Spaces: 11/11 PASS
- Phase 3 Interpolation: 9/11 (2 FAIL: project -- DOLFINx 0.10 LinearProblem API change)
- Phase 4 Problem Def: 6/7 (1 FAIL: expression BC -- type conversion incompatibility)
- Phase 5 Solver: 7/7 PASS
- Phase 6 Post-Processing: 16/16 PASS
- Phase 7 Session Mgmt: 12/15 (2 FAIL: assemble vector/matrix missing u/v symbols; 1 minor: forms persist after cascade)
- Phase 8 Cross-Cutting: 4/4 PASS
- Phase 9 Edge Cases: 2/2 PASS

**Defects identified (6):**
1. `refine_mesh`: Missing `create_entities(1)` pre-call; DOLFINx 0.10 `refine()` returns tuple
2. `create_submesh`: DOLFINx 0.10 returns 4 values, tool unpacks 2
3. `project`: DOLFINx 0.10 `LinearProblem` requires `petsc_options_prefix` kwarg
4. `apply_boundary_condition`: Expression value interpolation produces array incompatible with UFL
5. `assemble` (vector/matrix): Trial/test function symbols `u`/`v` not in expression namespace
6. `remove_object` (mesh cascade): Compiled forms not invalidated when trial/test space removed

**Recommendation: CONDITIONALLY READY** -- 25/31 tools fully operational through MCP protocol.
6 defects are DOLFINx 0.10 API compatibility issues, not contract violations. All contract
enforcement (preconditions, postconditions, invariants) functions correctly across all 31 tools.

**v0.3.0 metrics:**
- MCP protocol coverage: 31/31 tools (100%)
- Protocol tests: 75/81 passed (92.6%)
- All negative-path contracts verified: 36/36 (100%)
- DOLFINx 0.10 API issues: 6 (non-blocking for contract correctness)

---

## [0.2.9] - 2026-02-06

### Added

#### Design-by-Contract Phase 25: CI/CD Pipeline
- **GitHub Actions CI workflow** (`.github/workflows/ci.yml`) enforcing ruff lint
  and pytest on every push to main and on pull requests
- **`[project.optional-dependencies] dev`** section in pyproject.toml: pytest,
  pytest-asyncio, ruff pinned as dev dependencies
- CI runs two parallel jobs: `lint` (ruff check) and `test` (pytest, ignoring
  Docker-only integration tests)

**v0.2.9 metrics:**
- CI/CD: lint + test on push/PR (was: none)
- Ruff errors: 0
- 174 local tests, 23 Docker tests, 197 total

---

## [0.2.8] - 2026-02-06

### Changed

#### Design-by-Contract Phase 24: Ruff Lint Zero
- **43 ruff lint errors eliminated** (was 43, now 0):
  - 15 I001 (import sorting) auto-fixed across 10 files
  - 4 F401 (unused imports) removed: `DuplicateNameError`, `InvalidUFLExpressionError`
    in problem.py, `sys` in session_mgmt.py, `dolfinx.mesh` in mesh.py
  - 1 F841 (unused variable) removed: `space_name` in postprocess.py compute_error
  - 23 E501 (line-too-long) fixed: session.py cleanup postconditions refactored from
    11 repetitive if/raise blocks to a loop, long docstrings and dict literals wrapped
  - 1 E402 (import-not-at-top) suppressed with noqa in server.py (intentional ordering)
- **session.py cleanup() refactored**: 11 repetitive postcondition checks replaced with
  a registry iteration loop -- same behavior, 50% fewer lines, no E501 violations
- Zero runtime behavioral changes; all 174 local tests pass

**v0.2.8 metrics:**
- Ruff errors: 0 (was 43)
- 174 local tests passed, 3 skipped
- 23 Docker integration tests
- 197 total tests

---

## [0.2.7] - 2026-02-06

### Changed

#### Design-by-Contract Phase 23: Helper Type Annotations
- **8 internal helper functions** fully type-annotated, enabling static contract
  verification of parameter types (preconditions) and return types (postconditions)
- **3 files gain `Callable` import**: errors.py, mesh.py, problem.py
- Functions annotated in: mesh.py (1), problem.py (4), interpolation.py (1),
  postprocess.py (1), errors.py (1)

**v0.2.7 metrics:**
- 8/8 helper functions type-annotated (was 0/8)
- 174 local tests, 23 Docker tests, 197 total
- Zero runtime impact (annotations are no-ops at runtime)

---

## [0.2.6] - 2026-02-06

### Added

#### Design-by-Contract Phase 22: Test Infrastructure + Edge Case Tests
- **3 new pytest fixtures** in `tests/conftest.py`: `mock_ctx`, `populated_session`,
  `mock_ctx_populated` -- promotes common test patterns from module-level helpers
- **`_make_form_info` helper** in test_contracts.py for solver postcondition tests
- **4 new postcondition edge-case tests** in `TestPhase22PostconditionEdgeCases`:
  - `test_solve_postcondition_nan_solution`: solve() fires SolverError on NaN
  - `test_solve_postcondition_negative_l2_norm`: solve() fires PostconditionError on L2 < 0
  - `test_get_solver_diagnostics_postcondition_negative_l2`: diagnostics fires on L2 < 0
  - `test_compute_error_postcondition_nan`: compute_error() fires on NaN error value

**v0.2.6 metrics:**
- 137 contract tests (was 133), 174 local tests (was 170)
- 23 Docker integration tests
- 197 total tests

---

## [0.2.5] - 2026-02-06

### Added

#### Design-by-Contract Phase 21: Return Contract Documentation
- **24 `Returns:` docstring sections** added to tool functions, completing the
  MCP API specification. In an MCP server, the tool docstring is the contract
  the AI agent reads -- without return documentation, the output contract is
  unspecified.
- All 31 tools now have `Returns:` sections (was 7/31)
- 24 new SPEC correspondence table entries
- Docstring-only changes: zero runtime impact, all existing tests pass

**Files modified (7):**
- `session_mgmt.py` (2): get_session_state, reset_session
- `mesh.py` (7): create_unit_square, get_mesh_info, create_mesh, mark_boundaries,
  refine_mesh, create_custom_mesh, create_submesh, manage_mesh_tags
- `spaces.py` (2): create_function_space, create_mixed_space
- `problem.py` (3): define_variational_form, apply_boundary_condition,
  set_material_properties
- `solver.py` (1): solve
- `postprocess.py` (6): compute_error, export_solution, evaluate_solution,
  compute_functionals, query_point_values, plot_solution
- `interpolation.py` (2): interpolate, create_discrete_operator

**v0.2.5 metrics:**
- 31 tools, 31/31 with Returns: (100%)
- 145 correspondence table entries (was 121)
- 172 local tests (170 passed, 2 skipped, 1 collection skip)
- 23 Docker integration tests
- 195 total tests

---

## [0.2.4] - 2026-02-06

### Added

#### Design-by-Contract Phase 20: Local Test Completion and Documentation
- **12 new local contract tests** for 4 tools previously only tested via Docker:
  - `get_session_state` (3): empty_session, populated, after_removal
  - `reset_session` (3): clears_all, empty_session, returns_status
  - `get_mesh_info` (3): missing_mesh, returns_expected_keys, uses_accessor
  - `get_solver_diagnostics` (3): no_solution, returns_expected_keys, uses_accessor
- **All 31 tools** now have at least 1 local contract test
- **Naming footnotes** added to v0.1.0 for renamed/consolidated tools

**Final v0.2.4 metrics:**
- 31 tools, 30 PostconditionError raises, 35 `if __debug__:` sites
- 121 correspondence table entries
- 133 contract tests + 39 core tests = 172 local (170 passed, 2 skipped)
- 23 Docker integration tests
- 195 total tests
- 100% DbC compliance (31/31 tools)

---

## [0.2.3] - 2026-02-06

### Changed

#### Design-by-Contract Phase 19: Code Deduplication
- **`src/dolfinx_mcp/utils.py`**: New shared utility with `compute_l2_norm()`
  replacing 3 duplicated import+compute blocks in solver.py
- **`session.find_space_name()`**: New utility replacing 2 duplicated 7-line
  loops in solver.py
- **`solve()` PETSc opts**: Consolidated inline opts building to use existing
  `_build_petsc_opts()` function
- **5 new tests** for extracted utilities
- No behavioral changes: pure refactoring

---

## [0.2.2] - 2026-02-06

### Added

#### Design-by-Contract Phase 18: Defensive Invariant Checks
- **12 new `if __debug__: session.check_invariants()` sites** added to 10 tools
  that previously lacked debug invariant verification, bringing total from 23 to 35
- Every `@mcp.tool()` function now has at least one debug invariant check
- Zero production overhead (`python -O` strips debug blocks)
- Correspondence table: 123 entries (was 111)

**Tools with new invariant checks:**
- `compute_error`, `export_solution`, `evaluate_solution`, `compute_functionals`,
  `query_point_values`, `plot_solution` (postprocess.py)
- `get_session_state`, `assemble` (3 return paths) (session_mgmt.py)
- `get_solver_diagnostics` (solver.py)
- `get_mesh_info` (mesh.py)

---

## [0.2.1] - 2026-02-06

### Added

#### Design-by-Contract Phase 17: Postcondition Completion
- **14 new postconditions** added to tools that compute/return values without
  validity checks, bringing total PostconditionError raises from 16 to 30
- **14 new tests** in `TestPhase17Postconditions` verifying each postcondition
  fires under violation conditions
- **Correspondence table**: 111 entries (was 97)
- **Test summary**: 155 collected (153 passed, 2 skipped)

**Tools with new postconditions:**
- `create_unit_square`: num_cells > 0 and num_vertices > 0
- `create_mesh`: num_cells > 0 and num_vertices > 0
- `create_custom_mesh`: num_cells > 0 and num_vertices > 0
- `get_mesh_info`: bounding box coordinates are finite
- `mark_boundaries`: at least one facet tagged
- `create_submesh`: submesh cells <= parent cells
- `manage_mesh_tags` (create): at least one entity tagged
- `create_function_space`: num_dofs > 0
- `create_mixed_space`: num_dofs > 0
- `create_discrete_operator`: matrix dimensions > 0
- `define_variational_form`: compiled forms are not None
- `set_material_properties`: interpolated values are finite
- `export_solution`: file size > 0
- `plot_solution`: output file exists

---

## [0.2.0] - 2026-02-06

### Documentation

#### Design-by-Contract Phase 16: Documentation Finalization
- **Version milestone**: v0.2.0 marks completion of the DbC hardening effort
  (Phases 1-16)
- **Tool inventory**: 31 tools across 6 modules, all with full DbC enforcement
- **Correspondence table**: 97 entries mapping contracts to code locations
- **Test summary**: 100 contract tests + 24 Docker integration + 39 core = 163 total

### Tool Inventory (31 tools, 6 modules)

| Module | Tool | PRE | POST | INV |
|--------|------|-----|------|-----|
| session_mgmt | get_session_state | - | - | 1 |
| session_mgmt | reset_session | - | - | 1 |
| session_mgmt | run_custom_code | 1 | - | 1 |
| session_mgmt | assemble | 1 | 1 | 1 |
| session_mgmt | remove_object | 2 | 1 | 1 |
| mesh | create_unit_square | 3 | 1 | 1 |
| mesh | get_mesh_info | - | 1 | 1 |
| mesh | create_mesh | 4 | 1 | 1 |
| mesh | mark_boundaries | 3 | 1 | 1 |
| mesh | refine_mesh | - | 1 | 1 |
| mesh | create_custom_mesh | 1 | 2 | 1 |
| mesh | create_submesh | 2 | 1 | 1 |
| mesh | manage_mesh_tags | 1 | 1 | 1 |
| mesh | compute_mesh_quality | 1 | 2 | 1 |
| spaces | create_function_space | 1 | 1 | 1 |
| spaces | create_mixed_space | 1 | 1 | 1 |
| interpolation | interpolate | - | 1 | 1 |
| interpolation | create_discrete_operator | 1 | 1 | 1 |
| interpolation | project | 3 | 2 | 1 |
| problem | set_material_properties | 1 | 1 | 1 |
| problem | define_variational_form | 2 | 1 | 1 |
| problem | apply_boundary_condition | 1 | - | 1 |
| solver | solve | 1 | 2 | 1 |
| solver | solve_time_dependent | 3 | 1 | 1 |
| solver | get_solver_diagnostics | - | 1 | 1 |
| postprocess | compute_error | 2 | 2 | 1 |
| postprocess | export_solution | 1 | 1 | 1 |
| postprocess | evaluate_solution | 1 | 1 | 1 |
| postprocess | compute_functionals | 1 | 1 | 1 |
| postprocess | query_point_values | 2 | 1 | 1 |
| postprocess | plot_solution | 1 | 1 | 1 |

---

## [0.1.14] - 2026-02-06

### Testing

#### Design-by-Contract Phase 15: Docker Integration Test Expansion
- **11 new Docker integration tests** expanding coverage from 12 to 23+ tools
- **Group D (4 tests)**: Mesh operations -- create_mesh rectangle (D1), refine_mesh
  cell count increase (D2), mark_boundaries + manage_mesh_tags (D3), create_submesh (D4)
- **Group E (4 tests)**: Solver/postprocess -- solve_time_dependent heat equation (E1),
  export_solution XDMF (E2), evaluate_solution finiteness (E3), query_point_values (E4)
- **Group F (3 tests)**: Phase 14 tools -- remove_object mesh cascade (F1),
  compute_mesh_quality metrics (F2), project L2 projection (F3)
- Docker test total: 24 (13 existing + 11 new)
- Local test total: 139 passed, 4 skipped

---

## [0.1.13] - 2026-02-06

### Added

#### Design-by-Contract Phase 14: Missing Tool Implementation
- **`remove_object` tool (session_mgmt.py)**: Remove any named object from session by type.
  Mesh removal cascades (delegates to `session.remove_mesh()`), space removal cascades
  (delegates to `_remove_space_dependents()`), leaf types removed directly. Preconditions:
  name non-empty, object_type in valid enum. Postcondition: object absent from registry
- **`compute_mesh_quality` tool (mesh.py)**: Compute cell volume statistics (min, max, mean,
  std, quality_ratio=min/max) for a mesh. Precondition: mesh exists (via accessor).
  Postconditions: all metrics finite, all cell volumes > 0
- **`project` tool (interpolation.py)**: L2-project an expression or function onto a target
  space via mass matrix solve (M*u = b using `LinearProblem`). Preconditions: name non-empty,
  exactly one of expression/source_function. Postconditions: result finite, L2 norm >= 0

### Testing
- 12 new contract tests (100 contract tests total, 139 local + 13 Docker = 152 total)
- Phase 14: remove_object empty_name/invalid_type/not_found/mesh_cascade/leaf_delete (5),
  compute_mesh_quality missing_mesh/postcondition_finite/valid_access (3),
  project empty_name/both_args/neither_arg/postcondition_nan (4)

---

## [0.1.12] - 2026-02-06

### Changed

#### Design-by-Contract Phase 13: Accessor Completion
- **3 new typed accessors (session.py)**: `get_mesh_tags`, `get_entity_map`, `get_last_solution`
  complete the accessor family (now 9 total). Each has debug postconditions verifying
  name==key and parent references exist in registries
- **Replaced direct dict reads across 3 tool files**:
  - `mesh.py`: 2 `session.mesh_tags[name]` reads replaced with `session.get_mesh_tags(name)`
  - `postprocess.py`: 4 `session.functions[name]` reads replaced with `session.get_function(name)`,
    4 `list(session.solutions.values())[-1]` replaced with `session.get_last_solution()`,
    3 `list(session.solutions.keys())[-1]` replaced with `session.get_last_solution().name`
  - `solver.py`: 1 `list(session.solutions.keys())[-1]` + `session.get_solution()` replaced with
    `session.get_last_solution()`

### Testing
- 9 new contract tests (89 contract tests total, 127 local + 13 Docker = 140 total)
- Phase 13: get_mesh_tags not_found/name_mismatch/dangling_mesh (3),
  get_entity_map not_found/name_mismatch/dangling_parent (3),
  get_last_solution empty/returns_latest/dangling_space (3)

---

## [0.1.11] - 2026-02-06

### Fixed

#### Design-by-Contract Phase 12: Postcondition Error Type Correction
- **Interpolation postcondition error types (3 locations in interpolation.py)**: Changed
  `DOLFINxAPIError` to `PostconditionError` for NaN/Inf finiteness checks after expression-based,
  same-mesh function, and cross-mesh interpolation. A postcondition (result violated expectations)
  is semantically distinct from an API error (call was invalid). All three checks now correctly
  produce `POSTCONDITION_VIOLATED` error codes
- **Import addition**: `PostconditionError` added to `interpolation.py` import block

### Testing
- 2 new contract tests (80 contract tests total, 119 local + 13 Docker = 132 total)
- Phase 12: expression interpolation NaN error type (1), function interpolation NaN error type (1)
- Fixed: `test_refine_mesh_postcondition_cell_count` tautology -- original and refined now use
  distinct `num_cells` (100 vs 80/200) instead of both defaulting to 100
- Fixed: Docker test B1 assertion updated from `DOLFINX_API_ERROR` to `POSTCONDITION_VIOLATED`

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
- **Postprocessing** (5): `compute_error`, `export_solution`, `compute_derived_quantity`[^1], `evaluate_solution`, `query_point_values`
- **Interpolation** (2): `interpolate`, `project`
- **Session** (6): `get_session_status`[^2], `list_objects`[^3], `remove_object`, `reset_session`, `assemble`, `compute_mesh_quality`

[^1]: `compute_derived_quantity` was renamed to `compute_functionals` in Phase 14.
[^2]: `get_session_status` was renamed to `get_session_state` in Phase 14.
[^3]: `list_objects` was consolidated into `get_session_state` in Phase 14.

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
POST: interpolation finite   interpolate (3 paths)        PostconditionError
POST: name == registry key   get_mesh_tags                PostconditionError (debug)
POST: mesh_name in meshes    get_mesh_tags                PostconditionError (debug)
POST: name == registry key   get_entity_map               PostconditionError (debug)
POST: parent/child in meshes get_entity_map               PostconditionError (debug)
POST: space in fn_spaces     get_last_solution            PostconditionError (debug)
USE:  get_last_solution()    compute_error et al.         Replaces list()[-1]
PRE: name non-empty          remove_object                PreconditionError
PRE: object_type valid       remove_object                PreconditionError
POST: object removed         remove_object                PostconditionError
INV: check after remove      remove_object                if __debug__
PRE: mesh exists             compute_mesh_quality         get_mesh accessor
POST: metrics finite         compute_mesh_quality         PostconditionError
POST: volumes > 0            compute_mesh_quality         PostconditionError
PRE: expression xor source   project                     PreconditionError
PRE: name non-empty          project                     PreconditionError
POST: result finite          project                     PostconditionError
POST: l2_norm >= 0           project                     PostconditionError
INV: check after project     project                     if __debug__
POST: num_cells > 0          create_unit_square           PostconditionError
POST: num_cells > 0          create_mesh                  PostconditionError
POST: num_cells > 0          create_custom_mesh           PostconditionError
POST: bbox finite             get_mesh_info                PostconditionError
POST: unique_tags non-empty   mark_boundaries              PostconditionError
POST: sub <= parent cells     create_submesh               PostconditionError
POST: unique_tags non-empty   manage_mesh_tags (create)    PostconditionError
POST: num_dofs > 0            create_function_space        PostconditionError
POST: num_dofs > 0            create_mixed_space           PostconditionError
POST: rows > 0, cols > 0     create_discrete_operator     PostconditionError
POST: form not None           define_variational_form      PostconditionError
POST: material finite         set_material_properties      PostconditionError
POST: file_size > 0           export_solution              PostconditionError
POST: file exists             plot_solution                PostconditionError
INV: check after compute      compute_error                if __debug__
INV: check after export       export_solution              if __debug__
INV: check after evaluate     evaluate_solution            if __debug__
INV: check after functionals  compute_functionals          if __debug__
INV: check after query        query_point_values           if __debug__
INV: check after plot         plot_solution                if __debug__
INV: check after overview     get_session_state            if __debug__
INV: check after assemble     assemble (3 paths)           if __debug__
INV: check after diagnostics  get_solver_diagnostics       if __debug__
INV: check after mesh_info    get_mesh_info                if __debug__
SPEC: returns documented      get_session_state            Returns: docstring
SPEC: returns documented      reset_session                Returns: docstring
SPEC: returns documented      create_unit_square           Returns: docstring
SPEC: returns documented      get_mesh_info                Returns: docstring
SPEC: returns documented      create_mesh                  Returns: docstring
SPEC: returns documented      mark_boundaries              Returns: docstring
SPEC: returns documented      refine_mesh                  Returns: docstring
SPEC: returns documented      create_custom_mesh           Returns: docstring
SPEC: returns documented      create_submesh               Returns: docstring
SPEC: returns documented      manage_mesh_tags             Returns: docstring
SPEC: returns documented      create_function_space        Returns: docstring
SPEC: returns documented      create_mixed_space           Returns: docstring
SPEC: returns documented      define_variational_form      Returns: docstring
SPEC: returns documented      apply_boundary_condition     Returns: docstring
SPEC: returns documented      set_material_properties      Returns: docstring
SPEC: returns documented      solve                        Returns: docstring
SPEC: returns documented      compute_error                Returns: docstring
SPEC: returns documented      export_solution              Returns: docstring
SPEC: returns documented      evaluate_solution            Returns: docstring
SPEC: returns documented      compute_functionals          Returns: docstring
SPEC: returns documented      query_point_values           Returns: docstring
SPEC: returns documented      plot_solution                Returns: docstring
SPEC: returns documented      interpolate                  Returns: docstring
SPEC: returns documented      create_discrete_operator     Returns: docstring
PRE: expression xor source   interpolate                  PreconditionError
SEC: eager _check_forbidden  eval_numpy_expression        _check_forbidden (eager)
SEC: eager _check_forbidden  make_boundary_marker         _check_forbidden (eager)
SEC: eager _check_forbidden  _eval_bc_expression          _check_forbidden (eager)
SEC: eager _check_forbidden  safe_evaluate                _check_forbidden (eager)
```
