# DOLFINx MCP Server

MCP server for FEniCSx/DOLFINx finite element computing. Version 0.8.0.

## Quick Reference

```bash
# Install (dev mode)
pip install -e ".[dev]"

# Tests (no Docker needed, ~341 tests)
pytest tests/ --ignore=tests/test_runtime_contracts.py --ignore=tests/test_tutorial_workflows.py -v

# Tests with coverage (threshold set in pyproject.toml [tool.coverage.report].fail_under = 90)
pytest tests/ --ignore=tests/test_runtime_contracts.py --ignore=tests/test_tutorial_workflows.py --cov=dolfinx_mcp --cov-report=term-missing -v

# Lint
ruff check src/ tests/

# Type check
pyright src/dolfinx_mcp/

# Docker build + integration tests
docker build -t dolfinx-mcp .
pytest tests/test_runtime_contracts.py -v
```

## Architecture

### Import DAG (strict, no cycles)

```
errors.py, session.py, ufl_context.py, eval_helpers.py, utils.py  (leaf modules)
        ↓
     _app.py            (FastMCP instance + lifespan)
        ↓
   tools/*.py           (7 tool modules, each imports _app.mcp)
        ↓
    server.py           (imports all tools for registration)
        ↓
   __main__.py          (entry point)
```

**Rule**: Never import from a higher layer into a lower layer.

### Source Layout

```
src/dolfinx_mcp/
    __init__.py          Version string
    server.py            Entry point, imports all modules for registration
    _app.py              FastMCP instance + app_lifespan
    session.py           SessionState (8 registries, 7 invariants)
    errors.py            13 error classes + @handle_tool_errors decorator
    ufl_context.py       Token blocklist + restricted UFL expression evaluation
    eval_helpers.py      Shared numpy expression + boundary marker helpers
    utils.py             compute_l2_norm helper
    logging_config.py    stderr-only logging (stdout reserved for JSON-RPC)
    tools/
        mesh.py          9 mesh tools
        spaces.py        2 function space tools
        problem.py       3 problem definition tools
        solver.py        5 solver tools (solve [+nullspace_mode], solve_time_dependent, get_solver_diagnostics, solve_nonlinear, solve_eigenvalue)
        postprocess.py   6 post-processing tools
        interpolation.py 3 interpolation tools
        session_mgmt.py  5 session management tools
    prompts/templates.py 6 workflow prompt templates
    resources/providers.py 6 URI resources
```

## Design-by-Contract (DbC)

Every tool follows the pattern:
1. **Preconditions** first (before any imports or expensive operations)
2. **Business logic** with try/except for DOLFINx API calls
3. **Postconditions** after operations (always-on for critical checks, `if __debug__:` for expensive checks)
4. **Invariant check**: `if __debug__: session.check_invariants()` at end

### Error Hierarchy

All errors extend `DOLFINxMCPError` and have `error_code`, `suggestion`, `to_dict()`.

Key errors: `PreconditionError`, `PostconditionError`, `InvariantError`,
`NoActiveMeshError`, `MeshNotFoundError`, `FunctionSpaceNotFoundError`,
`InvalidUFLExpressionError`, `SolverError`, `DuplicateNameError`.

### Error Selection Guide

| Situation | Error Class |
|-----------|------------|
| Invalid tool input parameter | `PreconditionError` |
| Mutually exclusive arguments | `PreconditionError` |
| Named object not found | `MeshNotFoundError`, `FunctionSpaceNotFoundError`, `FunctionNotFoundError` |
| No active mesh set | `NoActiveMeshError` |
| Duplicate name in registry | `DuplicateNameError` |
| DOLFINx API call failed | `DOLFINxAPIError` (wrap caught Exception) |
| Solver did not converge | `SolverError` |
| Bad UFL/forbidden token | `InvalidUFLExpressionError` |
| File I/O failure | `FileIOError` |
| Result validation failed | `PostconditionError` |
| Session state inconsistent | `InvariantError` |
| Dataclass `__post_init__` | `InvariantError` (by design: data structure invariant) |

### @handle_tool_errors Decorator

Wraps every `@mcp.tool()` function. Catches `DOLFINxMCPError` -> returns structured dict.
Preserves `__signature__` for FastMCP schema generation.

## Session State

`SessionState` has 8 typed registries: meshes, function_spaces, functions, bcs, forms, solutions, mesh_tags, entity_maps.

7 referential integrity invariants (INV-1 through INV-7) verified by `check_invariants()`.
Cascade deletion: removing a mesh removes all dependent spaces, functions, BCs, solutions, tags, and entity maps.

## Security Model

UFL expressions are Python syntax (no separate parser). Expression evaluation is required.

**Three-layer defense**:
1. Token blocklist (`_check_forbidden()` in `ufl_context.py`) - blocks import, __, exec, open, os., sys., subprocess, etc.
2. Empty `__builtins__` dict in expression namespace
3. Docker container isolation (`--network none`, non-root, `--rm`)

**Important**: All expression evaluation sites MUST call `_check_forbidden()` EAGERLY (at creation time, not deferred inside lambdas).
The `run_custom_code` tool intentionally bypasses the blocklist (full `__builtins__`, documented) -- Docker is the only boundary there.

### Expression Evaluation Patterns

Two distinct evaluation contexts:

1. **Numpy expressions** (interpolation, exact solutions, materials): Use `eval_numpy_expression(expr, x)` from `eval_helpers.py`. Namespace provides numpy math functions operating on coordinate arrays.
2. **UFL symbolic expressions** (variational forms, functionals): Use `safe_evaluate(expr, namespace)` from `ufl_context.py` with `build_namespace(session)`. Namespace provides UFL operators for symbolic form assembly.

Both call `_check_forbidden()` eagerly. Never bypass this.

### UFL Namespace Features (v0.7.0)

- **`split()`**: Available for decomposing mixed-space functions into sub-components (e.g., `split(u)[0]` for the first sub-field of a mixed function).
- **`ds(tag)`**: When `mark_boundaries()` has been called, `ds` is automatically upgraded to a `ufl.Measure` with `subdomain_data`, enabling `ds(1)`, `ds(2)` etc. for tagged boundary integrals. Falls back to plain `ufl.ds` when no boundary tags exist.
- **`nullspace_mode`** on `solve()`: Accepts `"constant"` (scalar Neumann) or `"rigid_body"` (vector elasticity) to attach a PETSc nullspace to the system matrix for singular problems.

## Conventions

- **Lazy imports**: All DOLFINx/UFL/NumPy imports are inside function bodies (not at module level) because the MCP server runs on the host but DOLFINx is only in Docker.
- **Tool pattern**: `@mcp.tool()` then `@handle_tool_errors` (order matters).
- **Context sentinel**: `ctx: Context = None` is the FastMCP pattern (ruff B008 is ignored for this).
- **Logging**: All logging goes to stderr. stdout is reserved for JSON-RPC protocol.
- **Version**: Update in `__init__.py`, `pyproject.toml`, and `README.md` simultaneously.

## New Tool Checklist

1. **Choose the correct module** in `tools/` (mesh, spaces, problem, solver, postprocess, interpolation, session_mgmt).
2. **Decorator pair**: `@mcp.tool()` then `@handle_tool_errors` (reversed order breaks schema generation).
3. **Signature**: `async def tool_name(..., ctx: Context = None) -> dict[str, Any]:`
4. **Docstring**: First line = MCP tool description. Include Args and Returns sections.
5. **Preconditions first**: Validate all inputs BEFORE lazy imports. Use `PreconditionError`.
6. **Lazy imports**: `import dolfinx...` / `import numpy as np` inside the function body.
7. **Session access**: `session = _get_session(ctx)` (every tool module has this helper).
8. **Business logic**: Wrap DOLFINx API calls in try/except. Re-raise `DOLFINxMCPError`, wrap others in `DOLFINxAPIError`.
9. **Postconditions**: Check results (finite values, non-empty, positive). Use `PostconditionError`.
10. **Invariant check**: `if __debug__: session.check_invariants()` at end.
11. **Logging**: `logger.info(...)` with operation summary.
12. **Return dict**: All tools return `dict[str, Any]` with documented keys.
13. **Tests**: Add tests in `tests/` using the `mock_ctx` fixture from conftest.py.

## Anti-Patterns

- **DO NOT** import DOLFINx/UFL/NumPy at module level (lazy imports only).
- **DO NOT** reverse decorator order (`@handle_tool_errors` then `@mcp.tool()` breaks schema generation).
- **DO NOT** print to stdout (corrupts JSON-RPC transport). Use `logger.info()`.
- **DO NOT** use `DOLFINxAPIError` for input validation. Use `PreconditionError`.
- **DO NOT** skip `if __debug__: session.check_invariants()` at end of tool functions.
- **DO NOT** add expression evaluation without calling `_check_forbidden()` eagerly.

## Testing

- Tests mock all DOLFINx imports (no Docker needed for unit tests)
- `test_runtime_contracts.py` requires Docker container (28 tests)
- `test_tutorial_workflows.py` requires Docker container (19 tests covering all DOLFINx tutorial chapters)
  - T1 (2): Fundamentals — Poisson, membrane
  - T2 (3): Time-dependent/nonlinear — heat, elasticity, nonlinear Poisson
  - T3 (5): Boundary conditions — mixed BCs, multiple Dirichlet, subdomains, Robin, component-wise
  - T4 (4): Advanced — convergence study, Helmholtz, mixed Poisson, singular Poisson
  - T5 (5): Full coverage — Nitsche, hyperelasticity, electromagnetics (N1curl), AMR, Stokes (Taylor-Hood)
- `test_edge_case_contracts.py` requires Docker container (26 tests)
- Hypothesis property tests use 500 examples per property for SessionState invariants
- Coverage gate: 90% on core modules (tools/, server.py, resources/, prompts/, ufl_context.py are excluded)
- Fixtures in `conftest.py`: `session` (empty), `mock_ctx` (empty + MCP context), `populated_session` (mesh+space+function+BC+solution), `mock_ctx_populated` (populated + context)

## CI Pipeline

Three jobs: `lint` (ruff), `typecheck` (pyright), `test` (pytest + coverage).
All run on Python 3.12, Ubuntu latest.

## Plugin Layer (Claude Code Integration)

The `.claude/` directory adds FEM domain intelligence on top of the 32 MCP tools.

### Skills (`.claude/skills/`)

| Skill | Triggers |
|-------|----------|
| `fem-workflow-poisson` | "solve Poisson", "Laplace equation", "heat diffusion" |
| `fem-workflow-elasticity` | "linear elasticity", "stress strain", "elastic deformation" |
| `fem-workflow-stokes` | "Stokes flow", "creeping flow", "incompressible flow" |
| `fem-solver-selection` | "which solver", "KSP options", "preconditioner" |
| `fem-element-selection` | "which element", "Taylor-Hood", "P1 vs P2" |
| `fem-debugging` | "solver diverged", "NaN values", "convergence failure" |
| `fem-workflow-complex-poisson` | "complex-valued", "complex Poisson", "sesquilinear" |
| `fem-workflow-nitsche` | "Nitsche method", "weak Dirichlet", "penalty BC" |
| `fem-workflow-membrane` | "membrane deflection", "curved domain", "Gaussian load" |
| `fem-workflow-heat-equation` | "heat equation", "diffusion", "time-dependent", "transient" |
| `fem-workflow-nonlinear-poisson` | "nonlinear Poisson", "Newton", "nonlinear PDE" |
| `fem-workflow-navier-stokes` | "Navier-Stokes", "IPCS", "channel flow" |
| `fem-workflow-hyperelasticity` | "hyperelasticity", "large deformation", "neo-Hookean" |
| `fem-workflow-helmholtz` | "Helmholtz", "acoustics", "frequency domain" |
| `fem-workflow-adaptive-refinement` | "adaptive refinement", "AMR", "error estimator" |
| `fem-workflow-singular-poisson` | "singular Poisson", "nullspace", "pure Neumann" |
| `fem-workflow-mixed-bcs` | "mixed BCs", "Neumann and Dirichlet", "combined BCs" |
| `fem-workflow-multiple-dirichlet` | "multiple Dirichlet", "different BC values" |
| `fem-workflow-material-subdomains` | "material subdomains", "piecewise coefficients" |
| `fem-workflow-robin-bc` | "Robin BC", "impedance BC", "Newton cooling" |
| `fem-workflow-component-bc` | "component-wise BC", "fix x-component", "sub_space" |
| `fem-workflow-electromagnetics` | "electromagnetics", "Nedelec", "H(curl)", "Maxwell" |
| `fem-workflow-mixed-poisson` | "mixed Poisson", "Raviart-Thomas", "flux variable" |
| `fem-workflow-solver-config` | "PETSc options", "solver configuration", "JIT options" |
| `fem-workflow-custom-newton` | "custom Newton", "Newton loop", "load stepping" |
| `fem-workflow-convergence-rates` | "convergence rate", "mesh refinement study", "h-refinement" |
| `fem-workflow-cahn-hilliard` | "Cahn-Hilliard", "phase field", "spinodal decomposition", "Allen-Cahn" |
| `fem-workflow-biharmonic` | "biharmonic", "plate bending", "fourth-order PDE", "nabla^4" |
| `fem-workflow-dg-formulation` | "DG", "discontinuous Galerkin", "interior penalty", "SIPG", "jump" |
| `fem-workflow-eigenvalue` | "eigenvalue", "eigenmodes", "modal analysis", "natural frequencies" |
| `fem-workflow-interpolation` | "interpolate function", "L2 projection", "transfer between spaces" |
| `fem-workflow-visualization` | "plot solution", "export VTK", "visualize results" |
| `fem-workflow-functionals` | "compute integral", "evaluate at point", "functional value" |
| `fem-workflow-submesh` | "submesh", "extract subdomain", "domain decomposition" |
| `fem-workflow-mesh-quality` | "mesh quality", "element quality", "mesh statistics" |
| `fem-workflow-discrete-ops` | "discrete gradient", "discrete curl", "operator matrix" |

### Agents (`.claude/agents/`)

| Agent | Purpose | Model |
|-------|---------|-------|
| `fem-solver` | Complete PDE solve pipeline | sonnet |
| `convergence-study` | Automated mesh refinement study | sonnet |
| `mesh-quality` | Mesh quality analysis | haiku |
| `nonlinear-solver` | Nonlinear PDE solve with Newton | sonnet |
| `time-dependent-solver` | Time-dependent PDE workflows | sonnet |
| `boundary-condition-setup` | Complex BC configuration | haiku |

### Hooks (`.claude/settings.json`)

- **SessionStart**: Injects FEM workflow context, element families, and nonlinear solver capability
- **PreToolUse**: Client-side expression safety check (mirrors `_check_forbidden()` from `ufl_context.py`); covers `residual` and `jacobian` parameters for `solve_nonlinear`

### Commands (`.claude/commands/`)

- `/solve-poisson [mesh_size] [degree]`: End-to-end Poisson solve with manufactured solution
- `/run-tests`: Test suite with coverage
- `/add-tool`: New MCP tool scaffold
- `/check-contracts`: DbC compliance audit
- `/tutorial-chapter [chapter]`: Walk through a DOLFINx tutorial chapter step-by-step
- `/verify-installation`: Check Docker, container, and MCP server status
