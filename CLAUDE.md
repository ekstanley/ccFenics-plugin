# DOLFINx MCP Server

MCP server for FEniCSx/DOLFINx finite element computing. Version 0.6.1.

## Quick Reference

```bash
# Install (dev mode)
pip install -e ".[dev]"

# Tests (no Docker needed, ~238 tests)
pytest tests/ --ignore=tests/test_runtime_contracts.py -v

# Tests with coverage (threshold set in pyproject.toml [tool.coverage.report].fail_under = 90)
pytest tests/ --ignore=tests/test_runtime_contracts.py --cov=dolfinx_mcp --cov-report=term-missing -v

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
        solver.py        3 solver tools
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
- `test_runtime_contracts.py` requires Docker container
- Hypothesis property tests use 500 examples per property for SessionState invariants
- Coverage gate: 90% on core modules (tools/, server.py, resources/, prompts/ are excluded)
- Fixtures in `conftest.py`: `session` (empty), `mock_ctx` (empty + MCP context), `populated_session` (mesh+space+function+BC+solution), `mock_ctx_populated` (populated + context)

## CI Pipeline

Three jobs: `lint` (ruff), `typecheck` (pyright), `test` (pytest + coverage).
All run on Python 3.12, Ubuntu latest.
