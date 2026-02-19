# Contributing to dolfinx-mcp

Thanks for your interest in contributing! This document covers the development workflow.

## Prerequisites

- Python 3.10+
- Docker Desktop (for integration tests)
- Git

## Setup

```bash
git clone https://github.com/ekstanley/ccFenics-plugin.git
cd ccFenics-plugin
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Unit tests (no Docker needed)
pytest tests/ --ignore=tests/test_runtime_contracts.py --ignore=tests/test_tutorial_workflows.py -v

# Lint
ruff check src/ tests/

# Type check
pyright src/dolfinx_mcp/

# Docker integration tests (requires Docker)
docker build -t dolfinx-mcp:latest .
./scripts/run-docker-tests.sh
```

### Code Style

- **Line length**: 100 characters
- **Linter**: ruff (config in `pyproject.toml`)
- **Type checker**: pyright in standard mode
- **Lazy imports**: All DOLFINx/UFL/NumPy imports go inside function bodies, not at module level
- **Logging**: Use `logger.info()`, never `print()` (stdout is reserved for JSON-RPC)

### Adding a New Tool

See the "New Tool Checklist" in `CLAUDE.md` for the complete pattern. Key points:

1. Place in the correct module under `src/dolfinx_mcp/tools/`
2. Use `@mcp.tool()` then `@handle_tool_errors` (order matters)
3. Validate inputs with `PreconditionError` before any lazy imports
4. End with `if __debug__: session.check_invariants()`
5. Add tests in `tests/`

### Design-by-Contract

This project uses Design-by-Contract throughout:

- **Preconditions**: Validate all inputs at function entry
- **Postconditions**: Verify results before returning
- **Invariants**: 9 referential integrity invariants checked by `session.check_invariants()`

See `CLAUDE.md` for the full error hierarchy and when to use each error class.

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure `ruff check` and `pyright` pass
4. Ensure unit tests pass
5. Open a PR against `main`

## Reporting Issues

Use [GitHub Issues](https://github.com/ekstanley/ccFenics-plugin/issues) with the provided templates.
