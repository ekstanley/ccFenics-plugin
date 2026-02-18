Run the DOLFINx MCP test suite with coverage analysis.

Execute:
```bash
pytest tests/ --ignore=tests/test_runtime_contracts.py --cov=dolfinx_mcp --cov-report=term-missing -v
```

After tests complete:
1. If any tests FAIL: analyze the failure output and suggest specific fixes
2. If coverage is below 90% on core modules: identify the uncovered lines in session.py, errors.py, ufl_context.py, _app.py, logging_config.py
3. Report summary: total tests, passed, failed, coverage percentage

Note: The 90% coverage threshold is configured in `pyproject.toml` under `[tool.coverage.report].fail_under`. Tool modules (`tools/*`), `server.py`, `resources/*`, `prompts/*`, and `utils.py` are excluded from coverage measurement.

For Docker integration tests (requires Docker image built):
```bash
docker build -t dolfinx-mcp:latest .
./scripts/run-docker-tests.sh
```

This runs all 5 Docker suites (cowork, runtime, tutorial, edge, invariant) in a single container. Run specific suites with: `./scripts/run-docker-tests.sh runtime tutorial`
