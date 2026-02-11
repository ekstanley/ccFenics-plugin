Add a new MCP tool to the DOLFINx server.

Arguments: $ARGUMENTS (format: "tool_name module_name" e.g. "compute_gradient postprocess")

Follow the New Tool Checklist in CLAUDE.md exactly:

1. **Choose the correct module** in `src/dolfinx_mcp/tools/` based on the tool's purpose
2. **Add decorator pair**: `@mcp.tool()` then `@handle_tool_errors` (order matters)
3. **Signature**: `async def tool_name(..., ctx: Context = None) -> dict[str, Any]:`
4. **Docstring**: First line = MCP tool description. Include Args and Returns sections
5. **Preconditions first**: Validate all inputs BEFORE lazy imports. Use `PreconditionError`
6. **Lazy imports**: `import dolfinx...` / `import numpy as np` inside the function body
7. **Session access**: `session = _get_session(ctx)`
8. **Business logic**: Wrap DOLFINx API calls in try/except. Re-raise `DOLFINxMCPError`, wrap others in `DOLFINxAPIError`
9. **Postconditions**: Check results (finite values, non-empty, positive). Use `PostconditionError`
10. **Invariant check**: `if __debug__: session.check_invariants()` at end
11. **Logging**: `logger.info(...)` with operation summary
12. **Return dict**: All tools return `dict[str, Any]` with documented keys
13. **Import in server.py**: Ensure the tool module is imported in `server.py` for registration
14. **Tests**: Add tests in `tests/` using the `mock_ctx` fixture from conftest.py

After creating the tool, run `/check-contracts` to verify compliance.
