Audit Design-by-Contract (DbC) compliance across all tool functions in `src/dolfinx_mcp/tools/*.py`.

For each tool function (decorated with `@mcp.tool()` + `@handle_tool_errors`), verify:

1. **Decorator order**: `@mcp.tool()` THEN `@handle_tool_errors` (reversed breaks FastMCP schema generation)
2. **Preconditions before imports**: Input validation using `PreconditionError` appears BEFORE lazy `import dolfinx` statements
3. **Error class correctness**: Input validation uses `PreconditionError`, NOT `DOLFINxAPIError`. `DOLFINxAPIError` is only for wrapping caught DOLFINx API exceptions
4. **Business logic wrapping**: DOLFINx API calls are in try/except that re-raises `DOLFINxMCPError` and wraps generic `Exception` in `DOLFINxAPIError`
5. **Postconditions**: Results are validated (finite values, non-empty, positive where expected) using `PostconditionError`
6. **Invariant check**: `if __debug__: session.check_invariants()` appears at the end of the function
7. **Security**: All expression evaluation sites call `_check_forbidden()` EAGERLY (at creation time, not deferred inside lambdas)

Report violations as a table: | File | Tool | Violation | Suggested Fix |
