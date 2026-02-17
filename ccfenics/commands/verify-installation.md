Verify the DOLFINx MCP server installation by running a minimal test.

Steps:

1. Check if Docker is available:
   ```
   docker --version
   ```

2. Check if the DOLFINx container exists:
   ```
   docker images dolfinx-mcp --format "{{.Repository}}:{{.Tag}} ({{.Size}})"
   ```

3. If the container doesn't exist, offer to build it:
   ```
   docker build -t dolfinx-mcp .
   ```

4. Run a minimal Poisson solve to verify the full pipeline works:
   ```
   create_unit_square(nx=4, ny=4, name="test_mesh")
   create_function_space(family="Lagrange", degree=1, name="V")
   set_material_properties(name="f", value="1.0")
   define_variational_form(bilinear="inner(grad(u), grad(v)) * dx", linear="f * v * dx", name="test")
   apply_boundary_condition(value="0.0", boundary="True", name="bc")
   solve(solver_type="direct", solution_name="test_u")
   ```

5. Report results:
   - Docker status: available / not found
   - Container: built / not built
   - MCP server: responding / not responding
   - Minimal solve: converged / failed
   - Solution norm: (should be non-zero)

6. If anything fails, provide specific troubleshooting steps:
   - Docker not found: "Install Docker Desktop from docker.com"
   - Container not built: "Run: docker build -t dolfinx-mcp ."
   - Server not responding: "Check MCP server configuration in your client"
   - Solve failed: "Check server logs for DOLFINx import errors"
