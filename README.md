# dolfinx-mcp

MCP server for FEniCSx/DOLFINx finite element computing.

**Version**: 0.5.0 | **License**: MIT | **Python**: >= 3.10

Exposes 31 tools for mesh generation, function space creation, PDE solving,
and post-processing through the [Model Context Protocol](https://modelcontextprotocol.io/).
Runs inside a Docker container with the full DOLFINx stack.

## Quick Start

### 1. Build the Docker image

```bash
git clone https://github.com/estanley/ccFenics-plugin.git
cd ccFenics-plugin
docker build -t dolfinx-mcp .
```

### 2. Configure your MCP client

Add to your `.mcp.json` (Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "dolfinx": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--network", "none",
        "-v", "/tmp/dolfinx-workspace:/workspace",
        "dolfinx-mcp"
      ]
    }
  }
}
```

### 3. Use it

Ask your AI assistant to solve a Poisson equation, generate a mesh, or compute
error norms -- the MCP tools handle the DOLFINx operations inside the container.

## Tools (31)

### Mesh Operations (9)

| Tool | Description |
|------|-------------|
| `create_unit_square` | Create a unit square mesh with triangular elements |
| `create_mesh` | Create rectangle or box meshes with custom dimensions |
| `create_custom_mesh` | Import a mesh from Gmsh (.msh) file |
| `get_mesh_info` | Inspect mesh properties (cells, vertices, bounding box) |
| `compute_mesh_quality` | Compute mesh quality metrics (aspect ratio, volume) |
| `mark_boundaries` | Tag boundary facets using coordinate expressions |
| `manage_mesh_tags` | Create and manage cell/facet region markers |
| `refine_mesh` | Uniformly refine an existing mesh |
| `create_submesh` | Extract a submesh from tagged regions |

### Function Spaces (2)

| Tool | Description |
|------|-------------|
| `create_function_space` | Create a Lagrange, DG, or other finite element space |
| `create_mixed_space` | Create a mixed function space from components |

### Problem Definition (3)

| Tool | Description |
|------|-------------|
| `set_material_properties` | Define material coefficients and source terms |
| `define_variational_form` | Set up bilinear and linear forms (UFL expressions) |
| `apply_boundary_condition` | Apply Dirichlet boundary conditions |

### Solvers (3)

| Tool | Description |
|------|-------------|
| `solve` | Solve the assembled linear system (direct or iterative) |
| `solve_time_dependent` | Time-stepping solver (forward/backward Euler, Crank-Nicolson) |
| `get_solver_diagnostics` | Retrieve convergence info, iterations, residuals |

### Post-processing (6)

| Tool | Description |
|------|-------------|
| `compute_error` | Compute L2 or H1 error against an exact solution |
| `evaluate_solution` | Evaluate solution at specified points |
| `query_point_values` | Query values with geometric cell information |
| `compute_functionals` | Integrate UFL expressions over the domain |
| `export_solution` | Export to XDMF or VTK format |
| `plot_solution` | Generate contour or warp plots (PNG) |

### Interpolation (3)

| Tool | Description |
|------|-------------|
| `interpolate` | Interpolate a UFL expression into a function space |
| `project` | L2-project an expression onto a function space |
| `create_discrete_operator` | Build a discrete operator matrix from a form |

### Session Management (5)

| Tool | Description |
|------|-------------|
| `get_session_state` | List all registered meshes, spaces, functions, solutions |
| `reset_session` | Clear all session state |
| `run_custom_code` | Execute arbitrary Python code in the session namespace |
| `assemble` | Assemble UFL forms into scalars, vectors, or matrices |
| `remove_object` | Remove an object with cascade deletion of dependents |

## Architecture

```
Host Machine                         Docker Container (dolfinx-mcp)
+----------------------------------+ +-------------------------------+
| MCP Client (Claude, Cursor, ...) | | dolfinx/dolfinx:stable        |
|                                  | |                               |
|  stdio / streamable-http / sse   | |  FastMCP Server               |
|    |                             | |    |                           |
|    +-- JSON-RPC over stdin/out --+-+->  +-- 31 Tool Handlers        |
|                                  | |    +-- SessionState            |
|                                  | |    |   (meshes, spaces,        |
|                                  | |    |    forms, solutions)      |
|                                  | |    +-- /workspace/ output      |
+----------------------------------+ +-------------------------------+
```

All DOLFINx computation runs inside the container. The MCP protocol handles
serialization. Session state (meshes, function spaces, solutions) persists
across tool calls within a single container session.

## Transport Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **stdio** (default) | `docker run --rm -i dolfinx-mcp` | CLI clients, Claude Desktop |
| **streamable-http** | `docker run -p 8000:8000 dolfinx-mcp --transport streamable-http --host 0.0.0.0` | Web clients, JupyterLab |
| **sse** | `docker run -p 8000:8000 dolfinx-mcp --transport sse --host 0.0.0.0` | Legacy SSE clients |

## JupyterLab Integration

```python
%load_ext dolfinx_mcp_jupyter

# Configure connection (streamable-http transport)
%mcp_config url=http://localhost:8000/mcp

# Call tools with IPython magics
%mcp create_unit_square name=mesh nx=16 ny=16
%mcp solve solver_type=direct
%mcp compute_error exact="sin(pi*x[0])*sin(pi*x[1])" norm_type=L2
```

Install with: `pip install dolfinx-mcp[jupyter]`

## Development

### Prerequisites

- Python >= 3.10
- Docker
- `pip install -e ".[dev]"`

### Commands

```bash
# Run unit tests (no Docker required)
pytest tests/ --ignore=tests/test_runtime_contracts.py -v

# Run Docker integration tests
docker build -t dolfinx-mcp .
pytest tests/test_runtime_contracts.py -v

# Run production readiness suite (all 31 tools via MCP protocol)
python examples/production_readiness.py --verbose

# Lint
ruff check src/ tests/ examples/
```

### Design-by-Contract

All 31 tools enforce preconditions (input validation), postconditions (output
validation), and session state invariants (referential integrity). Contract
violations return structured error responses with error codes:

- `PRECONDITION_VIOLATED` -- invalid input parameters
- `POSTCONDITION_VIOLATED` -- operation produced invalid results
- `INVARIANT_VIOLATED` -- session state inconsistency

## License

MIT
