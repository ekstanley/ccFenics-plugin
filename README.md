# dolfinx-mcp

[![CI](https://github.com/ekstanley/ccFenics-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/ekstanley/ccFenics-plugin/actions/workflows/ci.yml) [![CodeQL](https://github.com/ekstanley/ccFenics-plugin/actions/workflows/codeql.yml/badge.svg)](https://github.com/ekstanley/ccFenics-plugin/actions/workflows/codeql.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

**Talk to your PDE solver.** An [MCP](https://modelcontextprotocol.io/) server that gives AI assistants direct access to the [FEniCSx/DOLFINx](https://fenicsproject.org/) finite element framework. Describe a PDE in natural language and the tools handle the rest — mesh generation, function spaces, assembly, solving, and post-processing — all inside a sandboxed Docker container.

<table>
<tr>
<td align="center"><strong>35</strong><br>MCP Tools</td>
<td align="center"><strong>37</strong><br>FEM Skills</td>
<td align="center"><strong>478</strong><br>Tests</td>
<td align="center"><strong>9</strong><br>Invariants</td>
<td align="center"><strong>20</strong><br>Lean 4 Theorems</td>
</tr>
</table>

```
              MCP Client                              Docker Container
         (Claude, Cursor, ...)                      (dolfinx/dolfinx:stable)
        +---------------------+                    +-----------------------+
        |                     |    JSON-RPC         |  FastMCP Server       |
        |  "Solve the 3D      | ── stdio/http ───> |    35 Tool Handlers   |
        |   Poisson equation" |                     |    SessionState       |
        |                     | <─────────────────  |    /workspace output  |
        +---------------------+                    +-----------------------+
```

---

## Quick Start

### 1. Build

```bash
git clone https://github.com/ekstanley/ccFenics-plugin.git
cd ccFenics-plugin
docker build -t dolfinx-mcp .
```

### 2. Configure Your MCP Client

Add to your client configuration (Claude Desktop, Cursor, VS Code, etc.).
Set `/path/to/workspace` to the directory for simulation output (VTK, XDMF):

```json
{
  "mcpServers": {
    "dolfinx": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--network", "none",
        "-v", "/path/to/workspace:/workspace",
        "dolfinx-mcp"
      ]
    }
  }
}
```

<details>
<summary><strong>Claude Code</strong> — zero-config setup</summary>

This repo includes `.mcp.json` for automatic MCP server discovery:

1. `docker build -t dolfinx-mcp .`
2. `claude` (from the repo root)
3. Start asking FEM questions — the server auto-connects

For the full FEM-aware experience (37 skills, 6 agents, 6 commands):

```bash
claude --plugin-dir ./ccfenics
```

</details>

<details>
<summary><strong>Claude Cowork</strong> — Desktop Extension</summary>

1. Build the Docker image: `docker build -t dolfinx-mcp .`
2. In Cowork, go to **Extensions** > **Install from folder**
3. Point to the `ccfenics/` directory in this repo
4. Choose a **Workspace Directory** for simulation output

The 35 MCP tools become available in your Cowork session.

</details>

### 3. Use

Ask your AI assistant to solve a PDE:

> "Create a 16x16 unit square mesh and solve the Poisson equation
> with f = 2pi^2 sin(pi x) sin(pi y), then compute the L2 error."

---

## Example: 3D Poisson on Unit Cube

A Jupyter notebook demonstrating a full 3D workflow is at
[`examples/3d_poisson_demo.ipynb`](examples/3d_poisson_demo.ipynb).

**Problem**: $-\nabla^2 u = 3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)$ on $[0,1]^3$, $u=0$ on boundary.

| Metric | Result |
|--------|--------|
| Mesh | 16x16x16 tetrahedra (24,576 cells, 4,913 vertices) |
| Space | P2 Lagrange (35,937 DOFs) |
| Solver | CG + Hypre AMG |
| max(u_h) | 1.000056 (expected 1.0) |
| L2 error | 1.21e-05 |
| H1 error | 1.15e-03 |

**Orthogonal slices through the solution field at x=0.5, y=0.5, z=0.5:**

![3D Poisson orthogonal slices](examples/3d_poisson_slices.png)

**Iso-surfaces at u = 0.2, 0.5, 0.8 (nested concentric level sets):**

![3D Poisson iso-surfaces](examples/3d_poisson_isosurfaces.png)

---

## Supported Problems

| Category | PDE Types |
|----------|-----------|
| **Elliptic** | Poisson, Helmholtz, mixed Poisson (RT elements), singular Poisson (nullspace), complex-valued |
| **Parabolic** | Heat equation, Cahn-Hilliard, Allen-Cahn |
| **Flow** | Stokes (Taylor-Hood), Navier-Stokes (IPCS) |
| **Solid Mechanics** | Linear elasticity, hyperelasticity (neo-Hookean), membrane deflection |
| **Electromagnetics** | H(curl) formulations with Nedelec edge elements |
| **Special** | Biharmonic (C/IP-DG), discontinuous Galerkin, eigenvalue (SLEPc), axisymmetric |

**Boundary conditions**: Dirichlet, Neumann, Robin, Nitsche (weak), component-wise, multi-region.
**Elements**: Lagrange (P1-P5), DG, Nedelec, Raviart-Thomas, BDM, Crouzeix-Raviart, mixed.
**Solvers**: Direct (LU, MUMPS), iterative (CG, GMRES, BiCGSTAB) with preconditioners (ILU, Hypre AMG, Jacobi), Newton (SNES), time-stepping (Euler, Crank-Nicolson).

---

## Tools (35)

### Mesh Operations (9)

| Tool | Description |
|------|-------------|
| `create_unit_square` | Unit square mesh with triangular or quadrilateral elements |
| `create_mesh` | Rectangle or box meshes with custom dimensions |
| `create_custom_mesh` | Import mesh from Gmsh (.msh) file |
| `get_mesh_info` | Inspect mesh properties (cells, vertices, bounding box) |
| `compute_mesh_quality` | Mesh quality metrics (aspect ratio, volume) |
| `mark_boundaries` | Tag boundary facets using coordinate expressions |
| `manage_mesh_tags` | Create and manage cell/facet region markers |
| `refine_mesh` | Uniformly refine an existing mesh |
| `create_submesh` | Extract submesh from tagged regions |

### Function Spaces (2)

| Tool | Description |
|------|-------------|
| `create_function_space` | Lagrange, DG, Nedelec, RT, BDM, or CR elements (degree 1-5) |
| `create_mixed_space` | Mixed function space from component subspaces |

### Problem Definition (3)

| Tool | Description |
|------|-------------|
| `set_material_properties` | Define material coefficients and source terms |
| `define_variational_form` | Bilinear and linear forms via UFL expressions |
| `apply_boundary_condition` | Dirichlet boundary conditions (value + locator) |

### Solvers (5)

| Tool | Description |
|------|-------------|
| `solve` | Direct (LU, MUMPS) or iterative (CG, GMRES) with preconditioners |
| `solve_time_dependent` | Time stepping: backward/forward Euler, Crank-Nicolson |
| `solve_nonlinear` | Newton solver (SNES) for nonlinear PDEs |
| `solve_eigenvalue` | Generalized eigenvalue problems via SLEPc |
| `get_solver_diagnostics` | Convergence info, iterations, residual norms |

### Post-processing (6)

| Tool | Description |
|------|-------------|
| `compute_error` | L2 or H1 error against an exact solution |
| `evaluate_solution` | Evaluate solution at specified coordinates |
| `query_point_values` | Point queries with geometric cell info |
| `compute_functionals` | Integrate UFL expressions over the domain |
| `export_solution` | Export to XDMF or VTK format |
| `plot_solution` | Contour or warp plots (PNG via PyVista) |

### Interpolation (4)

| Tool | Description |
|------|-------------|
| `create_function` | Create a function in a space, optionally initialized with an expression |
| `interpolate` | Interpolate UFL expression into a function space |
| `project` | L2-project an expression onto a function space |
| `create_discrete_operator` | Build a discrete operator matrix |

### Session Management (6)

| Tool | Description |
|------|-------------|
| `get_session_state` | List all registered meshes, spaces, functions, solutions |
| `reset_session` | Clear all session state |
| `run_custom_code` | Execute Python code in the session namespace |
| `assemble` | Assemble UFL forms into scalars, vectors, or matrices |
| `remove_object` | Remove an object with cascade deletion of dependents |
| `read_workspace_file` | Read files from /workspace/ as base64 (images) or text (VTK, CSV) |

---

## Claude Code Plugin

The `ccfenics/` directory adds FEM domain intelligence on top of the 35 MCP tools.
Load it with `claude --plugin-dir ./ccfenics`.

### Agents (6)

| Agent | What it does |
|-------|-------------|
| `fem-solver` | End-to-end PDE solve pipeline (mesh to post-processing) |
| `convergence-study` | Automated mesh refinement with convergence rate fitting |
| `nonlinear-solver` | Newton method with load stepping and convergence diagnostics |
| `time-dependent-solver` | Transient PDE workflows (initial conditions, time stepping, output) |
| `boundary-condition-setup` | Complex BC configuration (Dirichlet, Neumann, Robin, Nitsche) |
| `mesh-quality` | Mesh quality analysis and refinement recommendations |

### Skills (37)

Guided workflows for every supported PDE type, plus solver selection, element selection,
and debugging. Triggered automatically when you describe a problem:

> "Solve a Stokes flow problem" &rarr; `fem-workflow-stokes`
> "My solver diverged" &rarr; `fem-debugging`

### Commands (6)

| Command | Description |
|---------|-------------|
| `/solve-poisson` | End-to-end Poisson solve with manufactured solution |
| `/tutorial-chapter` | Walk through a DOLFINx tutorial chapter step-by-step |
| `/run-tests` | Run the test suite with coverage |
| `/check-contracts` | Audit Design-by-Contract compliance |
| `/add-tool` | Scaffold a new MCP tool |
| `/verify-installation` | Check Docker, container, and server status |

---

## MCP Protocol

### Prompt Templates (6)

| Prompt | Description |
|--------|-------------|
| `setup_poisson` | Guided Poisson equation setup |
| `setup_elasticity` | Linear elasticity problem workflow |
| `setup_stokes` | Stokes flow with mixed elements |
| `setup_navier_stokes` | Navier-Stokes with time stepping |
| `debug_convergence` | Diagnose solver convergence failures |
| `convergence_study` | h-refinement convergence rate study |

### Resources (6)

| URI | Content |
|-----|---------|
| `dolfinx://capabilities` | Element families, solvers, export formats |
| `dolfinx://mesh/schema` | JSON schema for mesh objects |
| `dolfinx://space/schema` | JSON schema for function space objects |
| `dolfinx://solution/schema` | JSON schema for solution objects |
| `dolfinx://session/overview` | Current session state summary |
| `dolfinx://solver/options` | Available solver and preconditioner options |

### Transport Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **stdio** (default) | `docker run --rm -i dolfinx-mcp` | CLI clients, Claude Desktop |
| **streamable-http** | `docker run -p 8000:8000 dolfinx-mcp --transport streamable-http --host 0.0.0.0` | Web clients, JupyterLab |
| **sse** | `docker run -p 8000:8000 dolfinx-mcp --transport sse --host 0.0.0.0` | Legacy SSE clients |

---

## JupyterLab Integration

Start with Docker Compose:

```bash
docker-compose -f docker-compose.lab.yml up --build
# Open http://localhost:8888
```

Or install directly:

```bash
pip install dolfinx-mcp[jupyter]
```

Use IPython magics inside JupyterLab:

```python
%load_ext dolfinx_mcp_jupyter

%mcp_config url=http://localhost:8000/mcp

%mcp create_unit_square name=mesh nx=16 ny=16
%mcp create_function_space name=V family=Lagrange degree=1
%mcp solve solver_type=direct
%mcp compute_error exact="sin(pi*x[0])*sin(pi*x[1])" norm_type=L2
```

---

## Design-by-Contract

All 35 tools enforce runtime contracts:

- **Preconditions**: Input validation (parameter types, ranges, existence checks)
- **Postconditions**: Output validation (return structure, value constraints)
- **Invariants**: 9 referential integrity invariants on SessionState

Contract violations return structured error responses:

```json
{
  "error": "PRECONDITION_VIOLATED",
  "message": "Mesh 'box' not found in session",
  "suggestion": "Create a mesh first with create_unit_square or create_mesh"
}
```

Error codes: `NO_ACTIVE_MESH`, `MESH_NOT_FOUND`, `FUNCTION_SPACE_NOT_FOUND`,
`FUNCTION_NOT_FOUND`, `INVALID_UFL_EXPRESSION`, `SOLVER_ERROR`, `DUPLICATE_NAME`,
`DOLFINX_API_ERROR`, `FILE_IO_ERROR`, `PRECONDITION_VIOLATED`,
`POSTCONDITION_VIOLATED`, `INVARIANT_VIOLATED`.

---

## Formal Verification (Lean 4)

8 of the 9 referential integrity invariants are formally verified in Lean 4
with machine-checked proofs. INV-9 (`FormInfo.trial_space_name`) is specified
in Quint and enforced at runtime.

**20 theorems, 4 helper lemmas, zero `sorry` placeholders.**

| Theorem | What it proves |
|---------|---------------|
| `freshState_valid` | Empty session satisfies all 8 invariants |
| `registerMesh_valid` | Adding a mesh preserves invariants |
| `registerFunctionSpace_valid` | Adding a function space preserves invariants |
| `removeMesh_valid` | **Cascade deletion** of mesh + dependents preserves invariants |
| `removeSpaceDeps_valid` | Removing space dependents preserves invariants |
| `cleanup_valid` | Session reset produces a valid state |
| `validateOutputPath_safe` | Output path validation is a containment check |
| `errorCodeString_injective` | All 13 error codes are pairwise distinct |

Verify locally (requires [Lean 4](https://leanprover.github.io/lean4/doc/quickstart.html)):

```bash
cd .outline/proofs/DolfinxProofs
lake build    # exits 0, zero warnings
```

---

## Security

Three-layer defense for UFL expression evaluation:

1. **Token blocklist** — `_check_forbidden()` rejects `import`, `__`, `exec`, `eval`, `open`, `os.`, `subprocess`, and 12 more tokens at parse time
2. **Empty `__builtins__`** — expression namespaces have no access to Python internals
3. **Docker isolation** — container runs with `--network none`, non-root user, `--rm`

`run_custom_code` intentionally bypasses the blocklist (full `__builtins__`).
Docker isolation is the sole boundary for custom code execution.

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## Development

```bash
pip install -e ".[dev]"
```

```bash
# Unit tests (no Docker, 230 tests)
pytest tests/ \
  --ignore=tests/test_runtime_contracts.py \
  --ignore=tests/test_tutorial_workflows.py \
  --ignore=tests/test_edge_case_contracts.py \
  --ignore=tests/test_poisson_workflow.py

# Docker integration tests (248 tests)
docker build -t dolfinx-mcp .
./scripts/run-docker-tests.sh

# Lint & type check
ruff check src/ tests/
pyright src/dolfinx_mcp/
```

### Project Structure

```
src/dolfinx_mcp/
    server.py              Entry point
    _app.py                FastMCP instance + lifespan
    session.py             SessionState (8 registries, 9 invariants)
    errors.py              13 error classes + decorator
    ufl_context.py         Restricted UFL expression evaluation
    tools/
        mesh.py            9 mesh tools
        spaces.py          2 function space tools
        problem.py         3 problem definition tools
        solver.py          5 solver tools
        postprocess.py     6 post-processing tools
        interpolation.py   4 interpolation tools
        session_mgmt.py    6 session management tools
    prompts/templates.py   6 workflow prompts
    resources/providers.py 6 URI resources

src/dolfinx_mcp_jupyter/   JupyterLab extension (5 IPython magics)

tests/                     17 test files, 478 tests (230 unit + 248 Docker)
examples/                  3D Poisson notebook + production readiness suite
.outline/proofs/           Lean 4 formal verification (20 theorems)
ccfenics/                  Claude Code plugin (37 skills, 6 agents, 6 commands)
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and the new tool checklist.

## License

MIT
