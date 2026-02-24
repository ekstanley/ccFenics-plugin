# DOLFINx Assistant

Structured FEM workflows for Claude Code. Guides simulation setup, enforces quality standards, diagnoses solver failures, and generates reports — all powered by the [DOLFINx MCP server](../README.md).

<table>
<tr>
<td align="center"><strong>22</strong><br>Commands</td>
<td align="center"><strong>20</strong><br>Skills</td>
<td align="center"><strong>6</strong><br>Agents</td>
<td align="center"><strong>4</strong><br>Safety Hooks</td>
</tr>
</table>

---

## Quick Start

### 1. Prerequisites

Build the Docker image (one time):

```bash
git clone https://github.com/ekstanley/ccFenics-plugin.git
cd ccFenics-plugin
docker build -t dolfinx-mcp .
```

### 2. Install the Plugin

<details>
<summary><strong>Claude Code</strong></summary>

```bash
claude --plugin-dir ./dolfinx-assistant
```

Or symlink into your global plugins directory:

```bash
ln -s "$(pwd)/dolfinx-assistant" ~/.claude/plugins/dolfinx-assistant
```

</details>

<details>
<summary><strong>Claude Cowork</strong></summary>

1. In Cowork, go to **Extensions** > **Install from folder**
2. Point to the `dolfinx-assistant/` directory in this repo
3. Choose a **Workspace Directory** for simulation output (VTK, XDMF files)

</details>

### 3. Configure the MCP Server

Add to your MCP client configuration. Set `/path/to/workspace` to the directory for simulation output:

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

> If using this repo directly, `.mcp.json` is already configured — just run `claude` from the repo root.

### 4. Start

```
/sim-setup       # Guided setup for a new simulation
/recipe poisson  # Jump straight to a pre-built PDE formulation
```

---

## Examples

### Guided PDE Solve

> **You:** `/solve-pde`
>
> **Assistant:** What PDE would you like to solve? (Poisson, elasticity, Stokes, heat equation, ...)
>
> **You:** Poisson on a unit square with f = sin(pi*x)*sin(pi*y)
>
> **Assistant:** *Creates mesh, function space, sets BCs, defines the form, solves, computes L2 error — step by step with explanations at each stage.*

### Quick Recipe Lookup

> **You:** `/recipe helmholtz`
>
> **Assistant:** *Returns the complete Helmholtz formulation: element choice (P2), variational form with wavenumber, Robin absorbing BCs, and recommended solver (direct/MUMPS for indefinite system).*

### Debugging a Failed Solver

> **You:** `/diagnose`
>
> **Assistant:** *Runs systematic checks: session completeness, mesh quality, BC coverage, form rank, solver-problem compatibility. Identifies root cause and suggests fixes.*

---

## Commands (22)

### Core Workflows

| Command | Description |
|---------|-------------|
| `/sim-setup` | Guided simulation configuration — mesh, elements, BCs, materials |
| `/solve-pde` | End-to-end PDE solve with interactive setup |
| `/recipe [pde-name]` | Quick PDE recipe lookup — 15 pre-built formulations |
| `/compose-form` | Interactive UFL form builder with operator guidance |

### Advanced Formulations

| Command | Description |
|---------|-------------|
| `/setup-dg [type]` | DG formulation with interior penalty and stabilization |
| `/setup-bc-advanced [type]` | Periodic, Nitsche, Robin, component-wise BC setup |
| `/setup-axisym [problem]` | Axisymmetric problem in cylindrical coordinates |
| `/setup-mms [pde]` | Manufactured solution verification workflow |
| `/couple-domains [type]` | Multi-domain / multi-physics coupling setup |

### Solver & Performance

| Command | Description |
|---------|-------------|
| `/newton-loop [config]` | Custom Newton solver with load stepping and monitoring |
| `/setup-matrix-free [type]` | Matrix-free solver and fieldsplit configuration |
| `/mpi-setup [nprocs]` | Parallel execution script generation |

### Verification & Diagnostics

| Command | Description |
|---------|-------------|
| `/check-mesh` | Mesh quality audit with pass/fail assessment |
| `/run-convergence` | Mesh convergence study with error rate analysis |
| `/diagnose` | Systematic diagnosis of a failed or suspicious simulation |
| `/parametric-sweep` | Parameter study across multiple values |
| `/explain-assembly` | Pedagogical assembly walkthrough for current session |

### Utilities

| Command | Description |
|---------|-------------|
| `/import-mesh` | Import a Gmsh .msh file with tag inspection |
| `/sim-report` | Generate a formatted HTML simulation report |
| `/compare-solutions` | Compare two solutions side by side |
| `/export-results` | Export to VTK/XDMF and generate plots |
| `/tutorial [chapter]` | Walk through a DOLFINx tutorial interactively |

---

## Skills (20)

Skills trigger automatically when your message matches their keywords. No slash command needed — just describe what you want.

<details>
<summary><strong>Formulation & Elements</strong> (7 skills)</summary>

| Skill | Triggers On |
|-------|------------|
| UFL Form Authoring | "UFL expression", "inner vs dot", "weak form", "form operators", "tensor algebra" |
| Element Selection | "which element", "P1 vs P2", "Taylor-Hood", element type questions |
| DG Formulations | "discontinuous Galerkin", "interior penalty", "jump", "average", "DG", "upwind" |
| Mixed Formulation | "mixed formulation", "saddle point", "inf-sup", "Stokes", "Taylor-Hood" |
| Advanced Boundary Conditions | "periodic BC", "Nitsche", "Robin BC", "component-wise BC", "weak Dirichlet" |
| Axisymmetric Formulations | "axisymmetric", "cylindrical coordinates", "r-z plane", "revolution" |
| PDE Cookbook | "recipe", "how to solve [PDE name]", "advection-diffusion", "wave equation", "Cahn-Hilliard" |

</details>

<details>
<summary><strong>Solvers & Performance</strong> (6 skills)</summary>

| Skill | Triggers On |
|-------|------------|
| Solver Selection | "which solver", "PETSc options", "preconditioner", solver configuration |
| Nonlinear Setup | "nonlinear", "Newton method", "load stepping", "SNES", "Jacobian" |
| Custom Newton Loops | "custom Newton", "Newton loop", "load stepping", "convergence monitoring" |
| Matrix-Free Solvers | "matrix-free", "shell matrix", "fieldsplit", "Schur complement", "nullspace" |
| JIT Performance Tuning | "FFCx", "JIT", "performance", "quadrature degree", "compilation" |
| Parallel MPI Awareness | "MPI", "parallel", "partitioning", "ghost mode", "distributed" |

</details>

<details>
<summary><strong>Applications & Verification</strong> (3 skills)</summary>

| Skill | Triggers On |
|-------|------------|
| Multi-Physics Coupling | "multi-physics", "FSI", "thermal-mechanical", "domain coupling", "submesh" |
| MMS Verification | "manufactured solution", "MMS", "convergence rate", "code verification" |
| Time-Dependent Setup | "time-dependent", "transient", "heat equation", "time stepping" |

</details>

<details>
<summary><strong>Mesh & Data</strong> (4 skills)</summary>

| Skill | Triggers On |
|-------|------------|
| Mesh Generation | "mesh generation", "Gmsh", "mesh refinement", "mesh import" |
| Results Validation | "validate results", "convergence verification", "results look wrong" |
| Debugging & Diagnostics | "solver diverged", "NaN values", "wrong results", "debug simulation" |
| Assembly Pedagogy | "how does assembly work", "what is lifting", "explain stiffness matrix" |

</details>

---

## Agents (6)

Specialized sub-agents that handle complex, multi-step tasks autonomously.

| Agent | Purpose | Model |
|-------|---------|-------|
| **simulation-designer** | Translates physical problem descriptions into DOLFINx simulation specs | sonnet |
| **formulation-architect** | UFL form language expert — guides PDE to weak form to code translation | sonnet |
| **solver-optimizer** | PETSc solver configuration, custom Newton, matrix-free techniques | sonnet |
| **parametric-study** | Designs and executes parameter sweeps with automated result collection | sonnet |
| **debugging-assistant** | Systematic diagnosis of solver failures, NaN values, wrong results | sonnet |
| **post-processor** | Automated post-processing: visualization, analysis, export, reports | sonnet |

---

## Safety Hooks (4)

Hooks run automatically before tool calls and at session start. They catch common mistakes before they reach the solver.

| Hook | When | What It Checks |
|------|------|----------------|
| **Pre-solve validation** | Before `solve`, `solve_nonlinear`, `solve_time_dependent`, `solve_eigenvalue` | Mesh exists, space defined, BCs present, form complete, DG penalty terms, solver-problem type match, expression safety |
| **Form definition validation** | Before `define_variational_form` | UFL form rank, operator usage, measure presence, DG interior facet terms, mixed sub-function access, expression safety |
| **Mesh import reminder** | After `create_custom_mesh` | Prompts for quality check (`/check-mesh`) and tag review |
| **Session start** | On session open | Welcome message with command suggestions |

---

## Architecture

This plugin is a **pure intelligence layer** — it contains no solver code. All computation happens through the 38 MCP tools provided by the DOLFINx MCP server running in Docker.

```
  Plugin Layer (this repo)              MCP Server (Docker)
  ┌──────────────────────┐            ┌─────────────────────┐
  │  22 Commands          │            │  38 Tool Handlers   │
  │  20 Skills            │──── MCP ──>│  SessionState       │
  │   6 Agents            │  (tools)   │  DOLFINx/FEniCSx    │
  │   4 Safety Hooks      │            │  /workspace output  │
  └──────────────────────┘            └─────────────────────┘
```

### Overlap with `.claude/` Layer

If using this plugin alongside the `.claude/skills/` layer in this repository, note that `element-selection`, `solver-selection`, and `/tutorial` exist in both. The plugin versions are self-contained alternatives optimized for guided workflows.

---

## Requirements

- Docker with the `dolfinx-mcp` image built
- Claude Code or Claude Cowork
- MCP server configured (see [Quick Start](#quick-start))

No environment variables required — the plugin communicates through MCP tool calls.
