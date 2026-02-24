# DOLFINx Assistant

FEA simulation workflow plugin for DOLFINx. Guides researchers and engineers through simulation setup, enforces quality standards, processes results, and generates reports.

## What It Does

This plugin adds structured FEM workflows on top of the DOLFINx MCP server. It covers the full simulation lifecycle: problem formulation, mesh management, element and solver selection, advanced formulations (DG, mixed, axisymmetric), result validation, debugging, parametric studies, multi-physics coupling, and reporting.

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

## Skills (20)

### Formulation & Elements

| Skill | Triggers On |
|-------|------------|
| UFL Form Authoring | "UFL expression", "inner vs dot", "weak form", "form operators", "tensor algebra" |
| Element Selection | "which element", "P1 vs P2", "Taylor-Hood", element type questions |
| DG Formulations | "discontinuous Galerkin", "interior penalty", "jump", "average", "DG", "upwind" |
| Mixed Formulation | "mixed formulation", "saddle point", "inf-sup", "Stokes", "Taylor-Hood" |
| Advanced Boundary Conditions | "periodic BC", "Nitsche", "Robin BC", "component-wise BC", "weak Dirichlet" |
| Axisymmetric Formulations | "axisymmetric", "cylindrical coordinates", "r-z plane", "revolution" |
| PDE Cookbook | "recipe", "how to solve [PDE name]", "advection-diffusion", "wave equation", "Cahn-Hilliard" |

### Solvers & Performance

| Skill | Triggers On |
|-------|------------|
| Solver Selection | "which solver", "PETSc options", "preconditioner", solver configuration |
| Nonlinear Setup | "nonlinear", "Newton method", "load stepping", "SNES", "Jacobian" |
| Custom Newton Loops | "custom Newton", "Newton loop", "load stepping", "convergence monitoring" |
| Matrix-Free Solvers | "matrix-free", "shell matrix", "fieldsplit", "Schur complement", "nullspace" |
| JIT Performance Tuning | "FFCx", "JIT", "performance", "quadrature degree", "compilation" |
| Parallel MPI Awareness | "MPI", "parallel", "partitioning", "ghost mode", "distributed" |

### Applications & Verification

| Skill | Triggers On |
|-------|------------|
| Multi-Physics Coupling | "multi-physics", "FSI", "thermal-mechanical", "domain coupling", "submesh" |
| MMS Verification | "manufactured solution", "MMS", "convergence rate", "code verification" |
| Time-Dependent Setup | "time-dependent", "transient", "heat equation", "time stepping" |

### Mesh & Data

| Skill | Triggers On |
|-------|------------|
| Mesh Generation | "mesh generation", "Gmsh", "mesh refinement", "mesh import" |
| Results Validation | "validate results", "convergence verification", "results look wrong" |
| Debugging & Diagnostics | "solver diverged", "NaN values", "wrong results", "debug simulation" |
| Assembly Pedagogy | "how does assembly work", "what is lifting", "explain stiffness matrix" |

## Agents (6)

| Agent | Purpose | Model |
|-------|---------|-------|
| **simulation-designer** | Translates physical problem descriptions into DOLFINx simulation specs | sonnet |
| **formulation-architect** | UFL form language expert — guides PDE → weak form → code translation | sonnet |
| **solver-optimizer** | PETSc solver configuration, custom Newton, matrix-free techniques | sonnet |
| **parametric-study** | Designs and executes parameter sweeps with automated result collection | sonnet |
| **debugging-assistant** | Systematic diagnosis of solver failures, NaN values, wrong results | sonnet |
| **post-processor** | Automated post-processing: visualization, analysis, export, reports | sonnet |

## Hooks

- **Pre-solve validation**: Checks mesh, function space, BCs, form completeness, DG penalty terms, solver-problem type compatibility. Blocks dangerous UFL expressions. Covers linear, nonlinear, time-dependent, and eigenvalue solves.
- **Form definition validation**: Checks UFL form rank, operator usage, measure presence, DG interior facet terms, mixed sub-function access, and expression safety.
- **Mesh import reminder**: After importing a Gmsh mesh, prompts for quality check and tag review.
- **Session start**: Welcome with categorized command overview.

## Requirements

- DOLFINx MCP server running in Docker
- Docker container with FEniCSx installed

## Setup

1. Install the plugin (place `dolfinx-assistant/` in your plugins directory or install via Claude Code)
2. Start the DOLFINx MCP server (`docker run ...`)
3. Use `/sim-setup` or `/recipe` to begin

No environment variables required — the plugin communicates through the DOLFINx MCP server tools.

## Relationship to .claude/ Layer

This plugin is designed for **standalone distribution**. If using alongside the existing `.claude/skills/` layer in this repository, note that `element-selection`, `solver-selection`, and `/tutorial` exist in both. The plugin versions are self-contained alternatives optimized for guided workflows.
