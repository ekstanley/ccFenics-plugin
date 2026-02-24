# DOLFINx Assistant

FEA simulation workflow plugin for DOLFINx. Guides researchers and engineers through simulation setup, enforces quality standards, processes results, and generates reports.

## What It Does

This plugin adds structured FEM workflows on top of the DOLFINx MCP server. It handles the full simulation lifecycle: problem definition, mesh quality assurance, element and solver selection, result validation, and reporting.

## Commands

| Command | Description |
|---------|-------------|
| `/sim-setup` | Guided simulation configuration — mesh, elements, BCs, materials |
| `/solve-pde` | End-to-end PDE solve with interactive setup |
| `/check-mesh` | Mesh quality audit with pass/fail assessment |
| `/run-convergence` | Mesh convergence study with error rate analysis |
| `/sim-report` | Generate a formatted HTML simulation report |
| `/compare-solutions` | Compare two solutions side by side |
| `/export-results` | Export to VTK/XDMF and generate plots |
| `/tutorial [chapter]` | Walk through a DOLFINx tutorial interactively |

## Skills

| Skill | Triggers On |
|-------|------------|
| Element Selection | "which element", "P1 vs P2", "Taylor-Hood", element type questions |
| Solver Selection | "which solver", "PETSc options", "preconditioner", solver configuration |
| Results Validation | "validate results", "manufactured solution", "convergence verification" |

## Agent

**simulation-designer** — Translates physical problem descriptions into DOLFINx simulation specifications through structured conversation. Activated when users describe a physical problem without FEM specifics.

## Hooks

- **Pre-solve validation**: Checks mesh, function space, BCs, and form completeness before any solve executes. Blocks solves with dangerous UFL expressions.
- **Session start**: Brief welcome with available commands.

## Requirements

- DOLFINx MCP server running in Docker
- Docker container with FEniCSx installed

## Setup

1. Install the plugin in Claude Cowork
2. Start the DOLFINx MCP server (`docker run ...`)
3. Use `/sim-setup` or `/solve-pde` to begin

No environment variables required — the plugin communicates through the DOLFINx MCP server tools.
