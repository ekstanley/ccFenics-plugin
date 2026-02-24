---
description: End-to-end PDE solve with guided setup
allowed-tools: Read, Write, Grep, Glob
model: sonnet
---

Walk the user through a complete PDE solve from scratch. This combines setup, solving, and basic validation in one flow.

## Phase 1: Problem Type

Ask the user what they want to solve. Map to a known PDE type:

| User Says | PDE Type | Key Tools |
|-----------|----------|-----------|
| Poisson, diffusion, heat (steady) | Scalar elliptic | Standard Lagrange |
| Elasticity, displacement | Vector elliptic | Vector Lagrange |
| Stokes, creeping flow | Saddle point | Taylor-Hood mixed |
| Heat equation, transient | Parabolic | Lagrange + time stepping |
| Nonlinear Poisson | Nonlinear elliptic | Newton solver |
| Helmholtz, acoustics | Complex elliptic | Lagrange (possibly complex) |

## Phase 2: Domain and Mesh

Ask about geometry. Create mesh with appropriate tool. Run `compute_mesh_quality` and report.

## Phase 3: Function Space

Reference element-selection skill. Create the appropriate function space.

## Phase 4: Coefficients and BCs

Ask about:
- Source terms, material coefficients
- Boundary conditions (type + value for each boundary)

Set up materials with `set_material_properties`, boundaries with `mark_boundaries` and `apply_boundary_condition`.

## Phase 5: Variational Form

Define the bilinear and linear forms with `define_variational_form`. Show the user what was defined.

## Phase 6: Solve

Reference solver-selection skill to pick the right solver. Call `solve` (or `solve_nonlinear` / `solve_time_dependent` as needed).

Report: convergence status, iterations, residual norm, wall time.

## Phase 7: Quick Validation

1. Plot the solution with `plot_solution`
2. Evaluate at a few characteristic points with `evaluate_solution`
3. If the user provided an exact solution, compute errors with `compute_error`
4. Check solution bounds are physically reasonable

Present a summary of the solve and suggest next steps (convergence study, parameter sweep, report generation).
