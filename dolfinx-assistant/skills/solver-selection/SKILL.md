---
name: solver-selection
description: >
  This skill should be used when the user asks "which solver should I use",
  "solver configuration", "PETSc options", "preconditioner", "KSP type",
  "iterative vs direct", "solver diverges", "solver performance",
  or needs help configuring linear/nonlinear solvers in DOLFINx.
version: 0.1.0
---

# Solver Selection Guide

Pick the right solver based on problem size, type, and conditioning. A bad solver choice turns a 10-second solve into a 10-minute failure.

## Quick Decision Tree

```
Problem size < 50,000 DOFs?
  YES → Direct solver (LU)
  NO → How big?
    < 500,000 DOFs → Iterative with good preconditioner
    > 500,000 DOFs → Iterative, consider multigrid
```

## Linear Solvers

### Direct Solvers

| Solver | DOLFINx Config | Best For |
|--------|---------------|----------|
| MUMPS (LU) | `solver_type="direct"` | Small-medium problems, robustness over speed |
| SuperLU | `pc_type="lu"`, `petsc_options={"pc_factor_mat_solver_type": "superlu"}` | Alternative to MUMPS |

**When to use direct**: Problems under ~50K DOFs, debugging (eliminates solver as error source), one-off solves, ill-conditioned systems.

**Limitation**: Memory scales as O(n^1.5) to O(n^2). A 1M DOF 3D problem may need 50+ GB RAM.

### Iterative Solvers — Symmetric Positive Definite (SPD)

Most scalar elliptic problems (Poisson, diffusion, elasticity) produce SPD systems.

| KSP | Preconditioner | Config | When |
|-----|---------------|--------|------|
| CG | Hypre AMG | `ksp_type="cg"`, `pc_type="hypre"` | Default for SPD. Fast, scalable |
| CG | ILU | `ksp_type="cg"`, `pc_type="ilu"` | Fallback if Hypre unavailable |
| CG | Jacobi | `ksp_type="cg"`, `pc_type="jacobi"` | Very large, well-conditioned systems |

**Default recommendation**: CG + Hypre AMG. Works for Poisson, elasticity, diffusion. Scales to millions of DOFs.

### Iterative Solvers — Non-Symmetric

Convection-dominated problems, Navier-Stokes, and non-self-adjoint operators produce non-symmetric systems.

| KSP | Preconditioner | Config | When |
|-----|---------------|--------|------|
| GMRES | ILU | `ksp_type="gmres"`, `pc_type="ilu"` | General non-symmetric |
| GMRES | Hypre AMG | `ksp_type="gmres"`, `pc_type="hypre"` | Large non-symmetric |
| BiCGStab | ILU | `ksp_type="bcgs"`, `pc_type="ilu"` | Alternative to GMRES, less memory |

### Iterative Solvers — Saddle Point (Mixed/Stokes)

| Strategy | Config | When |
|----------|--------|------|
| Direct (MUMPS) | `solver_type="direct"` | Under 100K DOFs, always works |
| Fieldsplit | Custom PETSc options | Large mixed problems |
| MINRES + block | `ksp_type="minres"` | Symmetric saddle point |

## Nonlinear Solvers

### Newton's Method

| SNES Type | Config | When |
|-----------|--------|------|
| Newton line search | `snes_type="newtonls"` | Default for most nonlinear problems |
| Newton trust region | `snes_type="newtontr"` | More robust for ill-conditioned problems |

**Key parameters**:
- `rtol=1e-10`: Relative tolerance. Tighten for high accuracy.
- `atol=1e-12`: Absolute tolerance. Matters when solution is near zero.
- `max_iter=50`: Increase for hard convergence. If you need >20 iterations, rethink the problem.

**Convergence tricks**:
- Start from a good initial guess (interpolate an approximate solution)
- Use load stepping for large deformations
- Check the Jacobian is correct (auto-differentiation vs hand-coded)

## Tolerance Guidelines

| Application | rtol | atol |
|-------------|------|------|
| Engineering design | 1e-6 | 1e-8 |
| Research / validation | 1e-10 | 1e-12 |
| Quick exploration | 1e-4 | 1e-6 |

## Convergence Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| KSP diverges immediately | Wrong solver for problem type | Check symmetry, try direct solver |
| KSP stalls after few iterations | Bad preconditioner | Switch to AMG or direct |
| SNES diverges | Bad initial guess or wrong Jacobian | Better initial guess, check Jacobian |
| Very slow convergence | Mesh too fine for preconditioner | Use multigrid, increase levels |
| NaN in residual | Division by zero, bad BCs | Check BCs, material parameters |

## PETSc Options Reference

Common options passed via `petsc_options` dict:

```python
# Monitor convergence
{"ksp_monitor": None, "ksp_converged_reason": None}

# GMRES restart
{"ksp_gmres_restart": "100"}

# AMG tuning
{"pc_hypre_boomeramg_strong_threshold": "0.7"}

# Verbose SNES
{"snes_monitor": None, "snes_converged_reason": None}
```

## Reference Material

For advanced solver configurations, block preconditioners, and fieldsplit setups, see `references/solver-details.md`.
