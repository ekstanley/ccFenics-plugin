# Solver Details Reference

## PETSc KSP Types

| KSP | Full Name | Symmetry | Memory | Best For |
|-----|-----------|----------|--------|----------|
| cg | Conjugate Gradient | SPD only | Low | Poisson, elasticity, diffusion |
| gmres | Generalized Minimal Residual | Any | Medium (restart) | Non-symmetric, general purpose |
| bcgs | BiCGStab | Any | Low | Alternative to GMRES, less memory |
| minres | Minimal Residual | Symmetric | Low | Saddle point (Stokes) |
| preonly | Preconditioner only | Any | Depends | Direct solvers (wraps LU/Cholesky) |
| richardson | Richardson iteration | Any | Low | Simple, needs good preconditioner |

## PETSc Preconditioner Types

| PC | Full Name | When to Use | Scalability |
|----|-----------|-------------|-------------|
| lu | LU factorization | Small problems, debugging | O(n^1.5-2) memory |
| ilu | Incomplete LU | General iterative fallback | Single process |
| jacobi | Diagonal scaling | Well-conditioned, large systems | Excellent |
| sor | Successive over-relaxation | Smoothing, simple problems | Limited |
| hypre | Hypre BoomerAMG | SPD systems, scalable | Excellent |
| gamg | PETSc native AMG | Alternative to Hypre | Good |
| fieldsplit | Block decomposition | Mixed/saddle point | Problem-dependent |

## AMG (Algebraic Multigrid) Tuning

Default Hypre BoomerAMG works well for most elliptic problems. Tune these if convergence is slow:

```python
petsc_options = {
    "pc_hypre_boomeramg_strong_threshold": "0.7",  # 3D: 0.5, 2D: 0.25
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_max_levels": "25",
}
```

## Block Preconditioners for Mixed Problems

For Stokes (velocity-pressure saddle point):

### Schur Complement Approach

```python
petsc_options = {
    "ksp_type": "minres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "diag",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "jacobi",
}
```

### Upper Triangular

```python
petsc_options = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
}
```

## Nonlinear Solver Details

### Newton Line Search Variants

| Line Search | PETSc Name | When |
|-------------|-----------|------|
| Basic (full step) | `basic` | Well-conditioned, close to solution |
| Backtracking | `bt` | Default, robust |
| Critical point | `cp` | Near bifurcation points |
| L2 norm | `l2` | Minimization-like problems |

### Load Stepping Strategy

For problems where Newton diverges with full load:

1. Start with 10% of the load
2. Solve
3. Use solution as initial guess for next increment
4. Increase load by 10-20%
5. Repeat until full load

Typical: 5-10 load steps for moderate nonlinearity, 20-50 for large deformation.

## Convergence Diagnostics

### KSP Convergence Reasons (positive = converged)

| Code | Meaning |
|------|---------|
| 1 | rtol satisfied |
| 2 | atol satisfied |
| 3 | its (reached max, but converged) |
| 9 | atol_normal |

### KSP Divergence Reasons (negative = failed)

| Code | Meaning | Fix |
|------|---------|-----|
| -2 | Null | Check matrix assembly |
| -3 | Its (max iterations, not converged) | Increase max_iter or improve preconditioner |
| -4 | Dtol (divergence tolerance) | Problem may be ill-posed |
| -5 | Breakdown | Preconditioner failure |
| -7 | Nonsymmetric | Using CG on non-symmetric matrix |

### SNES Convergence Reasons

| Code | Meaning |
|------|---------|
| 2 | FNORM_ABS (absolute residual) |
| 3 | FNORM_RELATIVE (relative residual) |
| -1 | Function domain error |
| -2 | LS failure (line search failed) |
| -6 | Max iterations |
