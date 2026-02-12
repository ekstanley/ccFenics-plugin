---
name: fem-solver-selection
description: |
  Helps select the right solver configuration for DOLFINx problems.
  Use when the user asks which solver to use, about KSP options, preconditioner selection,
  solver settings, iterative vs direct solvers, or PETSc configuration.
---

# Solver Selection Guide

## Decision Tree

### 1. Problem Size

| Problem size (DOFs) | Recommendation |
|---|---|
| < 50,000 | Direct solver (LU) -- always reliable |
| 50,000 - 500,000 | Iterative with good preconditioner |
| > 500,000 | Iterative required (memory constraint) |

### 2. Problem Type -> Solver

| Problem type | Matrix property | KSP | PC | Tool params |
|---|---|---|---|---|
| Poisson, elasticity | SPD | CG | ICC or ILU | `solver_type="cg", preconditioner="ilu"` |
| Stokes, saddle-point | Indefinite | GMRES | LU (direct) | `solver_type="gmres", preconditioner="lu"` |
| Convection-dominated | Nonsymmetric | GMRES | ILU | `solver_type="gmres", preconditioner="ilu"` |
| Navier-Stokes (nonlinear) | Nonsymmetric | GMRES | ILU or Jacobi | `solver_type="gmres", preconditioner="ilu"` |
| Any (small) | Any | Preonly | LU | `solver_type="lu"` |

### 3. Direct Solver (When in Doubt)

```
solve(form_name="...", solver_type="lu", name="...")
```

Pros: Always converges for well-posed problems, no tuning needed.
Cons: O(n^2) memory, O(n^3) time. Impractical for large 3D problems.

### 4. CG + ILU (Symmetric Positive Definite)

```
solve(form_name="...", solver_type="cg", preconditioner="ilu", name="...")
```

Use for: Poisson, elasticity, diffusion, any SPD system.
Not for: Stokes, Navier-Stokes, convection-dominated problems.

### 5. GMRES + ILU (General)

```
solve(form_name="...", solver_type="gmres", preconditioner="ilu", name="...")
```

Use for: Nonsymmetric systems, convection-diffusion, general purpose.
If it fails: Try increasing `max_iterations`, loosening `tolerance`, or switch to LU.

## Iterative Solver Tuning

| Parameter | Default | When to change |
|---|---|---|
| `max_iterations` | 1000 | Increase for ill-conditioned systems |
| `tolerance` | 1e-10 | Loosen (1e-6) for approximate solutions |
| `preconditioner` | "ilu" | Try "lu", "jacobi", "hypre" |

## Nonlinear Problems

Use `solve_time_dependent` with Newton iteration or use the nonlinear solver path:

```
solve(form_name="...", solver_type="gmres", preconditioner="ilu",
      max_iterations=50, tolerance=1e-8, name="...")
```

## Diagnosing Solver Failures

Use `get_solver_diagnostics(solution_name="...")` to retrieve:
- Iteration count
- Residual history
- Convergence status

If solver diverges:
1. Try direct LU first (rules out matrix issues)
2. Check BCs are sufficient (underconstrained -> singular matrix)
3. Check form is correct (typos in UFL expressions)
4. Check material properties are positive (negative stiffness -> indefinite)

## Common PETSc Solver Combos

| Name | KSP/PC | Best for |
|---|---|---|
| Direct | preonly/lu | Small problems, debugging |
| CG+ICC | cg/icc | SPD, structured meshes |
| CG+AMG | cg/hypre | Large SPD, unstructured meshes |
| GMRES+ILU | gmres/ilu | General nonsymmetric |
| GMRES+ASM | gmres/asm | Parallel, domain decomposition |
| MinRes | minres/none | Symmetric indefinite (Stokes) |
