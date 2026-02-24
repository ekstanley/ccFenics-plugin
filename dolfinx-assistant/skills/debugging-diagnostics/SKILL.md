---
name: debugging-diagnostics
description: >
  This skill should be used when the user says "solver diverged", "NaN values",
  "convergence failure", "wrong results", "solution looks wrong", "debug my simulation",
  "why did it fail", "residual blew up", "negative Jacobian", "ill-conditioned",
  or needs help diagnosing any DOLFINx simulation failure or suspicious result.
version: 0.1.0
---

# Debugging & Diagnostics Guide

Simulation failures fall into predictable categories. This guide walks through systematic diagnosis — from quick checks that catch 80% of issues to deep analysis for the remaining 20%.

## Triage: Classify the Failure

Start here. Ask the user what happened and map to a category:

| Symptom | Category | Jump To |
|---------|----------|---------|
| Solver returns "diverged" | Solver failure | §1 |
| NaN or Inf in solution | Numerical blowup | §2 |
| Solution is all zeros | Silent failure | §3 |
| Wrong magnitude or shape | Formulation error | §4 |
| Solver converges but results are unphysical | Modeling error | §5 |
| Solver is extremely slow | Performance issue | §6 |

## §1: Solver Failure (Diverged)

**First response**: Call `get_solver_diagnostics` and `get_session_state`.

### Check the convergence reason code

| Code | Meaning | Action |
|------|---------|--------|
| -3 | Max iterations reached | Increase `max_iter` or improve preconditioner |
| -4 | Divergence tolerance hit | System may be singular or ill-conditioned |
| -5 | Breakdown | Preconditioner failed — try a different one |
| -7 | Non-symmetric matrix with CG | Switch to GMRES or verify bilinear form symmetry |

### Systematic checks

1. **Are BCs applied?** A problem with no Dirichlet BCs and no other constraints is typically singular.
   - Call `get_session_state` → check `boundary_conditions` count
   - If zero BCs: likely singular system. Add BCs or use `nullspace_mode`.

2. **Is the matrix assembled correctly?** Use `run_custom_code` to check:
   - Matrix diagonal entries should be non-zero
   - Matrix should be well-conditioned (ratio of max/min diagonal ≈ mesh-dependent)

3. **Wrong solver for problem type?**
   - SPD problem → CG + AMG
   - Non-symmetric → GMRES + ILU
   - Saddle point → MINRES or direct solver
   - Mixed formulation → direct solver first (eliminates solver as variable)

4. **Mesh too coarse or too fine?**
   - Coarse mesh with high-degree elements → badly conditioned
   - Very fine mesh with direct solver → memory exhaustion

### Quick fix: Try direct solver

If iterative solver fails, switch to `solver_type="direct"` to isolate the issue. If direct also fails, the problem is in the formulation.

## §2: NaN / Inf in Solution

NaN means a division by zero or sqrt of negative number occurred somewhere.

### Diagnosis steps

1. **Check material properties**: Any zero or negative values where positive expected?
   - Diffusivity, Young's modulus, viscosity must be positive
   - Call `get_session_state` to list registered materials

2. **Check boundary conditions**: Conflicting BCs at corners produce singularities.
   - Two different Dirichlet values at the same DOF → undefined

3. **Check the variational form**: Common errors:
   - Missing `dx` in integral
   - Wrong sign on a term
   - Division by a function that can be zero

4. **For nonlinear problems**: Newton may have jumped to a bad state.
   - Check initial guess: should be physically reasonable
   - Reduce load/time step
   - Add line search: `snes_type="newtonls"` with backtracking

5. **For time-dependent problems**: Time step too large causes numerical instability.
   - Reduce `dt` by factor of 2-4
   - Check CFL condition for advection problems

## §3: Solution Is All Zeros

The solver "converged" but the solution is trivially zero.

### Common causes

1. **Zero source term and zero BCs**: The correct solution IS zero. Check that:
   - Source term `f` is actually non-zero
   - At least one non-homogeneous BC exists

2. **Material property not set**: If coefficient is zero or unregistered, the system has zero RHS.

3. **BC overwrites the entire domain**: Too many Dirichlet BCs can constrain all DOFs to zero.

4. **Form not assembled**: Check that `define_variational_form` was called before `solve`.

## §4: Wrong Magnitude or Shape

Solution converges but values are off by orders of magnitude or the spatial pattern is wrong.

### Checks

1. **Unit consistency**: Are all inputs in consistent units? Mixed SI/Imperial causes scale errors.

2. **Sign errors in weak form**: A minus sign flip reverses the solution direction.
   - Verify: `inner(grad(u), grad(v))*dx` vs `-inner(grad(u), grad(v))*dx`

3. **Wrong coefficient value**: Check material properties are in correct units and magnitude.

4. **Boundary condition on wrong boundary**: Verify boundary markers by evaluating at known points.
   - Use `evaluate_solution` at boundary points to confirm BC satisfaction.

5. **Neumann vs Dirichlet confusion**: Neumann BCs enter the linear form. Dirichlet BCs constrain DOFs directly. Mixing them up produces wrong results.

## §5: Unphysical Results

Solution has correct magnitude but violates physics.

### Red flags

- Negative temperature (in Kelvin)
- Displacement larger than domain
- Pressure oscillations (checkerboard pattern) → wrong element pair for Stokes
- Volumetric locking → P1 elements on near-incompressible material

### Fixes

- **Pressure oscillations**: Switch to inf-sup stable pair (Taylor-Hood P2/P1)
- **Locking**: Use P2 elements or mixed formulation
- **Overshoot/undershoot in advection**: Add stabilization (SUPG) or switch to DG
- **Non-physical stress concentrations**: Check mesh quality near those regions

## §6: Performance Issues

Solver runs but takes too long.

### Diagnosis

1. Call `get_solver_diagnostics` — check iteration count.
   - Direct solver slow: problem too large. Switch to iterative.
   - Iterative > 200 iterations: preconditioner is wrong.

2. **Preconditioner mismatch**:
   - Jacobi on ill-conditioned system → many iterations
   - AMG on non-SPD system → may not converge
   - ILU on parallel system → not scalable

3. **Element degree too high**: P3/P4 elements create much denser matrices. Consider P2 with finer mesh instead.

4. **Unnecessary mesh resolution**: Uniform fine mesh wastes DOFs in smooth regions. Consider adaptive refinement.

## Diagnostic Toolkit

### Session Inspection Tools

Before diving into solver diagnostics, inspect the full session state:

```
get_session_state()  # Lists all meshes, spaces, functions, BCs, forms, solutions
```

This reveals:
- **Missing BCs**: Inspect `boundary_conditions` dict — if empty, likely singular
- **Orphaned functions**: Functions referencing deleted spaces (invariant violation)
- **Stale forms**: Forms defined before current mesh/space changed
- **Active mesh**: Which mesh is currently active for new operations

**Reset if corrupted** (use as last resort):
```
reset_session()  # Clears ALL state and registries
```

**Remove specific objects** to fix inconsistencies:
```
remove_object(name="old_solution", object_type="solution")
remove_object(name="bad_mesh", object_type="mesh")  # Cascades: removes dependent spaces, functions, BCs
```

### Quick health check sequence

```
1. get_session_state          → Verify all components exist
2. compute_mesh_quality       → Ensure mesh is valid
3. get_solver_diagnostics     → Check convergence details
4. evaluate_solution at BCs   → Confirm BC satisfaction
5. compute_functionals        → Check conservation/integrals
```

### Emergency fallback: Simplify to known-good

When stuck, simplify the problem until it works, then add complexity back:

1. Start with unit square, P1 Lagrange, constant coefficients, direct solver
2. Verify with manufactured solution
3. Add complexity one piece at a time
4. When it breaks, the last addition is the culprit

## Reference Material

For detailed error codes and PETSc diagnostics, see `references/diagnostic-codes.md`.
