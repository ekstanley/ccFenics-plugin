---
name: fem-debugging
description: |
  Systematic debugging guide for DOLFINx solver failures and incorrect solutions.
  Use when the user encounters solver divergence, NaN values, convergence failure,
  wrong solution, unexpected results, or needs to debug a FEM simulation.
---

# FEM Debugging Checklist

## Quick Diagnosis

| Symptom | Likely cause | First action |
|---|---|---|
| Solver diverges | Missing BCs, wrong form | Check BCs with `get_session_state` |
| NaN in solution | Singular matrix, division by zero | Try direct LU solver |
| Wrong magnitude | Unit mismatch, wrong material props | Verify material values |
| Zero solution | BCs override everything, zero RHS | Check linear form and BC coverage |
| Oscillations | Wrong element, need stabilization | Try higher degree or DG |
| Slow convergence | Bad preconditioner, ill-conditioning | Switch to direct solver first |

## Step-by-Step Debugging Protocol

### Step 1: Inspect Session State

```
get_session_state()
```

Check:
- Is there at least one mesh? (No mesh = no problem)
- Is there at least one BC? (No BCs = likely singular system)
- Are forms defined? (No forms = nothing to solve)
- Do function space names in BCs match the form's space?

### Step 2: Verify Boundary Conditions

Common BC issues:
- **No BCs at all**: System is singular (Neumann-only needs special handling)
- **BCs on wrong boundary**: Check the boundary expression
- **Conflicting BCs**: Two BCs on same DOF with different values
- **BC value wrong type**: Scalar BC on vector space or vice versa

Debug with:
```
get_session_state()  # lists all BCs with their boundaries and values
```

### Step 3: Check the Variational Form

Common form issues:
- **Bilinear form not coercive**: Missing terms, wrong sign
- **Materials undefined**: Using `f` in form before `set_material_properties(name="f", ...)`
- **Wrong test/trial functions**: Mismatch between space and form variables
- **Missing measure**: Forgot `* dx` at the end

### Step 4: Try Direct Solver

If an iterative solver fails, always try LU first:
```
solve(form_name="...", solver_type="lu", name="test_solution")
```

If LU succeeds: the problem is well-posed, iterative solver needs tuning.
If LU fails: the problem is ill-posed (singular matrix, wrong BCs).

### Step 5: Check Solver Diagnostics

```
get_solver_diagnostics(solution_name="...")
```

Look for:
- **Iteration count at max**: Solver didn't converge, need more iterations or better PC
- **Residual increasing**: Diverging, likely CG on non-SPD system
- **Residual stuck**: Preconditioner is ineffective

### Step 6: Verify Solution Quality

```
compute_error(solution_name="u_h", exact_solution="...", error_type="L2")
compute_mesh_quality(mesh_name="...")
```

### Step 7: Mesh Quality

```
compute_mesh_quality(mesh_name="...")
```

Check for:
- **Aspect ratio > 10**: Severely distorted elements degrade accuracy
- **Min quality < 0.1**: Near-degenerate elements cause ill-conditioning
- **Suitable resolution**: Enough elements to resolve features

## Error Message -> Fix Mapping

| Error message (pattern) | Diagnosis | Fix |
|---|---|---|
| "KSP diverged" | Iterative solver failed | Try `solver_type="lu"` |
| "Singular matrix" | Insufficient BCs or wrong form | Add BCs, check form |
| "DIVERGED_ITS" | Max iterations reached | Increase max_iterations or use direct |
| "DIVERGED_DTOL" | Residual grew too large | Wrong solver type (CG on nonsymmetric) |
| "No active mesh" | Forgot to create mesh | Call `create_unit_square` or `create_mesh` |
| "Function space not found" | Typo in space name | Check `get_session_state` for names |
| "forbidden token" | Security check on expression | Remove import/exec/os from expression |

## Sanity Checks Before Solving

1. `get_session_state()` -- verify all objects exist
2. At least one BC is defined
3. Material properties referenced in forms are defined
4. Mesh resolution is adequate (start coarse, refine)
5. Element type matches problem type (vector for elasticity, mixed for Stokes)

## When All Else Fails

1. Simplify: Solve on a coarser mesh with simpler BCs
2. Use a known exact solution to verify the pipeline
3. Check the Poisson workflow first (simplest PDE) -- if that works, the MCP tools are functioning correctly
4. Use `get_session_state()` to audit all registered objects
