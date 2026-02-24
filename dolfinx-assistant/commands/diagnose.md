---
description: Diagnose a failed or suspicious simulation
allowed-tools: Read, Write
model: sonnet
---

Run a systematic diagnosis on the current simulation session. Use when a solve failed, produced NaN values, gave wrong results, or is running too slowly.

## Automatic Diagnosis Sequence

Execute these checks in order, stopping when the root cause is found:

### Check 1: Session Completeness

Call `get_session_state`. Verify all required components exist:

- [ ] Active mesh defined
- [ ] Function space created on active mesh
- [ ] At least one boundary condition applied
- [ ] Variational form defined (bilinear + linear)
- [ ] If nonlinear: unknown function exists

**If anything missing**: Report what's missing and stop. The problem is incomplete setup.

### Check 2: Mesh Validity

Call `get_mesh_info` and `compute_mesh_quality`:

- [ ] All cell volumes positive (no inverted elements)
- [ ] Quality ratio > 0.01
- [ ] Reasonable cell count for the problem

**If mesh invalid**: Recommend re-meshing. Inverted elements cause all solvers to fail.

### Check 3: Solver Convergence

Call `get_solver_diagnostics`:

- [ ] `converged == True`
- [ ] Positive convergence reason code
- [ ] Residual norm below tolerance
- [ ] Iterations well below max_iter

**If diverged**: Report the convergence reason code and suggest fixes based on the debugging-diagnostics skill.

### Check 4: Solution Sanity

If a solution exists:

1. Check min/max values: `evaluate_solution` at domain corners and center
2. Check BC satisfaction: evaluate at boundary points
3. Check for NaN: if min or max is NaN, flag immediately

**If NaN found**: Check material properties and BCs for zero/negative values where positive expected.

**If BCs not satisfied**: BC application may have failed — check boundary markers.

### Check 5: Formulation Audit

If all above pass but results look wrong:

1. Verify material property values and units
2. Check sign conventions in the variational form
3. Confirm Neumann vs Dirichlet BC application
4. For mixed problems: verify inf-sup stable element pair

## Output

Present findings as a diagnostic report:

```
DIAGNOSIS REPORT
================
Session: [mesh name], [space name], [element info]
Status: [PASS / FAIL / WARNING]

Checks:
  ✓ Session complete
  ✓ Mesh valid (quality ratio: 0.15)
  ✗ Solver diverged (reason: -3, max iterations)

Root cause: Iterative solver with ILU preconditioner on 200K DOF problem.
Recommendation: Switch to CG + Hypre AMG or use direct solver.
```

If $ARGUMENTS contains a specific complaint (e.g., "NaN", "slow", "wrong"), prioritize that category's checks.
