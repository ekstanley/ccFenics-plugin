# Diagnostic Codes Reference

## PETSc KSP Convergence Reason Codes

### Converged (positive values)

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | KSP_CONVERGED_RTOL_NORMAL | Residual norm decreased by rtol |
| 2 | KSP_CONVERGED_ATOL_NORMAL | Residual norm below atol |
| 3 | KSP_CONVERGED_RTOL | Relative tolerance satisfied |
| 4 | KSP_CONVERGED_ATOL | Absolute tolerance satisfied |
| 5 | KSP_CONVERGED_ITS | Converged within max iterations |
| 9 | KSP_CONVERGED_STEP_LENGTH | Step length criterion met |

### Diverged (negative values)

| Code | Constant | Meaning | Common Fix |
|------|----------|---------|------------|
| -2 | KSP_DIVERGED_NULL | Null preconditioner | Check PC setup |
| -3 | KSP_DIVERGED_ITS | Max iterations, not converged | Increase max_iter or better PC |
| -4 | KSP_DIVERGED_DTOL | Residual grew beyond dtol | Problem ill-posed or wrong solver |
| -5 | KSP_DIVERGED_BREAKDOWN | Method breakdown | Try different KSP/PC combo |
| -6 | KSP_DIVERGED_BREAKDOWN_BICG | BiCG breakdown | Switch to GMRES |
| -7 | KSP_DIVERGED_NONSYMMETRIC | CG on non-symmetric matrix | Use GMRES instead |
| -8 | KSP_DIVERGED_INDEFINITE_PC | Indefinite preconditioner | Check matrix properties |
| -9 | KSP_DIVERGED_NANORINF | NaN or Inf detected | Check formulation |
| -10 | KSP_DIVERGED_INDEFINITE_MAT | Indefinite matrix | Wrong problem type assumption |

## PETSc SNES Convergence Reason Codes

### Converged (positive values)

| Code | Constant | Meaning |
|------|----------|---------|
| 2 | SNES_CONVERGED_FNORM_ABS | Absolute function norm below atol |
| 3 | SNES_CONVERGED_FNORM_RELATIVE | Relative function norm below rtol |
| 4 | SNES_CONVERGED_SNORM_RELATIVE | Step norm below stol |

### Diverged (negative values)

| Code | Constant | Meaning | Common Fix |
|------|----------|---------|------------|
| -1 | SNES_DIVERGED_FUNCTION_DOMAIN | Function domain error | Check BCs, material params |
| -2 | SNES_DIVERGED_FUNCTION_COUNT | Max function evaluations | Increase limit or simplify |
| -3 | SNES_DIVERGED_LINEAR_SOLVE | Inner KSP failed | Fix linear solver config |
| -4 | SNES_DIVERGED_FNORM_NAN | NaN in function norm | Check Jacobian, reduce step |
| -5 | SNES_DIVERGED_MAX_IT | Max Newton iterations | Better initial guess or load stepping |
| -6 | SNES_DIVERGED_LINE_SEARCH | Line search failure | Try different line search type |
| -8 | SNES_DIVERGED_LOCAL_MIN | Stuck at local minimum | Different initial guess |

## SLEPc EPS Convergence Reason Codes

### Converged (positive values)

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | EPS_CONVERGED_TOL | Tolerance satisfied |
| 2 | EPS_CONVERGED_USER | User-defined convergence |

### Diverged (negative values)

| Code | Constant | Meaning |
|------|----------|---------|
| -1 | EPS_DIVERGED_ITS | Max iterations reached |
| -2 | EPS_DIVERGED_BREAKDOWN | Numerical breakdown |
| -3 | EPS_DIVERGED_SYMMETRY_LOST | Symmetry lost during iteration |

## Common Error Patterns and Root Causes

### Pattern: Residual oscillates without decreasing

- **Cause**: Wrong KSP type for problem symmetry
- **Fix**: If SPD → CG; if non-symmetric → GMRES; if saddle-point → MINRES

### Pattern: First Newton iteration OK, second diverges

- **Cause**: Jacobian inconsistent with residual (wrong derivative)
- **Fix**: Let DOLFINx auto-differentiate the Jacobian (omit `jacobian` parameter)

### Pattern: Converges on coarse mesh, diverges on fine

- **Cause**: Preconditioner doesn't scale
- **Fix**: Switch from ILU to AMG (Hypre BoomerAMG)

### Pattern: Solution has NaN only in corners

- **Cause**: Conflicting Dirichlet BCs at corner nodes
- **Fix**: Ensure consistent BC values where boundaries meet

### Pattern: Eigenvalue solver returns spurious modes

- **Cause**: Null space not handled
- **Fix**: Use spectral transform with shift-invert (`target` parameter)
