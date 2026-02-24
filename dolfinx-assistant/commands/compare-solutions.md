---
description: Compare two solutions side by side
allowed-tools: Read, Write
model: sonnet
---

Compare two DOLFINx solutions. Useful for verifying mesh independence, comparing element choices, or parameter studies.

## Input

If $ARGUMENTS is provided, parse it as two solution names. Otherwise, ask the user:
- Which two solutions to compare (by name)
- What varied between them (mesh size, element degree, parameters, solver)

## Comparison Steps

1. **Session check**: Call `get_session_state` to confirm both solutions exist.

2. **Metadata comparison**: Show a side-by-side table:

| Property | Solution A | Solution B |
|----------|-----------|-----------|
| Mesh cells | ... | ... |
| DOFs | ... | ... |
| Element | ... | ... |
| Solver time | ... | ... |

3. **Norm comparison**: For each solution, compute:
   - L2 norm via `compute_functionals` with `"inner(u, u)*dx"`
   - H1 seminorm via `compute_functionals` with `"inner(grad(u), grad(u))*dx"`

4. **Point-wise comparison**: Evaluate both solutions at 5-10 characteristic points using `evaluate_solution`. Present differences.

5. **Visual comparison**: Generate plots for both using `plot_solution` with the same colormap range.

6. **Difference analysis**: If solutions are on the same mesh, use `run_custom_code` to compute the pointwise difference and its L2 norm.

## Assessment

- If L2 difference < 1e-6: solutions agree (mesh independent or equivalent)
- If L2 difference is O(h^k): expected discretization error
- If large unexplained differences: flag potential issue

Present a clear verdict on whether the solutions are consistent.
