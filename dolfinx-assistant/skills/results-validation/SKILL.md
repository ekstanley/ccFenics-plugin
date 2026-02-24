---
name: results-validation
description: >
  This skill should be used when the user asks about "validating results",
  "checking solution accuracy", "manufactured solution", "convergence verification",
  "energy conservation", "mesh independence", "results look wrong",
  or needs guidance on verifying FEM simulation results in DOLFINx.
version: 0.1.0
---

# Results Validation Guide

Never trust a simulation result without verification. This guide covers systematic approaches to confirm your DOLFINx solutions are correct.

## Validation Hierarchy

Apply these checks in order. Each level catches different error classes.

### Level 1: Sanity Checks (Always Do These)

Run before any detailed analysis.

1. **Solution bounds**: Is the solution within physically reasonable limits?
   - Temperature can't be negative Kelvin
   - Displacement shouldn't exceed domain size
   - Pressure should match expected order of magnitude

2. **Boundary condition satisfaction**: Evaluate the solution at boundary points.
   ```
   Use evaluate_solution at boundary points to confirm BCs are satisfied.
   Tolerance: |u_computed - u_prescribed| < 1e-10 for Dirichlet BCs.
   ```

3. **Symmetry**: If the problem has geometric/loading symmetry, the solution must reflect it.

4. **Conservation**: Integrate relevant quantities.
   - Mass conservation: ∫ div(u) dx ≈ 0 for incompressible flow
   - Energy balance: input energy ≈ stored + dissipated energy

### Level 2: Manufactured Solutions

The gold standard for code verification. Works for any PDE.

**Process**:
1. Choose an exact solution u_exact (smooth, non-trivial)
2. Substitute into PDE to get the source term f
3. Solve with that source term
4. Compute L2 and H1 error norms

**Good manufactured solutions**:
- `sin(pi*x[0])*sin(pi*x[1])` — smooth, satisfies homogeneous Dirichlet on unit square
- `x[0]*(1-x[0])*x[1]*(1-x[1])` — polynomial, easy to differentiate
- `exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2)/0.1)` — localized, tests gradient resolution

**Expected errors** (P_k elements on quasi-uniform mesh with spacing h):

| Norm | Convergence Rate |
|------|-----------------|
| L2 | O(h^{k+1}) |
| H1 | O(h^k) |

If rates don't match theory, something is wrong with the implementation.

### Level 3: Mesh Convergence Study

Confirms the solution is mesh-independent.

**Process**:
1. Solve on mesh sizes N = 8, 16, 32, 64 (or finer)
2. Compute error norm at each level
3. Plot log(error) vs log(h)
4. Check slope matches theoretical rate

**Warning signs**:
- Error plateaus early → solver tolerance too loose
- Error increases on finer meshes → bug in formulation
- Rate is lower than expected → singularity, boundary layer, or wrong element

### Level 4: Comparison Benchmarks

Compare against known reference solutions.

- **Analytical solutions**: Beam bending, cavity flow at low Re, manufactured problems
- **Benchmark databases**: NAFEMS benchmarks for structural, lid-driven cavity for fluids
- **Published results**: Compare with literature values at specific points

### Level 5: Cross-Verification

Run the same problem with different:
- Element types (P1 vs P2)
- Solver configurations (direct vs iterative)
- Mesh types (triangles vs quadrilaterals)

Results should agree within discretization error bounds.

## Mesh Quality Checks

Run before solving. Bad meshes cause bad solutions regardless of formulation.

| Metric | Acceptable Range | Check With |
|--------|-----------------|------------|
| Min volume | > 0 (all positive) | `compute_mesh_quality` |
| Quality ratio (min/max vol) | > 0.01 | `compute_mesh_quality` |
| Aspect ratio | < 10 for tets, < 5 for tris | Custom check via `run_custom_code` |

## Solver Convergence Checks

After every solve, verify:

1. **Converged flag**: `converged == True` from solver output
2. **Residual norm**: Should be below tolerance (typically < 1e-10)
3. **Iteration count**: If near max_iter, solution may not be converged
4. **Convergence reason**: Positive reason codes = converged, negative = diverged

## Automated Report Generation

After validation, automatically generate a comprehensive HTML report containing all results and diagnostics:

```python
generate_report(
    title="Simulation Validation Report",
    include_plots=True,              # Embed solution plots
    include_solver_info=True,        # Solver diagnostics table
    include_mesh_info=True,          # Mesh statistics
    include_session_state=True,      # Full session JSON (for archiving)
    output_file="validation_report.html"
)
```

The generated report is self-contained and can be viewed in any web browser.

### File Management

Workspace utilities for organizing and archiving results:

```python
# List output files
list_workspace_files(pattern="*.png")  # Find all plot files
list_workspace_files(pattern="*.vtu")  # Find VTK exports

# Read files (base64 for images, text for data files)
read_workspace_file(file_path="plot.png")
read_workspace_file(file_path="solution.vtu")

# Bundle results into a single archive
bundle_workspace_files(
    file_paths=["*.vtu", "*.png", "report.html", "results.csv"],
    archive_name="results.zip"
)
```

Use these tools to:
- Archive all outputs from a simulation campaign
- Package results for sharing or publication
- Organize mesh and solution files for later post-processing

## Report Checklist

Every simulation report should include:

- [ ] Problem description (PDE, domain, BCs, material parameters)
- [ ] Mesh details (type, size, quality metrics)
- [ ] Element choice with justification
- [ ] Solver configuration and convergence info
- [ ] Solution plots (contour or warp)
- [ ] Quantitative validation (error norms, point values, integrals)
- [ ] Mesh convergence data (if applicable)
- [ ] Known limitations and assumptions

## Reference Material

For detailed benchmark problems and validation test cases, see `references/validation-benchmarks.md`.
