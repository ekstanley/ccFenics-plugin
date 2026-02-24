---
description: Generate a formatted simulation report
allowed-tools: Read, Write
model: sonnet
---

Generate a structured simulation report for the current session. Write the report as an HTML file in the workspace.

## Report Structure

### 1. Header
- Title: Simulation type + date
- Author: from user context or ask

### 2. Problem Description
- PDE formulation (weak form)
- Domain geometry and dimensions
- Material parameters and their values

### 3. Mesh Details
- Call `get_mesh_info` for cell type, count, vertices
- Call `compute_mesh_quality` for quality metrics
- Include mesh quality assessment (pass/warn/fail)

### 4. Discretization
- Element family and degree
- Number of DOFs
- Justification for element choice

### 5. Boundary Conditions
- List all BCs with type, location, and value
- Call `get_session_state` to enumerate registered BCs

### 6. Solver Configuration
- Solver type (direct/iterative)
- Call `get_solver_diagnostics` for convergence info
- Iterations, residual norm, wall time

### 7. Results
- Solution plots: call `plot_solution` with contour type
- Key quantities: call `compute_functionals` for relevant integrals
- Point evaluations at important locations using `evaluate_solution`

### 8. Validation
- Error norms if exact solution available
- Convergence data if available
- Comparison with benchmarks if applicable

### 9. Conclusions
- Summary of key findings
- Known limitations
- Recommendations for further analysis

## Formatting

Use the `generate_report` tool to create the HTML report. Include all available plots. Set a descriptive title.

If the user specified $ARGUMENTS, use it as the report title.
