---
description: Run a mesh convergence study with error analysis
allowed-tools: Read, Write
model: sonnet
---

Run a mesh convergence study for the current problem setup.

## Required Input

Ask the user for:
- The exact solution expression (for error computation). If unknown, ask if they want to use Richardson extrapolation instead.
- Mesh sizes to test (default: 8, 16, 32, 64)
- Element degree (default: use current setup)

## Process

For each mesh size N:

1. Create mesh with `create_mesh` (name it `conv_mesh_N`)
2. Create function space with same element family/degree
3. Re-apply material properties and boundary conditions
4. Define the variational form
5. Solve
6. Compute L2 and H1 errors against the exact solution using `compute_error`

## Analysis

After all solves:

1. Compute convergence rates between consecutive mesh sizes:
   - rate = log(e_coarse / e_fine) / log(h_coarse / h_fine)

2. Present results table:

| N | h | DOFs | L2 Error | L2 Rate | H1 Error | H1 Rate |
|---|---|------|----------|---------|----------|---------|

3. Compare observed rates against theoretical:
   - L2 expected: k+1 (where k = polynomial degree)
   - H1 expected: k

4. Assessment:
   - Rates match theory → implementation verified
   - Rates lower → possible singularity, boundary layer, or bug
   - Rates higher → superconvergence (usually fine, but note it)

5. Plot the convergence data using `plot_solution` or `run_custom_code` if matplotlib is available.

Save the convergence data to a summary file.
