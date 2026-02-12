---
name: fem-workflow-adaptive-refinement
description: |
  Guides the user through adaptive mesh refinement (AMR) using DOLFINx MCP tools.
  Use when the user asks about adaptive refinement, AMR, error estimators,
  local refinement, mesh adaptivity, or solve-estimate-mark-refine loops.
---

# Adaptive Mesh Refinement Workflow (Tutorial Ch2.9)

Automatically refine the mesh where the error is largest, using a **solve-estimate-mark-refine** loop.

## Algorithm

1. **Solve** the PDE on the current mesh
2. **Estimate** the error per cell (residual-based indicator)
3. **Mark** cells with large errors for refinement
4. **Refine** the marked cells
5. Repeat until tolerance is met

## Implementation via `run_custom_code`

AMR requires element-wise error estimation and selective refinement:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI

def solve_and_estimate(msh, degree=1):
    V = fem.functionspace(msh, ("Lagrange", degree))

    # Problem setup (Poisson with singular source)
    x = ufl.SpatialCoordinate(msh)
    f = 100 * ufl.exp(-200 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # BC
    tdim = msh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.exterior_facets(msh)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(fem.Function(V), dofs)

    # Solve
    problem = LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Error indicator (element residual)
    # h_K^2 * ||f + div(grad(u_h))||^2 on each cell
    W = fem.functionspace(msh, ("DG", 0))
    eta = fem.Function(W)
    # Simplified: use gradient jump as indicator
    eta_form = fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
    # ... compute cell-wise indicator

    return uh, eta

# AMR loop
msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

for level in range(5):
    uh, eta = solve_and_estimate(msh)
    num_cells = msh.topology.index_map(2).size_global
    print(f"Level {level}: {num_cells} cells")

    # Mark top 30% of cells
    threshold = np.percentile(eta.x.array, 70)
    cells_to_refine = np.where(eta.x.array > threshold)[0]

    # Refine
    edges = mesh.compute_incident_entities(msh.topology, cells_to_refine, 2, 1)
    msh = mesh.refine(msh, edges)

print("AMR complete")
""")
```

## Key Concepts

- **Error estimator**: Measures local error per cell without knowing the exact solution
- **Residual estimator**: `eta_K = h_K * ||f + div(grad(u_h))||_K` (cell residual)
- **Marking strategy**: Dorfler marking (refine cells contributing to top fraction of total error)
- **Refinement**: `dolfinx.mesh.refine` with edge markers for bisection refinement

## Marking Strategies

| Strategy | Description | Typical Parameter |
|---|---|---|
| Fixed fraction | Refine top X% of cells | 30% |
| Dorfler | Refine until fraction of total error is marked | 50% of total |
| Maximum | Refine cells above threshold*max(eta) | threshold = 0.5 |

## Convergence

- **Uniform refinement**: Error decreases as O(h^p) where p is element degree
- **Adaptive refinement**: Error decreases as O(N^{-p/d}) where N is DOF count, d is dimension
- Adaptive achieves **optimal** convergence rate even for singular solutions
