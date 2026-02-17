---
name: fem-workflow-singular-poisson
description: |
  Guides the user through solving a singular Poisson problem (pure Neumann BCs) using DOLFINx MCP tools.
  Use when the user asks about singular Poisson, nullspace, pure Neumann BCs,
  compatibility condition, or constant mode removal.
---

# Singular Poisson Workflow (Tutorial Ch2.3)

Solve: **-div(grad(u)) = f** with **pure Neumann BCs** (du/dn = g on all boundaries).

## The Singularity Problem

With only Neumann BCs, the solution is determined only up to a constant. The system matrix is singular (has a nullspace consisting of constant functions).

**Compatibility condition**: The source must satisfy `integral(f) dx + integral(g) ds = 0`.

## Implementation via `run_custom_code`

Handling the nullspace requires PETSc nullspace attachment:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import ufl
from mpi4py import MPI
from petsc4py import PETSc

msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(msh, ("Lagrange", 1))

# Source term satisfying compatibility: integral(f) = 0
x = ufl.SpatialCoordinate(msh)
f = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Assemble
A = assemble_matrix(fem.form(a))
A.assemble()
b = assemble_vector(fem.form(L))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Create nullspace (constant vector)
ns_vec = fem.Function(V)
ns_vec.x.array[:] = 1.0
nullspace = PETSc.NullSpace().create(constant=True)
A.setNullSpace(nullspace)
nullspace.remove(b)  # Orthogonalize RHS against nullspace

# Solve with CG + nullspace
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("cg")
ksp.getPC().setType("hypre")
ksp.setTolerances(rtol=1e-10)

uh = fem.Function(V)
ksp.solve(b, uh.x.petsc_vec)

# Remove mean (fix the constant)
mean = fem.assemble_scalar(fem.form(uh * ufl.dx))
area = fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx))
uh.x.array[:] -= mean / area

print(f"Solution norm: {np.linalg.norm(uh.x.array):.6e}")
print(f"Solution mean: {uh.x.array.mean():.2e} (should be ~0)")
""")
```

## Key Concepts

- **Nullspace**: The constant function `c` satisfies `A*c = 0` for pure Neumann problems
- **Compatibility**: `integral(f) dx = 0` is required for a solution to exist
- **PETSc NullSpace**: Attach via `A.setNullSpace()` so the Krylov solver handles it
- **Mean subtraction**: After solving, subtract the mean to pin the solution uniquely

## Steps Summary

1. Verify compatibility condition: `integral(f) dx + integral(g) ds = 0`
2. Assemble system without Dirichlet BCs
3. Create nullspace: `PETSc.NullSpace().create(constant=True)`
4. Attach nullspace to matrix and orthogonalize RHS
5. Solve with iterative solver (CG + AMG)
6. Post-process: subtract mean value

## Common Pitfalls

| Issue | Cause | Fix |
|---|---|---|
| Solver diverges | Incompatible RHS | Ensure compatibility condition |
| Non-zero mean | Nullspace not removed | Call `nullspace.remove(b)` |
| Slow convergence | No nullspace attached | Attach nullspace to matrix |
| Wrong solution | Direct solver without nullspace | Use iterative solver with nullspace |
