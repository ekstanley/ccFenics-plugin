---
name: fem-workflow-complex-poisson
description: |
  Guides the user through solving a complex-valued Poisson equation using DOLFINx MCP tools.
  Use when the user asks about complex-valued PDEs, sesquilinear forms, complex Poisson,
  or PETSc complex scalar type configuration.
---

# Complex-Valued Poisson Workflow (Tutorial Ch1.2)

Solve a Poisson equation with **complex-valued** source terms and solutions.

## Prerequisites

DOLFINx must be built with PETSc complex scalar type. This workflow uses `run_custom_code` for complex-specific operations.

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="complex_mesh")
```

### 2. Create Function Space

```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 3. Set Up Complex Problem via `run_custom_code`

Complex-valued problems require direct access to DOLFINx's complex scalar type:

```python
run_custom_code(code="""
import dolfinx
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Verify complex scalar type
assert np.dtype(PETSc.ScalarType).kind == 'c', "PETSc must be built with complex scalars"

# Create mesh and space
msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(msh, ("Lagrange", 1))

# Complex source term: f = 1 + 2j
f = fem.Constant(msh, PETSc.ScalarType(1 + 2j))

# Define sesquilinear form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary condition (complex-valued)
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.full(x.shape[1], 0.0 + 0.0j, dtype=PETSc.ScalarType))
tdim = msh.topology.dim
fdim = tdim - 1
boundary_facets = dolfinx.mesh.exterior_facets(msh)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_bc, dofs)

# Solve
problem = LinearProblem(a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

print(f"Solution norm: {np.linalg.norm(uh.x.array)}")
print(f"Real part range: [{uh.x.array.real.min():.6f}, {uh.x.array.real.max():.6f}]")
print(f"Imag part range: [{uh.x.array.imag.min():.6f}, {uh.x.array.imag.max():.6f}]")
""")
```

## Key Concepts

- **Sesquilinear form**: `inner(grad(u), grad(v))` where the test function v is conjugated
- **PETSc complex mode**: DOLFINx must be compiled with `--with-scalar-type=complex`
- **Complex constants**: Use `PETSc.ScalarType(1+2j)` for complex coefficients
- **Verification**: Check `PETSc.ScalarType` is complex before proceeding

## Common Pitfalls

- If PETSc is built with real scalars, complex-valued problems will fail silently or give wrong results
- Boundary conditions must also use complex-valued functions
- Use `np.dtype(PETSc.ScalarType).kind == 'c'` to verify complex mode
