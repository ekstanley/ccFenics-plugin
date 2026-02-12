---
name: fem-workflow-mixed-poisson
description: |
  Guides the user through solving the mixed formulation of the Poisson equation using Raviart-Thomas elements and DOLFINx MCP tools.
  Use when the user asks about mixed Poisson, Raviart-Thomas elements, flux variable,
  saddle-point systems, or Schur complement preconditioners.
---

# Mixed Poisson Workflow (Tutorial Ch4.1)

Solve the Poisson equation in **mixed form** with separate unknowns for the flux (sigma) and the scalar field (u).

## Mixed Formulation

Instead of solving `-div(grad(u)) = f`, introduce `sigma = -grad(u)`:

**sigma + grad(u) = 0** in Omega
**div(sigma) = -f** in Omega

## Function Spaces

- **sigma** in **H(div)**: Raviart-Thomas (RT) elements
- **u** in **L^2**: Discontinuous Galerkin (DG) elements
- This is a saddle-point (indefinite) system

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="mixed_mesh")
```

### 2. Create Function Spaces

```
create_function_space(family="RT", degree=1, name="Sigma")
create_function_space(family="DG", degree=0, name="U")
```

### 3. Create Mixed Space

```
create_mixed_space(spaces=["Sigma", "U"], name="W")
```

### 4. Define Source Term

```
set_material_properties(name="f", value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])")
```

### 5. Define Variational Form

Mixed form with trial functions `(sigma, u)` and test functions `(tau, v)`:

```
define_variational_form(
    bilinear="inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx",
    linear="-f * v * dx",
    name="mixed_poisson"
)
```

### 6. Apply Essential BC on Sigma

For the mixed formulation, the essential BC is on the flux (sigma . n on boundary):
This typically requires `run_custom_code` for proper setup.

### 7. Solve with Appropriate Preconditioner

Saddle-point systems need special solvers:

```
solve(
    solver_type="iterative",
    ksp_type="minres",
    pc_type="fieldsplit",
    petsc_options={
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "none"
    },
    solution_name="mixed_sol"
)
```

Or use a direct solver for small problems:
```
solve(solver_type="direct", solution_name="mixed_sol")
```

## Complete Implementation via `run_custom_code`

For full control over the mixed space and fieldsplit preconditioner:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
from basix.ufl import element, mixed_element

msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# Create mixed element
rt_el = element("RT", msh.basix_cell(), 1)
dg_el = element("DG", msh.basix_cell(), 0)
mel = mixed_element([rt_el, dg_el])
W = fem.functionspace(msh, mel)

# Trial and test functions
(sigma, u) = ufl.TrialFunctions(W)
(tau, v) = ufl.TestFunctions(W)

# Source term
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2) / 0.02)

# Mixed form
a = (ufl.inner(sigma, tau) + ufl.inner(u, ufl.div(tau)) + ufl.inner(ufl.div(sigma), v)) * ufl.dx
L = -f * v * ufl.dx

# Solve
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
wh = problem.solve()

sigma_h, u_h = wh.split()
print(f"Flux norm: {np.linalg.norm(sigma_h.x.array):.6e}")
print(f"Solution norm: {np.linalg.norm(u_h.x.array):.6e}")
""")
```

## Key Concepts

- **H(div) space**: RT elements have continuous normal components across edges
- **Saddle-point system**: The mixed system is indefinite (not positive definite)
- **Inf-sup stability**: RT_k x DG_{k-1} satisfies the LBB condition
- **Fieldsplit**: PETSc preconditioner that treats each field separately
- **Flux conservation**: The mixed formulation gives locally conservative fluxes

## Element Pairings

| Flux Space | Scalar Space | Order | Stability |
|---|---|---|---|
| RT_1 | DG_0 | Lowest order | Stable |
| RT_2 | DG_1 | Second order | Stable |
| BDM_1 | DG_0 | Alternative | Stable |
