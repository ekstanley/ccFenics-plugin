---
name: fem-workflow-hyperelasticity
description: |
  Guides the user through solving a hyperelasticity (large deformation) problem using DOLFINx MCP tools.
  Use when the user asks about hyperelasticity, large deformation, neo-Hookean material,
  nonlinear elasticity, finite strain, or deformation gradient.
---

# Hyperelasticity Workflow (Tutorial Ch2.7)

Solve a **geometrically nonlinear** elasticity problem using a stored energy function and Newton's method.

## Problem

Minimize the total potential energy:
**Pi(u) = integral(psi(F)) dx - integral(T . u) ds**

where:
- `F = I + grad(u)` = deformation gradient
- `psi(F)` = stored energy density (constitutive model)
- `T` = applied traction

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_mesh(shape="box", nx=8, ny=8, nz=8, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, name="hyper_mesh")
```

### 2. Create Vector Function Space

```
create_function_space(family="Lagrange", degree=1, shape=[3], name="V")
```

### 3. Create Initial Guess (Zero Displacement)

```
interpolate(name="u", expression="(0.0, 0.0, 0.0)", function_space="V")
```

### 4. Define Material Properties

```
set_material_properties(name="mu", value="1.0")
set_material_properties(name="lmbda", value="10.0")
```

### 5. Apply Boundary Conditions

Fixed face (x=0):
```
apply_boundary_condition(value="(0.0, 0.0, 0.0)", boundary="x[0] < 1e-14", name="clamp")
```

### 6. Solve Nonlinear Problem

The residual is derived from the strain energy. For a **compressible neo-Hookean** model:

```
solve_nonlinear(
    residual="mu * inner(Identity(3) + grad(u), grad(v)) * dx + (lmbda * ln(det(Identity(3) + grad(u))) - mu) * inner(inv(Identity(3) + grad(u)), grad(v)) * dx - dot(as_vector([0, 0, -0.1]), v) * ds",
    unknown="u",
    snes_type="newtonls",
    max_iter=50,
    solution_name="displacement"
)
```

For more complex energy functions, use `variable` and `diff`:

```python
run_custom_code(code="""
import ufl
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np

# Setup mesh and space
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)
V = fem.functionspace(mesh, ("Lagrange", 1, (3,)))

u = fem.Function(V)
v = ufl.TestFunction(V)

# Material parameters
mu = fem.Constant(mesh, 1.0)
lmbda = fem.Constant(mesh, 10.0)

# Kinematics
d = len(u)
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))
C = F.T * F
J = ufl.det(F)
Ic = ufl.tr(C)

# Neo-Hookean stored energy
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * ufl.ln(J)**2

# First Piola-Kirchhoff stress via automatic differentiation
P = ufl.diff(psi, F)

# Residual
F_form = ufl.inner(P, ufl.grad(v)) * ufl.dx

# Boundary conditions and solve...
""")
```

> **Namespace persistence**: Variables defined in `run_custom_code` persist across calls.
> You can split complex workflows into multiple calls without re-importing or re-defining objects.
> Session-registered objects (meshes, spaces, functions) are always injected fresh and override stale names.

### 7. Post-Process

```
export_solution(solution_name="displacement", filename="hyperelastic", format="vtk")
```

## Key Concepts

- **Deformation gradient**: `F = I + grad(u)` maps reference to deformed configuration
- **Strain energy**: `psi(F)` defines the material response (neo-Hookean, Mooney-Rivlin, etc.)
- **Automatic differentiation**: `ufl.diff(psi, F)` computes the 1st Piola-Kirchhoff stress
- **`ufl.variable`**: Required for `ufl.diff` -- wraps F so UFL can differentiate w.r.t. it
- **Load stepping**: For large deformations, apply the load incrementally

## Common Energy Functions

| Model | Energy `psi(F)` | Use Case |
|---|---|---|
| Neo-Hookean | `mu/2*(Ic-3) - mu*ln(J) + lmbda/2*ln(J)^2` | Rubber, soft tissue |
| Mooney-Rivlin | `c1*(I1-3) + c2*(I2-3)` | More accurate rubber |
| Saint Venant-Kirchhoff | `lmbda/2*tr(E)^2 + mu*tr(E^2)` | Small-moderate strain |
