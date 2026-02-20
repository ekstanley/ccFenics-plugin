---
name: fem-workflow-electromagnetics
description: |
  Guides the user through solving electromagnetics problems using Nedelec edge elements and DOLFINx MCP tools.
  Use when the user asks about electromagnetics, Nedelec elements, H(curl), Maxwell equations,
  edge elements, or curl-curl problems.
---

# Electromagnetics Workflow (Tutorial Ch3.6)

Solve the **curl-curl equation** using **Nedelec (edge) elements**: `curl(curl(E)) + k^2 E = J`.

## Background

Maxwell's equations in the frequency domain reduce to:
- **curl(curl(E)) - k^2 E = -j*omega*mu*J** (electric field formulation)
- H(curl)-conforming elements (Nedelec) are required for correct tangential continuity

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="em_mesh")
```

Or for 3D:
```
create_mesh(shape="box", nx=16, ny=16, nz=16, name="em_mesh")
```

### 2. Create Nedelec Function Space

```
create_function_space(family="N1curl", degree=1, name="V")
```

This creates an H(curl)-conforming Nedelec space. Alternative: `family="N2curl"` for second kind.

### 3. Define Source Current

```
set_material_properties(name="J_mag", value="1.0")
```

### 4. Define Variational Form

The curl-curl weak form:
```
define_variational_form(
    bilinear="inner(curl(u), curl(v)) * dx + k2 * inner(u, v) * dx",
    linear="J_mag * inner(as_vector([0, 0, 1]), v) * dx",
    name="maxwell"
)
```

For 2D (scalar curl):
```
define_variational_form(
    bilinear="inner(curl(u), curl(v)) * dx + inner(u, v) * dx",
    linear="inner(as_vector([1.0, 0.0]), v) * dx",
    name="maxwell_2d"
)
```

### 5. Apply Tangential BCs

Perfect electric conductor (PEC): tangential E = 0 on boundary. For Nedelec elements this is handled via `run_custom_code` for tangential projection:

```python
run_custom_code(code="""
from dolfinx import fem, mesh
import numpy as np
import ufl
from mpi4py import MPI

# After creating mesh and Nedelec space...
# PEC BC: n x E = 0 on boundary
tdim = msh.topology.dim
fdim = tdim - 1
boundary_facets = mesh.exterior_facets(msh)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
u_bc = fem.Function(V)
bc = fem.dirichletbc(u_bc, dofs)
""")
```

> **Namespace persistence**: Variables defined in `run_custom_code` persist across calls.
> You can split complex workflows into multiple calls without re-importing or re-defining objects.
> Session-registered objects (meshes, spaces, functions) are always injected fresh and override stale names.

### 6. Solve

Curl-curl systems are often indefinite. Use direct solver or AMS preconditioner:

```
solve(solver_type="direct", solution_name="E_field")
```

For large problems with the AMS (Auxiliary-space Maxwell Solver) preconditioner:
```
solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="hypre",
    petsc_options={"pc_hypre_type": "ams"},
    solution_name="E_field"
)
```

## Key Concepts

- **Nedelec elements**: DOFs are tangential components on edges (not nodal values)
- **H(curl) space**: Functions with square-integrable curl
- **Tangential BCs**: For PEC, tangential E = 0 (natural for Nedelec DOFs on boundary edges)
- **AMS preconditioner**: Specialized for curl-curl problems, available in Hypre

## Element Types for EM

| Space | DOLFINx Family | Continuity | Use Case |
|---|---|---|---|
| H(curl) | `N1curl` | Tangential | Electric field E |
| H(div) | `RT` | Normal | Magnetic flux B |
| H^1 | `Lagrange` | Full | Scalar potential phi |
| L^2 | `DG` | None | Charge density rho |

## Discrete Gradient for Preconditioning

The AMS preconditioner needs a discrete gradient operator. With existing tools, `create_discrete_operator` provides this.
