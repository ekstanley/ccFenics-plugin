---
name: fem-workflow-membrane
description: |
  Guides the user through solving a membrane deflection problem on a curved domain using DOLFINx MCP tools.
  Use when the user asks about membrane deflection, curved domains, Gaussian loads,
  circular meshes, or Gmsh integration with DOLFINx.
---

# Membrane Deflection Workflow (Tutorial Ch1.4)

Solve the Poisson equation on a **circular domain** with a Gaussian load, representing membrane deflection under pressure.

## Problem

**-div(grad(w)) = p(x,y)** on a circular domain of radius R, where:
- `w` = membrane deflection
- `p(x,y) = 4 * exp(-beta^2 * (x^2 + (y-R0)^2))` = Gaussian pressure load

## Step-by-Step Tool Sequence

### 1. Create Circular Mesh via Gmsh

The standard mesh tools only support rectangles/boxes. Use `run_custom_code` for Gmsh:

```python
run_custom_code(code="""
import gmsh
import numpy as np
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

gmsh.initialize()
gmsh.model.add("membrane")

# Circle domain
R = 1.0
membrane = gmsh.model.occ.addDisk(0, 0, 0, R, R)
gmsh.model.occ.synchronize()

# Mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(2)

# Import into DOLFINx
mesh, cell_tags, facet_tags = model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)
gmsh.finalize()

print(f"Circular mesh: {mesh.topology.index_map(2).size_global} cells")
""")
```

> **Namespace persistence**: Variables defined in `run_custom_code` persist across calls.
> You can split complex workflows into multiple calls without re-importing or re-defining objects.
> Session-registered objects (meshes, spaces, functions) are always injected fresh and override stale names.

### 2. Create Function Space on the Imported Mesh

After the mesh is created via `run_custom_code`, continue with standard tools if the mesh is registered in the session. Otherwise, complete the solve within `run_custom_code`:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI

# Assumes mesh is already created (from previous run_custom_code)
V = fem.functionspace(mesh, ("Lagrange", 1))

# Gaussian load parameters
beta = 8.0
R0 = 0.6
x = ufl.SpatialCoordinate(mesh)
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))

# Weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx

# Zero Dirichlet BC on boundary
tdim = mesh.topology.dim
fdim = tdim - 1
boundary_facets = dolfinx.mesh.exterior_facets(mesh)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(fem.Function(V), dofs)

# Solve
problem = LinearProblem(a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="s_")
w = problem.solve()

print(f"Max deflection: {w.x.array.max():.6e}")
print(f"Min deflection: {w.x.array.min():.6e}")
""")
```

### 3. Post-Process

Export the solution for visualization:
```
export_solution(solution_name="w", filename="membrane_deflection", format="vtk")
```

## Key Concepts

- **Gmsh integration**: DOLFINx can import meshes from Gmsh via `dolfinx.io.gmsh`
- **Gaussian load**: Models localized pressure on the membrane
- **Circular domain**: Requires non-standard mesh generation (Gmsh, not unit square)
- **Warp plot**: Deformation visualization shows the membrane shape under load

## Parameters to Explore

| Parameter | Effect |
|---|---|
| `beta` (load width) | Larger = more localized load |
| `R0` (load center) | Distance from center of membrane |
| `R` (radius) | Domain size |
| Mesh density | Finer mesh near load center improves accuracy |
