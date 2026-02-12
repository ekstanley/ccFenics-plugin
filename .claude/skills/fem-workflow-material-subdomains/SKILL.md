---
name: fem-workflow-material-subdomains
description: |
  Guides the user through setting up problems with different materials in different subdomains using DOLFINx MCP tools.
  Use when the user asks about material subdomains, piecewise coefficients, heterogeneous materials,
  multi-material problems, or subdomain-dependent properties.
---

# Material Subdomains Workflow (Tutorial Ch3.3)

Define **different material properties in different regions** of the domain.

## Approaches

### Approach 1: UFL `conditional` (Simple Cases)

For simple subdomain geometries, use UFL's `conditional` function directly in the variational form:

```
set_material_properties(name="kappa_val", value="1.0")
```

Then in the form, use conditional:
```
define_variational_form(
    bilinear="conditional(lt(x[0], 0.5), 1.0, 10.0) * inner(grad(u), grad(v)) * dx",
    linear="f * v * dx",
    name="heterogeneous"
)
```

UFL conditional operators available in the namespace:
- `conditional(condition, true_val, false_val)` -- ternary operator
- `lt(a, b)`, `gt(a, b)`, `le(a, b)`, `ge(a, b)` -- comparisons

### Approach 2: Cell Tags + `dx(tag)` (Complex Geometries)

For meshes imported from Gmsh with cell tags:

```
manage_mesh_tags(action="create", markers=[
    {"tag": 1, "condition": "x[0] < 0.5", "dimension": 2},
    {"tag": 2, "condition": "x[0] >= 0.5", "dimension": 2}
], name="material_tags")
```

Then integrate separately over each subdomain:
```
define_variational_form(
    bilinear="kappa1 * inner(grad(u), grad(v)) * dx(1) + kappa2 * inner(grad(u), grad(v)) * dx(2)",
    linear="f * v * dx",
    name="multi_material"
)
```

### Approach 3: DG0 Piecewise Function (via `run_custom_code`)

For the most general case, create a DG0 function with different values per cell:

```python
run_custom_code(code="""
from dolfinx import fem
import numpy as np

# DG0 space for piecewise constant material
Q = fem.functionspace(mesh, ("DG", 0))
kappa = fem.Function(Q)

# Assign different values based on cell midpoints
cells = mesh.topology.index_map(2).size_local
midpoints = mesh.geometry.x[:cells]  # approximate

for i in range(cells):
    if midpoints[i, 0] < 0.5:
        kappa.x.array[i] = 1.0
    else:
        kappa.x.array[i] = 10.0
""")
```

## Step-by-Step (Approach 1 -- Simplest)

### 1. Create Mesh
```
create_unit_square(nx=32, ny=32, name="multi_mat_mesh")
```

### 2. Create Function Space
```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 3. Define Source
```
set_material_properties(name="f", value="1.0")
```

### 4. Define Form with Conditional Coefficient
```
define_variational_form(
    bilinear="conditional(lt(x[0], 0.5), 1.0, 10.0) * inner(grad(u), grad(v)) * dx",
    linear="f * v * dx",
    name="multi_material"
)
```

### 5. Apply BCs and Solve
```
apply_boundary_condition(value="0.0", boundary="True", name="bc_zero")
solve(solver_type="direct", solution_name="u_h")
```

## Physical Examples

| Problem | Material 1 | Material 2 | Interface |
|---|---|---|---|
| Composite bar | Steel (k=50) | Copper (k=400) | x = 0.5 |
| Layered soil | Clay (k=0.5) | Sand (k=5.0) | y = 0.3 |
| Insulated region | Normal (k=1) | Insulator (k=0.01) | Circle r=0.2 |
