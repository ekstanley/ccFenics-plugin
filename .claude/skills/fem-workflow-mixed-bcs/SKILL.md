---
name: fem-workflow-mixed-bcs
description: |
  Guides the user through setting up mixed boundary conditions (Dirichlet + Neumann) using DOLFINx MCP tools.
  Use when the user asks about mixed BCs, combining Neumann and Dirichlet, natural boundary conditions,
  or traction boundary conditions.
---

# Mixed Boundary Conditions Workflow (Tutorial Ch3.1)

Set up a problem with **Dirichlet BCs on some boundaries** and **Neumann BCs on others**.

## Key Principle

- **Dirichlet BCs**: Applied explicitly via `apply_boundary_condition` (essential BCs)
- **Neumann BCs**: Included in the linear form as surface integrals (natural BCs)

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="mixed_mesh")
```

### 2. Mark Boundaries with Tags

Use `mark_boundaries` to assign integer tags to different boundary segments:

```
mark_boundaries(
    markers=[
        {"tag": 1, "condition": "np.isclose(x[0], 0.0)"},
        {"tag": 2, "condition": "np.isclose(x[0], 1.0)"},
        {"tag": 3, "condition": "np.isclose(x[1], 0.0)"},
        {"tag": 4, "condition": "np.isclose(x[1], 1.0)"}
    ],
    name="boundary_tags"
)
```

### 3. Create Function Space

```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 4. Define Source Term

```
set_material_properties(name="f", value="1.0")
```

### 5. Define Variational Form with Neumann Term

Dirichlet on left (tag 1), Neumann on right (tag 2), zero Neumann elsewhere:

```
set_material_properties(name="g", value="0.5")
```

```
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx",
    linear="f * v * dx + g * v * ds(2)",
    name="mixed_bc_problem"
)
```

The `ds(2)` integrates over facets tagged with 2 (right boundary). The Neumann condition `du/dn = g` appears naturally in the linear form.

### 6. Apply Dirichlet BC (Left Boundary Only)

```
apply_boundary_condition(value="0.0", boundary="np.isclose(x[0], 0.0)", name="bc_left")
```

### 7. Solve

```
solve(solver_type="direct", solution_name="u_h")
```

### 8. Post-Process

```
export_solution(solution_name="u_h", filename="mixed_bc_solution", format="vtk")
```

## Physical Explanation

- **Dirichlet**: "The temperature is fixed at 0 on the left wall"
- **Neumann**: "There is a heat flux of g=0.5 coming in from the right wall"
- **Natural Neumann**: Boundaries without explicit BCs get du/dn = 0 (insulated)

## Integration Measures with Tags

| Expression | Meaning |
|---|---|
| `f * v * dx` | Volume integral over entire domain |
| `g * v * ds` | Surface integral over ALL boundaries |
| `g * v * ds(2)` | Surface integral over tag-2 boundaries only |
| `g * v * ds(2) + h * v * ds(4)` | Different values on different boundaries |

## Feasibility

100% achievable with existing tools: `create_unit_square`, `mark_boundaries`, `create_function_space`, `set_material_properties`, `define_variational_form`, `apply_boundary_condition`, `solve`.
