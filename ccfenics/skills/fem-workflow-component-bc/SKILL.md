---
name: fem-workflow-component-bc
description: |
  Guides the user through applying component-wise boundary conditions on vector function spaces using DOLFINx MCP tools.
  Use when the user asks about component-wise BCs, fixing one component, sub_space,
  vector BC on single component, or roller boundary conditions.
---

# Component-Wise Boundary Conditions Workflow (Tutorial Ch3.5)

Apply **Dirichlet BCs on individual components** of a vector field (e.g., fix x-displacement but allow y-displacement).

## Key Principle

For vector function spaces, the `apply_boundary_condition` tool supports a `sub_space` parameter that targets individual components:
- `sub_space=0` -> x-component
- `sub_space=1` -> y-component
- `sub_space=2` -> z-component (3D)

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_mesh(shape="rectangle", nx=64, ny=16, x_min=0, x_max=4, y_min=0, y_max=1, name="beam")
```

### 2. Create Vector Function Space

```
create_function_space(family="Lagrange", degree=1, shape=[2], name="V")
```

### 3. Define Material Properties

```
set_material_properties(name="E", value="1e5")
set_material_properties(name="nu", value="0.3")
set_material_properties(name="lambda_", value="E*nu/((1+nu)*(1-2*nu))")
set_material_properties(name="mu", value="E/(2*(1+nu))")
```

### 4. Define Variational Form (Elasticity)

```
define_variational_form(
    bilinear="inner(lambda_*nabla_div(u)*Identity(2) + 2*mu*sym(grad(u)), sym(grad(v))) * dx",
    linear="dot(as_vector([0, -1.0]), v) * dx",
    name="elasticity"
)
```

### 5. Apply Component-Wise BCs

**Clamped left face** (both components zero):
```
apply_boundary_condition(value="0.0", boundary="x[0] < 1e-14", sub_space=0, name="clamp_x")
apply_boundary_condition(value="0.0", boundary="x[0] < 1e-14", sub_space=1, name="clamp_y")
```

**Roller on bottom** (fix y-component, allow x-sliding):
```
apply_boundary_condition(value="0.0", boundary="x[1] < 1e-14", sub_space=1, name="roller")
```

**Symmetry on centerline** (fix normal component):
```
apply_boundary_condition(value="0.0", boundary="np.isclose(x[1], 0.5)", sub_space=1, name="symmetry")
```

### 6. Solve

```
solve(solver_type="direct", solution_name="displacement")
```

## Common BC Patterns for Elasticity

| Physical BC | Components Fixed | Tool Call |
|---|---|---|
| Clamped | All | `sub_space=0` + `sub_space=1` (+ `sub_space=2` in 3D) |
| Roller (x-axis) | y only | `sub_space=1` with value 0 |
| Roller (y-axis) | x only | `sub_space=0` with value 0 |
| Symmetry (x-normal) | x only | `sub_space=0` with value 0 |
| Prescribed displacement | One component | `sub_space=i` with non-zero value |
| Pin support | All at a point | All sub_spaces at point boundary |

## How `sub_space` Works

The tool uses `.sub(i).collapse()` internally:
1. `V.sub(i)` returns a view of the i-th component subspace
2. `.collapse()` creates a standalone space for that component
3. DOFs are located on the collapsed space
4. The BC is applied to the full vector space at those DOF locations

## Feasibility

100% achievable with existing tools. The `sub_space` parameter in `apply_boundary_condition` handles component selection.
