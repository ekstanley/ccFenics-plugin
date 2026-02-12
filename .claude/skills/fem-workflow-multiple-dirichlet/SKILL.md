---
name: fem-workflow-multiple-dirichlet
description: |
  Guides the user through applying multiple Dirichlet BCs with different values on different boundaries using DOLFINx MCP tools.
  Use when the user asks about multiple Dirichlet conditions, different BC values on different boundaries,
  inhomogeneous BCs, or corner handling.
---

# Multiple Dirichlet BCs Workflow (Tutorial Ch3.2)

Apply **different Dirichlet values on different boundary segments**.

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="multi_bc_mesh")
```

### 2. Create Function Space

```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 3. Define Source Term

```
set_material_properties(name="f", value="0.0")
```

### 4. Define Variational Form

```
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx",
    linear="f * v * dx",
    name="laplace"
)
```

### 5. Apply Multiple Dirichlet BCs

Each call to `apply_boundary_condition` adds a separate BC:

```
apply_boundary_condition(value="0.0", boundary="np.isclose(x[0], 0.0)", name="bc_left")
apply_boundary_condition(value="1.0", boundary="np.isclose(x[0], 1.0)", name="bc_right")
apply_boundary_condition(value="sin(pi*x[0])", boundary="np.isclose(x[1], 0.0)", name="bc_bottom")
apply_boundary_condition(value="0.0", boundary="np.isclose(x[1], 1.0)", name="bc_top")
```

### 6. Solve

```
solve(solver_type="direct", solution_name="u_h")
```

## Corner Handling

When two boundaries meet at a corner, the DOF at the corner will be set by whichever BC is applied last. For consistent corners:

- Ensure corner values are compatible between the two BCs
- Or apply one BC with priority (the last one wins)

Example: If left wall has u=0 and bottom wall has u=sin(pi*x), then at corner (0,0):
- Left wall says u=0
- Bottom wall says u=sin(0)=0
- No conflict (both give 0)

## Expression-Based BCs

The `value` parameter accepts numpy expressions:

| Example | Description |
|---|---|
| `"0.0"` | Constant zero |
| `"1.0"` | Constant one |
| `"x[0]"` | Linear in x |
| `"sin(pi*x[0])"` | Sinusoidal |
| `"x[0]*(1-x[0])"` | Parabolic profile |
| `"exp(-10*(x[0]-0.5)**2)"` | Gaussian bump |

The `boundary` parameter accepts numpy boolean expressions:

| Example | Description |
|---|---|
| `"np.isclose(x[0], 0.0)"` | Left boundary |
| `"np.isclose(x[1], 1.0)"` | Top boundary |
| `"x[0] < 1e-14"` | Alternative left boundary |
| `"(x[0] < 1e-14) \| (x[0] > 1-1e-14)"` | Left OR right |
| `"True"` | Entire boundary |

## Feasibility

100% achievable with existing tools. Just use multiple `apply_boundary_condition` calls.
