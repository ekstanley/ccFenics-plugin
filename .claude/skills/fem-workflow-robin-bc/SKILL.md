---
name: fem-workflow-robin-bc
description: |
  Guides the user through applying Robin boundary conditions using DOLFINx MCP tools.
  Use when the user asks about Robin BCs, impedance BCs, Newton cooling,
  convective heat transfer BCs, or mixed BC types combining Dirichlet and Robin.
---

# Robin Boundary Conditions Workflow (Tutorial Ch3.4)

Apply **Robin BCs**: `du/dn + alpha*u = alpha*s` on part of the boundary.

Robin BCs model convective heat transfer (Newton's law of cooling), impedance conditions, and absorbing boundaries.

## Key Principle

Robin BCs appear as **additional terms in both the bilinear and linear forms** -- no `apply_boundary_condition` call needed for the Robin part.

Robin condition: `du/dn + alpha*u = alpha*s`

In the variational form:
- **Bilinear**: add `alpha * u * v * ds(robin_tag)`
- **Linear**: add `alpha * s * v * ds(robin_tag)`

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="robin_mesh")
```

### 2. Mark Boundaries

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

### 4. Define Material Properties

```
set_material_properties(name="f", value="1.0")
set_material_properties(name="alpha", value="10.0")
set_material_properties(name="s", value="0.5")
```

### 5. Define Variational Form with Robin Terms

Dirichlet on left (tag 1), Robin on right (tag 2), zero Neumann on top/bottom:

```
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx + alpha * u * v * ds(2)",
    linear="f * v * dx + alpha * s * v * ds(2)",
    name="robin_problem"
)
```

### 6. Apply Dirichlet BC (Left Only)

```
apply_boundary_condition(value="1.0", boundary="np.isclose(x[0], 0.0)", name="bc_left")
```

### 7. Solve

```
solve(solver_type="direct", solution_name="u_h")
```

## Physical Interpretation

For heat transfer:
- `alpha` = heat transfer coefficient (W/(m^2*K))
- `s` = ambient temperature
- Robin BC: `-k*du/dn = alpha*(u - s)` models convection at the surface
- Large `alpha`: approaches Dirichlet BC (u -> s)
- Small `alpha`: approaches Neumann BC (du/dn -> 0, insulated)

## Combining All Three BC Types

A single problem can have Dirichlet, Neumann, and Robin BCs simultaneously:

```
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx + alpha * u * v * ds(2)",
    linear="f * v * dx + g * v * ds(3) + alpha * s * v * ds(2)",
    name="all_bc_types"
)
```

- Tag 1: Dirichlet (via `apply_boundary_condition`)
- Tag 2: Robin (in bilinear + linear forms)
- Tag 3: Neumann (in linear form only)
- Tag 4: Zero Neumann (no terms needed -- natural BC)

## Feasibility

100% achievable with existing tools. Robin BCs are purely variational -- they need no special tool support.
