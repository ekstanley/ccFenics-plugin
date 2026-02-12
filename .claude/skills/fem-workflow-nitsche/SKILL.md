---
name: fem-workflow-nitsche
description: |
  Guides the user through applying Dirichlet BCs via Nitsche's method using DOLFINx MCP tools.
  Use when the user asks about Nitsche method, weak Dirichlet BCs, penalty boundary conditions,
  or weakly imposed boundary conditions.
---

# Nitsche Method Workflow (Tutorial Ch1.3)

Apply Dirichlet boundary conditions **weakly** via penalty terms in the variational form, instead of using `apply_boundary_condition`.

## When to Use Nitsche

- When strong Dirichlet BCs are difficult (e.g., non-matching meshes)
- For mixed element formulations where standard BCs are problematic
- When you want all BCs embedded in the variational form

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="nitsche_mesh")
```

### 2. Create Function Space

```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 3. Define Source Term and Exact Solution

```
set_material_properties(name="f", value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])")
```

### 4. Define Variational Form with Nitsche Terms

The Nitsche form replaces `apply_boundary_condition` by adding penalty terms.
For **-div(grad(u)) = f** with **u = 0** on the boundary:

```
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx - inner(dot(grad(u), n), v) * ds - inner(u, dot(grad(v), n)) * ds + 10 * 1 / h * inner(u, v) * ds",
    linear="f * v * dx",
    name="nitsche_poisson"
)
```

**Do NOT call `apply_boundary_condition`** -- the BC is already in the form.

Key symbols available in UFL namespace:
- `n` = `FacetNormal(mesh)` -- outward facet normal
- `h` = `CellDiameter(mesh)` -- cell diameter for scaling
- `ds` = boundary integral measure

### 5. Solve (No BCs needed)

```
solve(solver_type="direct", solution_name="u_h")
```

### 6. Verify

```
compute_error(exact_solution="sin(pi*x[0])*sin(pi*x[1])", norm_type="L2")
```

## Penalty Parameter

The penalty `alpha/h` must be large enough for coercivity:
- `alpha ~ 10 * degree^2` is a safe starting point
- Too small: solution is inaccurate near boundaries
- Too large: ill-conditioning

For degree p on the space:
- P1: `alpha = 10`
- P2: `alpha = 40`
- P3: `alpha = 90`

## Nitsche Form Structure

For **u = g** on the boundary (non-homogeneous):

**Bilinear form** (terms involving u and v):
```
inner(grad(u), grad(v)) * dx
- inner(dot(grad(u), n), v) * ds       # consistency
- inner(u, dot(grad(v), n)) * ds       # symmetry
+ (alpha/h) * inner(u, v) * ds         # penalty
```

**Linear form** (terms involving v only):
```
f * v * dx
- inner(g, dot(grad(v), n)) * ds       # symmetry (g part)
+ (alpha/h) * inner(g, v) * ds         # penalty (g part)
```

Where `g` is the Dirichlet boundary value (define as material property).

## Feasibility

100% achievable with existing tools: `create_unit_square`, `create_function_space`, `set_material_properties`, `define_variational_form`, `solve`. No `apply_boundary_condition` needed.
