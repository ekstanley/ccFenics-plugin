---
name: fem-workflow-axisymmetric
description: |
  Guides the user through solving axisymmetric PDE problems using DOLFINx MCP tools.
  Use when the user asks about axisymmetric domains, cylindrical coordinates, r-z plane,
  bodies of revolution, pipes, cylinders, axially symmetric problems, or radial weighting.
---

# Axisymmetric Domain Workflow (Official Demo)

Solve PDEs on **axisymmetric domains** (pipes, cylinders, shells) using a 2D r-z cross-section.

## Concept

For problems with rotational symmetry about an axis, the 3D problem reduces to a 2D problem in the (r, z) plane. The volume element in cylindrical coordinates is `2*pi*r*dr*dz`, so the weak form acquires an `r = x[0]` weighting factor (the `2*pi` cancels from both sides).

**Key modification**: Every integrand in `dx` gets multiplied by `x[0]` (the radial coordinate).

## Step-by-Step Tool Sequence

### Example: Axisymmetric Poisson

Solve `-1/r * d/dr(r * du/dr) - d^2u/dz^2 = f` on a rectangular r-z domain.

### 1. Create the Mesh (r-z cross-section)

The mesh represents the (r, z) cross-section. Use `x[0]` = r, `x[1]` = z.

**Important**: Offset the mesh away from r=0 to avoid the axis singularity, or ensure no integration occurs at r=0.

```
create_mesh(name="mesh", shape="rectangle", nx=16, ny=16,
            dimensions={"width": 1.0, "height": 1.0})
```

For an annular domain (r_inner to r_outer), shift the mesh origin:

```
create_custom_mesh(name="mesh",
    points=[[r_inner, 0], [r_outer, 0], [r_outer, H], [r_inner, H]],
    cell_type="triangle", n_refinements=3)
```

### 2. Create Function Space

```
create_function_space(name="V", family="Lagrange", degree=1)
```

### 3. Set Source Term

```
set_material_properties(name="f", value="1.0 + 0*x[0]")
```

### 4. Define Axisymmetric Variational Form

The standard Poisson weak form `inner(grad(u), grad(v))*dx` becomes `x[0]*inner(grad(u), grad(v))*dx` in axisymmetric coordinates:

```
define_variational_form(
    bilinear="x[0] * inner(grad(u), grad(v)) * dx",
    linear="x[0] * f * v * dx"
)
```

### 5. Apply Boundary Conditions

```
apply_boundary_condition(value=0.0, boundary="True")
```

### 6. Solve

```
solve(solver_type="direct")
```

## Axisymmetric Heat Equation

For time-dependent axisymmetric heat conduction:

```
define_variational_form(
    bilinear="x[0] * (u*v + dt*inner(grad(u), grad(v))) * dx",
    linear="x[0] * (u_n + dt*f) * v * dx"
)
```

## Axisymmetric Elasticity

For axisymmetric elasticity (displacement in r-z plane):

```
define_variational_form(
    bilinear="x[0] * inner(lambda_*nabla_div(u)*Identity(2) + 2*mu*sym(grad(u)), sym(grad(v))) * dx",
    linear="x[0] * inner(body_force, v) * dx"
)
```

**Note**: Full axisymmetric elasticity also includes hoop stress terms (`u_r/r`). For problems where hoop stress is significant, use `run_custom_code` with the full strain tensor in cylindrical coordinates.

> **Namespace persistence**: Variables defined in `run_custom_code` persist across calls.
> You can split complex workflows into multiple calls without re-importing or re-defining objects.
> Session-registered objects (meshes, spaces, functions) are always injected fresh and override stale names.

## Key Concepts

- **r = x[0], z = x[1]**: The mesh coordinates map to cylindrical (r, z)
- **Radial weighting**: Multiply all integrands by `x[0]` to account for the cylindrical volume element
- **Axis singularity**: At r=0, the weighting `x[0]=0` can cause numerical issues. Offset the mesh or ensure natural BCs at the axis
- **Symmetry BC**: On the axis of symmetry (r=0), the natural BC is du/dr=0 (homogeneous Neumann), which is automatically satisfied
- **No new tools needed**: All existing tools work; only the form expressions change

## Common Pitfalls

- Forgetting the `x[0]` weighting factor in either bilinear or linear form
- Placing mesh nodes at r=0 and applying Dirichlet BCs there (creates zero-weighted equations)
- Not recognizing that the "width" dimension of a rectangle mesh is the radial direction
- For elasticity: neglecting hoop strain terms for thick-walled problems
