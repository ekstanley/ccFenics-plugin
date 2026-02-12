---
name: fem-workflow-elasticity
description: |
  Guides the user through solving a linear elasticity problem using DOLFINx MCP tools.
  Use when the user asks about linear elasticity, elastic deformation, stress-strain analysis,
  structural mechanics, or displacement computation.
---

# Linear Elasticity Workflow

Solve: **-div(sigma(u)) = f** on Omega, where sigma = lambda*tr(epsilon)*I + 2*mu*epsilon and epsilon = sym(grad(u)).

## Step-by-Step Tool Sequence

### 1. Create the Mesh

Ask about geometry and resolution.

- 2D beam: `create_mesh(shape="rectangle", nx=64, ny=16, x_min=0, x_max=4, y_min=0, y_max=1, name="beam")`
- 3D block: `create_mesh(shape="box", nx=16, ny=16, nz=16, name="block")`

### 2. Create Vector Function Space

Elasticity requires a **vector** function space (displacement has gdim components):

- 2D: `create_function_space(family="Lagrange", degree=1, shape=[2], name="V")`
- 3D: `create_function_space(family="Lagrange", degree=1, shape=[3], name="V")`

The `shape` parameter makes this a vector space instead of scalar.

### 3. Define Material Properties

Ask for Young's modulus (E) and Poisson's ratio (nu). Compute Lame parameters:

```
set_material_properties(name="lambda_", value="E*nu/((1+nu)*(1-2*nu))")
set_material_properties(name="mu", value="E/(2*(1+nu))")
```

Where E and nu must already be defined:
```
set_material_properties(name="E", value="1e5")
set_material_properties(name="nu", value="0.3")
```

Typical values: Steel E=210 GPa, nu=0.3. Rubber E=0.01 GPa, nu=0.49.

### 4. Define Variational Form

The elasticity weak form uses symmetric gradient and the stress tensor:

```
define_variational_form(
    bilinear="inner(lambda_*nabla_div(u)*Identity(2) + 2*mu*sym(grad(u)), sym(grad(v))) * dx",
    linear="dot(f, v) * dx",
    name="elasticity"
)
```

For 3D, replace `Identity(2)` with `Identity(3)`.

Key UFL operators:
- `sym(grad(u))` = symmetric gradient (strain tensor epsilon)
- `nabla_div(u)` = divergence of vector field (tr(epsilon))
- `Identity(d)` = d x d identity matrix
- `inner(sigma, epsilon)` = double contraction of tensors

### 5. Apply Boundary Conditions

Typical elasticity BCs:

- **Clamped left face** (zero displacement):
  ```
  apply_boundary_condition(value="(0.0, 0.0)", boundary="x[0] < 1e-14", name="clamp")
  ```
- **Applied traction** on right face: Include in the linear form as `dot(t, v) * ds`
  or use a Neumann BC (natural BC -- no explicit tool call needed, just include in `linear`).

### 6. Solve

```
solve(form_name="elasticity", solver_type="lu", name="displacement")
```

Elasticity systems are symmetric positive definite (SPD) with proper BCs. For iterative: `solver_type="cg"`, `preconditioner="ilu"`.

### 7. Post-Process

Export displacement:
```
export_solution(solution_name="displacement", filename="elasticity_result", format="vtk")
```

Compute functionals (e.g., max displacement):
```
compute_functionals(solution_name="displacement", functional="max_component")
```

## Important Notes

- **Plane stress vs plane strain**: 2D elasticity assumes plane strain by default (Lame parameters as given). For plane stress, adjust lambda.
- **Nearly incompressible**: nu > 0.49 causes locking with P1 elements. Use P2 or mixed formulations.
- **Body forces**: Gravity = `(0, -rho*g)` in 2D. Define rho and g as material properties first.
- **Units**: Be consistent. SI (Pa, m, kg) is recommended.
