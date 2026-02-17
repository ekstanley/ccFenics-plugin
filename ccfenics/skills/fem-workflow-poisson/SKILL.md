---
name: fem-workflow-poisson
description: |
  Guides the user through solving a Poisson equation (-div(grad(u)) = f) using DOLFINx MCP tools.
  Use when the user asks to solve a Poisson problem, Laplace equation, heat diffusion,
  scalar diffusion, or set up a basic elliptic PDE.
---

# Poisson Equation Workflow

Solve: **-div(grad(u)) = f** on Omega, with u = g on dOmega.

## Step-by-Step Tool Sequence

### 1. Create the Mesh

Ask the user about domain shape and resolution.

- For [0,1]^2: `create_unit_square(nx=32, ny=32, name="poisson_mesh")`
- For custom rectangles: `create_mesh(shape="rectangle", nx=32, ny=32, x_min=0, x_max=2, y_min=0, y_max=1, name="poisson_mesh")`
- For 3D boxes: `create_mesh(shape="box", nx=16, ny=16, nz=16, name="poisson_mesh")`

Default recommendation: 32x32 for development, 64x64+ for production.

### 2. Create Function Space

- `create_function_space(family="Lagrange", degree=1, name="V")`
- P1 is simplest; P2 gives better accuracy (O(h^3) vs O(h^2) in L2 norm).

### 3. Define Material Properties / Source Term

Ask the user for f(x,y). Use `set_material_properties`:

```
set_material_properties(name="f", value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])")
```

This is a numpy expression evaluated over coordinate arrays. Available symbols: `x[0]`, `x[1]`, `x[2]`, `pi`, `sin`, `cos`, `exp`, `sqrt`, `np`.

### 4. Define Variational Form

The weak form of -div(grad(u)) = f:

```
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx",
    linear="f * v * dx",
    name="poisson"
)
```

This uses UFL symbolic expressions. Available operators: `grad`, `div`, `inner`, `dot`, `dx`, `ds`, plus any materials/functions defined in the session.

### 5. Apply Boundary Conditions

Ask which boundaries and what values:

- Homogeneous Dirichlet (u=0 everywhere): `apply_boundary_condition(value="0.0", boundary="True", name="bc_zero")`
- Non-homogeneous: `apply_boundary_condition(value="sin(pi*x[0])", boundary="x[1] < 1e-14", name="bc_bottom")`
- The `boundary` parameter is a numpy expression returning a boolean mask.
- The `value` parameter is a numpy expression for the BC values.

### 6. Solve

```
solve(form_name="poisson", solver_type="lu", name="u_h")
```

- Use `solver_type="lu"` (direct) for small-medium problems.
- For large problems: `solver_type="cg"` with `preconditioner="ilu"`.

### 7. Post-Process

Compute error if exact solution is known:
```
compute_error(solution_name="u_h", exact_solution="sin(pi*x[0])*sin(pi*x[1])", error_type="L2")
```

Export for visualization:
```
export_solution(solution_name="u_h", filename="poisson_solution", format="vtk")
```

## Physical Explanation

At each step, explain what is happening:
- **Mesh**: Discretizes the continuous domain into triangles/tetrahedra
- **Function space**: Defines the polynomial approximation (piecewise linear/quadratic)
- **Weak form**: Multiplies by test function v and integrates by parts, converting -div(grad(u))=f into inner(grad(u),grad(v))dx = f*v*dx
- **BCs**: Pin solution values on the boundary to make the problem well-posed
- **Solver**: Assembles the linear system Ku=F and solves it

## Common Exact Solutions for Testing

| Source term f | Exact solution u | Domain |
|---|---|---|
| `2*pi**2*sin(pi*x[0])*sin(pi*x[1])` | `sin(pi*x[0])*sin(pi*x[1])` | [0,1]^2, u=0 on dOmega |
| `-6` | `1 + x[0]**2 + 2*x[1]**2` | [0,1]^2, u=1+x^2+2y^2 on dOmega |
| `0` (Laplace) | `x[0]` | [0,1]^2, u=x on left/right |
