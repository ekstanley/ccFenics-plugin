---
name: fem-workflow-stokes
description: |
  Guides the user through solving a Stokes flow problem using DOLFINx MCP tools.
  Use when the user asks about Stokes flow, creeping flow, incompressible viscous flow,
  low Reynolds number flow, or mixed finite element formulations.
---

# Stokes Flow Workflow

Solve the Stokes equations (saddle-point system):
- **Momentum**: -div(2*mu*epsilon(u)) + grad(p) = f
- **Incompressibility**: div(u) = 0

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_mesh(shape="rectangle", nx=32, ny=32, x_min=0, x_max=1, y_min=0, y_max=1, name="channel")
```

For channel flow, use a rectangular domain with higher aspect ratio:
```
create_mesh(shape="rectangle", nx=64, ny=16, x_min=0, x_max=4, y_min=0, y_max=1, name="channel")
```

### 2. Create Function Spaces (Taylor-Hood P2/P1)

Stokes requires a **mixed function space** satisfying the inf-sup (LBB) condition:

```
create_function_space(family="Lagrange", degree=2, shape=[2], name="V")
create_function_space(family="Lagrange", degree=1, name="Q")
create_mixed_space(space_names=["V", "Q"], name="W")
```

**Taylor-Hood P2/P1** is the standard stable pair:
- Velocity: P2 vector (degree=2, shape=[gdim])
- Pressure: P1 scalar (degree=1)

### 3. Define Material Properties

```
set_material_properties(name="mu", value="1.0")
```

For the body force (if any):
```
set_material_properties(name="f", value="(0.0, 0.0)")
```

### 4. Define Variational Form

The Stokes weak form (monolithic):

```
define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx",
    linear="inner(f, v)*dx",
    name="stokes"
)
```

Where:
- `(u, p)` are trial functions (velocity, pressure)
- `(v, q)` are test functions
- First term: viscous stress (simplified for mu=1)
- Second term: pressure-velocity coupling
- Third term: incompressibility constraint

For variable viscosity, replace first term with `2*mu*inner(sym(grad(u)), sym(grad(v)))*dx`.

### 5. Apply Boundary Conditions

Typical Stokes BCs:

- **No-slip walls** (zero velocity):
  ```
  apply_boundary_condition(value="(0.0, 0.0)", boundary="x[1] < 1e-14 or x[1] > 1.0 - 1e-14", sub_space=0, name="noslip")
  ```
- **Inlet velocity** (parabolic profile):
  ```
  apply_boundary_condition(value="(4.0*x[1]*(1.0-x[1]), 0.0)", boundary="x[0] < 1e-14", sub_space=0, name="inlet")
  ```
- **Outlet**: Natural BC (do-nothing) -- no tool call needed, or set pressure:
  ```
  apply_boundary_condition(value="0.0", boundary="x[0] > 4.0 - 1e-14", sub_space=1, name="outlet_p")
  ```

The `sub_space` parameter indexes into the mixed space: 0=velocity, 1=pressure.

### 6. Solve

```
solve(form_name="stokes", solver_type="lu", name="stokes_solution")
```

For large problems, use iterative solvers with block preconditioning:
- `solver_type="gmres"` (not CG -- Stokes is indefinite)
- Direct LU is reliable for moderate sizes

### 7. Post-Process

```
export_solution(solution_name="stokes_solution", filename="stokes_flow", format="vtk")
```

Compute velocity magnitude, pressure contours:
```
compute_functionals(solution_name="stokes_solution", functional="l2_norm")
```

## Key Concepts

- **Saddle-point system**: The Stokes system is indefinite (not SPD). CG will fail; use GMRES or direct LU.
- **Inf-sup stability**: The velocity-pressure pair must satisfy the LBB condition. P2/P1 (Taylor-Hood) is stable. P1/P1 is NOT stable (needs stabilization).
- **Pressure nullspace**: For enclosed cavities (all Dirichlet on velocity, no pressure BC), pressure is determined only up to a constant. Pin pressure at one point or use a zero-mean constraint.
- **Reynolds number**: Stokes assumes Re << 1. For Re > 1, use Navier-Stokes (nonlinear convection term).

## Stable Element Pairs for Stokes

| Velocity | Pressure | Stable? | Notes |
|---|---|---|---|
| P2 | P1 | Yes | Taylor-Hood, standard choice |
| P2 + bubble | P1 | Yes | MINI element |
| P1 | P1 | No | Needs SUPG/PSPG stabilization |
| P2 | P0 | No | Fails inf-sup |
