---
name: fem-workflow-heat-equation
description: |
  Guides the user through solving the heat equation (time-dependent diffusion) using DOLFINx MCP tools.
  Use when the user asks about heat equation, diffusion, time-dependent problems,
  transient simulations, backward Euler, or temperature evolution.
---

# Heat Equation Workflow (Tutorial Ch2.1-2.2)

Solve: **du/dt - div(grad(u)) = f** on Omega x (0, T], with initial condition u(x,0) = u_0(x).

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="heat_mesh")
```

### 2. Create Function Space

```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 3. Define Initial Condition

Interpolate the initial condition into the function space:

```
interpolate(name="u_n", expression="exp(-5*((x[0]-0.5)**2 + (x[1]-0.5)**2))", function_space="V")
```

This creates `u_n` in the session -- it represents the solution at the previous time step.

### 4. Set Material Properties

Define the time step and source term:

```
set_material_properties(name="dt", value="0.01")
set_material_properties(name="f", value="0.0")
```

### 5. Define Variational Form (Backward Euler)

The backward Euler discretization gives: `(u - u_n)/dt - div(grad(u)) = f`

Weak form:
```
define_variational_form(
    bilinear="u * v * dx + dt * inner(grad(u), grad(v)) * dx",
    linear="(u_n + dt * f) * v * dx",
    name="heat"
)
```

Here `u_n` is the previous solution (already in session from the interpolate step), `dt` is the time step constant, and `f` is the source term.

### 6. Apply Boundary Conditions

```
apply_boundary_condition(value="0.0", boundary="True", name="bc_zero")
```

### 7. Solve Time-Dependent Problem

```
solve_time_dependent(
    t_end=0.5,
    dt=0.01,
    time_scheme="backward_euler",
    output_times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    solver_type="direct",
    solution_name="temperature"
)
```

### 8. Post-Process

```
export_solution(solution_name="temperature", filename="heat_solution", format="vtk")
```

## Physical Explanation

At each step, explain:
- **Initial condition**: Starting temperature distribution (e.g., Gaussian bump)
- **Time discretization**: Backward Euler replaces du/dt with (u^{n+1} - u^n)/dt
- **Stability**: Backward Euler is unconditionally stable (any dt works, but accuracy depends on dt)
- **Dissipation**: Heat equation smooths out the initial condition over time

## Common Configurations

| Problem | Initial Condition | Source | BCs |
|---|---|---|---|
| Gaussian diffusion | `exp(-5*((x[0]-0.5)**2+(x[1]-0.5)**2))` | `0.0` | u=0 on boundary |
| Heated plate | `0.0` | `1.0` | u=0 on boundary |
| Manufactured | `sin(pi*x[0])*sin(pi*x[1])` | `2*pi**2*sin(pi*x[0])*sin(pi*x[1])` (steady-state source) | u=0 on boundary |

## Time Step Selection

- **Stability**: Backward Euler is unconditionally stable, so dt is limited by accuracy, not stability
- **Accuracy**: dt ~ h^2 gives temporal error matching spatial error for P1 elements
- **Rule of thumb**: Start with dt = 0.01 for [0,1]^2 domain with nx=32

## Feasibility

100% achievable with existing tools: `create_unit_square`, `create_function_space`, `interpolate`, `set_material_properties`, `define_variational_form`, `apply_boundary_condition`, `solve_time_dependent`.
