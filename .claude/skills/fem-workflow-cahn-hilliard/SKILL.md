---
name: fem-workflow-cahn-hilliard
description: |
  Guides the user through solving the Cahn-Hilliard equation for phase separation using DOLFINx MCP tools.
  Use when the user asks about Cahn-Hilliard, phase field, spinodal decomposition,
  phase separation, binary alloy, diffuse interface, or Allen-Cahn equation.
---

# Cahn-Hilliard / Allen-Cahn Workflow (Official Demo)

Solve **phase-field** models for phase separation and pattern formation.

## Problem: Cahn-Hilliard Equation

**dc/dt = div(M grad(mu))** where **mu = df/dc - epsilon^2 lap(c)**

- `c` = concentration (order parameter)
- `mu` = chemical potential
- `M` = mobility
- `f(c) = 100 c^2 (1-c)^2` = free energy density (double-well)
- `epsilon` = interface width parameter

## Simplified Proxy: Allen-Cahn Steady State

For a direct MCP tool demonstration, use the **Allen-Cahn** steady state:

**-epsilon^2 lap(u) + u^3 - u = 0**

### Step-by-Step Tool Sequence

#### 1. Create the Mesh

```
create_unit_square(name="mesh", nx=32, ny=32)
```

#### 2. Create Function Space

```
create_function_space(name="V", family="Lagrange", degree=1)
```

#### 3. Set Initial Guess

Use a tanh-profile initial guess to help Newton convergence:

```
set_material_properties(name="u", value="np.tanh((x[0] - 0.5) / 0.1)", function_space="V")
```

#### 4. Apply Boundary Conditions

Fix values at domain boundaries to enforce phase separation:

```
apply_boundary_condition(value=-1.0, boundary="np.isclose(x[0], 0.0)")
apply_boundary_condition(value=1.0, boundary="np.isclose(x[0], 1.0)")
```

#### 5. Solve Nonlinear Problem

```
solve_nonlinear(
    residual="0.04*inner(grad(u), grad(v))*dx + (u**3 - u)*v*dx",
    unknown="u"
)
```

Here `epsilon^2 = 0.04` (epsilon=0.2). The residual is the weak form of `-epsilon^2 lap(u) + u^3 - u = 0`. Smaller epsilon gives sharper interfaces but requires finer meshes.

## Full Cahn-Hilliard (Time-Dependent, Mixed)

For the full Cahn-Hilliard with time stepping, use `run_custom_code` because it requires updating the previous-timestep function `c_n` inside a time loop with Newton solves at each step:

1. Create mixed space `W = V_c x V_mu` via `create_mixed_space`
2. Use `run_custom_code` to implement the time-stepping loop
3. Each timestep: assemble + Newton solve for (c, mu) pair, then update c_n

## Key Concepts

- **Double-well potential**: `f(c) = 100*c^2*(1-c)^2` drives phase separation
- **Mobility M**: Controls diffusion speed (constant or concentration-dependent)
- **Interface width epsilon**: Smaller = sharper interfaces but harder numerics
- **Time stepping**: Backward Euler for stability; small dt initially

## Common Pitfalls

- Newton divergence: Use a good initial guess (random perturbation around c=0.5)
- Interface resolution: Mesh must resolve epsilon (h < epsilon)
- Time step size: Start with dt ~ epsilon^2 for stability
