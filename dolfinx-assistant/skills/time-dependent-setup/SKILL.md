---
name: time-dependent-setup
description: >
  This skill should be used when the user asks about "time-dependent", "transient",
  "heat equation", "diffusion over time", "time stepping", "backward Euler",
  "Crank-Nicolson", "BDF", "CFL condition", "time step size", "initial condition",
  or needs guidance on configuring time-dependent simulations in DOLFINx.
version: 0.1.0
---

# Time-Dependent Problem Setup Guide

Transient problems add a time dimension to your PDE. Getting the time discretization right is as important as the spatial discretization.

## Problem Classification

| Physics | PDE Type | Time Behavior | Stiffness |
|---------|----------|--------------|-----------|
| Heat conduction | Parabolic | Diffusive, smooth | Moderate |
| Wave propagation | Hyperbolic | Oscillatory | Low-moderate |
| Advection-diffusion | Parabolic-hyperbolic | Mixed | Can be stiff |
| Reaction-diffusion | Parabolic | Can develop sharp fronts | Often stiff |
| Navier-Stokes | Mixed | Complex dynamics | Problem-dependent |
| Phase field (Allen-Cahn, Cahn-Hilliard) | Parabolic (4th order) | Slow evolution + fast interface dynamics | Very stiff |

## Time Discretization

### Available Schemes in DOLFINx MCP

The `solve_time_dependent` tool uses backward Euler by default. For other schemes, use `run_custom_code` with a manual time loop.

| Scheme | Order | Stability | When to Use |
|--------|-------|-----------|-------------|
| Backward Euler | 1st | A-stable (unconditionally) | Default, safe choice. Stiff problems. |
| Crank-Nicolson | 2nd | A-stable | Better accuracy, smooth solutions |
| BDF2 | 2nd | A-stable | Stiff problems needing 2nd-order accuracy |
| Explicit Euler | 1st | Conditionally stable | Fast dynamics, small dt required anyway |
| Theta-method (θ=0.5) | 2nd | A-stable | Equivalent to Crank-Nicolson |

### Setting Up the Time Loop

The key pattern for `solve_time_dependent`:

1. **Define `u_n`**: The previous-timestep solution. Create with `create_function` and set initial condition with `interpolate`.

2. **Include `u_n` in forms**: The bilinear and linear forms must reference `u_n` as a known quantity from the previous step.

3. **Example for heat equation** (backward Euler):
   - Bilinear: `inner(u, v)*dx + dt*inner(grad(u), grad(v))*dx`
   - Linear: `inner(u_n, v)*dx + dt*f*v*dx`

Where `dt` is set as a material property (constant) and `u_n` is a registered function.

## Time Step Selection

### Rule of Thumb by Problem Type

| Problem | Starting dt | Refine If |
|---------|------------|-----------|
| Heat diffusion | h²/α (α = diffusivity) | Solution oscillates |
| Wave equation | h/c (c = wave speed) | Dispersion errors |
| Advection-dominated | h/|v| (CFL condition) | Instability |
| Nonlinear dynamics | 0.01 * T_end | Newton fails to converge |
| Phase field | 0.1 * ε² (ε = interface width) | Interface smears |

### CFL Condition

For explicit schemes, the Courant-Friedrichs-Lewy condition MUST be satisfied:

```
CFL = |v| * dt / h ≤ 1
```

For implicit schemes, CFL doesn't limit stability but still affects accuracy. Large CFL numbers smear transient features.

### Adaptive Time Stepping Strategy

When using `run_custom_code` for manual time loops:

1. Solve with current dt
2. Estimate local error (compare with half-step or use embedded method)
3. If error > tolerance: reject step, halve dt
4. If error < tolerance/10: accept step, double dt
5. Otherwise: accept step, keep dt

## Initial Conditions

Set via `create_function` + `interpolate`:

```
create_function(name="u_n", function_space="V")
interpolate(target="u_n", expression="sin(pi*x[0])*sin(pi*x[1])")
```

### Common Initial Conditions

| Problem | Typical IC |
|---------|-----------|
| Heat diffusion | Spatial distribution: Gaussian, step function, sinusoidal |
| Wave equation | Initial displacement + initial velocity |
| Navier-Stokes | Zero velocity (impulsive start) or steady-state solution |
| Phase field | tanh profile at interface, random perturbation for spinodal |

## Output and Snapshots

The `solve_time_dependent` tool accepts `output_times` — a list of times at which to save snapshots.

**Strategy**: Don't save every time step. Pick 10-20 characteristic times:
- Early transient behavior
- Approach to steady state
- Key physical events (e.g., peak temperature)

## Verification for Time-Dependent Problems

### Manufactured solution approach

Choose u_exact that depends on both space and time:
```
u_exact = exp(-t) * sin(pi*x[0]) * sin(pi*x[1])
```

Compute source term by substituting into the PDE (including ∂u/∂t term).

### Temporal convergence

Fix a fine spatial mesh. Run with dt, dt/2, dt/4. Plot error vs dt on log-log scale. Slope should match scheme order (1 for backward Euler, 2 for Crank-Nicolson).

### Energy estimates

For dissipative problems (heat equation, diffusion): total energy should decrease monotonically. A sudden increase signals instability or a bug.

## Common Mistakes

- **Forgetting to update `u_n`**: The previous-timestep function must be updated after each solve. `solve_time_dependent` handles this automatically, but manual loops need explicit copy.
- **dt too large for implicit scheme**: Won't crash, but silently produces inaccurate results. Always verify temporal accuracy.
- **Wrong form assembly**: The mass matrix term `inner(u, v)*dx` must appear in the bilinear form for implicit methods.
- **Not registering `u_n` as a material/function**: The UFL namespace needs access to `u_n`. Create it as a function and register with `set_material_properties` or ensure it's in session state.

## Reference Material

For time integration theory and convergence proofs, see `references/time-integration-theory.md`.
