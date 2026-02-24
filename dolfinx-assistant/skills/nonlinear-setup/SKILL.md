---
name: nonlinear-setup
description: >
  This skill should be used when the user asks about "nonlinear", "Newton method",
  "Newton solver", "load stepping", "nonlinear Poisson", "hyperelasticity",
  "Navier-Stokes", "nonlinear convergence", "initial guess", "SNES",
  "Jacobian", "residual form", or needs guidance on setting up and solving
  nonlinear PDEs in DOLFINx.
version: 0.1.0
---

# Nonlinear Problem Setup Guide

Nonlinear PDEs require Newton's method (or a variant) to solve. The setup differs from linear problems: you define a residual form F(u; v) = 0 instead of separate bilinear and linear forms.

## When Is a Problem Nonlinear?

| Source of Nonlinearity | Example | Indicator |
|-----------------------|---------|-----------|
| Coefficient depends on solution | κ(u)∇u | Material property is a function of u |
| Nonlinear differential operator | -∇·(|∇u|^{p-2} ∇u) | p-Laplacian |
| Geometry changes with solution | Large deformation elasticity | Reference vs current configuration |
| Convective term | (u·∇)u | Navier-Stokes advection |
| Reaction term | u³, e^u | Nonlinear source |

## DOLFINx Nonlinear Solve Pattern

### Step 1: Create the unknown function

```
create_function(name="u_h", function_space="V", expression="0.0")
```

This is a mutable Function — Newton modifies it in place. Initialize with a reasonable guess (zero is fine for many problems, but not all).

### Step 2: Define the residual

Write F(u; v) = 0 as a UFL string using `u_h` (the unknown) and `v` (the test function):

```
residual = "inner(grad(u_h), grad(v))*dx + inner(u_h**3, v)*dx - inner(f, v)*dx"
```

Note: use `u_h` (the function name), NOT `u` (the trial function). This is the key difference from linear problems.

### Step 3: Apply BCs and solve

```
apply_boundary_condition(value=0.0, boundary="np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)")
solve_nonlinear(residual="...", unknown="u_h")
```

DOLFINx auto-differentiates the Jacobian from the residual. You can provide an explicit Jacobian via the `jacobian` parameter if auto-differentiation doesn't work for your case.

## Newton Solver Configuration

### SNES Types

| Type | When | Config |
|------|------|--------|
| `newtonls` | Default. Line search prevents overshoot. | `snes_type="newtonls"` |
| `newtontr` | Trust region. More robust near singularities. | `snes_type="newtontr"` |
| `nrichardson` | Nonlinear Richardson. Simple, for well-conditioned problems. | `snes_type="nrichardson"` |

### Tolerance Selection

| Accuracy Need | rtol | atol | max_iter |
|--------------|------|------|----------|
| Quick exploration | 1e-6 | 1e-8 | 20 |
| Engineering | 1e-8 | 1e-10 | 30 |
| Research validation | 1e-10 | 1e-12 | 50 |

### Inner Linear Solver

Newton needs to solve a linear system at each iteration. The inner solver choice matters:

- **Small problems (< 50K DOFs)**: `ksp_type="preonly"`, `pc_type="lu"` — direct solve, guaranteed convergence of inner problem
- **Large SPD Jacobian**: `ksp_type="cg"`, `pc_type="hypre"` — iterative with AMG
- **Non-symmetric Jacobian**: `ksp_type="gmres"`, `pc_type="ilu"` — general iterative

## Initial Guess Strategies

Newton converges quadratically near the solution but can diverge from a bad starting point.

### Safe starting points

| Problem Type | Good Initial Guess |
|-------------|-------------------|
| Nonlinear Poisson | Zero (if BCs are homogeneous) |
| Hyperelasticity | Zero displacement |
| Navier-Stokes | Stokes solution (solve linear problem first) |
| Phase field | Analytical interface profile (tanh) |
| Thermal with radiation | Linear temperature field between BC values |

### When zero doesn't work

If Newton diverges from zero:

1. **Linear approximation**: Solve the linearized problem (drop nonlinear terms), use as guess
2. **Continuation/load stepping**: Apply load incrementally (see below)
3. **Physical intuition**: Interpolate an approximate solution based on the physics

## Load Stepping (Continuation)

For strongly nonlinear problems where Newton can't converge in one shot:

### Pattern

1. Define a load parameter λ ∈ [0, 1]
2. Scale the forcing: f_λ = λ * f_full
3. For λ = 0.1, 0.2, ..., 1.0:
   a. Update the forcing
   b. Solve nonlinear problem (using previous solution as initial guess)
   c. If Newton converges: proceed to next λ
   d. If Newton diverges: halve the λ increment and retry

### Implementation via `run_custom_code`

```python
load_steps = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
for lam in load_steps:
    # Update load parameter
    session.ufl_symbols["load_factor"] = dolfinx.fem.Constant(mesh, lam)
    # Solve (u_h retains previous solution as initial guess)
    # ... call solve_nonlinear
```

### Typical load step counts

| Problem | Steps | Notes |
|---------|-------|-------|
| Mild nonlinearity | 1-3 | Direct solve often works |
| Moderate (hyperelasticity, small strain) | 5-10 | Linear increments |
| Severe (large deformation, buckling) | 20-50 | May need adaptive stepping |
| Near bifurcation | 50-100+ | Very small steps near critical point |

## Jacobian Considerations

### Auto-differentiation (default)

DOLFINx uses UFL's `derivative()` to compute the Jacobian automatically. This works for most problems and is the recommended approach.

### Manual Jacobian

Provide an explicit Jacobian when:
- Auto-diff doesn't handle your nonlinearity (rare)
- You want a simplified/approximate Jacobian for faster assembly
- You're debugging convergence issues

```
solve_nonlinear(
    residual="...",
    unknown="u_h",
    jacobian="inner(grad(du), grad(v))*dx + 3*inner(u_h**2*du, v)*dx"
)
```

Here `du` is the trial function (increment direction).

### Jacobian debugging

If Newton converges slowly (linear instead of quadratic rate), the Jacobian is probably wrong. Check:
1. Remove the explicit Jacobian and let auto-diff handle it
2. Compare residual reduction per iteration — should be quadratic near solution
3. Use finite differences to verify: set `snes_type="test"` in PETSc options

## Common Nonlinear Problems in DOLFINx

### Nonlinear Poisson (-∇·((1+u²)∇u) = f)

```
residual: "inner((1 + u_h**2)*grad(u_h), grad(v))*dx - inner(f, v)*dx"
```

### Hyperelasticity (Neo-Hookean)

Requires `run_custom_code` for strain energy density definition. The residual is the first variation of the energy functional.

### Navier-Stokes (Steady)

Nonlinear convective term: (u·∇)u. Often solved with Picard iteration (linearize convective term) before switching to Newton.

## Convergence Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Diverges immediately | Bad initial guess | Use linear solve as starting point |
| Converges then diverges | Overshoot | Enable line search (newtonls) |
| Very slow convergence | Wrong Jacobian | Remove manual Jacobian, use auto-diff |
| Oscillates without converging | Multiple solutions | Try different initial guess |
| NaN after few iterations | Negative quantity in log/sqrt | Add bounds or regularization |

## Reference Material

For advanced nonlinear strategies (arc-length, bifurcation tracking), see `references/nonlinear-strategies.md`.
