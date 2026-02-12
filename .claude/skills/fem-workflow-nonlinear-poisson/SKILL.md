---
name: fem-workflow-nonlinear-poisson
description: |
  Guides the user through solving a nonlinear Poisson equation using DOLFINx MCP tools.
  Use when the user asks about nonlinear Poisson, Newton method for PDEs, nonlinear PDE,
  q(u)*grad(u), or coefficient-dependent diffusion.
---

# Nonlinear Poisson Workflow (Tutorial Ch2.3)

Solve: **-div(q(u) * grad(u)) = f** where `q(u) = 1 + u^2` (or any nonlinear function of u).

This is a nonlinear problem because the coefficient `q` depends on the unknown `u`. We solve it with Newton's method via the `solve_nonlinear` tool.

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(nx=32, ny=32, name="nl_mesh")
```

### 2. Create Function Space

```
create_function_space(family="Lagrange", degree=1, name="V")
```

### 3. Create Initial Guess

The Newton solver needs a starting point. Interpolate a zero (or better) initial guess:

```
interpolate(name="u", expression="0.0", function_space="V")
```

### 4. Define Source Term

```
set_material_properties(name="f", value="10*exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2)/0.02)")
```

### 5. Apply Boundary Conditions

```
apply_boundary_condition(value="0.0", boundary="True", name="bc_zero")
```

### 6. Solve Nonlinear Problem

The residual form is `F(u; v) = q(u)*inner(grad(u),grad(v))*dx - f*v*dx`:

```
solve_nonlinear(
    residual="(1 + u**2) * inner(grad(u), grad(v)) * dx - f * v * dx",
    unknown="u",
    snes_type="newtonls",
    rtol=1e-10,
    max_iter=50,
    solution_name="u_h"
)
```

The Jacobian is automatically derived. For complex problems, you can provide it explicitly:

```
solve_nonlinear(
    residual="(1 + u**2) * inner(grad(u), grad(v)) * dx - f * v * dx",
    unknown="u",
    jacobian="2*u*du*inner(grad(u),grad(v))*dx + (1+u**2)*inner(grad(du),grad(v))*dx",
    solution_name="u_h"
)
```

### 7. Post-Process

```
export_solution(solution_name="u_h", filename="nonlinear_poisson", format="vtk")
```

## Key Concepts

- **Residual form**: `F(u; v) = 0` -- the weak form with the unknown `u` as a Function (not TrialFunction)
- **Newton linearization**: At each iteration, solve `J(u_k; du, v) = -F(u_k; v)` and update `u_{k+1} = u_k + du`
- **Automatic Jacobian**: `ufl.derivative(F, u, du)` computes J symbolically
- **Initial guess**: Newton needs a starting point. Zero works for mild nonlinearity; better guesses needed for strong nonlinearity

## Manufactured Solution Verification

For `q(u) = 1 + u^2` with exact solution `u_exact = sin(pi*x)*sin(pi*y)`:

Source term (compute by substituting into PDE):
```
f = -div(q(u_exact) * grad(u_exact))
```

This requires symbolic differentiation -- use `run_custom_code` for verification.

## Convergence Tips

| Issue | Solution |
|---|---|
| Newton doesn't converge | Better initial guess, or increase `max_iter` |
| Diverges immediately | Check residual form for sign errors |
| Slow convergence | Provide explicit Jacobian |
| Need load stepping | Break load into increments, solve sequentially |
