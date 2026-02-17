---
name: fem-workflow-convergence-rates
description: |
  Guides the user through computing convergence rates under mesh refinement using DOLFINx MCP tools.
  Use when the user asks about convergence rates, convergence study, mesh refinement study,
  h-refinement, p-refinement, error convergence, or verifying convergence order.
---

# Convergence Rate Study (Tutorial Ch4.4)

Compute **empirical convergence rates** by solving a PDE at multiple mesh resolutions and measuring how the error decreases.

## Theory

For a degree-p finite element, the a priori error estimate gives:

- **L2 error**: `||u - u_h||_L2 <= C * h^(p+1)`
- **H1 error**: `||u - u_h||_H1 <= C * h^p`

The convergence rate `r` is computed from consecutive mesh sizes:

```
r = ln(E_i / E_{i-1}) / ln(h_i / h_{i-1})
```

where `E_i` is the error at resolution `h_i = 1/N_i`.

## Step-by-Step Tool Sequence

### 1. Choose a Manufactured Solution

Pick a smooth exact solution so the error is purely discretization error:

```
u_exact = cos(2*pi*x[0]) * cos(2*pi*x[1])
f = 8*pi**2 * cos(2*pi*x[0]) * cos(2*pi*x[1])
```

### 2. Loop Over Mesh Resolutions

For each N in [4, 8, 16, 32, 64]:

**a) Create mesh:**
```
create_unit_square(nx=N, ny=N, name=f"mesh_{N}")
```

**b) Create function space:**
```
create_function_space(mesh_name=f"mesh_{N}", family="Lagrange", degree=p, name=f"V_{N}")
```

**c) Define variational form:**
```
define_variational_form(
    space_name=f"V_{N}",
    bilinear="inner(grad(u), grad(v)) * dx",
    linear="8*pi**2 * cos(2*pi*x[0]) * cos(2*pi*x[1]) * v * dx",
    form_name=f"poisson_{N}"
)
```

**d) Apply boundary condition (exact solution):**
```
apply_boundary_condition(
    space_name=f"V_{N}",
    boundary="True",
    value="cos(2*pi*x[0]) * cos(2*pi*x[1])",
    bc_name=f"bc_{N}"
)
```

**e) Solve:**
```
solve(form_name=f"poisson_{N}", bc_names=[f"bc_{N}"], solution_name=f"u_{N}")
```

**f) Compute error:**
```
compute_error(
    solution_name=f"u_{N}",
    exact_solution="cos(2*pi*x[0]) * cos(2*pi*x[1])",
    norm_type="L2"
)
```

### 3. Compute Convergence Rates

Collect errors `[E_4, E_8, E_16, E_32, E_64]` and mesh sizes `h = [1/4, 1/8, 1/16, 1/32, 1/64]`, then:

```
rate_i = ln(E_i / E_{i-1}) / ln(h_i / h_{i-1})
```

### 4. Present Results Table

```
| N   | h      | L2 error  | Rate |
|-----|--------|-----------|------|
|   4 | 0.2500 | ...       | --   |
|   8 | 0.1250 | ...       | ~2.0 |
|  16 | 0.0625 | ...       | ~2.0 |
|  32 | 0.0312 | ...       | ~2.0 |
|  64 | 0.0156 | ...       | ~2.0 |
```

For degree p, expect L2 rate ~ p+1, H1 rate ~ p.

## Expected Rates by Degree

| Degree | L2 Rate | H1 Rate |
|--------|---------|---------|
| 1      | 2       | 1       |
| 2      | 3       | 2       |
| 3      | 4       | 3       |
| 4      | 5       | 4       |

## Variations

- **p-refinement**: Fix N, increase degree p from 1 to 4, observe exponential convergence for smooth solutions
- **H1 norm**: Replace `norm_type="L2"` with `norm_type="H1"` to verify gradient convergence
- **Different PDEs**: Same methodology applies to elasticity, Stokes, etc.
- **Singular solutions**: Solutions with corner singularities show reduced rates (motivates adaptive refinement)

## Common Pitfalls

| Issue | Cause | Fix |
|---|---|---|
| Rate too low | Mesh not fine enough | Use more resolutions (N up to 128) |
| Rate too high | Pre-asymptotic regime | Skip coarsest meshes |
| Rate oscillates | Boundary approximation | Use finer meshes or exact geometry |
| NaN in rate | Zero error at coarse mesh | Check exact solution is correct |

## Cleanup

After the study, remove intermediate objects:

```
remove_object(name="mesh_4", object_type="mesh")
remove_object(name="mesh_8", object_type="mesh")
...
```

Or reset the full session: `reset_session()`
