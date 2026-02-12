---
name: fem-workflow-functionals
description: |
  Guides the user through computing integrals, functionals, and point evaluations of FEM solutions using DOLFINx MCP tools.
  Use when the user asks about computing an integral, evaluating at a point, computing a functional value,
  drag/lift coefficients, average values, or energy norms.
---

# Functionals & Point Evaluation Workflow

Compute derived quantities from FEM solutions: integrals, point values, norms.

## Step-by-Step Tool Sequence

### 1. Compute Integrals (Functionals)

Use `compute_functionals` to evaluate integrals over the domain or boundary:

```
compute_functionals(
    functional="u_h * dx",
    solution_name="u_h"
)
```

Common functionals:
- **Total integral**: `"u_h * dx"` — integral of solution over domain
- **L2 norm squared**: `"inner(u_h, u_h) * dx"`
- **Energy norm**: `"inner(grad(u_h), grad(u_h)) * dx"`
- **Boundary integral**: `"u_h * ds"` — integral over boundary
- **Tagged boundary**: `"u_h * ds(2)"` — integral on tagged facets (requires `mark_boundaries`)

### 2. Evaluate at Specific Points

Use `evaluate_solution` for pointwise queries:

```
evaluate_solution(
    solution_name="u_h",
    points=[[0.5, 0.5], [0.25, 0.75]]
)
```

Returns values at each coordinate. For vector solutions, returns all components.

### 3. Query Multiple Points

Use `query_point_values` for systematic sampling:

```
query_point_values(
    solution_name="u_h",
    direction="x",
    num_points=20,
    fixed_coord=0.5
)
```

Returns a line profile along a direction — useful for comparing against analytical solutions.

### 4. Compute Error Norms

Use `compute_error` with an exact solution:

```
compute_error(
    exact="sin(pi*x[0])*sin(pi*x[1])",
    norm_type="L2",
    solution_name="u_h"
)
```

Supported norms: `"L2"`, `"H1"`, `"Linf"`.

## Physical Quantities

| Quantity | Functional Expression | Context |
|---|---|---|
| Total heat | `"u_h * dx"` | Thermal problems |
| Average value | `"u_h * dx"` / domain area | Any scalar field |
| Strain energy | `"0.5 * inner(sigma, epsilon) * dx"` | Elasticity |
| Drag force | `"sigma_n[0] * ds(wall_tag)"` | Fluid mechanics |
| Flow rate | `"inner(u_h, n) * ds(outlet_tag)"` | Channel flow |

## Convergence Studies

Combine with mesh refinement to verify convergence rates:

```
# For each mesh size N in [4, 8, 16, 32]:
#   solve -> compute_error -> record error
# Then compute rate = log(e_1/e_2) / log(h_1/h_2)
```

Expected rates: P1 elements → O(h^2) in L2, O(h) in H1.
