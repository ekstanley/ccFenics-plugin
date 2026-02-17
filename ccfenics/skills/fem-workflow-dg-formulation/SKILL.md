---
name: fem-workflow-dg-formulation
description: |
  Guides the user through discontinuous Galerkin (DG) formulations using DOLFINx MCP tools.
  Use when the user asks about DG, discontinuous Galerkin, interior penalty,
  SIPG, NIPG, upwinding, jump operators, facet averages, or DG advection.
---

# Discontinuous Galerkin (DG) Formulation Workflow (Official Demo)

Solve PDEs using **discontinuous Galerkin** methods with interior penalty stabilization.

## Key Operators

- `jump(u)` = jump of u across interior facets: `u('+') - u('-')`
- `avg(u)` = average of u across interior facets: `0.5*(u('+') + u('-'))`
- `dS` = interior facet measure (integration over internal edges/faces)
- `ds` = exterior facet measure (integration over boundary)
- `n` = `FacetNormal(mesh)` -- outward normal (use `n('+')` or `n('-')` on interior facets)

## SIPG Poisson Example

Symmetric Interior Penalty Galerkin method for `-lap(u) = f`:

### 1. Create the Mesh

```
create_unit_square(name="mesh", nx=16, ny=16)
```

### 2. Create DG Function Space

```
create_function_space(name="V", family="DG", degree=1)
```

### 3. Set Source Term

```
set_material_properties(name="f", value=1.0)
```

### 4. Define Variational Form

The SIPG bilinear form has four components:

1. **Volume term**: `inner(grad(u), grad(v)) * dx`
2. **Consistency**: `- inner(avg(nabla_grad(u)), jump(v, n)) * dS`
3. **Symmetry**: `- inner(jump(u, n), avg(nabla_grad(v))) * dS`
4. **Penalty**: `(alpha/avg(h)) * inner(jump(u), jump(v)) * dS`

Plus Nitsche-style boundary terms for Dirichlet BCs:
5. `- inner(dot(grad(u), n), v) * ds`
6. `- inner(u, dot(grad(v), n)) * ds`
7. `(alpha/h) * inner(u, v) * ds`

```
define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx - inner(avg(nabla_grad(u)), jump(v, n))*dS - inner(jump(u, n), avg(nabla_grad(v)))*dS + (10.0)/avg(h) * inner(jump(u), jump(v))*dS + (10.0)/h * inner(u, v)*ds - inner(dot(grad(u), n), v)*ds - inner(u, dot(grad(v), n))*ds",
    linear="f*v*dx"
)
```

### 5. Solve

No Dirichlet BCs needed (imposed weakly via Nitsche terms). Use direct solver:

```
solve(solver_type="direct")
```

## Penalty Parameter Selection

- `alpha` should scale as `O(degree^2)` for stability
- Rule of thumb: `alpha = 10 * degree^2`
- Too small: unstable (oscillations)
- Too large: over-penalized (locking)

## DG Method Variants

| Method | Symmetry Term | Properties |
|--------|--------------|------------|
| SIPG | `-inner(jump(u,n), avg(grad(v)))*dS` | Symmetric, optimal convergence |
| NIPG | `+inner(jump(u,n), avg(grad(v)))*dS` | Non-symmetric, robust |
| IIPG | (omitted) | Incomplete, simplest |

## Common Pitfalls

- Forgetting the penalty term: System becomes singular without stabilization
- Wrong measure: Use `dS` (capital S) for interior facets, `ds` for boundary
- Normal direction: `n('+')` and `n('-')` point in opposite directions across facets
- DG degree 0: Only piecewise constant; use degree >= 1 for gradients
