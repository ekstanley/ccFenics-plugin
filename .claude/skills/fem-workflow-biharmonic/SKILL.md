---
name: fem-workflow-biharmonic
description: |
  Guides the user through solving the biharmonic equation using DOLFINx MCP tools.
  Use when the user asks about biharmonic equation, plate bending, fourth-order PDE,
  Kirchhoff plate, thin plate, nabla^4 u, or Ciarlet-Raviart mixed method.
---

# Biharmonic Equation Workflow (Official Demo)

Solve the **biharmonic equation** (fourth-order PDE) using a mixed formulation.

## Problem

**nabla^4 u = f** in Omega, u = 0 on boundary

This is a fourth-order PDE that cannot be solved directly with standard C0 Lagrange elements.

## Mixed Formulation (Ciarlet-Raviart)

Introduce auxiliary variable **sigma = -lap(u)** to split into two coupled second-order PDEs:

1. **sigma + lap(u) = 0**
2. **-lap(sigma) = f**

Weak form: Find (sigma, u) in W = V_sigma x V_u such that:
- **integral(sigma * tau) dx + integral(grad(u) . grad(tau)) dx = 0** for all tau
- **integral(grad(sigma) . grad(w)) dx = integral(f * w) dx** for all w

## Step-by-Step Tool Sequence

### 1. Create the Mesh

```
create_unit_square(name="mesh", nx=16, ny=16)
```

### 2. Create Function Spaces

Both sub-spaces use P2 elements for accuracy:

```
create_function_space(name="V_sigma", family="Lagrange", degree=2)
create_function_space(name="V_u", family="Lagrange", degree=2)
```

### 3. Create Mixed Space

```
create_mixed_space(name="W", subspaces=["V_sigma", "V_u"])
```

### 4. Set Source Term

```
set_material_properties(name="f", value=1.0)
```

### 5. Define Variational Form

Using `split()` to decompose the mixed trial/test functions:

```
define_variational_form(
    bilinear="(split(u)[0]*split(v)[0] - inner(grad(split(u)[1]), grad(split(v)[0])) + inner(grad(split(u)[0]), grad(split(v)[1])))*dx",
    linear="f * split(v)[1] * dx",
    trial_space="W",
    test_space="W"
)
```

Here `split(u)[0]` = sigma (auxiliary), `split(u)[1]` = u (primary unknown).

### 6. Apply Boundary Conditions

Apply u = 0 on the boundary (sub_space=1 for the u-component):

```
apply_boundary_condition(value=0.0, boundary="True", function_space="W", sub_space=1)
```

### 7. Solve

The saddle-point system requires a direct solver (MUMPS):

```
solve(solver_type="direct", petsc_options={"pc_factor_mat_solver_type": "mumps"})
```

## Alternative: DG Interior Penalty

The biharmonic can also be solved with DG elements and interior penalty (requires `jump()` and `avg()` operators). This avoids the mixed formulation but needs careful penalty parameter tuning.

## Key Concepts

- **C0 vs C1 elements**: Standard Lagrange elements are C0 (continuous values, discontinuous derivatives). Biharmonic needs C1 or a mixed formulation.
- **Saddle-point structure**: The mixed formulation produces a saddle-point system. Use MUMPS or a Schur complement preconditioner.
- **Convergence**: P2 elements give O(h^2) convergence for the biharmonic.

## Common Pitfalls

- Using P1 elements: Too low order for the biharmonic; use P2 or higher
- Forgetting MUMPS: Iterative solvers may struggle with the saddle-point structure
- Missing BCs: Both sigma and u may need boundary conditions depending on the problem
