---
name: fem-workflow-discrete-ops
description: |
  Guides the user through creating discrete differential operators using DOLFINx MCP tools.
  Use when the user asks about discrete gradient, discrete curl, operator matrices,
  or building matrix representations of differential operators.
---

# Discrete Operator Workflow

Build matrix representations of differential operators for advanced solver configurations.

## Step-by-Step Tool Sequence

### 1. Set Up Spaces

Create the domain and source/target function spaces:

```
create_unit_square(name="mesh", nx=16, ny=16)
create_function_space(name="V_nodal", family="Lagrange", degree=1)
create_function_space(name="V_edge", family="N1curl", degree=1)
```

### 2. Create Discrete Operator

Use `create_discrete_operator` to build the matrix:

```
create_discrete_operator(
    name="grad_matrix",
    operator_type="gradient",
    source_space="V_nodal",
    target_space="V_edge"
)
```

This creates the matrix G such that G * u_nodal = grad(u) in the edge space.

### 3. Available Operators

| Operator | Source Space | Target Space | Maps |
|---|---|---|---|
| Gradient | Lagrange (H1) | Nedelec (H(curl)) | Scalar → Edge |
| Curl | Nedelec (H(curl)) | RT (H(div)) | Edge → Face |
| Divergence | RT (H(div)) | DG (L2) | Face → Cell |

These form the de Rham complex: H1 → H(curl) → H(div) → L2.

### 4. Use in Solver Configuration

Discrete operators are used for:
- **AMS preconditioner**: Needs the discrete gradient for H(curl) problems
- **ADS preconditioner**: Needs discrete gradient and curl for H(div) problems
- **Multigrid**: Transfer operators between grid levels

Example with AMS:

```
solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "pc_hypre_type": "ams",
        "pc_hypre_ams_discrete_gradient": "grad_matrix"
    }
)
```

## When to Use

- Electromagnetics (Maxwell's equations) with Nedelec elements
- Mixed finite elements requiring specialized preconditioners
- Problems on the de Rham complex requiring inter-space operators
- Custom multigrid or domain decomposition methods
