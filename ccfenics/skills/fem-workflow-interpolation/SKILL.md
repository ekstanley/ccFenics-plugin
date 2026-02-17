---
name: fem-workflow-interpolation
description: |
  Guides the user through interpolating functions and projecting between function spaces using DOLFINx MCP tools.
  Use when the user asks about interpolating a function, L2 projection, transferring data between spaces,
  or creating initial conditions from expressions.
---

# Interpolation & Projection Workflow

Transfer data between function spaces or create functions from expressions.

## Step-by-Step Tool Sequence

### 1. Set Up Mesh and Function Space

Ensure a mesh and function space exist:

```
create_unit_square(name="mesh", nx=32, ny=32)
create_function_space(name="V", family="Lagrange", degree=2)
```

### 2. Interpolate an Expression

Use `interpolate` to create a function from a numpy expression:

```
interpolate(
    name="u_init",
    expression="np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])",
    function_space="V"
)
```

The expression operates on coordinate arrays: `x[0]`, `x[1]`, `x[2]`. Available symbols: `np`, `pi`, `sin`, `cos`, `exp`, `sqrt`.

### 3. Project Between Spaces (L2 Projection)

To transfer a function from one space to another:

```
create_function_space(name="V_coarse", family="Lagrange", degree=1)
project(source="u_init", target_space="V_coarse", name="u_projected")
```

L2 projection minimizes ||u - u_projected||_L2 over the target space.

### 4. Create Initial Conditions

For time-dependent problems, interpolate the initial condition:

```
interpolate(name="u_n", expression="0.0 + 0*x[0]", function_space="V")
```

Use `0*x[0]` to ensure the expression broadcasts over all DOFs.

### 5. Verify the Interpolation

Check the result:

```
evaluate_solution(solution_name="u_init", points=[[0.5, 0.5]])
compute_error(exact="np.sin(np.pi*x[0])*np.cos(np.pi*x[1])", norm_type="L2")
```

## Common Patterns

| Pattern | Expression | Use Case |
|---|---|---|
| Zero field | `"0.0 + 0*x[0]"` | Initial guess for nonlinear solvers |
| Constant | `"1.0 + 0*x[0]"` | Uniform material property |
| Gaussian | `"exp(-50*((x[0]-0.5)**2 + (x[1]-0.5)**2))"` | Localized load or IC |
| Step function | `"np.where(x[0] < 0.5, 1.0, 0.0)"` | Piecewise initial condition |
| Vector field | `"np.vstack([x[1], -x[0]])"` | Rotational velocity field |

## Physical Explanation

- **Interpolation**: Evaluates an analytical expression at DOF points to create a discrete function
- **L2 Projection**: Solves a mass matrix system to find the best approximation in a coarser space
- **Key difference**: Interpolation is pointwise (cheap), projection is variational (more accurate for non-polynomial functions)
