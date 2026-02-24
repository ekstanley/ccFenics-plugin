# UFL Form Authoring Guide

**Triggers**: "author UFL", "variational form", "bilinear form", "linear form", "UFL expression", "weak form", "define form"

**Version**: 0.1.0

**Team**: Alpha (Formulation Experts)

## Overview

UFL (Unified Form Language) is the symbolic mathematics layer in DOLFINx. This guide covers authoring correct, efficient variational forms for finite element problems.

## Core UFL Concepts

### Trial and Test Functions

```python
# Once a function space is defined (e.g., "V"), UFL provides:
# u = TrialFunction(V)    # Unknown (bilinear form first argument)
# v = TestFunction(V)     # Test function (weighted residual)
# f = Function(V)         # Known function (interpolated coefficient)
```

**Rule**: Bilinear form `a(u, v)` is linear in both arguments. Linear form `L(v)` is linear in test function only.

### Spatial Coordinates

Access coordinates via `x` (a 2-tuple or 3-tuple depending on mesh dimension):

```python
# x[0] = x-coordinate (or r in axisymmetric)
# x[1] = y-coordinate (or z in axisymmetric)
# x[2] = z-coordinate (3D only)

# Example: spatially-varying source term
f_expr = "sin(pi*x[0])*sin(pi*x[1])"  # UFL expression string
f = interpolate(expression=f_expr, target_space="V")
```

---

## Operator Reference (Quick)

| **What I Want** | **UFL Syntax** | **Input Ranks** | **Output Rank** | **Example** |
|---|---|---|---|---|
| Scalar product | `inner(u, v)` | 1+, 1+ | 0 | `inner(u, v)` → scalar |
| Dot product (2 vectors) | `dot(u, v)` | 1, 1 | 0 | `dot(grad(u), grad(v))` → scalar |
| Outer product | `outer(u, v)` | 1, 1 | 2 | `outer(u, v)` → matrix |
| Cross product | `cross(u, v)` | 1, 1 | 1 | `cross(u, v)` → vector |
| Gradient | `grad(u)` | 0 → 1, 1 → 2 | rank+1 | `grad(u)` on scalar → vector |
| Divergence | `div(u)` | 1 → 0, 2 → 1 | rank−1 | `div(sigma)` on matrix → scalar |
| Curl | `curl(u)` | 1 → 1, 2 → 1 | 1 | `curl(E)` → vector (3D) |
| Trace | `tr(A)` | 2 | 0 | `tr(stress)` → scalar |
| Determinant | `det(A)` | 2 | 0 | `det(F)` → scalar (Jacobian) |
| Deviatoric | `dev(A)` | 2 | 2 | `dev(sigma)` = sigma - (1/3)*tr(sigma)*I |
| Symmetric part | `sym(A)` | 2 | 2 | `sym(grad(u))` → ε (strain) |
| Skew part | `skew(A)` | 2 | 2 | `skew(grad(u))` → W (rotation) |
| Element-wise mult | `A * B` (not `inner`) | same rank | same | For matrix scaling |
| Conditional | `conditional(cond, val_true, val_false)` | bool, any, any | same as vals | Piecewise definitions |

**Key insight**: `inner()` is the most common choice—it contracts all indices, giving the right result for scalars, vectors, and matrices.

---

## Tensor Algebra Deep Dive

### Rank System

- **Rank 0**: Scalar (no indices)
- **Rank 1**: Vector (1 index) — represents `u_i`
- **Rank 2**: Matrix/Tensor (2 indices) — represents `u_ij`

### Indexing and Component Access

```python
# In a vector field v (rank 1):
v[0]  # First component
v[1]  # Second component

# In a matrix field A (rank 2):
A[0, 0]  # Component A_00
A[0, 1]  # Component A_01
A[i, j]  # Symbolic indexing (UFL indices)

# Access via grad of a scalar (rank 1):
du_dx = grad(u)[0]  # du/dx
du_dy = grad(u)[1]  # du/dy

# Access via grad of vector (rank 2):
strain = grad(u)           # Full strain matrix (rank 2)
strain[0, 0]              # ε_xx
strain[0, 1]              # ε_xy
sym_strain = sym(grad(u)) # Symmetric part only
```

### Tensor Contractions

```python
# Trace (contract all indices):
trace = tr(A)  # = A_ii (sum of diagonal)

# Inner product (all indices contracted):
scalarProduct = inner(A, B)  # = Σᵢⱼ Aᵢⱼ * Bᵢⱼ

# Double contraction (both sets of indices):
doubleContraction = inner(A, B)  # same as above for rank-2

# Deviatoric (traceless part):
devA = dev(A)  # = A - (1/3)*tr(A)*I
```

### Symmetric and Skew Parts

```python
# Strain tensor (symmetric part of gradient):
epsilon = sym(grad(u))  # = (1/2)*(∇u + ∇uᵀ)

# Rotation tensor (skew part):
omega = skew(grad(u))   # = (1/2)*(∇u - ∇uᵀ)

# Verify: grad(u) = sym(grad(u)) + skew(grad(u))
```

---

## Differentiation Operators

### Gradient (∇)

```python
# On scalar field u (rank 0 → rank 1):
du = grad(u)  # Returns vector: (∂u/∂x, ∂u/∂y, ∂u/∂z)

# On vector field v (rank 1 → rank 2):
Dv = grad(v)  # Returns matrix: Dv[i,j] = ∂v_i/∂x_j
```

### Divergence (∇·)

```python
# On vector field u (rank 1 → rank 0):
div_u = div(u)  # = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z

# On matrix field σ (rank 2 → rank 1):
div_sigma = div(sigma)  # [i] = Σⱼ ∂σ_ij/∂x_j
```

### Curl (∇×)

```python
# On vector field u in 3D (rank 1 → rank 1):
curl_u = curl(u)  # = (∂u_z/∂y - ∂u_y/∂z, ..., ...)

# Note: In 2D, curl is often computed differently; use carefully
```

### Symbolic Differentiation (`nabla_grad`, `nabla_div`)

For more control over UFL representation (rarely needed in standard FEM):

```python
# These exist but inner/grad/div are usually preferred
# Use `nabla_grad`, `nabla_div` only for mixed/complex forms
```

---

## Integration Measures

### Volume Integral

```python
# dx = volume integral (default measure on entire domain)
a = inner(grad(u), grad(v)) * dx
```

### Exterior Boundary Integral

```python
# ds = exterior boundary (requires mark_boundaries)
L = f * v * ds(tag)  # Boundary source on subdomain with tag

# Without tags (all exterior boundary):
L = f * v * ds
```

### Interior Facet Integral (DG)

```python
# dS = interior facets (between elements, for DG penalties)
penalty_form = alpha * (1/h_avg) * inner(jump(u), jump(v)) * dS

# jump(u) = u(+) - u(-)  (difference across facet)
# avg(u) = 0.5*(u(+) + u(-))  (average across facet)
# h_avg = average of cell diameters on both sides
```

### Cell Integral (Subdomain)

For domain decomposition, use custom cell tags (advanced):

```python
# After create_discrete_operator or manage_mesh_tags,
# access with dx(tag) or ds(tag) for marked regions
```

---

## Restrictions: (+) and (−)

Used exclusively in facet integrals (dS) for Discontinuous Galerkin:

```python
# u(+) = value on cell "+" side of facet
# u(-) = value on cell "−" side of facet
# avg(u) = 0.5*(u(+) + u(-))
# jump(u) = u(+) - u(-)

# Example: interior penalty DG Poisson
a = inner(grad(u('+')) - grad(u('-')), grad(v('+')) - grad(v('-'))) * dS \
  + inner(grad(u), grad(v)) * dx

# Note: Always pair (+) restrictions in both bilinear and linear forms
```

---

## Common Mistakes

### 1. Rank Mismatch

```python
# WRONG: Trying to add vector and scalar
a = u + grad(v) * dx  # Rank 1 + Rank 1 → OK, but multiplied by scalar dx → Wrong!

# CORRECT: Use inner to contract properly
a = inner(u, v) * dx  # Both vectors → scalar product → scalar form
```

### 2. Missing Integration Measure

```python
# WRONG: No dx at end
a = inner(grad(u), grad(v))  # Syntax error or silent failure

# CORRECT: Always end with dx, ds, or dS
a = inner(grad(u), grad(v)) * dx
```

### 3. Wrong Operator for Scalar vs Vector

```python
# If u and v are SCALARS, grad(u) is a VECTOR:
# WRONG: inner(u, grad(v)) * dx  # Rank 0 vs Rank 1 → Error
# CORRECT: inner(grad(u), grad(v)) * dx  # Both Rank 1 → OK

# If u and v are VECTORS, grad(u) is a MATRIX:
# WRONG: inner(grad(u), grad(v)) * dx  # Correct for elasticity
# CORRECT: (same thing)
```

### 4. Forgetting Boundary Tags

```python
# WRONG: Trying to use ds(1) without calling mark_boundaries first
L = g * v * ds(1)  # Will fail at solve time

# CORRECT: Call mark_boundaries, then use tags
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"}
], name="bnd_tags", mesh_name="mesh")

L = g * v * ds(1)  # Now valid
```

### 5. Mixing up Trial and Test

```python
# In bilinear form, FIRST argument is trial (u), SECOND is test (v):
# WRONG: a = inner(v, grad(u)) * dx  # Swapped!
# CORRECT: a = inner(grad(u), grad(v)) * dx  # u first, v second
```

---

## Conditional Expressions

```python
# Define piecewise-defined coefficients
condition = x[0] < 0.5
material = conditional(condition, 10.0, 1.0)  # κ = 10 if x<0.5, else 1

a = material * inner(grad(u), grad(v)) * dx
```

---

## Workflow: Composing a Form

### Step 1: Identify Trial/Test Spaces

```python
# Scalar problem:
create_function_space(name="V", family="Lagrange", degree=1)
# u = TrialFunction(V), v = TestFunction(V)

# Vector problem (elasticity):
create_function_space(name="V", family="Lagrange", degree=1, shape=[2])
# u = TrialFunction(V), v = TestFunction(V)

# Mixed problem (Stokes):
create_function_space(name="P", family="Lagrange", degree=2)
create_function_space(name="Q", family="Lagrange", degree=1)
create_mixed_space(name="W", subspaces=["P", "Q"])
# u, p = split(TrialFunction(W))
# v, q = split(TestFunction(W))
```

### Step 2: Define Material Properties

```python
set_material_properties(name="f", value="sin(pi*x[0])", function_space="V")
set_material_properties(name="kappa", value=1.0)  # constant
```

### Step 3: Build Bilinear Form

Start with main term, add boundary/domain modifications:

```python
# Poisson: inner(κ∇u, ∇v)
bilinear = "kappa * inner(grad(u), grad(v)) * dx"

# Add penalty for Nitsche (weak Dirichlet):
bilinear += "- inner(grad(u)*n, v)*ds(1) - inner(u, grad(v)*n)*ds(1) + (100.0/h)*inner(u, v)*ds(1)"
```

### Step 4: Build Linear Form

```python
# Poisson source:
linear = "f * v * dx"

# Add Neumann on boundary 2:
linear += "+ g * v * ds(2)"

# Add Nitsche RHS (weak Dirichlet):
linear += "+ (100.0/h)*u_d * v * ds(1)"  # u_d = desired value
```

### Step 5: Define Form

```python
define_variational_form(
    bilinear="kappa * inner(grad(u), grad(v)) * dx",
    linear="f * v * dx + g * v * ds(2)",
    trial_space="V",
    test_space="V"
)
```

---

## Quick Reference Table: "I want to compute X"

| Goal | UFL Expression | Notes |
|------|---|---|
| Heat diffusion (scalar) | `inner(grad(u), grad(v)) * dx` | Laplacian term |
| Advection | `dot(b, grad(u)) * v * dx` | b = velocity vector |
| Reaction | `u * v * dx` | Mass matrix term |
| Time derivative | `u/dt * v * dx` | u is time-discrete, dt is time step |
| Pressure (Stokes) | `-p * div(v) * dx` | Incompressibility constraint |
| Stress (elasticity) | `inner(sigma(u), sym(grad(v))) * dx` | sigma(u) = C:ε(u) |
| DG penalty | `alpha/h * inner(jump(u), jump(v)) * dS` | Interior penalty term |
| Nitsche penalty | `gamma/h * inner(u, v) * ds(tag)` | Weak BC term |
| Natural convection | `(u*grad(u)) * v * dx` | Nonlinear term |
| Eigenvalue (stiffness) | `inner(grad(u), grad(v)) * dx` | Bilinear for A in A*x=λ*M*x |
| Eigenvalue (mass) | `inner(u, v) * dx` | Bilinear for M in A*x=λ*M*x |

---

## Validation Checklist

Before calling `define_variational_form`:

- [ ] Bilinear form is linear in **both** u and v
- [ ] Linear form is linear in v only (no factors of u)
- [ ] All terms end with dx, ds, or dS
- [ ] Rank of all terms matches (should all be rank 0 after operators)
- [ ] Trial space and test space match (for symmetric bilinear forms)
- [ ] Boundary tags used in ds(tag) were created via mark_boundaries
- [ ] Material properties referenced in form exist via set_material_properties
- [ ] No forbidden tokens (import, exec, open, os, sys, subprocess) in expressions

---

## Example: Full Poisson Problem

```python
# 1. Create mesh and space
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1, mesh_name="mesh")

# 2. Mark boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left: x=0
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"} # Right: x=1
], name="bnd", mesh_name="mesh")

# 3. Define material and source
set_material_properties(name="f", value="10.0")
set_material_properties(name="g_neumann", value="0.0")

# 4. Define variational form
define_variational_form(
    bilinear="inner(grad(u), grad(v)) * dx",
    linear="f * v * dx + g_neumann * v * ds(2)",
    trial_space="V",
    test_space="V"
)

# 5. Apply boundary conditions
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")
apply_boundary_condition(value=1.0, boundary_tag=2, function_space="V")

# 6. Solve
solve(solver_type="direct", solution_name="u_h")
```

---

## See Also

- **operator-reference.md** — Detailed operator table with math notation
- **/advanced-boundary-conditions** — Nitsche, Robin, periodic BCs in UFL
- **/dg-formulations** — DG interior penalty forms
- **/axisymmetric-formulations** — Cylindrical coordinate adaptations
- **/pde-cookbook** — 15 PDE recipes with complete weak forms
