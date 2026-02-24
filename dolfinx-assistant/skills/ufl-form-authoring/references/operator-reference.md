# UFL Operator Reference Card

**Version**: 0.1.0

---

## Operator Summary Table

| Operator | Input Ranks | Output Rank | Example | Mathematical Notation | Notes |
|---|---|---|---|---|---|
| `inner(a, b)` | Any | 0 | `inner(u, v)` | ⟨a, b⟩ | Full contraction of all indices |
| `dot(a, b)` | 1, 1 | 0 | `dot(u, v)` | a·b | Alias for inner on vectors |
| `outer(a, b)` | 1, 1 | 2 | `outer(u, v)` | u⊗v | Dyadic product |
| `cross(a, b)` | 1, 1 | 1 | `cross(u, v)` | u×v | Cross product (3D) |
| `grad(a)` | 0→1, 1→2, n→n+1 | rank+1 | `grad(u)` | ∇u | Gradient (spatial derivative) |
| `div(a)` | 1→0, 2→1, n→n−1 | rank−1 | `div(sigma)` | ∇·σ | Divergence |
| `curl(a)` | 1→1 (3D) | 1 | `curl(E)` | ∇×E | Curl (3D); 2D specialized |
| `tr(A)` | 2 | 0 | `tr(A)` | tr(A) | Trace: Σᵢ Aᵢᵢ |
| `det(A)` | 2 | 0 | `det(F)` | det(A) | Determinant |
| `inv(A)` | 2 | 2 | `inv(A)` | A⁻¹ | Matrix inverse |
| `cofac(A)` | 2 | 2 | `cofac(F)` | cof(A) | Cofactor matrix |
| `dev(A)` | 2 | 2 | `dev(sigma)` | A - (1/3)tr(A)I | Deviatoric (traceless) part |
| `sym(A)` | 2 | 2 | `sym(grad(u))` | (A + Aᵀ)/2 | Symmetric part |
| `skew(A)` | 2 | 2 | `skew(grad(u))` | (A - Aᵀ)/2 | Skew-symmetric part |
| `transpose(A)` | 2 | 2 | `transpose(A)` | Aᵀ | Matrix transpose |
| `elem_mult(a, b)` | same | same | `elem_mult(A, B)` | A ⊙ B | Element-wise (Hadamard) product |
| `elem_div(a, b)` | same | same | `elem_div(A, B)` | A ⊘ B | Element-wise division |
| `elem_op(f, a)` | any | any | `elem_op(sqrt, A)` | f(A) | Apply function element-wise |
| `nabla_grad(u)` | any | rank+1 | `nabla_grad(u)` | ∇u | Explicit gradient (rare) |
| `nabla_div(u)` | any | rank−1 | `nabla_div(u)` | ∇·u | Explicit divergence (rare) |
| `conditional(cond, a, b)` | bool, any, any | same as a/b | `conditional(x[0]<0.5, 10, 1)` | χ_cond * a + χ_¬cond * b | Piecewise definition |
| `jump(u)` | any | any | `jump(u)` on dS | u(+) - u(-) | Facet jump (DG only) |
| `avg(u)` | any | any | `avg(u)` on dS | (u(+) + u(-))/2 | Facet average (DG only) |

---

## Detailed Operator Descriptions

### 1. `inner(a, b)` — Inner Product

**Definition**: Full contraction of all free indices.

**Input Requirements**: Same rank and compatible dimensions.

**Output**: Scalar (rank 0).

**Examples**:
```python
# Vector inner product: ⟨u, v⟩ = Σᵢ uᵢvᵢ
a = inner(u, v) * dx  # If u, v are rank 1, result is scalar

# Matrix inner product: ⟨A, B⟩ = Σᵢⱼ AᵢⱼBᵢⱼ
a = inner(sigma, epsilon) * dx  # If both rank 2

# Frobenius norm squared: ||A||²ꜰ = ⟨A, A⟩
a = inner(A, A) * dx
```

**When to use**: Bilinear and linear forms in almost all PDEs.

**Pitfall**: Do NOT use `*` for dot product on vectors; use `inner()` or `dot()`.

---

### 2. `dot(a, b)` — Dot Product (Vectors Only)

**Definition**: Same as `inner` but typically applied to vectors (rank 1).

**Input**: Two vectors (rank 1).

**Output**: Scalar (rank 0).

**Examples**:
```python
# Equivalent to inner for vectors:
a = dot(grad(u), grad(v)) * dx  # Same as inner(grad(u), grad(v)) * dx

# Velocity · force:
L = dot(f, v) * dx  # f and v both vectors
```

**When to use**: When you want to emphasize vector operations; otherwise `inner()` is preferred.

---

### 3. `outer(a, b)` — Outer/Dyadic Product

**Definition**: u ⊗ v with components (u ⊗ v)ᵢⱼ = uᵢvⱼ.

**Input**: Two vectors (rank 1 each).

**Output**: Matrix (rank 2).

**Examples**:
```python
# Stress from deformation gradient (rank-2 tensor):
F = grad(u)  # Rank 2
P = mu * F + lambda * tr(F) * Identity(2)  # Piola-Kirchhoff stress

# Rank-1 outer product (rarely used in standard FEM):
A = outer(u, v)  # Result: 2×2 or 3×3 matrix
```

**When to use**: Rare in standard FEM; used in hyperelasticity and tensor manipulations.

---

### 4. `cross(a, b)` — Cross Product (3D)

**Definition**: u × v (only in 3D).

**Input**: Two vectors (rank 1, length 3).

**Output**: Vector (rank 1, length 3).

**Examples**:
```python
# Angular momentum: L = r × p
L = cross(x_vec, p_vec)

# Curl in electromagnetics:
curl_E = curl(E)  # Equivalent to ∇ × E (in 3D)
```

**Note**: In 2D, cross product is either undefined or computed via embedded 3D.

---

### 5. `grad(u)` — Gradient

**Definition**: ∇u = [∂u/∂x, ∂u/∂y, ∂u/∂z]

**Input Rank**: Any (n).

**Output Rank**: n+1.

**Examples**:
```python
# Scalar field gradient (rank 0 → 1):
u_scalar = ...  # Scalar function
du = grad(u_scalar)  # Vector: (∂u/∂x, ∂u/∂y)

# Vector field gradient (rank 1 → 2):
u_vector = ...  # Vector function (2D: (u_x, u_y))
Du = grad(u_vector)  # Matrix: ∇u with Du[i,j] = ∂u_i/∂x_j

# Strain tensor:
epsilon = sym(grad(u_vector))  # Symmetric part of velocity gradient
```

**Key Rule**: Always available; one of the most frequent operators in FEM.

---

### 6. `div(u)` — Divergence

**Definition**: ∇·u = ∂u_x/∂x + ∂u_y/∂y + ... (sum of diagonal partial derivatives).

**Input Rank**: 1 → 0 (vector to scalar), 2 → 1 (matrix to vector), etc.

**Output Rank**: Input rank − 1.

**Examples**:
```python
# Vector divergence (rank 1 → 0):
u_vec = ...  # Vector field
div_u = div(u_vec)  # Scalar: ∂u_x/∂x + ∂u_y/∂y

# Divergence of stress tensor (rank 2 → 1):
sigma = ...  # Stress matrix
div_sigma = div(sigma)  # Vector: force balance ∇·σ

# Incompressibility constraint in Stokes:
L += -p * div(v) * dx  # p is scalar, v is vector (velocity)
```

**Key Rule**: Used for divergence-free constraints (incompressible flow) and force balance (elasticity).

---

### 7. `curl(u)` — Curl

**Definition**: ∇ × u (vorticity in 2D/3D).

**Input Rank**: 1 (vector).

**Output Rank**: 1 (vector, 3D).

**Examples**:
```python
# 3D curl (rank 1 → 1):
u_3d = ...  # 3D vector field
vort = curl(u_3d)  # ∇ × u

# 2D curl (scalar vorticity):
# In 2D, curl often returns a scalar (ω = ∂v/∂x - ∂u/∂y)
# Not directly available in standard UFL; use components
```

**Note**: Primarily used in electromagnetics (Maxwell equations) and vorticity analysis.

---

### 8. `tr(A)` — Trace

**Definition**: tr(A) = Σᵢ Aᵢᵢ (sum of diagonal elements).

**Input Rank**: 2 (matrix).

**Output Rank**: 0 (scalar).

**Examples**:
```python
# Trace of strain (dilatation):
I1 = tr(epsilon)  # First invariant

# Deviatoric decomposition:
A = sym(grad(u))  # Strain tensor
p = (1/3) * tr(A)  # Pressure (volumetric part)
dev_A = dev(A)     # Deviatoric (shear part)
```

---

### 9. `det(A)` — Determinant

**Definition**: det(A) = Πᵢ λᵢ (product of eigenvalues).

**Input Rank**: 2 (matrix).

**Output Rank**: 0 (scalar).

**Examples**:
```python
# Jacobian of deformation gradient:
F = grad(u)  # Deformation gradient
J = det(F)   # Volume change

# Neo-Hookean energy (hyperelasticity):
psi = (mu/2) * (J**(-2/3) * tr(F.T*F) - 3) + (lambda/2) * (J - 1)**2
```

---

### 10. `inv(A)` — Inverse

**Definition**: A⁻¹ such that A·A⁻¹ = I.

**Input Rank**: 2 (matrix).

**Output Rank**: 2 (matrix).

**Examples**:
```python
# Inverse of deformation gradient:
F = grad(u)
F_inv = inv(F)

# Inverse of Cauchy-Green:
C = F.T * F
C_inv = inv(C)
```

**Warning**: Use only in nonlinear forms; linear solvers may struggle with implicit inverts.

---

### 11. `cofac(A)` — Cofactor Matrix

**Definition**: Cofactor of A; related to adjugate matrix.

**Input Rank**: 2 (matrix).

**Output Rank**: 2 (matrix).

**Relation**: cofac(A) = det(A) * inv(A).T

**Examples**:
```python
# Often appears in hyperelasticity (via pull-back):
F = grad(u)
Cof_F = cofac(F)

# Used in constitutive laws for mixed formulations
```

---

### 12. `dev(A)` — Deviatoric Part

**Definition**: A_dev = A - (1/3)·tr(A)·I (removes hydrostatic pressure).

**Input Rank**: 2 (matrix).

**Output Rank**: 2 (matrix).

**Examples**:
```python
# Stress decomposition:
sigma = 2*mu*sym(grad(u)) + lambda*tr(sym(grad(u)))*Identity(dim)
sigma_dev = dev(sigma)  # Shear stress only
p = (1/3) * tr(sigma)   # Pressure

# J2 plasticity:
J2 = (1/2) * inner(dev(sigma), dev(sigma))
```

---

### 13. `sym(A)` — Symmetric Part

**Definition**: A_sym = (A + Aᵀ)/2.

**Input Rank**: 2 (matrix).

**Output Rank**: 2 (matrix).

**Examples**:
```python
# Strain tensor from velocity field:
epsilon = sym(grad(u))  # Symmetric strain

# Viscoelastic stress:
sigma = 2*mu*sym(grad(u)) + lambda*div(u)*Identity(dim)

# In solid mechanics, strain is always symmetric
A_sym = sym(A)
```

---

### 14. `skew(A)` — Skew-Symmetric Part

**Definition**: A_skew = (A - Aᵀ)/2 (rigid rotation, no strain).

**Input Rank**: 2 (matrix).

**Output Rank**: 2 (matrix).

**Property**: A = sym(A) + skew(A) (decomposition).

**Examples**:
```python
# Extract rotation from gradient:
Du = grad(u)
epsilon = sym(Du)   # Strain (symmetric)
omega = skew(Du)    # Rotation (skew, no contribution to work)

# Verify decomposition:
assert Du == epsilon + omega
```

---

### 15. `transpose(A)` — Transpose

**Definition**: Aᵀ with components (Aᵀ)ᵢⱼ = Aⱼᵢ.

**Input Rank**: 2 (matrix).

**Output Rank**: 2 (matrix).

**Examples**:
```python
# Right Cauchy-Green strain:
F = grad(u)
C = F.T * F

# Stiffness symmetry:
a = inner(C, sym(grad(v))) * dx  # C likely symmetric
```

---

### 16. `elem_mult(A, B)` — Element-wise Multiplication

**Definition**: (A ⊙ B)ᵢⱼ = AᵢⱼBᵢⱼ (Hadamard product).

**Input**: Operands of same shape.

**Output**: Same shape as inputs.

**Examples**:
```python
# Piecewise material property:
kappa = conditional(x[0] < 0.5, 100.0, 1.0)
a = kappa * inner(grad(u), grad(v)) * dx  # Scalar multiplication (preferred)

# Matrix-wise scaling:
# (Rare; usually use scalar multiplication instead)
```

**Note**: For scalars, this is equivalent to `*` operator. Not commonly used in standard FEM.

---

### 17. `elem_div(A, B)` — Element-wise Division

**Definition**: (A ⊘ B)ᵢⱼ = Aᵢⱼ / Bᵢⱼ.

**Input**: A and B of same shape.

**Output**: Same shape.

**Warning**: Risk of division by zero.

**Examples**:
```python
# Rarely used; usually handled via conditional
result = conditional(B != 0, elem_div(A, B), 0)
```

---

### 18. `conditional(cond, val_true, val_false)` — Piecewise Definition

**Definition**: Returns val_true where cond is True, val_false elsewhere.

**Inputs**: Boolean condition, two values (any rank).

**Output**: Same rank as values.

**Examples**:
```python
# Piecewise constant coefficient:
kappa = conditional(x[0] < 0.5, 100.0, 1.0)
a = kappa * inner(grad(u), grad(v)) * dx

# Piecewise material in domain:
mu = conditional(
    (x[0]**2 + x[1]**2) < 0.25,  # Inside circle
    1e2,
    1.0
)

# Enforce one-way coupling:
source = conditional(x[1] > 0.5, f_upper, f_lower)
```

**Performance**: Conditional expressions are computed everywhere; expensive conditions should be minimized.

---

### 19. `jump(u)` — Facet Jump

**Definition**: jump(u) = u(+) - u(-) (difference across interior facet).

**Valid On**: Interior facet integrals (dS) only.

**Input Rank**: Any.

**Output Rank**: Same as input.

**Examples**:
```python
# DG interior penalty:
# Assumes u(+) and u(-) are defined on opposite sides of facet
penalty = alpha/h_avg * inner(jump(u), jump(v)) * dS

# Jumping gradient (flux difference):
flux_jump = jump(grad(u) * n)
```

**Required Context**: Must be inside `* dS` integral; undefined for `* dx` or `* ds`.

---

### 20. `avg(u)` — Facet Average

**Definition**: avg(u) = (u(+) + u(-))/2 (average across interior facet).

**Valid On**: Interior facet integrals (dS) only.

**Input Rank**: Any.

**Output Rank**: Same as input.

**Examples**:
```python
# Stabilized DG form:
central_flux = avg(grad(u)) * jump(v*n) * dS

# Averaging of function values:
u_avg = avg(u)
```

**Note**: Often paired with `jump()` for consistent DG forms.

---

## Tensor Rank Reference

| Input Type | Rank | Example | grad() → | div() → | inner(·,·) → |
|---|---|---|---|---|---|
| Scalar | 0 | u (pressure) | Rank 1 | N/A | Rank 0 |
| Vector | 1 | u (velocity) | Rank 2 | Rank 0 | Rank 0 |
| Matrix | 2 | σ (stress), ε (strain) | Rank 3 | Rank 1 | Rank 0 |

---

## Operator Interaction Rules

### Rule 1: Linearity

All operators are linear:
```python
grad(a*u + b*v) = a*grad(u) + b*grad(v)
div(a*A + b*B) = a*div(A) + b*div(B)
```

### Rule 2: Composition

Operators can be composed (subject to rank compatibility):
```python
div(grad(u))  # Laplacian (on scalar u)
div(sym(grad(u)))  # Laplacian (on vector u)
grad(div(sigma))  # Gradient of divergence (advanced)
```

### Rule 3: Index Contraction Hierarchy

When multiple operators appear:
1. Compute all gradients and divergences
2. Then apply contractions (inner, dot, etc.)
3. Finally integrate

```python
# This computes ∇u · ∇v (both gradients first, then inner product)
a = inner(grad(u), grad(v)) * dx
```

---

## Common Weak Form Patterns

| PDE | Weak Form (Bilinear + Linear) |
|---|---|
| **Poisson** | `a = inner(grad(u), grad(v))*dx`; `L = f*v*dx` |
| **Helmholtz** | `a = inner(grad(u), grad(v))*dx - k**2*inner(u,v)*dx`; `L = f*v*dx` |
| **Elasticity** | `a = inner(sigma(u), sym(grad(v)))*dx`; `L = dot(f, v)*dx` (sigma = C:ε) |
| **Stokes** | `a = inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx`; `L = dot(f,v)*dx` |
| **Heat/Advection-Diffusion** | `a = inner(grad(u), grad(v))*dx + dot(b, grad(u))*v*dx`; `L = f*v*dx` |
| **DG Poisson** | `a = inner(grad(u),grad(v))*dx + penalty*inner(jump(u),jump(v))*dS + ...` |

---

## Performance Tips

1. **Reuse operators**: Store computed quantities (e.g., `epsilon = sym(grad(u))`) to avoid recomputation.
2. **Avoid expensive conditionals**: Use `conditional()` sparingly in hot forms.
3. **Use `avg()` and `jump()` only on dS**: Undefined elsewhere; will cause errors.
4. **Check operator precedence**: Use parentheses liberally for clarity.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `inner(A, b)` (rank 2, rank 1) | Rank mismatch | Use `dot(A[i,:], b)` or `A @ b` notation |
| Jumps/averages on `dx` | Wrong measure | Use `dS` for interior facets |
| `grad(p)` undefined | p not a UFL function | Define p via `TrialFunction()` or `Function()` |
| Nonlinear term in bilinear | Bilinearity violated | Move to linear form or define as nonlinear problem |

---

## See Also

- **ufl-form-authoring/SKILL.md** — Full guide to UFL composition
- **pde-cookbook/weak-form-derivations.md** — Derivations for 15 PDEs
