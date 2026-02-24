# Discontinuous Galerkin (DG) Formulations

**Triggers**: "DG", "discontinuous Galerkin", "interior penalty", "SIPG", "upwind", "advection", "conservation"

**Version**: 0.1.0

**Team**: Alpha (Formulation Experts)

## Overview

Discontinuous Galerkin methods allow finite element spaces with discontinuities across element boundaries. This enables: advection-dominated problems, conservation laws, hp-adaptive methods, and unfitted domain methods.

---

## When to Use DG

### DG is Ideal For:

✓ **Advection-dominated problems** (Péclet number > 1)
✓ **Hyperbolic systems** (conservation laws, Hamilton-Jacobi)
✓ **Steep gradients** (shocks, discontinuities)
✓ **hp-adaptive refinement** (easy hanging nodes, varying degrees)
✓ **Unfitted methods** (CutFEM, moving domains)
✓ **Problems requiring upwinding** (directional bias for stability)

### DG is **Not** Ideal For:

✗ **Simple elliptic problems** (standard Galerkin is simpler and equally accurate)
✗ **Very large systems** (DG typically needs smaller h or higher degree)
✗ **Real-time applications** (DG assembly and solve is slower per DOF)

---

## DG Spaces

### Creating a DG Function Space

```python
# Create standard continuous (H1) space:
create_function_space(name="V_cg", family="Lagrange", degree=2, mesh_name="mesh")

# Create discontinuous Galerkin space:
create_function_space(name="V_dg", family="DG", degree=2, mesh_name="mesh")

# Note: "DG" family always includes interior discontinuities
# Boundary values can still be continuous or discontinuous depending on BCs
```

### Space Properties

- **L2 space**: Elements may jump across facets
- **Interior continuity**: Degrees of freedom on shared facets are **not shared** between adjacent elements
- **Boundary continuity**: Boundary values can be enforced via weak formulation (no strong Dirichlet by default)
- **DOF count**: Roughly `degree × num_cells` (vs. `degree × num_vertices` for H1)

---

## Interior Penalty Methods for Elliptic Problems

### Symmetric Interior Penalty Galerkin (SIPG)

**Motivation**: For Poisson's equation, enforce weak continuity across facets.

**Weak Dirichlet BC on interior facets**:
- Penalty term: `(α/h_F) × jump(u) × jump(v)`
- Asymmetry terms: `-⟨∇u⟩ · n · jump(v) - jump(u) · ⟨∇v⟩ · n` (ensures symmetry)

**Full SIPG Bilinear Form** (DG Poisson):

```python
# Parameters:
# α = penalty parameter (≈ degree² for stability; α ≈ 10-20 for degree=1)
# h_F = element size at facet (averaged from both sides)
# jump(u) = u(+) - u(-)
# avg(∇u) = 0.5 * (grad(u)(+) + grad(u)(-))

# Interior penalty:
penalty_interior = (alpha / h_avg) * inner(jump(u), jump(v)) * dS

# Asymmetry terms (ensure symmetry):
symmetry_interior = -inner(avg(grad(u)), jump(v*n)) * dS \
                   -inner(jump(u*n), avg(grad(v))) * dS

# Complete SIPG form:
bilinear = inner(grad(u), grad(v)) * dx \
         + penalty_interior \
         + symmetry_interior

linear = f * v * dx
```

### Penalty Parameter Selection

**Principle**: Must be large enough for stability and coercivity.

**Practical formula**:
```
α ≈ C * degree * (degree + 1) / h_F

where C ≈ 2-10 (often C=10 is safe)
```

**In code**:
```python
# Compute cell diameter (minimum dimension for stability)
# h_F = average of h_E+ and h_E- (both sides of facet)

# Example with explicit computation:
alpha = 10 * degree * (degree + 1)  # Constant penalty

# More refined: use FacetArea and CellVolume (advanced)
# h_F = avg(CellVolume / FacetArea) (area-based estimate)
```

**IMPORTANT**: h_avg must be registered as a material property before use in forms. Compute h_avg from mesh: h_avg ≈ 1/nx for unit square with nx cells. Example:
```python
# For 20x20 mesh on unit square:
set_material_properties(name="h_avg", value="0.05")  # = 1.0/20
```

### Non-symmetric Interior Penalty Galerkin (NIPG)

Omit asymmetry terms for simplicity (less accurate but sometimes used):

```python
# NIPG (not recommended; reduced convergence):
bilinear_nipg = inner(grad(u), grad(v)) * dx \
              + (alpha / h_avg) * inner(jump(u), jump(v)) * dS
```

---

## Upwinding for Advection

### Upwind Numerical Flux

**Problem**: Pure advection (∂u/∂t + ∇·(bu) = f) needs stabilization.

**Upwind flux**: Bias flux from upstream (direction of flow).

**Jump definition (advection)**:
```python
# For scalar u and velocity b:
# Upwind flux: u* = u(ω) where ω is the upwind side
#
# Implemented as:
# u_upwind = (b·n > 0) ? u(-) : u(+)
#
# Using conditional:
flux = conditional(dot(b('+'), n) > 0, u('+'), u('-'))
```

**Advection-Diffusion Weak Form**:

```python
# Parameters:
# b = velocity field (vector)
# ε = diffusion parameter (small, possibly near zero)
# u_upwind = upwind value of u

# Volume term: diffusion + advection
a = eps * inner(grad(u), grad(v)) * dx \
  + inner(b, grad(u)) * v * dx

# Jump term for advection (upwind):
# Enforces weak continuity: ⟨u_upwind, v⟩_F on interior facets
a += -inner(u_upwind, b·n) * v('+') * dS

# Boundary terms:
# On inflow (b·n < 0): impose Dirichlet value u_D
# On outflow (b·n > 0): natural (outflow)
a += -inner(u_D, dot(b, n)) * v * ds(inflow_tag)

linear = f * v * dx
```

### Lax-Friedrichs Flux (Central + Dissipation)

Alternative to upwind; less diffusive but requires global CFL:

```python
# Central flux with dissipation:
# F_LF = 0.5 * (f(u+) + f(u-)) - 0.5 * λ_max * (u(+) - u(-))
#
# where λ_max is max wave speed (for scalar: |b|)

lf_flux = 0.5 * (u('+') + u('-')) \
        - 0.5 * lambda_max * jump(u)

a += -inner(lf_flux, jump(v*b·n)) * dS
```

---

## Jump and Average Operators

### Definitions (Interior Facets Only)

```
jump(u) = u(+) - u(-)         # Difference across facet
avg(u) = (u(+) + u(-)) / 2     # Average across facet
avg(grad(u)) = (grad(u)(+) + grad(u)(-)) / 2

# Restrictions:
u(+) = value on "+" side of facet
u(-) = value on "-" side of facet
```

### Rule of Signs

- **"+" side**: exterior (right/upper when viewed normally)
- **"-" side**: interior (left/lower when viewed normally)
- Arbitrary labeling per facet; use consistently

### Example: SIPG with Explicit Restrictions

```python
# All terms must use paired restrictions (both + or both -):

# CORRECT: Both + side
a_plus = inner(grad(u('+'), grad(v('+')))) * dS

# CORRECT: Mixing with avg is OK
a_mix = -inner(avg(grad(u)), jump(v*n)) * dS

# WRONG: Mixing + and - without averaging
a_bad = inner(grad(u('+')), grad(v('-'))) * dS  # No pairing!
```

---

## DG Poisson: Complete Example

### Problem

-∇²u = f in Ω, u = 0 on ∂Ω (pure Dirichlet)

### MCP Tool Sequence

```python
# 1. Create mesh
create_unit_square(name="mesh", nx=10, ny=10)

# 2. Create DG function space
create_function_space(name="V", family="DG", degree=1, mesh_name="mesh")

# 3. Mark boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "True"}  # All boundary (we'll enforce weakly)
], name="boundary_tags", mesh_name="mesh")

# 4. Define material (source)
set_material_properties(name="f", value="10.0")

# 5. Define SIPG form
# Parameters:
alpha_param = 10.0  # penalty (degree=1, so degree*(degree+1)=2, alpha≈10 is safe)

bilinear = """
inner(grad(u), grad(v))*dx
+ (alpha/h_avg) * inner(jump(u), jump(v))*dS
- inner(avg(grad(u)), jump(v*n))*dS
- inner(jump(u*n), avg(grad(v)))*dS
+ (alpha/h_avg) * u*v*ds(1)
- inner(grad(u)*n, v)*ds(1)
- inner(u*n, grad(v))*ds(1)
"""

linear = """
f*v*dx
"""

# Need to set alpha and h_avg in material properties:
set_material_properties(name="alpha", value=alpha_param)

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# 6. Solve
solve(solver_type="direct", solution_name="u_h")
```

### Handling Cell Diameter (h_avg)

**Challenge**: `h_avg` is the average cell diameter. In UFL, use:

```python
# UFL-native (if supported):
from ufl import cell_avg, Constant
h = Constant(mesh, 0.1)  # Set manually based on mesh size

# Or use run_custom_code to compute and set:
run_custom_code(code="""
import dolfinx
# Compute h_min, h_max from mesh
h_val = dolfinx.cpp.mesh.h(mesh, 2)  # Returns array
h_avg_val = float(h_val.mean())
print(f'Average h = {h_avg_val}')
""")

# Then use constant h in forms:
bilinear = f"... + (alpha/{h_avg_val})*inner(jump(u),jump(v))*dS ..."
```

---

## Boundary Conditions in DG

### Weak Dirichlet (No Strong BCs)

In DG, boundary Dirichlet BCs are enforced **weakly** via penalty:

```python
# On boundary with tag 1, enforce u = u_D:
# Add to bilinear:
penalty_boundary = (alpha/h_avg) * u*v*ds(1)
symmetry_boundary = -inner(grad(u)*n, v)*ds(1) - inner(u*n, grad(v))*ds(1)

# Add to linear:
source_boundary = (alpha/h_avg)*u_D*v*ds(1) + inner(u_D*n, grad(v))*ds(1)
```

### Weak Neumann (Natural)

Natural boundary conditions are automatic in DG (no weak terms needed):

```python
# On boundary with Neumann: du/dn = g
# Just add to linear form:
linear = f*v*dx + g*v*ds(neumann_tag)
```

---

## Stabilization for Advection-Dominated Problems

### SUPG (Streamline-Upwind Petrov-Galerkin)

For advection-diffusion with small ε, add residual-weighted stabilization:

```python
# Parameters:
# τ = stabilization parameter (proportional to |b|/ε)
# R(u) = residual = f - ∂u/∂t - div(b*u) + ε*∇²u

# Stabilized form:
# a(u,v) + ∫ τ*R(u)*(b·∇v)*dx

# Simplified (neglecting time derivative and diffusion in stabilization):
tau = 1.0 / (|b|/h + ε/h²)  # Stabilization parameter
R = f - div(b*u) + eps*div(grad(u))

a += tau * R * dot(b, grad(v)) * dx

linear = f * v * dx + ...
```

### GLS (Galerkin Least-Squares)

Similar to SUPG but uses residual in both test and trial:

```python
# a(u,v) + ∫ τ*R(u)*R(v)*dx
# (More accurate but more expensive)

tau = 1.0 / (|b|/h + ε/h²)
R_u = f - div(b*u) + eps*div(grad(u))
R_v = f - div(b*v) + eps*div(grad(v))  # Use v as test

a += tau * R_u * R_v * dx
```

---

## Ghost Penalties (CutFEM)

For unfitted mesh methods (domain boundary doesn't align with elements):

**Motivation**: Ensure coercivity when domain intersects element interiors.

**Ghost penalty term**: Penalize jumps of normal derivatives on "ghost" facets (element boundaries internal to domain):

```python
# On interior facets inside the domain:
# Add penalty: γ*h*(1/jump(u))*jump(v)*dS

h_ghost = 0.1  # element size
gamma_ghost = 0.01

ghost_penalty = gamma_ghost * h_ghost * inner(jump(grad(u)*n), jump(grad(v)*n)) * dS

bilinear += ghost_penalty
```

---

## Error Analysis and Convergence

### DG Convergence Rates

For smooth solutions u and DG space of degree p:

| Problem | Norm | Rate | Notes |
|---|---|---|---|
| **Poisson (SIPG)** | L² | O(h^(p+1)) | Optimal |
| **Poisson (SIPG)** | H¹ semi | O(h^p) | Optimal |
| **Advection** | L² | O(h^(p+1/2)) | CFL-dependent |
| **DG for BVP** | L² | O(h^p) | Super-convergence possible |

### Convergence Verification

Use `compute_error()` tool with exact solution:

```python
exact = "sin(pi*x[0])*sin(pi*x[1])"
compute_error(exact=exact, norm_type="L2", function_name="u_h")
```

---

## Common Pitfalls and Fixes

### Pitfall 1: Forgetting Interior Penalty Terms

```python
# WRONG: Poisson in DG without penalty
a = inner(grad(u), grad(v)) * dx  # Forms not stable!

# CORRECT: Add interior penalty and boundary penalty
a = inner(grad(u), grad(v)) * dx \
  + (alpha/h) * inner(jump(u), jump(v)) * dS \
  - inner(avg(grad(u)), jump(v*n)) * dS \
  - inner(jump(u*n), avg(grad(v))) * dS \
  + (alpha/h) * u*v*ds  # Boundary penalty
```

### Pitfall 2: Wrong Jump/Average Pairing

```python
# WRONG: Unpaired restrictions
a = inner(jump(u('+'), grad(v('-')))) * dS

# CORRECT: Use avg to pair, or both + or both -
a = -inner(avg(grad(u)), jump(v*n)) * dS  # avg pairs
```

### Pitfall 3: Penalty Parameter Too Small

```python
# WRONG: α too small → coercivity lost
alpha = 0.1
a += (alpha/h) * inner(jump(u), jump(v)) * dS

# CORRECT: α ≥ C * degree²
alpha = 10.0 * degree * (degree + 1)
a += (alpha/h) * inner(jump(u), jump(v)) * dS
```

### Pitfall 4: Forgetting Boundary Penalty for Weak Dirichlet

```python
# WRONG: DG form without boundary penalty
bilinear = inner(grad(u), grad(v)) * dx + (alpha/h)*inner(jump(u), jump(v))*dS
linear = f*v*dx  # No boundary condition!

# CORRECT: Enforce Dirichlet via penalty on boundary
bilinear += (alpha/h)*u*v*ds(boundary_tag) \
          - inner(grad(u)*n, v)*ds(boundary_tag) \
          - inner(u*n, grad(v))*ds(boundary_tag)
linear += (alpha/h)*u_D*v*ds(boundary_tag) + inner(u_D*n, grad(v))*ds(boundary_tag)
```

### Pitfall 5: Upwind Flux Implementation

```python
# WRONG: Treating upwind as simple jump
u_upwind = u('+') - u('-')  # This is jump(u), not upwind!

# CORRECT: Upwind is directional (depends on flow)
u_upwind = conditional(dot(b('+'), n) > 0, u('+'), u('-'))
```

---

## Performance Considerations

| Aspect | Impact | Mitigation |
|---|---|---|
| **DOF count** | DG has ~degree × num_cells DOFs (denser than H1) | Use coarser mesh or lower degree |
| **Assembly time** | Interior penalty has dS terms (more integrals) | Accept slower assembly; parallelize |
| **Solve time** | Sparser but less structured matrix | Use iterative solver with good preconditioner |
| **Memory** | Larger matrix (denser stiffness) | Consider low-rank approximations (advanced) |

---

## Workflow: Setting Up a DG Problem

1. **Mesh**: Create unit_square or custom_mesh
2. **Space**: `create_function_space(family="DG", degree=p)`
3. **Boundaries**: `mark_boundaries()` for weak BCs
4. **Form**: Build variational form with interior penalty + boundary penalty
5. **Penalty param**: Compute or set manually (α ≈ 10 for degree 1-2)
6. **Solve**: Use direct solver for small systems, iterative for large
7. **Validate**: Check convergence via `compute_error()` on refined meshes

---

## See Also

- **dg-methods-theory.md** — Mathematical foundations (coercivity, error bounds)
- **/ufl-form-authoring** — Complete UFL reference for penalty terms
- **/pde-cookbook** — DG formulation recipes for advection, Helmholtz
- **command: setup-dg.md** — Step-by-step DG setup workflow
