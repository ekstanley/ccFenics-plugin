# Advanced Boundary Conditions

**Triggers**: "boundary condition", "BC", "Dirichlet", "Neumann", "Robin", "Nitsche", "periodic", "component-wise", "mixed BC"

**Version**: 0.1.0

**Team**: Alpha (Formulation Experts)

## Overview

Standard Dirichlet BCs (via `apply_boundary_condition()`) work for many problems. Advanced BCs include:
- **Periodic** BCs (domain wrapping)
- **Nitsche's method** (weak enforcement)
- **Robin BCs** (mixed mode: α·u + β·du/dn = g)
- **Component-wise BCs** (fixing only one component of a vector field)
- **Spatially-varying BCs** (expression-based boundary values)
- **Mixed BC scenarios** (different types on different boundaries)
- **Point constraints** (enforce value at specific DOF)
- **Interior constraints** (Lagrange multipliers)

---

## Periodic Boundary Conditions

### Motivation

Problems with periodic domains (channels, tori) need periodic wrapping.

**Example**: Couette flow with periodic boundaries at inlet/outlet.

### Implementation Strategy

Periodic BCs require DOF identification across periodic boundaries:

```
Identify DOF pairs: u_left ↔ u_right
Eliminate one, couple the other: u_left = u_right
```

### Using `run_custom_code`

Since `apply_boundary_condition()` doesn't directly support periodic BCs, use custom code:

```python
# Step 1: Create mesh and function space
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1, mesh_name="mesh")

# Step 2: Mark periodic boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left (x=0)
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"} # Right (x=1)
], name="boundaries", mesh_name="mesh")

# Step 3: Implement periodic BC via run_custom_code
run_custom_code(code="""
import dolfinx
import numpy as np

# Get mesh and function space from session
mesh = session.meshes["mesh"]
V = session.function_spaces["V"]

# Find DOFs on left boundary (tag 1) and right boundary (tag 2)
boundary_facets = session.mesh_tags["boundaries"]  # MeshTags
left_facets = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1,
                                                   np.where(boundary_facets.values == 1)[0])
right_facets = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1,
                                                    np.where(boundary_facets.values == 2)[0])

# Create periodic constraint (simplified: assuming 1-1 correspondence)
# For complex geometries, use mpc (MultiPointConstraint) from dolfinx.fem
periodic_bc = dolfinx.fem.bcs.PeriodicBC(V, left_facets, right_facets)
session.boundary_conditions["periodic_bc"] = periodic_bc
""")

# Step 4: Reference in solve
# solve() will automatically use all BCs in session
solve(solver_type="direct", solution_name="u_h")
```

**Note**: Full periodic BC implementation requires DOLFINx's `PeriodicBC` or MultiPointConstraint (MPC). This is an advanced feature not fully exposed via MCP tools.

---

## Nitsche's Method (Weak Dirichlet)

### Motivation

Enforce Dirichlet BCs weakly via penalty instead of strong elimination. Advantages:
- No need to modify system matrix (no elimination of DOFs)
- Natural integration with variational formulation
- Works seamlessly with non-conforming elements

### Mathematical Form

**Weak Dirichlet u = u_D on Γ**:

Enforce via three terms added to bilinear form:

1. **Penalty term** (penalty parameter γ):
   ```
   (γ/h) · u · v   ds(tag)
   ```

2. **Asymmetry term 1** (ensures coercivity):
   ```
   -∇u·n · v   ds(tag)
   ```

3. **Asymmetry term 2** (ensures symmetry):
   ```
   -u · ∇v·n   ds(tag)
   ```

And corresponding linear form terms:

```
(γ/h) · u_D · v   ds(tag)
+ u_D · ∇v·n   ds(tag)
```

### Complete Poisson Example

**Problem**: -∇²u = f, u = u_D on Γ_D, du/dn = 0 elsewhere.

```python
# 1. Create mesh, space, mark boundary
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"}  # Left boundary
], name="bnd", mesh_name="mesh")

# 2. Set materials
set_material_properties(name="f", value="1.0")
set_material_properties(name="gamma", value=100.0)  # Penalty parameter
set_material_properties(name="u_D", value="0.0")    # Dirichlet value

# 3. Define Nitsche form
# Note: Assumes n = outward normal; h = cell diameter (set manually or via code)
h_val = 0.1  # Average element size

gamma_over_h = f"{100.0 / h_val}"

bilinear = f"""
inner(grad(u), grad(v))*dx
- inner(grad(u)*n, v)*ds(1)
- inner(u*n, grad(v))*ds(1)
+ {gamma_over_h}*u*v*ds(1)
"""

linear = f"""
f*v*dx
+ {gamma_over_h}*u_D*v*ds(1)
+ u_D*grad(v)*n*ds(1)
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# 4. Solve (no explicit BC application needed!)
solve(solver_type="direct", solution_name="u_h")
```

### Penalty Parameter Selection

```
γ ≥ C · degree²   (for stability)

Common values:
degree=1: γ ≈ 10-20
degree=2: γ ≈ 50-100
degree=3: γ ≈ 100-200
```

**Trade-off**: Larger γ → stronger enforcement but poorer conditioning.

### Validation

Nitsche enforcement should give error O(h^p) on boundary:

```python
# Evaluate solution at boundary points
boundary_points = [[0.0, y] for y in np.linspace(0, 1, 10)]
results = evaluate_solution(
    points=boundary_points,
    function_name="u_h"
)
# Verify u_h(0, y) ≈ u_D for all y
```

---

## Robin Boundary Conditions

### Definition

Mixed boundary condition: `α·u + β·du/dn = g` on boundary Γ_R.

**Common form**: `du/dn + σ·u = g` (Robin with σ = damping/conductance).

### Weak Formulation

Integrate by parts, keep natural BC term:

```
∫ ∇u·∇v dx = ∫ f·v dx + ∫_{∂Ω} g·v - σ·u·v dS
```

where σ plays the role of spring constant or transfer coefficient.

**Full bilinear + linear**:

```
Bilinear:
  a(u,v) = ∫ ∇u·∇v dx + ∫_{Γ_R} σ·u·v ds(tag)

Linear:
  L(v) = ∫ f·v dx + ∫_{Γ_R} g·v ds(tag)
```

### Example: Heat Conduction with Convection

**Problem**: -κ∇²T = 0, -κ·dT/dn = h(T - T_ambient) on boundary.

Rearranged: dT/dn = (h/κ)·(T - T_ambient).

Weak form adds: σ·T·v + T_ambient_weighted·v terms.

```python
# Setup
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=2)
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] > 0.9"},  # Right edge (convection)
    {"tag": 2, "condition": "x[0] < 0.1"}   # Left edge (fixed)
], name="bnd")

# Materials
set_material_properties(name="kappa", value=1.0)       # Thermal conductivity
set_material_properties(name="h_conv", value=0.1)      # Convection coeff
set_material_properties(name="T_ambient", value=0.0)   # Ambient temp
sigma = f"h_conv / kappa"  # Robin coefficient

# Variational form
bilinear = f"""
kappa * inner(grad(u), grad(v))*dx
+ (h_conv/kappa)*u*v*ds(1)
"""

linear = f"""
0*v*dx
+ (h_conv/kappa)*T_ambient*v*ds(1)
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# Dirichlet on left edge (fixed temperature)
apply_boundary_condition(value=1.0, boundary_tag=2, function_space="V")

solve(solver_type="direct", solution_name="T_h")
```

---

## Component-Wise Boundary Conditions

### Motivation

For vector problems (elasticity, Stokes), fix only **one component** on a boundary.

**Example**: u = (u_x, u_y). Fix u_x = 0 but allow u_y to move.

### Using `sub_space` Parameter

```python
# Create vector function space
create_function_space(name="V", family="Lagrange", degree=1, shape=[2])
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"}  # Left boundary
], name="bnd")

# Apply BC to x-component (sub_space=0) only
apply_boundary_condition(
    value=0.0,
    boundary_tag=1,
    function_space="V",
    sub_space=0,  # Fix x-component
    name="bc_ux_zero"
)

# y-component is free (no BC applied)

# Apply BC to y-component if needed
apply_boundary_condition(
    value=0.0,
    boundary_tag=2,
    function_space="V",
    sub_space=1,  # Fix y-component on different boundary
    name="bc_uy_zero"
)

define_variational_form(
    bilinear="inner(sym(grad(u)), sym(grad(v)))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

solve(solver_type="direct", solution_name="u_h")
```

**Key**: `sub_space=0` for x, `sub_space=1` for y, `sub_space=2` for z.

---

## Spatially-Varying Boundary Values

### Dirichlet with Expression

```python
# BC value as UFL expression (coordinate-dependent):
apply_boundary_condition(
    value="sin(pi*x[1])",  # Varies along y-axis
    boundary_tag=1,
    function_space="V"
)
```

The expression string is interpolated onto the boundary DOFs.

### Neumann with Expression

For natural BC (flux prescription), add to linear form:

```python
# Set material property as expression
set_material_properties(name="g_neumann", value="10.0*x[0]*(1.0-x[0])")

# Add to linear form
linear = "f*v*dx + g_neumann*v*ds(neumann_tag)"
```

---

## Mixed Boundary Conditions

### Scenario

Different BC types on different boundaries:
- **Boundary 1** (tag 1): Dirichlet u = 0
- **Boundary 2** (tag 2): Neumann du/dn = g
- **Boundary 3** (tag 3): Robin du/dn + σu = h

### Implementation

```python
# 1. Mark all boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left: Dirichlet
    {"tag": 2, "condition": "x[0] > 1.0-1e-14"},  # Right: Neumann
    {"tag": 3, "condition": "x[1] < 1e-14"}       # Bottom: Robin
], name="all_bnd")

# 2. Define materials
set_material_properties(name="f", value="1.0")
set_material_properties(name="g_neumann", value="0.5")
set_material_properties(name="sigma_robin", value="1.0")
set_material_properties(name="h_robin", value="2.0")

# 3. Build variational form
bilinear = """
inner(grad(u), grad(v))*dx
+ sigma_robin*u*v*ds(3)
"""

linear = """
f*v*dx
+ g_neumann*v*ds(2)
+ h_robin*v*ds(3)
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# 4. Apply strong Dirichlet BC (overrides weak terms if present)
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")

# 5. Solve
solve(solver_type="direct")
```

---

## Point Constraints

### Scenario

Fix value at specific vertex/point (e.g., reference node for pressure in singular systems).

### Using `run_custom_code`

```python
# Create mesh and space
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

# Mark point constraint
run_custom_code(code="""
import numpy as np
import dolfinx

mesh = session.meshes["mesh"]
V = session.function_spaces["V"]

# Find DOF closest to (0, 0)
target_point = np.array([0.0, 0.0])
boundary_facets = dolfinx.fem.locate_dofs_geometrical(
    V,
    lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
)

# Apply constraint: u(0,0) = 0
bc_point = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 0.0),
    boundary_facets[0:1],  # Single DOF
    V
)
session.boundary_conditions["bc_point"] = bc_point
""")

apply_boundary_condition(value=0.0, boundary="True", function_space="V")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

solve(solver_type="direct")
```

---

## Interior Constraints (Lagrange Multipliers)

### Scenario

Enforce constraint u = u_c in subdomain Ω_c (not on boundary).

**Example**: Force u = 1.0 in circle of radius 0.25.

### Advanced: Mixed Formulation

This requires a **mixed space** with Lagrange multiplier field λ.

```python
# Create primary and multiplier spaces
create_function_space(name="V_u", family="Lagrange", degree=1)
create_function_space(name="V_lam", family="Lagrange", degree=0)  # Lower degree
create_mixed_space(name="W", subspaces=["V_u", "V_lam"])

# Mark subdomain
mark_boundaries(markers=[
    {"tag": 1, "condition": "(x[0]**2 + x[1]**2) < 0.25**2"}
], name="constraint_region")

# Mixed variational form
# min_u max_λ [∫ ∇u·∇v + λ*(u-u_c)*v + μ*(lambda_λ) dx + ...]
bilinear = """
inner(grad(split(u)[0]), grad(split(v)[0]))*dx(1)  # Outside constraint
+ split(u)[1]*split(v)[0]*dx(1)  # Lagrange term
+ split(u)[0]*split(v)[1]*dx(1)  # Dual of Lagrange
"""

linear = """
f*split(v)[0]*dx + u_constraint*split(v)[1]*dx
"""

define_variational_form(bilinear=bilinear, linear=linear, trial_space="W", test_space="W")
solve(solver_type="direct", solution_name="u_lam")
```

**Note**: Full Lagrange multiplier implementation is advanced; consult FEniCS documentation.

---

## Common Mistakes

### Mistake 1: Wrong Boundary Tag

```python
# WRONG: Using tag that wasn't created
apply_boundary_condition(value=0.0, boundary_tag=99, function_space="V")
# Fails because tag 99 doesn't exist

# CORRECT: Mark boundary first
mark_boundaries(markers=[{"tag": 1, "condition": "x[0] < 1e-14"}], name="bnd")
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")
```

### Mistake 2: Forgetting Sub-space Index

```python
# WRONG: On vector space, applying BC without specifying component
create_function_space(name="V", family="Lagrange", degree=1, shape=[2])
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")
# Applies to entire vector (both components)

# CORRECT: Specify sub_space
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V", sub_space=0)
# Applies only to x-component
```

### Mistake 3: Conflicting Weak and Strong BCs

```python
# WRONG: Both weak (Nitsche) and strong Dirichlet on same boundary
bilinear = "... + (gamma/h)*u*v*ds(1)"  # Weak Dirichlet in form
linear = "... + (gamma/h)*u_D*v*ds(1)"

apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")  # Strong!
# System is over-constrained

# CORRECT: Choose one approach
# Option A: Pure Nitsche (weak only, no strong BC)
# Option B: Pure strong (BC via apply_boundary_condition, no weak terms)
```

### Mistake 4: Wrong Robin Coefficient Sign

```python
# WRONG: Adding Robin as diffusion instead of damping
linear = "... - h_robin*v*ds(3)"  # Wrong sign!

# CORRECT: Robin adds mass (positive coefficient)
linear = "... + h_robin*v*ds(3)"  # Positive Robin term
```

### Mistake 5: Nitsche Penalty Too Weak

```python
# WRONG: Small penalty → poor Dirichlet enforcement
gamma = 0.1  # Too small!
bilinear = f"... + (gamma/h)*u*v*ds(1)"
# BC value not enforced; solution drifts

# CORRECT: Adequate penalty
gamma = 100.0  # For degree=1
bilinear = f"... + (gamma/h)*u*v*ds(1)"
```

---

## Workflow: Choosing BC Type

**Decision Tree**:

```
Is BC essential (inherent to problem)?
  ├─ Yes (Dirichlet) → use apply_boundary_condition()
  └─ No (Natural/flux) → build into weak form (add to linear)

Does problem allow strong elimination?
  ├─ Yes → apply_boundary_condition()
  └─ No (unfitted mesh, DG) → use Nitsche's method

Need weak enforcement (no DOF elimination)?
  ├─ Yes → Nitsche with penalty
  └─ No → strong Dirichlet

Is BC spatially-varying?
  ├─ Yes → use expression in apply_boundary_condition() or set_material_properties()
  └─ No → scalar value

Vector space with component-wise BC?
  ├─ Yes → add sub_space=[0/1/2] to apply_boundary_condition()
  └─ No → standard BC

Multiple BC types on different regions?
  ├─ Yes → mark_boundaries() with different tags, apply BC per tag
  └─ No → single mark_boundaries() call
```

---

## Validation Checklist

- [ ] All boundaries used in forms (ds(tag), etc.) were marked via mark_boundaries()
- [ ] Dirichlet tags have corresponding apply_boundary_condition() calls
- [ ] Neumann/Robin contributions are in the linear form
- [ ] Weak Dirichlet (Nitsche) has penalty + asymmetry terms
- [ ] Penalty parameter is appropriate for degree (γ ≈ 10-100*degree²)
- [ ] No conflicting weak/strong BCs on same boundary
- [ ] Vector problems specify sub_space for component-wise BCs
- [ ] Spatially-varying expressions use valid coordinate variables (x[0], x[1], x[2])

---

## See Also

- **bc-patterns.md** — 10 BC implementation patterns with full examples
- **/ufl-form-authoring** — Weak form syntax for Neumann/Robin terms
- **/dg-formulations** — Weak BC enforcement in DG (no strong BCs used)
- **command: setup-bc-advanced.md** — Interactive BC setup workflow
