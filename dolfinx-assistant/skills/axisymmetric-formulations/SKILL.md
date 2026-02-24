# Axisymmetric Formulations

**Triggers**: "axisymmetric", "cylindrical", "rotational symmetry", "revolution", "r-z plane", "pressure vessel"

**Version**: 0.1.0

**Team**: Alpha (Formulation Experts)

## Overview

Axisymmetric problems have rotational symmetry about a central axis. Instead of solving in full 3D (r, θ, z), solve in 2D meridional plane (r-z), then reconstruct 3D solution via rotation.

**Benefit**: 3D physics with 2D computation cost.

---

## When to Use Axisymmetry

✓ **Problems with axis symmetry**:
- Pipes, cylinders, pressure vessels
- Rotating machinery, disks
- Thermal conduction in rods
- Elastic deformation of axially-symmetric structures
- Stokes flow in channels with circular cross-section

✗ **NOT applicable**:
- Problems with azimuthal variation (Fourier decomposition needed instead)
- Non-axially symmetric loading (use full 3D)

---

## Coordinate System

### Setup

- **Mesh domain**: r-z plane with r ≥ 0
- **Coordinates**: x[0] = r (radius), x[1] = z (axis)
- **3D interpretation**: Rotate mesh by 2π around z-axis

```
2D mesh (r-z plane):
    z
    ↑
    |----●-------- (r, z) point
    |   r
    +---→ r
   O (axis)
```

### Key Points

- **Axis singularity**: r = 0 is **singular** (no special BC needed there; it's natural)
- **Outer boundary**: r = r_max can have BCs
- **Azimuthal direction**: θ ∈ [0, 2π] handled implicitly

---

## Weak Form Modification

### Volume Measure

In 3D Cartesian: dV = dx·dy·dz

In cylindrical: dV = r·dr·dθ·dz

Since we solve only in (r, z) plane with implicit θ integration:

```
∫₀²π ∫∫_Ω f(r,z) · r dr dz dθ = 2π · ∫∫_Ω f(r,z) · r dr dz
                                     ↑ implicit
                                     in FEM
```

**In UFL**: Multiply bilinear and linear forms by r = x[0]:

```python
# Standard weak form (2D):
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Axisymmetric form (2D with revolution):
a = inner(grad(u), grad(v)) * x[0] * dx      # x[0] = r
L = f * v * x[0] * dx                         # x[0] = r
```

### Gradient in Cylindrical

For scalar u(r, z):

```
∇u = [∂u/∂r, ∂u/∂z]^T  (components in r, z directions)

In UFL: grad(u) = [∂u/∂x[0], ∂u/∂x[1]]
```

For vector u = [u_r, u_z]:

```
grad(u) = [[∂u_r/∂r, ∂u_r/∂z],
           [∂u_z/∂r, ∂u_z/∂z]]

In UFL: grad(u)[i,j] = ∂u_i/∂x[j]
```

---

## Axisymmetric Poisson

### Problem

**2D r-z plane**: -∇²u = f (with revolution to get 3D)

Strong form:
```
-1/r · ∂/∂r(r · ∂u/∂r) - ∂²u/∂z² = f
```

Weak form (integrate by parts with r·dr·dz measure):
```
∫ r · ∇u · ∇v dr dz = ∫ f · v · r dr dz
```

### MCP Implementation

```python
# 1. Create 2D mesh in r-z plane
create_unit_square(name="mesh", nx=20, ny=20)  # r=[0,1], z=[0,1]

# 2. Create function space
create_function_space(name="V", family="Lagrange", degree=1)

# 3. Mark boundaries
# r=0 is axis (automatically handled, no BC needed)
# r=1 is outer boundary, z=0 and z=1 are top/bottom
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] > 0.9"},        # r ≈ 1 (outer)
    {"tag": 2, "condition": "x[1] < 1e-14"},      # z = 0 (bottom)
    {"tag": 3, "condition": "x[1] > 1.0 - 1e-14"} # z = 1 (top)
], name="bnd_axi")

# 4. Set materials
set_material_properties(name="f", value="10.0")

# 5. Define axisymmetric form
# Key: multiply by x[0] (which is r)
bilinear = "inner(grad(u), grad(v))*x[0]*dx"
linear = "f*v*x[0]*dx"

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# 6. Apply boundary conditions
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V", name="bc_outer")
apply_boundary_condition(value=1.0, boundary_tag=2, function_space="V", name="bc_bottom")

solve(solver_type="direct", solution_name="u_axi")

# 7. Visualize: Remember plot is in (r, z) plane
plot_solution(function_name="u_axi", plot_type="contour", return_base64=True)
```

---

## Axisymmetric Heat Conduction

### Problem

2D heat equation on axisymmetric domain: ∂T/∂t - κ∇²T = q

**Time-dependent axisymmetric form**:

```python
# Create space for time-dependent solve
create_function_space(name="V_T", family="Lagrange", degree=1)
create_function_space(name="V_T_time", family="Lagrange", degree=0)  # For time integral

# Mark boundaries (same as above)
mark_boundaries(markers=[...], name="bnd_heat")

set_material_properties(name="kappa", value="1.0")
set_material_properties(name="q", value="0.0")

# Variational form for time-dependent:
# (u_n+1 - u_n)/dt · v · r + kappa · ∇u_n+1 · ∇v · r = q · v · r
bilinear = "inner(u, v)*x[0]*dx + dt*kappa*inner(grad(u), grad(v))*x[0]*dx"
linear = "u_n*v*x[0]*dx + dt*q*v*x[0]*dx"

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V_T",
    test_space="V_T"
)

# Initial condition (u_n = initial temperature)
interpolate(expression="1.0 + 0.5*x[0]", target_space="V_T")  # u_n in session as last function

# Apply boundary conditions
apply_boundary_condition(value=1.0, boundary_tag=1, function_space="V_T")

# Time-dependent solve
solve_time_dependent(
    t_end=1.0,
    dt=0.01,
    t_start=0.0,
    time_scheme="backward_euler",
    solution_name="T_axi_final"
)
```

---

## Axisymmetric Linear Elasticity

### Problem

Elastic deformation of axially-symmetric structure. Displacement u = [u_r, u_z].

**Strain tensor** (in cylindrical):
```
ε_rr = ∂u_r/∂r
ε_zz = ∂u_z/∂z
ε_rz = 1/2 · (∂u_r/∂z + ∂u_z/∂r)
ε_θθ = u_r/r                    ← Azimuthal strain (from rotation)

In 2D UFL: need to add u_r/r term explicitly
```

**Weak form**:

```python
create_function_space(name="V_u", family="Lagrange", degree=1, shape=[2])

set_material_properties(name="mu", value="1.0")      # Shear modulus
set_material_properties(name="lambda", value="1.0")  # Lame parameter
set_material_properties(name="f", value="[0.0, -10.0]")  # Body force (gravity)

# Strain tensor in UFL (need to include azimuthal term)
# Standard sym(grad(u)): [[∂u_r/∂r, 1/2·∂u_r/∂z],
#                         [1/2·∂u_z/∂r, ∂u_z/∂z]]
#
# Add azimuthal: ε_θθ = u_r/r (extra diagonal term)
#
# Isotropic constitutive: σ = 2μ·ε + λ·tr(ε)·I

bilinear = """
2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx
+ lambda*tr(sym(grad(u)))*tr(sym(grad(v)))*x[0]*dx
+ 2*mu*(u[0]/x[0])*(v[0]/x[0])*x[0]*dx
+ lambda*(u[0]/x[0])*tr(sym(grad(v)))*x[0]*dx
"""

linear = "inner(f, v)*x[0]*dx"

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V_u",
    test_space="V_u"
)

apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V_u", sub_space=0, name="bc_ur")
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V_u", sub_space=1, name="bc_uz")

solve(solver_type="direct", solution_name="u_elastic")
```

**Note on azimuthal term**: The term `(u[0]/x[0])*(v[0]/x[0])*x[0]` represents the contribution of azimuthal strain to strain energy. This is crucial for correct elastic behavior.

---

## Axis Singularity (r = 0)

### Natural Boundary Condition

At r = 0, the weak form naturally imposes:

```
∫ r·∇u·∇v dr dz → as r→0, integrand ~ r, so contribution vanishes

Therefore: du/dr → finite (no singularity in solution)
```

**Consequence**: No explicit BC needed at r = 0; use symmetry if desired.

### Symmetry Condition (Optional)

If physical symmetry requires u_r(0, z) = 0 (displacement along axis only):

```python
apply_boundary_condition(value=0.0, boundary="x[0] < 1e-14", function_space="V_u", sub_space=0)
# Force u_r = 0 on axis
```

### Verification

Solution should not diverge at r = 0. If it does, check:
1. Material properties are well-defined
2. No negative moduli
3. Mesh includes r = 0 or comes very close

---

## Visualization and Interpretation

### Result in (r, z) Plane

DOLFINx plots show axisymmetric solution in r-z meridional plane:

```
U_max ├─────────────────┐
      │                 │
      │    SOLUTION     │
      │    CONTOUR      │
      │                 │
U_min └─────────────────┘
      0          r_max
    (axis)     (outer boundary)
```

### 3D Reconstruction

To visualize full 3D, manually rotate solution:

```python
# In post-processing:
# u_3d(r, θ, z) = u_2d(r, z)  (scalar)
# u_3d_vec(r, θ, z) = [u_r·cos(θ), u_r·sin(θ), u_z]  (vector)
#
# Export to ParaView via export_solution() — ParaView can apply
# "Rotational Symmetry" filter to visualize 3D from 2D.
```

---

## Example: Pressurized Cylinder

### Problem Setup

Elastic cylinder, inner radius r_i, outer radius r_o, internal pressure p_i.

**Boundary conditions**:
- At r = r_i: -σ_rr = p_i (internal pressure)
- At r = r_o: σ_rr = 0 (free boundary)
- At z = 0, z = L: u_z = 0 (fixed ends, plane strain assumption)
- At r = 0: u_r = 0 (axis symmetry)

### MCP Workflow

```python
# 1. Create 2D mesh in (r-z) plane
# r ∈ [r_i, r_o], z ∈ [0, L]
create_mesh(name="cylinder_mesh", shape="rectangle",
            nx=20, ny=30,
            dimensions={"width": 0.1, "height": 1.0})

# 2. Function space for displacement (vector)
create_function_space(name="V_u", family="Lagrange", degree=1, shape=[2])

# 3. Mark boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 0.05 + 1e-14"},      # Inner surface (r_i)
    {"tag": 2, "condition": "x[0] > 0.15 - 1e-14"},      # Outer surface (r_o)
    {"tag": 3, "condition": "x[1] < 1e-14"},             # Bottom (z=0)
    {"tag": 4, "condition": "x[1] > 1.0 - 1e-14"}        # Top (z=L)
], name="cylinder_bnd")

# 4. Material properties
set_material_properties(name="mu", value="80.77e9")      # Shear modulus (GPa → Pa)
set_material_properties(name="lambda", value="121.15e9") # Lame parameter
set_material_properties(name="p_internal", value="1e8")  # Internal pressure (Pa)

# 5. Variational form (axisymmetric elasticity)
bilinear = """
2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx
+ lambda*tr(sym(grad(u)))*tr(sym(grad(v)))*x[0]*dx
+ 2*mu*(u[0]/x[0])*(v[0]/x[0])*x[0]*dx
+ lambda*(u[0]/x[0])*tr(sym(grad(v)))*x[0]*dx
"""

# Natural BC on inner surface: -σ_rr = p (traction)
# Applied as: p*v_r·n_r = p*v[0]  on boundary 1
linear = """
0*v[0]*dx
- p_internal*v[0]*ds(1)
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V_u",
    test_space="V_u"
)

# 6. Boundary conditions
# Axis symmetry: u_r = 0 at r = 0 (no explicit BC needed if mesh doesn't reach r=0)
apply_boundary_condition(value=0.0, boundary_tag=3, function_space="V_u", sub_space=1, name="bc_uz_bottom")
apply_boundary_condition(value=0.0, boundary_tag=4, function_space="V_u", sub_space=1, name="bc_uz_top")

# 7. Solve
solve(solver_type="direct", solution_name="u_cylinder")

# 8. Post-process: evaluate stress
# Stress: σ = 2μ·ε + λ·tr(ε)·I
# At center of cylinder: verify analytical solution
evaluate_solution(points=[[0.1, 0.5]], function_name="u_cylinder")
```

---

## Common Mistakes

### Mistake 1: Forgetting the r Multiplier

```python
# WRONG: Standard form without r
a = inner(grad(u), grad(v))*dx

# CORRECT: Multiply by r = x[0]
a = inner(grad(u), grad(v))*x[0]*dx
```

### Mistake 2: Missing Azimuthal Strain Term (Vector Problems)

```python
# WRONG: Elasticity without azimuthal contribution
bilinear = "2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx"

# CORRECT: Add azimuthal strain term
bilinear = """
2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx
+ 2*mu*(u[0]/x[0])*(v[0]/x[0])*x[0]*dx
+ lambda*(u[0]/x[0])*tr(sym(grad(v)))*x[0]*dx
"""
```

### Mistake 3: Treating r = 0 as Special Boundary

```python
# WRONG: Applying Dirichlet BC at r = 0 like any boundary
apply_boundary_condition(value=0.0, boundary="x[0] < 1e-14", function_space="V")

# CORRECT: No BC needed at axis; if required, use symmetry explicitly
# Usually omit; natural behavior at r = 0 is correct
```

### Mistake 4: Mesh Includes Negative r

```python
# WRONG: Domain is [-r_o, r_o] × [0, L]
# This mirrors the problem; breaks axisymmetry assumption

# CORRECT: Domain is [0, r_max] × [0, z_max]
# r ≥ 0 always
```

### Mistake 5: Forgetting to Interpret Results as Revolution

```python
# WRONG: Plotting (r, z) mesh and thinking it's the actual 3D shape
plot_solution(...)  # Shows meridional cross-section only

# CORRECT: Remember you're viewing meridional plane; full 3D is obtained by rotating
# Use ParaView "Rotational Symmetry" filter to visualize full 3D
```

---

## Convergence and Validation

### Verification Against Analytical Solution

For simple cases, compare to analytical solution:

**Poisson in disk** (r ∈ [0, 1], u = 0 at r=1):
```
-∇²u = 1  →  u_analytical = (1 - r²) / 4

compute_error(exact="(1.0 - x[0]**2)/4.0", norm_type="L2")
```

### Convergence Study

```python
# Refine mesh multiple times, check convergence rate
for nx in [10, 20, 40, 80]:
    create_unit_square(name=f"mesh_{nx}", nx=nx, ny=nx)
    create_function_space(name=f"V_{nx}", family="Lagrange", degree=1)
    # ... define form, solve, compute error
    error = compute_error(exact="...", function_name="u_h")
    print(f"h={1/nx:.4f}, error={error:.6f}")
```

Expected rate: O(h^(p+1)) for degree p elements.

---

## Workflow: Setting Up Axisymmetric Problem

1. **Recognize symmetry**: Problem has rotational symmetry around z-axis
2. **Create 2D mesh**: In (r, z) plane with r ≥ 0
3. **Mark boundaries**: Distinguish inner, outer, top, bottom
4. **Set up function space**: Scalar or vector (r-z components only)
5. **Multiply forms by r = x[0]**: Bilinear and linear
6. **Add azimuthal terms** (if vector): u_r/r strain contributions
7. **Apply BCs**: At boundaries (not at r = 0; it's natural)
8. **Solve**: Direct or iterative
9. **Visualize**: Remember plot is meridional (r-z) plane; use ParaView to rotate for 3D view
10. **Validate**: Check against analytical solution or convergence study

---

## See Also

- **cylindrical-coordinates.md** — Math reference for gradients, divergence, curl in cylindrical
- **/ufl-form-authoring** — UFL syntax for building forms
- **/pde-cookbook** — Recipes for Poisson, elasticity, heat in axisymmetric domains
