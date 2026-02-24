# Command: Setup Axisymmetric Problem

**Slug**: `setup-axisym`

**Description**: Configure and solve an axisymmetric FEM problem (2D meridional domain with implicit rotation about axis).

---

## Workflow

### Step 1: Confirm Problem Type

What physics does your axisymmetric problem involve?

**Options**:
- **Thermal**: Heat conduction (∂T/∂t - κ∇²T = q) in cylindrical domain
- **Elastic**: Deformation of axisymmetric structure (stress-strain in r-z plane)
- **Fluid**: Stokes or Navier-Stokes flow in pipe/channel
- **General Poisson**: Scalar diffusion in cylindrical geometry

**Your choice**: [Awaiting user input]

---

### Step 2: Understand Coordinate System

**Critical**: The 2D mesh domain is the **meridional plane** (r-z).

```
Coordinate assignment:
  x[0] = r (radial distance from z-axis, r ≥ 0)
  x[1] = z (height along axis)

Domain: r ∈ [0, r_max], z ∈ [0, z_max]

3D interpretation: Rotate meridional plane by θ ∈ [0, 2π] around z-axis
```

**Key constraint**: r ≥ 0 always. Mesh must NOT include negative r.

**Example**: For a cylinder (inner radius 0.05, outer 0.15, height 1.0):
```
mesh r-z plane: r ∈ [0.05, 0.15], z ∈ [0, 1.0]
DO NOT create mesh with r ∈ [-0.15, 0.15] (would be wrong!)
```

---

### Step 3: Create 2D Mesh in (r-z) Plane

```python
# Step 3a: Create rectangular mesh in (r-z) plane
# For simple case: r ∈ [0, 1], z ∈ [0, 1]
create_unit_square(name="mesh_axi", nx=20, ny=20)

# For custom domain (e.g., hollow cylinder):
# r ∈ [r_inner, r_outer], z ∈ [0, L]
create_mesh(name="mesh_axi", shape="rectangle", nx=20, ny=30,
            dimensions={"width": 0.1, "height": 1.0})
# Note: width = r_outer - r_inner, height = z_max
# Then translate mesh to start at r_inner (advanced)

# Step 3b: Mark boundaries (important for axisymmetric features)
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 0.05 + 1e-14"},      # Inner surface (r ≈ 0.05)
    {"tag": 2, "condition": "x[0] > 0.15 - 1e-14"},      # Outer surface (r ≈ 0.15)
    {"tag": 3, "condition": "x[1] < 1e-14"},             # Bottom (z = 0)
    {"tag": 4, "condition": "x[1] > 1.0 - 1e-14"}        # Top (z = 1.0)
], name="axi_boundaries", mesh_name="mesh_axi")

# Note: r = 0 (axis) is implicit; no explicit boundary marking needed
```

---

### Step 4: Create Function Space

**Scalar problem** (temperature, concentration):
```python
create_function_space(name="V_scalar", family="Lagrange", degree=1, mesh_name="mesh_axi")
```

**Vector problem** (displacement, velocity):
```python
create_function_space(name="V_vec", family="Lagrange", degree=1, shape=[2], mesh_name="mesh_axi")
# Components: u[0] = u_r (radial), u[1] = u_z (axial)
```

---

### Step 5: Set Up Material Properties

```python
# Scalar problem (heat conduction)
set_material_properties(name="kappa", value="1.0")        # Thermal conductivity
set_material_properties(name="q", value="0.0")            # Heat source

# Vector problem (elasticity)
set_material_properties(name="mu", value="1.0")           # Shear modulus
set_material_properties(name="lambda", value="1.0")       # Lame parameter
set_material_properties(name="f", value="[0.0, -10.0]")   # Body force [f_r, f_z]

# Time-dependent
set_material_properties(name="rho", value="1.0")          # Density
set_material_properties(name="c", value="1.0")            # Heat capacity
```

---

### Step 6: Define Axisymmetric Variational Form

**Key point**: Multiply all integrals by r = x[0] (the radial coordinate).

#### Option A: Scalar (Axisymmetric Poisson/Heat)

```python
# Step 6a: Bilinear form with r multiplier
bilinear = "inner(grad(u), grad(v))*x[0]*dx"  # Note: x[0] = r

# Step 6b: Linear form with r multiplier
linear = "f*v*x[0]*dx"

# Step 6c: Define variational form
define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V_scalar",
    test_space="V_scalar"
)
```

**Weak form interpretation**:
```
∫ r·∇u·∇v dr dz  (from 3D Cartesian integral ∫ ∇u·∇v r dr dθ dz, with ∫dθ = 2π factored into measure)
```

#### Option B: Vector (Axisymmetric Elasticity)

```python
# Critical: Include azimuthal strain term u_r/r
# This term represents the contribution of circumferential stretch to strain energy

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
    trial_space="V_vec",
    test_space="V_vec"
)
```

**Explanation of azimuthal terms**:
- Strain: ε_θθ = u_r / r (hoop strain from radial displacement)
- Stress: σ_θθ = 2μ·ε_θθ + λ·tr(ε)
- Energy contribution: ∫ σ_θθ·ε_θθ · r dr dz = ∫ [2μ(u_r/r)² + λ(u_r/r)·tr(ε)] · r dr dz

#### Option C: Time-Dependent (Transient Heat)

```python
# Backward Euler: (T^{n+1} - T^n)/Δt
dt = 0.01

bilinear = f"""
(rho*c/dt)*u*v*x[0]*dx
+ kappa*inner(grad(u), grad(v))*x[0]*dx
"""

linear = f"""
(rho*c/dt)*u_n*v*x[0]*dx
+ q*v*x[0]*dx
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V_scalar",
    test_space="V_scalar"
)
```

---

### Step 7: Apply Boundary Conditions

**Special handling at r = 0 (axis)**:
- No explicit BC needed at r = 0; it's natural (solution regular by construction)
- If symmetry requires u_r(0, z) = 0, apply it:

```python
apply_boundary_condition(value=0.0, boundary="x[0] < 1e-14", function_space="V_vec", sub_space=0)
# Force u_r = 0 on axis
```

**On r = r_max (outer boundary)**:
```python
apply_boundary_condition(value=0.0, boundary_tag=2, function_space="V_scalar")  # Dirichlet
```

**On z = 0, z = z_max**:
```python
apply_boundary_condition(value=0.0, boundary_tag=3, function_space="V_vec", sub_space=1)
# u_z = 0 at top/bottom (plane strain for elasticity)
```

---

### Step 8: Solve

**Steady-state**:
```python
solve(solver_type="direct", solution_name="u_axi_steady")
```

**Time-dependent**:
```python
solve_time_dependent(
    t_end=1.0,
    dt=0.01,
    t_start=0.0,
    time_scheme="backward_euler",
    solution_name="u_axi_final"
)
```

---

### Step 9: Visualize and Interpret Results

```python
# Step 9a: Plot solution in (r-z) plane
plot_solution(function_name="u_axi_steady", plot_type="contour", return_base64=True)
# This shows the meridional cross-section; remember it's a 2D plot of the r-z domain

# Step 9b: Evaluate at specific points
points_to_check = [[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]]  # Along vertical line
results = evaluate_solution(points=points_to_check, function_name="u_axi_steady")

# Step 9c: Verify symmetry at axis (if present)
axis_points = [[0.0, z] for z in [0.0, 0.25, 0.5, 0.75, 1.0]]
axis_results = evaluate_solution(points=axis_points, function_name="u_axi_steady")
# For elastic problems: u_r(0, z) should be 0; u_z(0, z) can be non-zero
```

**Important interpretation**:
- The plot you see is the **meridional plane** (r-z slice)
- The actual 3D structure is obtained by rotating this plot around the z-axis by θ ∈ [0, 2π]
- To visualize full 3D in ParaView: use "Rotational Symmetry" filter on the VTK/XDMF output

---

## Complete Working Example: Pressurized Cylinder

```python
# Axisymmetric elasticity: hollow cylinder with internal pressure

# Mesh: r ∈ [0.05, 0.15], z ∈ [0, 1.0]
create_mesh(name="cylinder", shape="rectangle", nx=15, ny=30,
            dimensions={"width": 0.1, "height": 1.0})

create_function_space(name="V_u", family="Lagrange", degree=1, shape=[2], mesh_name="cylinder")

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 0.05 + 1e-14"},      # Inner (pressure)
    {"tag": 2, "condition": "x[0] > 0.15 - 1e-14"},      # Outer (free)
    {"tag": 3, "condition": "x[1] < 1e-14"},             # Bottom (fixed)
    {"tag": 4, "condition": "x[1] > 1.0 - 1e-14"}        # Top (fixed)
], name="cyl_bnd", mesh_name="cylinder")

set_material_properties(name="mu", value="1.0")
set_material_properties(name="lambda", value="1.0")
set_material_properties(name="p_internal", value="1.0")

define_variational_form(
    bilinear="""
2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx
+ lambda*tr(sym(grad(u)))*tr(sym(grad(v)))*x[0]*dx
+ 2*mu*(u[0]/x[0])*(v[0]/x[0])*x[0]*dx
+ lambda*(u[0]/x[0])*tr(sym(grad(v)))*x[0]*dx
""",
    linear="-p_internal*v[0]*ds(1)"  # Traction on inner surface
)

# Fix bottom and top
apply_boundary_condition(value=0.0, boundary_tag=3, function_space="V_u", sub_space=1)
apply_boundary_condition(value=0.0, boundary_tag=4, function_space="V_u", sub_space=1)

solve(solver_type="direct", solution_name="u_cylinder_pressure")

# Verify: radial displacement should be largest at inner surface
evaluate_solution(points=[[0.05, 0.5]], function_name="u_cylinder_pressure")
# Expected: u_r > 0 (expansion)
```

---

## Tips and Common Issues

### Tip 1: Always Multiply by r = x[0]

```python
# WRONG:
bilinear = "inner(grad(u), grad(v))*dx"

# CORRECT:
bilinear = "inner(grad(u), grad(v))*x[0]*dx"
```

**Why**: In cylindrical coordinates, the volume element is r dr dθ dz. In 2D axisymmetric (r-z plane), this becomes r dr dz (with 2π factored into the time-averaged problem).

### Tip 2: Don't Forget Azimuthal Strain (Vector Problems)

```python
# INCOMPLETE (elasticity without hoop strain):
bilinear = "2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx"

# COMPLETE (includes hoop strain ε_θθ = u_r/r):
bilinear = """
2*mu*inner(sym(grad(u)), sym(grad(v)))*x[0]*dx
+ 2*mu*(u[0]/x[0])*(v[0]/x[0])*x[0]*dx  # Hoop term
+ lambda*(u[0]/x[0])*tr(sym(grad(v)))*x[0]*dx
"""
```

### Tip 3: Mesh Domain Must Have r ≥ 0

```python
# WRONG: Domain is r ∈ [-0.15, 0.15]
# This violates the axisymmetric assumption (mirrored about axis)

# CORRECT: Domain is r ∈ [0, r_max] or r ∈ [r_inner, r_outer] with r_inner ≥ 0
```

### Tip 4: Axis Singularity is Natural

No BC needed at r = 0. The weak form naturally enforces regularity:

```python
# NO need for:
apply_boundary_condition(value=0.0, boundary="x[0] < 1e-14", function_space="V")

# The integral ∫ r·f(r) dr naturally vanishes as r→0
```

### Issue: Solution Diverges at r = 0

**Symptom**: NaN or very large values near r = 0

**Cause**: Numerical instability, material properties, or bug in azimuthal term

**Fix**:
1. Verify material properties are positive definite
2. Check azimuthal strain term syntax
3. Ensure no division by r without proper handling (e.g., (u[0]/x[0]) only in elasticity context)

---

## Validation Checklist

- [ ] Mesh has r ≥ 0 (no negative radius)
- [ ] All integrals multiplied by x[0] (r coordinate)
- [ ] Azimuthal strain term included (if vector problem)
- [ ] Boundary conditions applied correctly (Dirichlet on appropriate tags)
- [ ] No BC at r = 0 (axis is natural)
- [ ] Solution evaluated at sample points to verify reasonableness
- [ ] Convergence study done (refine mesh, check O(h^p) convergence)

---

## See Also

- **axisymmetric-formulations/SKILL.md** — Full guide to axisymmetric FEM
- **cylindrical-coordinates.md** — Mathematical reference (gradients, divergence, strain in cylindrical)
- **pde-cookbook/SKILL.md** — Axisymmetric recipes (Poisson, elasticity, heat, Stokes)
- **command: setup-bc-advanced.md** — Advanced BCs applicable to axisymmetric problems
