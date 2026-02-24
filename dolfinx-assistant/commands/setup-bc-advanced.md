# Command: Setup Advanced Boundary Conditions

**Slug**: `setup-bc-advanced`

**Description**: Configure complex boundary conditions (Nitsche, Robin, component-wise, periodic, mixed) with proper weak form integration.

---

## Workflow

### Step 1: Choose BC Strategy

What type(s) of boundary conditions does your problem have?

**Options**:
- **Periodic**: Domain wraps around (inlet = outlet)
- **Nitsche (weak Dirichlet)**: No DOF elimination; penalty-based enforcement
- **Robin**: Mixed mode: du/dn + α·u = g
- **Component-wise**: Vector field with selective constraints
- **Mixed**: Multiple BC types on different boundaries
- **Spatially-varying**: Expression-based (e.g., u = sin(πy) on boundary)

**Your choice** (primary): [Awaiting user input]

---

### Step 2: Mark All Boundaries

Identify and tag each boundary region.

```python
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},            # Left
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"},      # Right
    {"tag": 3, "condition": "x[1] < 1e-14"},            # Bottom
    {"tag": 4, "condition": "x[1] > 1.0 - 1e-14"}       # Top
], name="all_boundaries", mesh_name="mesh")
```

**Your boundaries**:
- Boundary 1: [Description and tag]
- Boundary 2: [Description and tag]
- Boundary 3: (if needed)
- Boundary 4: (if needed)

**Verify** all boundaries used in `ds(tag)` or `apply_boundary_condition()` are marked.

---

### Step 3: Configure BC by Type

#### Option A: Periodic Boundary Conditions

**Concept**: Identify DOFs on "left" and couple them to "right" boundary.

```python
# Currently requires custom code (not fully exposed via MCP tools)
run_custom_code(code="""
import dolfinx
import numpy as np

mesh = session.meshes["mesh"]
V = session.function_spaces["V"]

# Find DOFs on left (tag 1) and right (tag 2) boundaries
left_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1,
                                                np.where(boundary_tags.values == 1)[0])
right_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1,
                                                 np.where(boundary_tags.values == 2)[0])

# Create periodic BC (if using PeriodicBC):
periodic_bc = dolfinx.fem.PeriodicBC(V, left_dofs, right_dofs)
session.boundary_conditions["periodic"] = periodic_bc
""")
```

**Next**: Go to Step 5 (Solve)

---

#### Option B: Nitsche Weak Dirichlet

**Concept**: Enforce u = u_D on boundary via penalty terms in weak form (no DOF elimination).

**Advantages**: Smooth dependence on u_D; no system modification.

**Setup**:

```python
# Step 3b(i): Choose boundary tag and penalty parameter
boundary_tag = 1  # Which boundary
u_D_value = 0.0   # Desired value
h_elem = 1.0 / 20  # Element size
gamma = 100.0      # Penalty parameter (≈ 10-100 · degree²)

set_material_properties(name="gamma", value=str(gamma))
set_material_properties(name="u_D", value=str(u_D_value))

# Step 3b(ii): Add Nitsche terms to bilinear form
bilinear = f"""
inner(grad(u), grad(v))*dx
- inner(grad(u)*n, v)*ds({boundary_tag})
- inner(u*n, grad(v))*ds({boundary_tag})
+ (gamma/{h_elem})*u*v*ds({boundary_tag})
"""

# Step 3b(iii): Add Nitsche RHS terms to linear form
linear = f"""
f*v*dx
+ (gamma/{h_elem})*u_D*v*ds({boundary_tag})
+ u_D*grad(v)*n*ds({boundary_tag})
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# NO apply_boundary_condition() call — BC is in the form!
```

**Verification**: Solution should approach u_D on boundary as γ increases.

**Next**: Go to Step 5 (Solve)

---

#### Option C: Robin Boundary Condition

**Concept**: Mixed BC: du/dn + α·u = g

**Setup**:

```python
# Step 3c(i): Identify Robin boundary
robin_tag = 3  # Which boundary
alpha_robin = 2.0  # Robin parameter (damping/conductance)
g_robin = 1.0      # RHS value

set_material_properties(name="alpha_robin", value=str(alpha_robin))
set_material_properties(name="g_robin", value=str(g_robin))

# Step 3c(ii): Add Robin terms to bilinear form
bilinear = f"""
inner(grad(u), grad(v))*dx
+ alpha_robin*u*v*ds({robin_tag})
"""

# Step 3c(iii): Add Robin RHS to linear form
linear = f"""
f*v*dx
+ g_robin*v*ds({robin_tag})
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# NO apply_boundary_condition() for Robin tag
```

**Verify**: On boundary, du/dn + α·u should equal g_robin.

**Next**: Go to Step 5 (Solve)

---

#### Option D: Component-Wise (Vector Fields)

**Concept**: For vector u = [u_x, u_y], fix only u_x on a boundary.

**Setup**:

```python
# Step 3d(i): Ensure you have a vector function space
create_function_space(name="V_vec", family="Lagrange", degree=1, shape=[2])

# Step 3d(ii): Apply BC to x-component only (sub_space=0)
apply_boundary_condition(
    value=0.0,                           # Value to fix
    boundary_tag=1,                      # Which boundary
    function_space="V_vec",              # Vector space
    sub_space=0,                         # 0=x, 1=y, 2=z
    name="bc_ux_fixed"
)

# Step 3d(iii): If needed, apply different BC to y-component
apply_boundary_condition(
    value=1.0,
    boundary_tag=2,
    function_space="V_vec",
    sub_space=1,
    name="bc_uy_fixed"
)

define_variational_form(
    bilinear="2*mu*inner(sym(grad(u)), sym(grad(v)))*dx",
    linear="inner(f, v)*dx",
    trial_space="V_vec",
    test_space="V_vec"
)
```

**Verify**: Solution has u_x = 0 on boundary 1, but u_y is free there.

**Next**: Go to Step 5 (Solve)

---

#### Option E: Mixed Dirichlet-Neumann-Robin

**Concept**: Different BC types on different boundaries.

**Setup**:

```python
# Step 3e(i): Mark all boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Dirichlet
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"},# Neumann
    {"tag": 3, "condition": "x[1] < 1e-14"}       # Robin
], name="mixed_bnd")

# Step 3e(ii): Define material properties
set_material_properties(name="f", value="1.0")
set_material_properties(name="g_neumann", value="0.5")
set_material_properties(name="alpha_robin", value="1.0")
set_material_properties(name="g_robin", value="2.0")

# Step 3e(iii): Build combined bilinear form
bilinear = """
inner(grad(u), grad(v))*dx
+ alpha_robin*u*v*ds(3)
"""

# Step 3e(iv): Build combined linear form
linear = """
f*v*dx
+ g_neumann*v*ds(2)
+ g_robin*v*ds(3)
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# Step 3e(v): Apply strong Dirichlet only
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")
# Neumann (tag 2) and Robin (tag 3) are implicit in the form
```

**Verify**:
- Boundary 1 (tag 1): u = 0 (Dirichlet) ✓
- Boundary 2 (tag 2): du/dn = g_neumann (Neumann) ✓
- Boundary 3 (tag 3): du/dn + α·u = g_robin (Robin) ✓

**Next**: Go to Step 5 (Solve)

---

#### Option F: Spatially-Varying Dirichlet

**Concept**: u = u(x, y) on boundary (expression-based).

**Setup**:

```python
# Step 3f(i): Provide boundary value as expression string
boundary_value = "sin(pi*x[1])"  # Varies along y-axis

# Step 3f(ii): Apply Dirichlet with expression
apply_boundary_condition(
    value=boundary_value,
    boundary_tag=1,
    function_space="V"
)

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)
```

**Verify**: Evaluate solution at boundary to check agreement with expression.

**Next**: Go to Step 5 (Solve)

---

### Step 4: Validate BC Configuration

**Checklist**:
- [ ] All boundaries in `ds(tag)` are marked via `mark_boundaries()`
- [ ] All Dirichlet tags have corresponding `apply_boundary_condition()` calls
- [ ] Neumann/Robin terms are in bilinear and linear forms (no strong BC)
- [ ] Weak Dirichlet (Nitsche) has penalty parameter γ set appropriately
- [ ] No conflicting strong + weak BCs on same boundary
- [ ] All material properties referenced in form are defined via `set_material_properties()`
- [ ] For vector problems, `sub_space` parameter specified for component-wise BCs

---

### Step 5: Solve and Validate

```python
# Step 5a: Solve
solve(solver_type="direct", solution_name="u_advanced_bc")

# Step 5b: Evaluate at boundary points
boundary_points = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]  # On tag 1
results = evaluate_solution(points=boundary_points, function_name="u_advanced_bc")
# Verify: u values should match expected BCs

# Step 5c: Check internal solution
interior_points = [[0.5, 0.5], [0.25, 0.75]]
results_interior = evaluate_solution(points=interior_points, function_name="u_advanced_bc")
# Verify: solution is smooth and reasonable

# Step 5d: Compute error (if exact solution known)
compute_error(exact="sin(pi*x[0])*sin(pi*x[1])", norm_type="L2", function_name="u_advanced_bc")
```

---

## Complete Working Example: Mixed BC Scenario

```python
# Setup: Poisson with mixed Dirichlet-Neumann-Robin

# Mesh and space
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)

# Mark boundaries
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left: Dirichlet
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"},# Right: Neumann
    {"tag": 3, "condition": "x[1] < 1e-14"},      # Bottom: Robin
    {"tag": 4, "condition": "x[1] > 1.0 - 1e-14"} # Top: Neumann
], name="mixed_bnd", mesh_name="mesh")

# Set properties
set_material_properties(name="f", value="1.0")
set_material_properties(name="g_neumann", value="0.0")
set_material_properties(name="alpha_robin", value="1.0")
set_material_properties(name="g_robin", value="0.0")

# Define form
define_variational_form(
    bilinear="""
inner(grad(u), grad(v))*dx
+ alpha_robin*u*v*ds(3)
""",
    linear="""
f*v*dx
+ g_neumann*v*ds(2)
+ g_neumann*v*ds(4)
+ g_robin*v*ds(3)
""",
    trial_space="V",
    test_space="V"
)

# Apply Dirichlet (strong)
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")

# Solve
solve(solver_type="direct", solution_name="u_mixed")

# Validate
evaluate_solution(points=[[0.0, 0.5]], function_name="u_mixed")  # Should be ≈ 0
plot_solution(function_name="u_mixed", plot_type="contour")
```

---

## Tips and Best Practices

1. **Always mark before applying**: Call `mark_boundaries()` first
2. **Choose one enforcement method per boundary**: Don't mix strong + weak on same tag
3. **Penalty parameter matters**: For Nitsche, γ too small → poor enforcement; too large → ill-conditioning
4. **Test on simple case first**: Verify BC behavior on unit square before complex geometry
5. **Use evaluation points**: Check solution at known boundary points to validate enforcement
6. **Convergence study**: Refine mesh and verify convergence rate (should be O(h^p) for degree p)

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| **Boundary value not enforced** | Penalty too weak or missing term | Increase γ or verify all Nitsche terms present |
| **"Unknown boundary tag" error** | Tag used in ds(tag) not in mark_boundaries | Add missing tag to mark_boundaries |
| **Conflicting BCs error** | Strong (apply_boundary_condition) + weak on same tag | Use only one method per boundary |
| **Non-physical solution** | Incorrect BC sign or type | Verify strong form matches weak form derivation |
| **Slow convergence** | Ill-conditioning from penalty or geometry | Adjust α or refine mesh uniformly |

---

## See Also

- **advanced-boundary-conditions/SKILL.md** — Full guide to each BC type
- **bc-patterns.md** — 10 working examples you can copy
- **ufl-form-authoring/SKILL.md** — Syntax for boundary integrals (ds, dS)
- **pde-cookbook/SKILL.md** — BC examples in context of specific PDEs
