# Boundary Condition Patterns Library

**Version**: 0.1.0

---

## Pattern 1: Homogeneous Dirichlet (All Boundaries)

**Use Case**: Classic Poisson with u = 0 on all boundaries.

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

# Single boundary mark for entire boundary
mark_boundaries(markers=[
    {"tag": 1, "condition": "True"}  # All boundaries
], name="all_bnd")

set_material_properties(name="f", value="1.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

# Apply homogeneous Dirichlet
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: None. Dirichlet enforced strongly.

**Validation**: Solution should be zero on all boundaries.

---

## Pattern 2: Inhomogeneous Dirichlet (Constant)

**Use Case**: u = 1 on one boundary, u = 0 on another.

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left: u=0
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"} # Right: u=1
], name="bnd")

set_material_properties(name="f", value="1.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

# Apply different values on different boundaries
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")
apply_boundary_condition(value=1.0, boundary_tag=2, function_space="V")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: None. Both enforced strongly.

**Validation**: Check solution at left (≈0) and right (≈1) boundaries.

---

## Pattern 3: Inhomogeneous Dirichlet (Spatially-Varying)

**Use Case**: u = sin(πy) on left boundary.

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left: u = sin(π*x[1])
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"} # Right: u = 0
], name="bnd")

set_material_properties(name="f", value="0.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

# Expression-based Dirichlet
apply_boundary_condition(value="sin(pi*x[1])", boundary_tag=1, function_space="V")
apply_boundary_condition(value=0.0, boundary_tag=2, function_space="V")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: None. Strong enforcement handles expressions automatically.

**Validation**: Check boundary 1 at y=0.5: u_h(0, 0.5) ≈ sin(π/2) = 1.

---

## Pattern 4: Natural Neumann (Zero Flux)

**Use Case**: du/dn = 0 (insulated boundary).

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"}  # Left: Neumann (zero flux, natural)
], name="bnd")

set_material_properties(name="f", value="1.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

# NO boundary condition! Neumann is natural (implicit in weak form)

# Apply Dirichlet on other boundary if needed
apply_boundary_condition(value=0.0, boundary="x[0] > 0.9", function_space="V")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: None. Natural BC is automatic (no ds(tag) term needed).

**Validation**: Solution should be smooth with zero normal derivative on boundary 1.

---

## Pattern 5: Non-Zero Neumann

**Use Case**: du/dn = g (prescribed flux).

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] > 0.9"},  # Right: Neumann, du/dn = 1
    {"tag": 2, "condition": "x[0] < 1e-14"} # Left: Dirichlet, u = 0
], name="bnd")

set_material_properties(name="f", value="0.0")
set_material_properties(name="g_neumann", value="1.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx + g_neumann*v*ds(1)",  # Add Neumann to linear form
    trial_space="V",
    test_space="V"
)

apply_boundary_condition(value=0.0, boundary_tag=2, function_space="V")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: Linear form includes `g_neumann*v*ds(1)`.

**Validation**: Integrate flux on boundary: ∫_{∂Ω} g·n ≈ ∫_Ω f (divergence theorem).

---

## Pattern 6: Robin Boundary Condition

**Use Case**: du/dn + α·u = g (convective cooling).

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] > 0.9"},  # Right: Robin
    {"tag": 2, "condition": "x[0] < 1e-14"} # Left: Dirichlet, u = 1
], name="bnd")

set_material_properties(name="f", value="0.0")
set_material_properties(name="alpha_robin", value="1.0")
set_material_properties(name="g_robin", value="0.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx + alpha_robin*u*v*ds(1)",  # Robin in bilinear
    linear="f*v*dx + g_robin*v*ds(1)",                               # Robin RHS in linear
    trial_space="V",
    test_space="V"
)

apply_boundary_condition(value=1.0, boundary_tag=2, function_space="V")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**:
- Bilinear: Add `alpha_robin*u*v*ds(1)`
- Linear: Add `g_robin*v*ds(1)`

**Validation**: At boundary, du/dn + α*u should equal g.

---

## Pattern 7: Nitsche Weak Dirichlet (Penalty)

**Use Case**: Enforce u = u_D weakly (for DG or unfitted methods).

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"}  # Left: Weak Dirichlet, u = 1
], name="bnd")

set_material_properties(name="f", value="1.0")
set_material_properties(name="gamma", value=100.0)  # Penalty parameter
set_material_properties(name="u_D", value="1.0")

h_val = 0.1  # Average element size

bilinear = f"""
inner(grad(u), grad(v))*dx
- inner(grad(u)*n, v)*ds(1)
- inner(u*n, grad(v))*ds(1)
+ {100.0/h_val}*u*v*ds(1)
"""

linear = f"""
f*v*dx
+ {100.0/h_val}*u_D*v*ds(1)
+ u_D*grad(v)*n*ds(1)
"""

define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V",
    test_space="V"
)

# NO apply_boundary_condition() — enforcement is in form!

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**:
- Bilinear: Add penalty + asymmetry terms
- Linear: Add penalty RHS + asymmetry RHS

**Validation**: Error on boundary should be O(1/γ). Increase γ to improve enforcement.

---

## Pattern 8: Component-Wise Dirichlet (Vector Space)

**Use Case**: Fix x-component of velocity on boundary, allow y-component free.

**Setup**:
```python
create_unit_square(name="mesh", nx=10, ny=10)
create_function_space(name="V", family="Lagrange", degree=1, shape=[2])  # Vector space

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"}   # Left: u_x = 0
], name="bnd")

set_material_properties(name="f", value="[1.0, 0.0]")  # Vector source

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="inner(f, v)*dx",
    trial_space="V",
    test_space="V"
)

# Fix only x-component (sub_space=0)
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V", sub_space=0)

# y-component (sub_space=1) is free

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: None. Sub-space parameter handled by apply_boundary_condition().

**Validation**: Check that u_x(0, y) = 0, but u_y(0, y) ≠ 0 in general.

---

## Pattern 9: Multiple Dirichlet Values (Different Boundaries)

**Use Case**: Complex geometry with different u values on different edges.

**Setup**:
```python
create_unit_square(name="mesh", nx=15, ny=15)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left: u = 0
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"},# Right: u = 1
    {"tag": 3, "condition": "x[1] < 1e-14"},      # Bottom: u = 0
    {"tag": 4, "condition": "x[1] > 1.0 - 1e-14"} # Top: u = 1
], name="bnd")

set_material_properties(name="f", value="0.0")

define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V",
    test_space="V"
)

# Apply different BCs per boundary
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V", name="bc_left")
apply_boundary_condition(value=1.0, boundary_tag=2, function_space="V", name="bc_right")
apply_boundary_condition(value=0.0, boundary_tag=3, function_space="V", name="bc_bottom")
apply_boundary_condition(value=1.0, boundary_tag=4, function_space="V", name="bc_top")

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**: None. Multiple BCs compose naturally.

**Validation**: Solution should be 0 on left/bottom, 1 on right/top.

---

## Pattern 10: Mixed Dirichlet-Neumann-Robin

**Use Case**: All three BC types on different parts of boundary.

**Setup**:
```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)

mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},       # Left: Dirichlet
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"}, # Right: Neumann
    {"tag": 3, "condition": "x[1] < 1e-14"}        # Bottom: Robin
], name="bnd")

set_material_properties(name="f", value="1.0")
set_material_properties(name="g_neumann", value="0.5")
set_material_properties(name="alpha_robin", value="2.0")
set_material_properties(name="g_robin", value="1.0")

define_variational_form(
    bilinear="""
inner(grad(u), grad(v))*dx
+ alpha_robin*u*v*ds(3)
""",
    linear="""
f*v*dx
+ g_neumann*v*ds(2)
+ g_robin*v*ds(3)
""",
    trial_space="V",
    test_space="V"
)

# Apply Dirichlet (strong)
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V", name="bc_dirichlet")

# Neumann and Robin are in weak form; no additional BC calls

solve(solver_type="direct", solution_name="u_h")
```

**Form modifications**:
- Bilinear: Add Robin mass term `alpha_robin*u*v*ds(3)`
- Linear: Add Neumann `g_neumann*v*ds(2)` and Robin RHS `g_robin*v*ds(3)`

**Validation**:
- Check u(0, y) ≈ 0 (Dirichlet)
- Integrate flux on right boundary (Neumann)
- Check du/dn + α*u = g on bottom (Robin)

---

## Quick Reference Table

| Pattern | Primary BC | Form Terms | Strong BC Call? | Boundary Tags Needed |
|---|---|---|---|---|
| 1: Homo Dirichlet | u = 0 | None | Yes | All boundaries |
| 2: Inhomo Dirichlet | u = const | None | Yes | Per boundary value |
| 3: Varying Dirichlet | u = u(x) | None | Yes | Expression-based |
| 4: Neumann (zero) | du/dn = 0 | None | No | Not needed |
| 5: Neumann (nonzero) | du/dn = g | Linear: `g*v*ds(tag)` | No | Neumann boundary |
| 6: Robin | du/dn + α*u = g | Both: `α*u*v*ds(tag)` + `g*v*ds(tag)` | No | Robin boundary |
| 7: Nitsche Dirichlet | u = u_D (weak) | Both + asymmetry | No | Nitsche boundary |
| 8: Component BC | u_i = const | None, use sub_space | Yes | Per component |
| 9: Multiple Dirichlet | u = u_i per region | None | Yes per tag | Multiple |
| 10: Mixed | Dirichlet + Neumann + Robin | Combine patterns | Partial | Multiple |

---

## Assembly Checklist per Pattern

- [ ] Pattern 1-3, 8, 9: Mark all boundaries; call apply_boundary_condition() per BC
- [ ] Pattern 4: Mark Neumann boundary (optional, auto-natural); no BC call
- [ ] Pattern 5: Mark Neumann boundary; add `g*v*ds(tag)` to linear form
- [ ] Pattern 6: Mark Robin boundary; add `α*u*v*ds(tag)` to bilinear, `g*v*ds(tag)` to linear
- [ ] Pattern 7: Mark Nitsche boundary; add penalty + asymmetry terms; NO apply_boundary_condition()
- [ ] Pattern 10: Mark all boundaries; call apply_boundary_condition() for strong BCs only; add weak terms to form for Neumann/Robin

---

## Validation Workflow for All Patterns

1. **Define exact solution** (if available): `u_exact(x, y) = ...`
2. **Solve problem** via solve()
3. **Evaluate error**:
   ```python
   compute_error(exact="sin(pi*x[0])*sin(pi*x[1])", norm_type="L2")
   ```
4. **Check boundary values** via evaluate_solution():
   ```python
   boundary_points = [[0.0, 0.5], [0.0, 0.75]]  # Points on left boundary
   evaluate_solution(points=boundary_points, function_name="u_h")
   ```
5. **Verify convergence**: Refine mesh, resolve, check error decreases at rate O(h^(p+1))

---

## See Also

- **advanced-boundary-conditions/SKILL.md** — Full guide to each BC type
- **/ufl-form-authoring** — Syntax for building weak forms
