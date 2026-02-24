# Command: Setup DG Formulation

**Slug**: `setup-dg`

**Description**: Configure a Discontinuous Galerkin FEM problem with interior penalty and proper penalty parameter selection.

---

## Workflow

### Step 1: Choose Problem Type

What is the primary application?

**Options**:
- **Elliptic** (Poisson with interior penalty): Good for pure diffusion where you want DG's flexibility
- **Advection** (advection-diffusion or pure advection): Need upwinding stabilization
- **Mixed** (advection + diffusion + reaction): General conservation laws

**Your choice**: [Awaiting user input]

---

### Step 2: Create Mesh and DG Space

Define computational domain and create Discontinuous Galerkin function space.

```python
# Step 2a: Create mesh
create_unit_square(name="mesh", nx=20, ny=20)
# Or: create_mesh(name="mesh", shape="rectangle", nx=20, ny=20,
#                 dimensions={"width": 1.0, "height": 1.0})

# Step 2b: Create DG function space (key: family="DG")
create_function_space(name="V_dg", family="DG", degree=1, mesh_name="mesh")
# Note: degree 1 is typical; increase for higher accuracy (degree=2, etc.)

# Step 2c: Mark boundaries for weak BCs
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"},      # Left
    {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"} # Right
], name="dg_boundaries", mesh_name="mesh")
```

**Mesh size** (h): [Record approximate element size]
h ≈ 1.0 / nx = 1.0 / 20 = 0.05

---

### Step 3: Compute Penalty Parameter

The interior penalty method requires a penalty parameter α.

**Formula**:
```
α = C · degree · (degree + 1) / h_avg

where C ≈ 5-20 (use C=10 as default)
```

**Calculation** (for degree 1):
```
α = 10 · 1 · (1+1) / h_avg
  = 20 / h_avg
  = 20 / 0.05
  = 400
```

**For your setup**:
- Polynomial degree: [1, 2, 3, ...] → degree = 1
- Element size h: [h_min, h_typical, h_max] → h = 0.05
- Safety factor C: [5-20] → C = 10
- **Computed α**: [20/0.05] = **400**

```python
# Step 3: Set penalty parameter
degree = 1
h_avg = 1.0 / 20  # 0.05
alpha = 10.0 * degree * (degree + 1) / h_avg  # = 400.0
```

**Verify α is adequate**:
- Too small (α < 10): ✗ Form not stable
- Adequate (α ≈ 20-400): ✓
- Large (α > 10000): ✓ (stable but ill-conditioned)

---

### Step 4: Build DG Variational Form

Construct the Discontinuous Galerkin bilinear and linear forms.

#### For Elliptic (DG Poisson)

**Bilinear form** includes:
1. **Interior diffusion**: ∫ ∇u·∇v dx
2. **Interior penalty**: (α/h_avg) · ∫ jump(u)·jump(v) dS
3. **Interior asymmetry**: ∫ [-⟨∇u⟩·[v] - [u]·⟨∇v⟩] dS
4. **Boundary penalty**: (α/h_avg) · ∫ u·v ds (weak Dirichlet)
5. **Boundary asymmetry**: ∫ [-∇u·n·v - u·∇v·n] ds

```python
# Step 4a: Set parameters
set_material_properties(name="f", value="1.0")
set_material_properties(name="alpha", value="400.0")
set_material_properties(name="h_avg", value="0.05")  # Average mesh size = 1/nx

# Step 4b: Define DG bilinear form
bilinear = f"""
inner(grad(u), grad(v))*dx
+ (alpha/h_avg)*inner(jump(u), jump(v))*dS
- inner(avg(grad(u)), jump(v*n))*dS
- inner(jump(u*n), avg(grad(v)))*dS
+ (alpha/h_avg)*u*v*ds
- inner(grad(u)*n, v)*ds
- inner(u*n, grad(v))*ds
"""

# Step 4c: Define linear form
linear = "f*v*dx"

# Step 4d: Define form
define_variational_form(
    bilinear=bilinear,
    linear=linear,
    trial_space="V_dg",
    test_space="V_dg"
)
```

**Checklist**:
- [ ] Interior penalty term: (alpha/h_avg)*jump(u)*jump(v)*dS
- [ ] Interior asymmetry: -avg(grad(u))*jump(v*n)*dS, -jump(u*n)*avg(grad(v))*dS
- [ ] Boundary penalty: (alpha/h_avg)*u*v*ds
- [ ] Boundary asymmetry: -grad(u)*n*v*ds, -u*grad(v)*n*ds

#### For Advection (Upwind DG)

**Bilinear form** includes:
1. **Interior diffusion**: ∫ ε·∇u·∇v dx (if ε > 0)
2. **Interior advection**: ∫ b·∇u·v dx (volume)
3. **Upwind flux**: ∫ [upwind flux jump term] dS (interior facets)
4. **Boundary treatment**: inflow (Dirichlet), outflow (natural)

```python
# Step 4 (Advection variant):
set_material_properties(name="eps", value="0.01")
set_material_properties(name="b_x", value="1.0")
set_material_properties(name="b_y", value="0.0")

# Upwind flux (u_up = conditional-based choice)
bilinear = """
eps*inner(grad(u), grad(v))*dx
+ (b_x*grad(u)[0] + b_y*grad(u)[1])*v*dx
- u_upwind*(b*n)*v('+')dS
"""
# Note: u_upwind = conditional(dot(b('+'), n) > 0, u('+'), u('-'))
# This requires u_upwind defined elsewhere or simplified form

linear = "0.0*v*dx"  # Homogeneous advection
# Add inflow BC via set_material_properties + linear form term
```

---

### Step 5: Apply Boundary Conditions

In DG, apply Dirichlet BCs **weakly** (via penalty in form) or **not at all** for natural boundaries.

**For weak Dirichlet on boundary tag 1** (already in bilinear form above):
```python
# No additional apply_boundary_condition needed!
# BC is enforced via the penalty terms in the form

# However, set the Dirichlet value if needed:
set_material_properties(name="u_D", value="0.0")
```

**For Neumann** (natural BC, zero flux):
```python
# No terms added to form; natural by default
```

**Important**: Standard `apply_boundary_condition()` may conflict with weak DG enforcement.
Avoid using it; enforce BCs entirely in the weak form.

---

### Step 6: Solve and Validate

```python
# Step 6a: Solve
solve(solver_type="direct", solution_name="u_dg")

# Step 6b: Check solution bounds (sanity check)
evaluate_solution(points=[[0.5, 0.5]], function_name="u_dg")
# Expected: reasonable values (not NaN, not excessively large)

# Step 6c: Verify convergence (h-refinement)
# Refine mesh, re-solve, compute error
compute_error(exact="sin(pi*x[0])*sin(pi*x[1])", norm_type="L2", function_name="u_dg")
# Expected rate: O(h^(p+1)) for degree p
```

---

### Step 7: Parameter Sensitivity Study (Optional)

**Question**: Is α too small or too large?

**Experiment**: Solve with different α values and compare solutions.

```python
for alpha_test in [100, 200, 400, 800]:
    # Update penalty parameter
    set_material_properties(name="alpha", value=str(alpha_test))
    define_variational_form(bilinear=..., linear=..., ...)
    solve(solver_type="direct", solution_name=f"u_alpha_{alpha_test}")
    error = compute_error(exact="...", function_name=f"u_alpha_{alpha_test}")
    print(f"alpha={alpha_test}, error={error:.6f}")
```

**Observation**:
- If error decreases significantly with larger α: Original α was too small → increase
- If error plateaus: α is adequate
- If error increases or conditioning degrades: α too large → decrease

---

## Complete Working Example: DG Poisson

```python
# Full minimal example
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V_dg", family="DG", degree=1, mesh_name="mesh")
mark_boundaries(markers=[
    {"tag": 1, "condition": "True"}
], name="bnd", mesh_name="mesh")

set_material_properties(name="f", value="10.0")
set_material_properties(name="alpha", value="400.0")

define_variational_form(
    bilinear="""
inner(grad(u), grad(v))*dx
+ (alpha/0.05)*inner(jump(u), jump(v))*dS
- inner(avg(grad(u)), jump(v*n))*dS
- inner(jump(u*n), avg(grad(v)))*dS
+ (alpha/0.05)*u*v*ds
- inner(grad(u)*n, v)*ds
- inner(u*n, grad(v))*ds
""",
    linear="f*v*dx",
    trial_space="V_dg",
    test_space="V_dg"
)

solve(solver_type="direct", solution_name="u_dg")
plot_solution(function_name="u_dg", plot_type="contour", return_base64=True)
```

---

## Common Issues and Fixes

| Issue | Symptom | Solution |
|---|---|---|
| **α too small** | Solver converges but solution oscillates or violates BCs | Increase α (try 2× current value) |
| **α too large** | Solver slow, conditioning bad | Reduce α or use iterative solver with preconditioner |
| **h_avg not computed** | Form uses h_avg but undefined | Replace with computed value: h_avg = 1.0/nx or compute from mesh |
| **jump/avg on dx** | Error about undefined operators | Use dS (interior facets) only; not dx or ds |
| **Mismatch in restrictions** | Form assembly error | Pair (+) and (-) correctly; use avg() to bridge them |
| **Solution not matching BC** | Weak Dirichlet u_D not enforced strongly | Verify α is large enough; add boundary penalty term |

---

## Tips

1. **Start with P1 (degree=1)**: Standard in DG; easier to debug before trying degree 2+
2. **α ≈ 10-20 is usually safe**: Unless elements vary drastically in size
3. **Use direct solver for small problems** (<100k DOFs): Avoid iterative solver configuration
4. **Test on simple domains first**: Unit square with known solution (e.g., polynomial)
5. **Check jump/average directions**: (+) and (-) sides matter for consistency

---

## See Also

- **dg-formulations/SKILL.md** — Full guide to DG theory and practice
- **dg-methods-theory.md** — Mathematical foundations, penalty parameter theory
- **operator-reference.md** — `jump()`, `avg()`, and restriction syntax
- **pde-cookbook/SKILL.md** — Recipe #13 covers DG Poisson in detail
