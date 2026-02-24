# Command: Compose UFL Form Interactively

**Slug**: `compose-form`

**Description**: Build a complete variational form step-by-step with guidance and validation.

---

## Workflow

### Step 1: Define Problem Type

What is the nature of your problem?

**Options**:
- **Scalar field** (temperature, pressure, concentration)
- **Vector field** (velocity, displacement, magnetic field)
- **Mixed** (velocity + pressure in Stokes, displacement + strain)

**Your choice**: [Awaiting user input]

---

### Step 2: Identify Primary Operators

Which mathematical operators appear in your weak form?

**Common combinations**:
- ✓ **Diffusion only**: ∫ κ·∇u·∇v dx
- ✓ **Diffusion + Reaction**: ∫ κ·∇u·∇v + c·u·v dx
- ✓ **Advection + Diffusion**: ∫ ε·∇u·∇v + b·∇u·v dx
- ✓ **Strain energy** (elasticity): ∫ σ(u):ε(v) dx
- ✓ **Pressure coupling** (Stokes): ∫ μ·∇u:∇v - p·div(v) dx

**Mark all that apply**: [Awaiting user choices]

---

### Step 3: Build Bilinear Form

Now we'll construct the **bilinear form** a(u, v) piece by piece.

**Template structure**:
```python
bilinear = "term1 + term2 + ... + termN"
```

#### Piece A: Main Operator
```
Enter the primary differential operator:
- For Poisson-like: "inner(grad(u), grad(v))*dx"
- For elasticity: "2*mu*inner(sym(grad(u)), sym(grad(v)))*dx"
- For Stokes: "mu*inner(grad(u), grad(v))*dx"
```

**Your term**: [Awaiting user input]

#### Piece B: Reaction/Mass (if needed)
```
Enter reaction or mass term (use 0 if none):
- Example: "c*u*v*dx"
- Example: "u*v*dx" (for time-dependent, coefficient is 1/dt added elsewhere)
```

**Your term**: [Awaiting user input]

#### Piece C: Incompressibility Constraint (if mixed problem)
```
For mixed spaces (velocity + pressure), add pressure terms:
- Example: "- p*div(v)*dx - div(u)*q*dx"
```

**Your term**: [Awaiting user input or skip]

#### Piece D: Boundary/Penalty Terms (if Nitsche or Robin)
```
Add boundary contributions:
- Nitsche weak Dirichlet: "(gamma/h)*u*v*ds(1) - inner(grad(u)*n, v)*ds(1) - inner(u*n, grad(v))*ds(1)"
- Robin: "alpha*u*v*ds(boundary_tag)"
```

**Your term**: [Awaiting user input or skip]

**Assembled bilinear form**:
```
a(u,v) = [Piece A] + [Piece B] + [Piece C] + [Piece D]
```

**Verification**:
- [ ] Both u and v appear exactly once in each term
- [ ] All indices are contracted (result is scalar)
- [ ] All terms end with dx, ds, or dS
- [ ] Rank mismatch? Use `inner()` to contract

---

### Step 4: Build Linear Form

Construct the **linear form** L(v) (depends on v only, not on u).

#### Piece A: Source/Load
```
Enter source term:
- For Poisson: "f*v*dx"
- For elasticity: "inner(f, v)*dx" where f is body force
- Example: "f*v*dx + g*v*ds(neumann_tag)"
```

**Your term**: [Awaiting user input]

#### Piece B: Boundary Conditions (Neumann/Robin)
```
Add natural BC contributions:
- Neumann flux: "g*v*ds(neumann_tag)"
- Robin RHS: "h*v*ds(robin_tag)"
- Nitsche Dirichlet RHS: "(gamma/h)*u_D*v*ds(1) + u_D*grad(v)*n*ds(1)"
```

**Your term**: [Awaiting user input or skip]

#### Piece C: Time-Dependent Term (if applicable)
```
For unsteady problems (u_n from previous timestep):
- Example: "u_n*v*dx / dt"
```

**Your term**: [Awaiting user input or skip]

**Assembled linear form**:
```
L(v) = [Piece A] + [Piece B] + [Piece C]
```

**Verification**:
- [ ] Only v appears (no u, only u_n from previous step)
- [ ] All terms end with dx or ds
- [ ] All indices contracted to scalar

---

### Step 5: Validate and Generate Code

**Your completed forms**:
```
Bilinear:  a(u,v) = [assembled]
Linear:    L(v) = [assembled]
```

**Validation checks**:
1. **Rank consistency**: ✓
2. **Test vs Trial**: ✓ (u is trial, v is test)
3. **Measure coverage**: ✓ (all integrals have dx/ds/dS)
4. **Material properties**: ✓ (all coefficients exist or will be defined)

**Generated MCP code**:

```python
# Step 1: Create mesh
create_unit_square(name="mesh", nx=20, ny=20)

# Step 2: Create function space
create_function_space(name="V", family="Lagrange", degree=1)

# Step 3: Mark boundaries (if needed for ds(tag) or BCs)
mark_boundaries(markers=[
    {"tag": 1, "condition": "x[0] < 1e-14"}    # Adjust geometry
], name="boundaries", mesh_name="mesh")

# Step 4: Define material properties
set_material_properties(name="f", value="1.0")
set_material_properties(name="kappa", value="1.0")
# Add others as needed

# Step 5: Define variational form
define_variational_form(
    bilinear="[YOUR BILINEAR FORM]",
    linear="[YOUR LINEAR FORM]",
    trial_space="V",
    test_space="V"
)

# Step 6: Apply boundary conditions (Dirichlet only)
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")

# Step 7: Solve
solve(solver_type="direct", solution_name="u_h")
```

**Next steps**:
- [ ] Verify material property names in set_material_properties match those in forms
- [ ] Check boundary tags in mark_boundaries match those in weak form (ds(tag))
- [ ] Run solver and validate solution (check boundary values, compute error, etc.)

---

## Troubleshooting

### "Rank mismatch error"
**Problem**: Form has mixed ranks (e.g., scalar + vector)

**Solution**: Use `inner()` to properly contract:
```python
# WRONG: u + grad(v)  (rank 0 + rank 1)
# CORRECT: inner(u, v) or dot(u, v)  if same rank
```

### "Unknown material property"
**Problem**: Form references "kappa" but never defined

**Solution**: Call set_material_properties("kappa", value=...) before define_variational_form()

### "Boundary tag not found"
**Problem**: Form uses ds(5) but mark_boundaries only created tags 1-3

**Solution**: Add missing tag to mark_boundaries with appropriate geometric condition

### Form looks correct but solution is wrong
**Problem**: Weak form is syntactically valid but mathematically incorrect

**Diagnostic**:
1. Verify against /pde-cookbook reference for your PDE type
2. Check sign of all terms (esp. Neumann/Robin RHS)
3. Validate on simple test case (e.g., exact polynomial solution)

---

## Tips

1. **Start simple**: Build bilinear form first (main operator only), validate, then add terms
2. **Use reference cards**: Consult /ufl-form-authoring/operator-reference.md for correct operators
3. **Check the cookbook**: /pde-cookbook/SKILL.md likely has your PDE type already
4. **Test on unit square**: Start on [0,1]² with homogeneous BCs to eliminate variables
5. **Print assembled form**: Before solving, display the complete form strings to catch typos

---

## See Also

- **ufl-form-authoring/SKILL.md** — Complete guide to UFL operators and syntax
- **ufl-form-authoring/references/operator-reference.md** — Quick operator lookup
- **pde-cookbook/SKILL.md** — 15 pre-built PDE recipes to copy from
- **weak-form-derivations.md** — Full mathematical derivations if stuck
