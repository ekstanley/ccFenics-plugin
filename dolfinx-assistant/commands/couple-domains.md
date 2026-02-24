# Multi-Domain Coupling Setup

**Purpose**: Interactively set up and solve coupled multi-physics problems (thermal-mechanical, fluid-structure, custom).

---

## Step 1: Select Coupling Type

Which multi-physics scenario?

- **Thermal-Mechanical**: Heat equation (domain 1) → Temperature interpolated → Elasticity with thermal strain (domain 2). *One-way coupling.*
- **Fluid-Structure Interaction (FSI)**: Stokes/NS (fluid domain) ↔ Elasticity (structure domain) via interface tractions. *Bidirectional iterative coupling.*
- **Thermoconvection**: Heat equation (temperature) ↔ Stokes flow (buoyancy coupling). *Two-way iterative.*
- **Custom**: You define the domains, PDEs, and coupling strategy.

**Select**: `thermal-mechanical | fsi | thermoconvection | custom`

---

## Step 2: Check Mesh Structure

Your mesh must define **subdomains** (regions for each physics).

### Option A: Import from Gmsh with cell_tags

If you have a `.msh` file with cell regions:

```bash
create_custom_mesh name=coupled_mesh filename=my_domains.msh
```

Output will show:
```
{
  "name": "coupled_mesh",
  "cell_tags": "cell_tags",      # ← Auto-created entity map
  "facet_tags": "facet_tags",    # ← Boundary facets
  ...
}
```

The `cell_tags` object maps each cell to a region ID (1, 2, 3, ...).

### Option B: Mark Regions Programmatically

If you have a single mesh, mark subdomains:

```bash
create_mesh name=base_mesh shape=unit_square nx=20 ny=20
mark_boundaries name=domain_tags markers='[{"tag": 1, "condition": "x[0] < 0.5"}, {"tag": 2, "condition": "x[0] >= 0.5"}]'
```

Creates `domain_tags` (cell-based).

**Verify**:
```bash
manage_mesh_tags action=query name=domain_tags
```

Output:
```
{
  "name": "domain_tags",
  "dimension": 2,           # ← Cell-based (not facets)
  "unique_tags": [1, 2],
  "tag_counts": {1: 100, 2: 100}
}
```

---

## Step 3: Extract Submeshes for Each Domain

For each physics domain, extract its submesh:

```bash
# Thermal domain (region 1)
create_submesh name=thermal_mesh tags_name=domain_tags tag_values=[1]

# Mechanical domain (region 2)
create_submesh name=mechanical_mesh tags_name=domain_tags tag_values=[2]
```

**Output**: Each submesh has auto-created `entity_map` tracking parent-child relationships.

**Verify**:
```bash
get_mesh_info name=thermal_mesh
get_mesh_info name=mechanical_mesh
```

Should show:
```
{"name": "thermal_mesh", "num_cells": 100, "num_vertices": ...}
{"name": "mechanical_mesh", "num_cells": 100, "num_vertices": ...}
```

---

## Step 4: Create Function Spaces on Each Submesh

**Thermal domain** (scalar temperature):
```bash
create_function_space name=T_space family=Lagrange degree=1 mesh_name=thermal_mesh
```

**Mechanical domain** (vector displacement):
```bash
create_function_space name=u_space family=Lagrange degree=1 shape=[2] mesh_name=mechanical_mesh
```

(For FSI, create velocity space on fluid domain and displacement space on structure domain, etc.)

---

## Step 5: Solve First Domain Problem

### Example: Thermal Problem (Domain 1)

**Set up**:
```bash
set_material_properties name=kappa value=1.0  # Thermal conductivity
set_material_properties name=f_thermal value=1.0  # Heat source
```

**Boundary conditions**:
```bash
apply_boundary_condition value=300 boundary='x[1]==0' function_space=T_space name=bc_cold
apply_boundary_condition value=400 boundary='x[1]==1' function_space=T_space name=bc_hot
```

**Variational form**:
```bash
define_variational_form bilinear='kappa*inner(grad(u), grad(v))*dx' linear='f_thermal*v*dx'
```

**Solve**:
```bash
solve solution_name=T_h
```

**Verify**: Plot and evaluate:
```bash
plot_solution function_name=T_h plot_type=contour colormap=hot
evaluate_solution function_name=T_h points=[[0.5, 0.5]]
```

---

## Step 6: Transfer Solution to Second Domain

**Problem**: T_h lives on `thermal_mesh`, but we need it on `mechanical_mesh`.

**Solution**: Create a function on mechanical space and interpolate T_h into it:

```bash
# Create zero-initialized function on mechanical space
create_function name=T_mechanical function_space=u_space
```

**Interpolate from thermal to mechanical**:
```bash
interpolate target=T_mechanical source_function=T_h source_mesh=thermal_mesh
```

**Result**: `T_mechanical` now contains temperature values transferred to the mechanical mesh via point evaluation.

**Check transfer**:
```bash
evaluate_solution function_name=T_mechanical points=[[0.5, 0.5]]
```

---

## Step 7: Solve Second Domain Problem with Coupling

### Example: Mechanical Problem with Thermal Strain

**Thermal strain**: ε_th = α(T - T_ref) * I

**Material properties**:
```bash
set_material_properties name=E value=1e11      # Young's modulus
set_material_properties name=nu value=0.3      # Poisson ratio
set_material_properties name=alpha value=1e-5  # Thermal expansion
set_material_properties name=T_ref value=300   # Reference temperature
```

**Compute Lamé parameters**:
```bash
set_material_properties name=mu value='E/(2*(1+nu))'
set_material_properties name=lambda value='E*nu/((1+nu)*(1-2*nu))'
```

**Boundary conditions** (mechanical):
```bash
apply_boundary_condition value=[0, 0] boundary='x[0]==0' function_space=u_space
```

**Variational form with thermal coupling**:
The weak form includes both elastic stiffness and thermal strain:
```
∫ σ_elastic(u):ε(v) dx = ∫ σ_thermal(T_mech):ε(v) dx + ∫ f_mech*v dx
```

where σ_thermal = λ*α*(T - T_ref)*tr(I)*I + 2μ*α*(T - T_ref)*I

Define in MCP:
```bash
define_variational_form \
  bilinear='lambda*inner(div(u), div(v))*dx + 2*mu*inner(sym(grad(u)), sym(grad(v)))*dx' \
  linear='alpha*(T_mechanical - T_ref)*(lambda*trace(Identity(2)) + 2*mu)*inner(Identity(2), sym(grad(v)))*dx'
```

(Alternatively, use custom assembly with `run_custom_code` if UFL expression is complex.)

**Solve mechanical**:
```bash
solve solution_name=u_mechanical
```

**Verify**: Plot and evaluate:
```bash
plot_solution function_name=u_mechanical component=0 colormap=RdBu_r
evaluate_solution function_name=u_mechanical points=[[0.5, 0.5]]
```

---

## Step 8: (Optional) Iterative Coupling for Bidirectional Problems

For **FSI** or **thermoconvection** where both domains feed back into each other:

### Gauss-Seidel Loop Pattern

```bash
# Initialize: u_struct_old = 0
create_function name=u_struct_old function_space=u_space expression='0'

# Iteration loop
for i in range(1, 20):
    # ===== Fluid solve (using u_struct from previous iteration) =====
    # Apply Dirichlet BC on interface: u_fluid = u_struct_old
    apply_boundary_condition value=u_struct_old boundary='interface_tag' function_space=V_fluid

    # Solve Stokes
    define_variational_form bilinear='...' linear='...'
    solve solution_name=u_fluid_i

    # Extract traction at interface: tau = sigma_fluid * n
    # (Use run_custom_code for this; MCP doesn't have direct traction extraction)
    run_custom_code code='
      import dolfinx.fem as fem
      import ufl

      # Get fluid solution
      u_fluid = session.functions["u_fluid_i"]

      # Compute traction (simplified; full version needs facet integration)
      # tau = sigma_fluid @ n on interface
      session.functions["tau_interface"] = tau
    '

    # ===== Structure solve (using tau from fluid) =====
    # Apply Neumann BC on interface: sigma_struct*n = tau_interface
    apply_boundary_condition value=tau_interface boundary='interface_tag' function_space=V_struct

    # Solve elasticity
    define_variational_form bilinear='...' linear='...'
    solve solution_name=u_struct_i

    # ===== Check convergence =====
    run_custom_code code='
      import numpy as np

      u_new = session.functions["u_struct_i"]
      u_old = session.functions["u_struct_old"]

      residual = np.linalg.norm((u_new.x.array - u_old.x.array))
      residual_rel = residual / np.linalg.norm(u_new.x.array)

      print(f"Iteration {i}: residual = {residual:.4e}, relative = {residual_rel:.4e}")

      if residual_rel < 1e-4:
        print("Converged!")
        break

      # Update for next iteration
      u_old.x.array[:] = u_new.x.array
    '

    # u_struct_old = u_struct_i for next iteration
```

---

## Step 9: Multi-Domain Visualization and Export

Export both domain solutions together:

```bash
export_solution filename=thermal_mech_coupled.xdmf format=xdmf functions=['T_h', 'u_mechanical']
```

This creates a single VTK file with both solutions (though on different meshes; visualization tool will handle interpolation).

---

## Step 10: Report Summary

Gather key results:

```bash
run_custom_code code='
import numpy as np

# Get solutions
T_solution = session.functions.get("T_h")
u_solution = session.functions.get("u_mechanical")

# Compute norms
T_L2 = T_solution.vector.norm()
u_L2 = u_solution.vector.norm()

# Find extrema
T_vals = T_solution.x.array
u_vals = u_solution.x.array

print(f"=== Thermal-Mechanical Coupling Summary ===")
print(f"Temperature:")
print(f"  L2 norm: {T_L2:.4e}")
print(f"  Min: {T_vals.min():.4f}, Max: {T_vals.max():.4f}")
print(f"Displacement (magnitude):")
print(f"  L2 norm: {u_L2:.4e}")
print(f"  Max: {u_vals.max():.4e}")
print(f"Coupling: {u_L2 / (T_L2 * 1e-5):.2f} (relative scale)")
'
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Interpolation produces zeros | Source mesh ≠ target mesh coordinates | Verify both meshes created correctly; check entity maps |
| Solver diverges after coupling | BC incompatible or form wrong | Check that BC values match across domains; print intermediate T values |
| Slow convergence (Gauss-Seidel) | Strong coupling, no relaxation | Add Aitken acceleration or use underrelaxation ω=0.5 |
| NaN in second domain solution | Source field out of bounds | Evaluate first domain solution at points; check for extreme values |
| Different mesh resolutions mismatch | Submeshes from different parents | Use single parent mesh with `mark_boundaries`; extract submeshes from same source |

---

## Advanced: Custom Coupling Logic with run_custom_code

For complex iteration, relaxation, or energy-based coupling:

```bash
run_custom_code code='
import numpy as np
from dolfinx import fem

# Aitken acceleration
omega = 0.1
residuals = []

for k in range(50):
    # Solve A
    # ... (define form, solve, extract u_A)

    # Solve B with u_A
    # ... (define form, solve, extract u_B)

    # Compute residual
    r = np.linalg.norm((u_B - u_B_old).vector)
    residuals.append(r)

    if r < 1e-5:
        print("Converged!")
        break

    # Aitken: adjust omega
    if len(residuals) > 1:
        denom = (residuals[-1] - residuals[-2])**2
        if abs(denom) > 1e-14:
            omega = omega * (omega - 1) / (1 - 2*omega + denom / r)
            omega = max(0.01, min(omega, 0.9))

    print(f"Iter {k}: r = {r:.4e}, omega = {omega:.3f}")

    # Update with relaxation
    u_B.x.array[:] = u_B_old.x.array + omega * (u_B.x.array - u_B_old.x.array)
    u_B_old.x.array[:] = u_B.x.array
'
```

---

## See Also

- **Coupling strategies reference**: `skills/multi-physics-coupling/references/coupling-strategies.md`
- **Full multi-physics skill**: `skills/multi-physics-coupling/SKILL.md`
- **Single-domain recipes**: `commands/recipe.md`
