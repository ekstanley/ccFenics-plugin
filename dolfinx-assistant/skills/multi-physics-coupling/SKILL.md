# Multi-Physics and Multi-Domain Coupling in DOLFINx

**When to use**: Solve coupled problems across multiple domains or physics regimes (thermal-mechanical, fluid-structure, electrochemistry, etc.).

## Core Concepts

### Multi-Domain Problems
- **Definition**: Different PDEs solved on different spatial regions with interface conditions
- **Example**: Thermal problem on domain Ω₁, mechanical on domain Ω₂, coupled at interface Γ
- **Workflow**: Create mesh with subdomains → extract submeshes → solve each domain → couple via interpolation

### Interface Conditions
| Type | Condition | Implementation |
|------|-----------|-----------------|
| Dirichlet trace | u continuous across interface | `interpolate(u_from_domain1, space_domain2)` |
| Neumann trace | flux continuous (natural in weak form) | Both domains share boundary integral |
| Lagrange multiplier | Enforce constraint weakly | Advanced (requires run_custom_code) |
| Robin (mixed) | Combined: a*u + b*∇u·n = c | Applied as BC on both sides |

---

## Workflow 1: Thermal-Mechanical Coupling

**Problem**: Solve heat equation → interpolate temperature → solve elasticity with thermal strain.

### Step 1: Create mesh with tagged regions (thermal domain = 1, mechanical = 2)
```bash
# Via Gmsh import if available, or programmatically:
create_custom_mesh name=coupled_mesh filename=thermal_mech.msh
```

Or mark regions on existing mesh:
```bash
mark_boundaries name=domain_tags markers='[{"tag": 1, "condition": "x[0] < 0.5"}, {"tag": 2, "condition": "x[0] >= 0.5"}]'
```

### Step 2: Extract submeshes
```bash
create_submesh name=thermal_mesh tags_name=domain_tags tag_values=[1]
create_submesh name=mechanical_mesh tags_name=domain_tags tag_values=[2]
```

### Step 3: Create function spaces on each submesh
```bash
create_function_space name=T_space family=Lagrange degree=1 mesh_name=thermal_mesh
create_function_space name=u_space family=Lagrange degree=1 shape=[2] mesh_name=mechanical_mesh
```

### Step 4: Solve thermal problem
Set material properties (thermal conductivity) and BCs:
```bash
set_material_properties name=k value=1.0
set_material_properties name=f_thermal value=1.0
apply_boundary_condition value=300 boundary='x[1]==0' function_space=T_space name=bc_cold
apply_boundary_condition value=400 boundary='x[1]==1' function_space=T_space name=bc_hot
```

Define variational form:
```bash
define_variational_form bilinear='k * inner(grad(u), grad(v)) * dx' linear='f_thermal * v * dx'
```

Solve:
```bash
solve solution_name=T_h
```

### Step 5: Interpolate temperature to mechanical mesh
```bash
interpolate target=u_mech source_function=T_h source_mesh=thermal_mesh
```

**Result**: `u_mech` is zero-initialized function on mechanical space; can be pre-populated if T_h matches shape. More useful: create thermal strain field:

```bash
create_function name=T_transfer function_space=u_space
# Then use run_custom_code to copy/interpolate T_h→T_transfer
interpolate target=T_transfer source_function=T_h source_mesh=thermal_mesh
```

### Step 6: Solve elasticity with thermal loading
Create thermal strain field as material property:
```bash
set_material_properties name=alpha value=1e-5
set_material_properties name=T_ref value=300.0
# Thermal strain: epsilon_th = alpha * (T - T_ref) * I
# Define in bilinear form manually
define_variational_form bilinear='inner(sigma(u, lambda, mu), epsilon(v)) * dx' linear='- inner(sigma_thermal(T_transfer, alpha, T_ref, lambda, mu), epsilon(v)) * dx'
# Where sigma_thermal = lambda * tr(epsilon_thermal) * I + 2*mu*epsilon_thermal
```

Apply mechanical BCs:
```bash
apply_boundary_condition value=0 boundary='x[0]==0' sub_space=0 function_space=u_space
apply_boundary_condition value=0 boundary='x[0]==0' sub_space=1 function_space=u_space
```

Solve:
```bash
solve solution_name=u_mechanical
```

### Step 7: Evaluate results
```bash
evaluate_solution function_name=u_mechanical points=[[0.5, 0.5], [0.75, 0.75]]
plot_solution function_name=u_mechanical plot_type=contour colormap=viridis
```

---

## Workflow 2: Fluid-Structure Interaction (FSI)

**Problem**: Stokes flow in Ω_fluid coupled to elasticity in Ω_structure via interface traction.

### Setup: Three meshes (fluid, structure, interface boundary)

```bash
create_submesh name=fluid_mesh tags_name=domain_tags tag_values=[1]
create_submesh name=structure_mesh tags_name=domain_tags tag_values=[2]
# Interface is marked as facet_tags from Gmsh or mark_boundaries
```

### Iteration pattern (Gauss-Seidel partitioned coupling)

**Initialize**: u_struct = 0, p_fluid = 0, velocity_fluid = (0, 0)

**Loop** for i = 1..N_iter:

1. **Solve Stokes on fluid domain** with no-slip at structure interface
   - Extract interface velocity from previous u_struct iteration
   - Apply as Dirichlet BC on fluid interface

2. **Extract traction from Stokes solution**
   - Compute σ_fluid·n at interface
   - Transfer to structure mesh

3. **Solve elasticity on structure domain** with traction BC
   ```bash
   apply_boundary_condition value=stress_vector boundary=interface_tag function_space=u_space
   ```
   - This enforces g = σ_fluid·n on structure boundary

4. **Check convergence**: ||u_struct_new - u_struct_old||_L2 < tol
   - If converged: exit loop
   - Else: u_struct ← u_struct_new, return to step 1

### Relaxation improvement (Aitken acceleration)
For slow partitioned coupling, apply relaxation:
```
u_struct_relaxed = u_struct_old + omega * (u_struct_new - u_struct_old)
```
where omega is auto-adjusted based on residual history.

---

## Workflow 3: Custom Multi-Physics (run_custom_code approach)

For complex coupling logic (e.g., implicit monolithic, reaction-based transfer, ALE), use:

```bash
run_custom_code code='
import numpy as np
from dolfinx import fem, mesh

# Get session state
session_state = session.list_objects()
thermal_solution = session.functions["T_h"]
mechanical_space = session.spaces["u_space"]

# Custom transfer: interpolate + smooth + scale
from dolfinx.fem import Function
T_mech = Function(mechanical_space)
# ... scipy interpolation or DOLFINx interpolate_nonmatching_meshes

# Update material properties
session.materials["alpha"].value = compute_alpha(T_mech)
'
```

---

## Best Practices

| Pattern | Rationale |
|---------|-----------|
| **Store intermediate solutions** | Save T_h before overwriting for coupling reference |
| **Use entity maps** | Track parent↔child mesh relationships for cross-mesh operations |
| **Verify energy balance** | Integrate thermal power into mechanical work as sanity check |
| **Start with staggered** | Easier to debug than monolithic; convergence guaranteed |
| **Export all fields** | Bundle solutions from both domains into single VTK for post-processing |
| **Check mesh alignment** | Interface nodes should align or use mortar method |

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Interpolation produces zeros | Source mesh ≠ target mesh range | Use `source_mesh` parameter in `interpolate` |
| Solver diverges after coupling | Incompatible BCs at interface | Ensure BC values match across domains |
| Slow convergence (partitioned) | Too few coupling iterations | Increase N_iter or apply Aitken relaxation |
| NaN in thermal strain | Source T_h out of bounds | Check T_h values with `evaluate_solution` |
| Interface tractions don't balance | Normal direction sign | Verify `n` direction is outward from domain |

---

## Post-Processing Multi-Physics Results

After coupling iterations converge, validate and export:

```python
# Evaluate at interface points
evaluate_solution(points=[[0.5, 0.0], [0.5, 0.5]], function_name="T_fluid")
evaluate_solution(points=[[0.5, 0.0], [0.5, 0.5]], function_name="T_solid")

# Check interface continuity (values should match)
query_point_values(points=[[0.5, 0.0]], function_name="T_fluid")
query_point_values(points=[[0.5, 0.0]], function_name="T_solid")

# Export both domain solutions
export_solution(filename="fluid_solution", format="xdmf", functions=["T_fluid"])
export_solution(filename="solid_solution", format="xdmf", functions=["T_solid"])

# Generate combined report
generate_report(
    title="Multi-Physics Coupling Report",
    include_plots=True,
    include_solver_info=True,
    output_file="coupling_report.html"
)

# Bundle all outputs
bundle_workspace_files(
    file_paths=["*.xdmf", "*.h5", "*.png", "coupling_report.html"],
    archive_name="multi_physics_results.zip"
)
```

---

## Tools Used

- `create_submesh`: Extract domain-specific mesh
- `interpolate`: Transfer solutions across meshes
- `create_function`: Store coupling intermediate data
- `apply_boundary_condition`: Enforce interface conditions
- `solve`: Domain-specific solve
- `run_custom_code`: Complex custom coupling logic
- `evaluate_solution`: Check intermediate values
- `export_solution`: Bundle all physics for visualization
- `query_point_values`: Verify interface solution continuity
- `generate_report`: Create coupled simulation summary
- `bundle_workspace_files`: Package all results for delivery
