# Method of Manufactured Solutions (MMS) for Code Verification

**When to use**: Verify that your PDE solver implementation achieves correct convergence rates under mesh refinement.

**Why MMS**: Code verification ≠ validation
- **Verification**: Does code solve PDE correctly? (mathematical correctness)
- **Validation**: Does PDE model match physics? (realism)
- **MMS verifies**: Your discretization, assembly, and solver are correct

## Core Principle

1. Choose analytical solution u_exact(x) (smooth, non-trivial)
2. Substitute into PDE operator → derive source term f_exact
3. Solve PDE numerically with f = f_exact and BC = u_exact
4. Compare u_numerical vs u_exact on sequence of meshes
5. Verify convergence rate: should be O(h^{p+1}) for L2, O(h^p) for H1 (p = element degree)

**Key insight**: If you choose u_exact and derive f, the exact solution is guaranteed; any discrepancy means your code has a bug.

---

## Step-by-Step Workflow

### Phase 1: Design the Manufactured Solution

**Choose u_exact** (smooth, boundary-compatible):
- Good: u_exact = sin(πx)sin(πy), u_exact = (1-x²)(1-y²)
- Bad: u_exact = x, u_exact = |x| (singular, discontinuous derivatives)
- Bad: u_exact = constant (no information)

**Requirements**:
- Must satisfy essential BCs (e.g., if BC is u=0 at x=0, then u_exact(0,y)=0)
- Must be sufficiently smooth for the element degree p (at least C^{p+1})

**Example for Poisson on [0,1]²** with zero Dirichlet BCs:
```
u_exact(x,y) = sin(π*x) * sin(π*y)
```
Already satisfies u=0 on all boundaries. ✓

---

### Phase 2: Derive the Source Term

Substitute u_exact into strong form PDE:
```
-∇²u = f   →   f = -∇²u_exact
```

**For u_exact = sin(πx)sin(πy)**:
```
∂u/∂x = π*cos(πx)*sin(πy)
∂²u/∂x² = -π²*sin(πx)*sin(πy)

∂u/∂y = sin(πx)*π*cos(πy)
∂²u/∂y² = -π²*sin(πx)*sin(πy)

∇²u = ∂²u/∂x² + ∂²u/∂y² = -2π²*sin(πx)*sin(πy)

f = -∇²u = 2π²*sin(πx)*sin(πy)
```

**MCP call** (register source term):
```bash
set_material_properties name=f value='2*pi**2*sin(pi*x[0])*sin(pi*x[1])'
```

---

### Phase 3: Set up Problem with Exact BCs

**Boundary conditions**: Must match u_exact exactly

```bash
# For u_exact = sin(πx)sin(πy), which is 0 on all boundaries:
apply_boundary_condition value=0 boundary='x[0]==0' function_space=V_space
apply_boundary_condition value=0 boundary='x[0]==1' function_space=V_space
apply_boundary_condition value=0 boundary='x[1]==0' function_space=V_space
apply_boundary_condition value=0 boundary='x[1]==1' function_space=V_space
```

Or use expression form (newer MCP syntax):
```bash
apply_boundary_condition value='sin(pi*x[0])*sin(pi*x[1])' boundary='True' function_space=V_space
```

---

### Phase 4: Create Function Space and Variational Form

```bash
create_mesh name=mesh0 shape=unit_square nx=8 ny=8
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh0
```

Define variational form (weak Poisson):
```bash
define_variational_form bilinear='inner(grad(u), grad(v)) * dx' linear='f * v * dx'
```

---

### Phase 5: Solve and Compute Error

```bash
solve solution_name=u_h0
```

Compute L2 error:
```bash
compute_error function_name=u_h0 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2
```

Returns:
```
{
  "error_value": 0.00456,    # ||u_h - u_exact||_L2
  "norm_type": "L2",
  "function_name": "u_h0"
}
```

Compute H1 error (seminorm, for reference):
```bash
compute_error function_name=u_h0 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=H1
```

---

### Phase 6: Refine Mesh and Repeat

**Loop over refinement levels**:

```
FOR k = 1..4:
  1. If k=0: mesh_k = mesh0 (already created)
  2. Else: refine_mesh name=mesh{k} new_name=mesh{k+1}
  3. create_function_space name=V{k+1} family=Lagrange degree=1 mesh_name=mesh{k+1}
  4. define_variational_form ... (same form)
  5. solve solution_name=u_h{k+1}
  6. compute_error function_name=u_h{k+1} exact='...' norm_type=L2 → error_{k+1}
  7. Store: h_k, error_k
```

**Collect errors**:
```
Level | h       | Error L2  | Error H1  | Rate L2 | Rate H1
------|---------|-----------|-----------|---------|----------
0     | 1/8     | 4.56e-3   | 4.23e-2   |    -    |    -
1     | 1/16    | 1.14e-3   | 2.16e-2   | 2.00    | 0.97
2     | 1/32    | 2.85e-4   | 1.08e-2   | 2.00    | 1.00
3     | 1/64    | 7.13e-5   | 5.42e-3   | 2.00    | 1.00
```

---

### Phase 7: Compute Convergence Rates

**Richardson extrapolation** (from two consecutive refinements):

For 2D with uniform refinement (h_new = h_old / 2):
```
Rate = log(error_old / error_new) / log(2)
```

**Expected rates**:
- **P1 Poisson**: L2 = 2, H1 = 1
- **P2 Poisson**: L2 = 3, H1 = 2
- **DG-P1 Poisson** (with penalty): L2 = 2, H1 = 1
- **P1 Heat equation (time + space)**: Depends on time scheme
  - Implicit Euler + P1: time order 1, space order 2 → global ~1
  - Crank-Nicolson + P1: time order 2, space order 2 → global 2
- **P1 Elasticity**: Same as Poisson for each component

---

## MMS with DOLFINx MCP: Complete Example

### Poisson MMS (P1, 4 refinements)

```bash
# Initialize
reset_session

# Level 0
create_mesh name=mesh0 shape=unit_square nx=8 ny=8
create_function_space name=V0 family=Lagrange degree=1 mesh_name=mesh0
set_material_properties name=f value='2*pi**2*sin(pi*x[0])*sin(pi*x[1])'
apply_boundary_condition value=0 boundary=True function_space=V0
define_variational_form bilinear='inner(grad(u), grad(v)) * dx' linear='f * v * dx'
solve solution_name=u_h0
compute_error function_name=u_h0 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2

# Level 1
refine_mesh name=mesh0 new_name=mesh1
create_function_space name=V1 family=Lagrange degree=1 mesh_name=mesh1
# Note: forms and BCs are session-global; they apply to new space
solve solution_name=u_h1
compute_error function_name=u_h1 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2

# Level 2
refine_mesh name=mesh1 new_name=mesh2
create_function_space name=V2 family=Lagrange degree=1 mesh_name=mesh2
solve solution_name=u_h2
compute_error function_name=u_h2 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2

# Level 3
refine_mesh name=mesh2 new_name=mesh3
create_function_space name=V3 family=Lagrange degree=1 mesh_name=mesh3
solve solution_name=u_h3
compute_error function_name=u_h3 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2

# Use run_custom_code to compute convergence rates
run_custom_code code='
import numpy as np

# Collected errors from compute_error outputs (hardcode or read from log)
h_values = np.array([1/8, 1/16, 1/32, 1/64])
errors_L2 = np.array([4.56e-3, 1.14e-3, 2.85e-4, 7.13e-5])

rates = np.log(errors_L2[:-1] / errors_L2[1:]) / np.log(2.0)

print("Convergence Rates (L2 norm):")
for i, rate in enumerate(rates):
    print(f"  Level {i}→{i+1}: rate = {rate:.3f} (expected 2.0)")

if np.allclose(rates, 2.0, atol=0.1):
    print("✓ PASS: Convergence rates match expected O(h²)")
else:
    print("✗ FAIL: Rates deviate from expected")
'
```

---

## Manufactured Solutions Library

### Poisson 2D

**u_exact**:
```
sin(π*x)*sin(π*y)
```

**f**:
```
2π²*sin(π*x)*sin(π*y)
```

**BCs**: u = 0 on ∂Ω (already satisfied by u_exact)

**Expected rates**: P1: L2=2, H1=1; P2: L2=3, H1=2

---

### Poisson 3D

**u_exact**:
```
sin(π*x)*sin(π*y)*sin(π*z)
```

**f**:
```
3π²*sin(π*x)*sin(π*y)*sin(π*z)
```

**BCs**: u = 0 on ∂Ω

**Expected rates**: Same as 2D

---

### Heat Equation (Parabolic, Time-Dependent)

**Domain**: Ω = [0,1]², T ∈ [0, T_final]

**u_exact(x,y,t)**:
```
exp(-t) * sin(π*x) * sin(π*y)
```

**Source f(x,y,t)**:
```
From u_t - κ∇²u = f:
u_t = -exp(-t)*sin(π*x)*sin(π*y)
∇²u = -2π²*exp(-t)*sin(π*x)*sin(π*y)

f = u_t + κ∇²u = exp(-t)*sin(π*x)*sin(π*y) * (2κπ² - 1)
```

With κ=1: f = (2π² - 1)*exp(-t)*sin(π*x)*sin(π*y)

**BCs**: u(x,y,0) = sin(π*x)*sin(π*y); u=0 on ∂Ω

**Expected rates** (Implicit Euler in time + P1 space):
- Spatial: L2=2 (h²)
- Temporal: O(Δt)
- Global: dominated by coarser rate

---

### Linear Elasticity 2D

**u_exact** (displacement):
```
u_x = sin(π*x)*sin(π*y)
u_y = cos(π*x)*cos(π*y)
```

**Strain ε = (∇u + ∇u^T)/2**:
```
ε_xx = π*cos(π*x)*sin(π*y)
ε_yy = -π*sin(π*x)*cos(π*y)
ε_xy = 0.5*(π*cos(π*x)*cos(π*y) - π*sin(π*x)*sin(π*y))
```

**Stress σ = λ(tr ε)I + 2μ ε** (with λ, μ material constants)

**Source f = -∇·σ**:
```
f_x = -∂σ_xx/∂x - ∂σ_xy/∂y
f_y = -∂σ_xy/∂x - ∂σ_yy/∂y
```

Derive explicitly (tedious, but straightforward).

**BCs**: u = u_exact on ∂Ω

**Expected rates**: P1: L2=2, H1=1

---

### Helmholtz (Frequency Domain Wave)

**u_exact**:
```
sin(π*x)*sin(π*y)
```

**From -∇²u - k²u = f**:
```
f = -(−2π² − k²)*sin(π*x)*sin(π*y) = (2π² + k²)*sin(π*x)*sin(π*y)
```

Set k = π (wavenumber).

**BCs**: u = 0 on ∂Ω

**Expected rates**: P1: L2=2, H1=1

---

## Common MMS Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Wrong source derivation | Solution doesn't match | Double-check ∇²u computation symbolically |
| BC doesn't match u_exact | Boundary layers, oscillations | Verify u_exact at all boundaries |
| Element too low order | Low convergence rate | Use P2 or higher; or ensure u_exact ∈ polynomial space |
| Mesh too coarse initially | Pre-asymptotic regime | Start with nx=16 or higher |
| Using one level only | Can't verify rate | Need at least 3-4 refinement levels |
| Confusing L2 vs H1 | Rate check fails | L2 is O(h^{p+1}), H1 is O(h^p) |
| Source term f not set | Solver solves different PDE | Verify f is registered before solve |
| Numeric rate calculation | Rounding error | Use at least 4 decimal places in error |

---

## Troubleshooting

### Rates Too Low
**Symptoms**: Observed rate 1.5 when expecting 2.0

**Causes**:
1. Source term wrong → verify derivation
2. BCs don't match u_exact → check boundaries
3. Element order mismatch → if degree=1 but u_exact ∈ P2 space
4. Pre-asymptotic regime → coarsen initial mesh or use finer starting point

**Fix**: Plot error vs h on log-log scale; line should be straight and parallel to expected rate line.

### Rates Too High
**Symptoms**: Observed rate 2.5 when expecting 2.0

**Likely cause**: Superconvergence (rare but possible in special norms). Usually indicates:
- Problem is simpler than expected
- Norm used is not standard (e.g., nodal values instead of L2)

**Not a bug** — actually good sign of code quality. Verify with different u_exact.

### Negative Rate
**Symptoms**: Error increases with refinement

**Cause**: Implementation bug in error computation, not actually refining, or code reusing old solution

**Fix**: Check that mesh is actually refined with `get_mesh_info`; verify solution changes with `evaluate_solution`.

---

### Reporting MMS Results

After computing errors across mesh refinements:

```python
# Plot error convergence (if solution exists at finest mesh)
plot_solution(function_name="u_h", plot_type="contour", output_file="mms_solution.png")

# Generate verification report
generate_report(
    title="MMS Verification Report",
    include_plots=True,
    include_solver_info=True,
    include_mesh_info=True,
    output_file="mms_report.html"
)

# Export data for external plotting
bundle_workspace_files(
    file_paths=["mms_*.png", "mms_report.html"],
    archive_name="mms_verification.zip"
)
```

---

## Integration with DOLFINx Workflows

MMS fits naturally into development cycle:
```
1. Implement new PDE solver
2. Run MMS on simple case (Poisson, P1)
3. If rates correct: proceed to more complex physics
4. If rates wrong: debug formulation/implementation
5. Use MMS periodically (code maintenance, regression testing)
```

Automate via script:
```bash
for pde in poisson heat elasticity; do
  /setup-mms.md $pde
done
```

This will execute workflow as part of CI/CD.
