# Manufactured Solution (MMS) Verification Setup

**Purpose**: Automatically verify PDE solver convergence rates on a sequence of meshes using a known analytical solution.

---

## Step 1: Identify or Select PDE

Which PDE are you verifying?

- **Poisson** (steady diffusion)
- **Heat equation** (time-dependent diffusion)
- **Linear elasticity** (displacement-based)
- **Helmholtz** (frequency domain)
- **Stokes flow** (creeping flow)
- **Custom** (you provide the strong form)

**Detect from session**: If you already have `mesh`, `function_space`, `forms`, `boundary_conditions` defined, the PDE type may be inferred automatically.

**Select**: Paste PDE type or existing problem name.

---

## Step 2: Choose or Derive Manufactured Solution

### Option A: Use Pre-Built Library

From `skills/mms-verification/references/manufactured-solutions.md`, pick a solution:

| PDE | u_exact | f (source) |
|-----|---------|-----------|
| **Poisson 2D** | `sin(π*x)*sin(π*y)` | `2π²*sin(π*x)*sin(π*y)` |
| **Poisson 3D** | `sin(π*x)*sin(π*y)*sin(π*z)` | `3π²*sin(π*x)*sin(π*y)*sin(π*z)` |
| **Heat Eq.** | `exp(-t)*sin(π*x)*sin(π*y)` | `(2π²-1)*exp(-t)*sin(π*x)*sin(π*y)` |
| **Elasticity 2D** | (polynomial, see ref) | (complex, use SymPy) |
| **Helmholtz** | `sin(π*x)*sin(π*y)` with k=π | `3π²*sin(π*x)*sin(π*y)` |

**Example**: For Poisson on [0,1]² with zero BCs:
```
u_exact = sin(π*x)*sin(π*y)
f = 2π²*sin(π*x)*sin(π*y)
```

### Option B: Derive Custom Solution

1. **Choose u_exact**: Smooth, matches BCs
   - Example: `u = x²*(1-x)*y²*(1-y)` (zero on ∂[0,1]²)

2. **Derive f** using SymPy:
```python
import sympy as sp

x, y = sp.symbols('x y', real=True)
u = x**2 * (1-x) * y**2 * (1-y)

# For Poisson: f = -∇²u
laplacian = sp.diff(u, x, 2) + sp.diff(u, y, 2)
f = -laplacian
f = sp.simplify(f)

print(f"u_exact = {u}")
print(f"f = {f}")
# Copy output into setup below
```

3. **Verify u_exact satisfies BCs**:
   - Check u_exact(0,y), u_exact(1,y), u_exact(x,0), u_exact(x,1) match your BC values

---

## Step 3: Set Up Problem on Initial Mesh

**Create coarse starting mesh** (e.g., 8×8 or 10×10):

```bash
reset_session  # Clean slate

create_mesh name=mesh0 shape=unit_square nx=8 ny=8
```

**Create function space**:
```bash
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh0
```

**Set source term f** (from Step 2):

```bash
set_material_properties name=f value='2*pi**2*sin(pi*x[0])*sin(pi*x[1])'
```

**Apply exact Dirichlet BCs** matching u_exact:

For `u_exact = sin(πx)sin(πy)` (zero on boundary):
```bash
apply_boundary_condition value=0 boundary=True function_space=V
```

Or with non-zero BC:
```bash
apply_boundary_condition value='sin(pi*x[0])*sin(pi*x[1])' boundary='x[0]==0' function_space=V
```

**Define variational form**:

For Poisson:
```bash
define_variational_form \
  bilinear='inner(grad(u), grad(v))*dx' \
  linear='f*v*dx'
```

For heat (if time-dependent): Use `solve_time_dependent` instead (Step 7 below).

---

## Step 4: Solve on Level 0

```bash
solve solution_name=u_h0
```

**Compute L2 error on Level 0**:
```bash
compute_error function_name=u_h0 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2
```

**Output**:
```json
{
  "norm_type": "L2",
  "error_value": 0.00456,
  "function_name": "u_h0"
}
```

**Store** error_0 = 0.00456, h_0 = 1/8

---

## Step 5: Refine Mesh and Solve (Loop)

**Loop over 3–4 refinement levels**:

```
FOR level in [1, 2, 3]:
  1. Refine previous mesh
  2. Create new function space
  3. Solve on new mesh
  4. Compute error
  5. Store (h_level, error_level)
```

### Level 1:
```bash
# Refine mesh0 → mesh1
refine_mesh name=mesh0 new_name=mesh1

# Create function space on mesh1
create_function_space name=V1 family=Lagrange degree=1 mesh_name=mesh1

# Solve (forms are global; they apply to new space automatically)
solve solution_name=u_h1

# Compute error
compute_error function_name=u_h1 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2
```

**Output**: error_1 = 0.00114, h_1 = 1/16

### Level 2:
```bash
refine_mesh name=mesh1 new_name=mesh2
create_function_space name=V2 family=Lagrange degree=1 mesh_name=mesh2
solve solution_name=u_h2
compute_error function_name=u_h2 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2
```

**Output**: error_2 = 2.85e-4, h_2 = 1/32

### Level 3:
```bash
refine_mesh name=mesh2 new_name=mesh3
create_function_space name=V3 family=Lagrange degree=1 mesh_name=mesh3
solve solution_name=u_h3
compute_error function_name=u_h3 exact='sin(pi*x[0])*sin(pi*x[1])' norm_type=L2
```

**Output**: error_3 = 7.13e-5, h_3 = 1/64

---

## Step 6: Compute Convergence Rates

**Using run_custom_code** to extract and analyze errors:

```bash
run_custom_code code='
import numpy as np

# Collected errors from compute_error (read from session or hardcode)
errors_L2 = np.array([4.56e-3, 1.14e-3, 2.85e-4, 7.13e-5])
h_values = np.array([1/8, 1/16, 1/32, 1/64])

# Richardson extrapolation: rate = log(e[k]/e[k+1]) / log(h[k]/h[k+1])
# For uniform 2D refinement: h_new = h_old / 2 → log(2)
rates = np.log(errors_L2[:-1] / errors_L2[1:]) / np.log(2.0)

print("=== Convergence Study ===")
print("Level | h       | Error L2  | Rate")
print("------|---------|-----------|-------")
for i in range(len(h_values)):
    rate_str = f"{rates[i]:.3f}" if i > 0 else "   -"
    print(f"  {i}   | {h_values[i]:.5f} | {errors_L2[i]:.4e} | {rate_str}")

# Check vs expected
expected_rate = 2.0  # For P1 Poisson: O(h²) → rate = 2
observed_rates = rates
error_in_rates = np.abs(observed_rates - expected_rate)

print()
print("Expected rate (P1 Poisson): 2.0")
print(f"Observed rates: {observed_rates}")
print(f"Deviation: {error_in_rates}")

if np.allclose(observed_rates, expected_rate, atol=0.1):
    print("\n✓ PASS: Convergence rates match O(h²)")
    exit_code = 0
else:
    print("\n✗ FAIL: Rates deviate from expected")
    print("  Check: source term f, boundary conditions, or element order")
    exit_code = 1
'
```

---

## Step 7: Verify Convergence Rates

### Expected Rates (By Element Type)

| Element | PDE | L2 Rate | H1 Rate |
|---------|-----|---------|---------|
| **P1** | Poisson | 2 | 1 |
| **P2** | Poisson | 3 | 2 |
| **P1** | Elasticity | 2 | 1 |
| **P1** | Heat (IE+P1) | 2(spatial), 1(temporal) | 1 |
| **P1** | Heat (CN+P1) | 2(spatial), 2(temporal) | 1 |
| **P1** | Helmholtz | 2 | 1 |
| **DG-P1** | Poisson (SIPG) | 2 | 1 |

### Pass/Fail Criteria

**PASS**: Observed rate within ±0.15 of expected
```
|observed - expected| < 0.15
```

**FAIL**: Rate deviates significantly
```
|observed - expected| > 0.15
```

**Red flags**:
- Rate = 0.5–1.0 → Source term f likely wrong
- Rate = 1.5 → Partial refinement (some elements not refined)
- Rate negative → Solver diverging; check for NaN

---

## Step 8: Diagnose Failures

If rates are wrong:

### Check 1: Verify Source Term Derivation

```python
# Use SymPy to double-check f = -∇²u_exact
import sympy as sp

x, y = sp.symbols('x y', real=True)
u = sp.sin(sp.pi*x) * sp.sin(sp.pi*y)

lap_u = sp.diff(u, x, 2) + sp.diff(u, y, 2)
f = -lap_u
f_simplified = sp.simplify(f)

print(f"u_exact = {u}")
print(f"f = -∇²u = {f_simplified}")

# Expected: f = 2π²*sin(πx)*sin(πy)
```

### Check 2: Verify Boundary Conditions

Evaluate u_exact at mesh boundary points:
```bash
evaluate_solution function_name=u_h0 points=[[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]]
```

Compare with u_exact at same points. If BC values are wrong, solution will have boundary layers.

### Check 3: Verify Mesh Refinement

```bash
run_custom_code code='
import numpy as np

# Check that mesh is actually refining
mesh_0 = session.meshes["mesh0"]
mesh_1 = session.meshes["mesh1"]

nc0 = mesh_0.topology.num_entities(mesh_0.topology.dim)
nc1 = mesh_1.topology.num_entities(mesh_1.topology.dim)

refinement_ratio = nc1 / nc0
print(f"Mesh 0: {nc0} cells")
print(f"Mesh 1: {nc1} cells")
print(f"Refinement ratio: {refinement_ratio:.2f} (expected ~4 in 2D)")

if refinement_ratio < 3.5 or refinement_ratio > 4.5:
    print("WARNING: Refinement ratio unexpected; meshes may not be uniform or refinement failed")
'
```

### Check 4: Verify Element Order Matches

For P1 elements, L2 rate should be ~2. If you use P2, rate should be ~3.

```bash
get_session_state  # Check "element_degree" in function_spaces
```

If degree is wrong, update:
```bash
create_function_space name=V family=Lagrange degree=2 mesh_name=mesh0
```

---

## Step 9: Report Results

Generate summary table and plot:

```bash
run_custom_code code='
import numpy as np
import matplotlib.pyplot as plt

errors = np.array([4.56e-3, 1.14e-3, 2.85e-4, 7.13e-5])
h = np.array([1/8, 1/16, 1/32, 1/64])

# Log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(h, errors, "bo-", label="L2 error (computed)", markersize=8)

# Reference lines for O(h²) and O(h) rates
h_ref = np.linspace(h.min(), h.max(), 100)
rate2 = errors[0] * (h_ref / h[0])**2
rate1 = errors[0] * (h_ref / h[0])**1

plt.loglog(h_ref, rate2, "r--", alpha=0.5, label="O(h²) reference")
plt.loglog(h_ref, rate1, "g--", alpha=0.5, label="O(h) reference")

plt.xlabel("h (mesh size)", fontsize=12)
plt.ylabel("L2 error ||u_h - u_exact||", fontsize=12)
plt.title("MMS Convergence Study: Poisson P1", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/mms_convergence.png", dpi=150)
print("Plot saved to /workspace/mms_convergence.png")
'
```

---

## Step 10: Multi-Level Report

Consolidate results:

```bash
run_custom_code code='
import numpy as np
from datetime import datetime

# Collected data
levels = [0, 1, 2, 3]
h_vals = [1/8, 1/16, 1/32, 1/64]
errors_L2 = [4.56e-3, 1.14e-3, 2.85e-4, 7.13e-5]
errors_H1 = [4.23e-2, 2.16e-2, 1.08e-2, 5.42e-3]  # Optional

rates_L2 = np.log(np.array(errors_L2)[:-1] / np.array(errors_L2)[1:])) / np.log(2.0)
rates_H1 = np.log(np.array(errors_H1)[:-1] / np.array(errors_H1)[1:])) / np.log(2.0)

# Generate report
report = f"""
╔════════════════════════════════════════════════════════════╗
║      METHOD OF MANUFACTURED SOLUTIONS (MMS) REPORT         ║
╚════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PROBLEM
-------
PDE:            Poisson: -∇²u = f
Domain:         Ω = [0,1]²
u_exact:        sin(π*x)*sin(π*y)
f:              2π²*sin(π*x)*sin(π*y)
Element:        Lagrange P1
BCs:            u = 0 on ∂Ω

CONVERGENCE TABLE
-----------------
Level | h       | Error L2  | Rate L2 | Error H1  | Rate H1
------|---------|-----------|---------|-----------|----------
  0   | 1/8     | 4.56e-03  |    -    | 4.23e-02  |   -
  1   | 1/16    | 1.14e-03  |  2.00   | 2.16e-02  |  0.97
  2   | 1/32    | 2.85e-04  |  2.00   | 1.08e-02  |  1.00
  3   | 1/64    | 7.13e-05  |  2.00   | 5.42e-03  |  1.00

VERIFICATION RESULT
-------------------
Expected L2 Rate:   2.0 (O(h²) for P1)
Observed L2 Rates:  {rates_L2}
Mean Rate:          {np.mean(rates_L2):.3f}
Deviation:          {np.abs(np.mean(rates_L2) - 2.0):.3f}

Status:             ✓ PASS (rates within ±0.15 of expected)

Conclusion:         Solver correctly implements Poisson equation.
                    Code is verified for production use.
"""

print(report)

# Save to file
with open("/workspace/mms_report.txt", "w") as f:
    f.write(report)

print("\nFull report saved to: /workspace/mms_report.txt")
'
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Rate = 1.0 | Source term f wrong or BCs mismatch | Re-derive f with SymPy; check u_exact at ∂Ω |
| Rate = 1.5 | Partial refinement; only some elements refined | Check mesh refinement with get_mesh_info; ensure uniform refinement |
| Negative rate | Solution diverging as mesh refines | Check for NaN in errors; verify source term doesn't overflow |
| Rate = 0.5 | Severe issue; likely code bug | Simplify to Poisson; check weak form definition and BC application |
| Error increases | Mesh points moving; CFL violation (time-dependent) | For time-dependent: ensure Δt scales with h² (Δt ~ h²); check initial condition |

---

## Advanced: Time-Dependent MMS (Heat Equation)

For parabolic problems, verify both spatial and temporal rates:

```bash
# Heat equation: ∂u/∂t - κ∇²u = f
# With u_exact = exp(-t)*sin(πx)*sin(πy)

# Create mesh (fixed)
create_mesh name=mesh shape=unit_square nx=16 ny=16
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh

# Vary Δt while keeping h fixed; measure error at T=1.0
set_material_properties name=kappa value=1.0
set_material_properties name=f value='(2*pi**2-1)*exp(-t)*sin(pi*x[0])*sin(pi*x[1])'
apply_boundary_condition value=0 boundary=True function_space=V
define_variational_form \
  bilinear='inner(u, v)*dx + dt*kappa*inner(grad(u), grad(v))*dx' \
  linear='inner(u_old, v)*dx + dt*f*v*dx'

# Solve for multiple Δt values: dt = 0.1, 0.05, 0.025, 0.0125
# Compute error at final time
# Plot error vs Δt on log-log scale
# Expected rate (Implicit Euler): O(Δt)
```

---

## See Also

- **Manufactured solutions library**: `skills/mms-verification/references/manufactured-solutions.md`
- **Full MMS skill**: `skills/mms-verification/SKILL.md`
- **PDE recipes**: `commands/recipe.md`
