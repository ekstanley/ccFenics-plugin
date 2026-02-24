# Matrix-Free Solver Configuration

Interactive command to configure and test advanced solver techniques for large-scale systems.

## Step 1: Problem Assessment

**Question:** Why are you considering matrix-free or advanced solvers?

- **Very large system** (>1M DOFs, memory-limited)
- **Slow solver** (seeking better preconditioner)
- **Block structure** (Stokes, mixed systems)
- **Custom operator** (FFT-based, tensor-product structure)
- **Singular system** (pure Neumann, requires nullspace)
- **Other:** (describe)

**Action:** Your answer guides preconditioner selection.

---

## Step 2: Verify Session State

**Question:** Check what's currently in the session.

```python
# Call via MCP tool
result = get_session_state()
```

**Look for:**
- **Active mesh:** size, cell type, number of cells
- **Function spaces:** names, families, degrees, DOF counts
- **Forms defined:** bilinear and linear forms
- **Boundary conditions:** number and type

**Example output:**
```
Active mesh: "mesh_1"
  - Cells: 100,000
  - Vertices: 50,001

Function spaces:
  - "V": Lagrange degree 1, 50,001 DOFs
  - "Q": Lagrange degree 0, 100,000 DOFs

Forms defined:
  - Bilinear: a_form (compiled)
  - Linear: L_form (compiled)

Boundary conditions: 4 Dirichlet BCs
```

---

## Step 3: Problem Classification

**Based on problem type, recommend solver strategy:**

### Poisson / Elliptic (Laplace, Heat)

**Property:** Symmetric positive definite (SPD)

**Recommended:**
```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",           # Conjugate Gradient (best for SPD)
    pc_type="hypre",         # AMG preconditioner
    petsc_options={
        "-pc_hypre_type": "boomeramg",
        "-pc_hypre_boomeramg_strong_threshold": "0.25"
    }
)
```

**Alternative (if AMG fails):**
```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="gamg"           # GAMG (similar to Hypre)
)
```

**Size guidelines:**
- <100k DOFs: Direct solver sufficient
- 100k-1M DOFs: CG + AMG (10-50 iterations typical)
- >1M DOFs: CG + AMG with multilevel scaling (100+ iterations possible)

---

### Convection-Diffusion (Non-Symmetric)

**Property:** Non-symmetric due to advection term

**Recommended:**
```python
result = solve(
    solver_type="iterative",
    ksp_type="gmres",        # GMRES for non-symmetric
    pc_type="ilu",           # ILU preconditioner
    petsc_options={
        "-pc_factor_levels": "2",
        "-ksp_gmres_restart": "100"
    }
)
```

**ILU parameters:**
- `-pc_factor_levels 1` or `2`: Balance between fill and robustness
- `-ksp_gmres_restart 50` or `100`: GMRES restart (larger = more memory but faster)

**If ILU stalls:**
```python
# Try AMG (may help if diffusion-dominated)
result = solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="gamg"
)
```

---

### Stokes / Mixed System

**Property:** Saddle-point block structure

```
[A   B^T] [u]   [f]
[B   0  ] [p] = [g]
```

**Recommended:**
```python
result = solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="fieldsplit",
    petsc_options={
        "-pc_fieldsplit_type": "multiplicative",
        "-pc_fieldsplit_schur_fact_type": "lower",
        "-pc_fieldsplit_schur_precondition": "selfp",
        "-fieldsplit_u_ksp_type": "cg",
        "-fieldsplit_u_pc_type": "hypre",
        "-fieldsplit_p_ksp_type": "minres",
        "-fieldsplit_p_pc_type": "jacobi"
    }
)
```

**Key insight:**
- Velocity block (`u`): Use CG + AMG (Poisson-like)
- Pressure block (`p`): Use pressure mass matrix (`selfp`) or lumped mass
- Coupling: Schur complement with multiplicative (Gauss-Seidel style)

---

### Elasticity

**Property:** SPD, higher condition number than Poisson

**Recommended:**
```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "-pc_hypre_type": "boomeramg",
        "-pc_hypre_boomeramg_coarsen_type": "HMIS"  # Aggressive coarsening
    }
)
```

**For pure Neumann (no Dirichlet BCs):**
```python
# Attach rigid body modes (6 modes for 3D: 3 translations + 3 rotations)
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    nullspace_mode="rigid_body"  # MCP tool handles attachment
)
```

---

## Step 4: Configure PETSc Options

**Question:** Are you happy with the recommended configuration, or customize?

### Option A: Use Recommended (Quick)

```python
# For your problem type (e.g., Poisson):
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre"
)
```

### Option B: Fine-Tune Options

**Step 4a: KSP (Solver) Options**

```python
# Set tolerance and max iterations
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    rtol=1e-6,           # relative tolerance
    atol=1e-12,          # absolute tolerance
    max_iter=1000,       # max iterations
    pc_type="hypre"
)
```

**Step 4b: Preconditioner Options**

```python
# For AMG fine-tuning
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "-pc_hypre_type": "boomeramg",
        "-pc_hypre_boomeramg_strong_threshold": "0.3",     # higher = coarsen more
        "-pc_hypre_boomeramg_coarsen_type": "PMIS",        # parallel coarsening
        "-pc_hypre_boomeramg_num_sweeps": "2",             # smoother iterations
        "-pc_hypre_boomeramg_relax_type_up": "SOR/SSOR",   # interpolation
        "-pc_hypre_boomeramg_relax_type_down": "SOR/SSOR"
    }
)
```

**Common tuning:**

| Parameter | Low Value | High Value | When to Adjust |
|---|---|---|---|
| `strong_threshold` | 0.1 | 0.5 | 0.1 → coarse problem harder; 0.5 → fewer levels |
| `num_sweeps` | 1 | 3 | Increase if oscillation; decrease if slow |
| `coarsen_type` | CLJP | HMIS, PMIS | PMIS faster, sometimes less stable |

---

## Step 5: Handle Nullspace (if Pure Neumann)

**Question:** Are Dirichlet boundary conditions applied?

- **YES** (Dirichlet BCs on part or all boundary) → Skip, no nullspace needed
- **NO** (Pure Neumann everywhere) → Must attach nullspace

**For pure Neumann Poisson:**
```python
result = solve(
    nullspace_mode="constant"  # attaches constant mode (rank-1 nullspace)
)
```

**For pure Neumann elasticity:**
```python
result = solve(
    nullspace_mode="rigid_body"  # attaches 3 translations + 3 rotations
)
```

**For custom nullspace** (advanced, via `run_custom_code`):
```python
# See matrix-free-solvers/SKILL.md for custom nullspace implementation
```

---

## Step 6: Test Convergence

**Action:** Run solver and inspect convergence.

```python
# Solve with monitoring
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "-ksp_monitor": None,        # print residual at each iteration
        "-ksp_monitor_short": None   # compact format
    }
)
```

**Expected output:**
```
  0 KSP Residual norm 1.234e+00
  1 KSP Residual norm 1.200e-01
  2 KSP Residual norm 1.500e-02
  3 KSP Residual norm 1.800e-03
  4 KSP Residual norm 2.100e-04
  5 KSP Residual norm 2.400e-05
  6 KSP Residual norm 2.600e-06  ← converged (rtol=1e-6)
KSP Object: CONVERGED RTOL
```

**Convergence quality:**

| Pattern | Meaning | Status |
|---|---|---|
| Steady 10x decrease → quadratic at end | Excellent (ideal) | ✓ Accept |
| Linear 10x decrease throughout | Good (acceptable) | ✓ Accept |
| Stagnation (ratio 0.9) after few iterations | Preconditioner weak | ⚠ Tune parameters |
| Non-monotone, oscillations | Saddle-point or ill-conditioned | ⚠ Check problem setup |
| Divergence (increasing residual) | Solver/preconditioner failing | ✗ Reconfigure |

---

## Step 7: Compare Solver Performance

**Question:** How does the new solver compare to the old one?

**Benchmark structure:**

```python
import time

# Baseline (direct solver or previous approach)
t0 = time.time()
result_direct = solve(solver_type="direct")
t_direct = time.time() - t0

# New approach (iterative + preconditioner)
t0 = time.time()
result_iter = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={"-ksp_monitor": None}
)
t_iter = time.time() - t0

print(f"Direct solver:  {t_direct:.4f}s")
print(f"Iterative solver: {t_iter:.4f}s (speedup: {t_direct/t_iter:.1f}x)")
```

**Interpretation:**

| Speedup | Status | Meaning |
|---|---|---|
| <1x | ✗ Worse | Preconditioner overhead too high; switch back to direct |
| 1-2x | ⚠ Marginal | May improve on larger problems; tune parameters |
| 2-10x | ✓ Good | Worthwhile improvement; keep iterative |
| >10x | ✓ Excellent | Significant scaling advantage; iterative is better |

---

## Step 8: Profile Assembly vs Solve

**Question:** Where is the time spent?

```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "-log_view": None,      # full PETSc breakdown
        "-log_view_memory": None # include memory usage
    }
)

# Output includes:
# - MatAssembly time
# - KSPSetup time
# - KSPSolve time
# - MatMult (most expensive in iterative)
# - PCApply (preconditioner application)
```

**Typical breakdown (1M DOF Poisson with AMG):**
```
Assembly: 2.0s  (20%)
Solver setup: 1.0s (10%)
Solver iterations: 7.0s (70%)
  - MatMult: 4.5s
  - PCApply: 2.5s
Total: 10s
```

**Action based on breakdown:**
- **Assembly dominates:** Form is complex; can't optimize much (accept cost)
- **Solve setup dominates:** Preconditioner expensive; use cheaper PC or accept cost
- **Solve dominates:** Choose different iterative method or better preconditioner

---

## Step 9: Scaling Study (Advanced)

**Question:** How does performance scale with mesh refinement?

```python
# Refinement sequence
for ref in range(4):
    mesh_refined = refine_mesh(mesh, new_name=f"mesh_ref{ref}")
    V = create_function_space("V", family="Lagrange", degree=1, mesh_name=f"mesh_ref{ref}")

    # Create problem at this refinement
    # ... define a, L, bcs ...

    # Solve and time
    t0 = time.time()
    result = solve(solver_type="iterative", ksp_type="cg", pc_type="hypre")
    t_solve = time.time() - t0

    n_dofs = V.dofmap.index_map.size_global
    n_iters = result["iterations"]

    print(f"Refinement {ref}: {n_dofs:8d} DOFs, {n_iters:3d} iters, {t_solve:.4f}s")
```

**Expected scaling:**

| Preconditioner | Iteration Count | Time |
|---|---|---|
| None | O(√n) iterations | O(n^1.5) time |
| Jacobi | O(n^0.5) iterations | O(n) time |
| ILU | ~O(1) iterations | O(n) time |
| **AMG** | **~O(1) iterations** | **O(n) time** (optimal) |

**Ideal output:**
```
Refinement 0:     1,000 DOFs,  15 iters, 0.0001s
Refinement 1:     4,000 DOFs,  18 iters, 0.0005s
Refinement 2:    16,000 DOFs,  20 iters, 0.0020s
Refinement 3:    64,000 DOFs,  22 iters, 0.0085s
Refinement 4:   256,000 DOFs,  23 iters, 0.0350s
  ↑ Iterations nearly constant (AMG working optimally)
  ↑ Time scales roughly linearly (good)
```

---

## Step 10: Production Configuration

**Action:** Save recommended options for production runs.

```python
# Store configuration
solver_config = {
    "solver_type": "iterative",
    "ksp_type": "cg",
    "pc_type": "hypre",
    "rtol": 1e-6,
    "atol": 1e-12,
    "max_iter": 1000,
    "petsc_options": {
        "-pc_hypre_type": "boomeramg",
        "-pc_hypre_boomeramg_strong_threshold": "0.25",
        "-pc_hypre_boomeramg_coarsen_type": "PMIS"
    }
}

# Use in solve
result = solve(**solver_config)
```

---

## Troubleshooting

| Issue | Symptom | Fix |
|---|---|---|
| **Divergence** | Residual increases | Check matrix is SPD; try `-pc_type ilu` |
| **Stagnation** | Iterations plateau | Increase `strong_threshold`, use `-pc_type gamg` |
| **Memory bloat** | Out-of-memory during setup | Reduce coarse level size (`-pc_gamg_coarse_eq_limit 50`) |
| **Slow preconditioner** | `PCSetUp` time >50% | Use simpler PC (e.g., Jacobi) or `-pc_factor_levels 1` |
| **Nonconvergence** | `KSP did not converge` | Increase `max_iter`, relax `rtol`, check BCs |
| **Singular matrix** | `MAT_FACTOR_NUMERIC_ZEROPIVOT` | Attach nullspace (`nullspace_mode="constant"`) |

---

## Summary Checklist

- [ ] Problem type identified (elliptic, non-symmetric, block system, singular)
- [ ] Preconditioner recommended based on type
- [ ] Tolerance and iteration limits set appropriately
- [ ] Nullspace attached if pure Neumann (via `nullspace_mode`)
- [ ] PETSc options configured for target system size
- [ ] Convergence monitored (`-ksp_monitor`)
- [ ] Performance compared to baseline (speedup measured)
- [ ] Scaling study done on refined meshes
- [ ] Configuration saved for production use

---

## See Also

- **SKILL:** `matrix-free-solvers/SKILL.md` — Full technical details
- **Reference:** `matrix-free-solvers/references/preconditioner-guide.md` — Detailed preconditioner table
- **Tool:** `solve()` MCP tool — Core solver interface
- **Command:** `/newton-loop` — For nonlinear systems
- **Command:** `/mpi-setup` — For parallel execution
