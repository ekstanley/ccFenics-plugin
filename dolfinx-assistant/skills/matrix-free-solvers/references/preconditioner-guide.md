# Preconditioner Selection Guide

Detailed decision tree and configuration reference for choosing the right preconditioner.

## Quick Selection Decision Tree

```
Problem Type?
├─ Symmetric Positive Definite (SPD)
│  ├─ Small (<100k DOFs)
│  │  └─> DIRECT: solve(solver_type="direct")
│  └─ Large (>100k DOFs)
│     ├─ Elliptic / Diffusion?
│     │  └─> CG + AMG: ksp_type="cg", pc_type="gamg" or "hypre"
│     └─ Other structured?
│        └─> CG + ILU: ksp_type="cg", pc_type="ilu"
├─ Non-symmetric (Convection, Advection)
│  ├─ Small (<100k DOFs)
│  │  └─> DIRECT: solve(solver_type="direct")
│  └─ Large
│     ├─ Dominated by diffusion?
│     │  └─> GMRES + ILU: ksp_type="gmres", pc_type="ilu"
│     └─ Strongly advective?
│        └─> GMRES + FieldSplit or AMG: depends on structure
└─ Saddle-Point (Stokes, Mixed)
   ├─ Small
   │  └─> DIRECT
   └─ Large
      └─> GMRES + FieldSplit + Schur: fieldsplit preconditioner
```

## Problem Type → Solver → Preconditioner Matrix

| Problem | Matrix Property | Recommended KSP | Recommended PC | Notes |
|---------|---|---|---|---|
| Poisson (Dirichlet) | SPD | cg | gamg, hypre | AMG optimal for large |
| Poisson (Neumann) | SPD + nullspace | cg | gamg + nullspace | Attach constant mode |
| Diffusion (time-dependent) | SPD, mass+stiffness | cg | ilu, hypre | Mass matrix well-conditioned |
| Convection-diffusion | Non-sym, Re dependent | gmres | ilu | Increase KSP restarts if needed |
| Navier-Stokes (IPCS) | Pressure Poisson | cg | gamg, hypre | Precondition velocity, pressure separately |
| Stokes (monolithic) | Saddle-point 2x2 block | gmres | fieldsplit+schur | Schur complement for pressure |
| Elasticity (pure Dirichlet) | SPD, 3-4x condition number | cg | gamg, hypre | Comparable to Poisson |
| Elasticity (pure Neumann) | SPD + rigid body nullspace | cg | gamg + rigid_body nullspace | Attach 6 rigid body modes |
| Helmholtz (high freq) | Indefinite, hard to precondition | gmres | ilu, mumps (direct) | Often requires direct solver |
| Mixed Poisson (RT/BDM) | Saddle-point block | gmres | fieldsplit | Similar to Stokes |
| DG formulation | Denser matrix, higher condition | gmres | ilu, block-jacobi | May need denser preconditioner |

## Preconditioner Configuration Examples

### AMG (GAMG) for Elliptic Problems

**When to use:** Poisson, diffusion, elasticity; >100k DOFs.

**Configuration:**

```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="gamg",
    petsc_options={
        "-pc_gamg_type": "agg",           # aggregation-based
        "-pc_gamg_threshold": "0.05",     # strong coupling threshold
        "-pc_gamg_square_graph": "3",     # square graph 3 times
        "-pc_gamg_coarse_eq_limit": "100" # coarsen until 100 DOFs
    }
)
```

**For Hypre BoomerAMG** (often faster):

```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "-pc_hypre_type": "boomeramg",
        "-pc_hypre_boomeramg_strong_threshold": "0.25",
        "-pc_hypre_boomeramg_coarsen_type": "PMIS",
        "-pc_hypre_boomeramg_num_sweeps": "2"
    }
)
```

### ILU for General Matrices

**When to use:** Non-symmetric, no clear structure; small-medium systems.

```python
result = solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="ilu",
    petsc_options={
        "-pc_factor_levels": "2",      # ILU(2) fill level
        "-pc_factor_shift_type": "INBLOCKS"  # numerical stability
    }
)
```

**For better stability (ILUT with threshold):**

```python
petsc_options={
    "-pc_factor_matrix_ordering": "rcm",  # reverse Cuthill-McKee
    "-pc_factor_shift_type": "INBLOCKS",
    "-pc_factor_diagonal_fill": "1"       # fill diagonal if zero
}
```

### FieldSplit + Schur for Stokes

**Problem:** 2x2 block system (velocity-pressure).

```python
result = solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="fieldsplit",
    petsc_options={
        "-pc_fieldsplit_type": "multiplicative",  # Gauss-Seidel type
        "-pc_fieldsplit_0_ksp_type": "cg",
        "-pc_fieldsplit_0_pc_type": "hypre",
        "-pc_fieldsplit_1_ksp_type": "minres",
        "-pc_fieldsplit_1_pc_type": "none",
        "-pc_fieldsplit_schur_fact_type": "lower",
        "-pc_fieldsplit_schur_precondition": "selfp"  # pressure mass matrix
    }
)
```

### Block Jacobi / Additive Schwarz for Domain Decomposition

**When to use:** Parallel systems with clear subdomain structure.

```python
result = solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="bjacobi",  # block Jacobi
    petsc_options={
        "-sub_ksp_type": "cg",
        "-sub_pc_type": "ilu",
        "-sub_pc_factor_levels": "2"
    }
)

# OR Additive Schwarz
# petsc_options={
#     "-pc_type": "asm",
#     "-sub_ksp_type": "cg",
#     "-sub_pc_type": "ilu"
# }
```

## Preconditioner Strength vs. Cost

| Preconditioner | Setup Time | Memory | Iteration Count | Total Time |
|---|---|---|---|---|
| **None (CG only)** | O(1) | O(n) | Very high | Very slow |
| **Jacobi** | O(n) | O(n) | High | Slow |
| **ILU(1)** | O(n) | O(n) | Medium | Moderate |
| **ILU(2)** | O(n) | O(n) | Lower | Moderate-fast |
| **GAMG** | O(n) | O(n log n) | Low | Fast |
| **Hypre AMG** | O(n) | O(n log n) | Low | Fast |
| **DIRECT (LU)** | O(n^{1.5}-n^3) | O(n-n^2) | 1 (converged) | Fast |

**Typical convergence curves:**
- No preconditioner: 10,000+ iterations for 1M DOF system
- ILU(2): 100-500 iterations
- AMG: 10-50 iterations
- Direct solver: 1 iteration (but expensive setup)

## AMG Parameters Tuning Guide

| Parameter | Low Value | High Value | Effect |
|---|---|---|---|
| `-pc_gamg_threshold` | 0.01 | 0.1 | Higher → fewer strong couplings → coarser grids → faster setup, slower solve |
| `-pc_gamg_square_graph` | 0 | 3 | Higher → smoother aggregates → better convergence → slower setup |
| `-pc_gamg_coarse_eq_limit` | 50 | 500 | Lower → coarsen more → more levels → slower iteration but faster total time |
| `-pc_hypre_boomeramg_strong_threshold` | 0.1 | 0.5 | Higher → fewer strong edges → more aggressive coarsening |
| `-pc_hypre_boomeramg_num_sweeps` | 1 | 4 | Higher → more smoothing → better convergence → slower iteration |

**Recommended starting points:**
- Poisson / Elliptic: threshold=0.05, square_graph=3
- Convection-dominated: threshold=0.1, square_graph=1
- Elasticity: threshold=0.05, square_graph=3, coarse_eq_limit=100

## Schur Complement Strategies

For block systems with block structure:
```
[A  B^T] [u]   [f]
[B  C  ] [p] = [g]
```

**Strategy 1: Diagonal Schur (S = B*diag(A)^{-1}*B^T)**
- Fast, but rough approximation
- Use for initial solves or debugging

**Strategy 2: Pressure Mass Matrix (S ≈ M_p)**
- For Stokes: use `-pc_fieldsplit_schur_precondition selfp`
- Good balance between accuracy and cost

**Strategy 3: Lumped Mass (M_p diagonal)**
- Faster than full mass matrix
- Reasonable for incompressible flow

**Strategy 4: LSC (Least-Squares Commutator)**
- Advanced: `-pc_fieldsplit_schur_precondition lsc`
- Good for variable viscosity or non-Newton flows

## Near-Singular Systems (Pure Neumann)

When matrix has 1D nullspace (constant mode for Poisson), attach via `nullspace_mode`:

```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    nullspace_mode="constant"  # MCP tool will attach it
)
```

**Or manually** (via `run_custom_code`):

```python
# Build constant vector
u_const = fem.Function(V)
u_const.x.array[:] = 1.0 / (V.dofmap.index_map.size_global)**0.5  # normalized

# Attach to matrix
nullspace = PETSc.NullSpace().create(comm=mesh.comm, vectors=[u_const.x.petsc_vec])
A.setNullSpace(nullspace)
```

## Saddle-Point Preconditioner Recipes

### Recipe 1: Stokes with velocity-pressure decoupling

```python
petsc_options={
    "-ksp_type": "gmres",
    "-ksp_gmres_restart": "100",
    "-pc_type": "fieldsplit",
    "-pc_fieldsplit_type": "schur",
    "-pc_fieldsplit_schur_fact_type": "lower",
    "-pc_fieldsplit_schur_precondition": "selfp",
    # Velocity block
    "-fieldsplit_u_ksp_type": "cg",
    "-fieldsplit_u_pc_type": "hypre",
    "-fieldsplit_u_pc_hypre_type": "boomeramg",
    # Pressure block
    "-fieldsplit_p_ksp_type": "minres",
    "-fieldsplit_p_pc_type": "none"
}
```

### Recipe 2: Mixed Poisson (Raviart-Thomas / BDM)

```python
petsc_options={
    "-ksp_type": "gmres",
    "-pc_type": "fieldsplit",
    "-pc_fieldsplit_type": "multiplicative",
    # Flux block
    "-fieldsplit_flux_ksp_type": "cg",
    "-fieldsplit_flux_pc_type": "ilu",
    "-fieldsplit_flux_pc_factor_levels": "2",
    # Potential block
    "-fieldsplit_potential_ksp_type": "preonly",
    "-fieldsplit_potential_pc_type": "jacobi"
}
```

## Deflation-Based Preconditioning

When you have known low-energy modes, deflate them:

```python
# Build deflation space: e.g., first few eigenmodes
E = [e1_vec, e2_vec, ...]  # modes to deflate

# Create deflation preconditioner (advanced)
# Use in custom code with PETSc API
pc.setType("deflation")
pc.setDeflationSpace(E)
```

## Convergence Diagnostics

**Check convergence with:**

```python
result = solve(
    solver_type="iterative",
    petsc_options={
        "-ksp_monitor": None,  # prints residual at each iteration
        "-ksp_view": None      # prints final solver summary
    }
)
```

**Interpretation:**
- Linear decay → well-preconditioned (ideal)
- Oscillations → saddle-point or indefinite system
- Sudden drop → deflation or restart effect
- Plateauing → preconditioner too weak, need tuning

## Common Pitfalls

| Issue | Symptom | Fix |
|---|---|---|
| No preconditioner | >1000 iterations for 100k DOFs | Add `-pc_type gamg` or `hypre` |
| Preconditioner not converging | `DIVERGED_INDEFINITE_PC` | Check matrix is SPD; try `-pc_type ilu` |
| Memory explosion with AMG | Out-of-memory on coarse grid | Reduce levels with `-pc_gamg_coarse_eq_limit 50` |
| FieldSplit not converging | `KSP did not converge` | Check block structure; try different factorization type |
| ILU fill exploding | Memory overflow | Reduce `-pc_factor_levels` (use ILU(1) or (0)) |
| Schur complement singular | Pressure solver fails | Use `-pc_fieldsplit_schur_precondition selfp` or `lsc` |

## Performance Benchmarks

Typical iteration counts for solving Ax=b (1M DOF, SPD Poisson):

| Preconditioner | Iterations | Setup (s) | Solve (s) | Total |
|---|---|---|---|---|
| None | 12,000+ | 0 | 180+ | 180+ |
| Jacobi | 4,500 | 0.1 | 65 | 65 |
| ILU(1) | 400 | 1 | 6 | 7 |
| ILU(2) | 150 | 3 | 2 | 5 |
| GAMG | 35 | 5 | 0.5 | 5.5 |
| Hypre AMG | 30 | 3 | 0.4 | 3.4 |
| Direct (LU) | 1 | 50 | 0.3 | 50 |

**Lesson:** For medium-large problems, AMG beats direct solver by >10x total time.

## See Also

- `matrix-free-solvers/SKILL.md`: Implementation details for matrix-free operators
- `solve()` tool documentation: PETSc options passthrough
- PETSc Manual: http://www.mcs.anl.gov/petsc/documentation/
