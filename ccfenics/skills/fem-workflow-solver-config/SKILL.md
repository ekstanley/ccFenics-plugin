---
name: fem-workflow-solver-config
description: |
  Comprehensive guide to PETSc solver configuration and JIT optimization in DOLFINx.
  Use when the user asks about PETSc options, solver configuration, preconditioner tuning,
  KSP settings, JIT compilation, FFCx options, or solver performance.
---

# Solver Configuration Workflow (Tutorial Ch4.2 + Ch4.3)

Configure PETSc solvers and FFCx JIT compilation for optimal performance.

## PETSc Solver Options via `petsc_options`

The `solve` tool accepts a `petsc_options` dict for fine-grained control.

### Direct Solvers

```
solve(
    solver_type="direct",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    },
    solution_name="u_h"
)
```

Available direct solvers:
| Solver | `pc_factor_mat_solver_type` | Notes |
|---|---|---|
| MUMPS | `"mumps"` | Parallel, default for MPI |
| SuperLU_dist | `"superlu_dist"` | Alternative parallel |
| PETSc built-in | (omit option) | Sequential only |

### Iterative: CG + Algebraic Multigrid

For SPD systems (Poisson, elasticity):
```
solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    petsc_options={
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_max_iter": 1,
        "pc_hypre_boomeramg_strong_threshold": 0.7,
        "ksp_rtol": 1e-8,
        "ksp_monitor": ""
    },
    solution_name="u_h"
)
```

### Iterative: GMRES + ILU

For non-symmetric systems:
```
solve(
    solver_type="iterative",
    ksp_type="gmres",
    pc_type="ilu",
    petsc_options={
        "ksp_gmres_restart": 100,
        "pc_factor_levels": 2,
        "ksp_rtol": 1e-8
    },
    solution_name="u_h"
)
```

### Fieldsplit for Mixed Problems

For block-structured systems (Stokes, mixed Poisson):
```
solve(
    solver_type="iterative",
    ksp_type="minres",
    pc_type="fieldsplit",
    petsc_options={
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "none"
    },
    solution_name="u_h"
)
```

### Monitoring Convergence

Add monitoring options to see convergence history:
```
petsc_options={
    "ksp_monitor": "",           # Print residual each iteration
    "ksp_converged_reason": "",  # Print reason for convergence/divergence
    "ksp_view": ""               # Print solver configuration
}
```

## Solver Selection Guide

| Problem Type | Recommended KSP | Recommended PC | Notes |
|---|---|---|---|
| SPD (Poisson) | `cg` | `hypre` (AMG) | Optimal O(n) |
| SPD (small) | `preonly` | `lu` (MUMPS) | Robust, not scalable |
| Non-symmetric | `gmres` | `ilu` | Restart at 100+ |
| Saddle-point | `minres` | `fieldsplit` | Block preconditioner |
| Indefinite | `gmres` | `lu` | Direct solver safest |
| Complex-valued | `gmres` | `lu` | CG requires Hermitian |

## JIT Compilation Options

DOLFINx uses FFCx to JIT-compile variational forms. Optimization flags:

```python
run_custom_code(code="""
from dolfinx import fem
import cffi

# Set FFCx options for optimized compilation
jit_options = {
    "cffi_extra_compile_args": ["-O2", "-march=native"],
    "cffi_libraries": ["m"],
}

# Apply when compiling forms
a_compiled = fem.form(a_form, jit_options=jit_options)
L_compiled = fem.form(L_form, jit_options=jit_options)
""")
```

## Diagnostics

After solving, use `get_solver_diagnostics` to inspect solver performance:

```
get_solver_diagnostics()
```

Returns: iterations, residual norm, wall time, convergence reason, DOF count.

## Feasibility

100% achievable with existing tools. The `petsc_options` parameter on `solve` provides full PETSc configuration access.
