---
name: mixed-formulation
description: >
  This skill should be used when the user asks about "mixed formulation",
  "saddle point", "inf-sup", "LBB condition", "Taylor-Hood", "Stokes",
  "mixed Poisson", "Raviart-Thomas", "flux variable", "pressure-velocity",
  "incompressible", "divergence-free", "block system", or needs guidance
  on mixed finite element problems in DOLFINx.
version: 0.1.0
---

# Mixed Formulation Guide

Mixed formulations introduce multiple unknown fields (e.g., velocity + pressure) that live in different function spaces. Getting the pairing right prevents spurious oscillations and ensures convergence.

## When to Use Mixed Formulations

| Problem | Fields | Why Mixed? |
|---------|--------|-----------|
| Stokes/Navier-Stokes | velocity + pressure | Incompressibility constraint |
| Mixed Poisson | flux σ + scalar u | Conservation of flux |
| Darcy flow | velocity + pressure | Mass conservation |
| Nearly incompressible elasticity | displacement + pressure | Avoid volumetric locking |
| Biot poroelasticity | displacement + pressure | Coupled solid-fluid |

## The Inf-Sup (LBB) Condition

Mixed formulations produce saddle-point systems:

```
[A  B^T] [u]   [f]
[B  0  ] [p] = [g]
```

The element pair (V_h, Q_h) must satisfy the discrete inf-sup condition:

```
inf_{q_h ∈ Q_h} sup_{v_h ∈ V_h} (b(v_h, q_h)) / (||v_h|| ||q_h||) ≥ β > 0
```

If violated: spurious pressure modes, non-convergent solutions, or singular matrix.

## Stable Element Pairs

### Stokes / Incompressible Flow

| Pair | Velocity Space | Pressure Space | Setup |
|------|---------------|---------------|-------|
| **Taylor-Hood (P2/P1)** | Lagrange deg=2, shape=[gdim] | Lagrange deg=1 | Default recommendation |
| P2/P0 | Lagrange deg=2, shape=[gdim] | DG deg=0 | Stable, poor pressure |
| MINI (P1+bubble/P1) | Requires custom enrichment | Lagrange deg=1 | Cheaper, less accurate |

### Mixed Poisson / Darcy

| Pair | Flux Space | Scalar Space | Setup |
|------|-----------|-------------|-------|
| **RT1/DG0** | RT deg=1 | DG deg=0 | Lowest-order, conservative |
| RT2/DG1 | RT deg=2 | DG deg=1 | Better accuracy |
| BDM1/DG0 | BDM deg=1 | DG deg=0 | Higher accuracy flux |

### Unstable Pairs (DO NOT USE)

- **P1/P1 for Stokes**: Checkerboard pressure oscillations
- **Equal-order without stabilization**: Always unstable
- **RT0/DG0**: Doesn't exist — RT starts at degree 1

## DOLFINx Setup Pattern

### Step 1: Create individual spaces

```
create_function_space(name="V", family="Lagrange", degree=2, shape=[2])
create_function_space(name="Q", family="Lagrange", degree=1)
```

### Step 2: Create mixed space

```
create_mixed_space(name="W", subspaces=["V", "Q"])
```

### Step 3: Define variational form

For Stokes:
```
bilinear: "inner(grad(u), grad(v))*dx - inner(p, div(v))*dx - inner(div(u), q)*dx"
linear: "inner(f, v)*dx"
```

Where `u, v` are velocity trial/test and `p, q` are pressure trial/test. The MCP tools use `split()` to decompose the mixed trial and test functions.

### Step 4: Apply BCs

For Stokes, Dirichlet BCs go on the velocity sub-space:
```
apply_boundary_condition(value=[0.0, 0.0], boundary="...", sub_space=0)
```

The `sub_space=0` indexes the velocity component of the mixed space.

### Step 5: Solve

Direct solver is recommended for mixed problems under 100K DOFs:
```
solve(solver_type="direct")
```

For larger problems, use fieldsplit preconditioner (see solver-selection skill).

## Solver Strategies for Saddle-Point Systems

### Small problems (< 100K DOFs)

Direct solver (MUMPS). Works always, no parameter tuning needed.

### Medium problems (100K - 1M DOFs)

Block preconditioners via PETSc fieldsplit:

```python
petsc_options = {
    "ksp_type": "minres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "diag",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "jacobi",
}
```

### Large problems (> 1M DOFs)

Multigrid-based block preconditioners. Requires careful setup — reference the solver-selection skill for details.

## Post-Processing Mixed Solutions

After solving, the mixed solution contains all fields. Extract components:

- Use `split()` in UFL expressions: `split(u_h)[0]` for velocity, `split(u_h)[1]` for pressure
- For plotting: use `run_custom_code` to extract sub-functions
- For functionals: reference components directly in UFL expressions

### Useful functionals for flow problems

- **Drag/lift**: Integrate stress tensor components on obstacle boundary
- **Flow rate**: `inner(u, n)*ds(outlet_tag)` where n is the outward normal
- **Pressure drop**: `evaluate_solution` at inlet and outlet points
- **Divergence error**: `inner(div(u), div(u))*dx` — should be near zero

## Common Mistakes

1. **Wrong sub_space index for BCs**: Velocity is sub_space=0, pressure is sub_space=1 (in the order they were passed to `create_mixed_space`).

2. **Pressure needs pinning**: For enclosed flow (no pressure BC), pressure is determined only up to a constant. Pin pressure at one point or use `nullspace_mode="constant"`.

3. **Missing div-terms**: The Stokes bilinear form has three terms: viscous + pressure-velocity coupling + continuity. Missing any term breaks the formulation.

4. **Wrong sign convention**: The pressure-velocity coupling terms should have opposite signs: `-p*div(v)` and `-div(u)*q` (or `+p*div(v)` and `+div(u)*q` depending on convention).

5. **Forgetting shape parameter**: Velocity spaces need `shape=[gdim]`. Without it, you get a scalar space.

## Reference Material

For block preconditioner theory and Schur complement approximations, see `references/saddle-point-theory.md`.
