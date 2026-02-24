# Saddle-Point Theory Reference

## Block Structure

The generic saddle-point system:

```
[A   B^T] [u]   [f]
[B   -C ] [p] = [g]
```

Where:
- A: elliptic block (e.g., viscous operator)
- B: coupling operator (e.g., divergence)
- C: stabilization (often zero for standard formulations)

## Schur Complement

The Schur complement of A is:

```
S = -C - B A^{-1} B^T
```

Eliminating u:
```
S p = g - B A^{-1} f
u = A^{-1} (f - B^T p)
```

### Computational cost

Forming S explicitly is prohibitively expensive (dense matrix). Instead, precondition the block system.

## Block Preconditioners

### Block diagonal (simplest)

```
P = [A    0 ]
    [0   -S̃ ]
```

Where S̃ ≈ S is a Schur complement approximation.

For Stokes with uniform viscosity: S̃ ≈ (1/ν) M_p (pressure mass matrix).

### Block triangular (better convergence)

```
P = [A   B^T]
    [0   -S̃ ]
```

Fewer outer iterations but each application costs more.

## Schur Complement Approximations

### Stokes

| Approximation | Formula | Quality |
|--------------|---------|---------|
| Pressure mass matrix | S̃ = (1/ν) M_p | Good for uniform ν |
| BFBt | S̃ = B diag(A)^{-1} B^T | Better for variable ν |
| LSC (least-squares commutator) | More complex | Best general-purpose |

### Navier-Stokes

The convective term makes the Schur complement harder to approximate. PCD (pressure convection-diffusion) and LSC are common choices.

## Inf-Sup Constant

The discrete inf-sup constant β_h determines:
- Pressure stability: ||p - p_h|| ≤ (C/β_h) ||u - u_h||
- Preconditioner quality: condition number depends on β_h

### Known β_h values

| Pair | β_h behavior | Notes |
|------|-------------|-------|
| Taylor-Hood (P2/P1) | Mesh-independent | Well-established |
| MINI (P1+B/P1) | Mesh-independent | Smaller constant |
| P1/P0 | Mesh-dependent | Conditionally stable only |

## PETSc Fieldsplit Configuration

### Factorization types

| Type | Formula | Iterations | Cost per iteration |
|------|---------|-----------|-------------------|
| `additive` | P = diag(A, S) | More | Cheaper |
| `multiplicative` | P = triangular | Fewer | More expensive |
| `schur` | Schur complement | Depends on S̃ | Depends on S̃ |
| `symmetric_multiplicative` | Symmetric Gauss-Seidel | Enables MINRES | Moderate |

### Schur factorization subtypes

| Subtype | Description |
|---------|-------------|
| `diag` | Block diagonal with S̃ | Use with MINRES |
| `lower` | Lower triangular | Use with GMRES |
| `upper` | Upper triangular | Use with GMRES |
| `full` | Full factorization | Most expensive per iteration |

### Example: Stokes with Schur complement

```python
petsc_options = {
    "ksp_type": "minres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "diag",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "jacobi",
}
```

## Convergence Rates for Mixed Methods

### Stokes (Taylor-Hood P2/P1)

| Quantity | Rate | Norm |
|----------|------|------|
| Velocity | O(h³) | L2 |
| Velocity | O(h²) | H1 |
| Pressure | O(h²) | L2 |

### Mixed Poisson (RT1/DG0)

| Quantity | Rate | Norm |
|----------|------|------|
| Flux σ | O(h) | L2 |
| Flux div(σ) | O(h) | L2 |
| Scalar u | O(h) | L2 |
