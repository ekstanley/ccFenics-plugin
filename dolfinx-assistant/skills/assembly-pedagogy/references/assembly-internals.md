# Assembly Internals: Deep Dive

Technical reference for how DOLFINx performs cell-by-cell assembly.

## Element Stiffness Matrix Computation

For a single cell, assembly computes the local element matrix via Gaussian quadrature.

### Example: Poisson on a Triangle

Given:
- Bilinear form: `a(u,v) = ∫ ∇u · ∇v dx`
- Reference triangle: unit right triangle with vertices (0,0), (1,0), (0,1)
- P1 elements: 3 basis functions φ₀, φ₁, φ₂ (one per vertex)

Local element matrix (3×3):
```
K_local[i,j] = ∫_ref_triangle ∇φⱼ · ∇φᵢ dx
```

Gaussian quadrature with n_qp quadrature points:
```
K_local[i,j] ≈ Σ_q weight_q · (∇φⱼ|_xq) · (∇φᵢ|_xq) · det(J_q)
```

Where:
- `xq` = quadrature point q on reference element
- `weight_q` = quadrature weight
- `∇φᵢ|_xq` = gradient of basis i evaluated at xq
- `det(J_q)` = Jacobian determinant (accounts for cell size/shape)

### Quadrature Rules

Standard Gaussian quadrature on reference elements:

| Element | Degree | Points | Rule |
|---------|--------|--------|------|
| Triangle | 1 | 1 | Centroid (1/3, 1/3) |
| Triangle | 2 | 3 | Midpoints of edges |
| Triangle | 3 | 4 | Centroid + 3 edge points |
| Tetrahedron | 1 | 1 | Centroid (1/4, 1/4, 1/4) |
| Tetrahedron | 2 | 4 | Vertices of dual |

FFCx automatically selects appropriate quadrature degree to exactly integrate the form (or within tolerance).

## Reference Element Mapping

Assembly maps between:
- **Reference element** Ê (fixed, [0,1]² for triangle)
- **Physical element** K (the actual mesh cell)

### Affine Mapping (used for P1, P2, etc.)

```
x_phys = x_ref_basis[0] + J @ x_ref
```

Where J is 2×2 Jacobian matrix of derivatives of the mapping.

Gradient transform (pull-back):
```
∇_phys f = J^{-T} ∇_ref f
```

Volume transform:
```
dx_phys = |det(J)| dx_ref
```

### Non-Affine Mapping (curved elements)

For curved boundary elements or higher-order elements:
```
x_phys = Σ_i φᵢ(x_ref) · x_phys_vertex_i
```

Uses parametric coordinate transformations. More expensive but necessary for geometric accuracy near curved boundaries.

## DOF Numbering and Local-to-Global Mapping

Mesh has global DOF numbering across all cells. Assembly must map:
- Local DOFs (0, 1, 2 on a triangle) → Global DOFs (e.g., 15, 42, 103)

DOF map for cell:
```python
local_dofs = dofmap.cell_dofs(cell_index)
# Returns: [global_dof_0, global_dof_1, global_dof_2, ...]
```

When inserting local matrix into global:
```python
for i in range(num_local_dofs):
    for j in range(num_local_dofs):
        global_i = local_dofs[i]
        global_j = local_dofs[j]
        K[global_i, global_j] += K_local[i, j]
```

This "assembly loop" scatters local contributions into the global sparse matrix.

## Sparse Matrix Format: Compressed Sparse Row (CSR)

DOLFINx uses PETSc's default CSR format:

```
Matrix K (3×3 with sparsity pattern):
  [ x . x ]
  [ x x . ]
  [ . x x ]

CSR storage:
  row_offsets = [0, 2, 4, 6]     // Start index of each row
  col_indices  = [0, 2, 0, 1, 1, 2]  // Column of each nonzero
  values       = [k00, k02, k10, k11, k21, k22]  // Nonzero values
```

Access K[i,j]:
```
for idx in range(row_offsets[i], row_offsets[i+1]):
    if col_indices[idx] == j:
        return values[idx]
```

**Advantages:**
- Fast row operations (solver sweeps)
- Compact storage: only O(nnz) memory vs O(n²) for dense
- Efficient sparse-dense multiplication

## Ghost DOF Communication (Parallel)

In parallel DOLFINx:

1. **Local DOFs**: Owned by this process
2. **Ghost DOFs**: Owned by neighboring process, needed locally

Assembly on process 0:
```
Cell 5 (on process 0) includes global DOFs: [10, 11, 50]
  - DOF 10, 11 are local on process 0
  - DOF 50 is ghost (owned by process 1)

K_local computed fully. When inserting:
  - K[10, 10], K[10, 11] → inserted locally
  - K[10, 50] → queued for communication to process 1
```

After assembly on all processes: `K.assemble()` communicates ghost contributions via AllReduce.

## Form Compilation: UFL → FFCx → C Code

### UFL Expression Tree

```python
bilinear = inner(grad(u), grad(v)) * dx
```

Parsed into AST:
```
Integral(
  integrand = Inner(
    Grad(Argument(u, 0)),  # trial (numbered 0)
    Grad(Argument(v, 1))   # test (numbered 1)
  ),
  measure = dx,
  metadata = {...}
)
```

### FFCx IR Generation

FFCx transforms AST into intermediate representation (IR):
- Expand tensor operations into scalar loops
- Identify which basis functions are non-zero in each integral
- Generate quadrature code

Example IR for Poisson (simplified):
```
for qp in range(num_qp):
  J = compute_jacobian(cell, qp)
  det_J = determinant(J)
  for i in range(num_basis):
    for j in range(num_basis):
      grad_phi_i = evaluate_gradient(phi_i, qp, J)
      grad_phi_j = evaluate_gradient(phi_j, qp, J)
      A[i,j] += weight[qp] * dot(grad_phi_i, grad_phi_j) * det_J
```

### C Code Generation

FFCx emits optimized C code (10,000+ lines for complex forms):
```c
// Excerpt: Poisson form kernel
void tabulate_tensor_poisson(
    double* A, const double* w, const double* c,
    const double* coordinate_dofs,
    const int* entity_local_index, const uint8_t* quadrature_rule) {

  // Precomputed basis gradients and quadrature points
  static const double phi_grad[3][2] = { ... };
  static const double weights[3] = { ... };

  double J[4], detJ, K_inv[4];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A[3*i + j] = 0.0;
    }
  }

  // Quadrature loop
  for (int qp = 0; qp < 3; qp++) {
    compute_jacobian(J, coordinate_dofs, /* ... */);
    detJ = J[0]*J[3] - J[1]*J[2];
    invert_2x2(J, K_inv);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double grad_i[2], grad_j[2];
        transform_gradient(grad_i, K_inv, phi_grad[i]);
        transform_gradient(grad_j, K_inv, phi_grad[j]);
        A[3*i + j] += weights[qp] * dot(grad_i, grad_j) * fabs(detJ);
      }
    }
  }
}
```

This C code is compiled to a shared object (.so), then called via CFFI during assembly.

### Performance Implications

- **JIT compilation** (~1-2 seconds first call): FFCx generates and compiles C code on first solve
- **Cached**: Subsequent solves reuse compiled kernel
- **Vectorization**: C compiler applies SIMD optimizations
- **Inlining**: Small routines (basis evaluation) inlined for speed

## Assembly Cost Analysis

Total time = Kernel time + Communication time

### Kernel (Quadrature) Time

Per cell:
```
FLOP ≈ n_qp × (n_basis² × ops_per_qp + overhead)
```

For P1 Poisson (3 basis, 1 quadrature point, 2D):
```
FLOP ≈ 1 × (9 × 10 + 50) ≈ 140 FLOP per cell
```

Typical machine: 10 GigaFLOP/s → 14 nanoseconds per cell → ~10 million cells/second

### Memory Access

Assembly is **memory-bound** on modern CPUs:
- Read: coordinates, basis values, quadrature weights
- Write: sparse matrix entries (scattered)
- Cache efficiency: Good (spatial locality on cells), but scattered writes hurt

### Communication (Parallel)

After local assembly:
- `A.assemble()` → AllReduce on ghost DOF contributions
- Scales as O(n_ghost) per process
- Usually < 5% of total time for <100k DOFs per process

## Form Compilation Performance Checklist

| Aspect | Impact | Optimization |
|--------|--------|-------------|
| Form complexity | FLOP per cell | Simplify form (factor out constants) |
| Basis degree | n_qp, n_basis² growth | Lower degree if geometry permits |
| Tensor rank | loop nesting depth | Use vector spaces instead of tensors |
| Mesh size | total cells | Refine adaptively, use coarse precond |
| Quadrature order | n_qp per cell | FFCx chooses automatically, trust it |

## Boundary Integral Assembly

For `ds` integrals (boundary):

1. Loop over **facets** (edges in 2D, faces in 3D), not cells
2. Identify orientation (inward/outward normal)
3. Compute restricted basis functions (only DOFs on that facet)
4. Apply quadrature on (d-1)-dimensional reference element

Facet assembly typically 10-20% of volume assembly time (fewer facets than cells).

## Mixed-Space Assembly

For mixed spaces `(P1, P0)`:
- Basis functions include: scalar P1 (linear) + scalar P0 (constant)
- Local element matrix is block-structured:
  ```
  K_local = [ K_P1P1   K_P1P0 ]
            [ K_P0P1   K_P0P0 ]
  ```
- FFCx generates kernel for full block system
- Sparsity: diagonal coupling between blocks (inf-sup stable pairs)

## DG (Discontinuous Galerkin) Assembly

Interior penalties require `dS` integral (interior facets):

```python
bilinear = inner(grad(u_h('-')), grad(v_h('-'))) * dS  # Both sides of facet
```

Assembly:
1. Loop over interior facets
2. Get DOFs from cell on '-' side and cell on '+' side
3. Evaluate basis from both cells at facet quadrature points
4. Compute interior penalty contribution
5. Insert into K: couples '-' DOFs with '+' DOFs (off-diagonal blocks)

Sparsity: Much denser than CG (more non-local coupling).

## Edge Cases and Gotchas

### Coefficients in Forms

If form includes material property `c(x)`:
```python
a = c*inner(grad(u), grad(v))*dx
```

FFCx can:
1. Inline `c` at compile time (if constant)
2. Tabulate `c` at quadrature points (if function)
3. Evaluate `c` on-the-fly during assembly (if Expression)

Option 3 is slowest. Pre-compute `c` on function space for speed.

### Subdomain Integrals

```python
a = a1*inner(grad(u), grad(v))*dx(1) + a2*inner(grad(u), grad(v))*dx(2)
```

FFCx generates **separate kernels** for each subdomain. Assembly loops:
1. Over cells with tag=1 → call kernel_1
2. Over cells with tag=2 → call kernel_2

No performance penalty (same total FLOP), but requires cell tagging.

### Exterior Facet Integrals with BCs

```python
L = f*v*dx + g*v*ds(boundary_tag)
```

Assembly:
1. Loop cells with tag → boundary kernel
2. Flux `g` can be coefficient or function
3. Lifted automatically into RHS during solve

Must call `mark_boundaries()` first to tag facets.

## Debugging Assembly

Common issues and diagnostics:

| Problem | Check |
|---------|-------|
| Matrix singular | Sufficient BCs applied? Nullspace attached? |
| Matrix ill-conditioned | Coefficient scaling issue? Element degree too low? |
| Matrix zero | Bilinear form = 0? Check algebra, gradient signs |
| Sparse pattern wrong | Cell tagging missing? DOF numbering correct? |
| Slow assembly | Form too complex? Coefficient function inefficient? |

Use `get_solver_diagnostics()` to inspect:
- Matrix dimensions
- Nonzero count and sparsity
- Condition number estimate
- Solver iteration count

## References

- FFCx documentation: Compiler Intermediate Representation (CIR) format
- PETSc Mat format: CSR, sparsity patterns, assembly
- FEniCS Book, Chapter 3: Detailed assembly algorithm walkthrough
- Logg et al. "Automated Solution of Differential Equations by the Finite Element Method" (2012)
