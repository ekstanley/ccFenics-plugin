# Assembly Pedagogy: From Weak Forms to Linear Algebra

This skill explains how DOLFINx converts your mathematical weak form into the discrete linear system that solvers actually compute.

## What is Assembly?

Assembly transforms a continuous weak form into discrete linear algebra:

**Weak form (mathematics):**
Find u ∈ V such that: `a(u, v) = L(v)` for all v ∈ V̂

**Discrete system (what the solver gets):**
`K u = F`

Where:
- `K` is the stiffness matrix (from bilinear form `a`)
- `F` is the load vector (from linear form `L`)
- `u` is the nodal solution vector

## The Assembly Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. You write UFL expressions                                │
│    bilinear = inner(grad(u), grad(v)) * dx                 │
│    linear = f * v * dx                                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 2. UFL symbolic processing                                  │
│    - Validate form rank (bilinear needs trial + test)       │
│    - Determine element families and degree                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 3. FFCx form compilation                                    │
│    - Generate C kernel code                                 │
│    - Create quadrature points and weights                   │
│    - Build reference element mappings                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 4. Cell-by-cell assembly loop                               │
│    for each cell in mesh:                                   │
│      1. Evaluate C kernel at quadrature points              │
│      2. Compute local element matrix/vector                 │
│      3. Map local DOFs → global DOFs                        │
│      4. Insert into global matrix K and vector F            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 5. Boundary condition handling (lifting)                    │
│    - Modify RHS to account for known BC values              │
│    - Constraint DOFs are eliminated from solve              │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 6. Solver receives Ku = F                                   │
│    - Solves for unconstrained DOFs                          │
│    - Known BC values already in solution                    │
└─────────────────────────────────────────────────────────────┘
```

## The Bilinear Form → Stiffness Matrix

For Poisson problem: `a(u, v) = ∫ ∇u · ∇v dx`

Assembly computes:
```
K[i,j] = ∫_Ω (∇φ_j · ∇φ_i) dx
```

Where φ_i, φ_j are basis functions (usually linear or quadratic polynomials on each cell).

In practice:
1. Divide integral into cell integrals: `∫_Ω = Σ_cells ∫_cell`
2. On reference element, use Gaussian quadrature: `∫ f dx ≈ Σ_q w_q f(x_q)`
3. Map quadrature points from reference cell to physical cell using Jacobian
4. Accumulate local contributions into global matrix

**Sparsity**: K[i,j] is nonzero only if basis functions φ_i and φ_j overlap (non-disjoint support).
This creates a sparse matrix — only ~10-100 nonzeros per row for typical FEM problems.

## The Linear Form → Load Vector

For Poisson with source: `L(v) = ∫ f v dx`

Assembly computes:
```
F[i] = ∫_Ω f(x) φ_i(x) dx
```

Same cell-by-cell integration process.

**Boundary integrals** use `ds` or `dS` (not `dx`):
- `a_robin(u, v) = ∫_boundary (c*u*v) ds`
- Integrated over boundary facets instead of cells
- Often appears in Robin BC: `∫_∂Ω α·u·v ds`

## Boundary Condition Application: Lifting

When you apply a Dirichlet BC (e.g., u = g on boundary):

1. **Identify constrained DOFs**: Which DOFs lie on the boundary where u = g?
2. **Compute "lifting"**: A function u_D that satisfies u_D = g on boundary
3. **Modify RHS**: F_new = F - K·u_D (moves known values to RHS)
4. **Solve reduced system**: K·u_interior = F_new for interior DOFs only
5. **Combine**: Final solution is u_interior + u_D

**Why lifting?** Preserves matrix symmetry. K remains symmetric even with BCs.

Example with MCP:
```python
# Apply Dirichlet BC: u = 0 on left boundary (x=0)
apply_boundary_condition(
    value=0,
    boundary="np.isclose(x[0], 0.0)",
    function_space="V"
)
# DOLFINx automatically lifts during solve()
```

## Assembly Modes: The `assemble` Tool

DOLFINx exposes three assembly modes via the `assemble` MCP tool:

### Scalar Assembly
```python
assemble(target="scalar", form="u_h*u_h*dx")
# Returns: float (the integral ∫ u_h² dx)
# Typical uses: compute norms, energy functionals, error measures
```

Example: L² norm
```python
assemble(target="scalar", form="inner(u_h, u_h)*dx")
# Returns ∫_Ω u_h² dx = ||u||_L2²
```

### Vector Assembly
```python
assemble(target="vector", form="f*v*dx")
# Returns: dict with "norm": float, "size": int
# The actual vector is assembled internally and its L2 norm returned
```

Example: Load vector from source term
```python
set_material_properties(name="f", value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])")
assemble(target="vector", form="f*v*dx", apply_bcs=True)
# Returns assembled load vector F, with BC modifications applied
```

### Matrix Assembly
```python
assemble(target="matrix", form="inner(grad(u), grad(v))*dx")
# Returns: dict with "dims": [int, int], "nnz": int
# The stiffness matrix K is assembled
```

Example: Poisson stiffness
```python
assemble(
    target="matrix",
    form="inner(grad(u), grad(v))*dx",
    name="poisson_matrix",
    apply_bcs=True
)
# Returns matrix dimensions and sparsity info
# With apply_bcs=True, boundary rows are zeroed and diagonal set to 1
```

## Form Validation: What Assembly Expects

Before assembly succeeds, your forms must satisfy:

| Check | Passes When | Error If |
|-------|------------|----------|
| Form rank | Bilinear has 2 function arguments (trial + test) | Linear form has 2 functions, or bilinear has 1 |
| Measure | Form includes `dx`, `ds`, or `dS` | Form is just a UFL expression, no measure |
| Trial/test spaces | Both from same function space family | Test space incompatible with trial (e.g., DG vs CG) |
| Tensor rank | `inner()` both args have same rank | `inner(u, grad(v))` where u is scalar, grad(v) is vector |
| Vector/scalar | Multiplication well-defined | Trying to add vector + scalar |

Common errors during assembly:

```python
# ERROR: Missing test function
bilinear = "u*u*dx"  # Just u, no v
define_variational_form(bilinear=bilinear, linear="f*v*dx")
# → Form rank error: need trial and test

# ERROR: Rank mismatch
bilinear = "inner(grad(u), grad(v)) + u*v*dx"
# All terms in bilinear form must have same rank (both quadratic in u,v)
# This works; inner(grad(u), grad(v)) is bilinear, u*v is also bilinear

# ERROR: Form measure
bilinear = "inner(grad(u), grad(v))"  # No dx!
define_variational_form(bilinear=bilinear, linear="f*v*dx")
# → Assembly error: form has no measure (dx, ds, dS)
```

## Sparsity and Performance

The sparsity pattern of K is determined by:
1. **Mesh connectivity**: Which cells are neighbors?
2. **Element DOF map**: How many DOFs per cell, which cells share DOFs?
3. **Form structure**: Which DOFs interact in the form?

For P1 Lagrange (linear elements on triangles):
- 1 DOF per vertex
- Each cell has 3 DOFs (vertices)
- K[i,j] nonzero only if vertices i, j share an edge in the mesh

Typical sparsity:
- 2D triangle mesh: ~9 nnz per row (dense neighborhoods)
- 3D tetrahedral mesh: ~27 nnz per row

This sparsity is what makes assembled systems tractable for large problems.

## Dirichlet vs Neumann vs Robin

### Dirichlet (essential BC)
"u = value on boundary"
- Enforced during assembly via lifting
- Eliminates DOFs from the system
- Example: `apply_boundary_condition(value=0, boundary="x[0] < 1e-14")`

### Neumann (natural BC)
"∇u·n = flux on boundary"
- No code needed; naturally included in weak form if you write `ds` integral
- Example: `L = f*v*dx + g*v*ds(tag=1)` (flux g on tagged boundary)

### Robin (mixed BC)
"α·u + β·∇u·n = γ on boundary"
- Applied as Neumann data in the weak form
- Example: `L = f*v*dx + (c*u_D - flux)*v*ds(tag=1)`

## Bilinear vs. Linear: Clear Distinction

| Aspect | Bilinear a(u,v) | Linear L(v) |
|--------|-----------------|-----------|
| Arguments | 2: trial u, test v | 1: test v only |
| Produces | Stiffness matrix K | Load vector F |
| MCP call | `define_variational_form(bilinear=...)` | `define_variational_form(linear=...)` |
| Example | `inner(grad(u), grad(v))*dx` | `f*v*dx` |
| Rank | Quadratic in (u,v) | Linear in v |
| DOF coupling | K[i,j] couples DOFs i, j | F[i] depends on only DOF i region |

## Assembly Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Form rank mismatch" | Bilinear has ≠2 functions or wrong ranks | Ensure bilinear has exactly trial u and test v |
| "Measure required" | Form is bare UFL, no dx/ds/dS | Add `*dx` to form expression |
| "Element mismatch" | Mixing incompatible elements in bilinear | Both u and v from same space (or compatible mixed spaces) |
| "Shape mismatch" | inner() on incompatible ranks | Both arguments same rank: scalar·scalar, vector·vector, tensor·tensor |
| "Function not found" | Reference undefined function in form | Create and register function via `create_function`, `interpolate`, or `project` |

## Discrete Differential Operators

Beyond the bilinear and linear forms, DOLFINx can assemble discrete differential operators — matrices that represent gradient, curl, or interpolation between spaces.

### Creating Operators

```python
create_discrete_operator(
    operator_type="gradient",      # "gradient", "curl", or "interpolation"
    source_space="V_scalar",       # Source function space (e.g., CG1)
    target_space="V_vector",       # Target space (e.g., Nedelec or vector DG)
    name="grad_operator"           # Optional: name the operator
)
```

### Operator Types

| Type | Source Space | Target Space | Meaning |
|------|-------------|--------------|---------|
| `"gradient"` | Scalar (H1) | Vector (L2 or H(div)) | K[i,j] = ∫ ∇φ_i·ψ_j dx |
| `"curl"` | Vector (H(curl)) | Vector (L2) | K[i,j] = ∫ (∇×ψ_i)·ψ_j dx |
| `"interpolation"` | Any source | Any target | L2 projection operator |

### Use Cases

**Auxiliary space preconditioning**: Build a coarse preconditioner using gradient operators.

**Transfer operators for multigrid**: Discretize restriction/prolongation between nested spaces.

**Flux extraction**: Apply gradient operator to scalar solution to get vector field.

### Example: Gradient Operator

```python
# Poisson with scalar solution u
create_function_space(name="V", family="Lagrange", degree=1)
create_function_space(name="grad_V", family="DG", degree=0, shape=[2])

# Create gradient matrix
create_discrete_operator(
    operator_type="gradient",
    source_space="V",
    target_space="grad_V"
)
# Result: matrix that maps scalar DOFs → vector DOFs
# Can be used in custom preconditioner or as part of larger system
```

Returns: matrix size, number of non-zeros (nnz), and operator type.

## Practical Workflow

When you call `solve()`:
```python
# User code
define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx + c*u*v*ds(1)",
    linear="f*v*dx + g*v*ds(2)"
)
apply_boundary_condition(value=0, boundary_tag=1)
solve(solver_type="direct")
```

Internally:
1. Assemble bilinear form → K matrix
2. Assemble linear form → F vector
3. Apply BCs via lifting → modify F, constrain DOF rows of K
4. Solve Ku = F → u_h
5. Return solution function u_h

The `solve()` tool hides assembly details, but understanding this process helps you:
- Debug form errors quickly
- Choose efficient element/BC combinations
- Predict solver performance (matrix size, sparsity, condition number)
- Optimize custom Newton loops

## Visualization: Matrix Spy Plot

For a simple 2×2 unit square with triangular mesh (4 triangles, 6 vertices):
```
Assembled K for Poisson:
┌                  ┐
│ x . . x . .      │  Node 0 couples with nodes 0, 3
│ . x . x . .      │  Node 1 couples with nodes 1, 3
│ . . x x . .      │  Node 2 couples with nodes 2, 3
│ x x x x x x      │  Node 3 (center) couples with all nodes
│ . . . x x .      │  Node 4 couples with nodes 3, 4
│ . . . x . x      │  Node 5 couples with nodes 3, 5
└                  ┘
```

This sparsity is created automatically by assembly—you don't need to compute it yourself.
