# Matrix-Free and Advanced Solvers

Memory-efficient and specialized solver techniques for large-scale or structured problems.

## When to Use Matrix-Free

Use matrix-free methods when:

- **Very large systems:** Millions+ DOFs, full matrix won't fit in memory
- **Structured problems:** Operators with special structure (e.g., FFT-based, tensor products)
- **Action-only operators:** Can compute matrix-vector products but not the matrix itself
- **Iterative solver already chosen:** Iterative methods naturally support matrix-free via `matvec` callbacks
- **Sparse structure:** Pattern-based operators where explicit assembly is wasteful

**When NOT to use:** If a direct solver (LU) fits in memory, prefer it (faster, more robust).

## Shell Matrix Concept

In PETSc, a "shell matrix" defines matrix-vector products via callbacks instead of storing entries.

```python
from petsc4py import PETSc
from dolfinx import fem
import numpy as np

# Example: Laplacian operator via matrix-free form

class LaplacianOperator:
    """Shell matrix for -div(grad u)"""

    def __init__(self, form, bcs, V, mesh):
        self.form = form
        self.bcs = bcs
        self.V = V
        self.mesh = mesh

    def mult(self, mat, x, y):
        """
        Compute y = A*x (matrix-vector product).

        Args:
            mat: The shell matrix (unused, self is the context)
            x: Input vector (PETSc Vec)
            y: Output vector (PETSc Vec)
        """
        # Create temporary function for x
        u_temp = fem.Function(self.V)
        u_temp.x.array[:] = x.array

        # Assemble form: form should be bilinear in u_temp, v
        v = fem.TestFunction(self.V)
        form_assembled = fem.form(self.form(u_temp, v))

        # Assemble into vector
        y_vec = fem.assemble_vector(form_assembled)

        # Apply boundary conditions (zero for Dirichlet)
        fem.apply_lifting(y_vec, [form_assembled], [self.bcs])
        y_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.BEGIN)
        y_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.END)

        y.array[:] = y_vec.array


# Create shell matrix
mesh = ...  # your mesh
V = fem.FunctionSpace(mesh, ("Lagrange", 1))
bcs = [...]

# Define bilinear form (curried in first argument)
def bilinear(u, v):
    return fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)

operator = LaplacianOperator(bilinear, bcs, V, mesh)

# Create PETSc shell matrix
size = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
A = PETSc.Mat().createPython(size, comm=mesh.comm)
A.setPythonContext(operator)
A.setUp()

# Use in KSP solver
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("cg")
ksp.pc.setType("none")  # or "jacobi" for preconditioning

# Solve A*x = b
b = fem.assemble_vector(fem.form(f * v * ufl.dx))
x = A.createVecRight()
ksp.solve(b, x)

print(f"Converged: {ksp.converged}, iterations: {ksp.getIterationNumber()}")
```

**Key points:**
- `mult` receives input vector `x`, must fill output vector `y`
- `ghostUpdate` ensures communication across MPI boundaries
- No storage of matrix entries
- **Preconditioning critical:** Without good preconditioner, convergence is poor

## Near-Nullspace for Singular Systems

For pure Neumann problems (no Dirichlet BCs), attach a nullspace to the system matrix.

The `solve()` tool supports `nullspace_mode`:

```python
# Using MCP solve() tool with nullspace_mode

# For Poisson with pure Neumann:
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    nullspace_mode="constant"  # attaches constant mode
)

# For elasticity with pure Neumann:
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",
    nullspace_mode="rigid_body"  # attaches rigid body modes
)
```

**For custom nullspace** (via `run_custom_code`):

```python
from dolfinx import fem
from petsc4py import PETSc
import numpy as np

# Create nullspace vectors
# Example: constant mode for scalar problem
nullspace_vecs = []

# Constant vector
u_const = fem.Function(V)
u_const.x.array[:] = 1.0
u_const.x.scatter_forward()
nullspace_vecs.append(u_const.x.petsc_vec)

# For elasticity, add rigid body modes (translations + rotations)
# u_rigid_x, u_rigid_y, u_rigid_z, omega_x, omega_y, omega_z

# Create nullspace
nullspace = PETSc.NullSpace().create(comm=mesh.comm, vectors=nullspace_vecs)

# Attach to matrix
A = fem.assemble_matrix(...)
nullspace.remove(A)  # orthogonalize matrix against nullspace

# Attach nullspace to matrix
A.setNullSpace(nullspace)

# KSP solver automatically removes nullspace component from RHS
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.solve(b, x)
```

## FieldSplit Preconditioner (Block Systems)

For coupled multi-physics (e.g., Stokes velocity-pressure), use block preconditioning.

**Example: Stokes flow**

```
[A  B^T] [u]   [f_u]
[B  0  ] [p] = [f_p]

where A is velocity Laplacian, B is divergence operator.
```

```python
from dolfinx import fem
import ufl

# Define mixed space
P1 = fem.FunctionSpace(mesh, ("Lagrange", 2))  # velocity
P0 = fem.FunctionSpace(mesh, ("Lagrange", 1))  # pressure
W = fem.FunctionSpace(mesh, ("Lagrange", 2) * 2 + ("Lagrange", 1))  # mixed

# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# Stokes system
nu = 1.0  # viscosity
a = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    - p * ufl.div(v) * ufl.dx \
    - q * ufl.div(u) * ufl.dx

L = ufl.inner(f, v) * ufl.dx

# Assemble system
A = fem.assemble_matrix(fem.form(a), bcs=bcs)
b = fem.assemble_vector(fem.form(L))

# Define index sets for blocks
# u: DOFs 0 to N_u-1
# p: DOFs N_u to N_u+N_p-1

# Extract subspace DOF information
V_dofs = W.sub(0).dofmap.index_map.size_global  # velocity DOFs
P_dofs = W.sub(1).dofmap.index_map.size_global  # pressure DOFs

IS_u = PETSc.IS().createStride(comm=mesh.comm, size=V_dofs, first=0, step=1)
IS_p = PETSc.IS().createStride(comm=mesh.comm, size=P_dofs, first=V_dofs, step=1)

# Configure FieldSplit preconditioner
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("gmres")

pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitIS(
    ("u", IS_u),
    ("p", IS_p)
)

# Set type of fieldsplit: additive, multiplicative, etc.
pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

# Configure sub-KSPs for each block
ksp_u, pc_u = pc.getFieldSplitSubKSP()[0]
ksp_u.setType("cg")
pc_u.setType("hypre")

ksp_p, pc_p = pc.getFieldSplitSubKSP()[1]
ksp_p.setType("minres")
pc_p.setType("none")

# Solve
x = A.createVecRight()
ksp.solve(b, x)

print(f"Converged: {ksp.converged}, iterations: {ksp.getIterationNumber()}")
```

## Schur Complement Preconditioner

For saddle-point problems, use a Schur complement approach to reduce the coupled system.

**Idea:** Given block system:
```
[A  B^T] [u]   [f]
[B  0  ] [p] = [g]
```

Eliminate u to get: `(B*A^{-1}*B^T)*p = g - B*A^{-1}*f` (Schur complement system).

```python
# Advanced configuration (PETSc options)
pc.setFieldSplitSchurPreType(
    PETSc.PC.SchurComplementType.LOWER,  # use lower block system
    PETSc.PC.SchurComplementType.SELFP   # pressure mass matrix preconditioner
)

# Or configure explicitly
pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.UPPER)

# Estimate of A^{-1}: use approximation (e.g., lumped mass)
# This is problem-specific; consult PETSc documentation
```

## Algebraic Multigrid (AMG) — GAMG and Hypre

Most effective for large elliptic systems.

**GAMG (Geometric-Algebraic Multigrid):**

```python
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("cg")

pc = ksp.getPC()
pc.setType("gamg")

# Configure GAMG
pc.setGAMGType("agg")  # aggregation-based
pc.setGAMGLevels(3)    # target 3 levels
pc.setGAMGThreshold(0.05)  # threshold for strong coupling

# Smoother: symmetric Gauss-Seidel
pc.setGAMGSmootherType("jacobi")

ksp.setFromOptions()
```

**Hypre BoomerAMG:**

```python
ksp.setType("cg")
pc.setType("hypre")
pc.setHYPREType("boomeramg")

# Fine-tune BoomerAMG
pc.hypre_set_options(
    -hypre_boomeramg_strong_threshold 0.25
    -hypre_boomeramg_coarsen_type PMIS
    -hypre_boomeramg_num_sweeps 2
)
```

**Use when:**
- Poisson / diffusion with millions of DOFs
- Need automatic setup (no tuning needed)
- Memory is plentiful (AMG uses extra levels)

**Typical speedup:** 10-100x vs unpreconditioned CG.

## Geometric Multigrid

Use mesh hierarchy (coarse, medium, fine) with explicit prolongation/restriction.

```python
# Create mesh hierarchy
mesh_coarse = ... # coarser mesh
mesh_fine = refine_mesh(mesh_coarse)
mesh_finer = refine_mesh(mesh_fine)

# Define spaces on each level
V_coarse = fem.FunctionSpace(mesh_coarse, ("Lagrange", 1))
V_fine = fem.FunctionSpace(mesh_fine, ("Lagrange", 1))
V_finer = fem.FunctionSpace(mesh_finer, ("Lagrange", 1))

# Assemble system on finest mesh
A_finer = fem.assemble_matrix(...)
b_finer = fem.assemble_vector(...)

# Compute prolongation (interpolation from coarse to fine)
# In DOLFINx, interpolate via FunctionSpace.interpolate_constant

# Use geometric MG via external library (e.g., PyAMG)
# or implement custom V-cycle
```

For **automatic mesh refinement**, use the DOLFINx MCP tool `refine_mesh`:

```python
# Via MCP: call refine_mesh multiple times
# Then assemble systems at each level
# then use solver.setMGLevels(...)
```

## Preconditioning Strategies Comparison

| Preconditioner | Problem Type | Setup Cost | Memory | Robustness |
|---|---|---|---|---|
| **Jacobi** | Any | O(N) | O(N) | Low (diagonal-only) |
| **ILU(k)** | General | O(N^{3/2}) | O(N) | Good |
| **AMG (GAMG)** | Elliptic | O(N) | O(N log N) | Excellent |
| **Hypre AMG** | Elliptic | O(N) | O(N log N) | Excellent |
| **FieldSplit** | Block systems | O(blocks) | Problem-dep | Good (if blocks decoupled) |
| **Multigrid (GMG)** | Smooth systems | O(N) | O(N) | Excellent (if hierarchy good) |
| **None (CG only)** | Very smooth | None | O(1) | Poor (only if well-conditioned) |

## Integration with DOLFINx MCP Tools

### Using `solve()` with Matrix-Free Preconditioning

```python
result = solve(
    solver_type="iterative",
    ksp_type="cg",
    pc_type="hypre",  # or "gamg"
    petsc_options={
        "-pc_hypre_type": "boomeramg",
        "-pc_hypre_boomeramg_strong_threshold": "0.25"
    }
)
```

### Using `run_custom_code` for Custom Shell Matrices

```python
code = """
# Define shell matrix operator class
class MyOperator:
    def mult(self, mat, x, y):
        # y = A*x
        ...

# Assemble system and solve
"""

run_custom_code(code=code)
```

## Performance Optimization Checklist

- [ ] Preconditioner chosen based on problem type (elliptic → AMG, general → ILU, block → FieldSplit)
- [ ] Convergence tolerance reasonable (rtol=1e-6, atol=1e-12 typical)
- [ ] Initial guess good (set from previous solution if available)
- [ ] Matrix sparsity exploited (don't assemble dense matrices)
- [ ] Near-nullspace attached if needed (pure Neumann)
- [ ] PETSc compiled with optimizations (-O3, -march=native)
- [ ] Profiling done (-log_view) to identify bottlenecks
- [ ] Krylov method appropriate (CG for SPD, GMRES for general, MINRES for saddle-point)

## See Also

- `preconditioner-guide.md`: Detailed preconditioner selection table
- `solve()` tool: Built-in solver with PETSc options support
- `parallel-mpi-awareness/SKILL.md`: Parallel considerations for large systems
