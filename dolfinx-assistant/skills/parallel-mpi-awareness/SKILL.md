# Parallel MPI Awareness

Essential concepts for running DOLFINx simulations on distributed-memory systems.

## Key Principle

**DOLFINx is always MPI-aware.** Even serial code runs on `MPI_COMM_WORLD` with 1 rank. To run in parallel:

```bash
mpirun -np 4 python3 script.py
```

The same Python script automatically distributes computation across 4 ranks.

## Mesh Partitioning

DOLFINx automatically partitions meshes across ranks using ParMETIS, KaHIP, or SCOTCH (compile-time selection).

```python
from dolfinx import mesh

# Automatic partitioning: each rank gets a local partition
mesh = mesh.create_unit_square(
    MPI.COMM_WORLD,  # collective call: all ranks execute
    nx=100, ny=100,
    cell_type=mesh.CellType.triangle
)

# mesh.topology.original_cell_index: maps local to global
# mesh.geometry.x: local coordinates
print(f"Rank {MPI.COMM_WORLD.rank}: {mesh.num_entities(mesh.topology.dim)} cells")
# Each rank prints different count (its local cells)
```

**Each rank stores only its partition + ghost layers.**

## Ghost Modes

Control how much of neighboring partitions are stored locally (communication overhead).

### GhostMode.none
- **No ghost cells** stored
- **Fastest local operations** (smallest local mesh)
- **Limited:** Can't assemble DG forms, can't do local neighbor queries
- **When to use:** Never (rarely sufficient)

### GhostMode.shared_facet (Default)
- **Ghost cells sharing facets** with local cells
- **Needed for:** DG forms, edge-based assembly, facet integrals
- **Trade-off:** Slightly more storage/communication, but essential for correctness
- **Always use this unless you have specific reason not to**

```python
# Explicitly set ghost mode (usually default)
from dolfinx.mesh import GhostMode

mesh = mesh.create_unit_square(
    MPI.COMM_WORLD,
    nx=100, ny=100,
    ghost_mode=GhostMode.shared_facet  # explicit, usually default
)
```

## Rank-0 Pattern (Serial I/O)

Only rank 0 should print and write files (avoid duplicated output):

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

# Print only from rank 0
if rank == 0:
    print("Starting simulation...")

# Assembly happens on all ranks
b = fem.assemble_vector(fem.form(L))

# Print solver info only from rank 0
if rank == 0:
    print(f"System assembled, RHS norm = {b.norm()}")
```

## Collective Operations

All ranks must call collective operations in the same order.

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank

# CORRECT: all ranks execute create and solve
u = fem.Function(V)
A = fem.assemble_matrix(...)
b = fem.assemble_vector(...)
ksp.solve(b, u.x.petsc_vec)  # collective within KSP

# WRONG: only rank 0 creates function
if rank == 0:
    u = fem.Function(V)  # DEADLOCK: other ranks waiting

# WRONG: ranks scatter without barrier
if rank == 0:
    data = [1, 2, 3, 4]
else:
    data = None
result = comm.scatter(data)  # DEADLOCK if not in same collective call
```

**Rule:** Mesh creation, function space creation, assembly, and solve must be called by ALL ranks (even if local computation differs).

## Ghost Exchange

PETSc vectors track ghosts. After assembly, call `ghostUpdate`:

```python
# Assemble vector
b = fem.assemble_vector(fem.form(L))

# Ghost values are stale; update them (communicate with neighbors)
b.ghostUpdate(
    addv=PETSc.InsertMode.ADD,     # sum contributions from ghost ranks
    mode=PETSc.ScatterMode.BEGIN   # non-blocking send/receive
)
b.ghostUpdate(
    addv=PETSc.InsertMode.ADD,
    mode=PETSc.ScatterMode.END     # wait for completion
)
```

**Two-phase pattern:** BEGIN / END allows computation/communication overlap.

```python
# Optimized: overlap communication with computation
for b in [b1, b2, b3]:
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)

# Do some computation while sends/receives in flight
# ...

for b in [b1, b2, b3]:
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)
```

## Distributed Function Evaluation

Point evaluation may fail if point is not on current rank.

```python
from dolfinx.fem import evaluate_function_at_points

u = fem.Function(V)
points = np.array([[0.5, 0.5], [0.3, 0.3], ...]).T  # shape (gdim, num_points)

# May return NaN if point not in local mesh
values = evaluate_function_at_points(u, points)

# For robustness, gather all points to rank 0, evaluate, scatter back
all_points = comm.allgather(points)  # all ranks have all points
all_values = [evaluate_function_at_points(u, p) for p in all_points]
# ... postprocess
```

## Parallel I/O

### XDMF (Recommended for Parallel)

```python
from dolfinx.io import XDMFFile

# Parallel write to single file
with XDMFFile(comm, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)

# At each timestep
with XDMFFile(comm, "solution.xdmf", "a") as xdmf:
    xdmf.write_function(u, t)

# Read on same or different rank count
with XDMFFile(comm, "solution.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    u = xdmf.read_function(V, "u")
```

### VTK (Partition-based)

```python
from dolfinx.io import VTKFile

# Each rank writes its partition; Paraview assembles on read
with VTKFile(comm, "solution.pvd", "w") as vtk:
    vtk.write_mesh(mesh)

with VTKFile(comm, "solution.pvd", "a") as vtk:
    vtk.write_function(u, t)
```

XDMF is preferred for parallel I/O (single unified file).

## Communication Patterns

### AllReduce (Norm, Integral)

```python
from dolfinx.fem import assemble_scalar, form

# Local integral (each rank computes on its partition)
local_integral = assemble_scalar(form(u**2 * ufl.dx))

# Global integral (sum over all ranks)
global_integral = comm.allreduce(local_integral, op=MPI.SUM)

# Norm (already collective in PETSc)
norm = b.norm()  # automatically allreduce
```

### Scatter / Gather

```python
# Rank 0 has data, scatter to all
if rank == 0:
    all_data = [1, 2, 3, 4, 5, 6, 7, 8]
else:
    all_data = None

local_data = comm.scatter(all_data, root=0)
print(f"Rank {rank}: {local_data}")  # rank 0: [1,2], rank 1: [3,4], ...

# Gather all data back to rank 0
gathered = comm.gather(local_data, root=0)
if rank == 0:
    print(f"All data: {gathered}")
```

## Scaling Considerations

### Mesh Size Per Rank

- **Small mesh (<10k cells per rank):** Communication overhead dominates (poor scaling)
- **Medium mesh (100k cells per rank):** Good scaling (typical sweet spot)
- **Large mesh (>1M cells per rank):** Excellent scaling if preconditioner is scalable

**Rule of thumb:** Minimum 100k DOFs per rank for good parallel efficiency.

### Communication Overhead

```
Total time = Computation + Communication

For fixed global problem size:
- More ranks → less computation per rank
- More ranks → more communication (fewer elements per boundary)
- Optimal rank count ≈ where computation ≈ communication
```

For **weak scaling** (problem size ∝ rank count):
- Computation per rank stays constant
- Communication grows (surface-to-volume ratio)
- Target: >70% of peak speed on N ranks

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Only rank 0 creates function | Deadlock / segfault | Call fem.Function on all ranks |
| Print from all ranks | Spam output (duplicates) | Use `if rank == 0: print(...)` |
| Rank-specific assembly | Inconsistent matrix | Same forms on all ranks |
| Missing ghostUpdate | Wrong results | Call `b.ghostUpdate(...)` after assembly |
| Point not in local mesh | Returned NaN | Use `comm.allgather` + robustness |
| BCs applied differently | Solver divergence | Apply same BCs on all ranks |
| File I/O from rank 0 only in loop | File gets truncated | Coordinate rank 0 writes or use XDMF |
| Calling MPI_Abort from rank 0 only | Other ranks hang | Call collective operations or use allreduce + check |

## Profiling Parallel Code

```bash
# Run with profiling
mpirun -np 4 python3 script.py -log_view

# Output shows:
# - Time in each operation (assembly, solve, etc.)
# - Load imbalance (if one rank much slower)
# - Communication volume
```

## Example: Parallel Poisson Solve

```python
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import numpy as np
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Create mesh (collective)
mesh_obj = mesh.create_unit_square(comm, nx=100, ny=100)

# Create function space (collective)
V = fem.FunctionSpace(mesh_obj, ("Lagrange", 1))

# Define problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# RHS (parametric)
f = fem.Constant(mesh_obj, PETSc.ScalarType(1.0))

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Boundary conditions
bcs = [fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_geometrical(
    V, lambda x: np.logical_or(x[0] == 0, x[1] == 0)), V)]

# Solve (collective)
problem = fem.petsc.LinearProblem(
    a, L, bcs=bcs,
    petsc_options={
        "-ksp_type": "cg",
        "-pc_type": "hypre",
        "-pc_hypre_type": "boomeramg",
        "-ksp_monitor": None
    }
)

uh = problem.solve()

# Compute norm (collective within PETSc)
norm = uh.x.petsc_vec.norm()

# Print from rank 0 only
if rank == 0:
    print(f"Solution norm: {norm:.6e}")

# Write output (collective XDMF)
from dolfinx.io import XDMFFile
with XDMFFile(comm, "poisson_parallel.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_obj)
    xdmf.write_function(uh)
```

**Run as:**
```bash
mpirun -np 4 python3 parallel_poisson.py
```

## Hybrid OpenMP+MPI

DOLFINx supports OpenMP within each MPI rank:

```bash
# 4 MPI ranks, each with 4 threads
mpirun -np 4 OMP_NUM_THREADS=4 python3 script.py
```

**When to use:**
- Very large shared-memory nodes (>16 cores)
- Unbalanced load distribution (OpenMP load balancing helps)
- Typically: 2-4 MPI ranks per socket, rest OpenMP

## Checkpoint / Restart

Save solution and state for restart:

```python
from dolfinx.io import XDMFFile

# Checkpoint at timestep
if timestep % 10 == 0:
    with XDMFFile(comm, f"checkpoint_t{timestep}.xdmf", "w") as xdmf:
        xdmf.write_function(u)
        xdmf.write_function(u_prev)

# Restart from checkpoint
with XDMFFile(comm, "checkpoint_t100.xdmf", "r") as xdmf:
    u = xdmf.read_function(V, "u")
    u_prev = xdmf.read_function(V, "u_prev")
```

**Best practice:** Store mesh once, append functions at each step (XDMF format).

## Note: MCP Server Single-Rank

The DOLFINx MCP server runs single-rank. To execute parallel code:

1. **Generate script via `run_custom_code`**
2. **Run script externally with `mpirun`**

```python
# In MCP skill/command:
code = """
# Generate parallel script
script = '''
from mpi4py import MPI
from dolfinx import mesh, fem
...
if __name__ == "__main__":
    main()
'''

with open('parallel_solve.py', 'w') as f:
    f.write(script)
"""

run_custom_code(code=code)
# Then user runs: mpirun -np 4 python3 parallel_solve.py
```

## See Also

- `mpi-patterns.md`: Reusable parallel code patterns
- `setup-mpi` command: Interactive parallel configuration
- `parallel-mpi-awareness/SKILL.md`: Full parallel tutorial
