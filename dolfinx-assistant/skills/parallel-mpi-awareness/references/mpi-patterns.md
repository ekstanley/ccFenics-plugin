# MPI Pattern Reference

Reusable parallel code patterns for DOLFINx + MPI.

## Pattern 1: Rank-0 Output

**Use:** Controlling printed output and file I/O.

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

# Single print statement (rank 0 only)
if rank == 0:
    print("Starting simulation...")

# Conditional logging
if rank == 0:
    logger.info(f"Assembling system with {V.dofmap.index_map.size_global} DOFs")

# Barrier to ensure rank 0 finishes before others continue (if needed)
comm.Barrier()

# File I/O on rank 0
if rank == 0:
    with open("results.txt", "w") as f:
        f.write("Results\n")
        f.write("-------\n")

# Collective file writes use XDMF or VTK (parallel-aware)
```

## Pattern 2: Collective Norm and Integral

**Use:** Computing global norms and integrals across all ranks.

```python
from dolfinx.fem import assemble_scalar, form

u = fem.Function(V)

# L2 norm (automatically collective in PETSc)
norm_L2 = u.x.petsc_vec.norm(PETSc.NormType.N2)

# Integral (assemble locally, allreduce globally)
local_integral = assemble_scalar(form(u * ufl.dx))
global_integral = comm.allreduce(local_integral, op=MPI.SUM)

# Error integral (l2 norm of u - u_exact)
local_err = assemble_scalar(form((u - u_exact)**2 * ufl.dx))
global_err = comm.allreduce(local_err, op=MPI.SUM)
error = np.sqrt(global_err)

if rank == 0:
    print(f"Global error: {error:.6e}")
```

## Pattern 3: Parallel Ghost Exchange

**Use:** After assembly, communicate ghost values.

```python
# Standard pattern (non-blocking)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

# Optimized pattern: overlap with computation
def assemble_and_communicate(forms):
    """Assemble multiple forms with communication overlap."""
    vectors = []
    for form in forms:
        v = fem.assemble_vector(fem.form(form))
        v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
        vectors.append(v)

    # Computation during communication
    # ... do work here ...

    for v in vectors:
        v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

    return vectors
```

## Pattern 4: Broadcasting Data

**Use:** One rank generates data, all ranks need it.

```python
# Broadcast scalar
if rank == 0:
    value = 3.14159
else:
    value = None

value = comm.bcast(value, root=0)
print(f"Rank {rank}: value = {value}")  # all ranks print 3.14159

# Broadcast array
if rank == 0:
    data = np.array([1, 2, 3, 4, 5])
else:
    data = None

data = comm.bcast(data, root=0)
# Now all ranks have the array

# Broadcast string (e.g., filename)
if rank == 0:
    filename = "solution_t100.xdmf"
else:
    filename = None

filename = comm.bcast(filename, root=0)

# All ranks read same file with same filename
with XDMFFile(comm, filename, "r") as xdmf:
    u = xdmf.read_function(V, "u")
```

## Pattern 5: Scatter / Gather

**Use:** Distribute work from rank 0, or collect results.

```python
# Scatter: rank 0 has work, distribute to all
if rank == 0:
    load_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
else:
    load_steps = None

# Each rank gets portion
my_steps = comm.scatter(load_steps, root=0)
print(f"Rank {rank}: working on steps {my_steps}")

# Process locally
local_results = [process(step) for step in my_steps]

# Gather: collect all results back to rank 0
all_results = comm.gather(local_results, root=0)

if rank == 0:
    # Flatten and write results
    flat_results = [r for sublist in all_results for r in sublist]
    print(f"Collected {len(flat_results)} results")
```

## Pattern 6: AllGather

**Use:** All ranks collect data from all ranks.

```python
# Each rank computes local error
local_error = assemble_scalar(form((u - u_exact)**2 * ufl.dx))

# All ranks get all errors
all_errors = comm.allgather(local_error)

if rank == 0:
    # Analyze global errors
    global_error = np.sqrt(sum(all_errors))
    max_error = np.sqrt(np.max(all_errors))
    print(f"Global error: {global_error:.6e}, max: {max_error:.6e}")
```

## Pattern 7: Parallel Mesh Creation

**Use:** Creating mesh that's automatically distributed.

```python
# Collective: all ranks call, DOLFINx partitions automatically
mesh_obj = mesh.create_unit_square(
    comm,
    nx=100, ny=100,
    cell_type=mesh.CellType.triangle,
    ghost_mode=mesh.GhostMode.shared_facet
)

# All ranks have their partition
if rank == 0:
    print(f"Total cells: {mesh_obj.topology.index_map(mesh_obj.topology.dim).size_global}")
print(f"Rank {rank}: {mesh_obj.num_entities(mesh_obj.topology.dim)} local cells")

# Create function spaces (collective)
V = fem.FunctionSpace(mesh_obj, ("Lagrange", 1))
Q = fem.FunctionSpace(mesh_obj, ("Lagrange", 0))
```

## Pattern 8: Load Balancing Check

**Use:** Detecting load imbalance.

```python
# Timer for critical section
import time

t0 = time.time()
# ... perform assembly ...
local_time = time.time() - t0

# Gather all times
all_times = comm.gather(local_time, root=0)

if rank == 0:
    max_time = np.max(all_times)
    min_time = np.min(all_times)
    imbalance = (max_time - min_time) / min_time if min_time > 0 else 0
    print(f"Load imbalance: {100*imbalance:.1f}%")
    if imbalance > 0.2:
        print("Warning: significant load imbalance detected")
```

## Pattern 9: Conditional Barrier

**Use:** Ensuring synchronization before proceeding.

```python
# Implicit barrier in collective operations
comm.Barrier()  # explicit wait for all ranks

# After computation-intensive step
compute_something()
comm.Barrier()  # ensure all done before next step

# Conditional barrier (if only some ranks do work)
if rank < 2:
    heavy_computation()
comm.Barrier()  # wait for everyone before proceeding
```

## Pattern 10: Point Evaluation (Robust)

**Use:** Evaluate function at arbitrary points safely.

```python
from dolfinx.fem import evaluate_function_at_points

# Points of interest
points = np.array([
    [0.25, 0.25],
    [0.5, 0.5],
    [0.75, 0.75]
]).T  # shape (2, 3)

# Strategy 1: Local evaluation (may fail if point not local)
try:
    local_values = evaluate_function_at_points(u, points)
except RuntimeError:
    local_values = np.full(points.shape[1], np.nan)

# Strategy 2: Robust gathering (all points to all ranks)
all_points_list = comm.allgather(points)

# Evaluate on each rank's mesh
rank_values = []
for pts in all_points_list:
    try:
        vals = evaluate_function_at_points(u, pts)
        rank_values.append(vals)
    except RuntimeError:
        rank_values.append(np.full(pts.shape[1], np.nan))

# Combine (first valid evaluation per point)
final_values = []
for pt_idx in range(points.shape[1]):
    for rank_vals in rank_values:
        if not np.isnan(rank_vals[pt_idx]):
            final_values.append(rank_vals[pt_idx])
            break
    else:
        final_values.append(np.nan)

if rank == 0:
    print(f"Point evaluations: {final_values}")
```

## Pattern 11: Parallel Checkpoint/Restart

**Use:** Saving and loading distributed solutions.

```python
# === Checkpoint (all timesteps) ===

def checkpoint(u, u_prev, t):
    """Save solution at timestep."""
    with XDMFFile(comm, "checkpoint.xdmf", "a") as xdmf:
        xdmf.write_function(u, t)

# In time loop
for t in np.linspace(0, T, Nt):
    # Solve...
    solve_step(u, u_prev)

    if timestep % checkpoint_interval == 0:
        checkpoint(u, u_prev, t)
        if rank == 0:
            print(f"Checkpointed at t={t:.4f}")

# === Restart (from checkpoint) ===

def restart_from_checkpoint(V, t_restart):
    """Load solution from checkpoint."""
    with XDMFFile(comm, "checkpoint.xdmf", "r") as xdmf:
        u = fem.Function(V)
        u = xdmf.read_function(V, name="u", time=t_restart)
        u_prev = fem.Function(V)
        u_prev = xdmf.read_function(V, name="u_prev", time=t_restart)
    return u, u_prev

u, u_prev = restart_from_checkpoint(V, t_restart=0.5)
```

## Pattern 12: Reduce (Min/Max/Sum)

**Use:** Computing global min/max/sum across all ranks.

```python
# Find maximum value in solution
local_max = np.max(u.x.array)
global_max = comm.allreduce(local_max, op=MPI.MAX)

# Find minimum
local_min = np.min(u.x.array)
global_min = comm.allreduce(local_min, op=MPI.MIN)

# Sum across ranks
local_sum = np.sum(u.x.array)
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

# Count
local_count = len(u.x.array)
global_count = comm.allreduce(local_count, op=MPI.SUM)

# Average
global_avg = global_sum / global_count

if rank == 0:
    print(f"Solution range: [{global_min:.6e}, {global_max:.6e}]")
    print(f"Average: {global_avg:.6e}")
```

## Pattern 13: Scaling Test

**Use:** Measuring parallel efficiency.

```python
import time

def strong_scaling_test():
    """Fixed global problem, vary rank count."""
    # Create fixed-size global mesh
    mesh_obj = mesh.create_unit_square(comm, nx=1000, ny=1000)
    V = fem.FunctionSpace(mesh_obj, ("Lagrange", 1))

    # Problem setup
    a, L = define_problem(V)

    # Time assembly
    comm.Barrier()
    t0 = time.time()
    A = fem.assemble_matrix(fem.form(a), bcs=bcs)
    b = fem.assemble_vector(fem.form(L))
    comm.Barrier()
    t_assembly = time.time() - t0

    # Time solve
    comm.Barrier()
    t0 = time.time()
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()
    comm.Barrier()
    t_solve = time.time() - t0

    # Gather timing
    all_assembly = comm.gather(t_assembly, root=0)
    all_solve = comm.gather(t_solve, root=0)

    if rank == 0:
        avg_assembly = np.mean(all_assembly)
        avg_solve = np.mean(all_solve)
        print(f"{comm.size:2d} ranks: assembly={avg_assembly:8.4f}s, solve={avg_solve:8.4f}s")

# Run with: mpirun -np 1 python3 ...
#           mpirun -np 2 python3 ...
#           mpirun -np 4 python3 ...
#           mpirun -np 8 python3 ...
```

## Pattern 14: Nonblocking Isend/Irecv

**Use:** Advanced: overlapping computation with custom communication.

```python
# Most users: avoid this, use collective operations instead.
# DOLFINx/PETSc usually handle communication implicitly.

# Example: if you implement custom parallel algorithm
from mpi4py import MPI

# Send request
request = comm.Isend(sendbuf, dest=(rank+1)%size)

# Do computation while message in flight
compute()

# Wait for completion
request.Wait()
receive_data = recvbuf
```

**Note:** DOLFINx assembly and KSP solve handle communication internally; avoid custom Isend/Irecv unless truly necessary.

## Pattern 15: Profiling Region (PETSc Logging)

**Use:** Detailed performance analysis of specific code regions.

```python
from petsc4py import PETSc

# Start logging
event_id = PETSc.Log.getEventId("MyCustomEvent")
PETSc.Log.eventBegin(event_id)

# Code to time
do_work()

PETSc.Log.eventEnd(event_id)

# At end of program, PETSc prints detailed breakdown
# Run with: python3 script.py -log_view
```

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Only rank 0 creates Function | Deadlock | Create Function on all ranks |
| Rank-specific mesh creation | Inconsistent topology | Use comm to create mesh, not if/rank |
| Missing ghostUpdate | Wrong assembly results | Call ghostUpdate after assembly |
| AllGather in tight loop | Excessive overhead | Batch communication outside loop |
| Print from all ranks | Spam output | Use `if rank == 0: print(...)` |
| Non-collective I/O | File corruption | Use XDMF or rank 0 with barrier |
| Hardcoded rank-0 assumptions | Fails on different rank counts | Use rank/size dynamically |
| Sync all ranks before every operation | Extreme slowdown | Let collective ops handle sync |

## See Also

- `parallel-mpi-awareness/SKILL.md`: Full parallel tutorial
- PETSc documentation: https://petsc.org/
- mpi4py documentation: https://mpi4py.readthedocs.io/
