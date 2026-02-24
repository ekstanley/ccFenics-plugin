# Parallel MPI Execution Setup

Interactive command to configure and test parallel DOLFINx simulations with MPI.

## Step 1: Important Clarification

**Key fact:** The DOLFINx MCP server runs **single-rank (1 process)**.

This command helps you:
1. Generate parallel-ready Python scripts
2. Configure MPI settings
3. Provide instructions to run externally with `mpirun`

**Workflow:**
```
[You] → [MCP server generates script] → [You run: mpirun -np 4 python3 script.py]
```

---

## Step 2: Problem Size Assessment

**Question:** How large is your problem?

**Provide:**
- **Global DOF count:** Total degrees of freedom in full mesh
  - Small: <100k DOFs
  - Medium: 100k-1M DOFs
  - Large: 1M-10M DOFs
  - Very large: >10M DOFs

- **Current single-rank performance:** (if known)
  - Assembly time
  - Solve time

**Example:**
```
Problem: 2D Poisson on [0,1]^2
Mesh: 1000×1000 cells (2M triangles)
Unknowns: 1.01M DOFs (P1 Lagrange)
Assembly: 2.5s
Solve (CG+AMG): 5.0s
Total: 7.5s per run
```

---

## Step 3: Target Rank Count

**Question:** How many MPI processes to use?

**Decision factors:**

| DOF Count | Recommended Ranks | Reason |
|---|---|---|
| <100k | 1 | No parallelism benefit |
| 100k-1M | 2-4 | Start with 2-4 to test scaling |
| 1M-10M | 4-16 | 250k-1M DOF per rank optimal |
| >10M | 16-128 | Aim for 500k-1M DOF per rank |

**Rule of thumb:** Minimum 100k-500k DOFs per rank for communication overhead to be < 50% of computation.

**Scaling test recommendation:**
```
Problem size fixed.
Try: np=1, 2, 4, 8, 16
Measure: Solve time, memory usage
Sweet spot: Maximum speedup before communication overhead dominates
```

---

## Step 4: Generate Parallel Script

**Action:** System generates a parallel-ready Python script from your single-rank session.

**Script template:**

```python
#!/usr/bin/env python3
"""
Parallel MPI DOLFINx script (auto-generated)

Run as:
  mpirun -np 4 python3 parallel_solve.py
"""

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        print(f"Running on {size} MPI processes")

    # === Create distributed mesh ===
    mesh_obj = mesh.create_unit_square(
        comm,
        nx=1000, ny=1000,
        cell_type=mesh.CellType.triangle,
        ghost_mode=mesh.GhostMode.shared_facet
    )

    # All ranks create function space (collective)
    V = fem.FunctionSpace(mesh_obj, ("Lagrange", 1))

    if rank == 0:
        print(f"Global DOF count: {V.dofmap.index_map.size_global}")
        print(f"Local DOF count (rank 0): {V.dofmap.index_map.local_range[1] - V.dofmap.index_map.local_range[0]}")

    # === Define variational problem ===
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(mesh_obj, PETSc.ScalarType(1.0))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # === Apply boundary conditions ===
    bcs = [fem.dirichletbc(
        PETSc.ScalarType(0),
        fem.locate_dofs_geometrical(V, lambda x: np.logical_or(x[0] == 0, x[1] == 0)),
        V
    )]

    # === Solve in parallel ===
    problem = fem.petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "-ksp_type": "cg",
            "-pc_type": "hypre",
            "-pc_hypre_type": "boomeramg",
            "-ksp_rtol": "1e-6"
        }
    )

    uh = problem.solve()

    # === Post-processing (rank 0 only) ===
    if rank == 0:
        print("Solve complete!")

    # === Parallel I/O ===
    from dolfinx.io import XDMFFile
    with XDMFFile(comm, "parallel_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh_obj)
        xdmf.write_function(uh)

    if rank == 0:
        print("Solution written to parallel_solution.xdmf")

if __name__ == "__main__":
    main()
```

**Action:** Copy generated script to your local machine.

---

## Step 5: Ghost Mode Configuration

**Question:** Which ghost mode do you need?

**Options:**

### GhostMode.shared_facet (Recommended)

```python
ghost_mode=mesh.GhostMode.shared_facet
```

**Use when:**
- Using DG (discontinuous Galerkin) elements
- Computing facet integrals (interior/exterior)
- Assembling any form with `ds`, `dS` measures
- **Default for most problems**

**Trade-off:** Slightly more communication, but essential for correctness.

### GhostMode.none (Rare)

```python
ghost_mode=mesh.GhostMode.none
```

**Use only when:**
- Cell-local computations only (no edge/vertex coupling)
- Minimizing memory footprint is critical
- **Rarely recommended**

**Answer:** Use `GhostMode.shared_facet` (already in generated script).

---

## Step 6: Parallel I/O Configuration

**Question:** How to save/load results in parallel?

**Option A: XDMF (Recommended)**

```python
from dolfinx.io import XDMFFile

# Write (all ranks collaborate on single file)
with XDMFFile(comm, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_obj)

# Append at each timestep
with XDMFFile(comm, "solution.xdmf", "a") as xdmf:
    xdmf.write_function(u, t=0.0)
    xdmf.write_function(v, t=0.0)

# Read back (works with any rank count)
with XDMFFile(comm, "solution.xdmf", "r") as xdmf:
    mesh_read = xdmf.read_mesh()
    u = xdmf.read_function(V, name="u")
```

**Advantages:**
- Single unified file (easy archival)
- Works with different rank counts on read
- Efficient parallel I/O

**Option B: VTK (Partition-based)**

```python
from dolfinx.io import VTKFile

# Each rank writes partition; Paraview assembles on read
with VTKFile(comm, "solution.pvd", "w") as vtk:
    vtk.write_mesh(mesh_obj)

with VTKFile(comm, "solution.pvd", "a") as vtk:
    vtk.write_function(u, t=0.0)
```

**Disadvantages:**
- Multiple files (one per rank per timestep)
- Requires visualization tool to assemble

**Recommendation:** Use XDMF for checkpointing, VTK for visualization only.

---

## Step 7: Running in Parallel

**Question:** How to execute the script?

**Commands (run from your machine, not MCP server):**

### Basic Execution

```bash
# Run on 4 processes
mpirun -np 4 python3 parallel_solve.py

# Run on 8 processes
mpirun -np 8 python3 parallel_solve.py

# Run on all cores (Linux)
mpirun -np $(nproc) python3 parallel_solve.py
```

### With Output Control

```bash
# Show output from rank 0 only (recommended)
mpirun -np 4 python3 parallel_solve.py

# Show output from all ranks (noisy)
mpirun -np 4 python3 parallel_solve.py 2>&1 | head -100

# Write rank-specific logs
mpirun -np 4 python3 parallel_solve.py 2>&1 > run_log.txt
```

### With Profiling

```bash
# Profile with PETSc (-log_view)
mpirun -np 4 python3 parallel_solve.py -log_view > profile.txt

# Summarize profiling
tail -100 profile.txt  # shows breakdown by operation
```

### On HPC Cluster

```bash
# SLURM submission script (example)
cat > run.sh << 'EOF'
#!/bin/bash
#SBATCH -N 2                # 2 nodes
#SBATCH -n 16               # 16 processes total
#SBATCH -t 01:00:00         # 1 hour timeout
#SBATCH --ntasks-per-node 8 # 8 tasks per node

module load mpi/openmpi-4.0
mpirun python3 parallel_solve.py
EOF

sbatch run.sh
```

---

## Step 8: Verify Parallel Setup

**Question:** Is the parallelization working?

**Check 1: Rank-specific output**

```bash
# Run and grep rank info
mpirun -np 4 python3 parallel_solve.py 2>&1 | grep -i "rank"

# Expected:
# Rank 0: 250,050 local cells
# Rank 1: 250,050 local cells
# Rank 2: 250,050 local cells
# Rank 3: 250,050 local cells
```

**Check 2: DOF distribution**

```python
# Add to script after creating V:
local_start, local_end = V.dofmap.index_map.local_range
print(f"Rank {rank}: DOFs [{local_start}, {local_end})")

# Expected: roughly equal distribution
# Rank 0: DOFs [0, 252500)
# Rank 1: DOFs [252500, 505000)
# ...
```

**Check 3: Communication test**

```python
# Add to script:
import time

comm.Barrier()  # sync all ranks
t0 = time.time()

# Collective operation
b = fem.assemble_vector(fem.form(L))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

comm.Barrier()  # sync again
t_comm = time.time() - t0

if rank == 0:
    print(f"Ghost communication time: {t_comm:.4f}s")
```

---

## Step 9: Scaling Study

**Question:** How does performance scale with rank count?

**Procedure:**

```bash
# Run with different rank counts
for np in 1 2 4 8; do
    echo "=== Running with $np ranks ==="
    mpirun -np $np python3 parallel_solve.py | grep "Solve time"
done
```

**Expected output:**
```
=== Running with 1 ranks ===
Solve time: 5.000s

=== Running with 2 ranks ===
Solve time: 2.800s (1.8x speedup)

=== Running with 4 ranks ===
Solve time: 1.600s (3.1x speedup)

=== Running with 8 ranks ===
Solve time: 0.950s (5.3x speedup)
```

**Scaling interpretation:**

| Speedup | Efficiency | Assessment |
|---|---|---|
| Linear (np speedup) | 100% | Ideal (rare) |
| Near-linear (0.8×np) | 80% | Excellent (typical) |
| Sublinear (0.5×np) | 50% | Good (acceptable) |
| Very sublinear (0.2×np) | 20% | Communication-bound (problem too small or too many ranks) |

**If poor scaling:**
- Increase mesh size
- Reduce rank count
- Improve preconditioner (AMG helps scaling)
- Check network bandwidth (HPC cluster issue)

---

## Step 10: Recommended Settings by Problem Size

**Use these as starting points, adjust based on scaling test.**

### Small Problem (100k-1M DOFs)

```bash
# Single rank (no parallelism overhead)
python3 parallel_solve.py

# Or 2 ranks if exploring MPI
mpirun -np 2 python3 parallel_solve.py
```

### Medium Problem (1M-10M DOFs)

```bash
# Start with 4 ranks
mpirun -np 4 python3 parallel_solve.py

# Or match available cores
mpirun -np $(nproc) python3 parallel_solve.py
```

### Large Problem (>10M DOFs)

```bash
# Use more ranks (8-32)
mpirun -np 16 python3 parallel_solve.py

# On cluster: use multiple nodes
srun -N 2 -n 16 python3 parallel_solve.py
```

---

## Step 11: Production Checklist

- [ ] Script generated from single-rank session (or written manually)
- [ ] All collective operations (mesh creation, space creation) present on all ranks
- [ ] Rank-0-only output: `if rank == 0: print(...)`
- [ ] Collective I/O: XDMF with `comm` parameter, or rank 0 with barrier
- [ ] Ghost mode: `GhostMode.shared_facet` used
- [ ] Null boundary conditions applied: `apply_lifting` used consistently
- [ ] Solver uses good preconditioner (AMG for elliptic)
- [ ] Scaling test done: speedup curve measured
- [ ] Script runs without deadlocks or hangs
- [ ] Output correctly written to file(s)

---

## Common Parallel Pitfalls

| Issue | Symptom | Fix |
|---|---|---|
| **Deadlock** | Process hangs indefinitely | Check all ranks execute collective ops; avoid if/rank in mesh creation |
| **Segmentation fault** | Crash on some ranks only | Unequal mesh distribution; check `ghost_mode` setting |
| **Wrong results** | Convergence issues | BCs applied inconsistently; missing `apply_lifting`; ghostUpdate missing |
| **Memory bloat** | OOM on some ranks | Unbalanced partition; use partitioner (DOLFINx handles automatically) |
| **I/O corruption** | File unreadable | Mix of XDMF and VTK writing; use single I/O strategy |
| **Slow communication** | ghostUpdate very slow | Network bandwidth issue; check -log_view for comm overhead |
| **Poor scaling** | Speedup < 2x with 4 ranks | Problem too small (reduce ranks) or preconditioner weak (use AMG) |

---

## Advanced: Hybrid OpenMP+MPI

**When rank count is high and shared-memory cores available:**

```bash
# 4 MPI ranks, each with 4 OpenMP threads
export OMP_NUM_THREADS=4
mpirun -np 4 python3 parallel_solve.py

# Or explicit placement (SLURM)
srun --ntasks-per-node=2 --cpus-per-task=4 python3 parallel_solve.py
```

**Benefit:** Better load balancing, lower communication overhead.

**When to use:** >16 core shared-memory nodes + distributed memory MPI.

---

## See Also

- **SKILL:** `parallel-mpi-awareness/SKILL.md` — Full parallel tutorial
- **Reference:** `parallel-mpi-awareness/references/mpi-patterns.md` — Reusable patterns
- **Tool:** `solve()` MCP tool — Works on single-rank; use for development
- **Command:** `/setup-matrix-free` — Preconditioner choice for parallel systems

---

## Summary

1. ✓ MCP server is single-rank; generate script for parallel execution
2. ✓ Provide `comm` to all mesh/space creation calls
3. ✓ Use `if rank == 0: print(...)` for output
4. ✓ Use XDMF for parallel I/O
5. ✓ Run with `mpirun -np <N> python3 script.py`
6. ✓ Do scaling test to find optimal rank count
7. ✓ Target 500k-1M DOFs per rank for best performance
