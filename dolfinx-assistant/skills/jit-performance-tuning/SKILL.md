# JIT Performance Tuning

Optimize DOLFINx simulations through FFCx compiler configuration, quadrature strategy, and profiling.

## FFCx Compiler Fundamentals

FFCx (FEniCS Form Compiler) converts UFL forms into C code, which is then compiled via CFFI.

**Pipeline:**
```
UFL Form → FFCx → C code → CFFI compilation → .so module → PETSc calls
```

**Cost breakdown:**
- FFCx: ~0.1-1 second per form (one-time)
- CFFI compilation: ~1-10 seconds per form (one-time, most expensive)
- Cached: Future runs load .so from cache (~0.01s)

**Cache location:** `~/.cache/fenics/` (Unix) or `%APPDATA%\fenics\` (Windows)

## JIT Options Configuration

Pass `jit_options` dict to `fem.form()` or assembly functions:

```python
from dolfinx import fem
import dolfinx

# Option 1: Simple form compilation
jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native"],
    "optimize": True
}

compiled_form = fem.form(a, jit_options=jit_options)
A = fem.assemble_matrix(compiled_form, bcs=bcs)

# Option 2: Module-level default (all forms use these options)
dolfinx.default_jit_options = {
    "cffi_extra_compile_args": ["-O3"],
    "cffi_libraries": ["m"],  # math library
}

# Option 3: Environment variable (lowest priority)
import os
os.environ["DOLFINX_JIT_OPTIONS"] = ...  # format: key=value pairs
```

## Key JIT Options Reference

| Option | Values | Effect |
|---|---|---|
| **cffi_extra_compile_args** | `["-O0"]`, `["-O2"]`, `["-O3"]`, `["-march=native"]` | Compiler optimization level |
| **cffi_libraries** | `["m"]` (math), `["c"]` (C library) | Link additional libraries |
| **cffi_include_dirs** | `["/usr/include/petsc"]` | Include search paths |
| **cffi_library_dirs** | `["/usr/lib64"]` | Library search paths |
| **optimize** | `True` / `False` | FFCx form optimization (default: True) |
| **cpp_optimize_flags** | String | C++ flags passed to compiler |

**Recommended defaults:**

```python
default_jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"],
    "optimize": True,
    "cpp_optimize_flags": "-O3 -march=native"
}
```

## Quadrature Degree Strategy

Quadrature degree determines number of integration points per cell. Higher degree = more accurate but slower.

### Automatic (Default)

```python
# FFCx estimates quadrature degree from form
# Works well for polynomial integrands
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # degree = 2 (estimated)
```

**FFCx estimation rule:**
- For bilinear form a(u,v): degree = deg(u) + deg(v)
- For nonlinear form F(u;v): degree = 2*deg(u) + deg(v)

### Explicit Override

```python
from ufl import inner, grad, dx, Measure

# Specify degree explicitly
metadata = {"quadrature_degree": 4}
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(metadata=metadata)

# Or higher-level
dx_custom = Measure("cell", metadata={"quadrature_degree": 6})
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_custom
```

### Custom Quadrature Rule

```python
# For specialized problems (rare)
metadata = {
    "quadrature_rule": "vertex",  # evaluate at vertices only
    "quadrature_degree": 0         # minimal integration
}

# Or use default adaptive rule
metadata = {"quadrature_rule": "default"}
```

## When to Increase Quadrature Degree

Increase quadrature degree when the integrand is **not a polynomial**:

```python
# Example 1: Nonlinear material law
# Stress σ = (1 + ||ε||^2) * ε  (nonlinear in strain)

e = ufl.sym(ufl.grad(u))        # strain: degree 1 (u is P2)
sigma = (1 + ufl.inner(e, e)) * e  # degree 5!

# Automatic: degree = 2*2 + 1 = 5 (OK)
# But may miss higher-order effects; safer to use degree=8

metadata = {"quadrature_degree": 8}
a_form = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx(metadata=metadata)

# Example 2: Curved mesh with isoparametric elements
# Element Jacobian is nonlinear in reference coordinates
# DOLFINx handles automatically, but high-order elements need higher quadrature

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(metadata={"quadrature_degree": 4})

# Example 3: Exponential nonlinearity
a = ufl.exp(u) * v * ufl.dx(metadata={"quadrature_degree": 6})
```

## Performance Profiling

### Timing Assembly vs Solve

```python
import time

# Time form compilation + assembly
t0 = time.time()
form_compiled = fem.form(a)
t_form = time.time() - t0

t0 = time.time()
A = fem.assemble_matrix(form_compiled, bcs=bcs)
t_assembly = time.time() - t0

# Time solve
t0 = time.time()
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.solve(b, x)
t_solve = time.time() - t0

if mesh.comm.rank == 0:
    print(f"Form compilation: {t_form:.4f}s")
    print(f"Assembly:        {t_assembly:.4f}s")
    print(f"Solve:           {t_solve:.4f}s")
    print(f"Assembly/Solve ratio: {t_assembly/t_solve:.2f}")
```

**Typical bottleneck:**
- **Small systems** (<10k DOFs): Assembly dominates
- **Large systems** (>1M DOFs): Solve dominates

### PETSc Logging

Detailed breakdown of every operation:

```bash
# Run with logging enabled
python3 script.py -log_view > profile.txt

# Inspect profile.txt for stage/event breakdowns
# Look for:
# - MatAssembly time
# - KSPSetup time
# - KSPSolve time
# - MatMult time (most expensive in KSP loop)
```

Output example:
```
Event                Count      Time (sec)     % of total
MatAssembly            1         0.250         10.5%
KSPSetup               1         0.150          6.3%
KSPSolve               1         1.800         75.4%
  MatMult             50         1.200         50.4%
  PCApply             50         0.450         18.9%
```

### Flop Rate Analysis

```python
# Estimate FLOPs in assembly
# Bilinear form a(u,v) with dim=2, P1 elements
# Approximate FLOP count: ~100-300 FLOPs per cell

n_cells = mesh.num_entities(mesh.topology.dim)
estimated_flops = 200 * n_cells  # rough estimate

actual_time = t_assembly

flops_per_sec = estimated_flops / actual_time

if mesh.comm.rank == 0:
    print(f"Assembly speed: {flops_per_sec/1e9:.2f} GFLOPs/s")
    # Typical: 5-20 GFLOPs/s on modern CPU (assembly is not compute-intensive)
```

## Form Caching and Cache Management

DOLFINx caches compiled forms by **hash of UFL + JIT options**.

### Cache Hit (Reuse)

```python
# First call: compile and cache
a1 = fem.form(a, jit_options={"cffi_extra_compile_args": ["-O3"]})

# Second call (same form + options): cache hit, loads from disk
a2 = fem.form(a, jit_options={"cffi_extra_compile_args": ["-O3"]})

# Time difference: ~0.01s vs ~10s (100x speedup)
```

### Cache Location

```bash
# View cache
ls ~/.cache/fenics/

# Size of cache
du -sh ~/.cache/fenics/

# Clear cache (start fresh)
rm -rf ~/.cache/fenics/

# Rebuilds everything but ensures fresh compilation
```

**Note:** Cache is **not invalidated** on DOLFINx/FEniCS version change!
If upgrading libraries, clear cache to avoid ABI mismatches.

### Disable Caching

```python
# Rarely needed (debugging only)
import dolfinx
dolfinx.use_cache = False  # disable caching

# Forms will recompile every call (slow, but ensures fresh build)
```

## Matrix Storage and Sparsity

DOLFINx uses **CSR (Compressed Sparse Row)** format for matrices.

### Sparsity Pattern Analysis

```python
# Inspect matrix structure
A = fem.assemble_matrix(fem.form(a), bcs=bcs)

# Number of nonzeros
nnz = A.getInfo()["nz_allocated"]  # allocated
nnz_actual = A.getInfo()["nz_used"]  # actually used

# Density
n_rows, n_cols = A.getSize()
density = nnz_actual / (n_rows * n_cols)

print(f"Matrix size: {n_rows}x{n_cols}")
print(f"Nonzeros: {nnz_actual} ({100*density:.2f}%)")
print(f"Average nnz per row: {nnz_actual/n_rows:.1f}")
```

**Rules of thumb:**
- **Poisson (P1 FEM, 2D):** ~5-7 nonzeros per row, 0.1% density
- **Elasticity (P2 FEM, 3D):** ~100+ nonzeros per row, 10-20% density
- **DG elements:** 10-100x denser than CG (discontinuous → more couplings)

### Preallocate Sparsity (Advanced)

```python
# For custom assembly, preallocate pattern to avoid reallocation
d_nnz = [V.dofmap.index_map.local_range[1] for _ in range(n_rows)]  # diag blocks
o_nnz = [V.dofmap.index_map.size - d for d in d_nnz]  # off-diag blocks

A = PETSc.Mat().create(comm=mesh.comm)
A.setSizes((n_rows, n_cols))
A.setType("aij")
A.setPreallocationNNZ((d_nnz, o_nnz))

# Now insert values without reallocation
```

## Assembly Optimization Checklist

- [ ] Quadrature degree appropriate (not too high, not too low)
- [ ] Form compiled only once (reuse compiled form, don't recompile)
- [ ] Cache enabled (delete `~/.cache/fenics/` only if debugging)
- [ ] Boundary conditions applied efficiently (use `apply_lifting` post-assembly)
- [ ] Matrix sparsity profiled (inspect nnz count)
- [ ] JIT options set to `-O3` and `-march=native` (if CPU-bound)
- [ ] PETSc profiling done (`-log_view`) to identify bottlenecks

## Memory Optimization

### Matrix Memory

```python
# CSR format: memory = nnz * 16 + n_rows * 8 (roughly)
# Example: 1M DOFs, 10 nnz/row
nnz = 1e6 * 10
memory_GB = (nnz * 16) / 1e9
# ≈ 160 GB (beware!)

# Solution vector memory
# complex128: 1M * 8 bytes = 8 MB (negligible)

# Preconditioner memory
# AMG: 2-3x matrix memory (due to coarse grids)
```

### Reduce Memory

```python
# Use single-precision (float32) if acceptable
scalar_type = "float32"  # instead of "float64"

# Requires rebuilding DOLFINx with PETSC_SCALAR_TYPE=real:float32
# Not recommended unless truly memory-constrained

# Better: reduce problem size via subdomains, domain decomposition
```

## When NOT to Optimize

- **Small problems** (<10k DOFs): Assembly is instantaneous; optimize solver instead
- **First run:** Don't optimize; measure first
- **Premature optimization:** Profile before optimizing (Amdahl's law)
- **Complex quadrature rules:** Cost of evaluating nonlinear kernels >> quadrature overhead

**Focus optimization effort on:**
1. Solver (preconditioner choice)
2. Quadrature degree (if automatic is wrong)
3. Algorithm (use multigrid, not direct solve for large systems)

## Common Pitfalls

| Issue | Symptom | Fix |
|---|---|---|
| Form recompiles every call | Slow (10s) on first use, then 10s again | Save compiled form: `a_comp = fem.form(a)` |
| Quadrature too high | Slow assembly, high memory | Use automatic quadrature, or lower degree |
| Quadrature too low | Inaccurate results, wrong error | Increase quadrature_degree in metadata |
| Old cache conflicts | Mysterious segfault or wrong results | `rm -rf ~/.cache/fenics/` |
| `-march=native` not portable | Compile on CPU A, run on CPU B → segfault | Remove from cffi_extra_compile_args for portable builds |
| `-ffast-math` non-conforming | Results differ from reference (IEEE violations) | Remove if reproducibility required |
| CFFI compilation fails | `RuntimeError: cffi compilation failed` | Check compiler flags, disable `-ffast-math` |

## Integration with DOLFINx MCP Tools

### Pass JIT Options to `solve()`

The MCP `solve()` tool doesn't yet support custom JIT options directly. **Workaround:**

```python
# Use run_custom_code with custom jit_options
code = """
from dolfinx import fem

jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native"],
    "optimize": True
}

a_form = fem.form(a, jit_options=jit_options)
L_form = fem.form(L, jit_options=jit_options)

problem = fem.petsc.LinearProblem(
    a_form, L_form, bcs=bcs,
    petsc_options={"-ksp_type": "cg", "-pc_type": "hypre"}
)

uh = problem.solve()
"""

result = run_custom_code(code=code)
```

### Profiling via PETSc Options

The `solve()` tool accepts `petsc_options`:

```python
result = solve(
    solver_type="iterative",
    petsc_options={
        "-log_view": None,  # Enable PETSc profiling
        "-log_view_memory": None  # Include memory usage
    }
)

# Inspect output for assembly/solve breakdown
```

## See Also

- `ffcx-options.md`: Complete FFCx option reference
- `solve()` tool: Solver configuration with PETSc options
- DOLFINx documentation: https://docs.fenicsproject.org/dolfinx/
- PETSc manual: https://petsc.org/release/docs/manual/
