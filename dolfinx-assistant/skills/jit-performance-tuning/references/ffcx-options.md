# FFCx Compiler Options Reference

Comprehensive reference for controlling FFCx form compilation and CFFI code generation.

## FFCx High-Level Options

| Option | Type | Default | Purpose |
|---|---|---|---|
| `optimize` | bool | True | Enable FFCx form simplification and optimization |
| `scalar_type` | str | "float64" | Numeric type: "float64", "float32", "complex64", "complex128" |
| `epsilon` | float | 1e-14 | Threshold for zero detection (finite differences) |
| `verbosity` | int | 0 | Logging level (0=silent, 1=info, 2=debug) |
| `quadrature_degree` | int | auto | Override automatic quadrature degree selection |
| `quadrature_rule` | str | "default" | "default", "vertex", "representation", "custom" |

### Example: Control FFCx Optimization

```python
from dolfinx import fem

# Disable form optimization (rarely needed, debugging only)
jit_options = {"optimize": False}

a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
             jit_options=jit_options)

# This skips FFCx's form simplification, useful if optimization is causing issues
```

### Example: Switch Scalar Type

```python
# Single-precision (32-bit float)
jit_options = {"scalar_type": "float32"}

# Complex-valued
jit_options = {"scalar_type": "complex128"}

a_complex = fem.form(f * ufl.conj(v) * ufl.dx, jit_options=jit_options)
```

## CFFI Compiler Options

Control how FFCx-generated C code is compiled via CFFI.

| Option | Type | Example | Purpose |
|---|---|---|---|
| `cffi_extra_compile_args` | list | `["-O3", "-march=native"]` | Extra flags to C compiler (gcc/clang) |
| `cffi_extra_link_args` | list | `["-lm"]` | Extra flags to linker |
| `cffi_include_dirs` | list | `["/usr/include"]` | C header search paths |
| `cffi_library_dirs` | list | `["/usr/lib64"]` | Library search paths |
| `cffi_libraries` | list | `["m", "c"]` | Libraries to link against |

### Optimization Flags

```python
# Low optimization (fast compilation, slow execution)
jit_options = {
    "cffi_extra_compile_args": ["-O0"]
}

# Medium optimization (balance)
jit_options = {
    "cffi_extra_compile_args": ["-O2"]
}

# High optimization (slow compilation, fast execution)
jit_options = {
    "cffi_extra_compile_args": ["-O3"]
}

# Architecture-specific optimization
jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native", "-mtune=native"]
}

# SIMD vectorization
jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native", "-mavx2"]
}

# Performance compromise (not IEEE-754 compliant, faster)
jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"]
}
```

**Performance impact (typical):**
- `-O0` vs `-O3`: 2-5x slowdown
- `-march=native` vs generic: 10-20% speedup on same CPU
- `-ffast-math`: 10-30% speedup but violates IEEE-754 (use cautiously)

### Platform-Specific Examples

**Linux (GCC/Clang):**
```python
jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native", "-fPIC"],
    "cffi_extra_link_args": ["-lm"]
}
```

**macOS (Clang):**
```python
jit_options = {
    "cffi_extra_compile_args": ["-O3", "-march=native", "-fPIC"],
    "cffi_extra_link_args": ["-lm"]
}
```

**Windows (MSVC via CFFI):**
```python
jit_options = {
    "cffi_extra_compile_args": ["/O2", "/arch:AVX2"],
    # Note: Windows paths differ
}
```

## Common Compilation Issues

| Error | Cause | Fix |
|---|---|---|
| `cffi compilation failed` | Missing C compiler | Install gcc/clang, ensure PATH correct |
| `undefined reference to 'sin'` | Math library not linked | Add `-lm` to cffi_extra_link_args |
| `cannot find -lpetsc` | PETSc library path not found | Add to cffi_library_dirs |
| `illegal instruction (core dumped)` | `-march=native` runs on incompatible CPU | Remove `-march=native`, use generic flags |
| `warning: unused variable` | Harmless FFCx-generated code | Suppress with `-Wno-unused` if desired |

## C++ Compiler Compatibility

DOLFINx generates C code (not C++), which CFFI compiles with system C compiler.

**Typical compiler:** gcc or clang/LLVM.

**Verify compiler:**
```bash
gcc --version
# or
clang --version
```

**Minimum versions:**
- GCC: 5.0+
- Clang: 3.8+

## PETSc Integration Options

While not strictly CFFI options, these control how FFCx-generated code interfaces with PETSc:

| PETSc Configuration | Effect | Check Via |
|---|---|---|
| `--with-scalar-type=real:float64` (default) | 64-bit float | `python3 -c "from petsc4py import PETSc; print(PETSc.ScalarType)"` |
| `--with-scalar-type=real:float32` | 32-bit float | Same check |
| `--with-scalar-type=complex` | Complex numbers | Same check |
| `--with-precision=single` | Single precision (obsolete) | Newer PETSc uses scalar-type |

**Ensure FFCx scalar_type matches PETSc scalar-type**, otherwise type mismatches cause segfaults.

```python
# Check PETSc scalar type
from petsc4py import PETSc
print(f"PETSc scalar type: {PETSc.ScalarType}")

# Match in FFCx
if PETSc.ScalarType == complex:
    jit_options = {"scalar_type": "complex128"}
else:
    jit_options = {"scalar_type": "float64"}
```

## Advanced: Custom Quadrature

```python
# Override quadrature rule for specific integrals
from ufl import Measure

# Vertex-rule (evaluate at cell vertices only)
dx_vertex = Measure("cell", metadata={"quadrature_rule": "vertex"})

# Custom degree
dx_custom = Measure("cell", metadata={"quadrature_degree": 5})

# For facet integrals
ds_custom = Measure("exterior_facet", metadata={"quadrature_degree": 3})
dS_custom = Measure("interior_facet", metadata={"quadrature_degree": 4})
```

## Epsilon for Numerical Differentiation

When FFCx cannot symbolically differentiate (rare), it uses finite differences with step size `epsilon`:

```python
# Increase epsilon if finite differences oscillate
jit_options = {"epsilon": 1e-6}

# Decrease epsilon if differentiation is inaccurate
jit_options = {"epsilon": 1e-16}

# Default 1e-14 is usually fine
```

## Cache Control Environment Variables

```bash
# Disable cache (forces recompilation every run)
export DOLFINX_JIT_CACHE_DISABLE=1
python3 script.py

# Cache directory (default: ~/.cache/fenics/)
export XDG_CACHE_HOME=/custom/cache/path
# Now cache goes to /custom/cache/path/fenics/

# Verbosity during compilation
export DOLFINX_JIT_LOGGING_LEVEL=DEBUG
python3 script.py
```

## Profiling Compilation Time

```python
import time
from dolfinx import fem

# Time form compilation
t0 = time.time()
form_compiled = fem.form(a)  # FFCx + CFFI
t_compile = time.time() - t0

print(f"Compilation time: {t_compile:.2f}s")
# Typical: 2-10s (first time), 0.01s (cache hit)
```

## Memory Considerations for Kernel Size

Larger forms → larger generated C code → larger memory during compilation.

```python
# Simple form: small kernel
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
# Generated C code: ~10 KB, compile time: 1-2s

# Complex nonlinear form: large kernel
a = ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(v)) * ufl.dx
# Generated C code: ~50 KB, compile time: 3-5s

# Very complex coupled system: huge kernel
# Generated C code: >500 KB, compile time: 10-30s, memory: 1-2 GB during compilation
```

If compilation uses excessive memory, split into separate forms:
```python
# Instead of one big form
F = F_poisson + F_advection + F_reaction

# Use separate forms
F_poisson = ...
F_advection = ...
F_reaction = ...
```

## Recommended Default Configuration

```python
# Development (balance between compile-time and accuracy)
dev_jit_options = {
    "optimize": True,
    "scalar_type": "float64",
    "cffi_extra_compile_args": ["-O2"],
    "quadrature_degree": None  # automatic
}

# Production (maximize performance)
prod_jit_options = {
    "optimize": True,
    "scalar_type": "float64",
    "cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"],
    "quadrature_degree": None  # automatic
}

# Debugging (focus on compilation speed and reproducibility)
debug_jit_options = {
    "optimize": False,
    "scalar_type": "float64",
    "cffi_extra_compile_args": ["-O0", "-g"],
    "quadrature_degree": 2  # minimal
}
```

## Integration with DOLFINx Workflow

### Set Global Default

```python
from dolfinx import default_jit_options

default_jit_options.update({
    "cffi_extra_compile_args": ["-O3", "-march=native"],
    "optimize": True
})

# All subsequent fem.form() calls use these options
a1 = fem.form(...)  # uses default options
a2 = fem.form(...)  # uses default options
a3 = fem.form(..., jit_options={"cffi_extra_compile_args": ["-O0"]})  # overrides
```

### Per-Form Control

```python
# Slow, rarely-used form: minimize compile time
slow_form_options = {"cffi_extra_compile_args": ["-O0"]}
F_slow = fem.form(F_slow_expr, jit_options=slow_form_options)

# Hot-path form: maximize performance
hot_path_options = {"cffi_extra_compile_args": ["-O3", "-march=native"]}
F_hot = fem.form(F_hot_expr, jit_options=hot_path_options)
```

## Troubleshooting Checklist

- [ ] Is FFCx installed? `python3 -c "import ffcx"` should work
- [ ] Is CFFI installed? `python3 -c "from cffi import FFI"` should work
- [ ] Is C compiler available? `gcc --version` or `clang --version`
- [ ] Do compile flags match target CPU? (`-march=native` only on target CPU)
- [ ] Is PETSc scalar type matching FFCx scalar type?
- [ ] Have you cleared cache after major DOLFINx upgrade? `rm -rf ~/.cache/fenics/`
- [ ] Are quadrature degrees appropriate for polynomial order?
- [ ] Is `-ffast-math` acceptable for your precision requirements?

## See Also

- `jit-performance-tuning/SKILL.md`: Full JIT tuning guide
- DOLFINx documentation: https://docs.fenicsproject.org/
- FFCx repository: https://github.com/FEniCS/ffcx
- PETSc manual: https://petsc.org/release/docs/manual/
