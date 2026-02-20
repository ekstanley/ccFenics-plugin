---
name: fem-workflow-helmholtz
description: |
  Guides the user through solving the Helmholtz equation using DOLFINx MCP tools.
  Use when the user asks about Helmholtz equation, acoustics, frequency domain,
  wave equation, impedance BCs, or scattering problems.
---

# Helmholtz Equation Workflow (Tutorial Ch2.8)

Solve: **nabla^2 p + k^2 p = 0** in Omega, with impedance boundary conditions.

## Problem

The Helmholtz equation models time-harmonic wave propagation (acoustics, electromagnetics):
- `k = omega/c` = wave number
- `p` = complex pressure amplitude
- Impedance BC: `dp/dn = -jkp` on absorbing boundaries

## Implementation via `run_custom_code`

Helmholtz requires complex arithmetic and frequency-dependent coefficients:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Create mesh
msh = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
V = fem.functionspace(msh, ("Lagrange", 2))

# Wave number
k0 = 2 * np.pi * 5  # 5 wavelengths across domain

# Source term (point-like source at center)
x = ufl.SpatialCoordinate(msh)
f = 100 * ufl.exp(-200 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

# Variational form: -k^2*p*v*dx + inner(grad(p), grad(v))*dx - jk*p*v*ds = f*v*dx
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Bilinear form with impedance BC
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \\
  - k0**2 * ufl.inner(u, v) * ufl.dx \\
  - 1j * k0 * ufl.inner(u, v) * ufl.ds

L = ufl.inner(f, v) * ufl.dx

# Solve (requires complex PETSc)
problem = LinearProblem(a, L, bcs=[],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="s_")
p_h = problem.solve()

print(f"Solution norm: {np.linalg.norm(p_h.x.array):.6e}")
print(f"Max |p|: {np.abs(p_h.x.array).max():.6e}")
""")
```

> **Namespace persistence**: Variables defined in `run_custom_code` persist across calls.
> You can split complex workflows into multiple calls without re-importing or re-defining objects.
> Session-registered objects (meshes, spaces, functions) are always injected fresh and override stale names.

## Key Concepts

- **Wave number**: `k = 2*pi*freq/c` where freq is frequency and c is wave speed
- **Impedance BC**: `dp/dn + jkp = 0` on absorbing boundaries (Sommerfeld radiation condition)
- **Complex solver**: Requires PETSc complex scalar type
- **Resolution rule**: Need ~10 elements per wavelength: `h < lambda/10 = 2*pi/(10*k)`

## Frequency Sweep

To study response over a range of frequencies:

```python
run_custom_code(code="""
# Sweep over frequencies
frequencies = [1, 2, 5, 10, 20]
for freq in frequencies:
    k0 = 2 * np.pi * freq
    # Re-assemble and solve for each k0
    # Record max amplitude
    ...
print("Frequency sweep complete")
""")
```

## Boundary Condition Types

| BC Type | Formula | Physical Meaning |
|---|---|---|
| Sound-hard (Neumann) | `dp/dn = 0` | Rigid wall reflection |
| Sound-soft (Dirichlet) | `p = 0` | Pressure release surface |
| Impedance (Robin) | `dp/dn + jkp = 0` | Absorbing boundary |
| Incident wave | `dp/dn + jkp = 2jk*p_inc` | Scattering problem |
