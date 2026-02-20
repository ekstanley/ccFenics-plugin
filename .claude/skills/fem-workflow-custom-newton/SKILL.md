---
name: fem-workflow-custom-newton
description: |
  Guides the user through implementing custom Newton solvers and load stepping for nonlinear problems using DOLFINx MCP tools.
  Use when the user asks about custom Newton loops, load stepping, manual Newton iteration,
  nonlinear convergence control, or incremental loading.
---

# Custom Newton Solver Workflow (Tutorial Ch4.5)

Implement **manual Newton iteration** for full control over the nonlinear solve process.

## Two Approaches

### Approach 1: `solve_nonlinear` (Standard Newton)

For most problems, the built-in Newton solver is sufficient:

```
solve_nonlinear(
    residual="(1 + u**2) * inner(grad(u), grad(v)) * dx - f * v * dx",
    unknown="u",
    snes_type="newtonls",
    rtol=1e-10,
    max_iter=50,
    solution_name="u_h"
)
```

### Approach 2: Manual Newton Loop (Full Control)

For load stepping, custom convergence criteria, or line search modifications:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, apply_lifting, set_bc
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Setup (mesh, space, BCs already created)
V = fem.functionspace(mesh, ("Lagrange", 1))
u = fem.Function(V)       # current solution
du = ufl.TrialFunction(V) # Newton increment
v = ufl.TestFunction(V)

# Residual F and Jacobian J
F = (1 + u**2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
J = ufl.derivative(F, u, du)

# Compile forms
F_form = fem.form(F)
J_form = fem.form(J)

# Assemble structures
A = assemble_matrix(J_form, bcs=bcs)
b = create_vector(F_form)

# Newton solver
solver = PETSc.KSP().create(mesh.comm)
solver.setType("preonly")
solver.getPC().setType("lu")

# Newton loop
du_vec = fem.Function(V)
rtol = 1e-10
max_iter = 25

for i in range(max_iter):
    # Assemble residual
    with b.localForm() as b_local:
        b_local.set(0.0)
    assemble_vector(b, F_form)
    apply_lifting(b, [J_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Check convergence
    residual_norm = b.norm()
    if i == 0:
        residual_norm_0 = residual_norm
    print(f"  Newton iter {i}: residual = {residual_norm:.3e}")

    if residual_norm / residual_norm_0 < rtol:
        print(f"  Converged in {i} iterations")
        break

    # Assemble Jacobian
    A.zeroEntries()
    assemble_matrix(A, J_form, bcs=bcs)
    A.assemble()

    # Solve linear system
    solver.setOperators(A)
    solver.solve(b, du_vec.x.petsc_vec)
    du_vec.x.scatter_forward()

    # Update solution: u -= du (note: residual has opposite sign convention)
    u.x.array[:] -= du_vec.x.array

print(f"Final residual: {b.norm():.3e}")
print(f"Solution norm: {np.linalg.norm(u.x.array):.6e}")
""")
```

> **Namespace persistence**: Variables defined in `run_custom_code` persist across calls.
> You can split complex workflows into multiple calls without re-importing or re-defining objects.
> Session-registered objects (meshes, spaces, functions) are always injected fresh and override stale names.

## Load Stepping

For problems with large deformations or strong nonlinearity, apply the load incrementally:

```python
run_custom_code(code="""
# Load stepping for hyperelasticity
load_steps = np.linspace(0, 1.0, 10)  # 10 increments

for step, load_factor in enumerate(load_steps[1:], 1):
    # Update applied load
    traction.value = load_factor * max_traction

    # Newton iteration (inner loop)
    for i in range(max_iter):
        # Assemble, solve, update...
        if converged:
            break

    print(f"Load step {step}: factor={load_factor:.2f}, Newton iters={i}")
""")
```

## Key Concepts

- **Newton increment**: `du` is the correction at each iteration; `u_{k+1} = u_k - du`
- **Residual assembly**: Must zero the vector, assemble, apply lifting, ghost update, set BC
- **Jacobian assembly**: Must zero entries, assemble, then `A.assemble()`
- **Convergence criterion**: `||F(u_k)|| / ||F(u_0)|| < rtol` (relative residual)
- **Load stepping**: Break the applied load into increments; use previous solution as initial guess

## When to Use Custom Newton

| Use `solve_nonlinear` | Use custom Newton loop |
|---|---|
| Standard Newton sufficient | Need load stepping |
| No special convergence control | Custom convergence criteria |
| Simple problems | Need to monitor/log each iteration |
| First attempt | `solve_nonlinear` didn't converge |

## Convergence Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Diverges immediately | Bad initial guess | Start from linear solution |
| Oscillates | No line search | Add damping: `u -= alpha*du` with alpha < 1 |
| Converges slowly | Weak nonlinearity | May be fine, check tolerance |
| Stalls | Strong nonlinearity | Load stepping |
