---
name: fem-workflow-navier-stokes
description: |
  Guides the user through solving the incompressible Navier-Stokes equations using DOLFINx MCP tools.
  Use when the user asks about Navier-Stokes, IPCS splitting scheme, channel flow,
  cylinder flow, incompressible flow, or Reynolds number.
---

# Navier-Stokes Workflow (Tutorial Ch2.5)

Solve the incompressible Navier-Stokes equations using the **Incremental Pressure Correction Scheme (IPCS)**.

## Equations

**du/dt + (u . grad)u - div(sigma) = f** in Omega
**div(u) = 0** in Omega

where sigma = 2*mu*epsilon(u) - pI, epsilon(u) = sym(grad(u)).

## IPCS Splitting Scheme

IPCS splits each time step into 3 sub-problems:
1. **Tentative velocity**: Solve for u* (momentum without pressure gradient)
2. **Pressure correction**: Solve Poisson for pressure p^{n+1}
3. **Velocity correction**: Project u* onto divergence-free space

## Implementation via `run_custom_code`

IPCS requires 3 separate variational forms per time step, so `run_custom_code` is the primary approach:

```python
run_custom_code(code="""
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI

# Create channel mesh
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([2.2, 0.41])],
    [220, 41],
    cell_type=mesh.CellType.triangle
)

# Taylor-Hood elements (P2/P1)
v_el = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
p_el = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
V = fem.functionspace(msh, v_el)
Q = fem.functionspace(msh, p_el)

# Functions
u_n = fem.Function(V, name="u_n")   # velocity at t_n
u_ = fem.Function(V, name="u_")     # tentative velocity
p_n = fem.Function(Q, name="p_n")   # pressure at t_n
p_ = fem.Function(Q, name="p_")     # pressure correction

# Trial/test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

# Parameters
dt = 0.001
mu = 0.001  # dynamic viscosity (Re = U*L/mu)
rho = 1.0

# Step 1: Tentative velocity
F1 = (rho / dt) * ufl.inner(u - u_n, v) * ufl.dx \\
   + rho * ufl.inner(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx \\
   + ufl.inner(2*mu*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx \\
   - ufl.inner(p_n, ufl.div(v)) * ufl.dx
a1 = ufl.lhs(F1)
L1 = ufl.rhs(F1)

# Step 2: Pressure correction
a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
L2 = -(rho / dt) * ufl.div(u_) * q * ufl.dx

# Step 3: Velocity correction
a3 = ufl.inner(u, v) * ufl.dx
L3 = ufl.inner(u_, v) * ufl.dx - (dt / rho) * ufl.inner(ufl.grad(p_ - p_n), v) * ufl.dx

# Time loop
T = 5.0
t = 0
num_steps = int(T / dt)

for step in range(num_steps):
    t += dt
    # Update BCs if time-dependent
    # ... (set inflow velocity, no-slip walls, outflow pressure)

    # Solve step 1
    problem1 = LinearProblem(a1, L1, bcs=bcs_v)
    u_.x.array[:] = problem1.solve().x.array

    # Solve step 2
    problem2 = LinearProblem(a2, L2, bcs=bcs_p)
    p_.x.array[:] = problem2.solve().x.array

    # Solve step 3
    problem3 = LinearProblem(a3, L3, bcs=[])
    u_new = problem3.solve()

    # Update for next step
    u_n.x.array[:] = u_new.x.array
    p_n.x.array[:] = p_.x.array

    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, t={t:.3f}")
""")
```

## Key Concepts

- **IPCS**: Operator splitting decouples velocity and pressure, avoiding saddle-point systems
- **Taylor-Hood**: P2 velocity / P1 pressure satisfies the inf-sup condition
- **Reynolds number**: Re = U*L/mu. Low Re (~1) = creeping flow, high Re (~1000+) = turbulent
- **CFL condition**: dt should satisfy CFL ~ u*dt/h < 1 for explicit advection terms

## Typical BCs

| Boundary | Velocity | Pressure |
|---|---|---|
| Inflow | Parabolic profile | (natural) |
| Walls | No-slip u=0 | (natural) |
| Outflow | (natural) | p=0 |

## Why `run_custom_code`?

IPCS requires 3 distinct variational forms and 3 linear solves per timestep with intermediate function updates. This cannot be expressed as a single `define_variational_form` + `solve` cycle.
