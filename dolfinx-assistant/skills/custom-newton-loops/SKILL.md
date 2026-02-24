# Custom Newton Solvers

Build Newton-Raphson loops beyond the built-in `solve_nonlinear` tool. Essential for load stepping, adaptive convergence, nonlinear constraint handling, and debugging.

## When to Use Custom Newton

Use custom Newton loops when:

- **Load stepping / continuation**: Incrementally increase a parameter (load factor, material constant) from 0 to target to avoid convergence failure
- **Adaptive convergence criteria**: Change tolerance or max iterations based on iteration history
- **Monitoring and damping**: Log residual norms at each step, apply damping factors, detect stagnation
- **Nonlinear constraints**: Add projection steps or penalty enforcement between Newton iterations
- **Modified Newton**: Reuse Jacobian for multiple steps (skip reassembly for speed)
- **Line search or trust region**: Custom backtracking strategy beyond SNES defaults
- **Multiple nonlinearities**: Alternate between Newton and Picard for coupled systems

For simple nonlinear problems: use built-in `solve_nonlinear` tool (SNES handles most cases).

## Basic Newton-Raphson Loop Structure

```python
# Setup via run_custom_code
from dolfinx import fem
from petsc4py import PETSc
import numpy as np

# Define function and form
u = fem.Function(V)           # current iterate
u.x.array[:] = u0.x.array     # set initial guess (critical!)

v = fem.TestFunction(V)
du = fem.TrialFunction(V)

# Define nonlinear residual F(u; v)
F = ... # your UFL expression

# Jacobian J(u; du, v) = dF/du[du]
from ufl import derivative
J = derivative(F, u, du)

# Compile forms
f_form = fem.form(F)
j_form = fem.form(J)

# Extract boundary conditions
bcs = [bc1, bc2, ...]

# Newton iteration
atol = 1e-12
rtol = 1e-10
max_iter = 50

for it in range(max_iter):
    # Assemble residual
    b = fem.assemble_vector(f_form)
    fem.apply_lifting(b, [j_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

    # Check convergence on residual
    norm_b = b.norm()
    print(f"Iteration {it}: residual norm = {norm_b:.6e}")

    if norm_b < atol or (it > 0 and norm_b < rtol * norm_b_prev):
        print("Newton converged")
        break

    norm_b_prev = norm_b

    # Assemble Jacobian
    A = fem.assemble_matrix(j_form, bcs=bcs)
    A.assemble()

    # Solve for update: A * delta_u = -b
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.pc.setType("hypre")

    du_vec = A.createVecRight()
    ksp.solve(-b, du_vec)

    # Line search: optional backtracking
    alpha = 1.0
    u_old = u.x.copy()

    # Update solution
    u.x.array[:] += alpha * du_vec.array

else:
    print(f"Warning: Newton did not converge after {max_iter} iterations")
```

**Key points:**
- Initial guess `u0` strongly affects convergence (start close to solution)
- `derivative(F, u, du)` computes J = dF/du in UFL
- `apply_lifting` applies BCs to residual
- Check convergence on residual norm BEFORE solving
- Jacobian reuse: assemble once, use for multiple RHS solves (modified Newton)
- Line search: reduce step size if residual increases

## Load Stepping

Gradually increase load from 0 to target to maintain convergence:

```python
# Parameter-dependent residual
t_load = fem.Constant(mesh, PETSc.ScalarType(0.0))  # load factor [0, 1]

# Define F(u, t_load; v) with t_load multiplying the load
F = ... * t_load * ...  # e.g., inner(grad(u), grad(v))*dx - t_load*f*v*dx

num_steps = 10
load_factor_list = np.linspace(0, 1, num_steps + 1)[1:]  # 0.1, 0.2, ..., 1.0

for load_factor in load_factor_list:
    print(f"\n=== Load step {load_factor:.2f} ===")
    t_load.value = load_factor

    # Newton loop (as above)
    for it in range(max_iter):
        # ... Newton iteration
        if norm_b < atol:
            break

    # Store solution for post-processing
    u_solutions.append(u.x.copy())
```

**Strategy:**
- Start with small steps (0.1) to find stability region
- Increase step size if converges in few iterations
- Reduce step size if Newton stalls or diverges
- Adaptive stepping: if Newton fails, backtrack and retry with smaller step

## Modified Newton

Reuse Jacobian over multiple steps (skip reassembly):

```python
# Assemble Jacobian once
A = fem.assemble_matrix(j_form, bcs=bcs)
A.assemble()

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("cg")
ksp.pc.setType("hypre")
ksp.setFromOptions()

reuse_steps = 3

for it in range(max_iter):
    # Assemble residual (every iteration)
    b = fem.assemble_vector(f_form)
    fem.apply_lifting(b, [j_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

    norm_b = b.norm()
    print(f"Iteration {it}: residual norm = {norm_b:.6e}")

    if norm_b < atol:
        break

    # Solve with REUSED Jacobian
    du_vec = A.createVecRight()
    ksp.solve(-b, du_vec)

    u.x.array[:] += du_vec.array

    # Reassemble Jacobian every reuse_steps
    if (it + 1) % reuse_steps == 0:
        A = fem.assemble_matrix(j_form, bcs=bcs)
        A.assemble()
        ksp.setOperators(A)
        print(f"  -> Jacobian reassembled")
```

**Trade-off:** Fewer assemblies (faster) but convergence may slow (more iterations).

## Line Search (Armijo Backtracking)

Reduce step size if residual increases:

```python
# After computing du_vec
alpha = 1.0
alpha_min = 1e-8
c_armijo = 1e-4  # sufficient decrease parameter

u_old = u.x.copy()
norm_b_old = norm_b

# Save old residual for comparison
b_old = fem.assemble_vector(f_form)
fem.apply_lifting(b_old, [j_form], [bcs])
b_old.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
b_old.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

ls_iter = 0
while alpha > alpha_min:
    # Try update with current alpha
    u.x.array[:] = u_old + alpha * du_vec.array

    # Compute new residual
    b_new = fem.assemble_vector(f_form)
    fem.apply_lifting(b_new, [j_form], [bcs])
    b_new.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
    b_new.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)

    norm_b_new = b_new.norm()

    # Armijo condition: new residual sufficiently decreased
    if norm_b_new <= (1 - c_armijo * alpha) * norm_b_old:
        print(f"  Line search: alpha = {alpha:.6f}, |b_new| = {norm_b_new:.6e}")
        break

    alpha *= 0.5
    ls_iter += 1

    if ls_iter > 10:
        print("  Line search failed, exiting")
        break
else:
    # Revert to old solution if line search failed
    u.x.array[:] = u_old
    print("  Warning: Line search exhausted, reverted to old solution")
```

## Convergence Monitoring

```python
residuals = []
updates = []

for it in range(max_iter):
    # Residual norm
    b = fem.assemble_vector(f_form)
    fem.apply_lifting(b, [j_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.BEGIN)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.END)
    norm_b = b.norm()
    residuals.append(norm_b)

    # Solve for update
    A = fem.assemble_matrix(j_form, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.pc.setType("hypre")
    du_vec = A.createVecRight()
    ksp.solve(-b, du_vec)

    norm_du = du_vec.norm()
    updates.append(norm_du)

    # Stagnation detection
    if it > 2:
        ratio = residuals[-1] / residuals[-2]
        if ratio > 0.99:
            print(f"Warning: Convergence stagnating (ratio {ratio:.4f})")

    # Convergence report
    print(f"Iteration {it}: |res| = {norm_b:.6e}, |du| = {norm_du:.6e}, "
          f"ratio = {norm_du / norm_b:.6e}")

    u.x.array[:] += du_vec.array

    if norm_b < atol or norm_du < 1e-14:
        print(f"Converged in {it+1} iterations")
        break

# Plot convergence
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.semilogy(residuals, 'o-', label='|residual|')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Residual norm')
ax1.legend()
ax1.grid()

ax2.semilogy(updates, 'o-', label='|update|')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Update norm')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.savefig('newton_convergence.png')
```

### Monitoring and Diagnostics

After a custom Newton solve, inspect diagnostics:

```python
# Check solver state
get_solver_diagnostics()  # Returns convergence info, residual norms, wall time

# Inspect session to verify solution was registered
get_session_state()  # Confirm solution appears in solutions dict

# Evaluate solution at key points
evaluate_solution(points=[[0.5, 0.5]], function_name="u_newton")
```

If Newton fails to converge:
```python
# Reset and try with different parameters
remove_object(name="u_newton", object_type="function")  # Clean up failed attempt
# Adjust load stepping, damping, or initial guess and retry
```

## Damped Newton

Apply damping factor (step size < 1) for stability:

```python
# Conservative start with small step
step = 0.1  # or 0.5

# After computing du_vec:
u.x.array[:] += step * du_vec.array

# Can increase step if converging well
if it > 0 and residuals[-1] < 0.1 * residuals[-2]:
    step = min(step * 1.2, 1.0)
    print(f"  Increasing step to {step}")
```

## SNES vs Custom Newton

| Aspect | `solve_nonlinear` (SNES) | Custom Newton |
|--------|-------------------------|---------------|
| **Setup** | Specify residual + optional Jacobian | Full control over residual, Jacobian, solver |
| **Convergence monitoring** | Summary only | Every iteration |
| **Load stepping** | Via residual form parameter | Manual loop |
| **Line search** | Built-in (backtracking, quadratic, cubic) | Implement yourself |
| **Jacobian reuse** | SNES handles automatically | Manual reuse_steps |
| **Adaptive criteria** | Fixed atol/rtol | Change per iteration |
| **Debugging** | Harder (SNES black-box) | Full visibility |
| **When to use** | Standard nonlinear PDEs | Complex workflows, debugging |

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Divergence on first step | Poor initial guess | Start closer to solution or use load stepping |
| Slow convergence (>50 iter) | Jacobian not accurate (numerical issues) | Use `epsilon=1e-8` in derivative if using finite differences; switch to analytical Jacobian |
| Stagnation after few iterations | Jacobian singular (near bifurcation) | Add damping, reduce load step, perturb initial condition |
| NaN in residual | Jacobian assembly failed | Check for division by zero in UFL, ensure BCs applied correctly |
| Memory growth | Not destroying PETSc objects | Call `A.destroy()`, `ksp.destroy()` if iterating over many load steps |
| Line search fails repeatedly | Step size too large for problem structure | Increase `c_armijo`, reduce initial `alpha`, add regularization |

## Integration with DOLFINx MCP Tools

Use `run_custom_code` to execute Newton loops:

```python
# Call via MCP tool: run_custom_code
code = """
from dolfinx import fem
from petsc4py import PETSc
import numpy as np

# session object and mesh, function spaces already available from session state
# Access via: session.meshes['mesh_name'], session.function_spaces['space_name']

u = fem.Function(V)  # V is from session
u.x.array[:] = 0.0  # initial guess

# Newton loop here...
"""
```

After loop completes, register solution with session:

```python
# Inside run_custom_code, at end:
session.solutions['u_final'] = u  # makes it available for post-processing

# Can then call plot_solution('u_final') or evaluate_solution(..., function_name='u_final')
```

## See Also

- `solve_nonlinear`: Built-in tool using SNES (simpler for most cases)
- `newton-strategies.md`: Arc-length, Picard, quasi-Newton, deflation
- `/setup-matrix-free` command: Preconditioner configuration for large systems
