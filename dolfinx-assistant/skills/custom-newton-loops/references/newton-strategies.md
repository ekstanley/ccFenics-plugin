# Advanced Newton Strategies Reference

Specialized Newton variants and related iterative methods for challenging nonlinear problems.

## Arc-Length Method (Riks Method)

For problems with limit points or snap-through behavior (load vs displacement non-monotone).

**Problem:** Standard Newton fails when dF/du becomes singular (limit point).

**Solution:** Introduce arc-length constraint to track the solution branch through the limit point.

```python
# Pseudo-code for arc-length method
# Constraint: ||u - u_old||^2 + (lambda - lambda_old)^2 = ds^2
# where lambda is load factor, ds is arc-length step size

from dolfinx import fem
from petsc4py import PETSc
import numpy as np

ds = 0.1  # arc-length step size
lam = fem.Constant(mesh, PETSc.ScalarType(0.0))  # load factor

u = fem.Function(V)
u_old = fem.Function(V)
lam_old = 0.0

for step in range(num_steps):
    # Define arc-length constraint
    # Extended system: [F(u, lam); g(u, lam)] = 0
    # where g = ||u - u_old||^2 + (lam - lam_old)^2 - ds^2

    # Newton on extended system
    # [J  dF/dlam] [du    ]   [-F            ]
    # [2(u-u_old)T  2(lam-lam_old)] [dlam] = [-g]

    v = fem.TestFunction(V)
    du = fem.TrialFunction(V)

    # Residual
    F = ... * lam * ...  # load-dependent form
    dF_dlam = ...  # derivative w.r.t. load factor

    for it in range(max_iter):
        # Assemble residual F
        b_u = fem.assemble_vector(fem.form(F))
        fem.apply_lifting(b_u, [...], [bcs])

        # Scalar equation for load factor
        g = (u.x.norm() - u_old.x.norm())**2 + (lam.value - lam_old)**2 - ds**2

        # Assemble extended Jacobian
        J = fem.form(ufl.derivative(F, u, du))
        A = fem.assemble_matrix(J, bcs=bcs)

        # Solve extended system (simplified)
        # ... solve for du and dlam

        u.x.array[:] += du_vec.array
        lam_val = lam.value + dlam

        if convergence:
            break

    u_old.x.array[:] = u.x.array
    lam_old = lam.value
```

**Use when:**
- Load-displacement curve is non-monotone
- Need to trace unstable equilibrium branches
- Studying bifurcations or snap-through

**Trade-offs:** More complex code, requires proper initial direction selection.

## Picard Iteration (Fixed-Point)

Linear iteration without computing Jacobian.

**Method:** Decouple nonlinearity via fixed-point iteration:
- Form F(u, v) = 0
- Rearrange: u = G(u) (extract linear and nonlinear parts)
- Iterate: u_{n+1} = G(u_n)

```python
# Example: nonlinear diffusion u_t - div(a(u) grad u) = 0
# where a(u) = 1 + u^2

from dolfinx import fem
import ufl

u = fem.Function(V)
u_prev = fem.Function(V)

# In fixed-point iteration, treat u_prev as known
a_prev = 1 + u_prev**2  # coefficient from previous iterate

v = fem.TestFunction(V)
du = fem.TrialFunction(V)

# Linear form: bilinear in du, but nonlinearity handled by u_prev
a_form = a_prev * ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx
L_form = ...

for it in range(max_iter):
    # Solve linear system with u_prev's coefficient
    problem = fem.petsc.LinearProblem(
        a_form, L_form, bcs=[...],
        petsc_options={...}
    )
    u.x.array[:] = problem.solve().x.array

    # Check convergence
    error = fem.assemble_scalar(fem.form((u - u_prev)**2 * ufl.dx))**0.5
    print(f"Iteration {it}: error = {error:.6e}")

    if error < tol:
        print("Picard converged")
        break

    u_prev.x.array[:] = u.x.array
```

**Convergence rate:** Linear (slower than Newton's quadratic).

**Advantage:** No Jacobian computation, simpler code.

**Disadvantage:** May not converge if nonlinearity is strong. Convergence basin smaller than Newton.

## Picard-to-Newton Switching

Start with Picard (robust), switch to Newton (fast) once near solution:

```python
# Hybrid strategy
use_picard = True
picard_residual_threshold = 1e-2  # switch when residual < this

for it in range(max_iter):
    if use_picard:
        # Picard step (linear solve)
        # ... solve linear system with u_prev's coefficient
        u.x.array[:] = u_linear.x.array

        # Check residual
        res = fem.assemble_scalar(fem.form((u - u_prev)**2 * ufl.dx))**0.5

        if res < picard_residual_threshold:
            print(f"Switching to Newton at iteration {it}")
            use_picard = False

    else:
        # Newton step (nonlinear solve)
        # ... compute Jacobian, solve Newton system
        u.x.array[:] += du_vec.array

    u_prev.x.array[:] = u.x.array

    if residual < atol:
        break
```

## Parameter Continuation (Homotopy)

Gradually deform problem from solvable initial state to target problem.

```python
# Problem: F(u, mu) = 0, want to solve for mu=1
# Homotopy: H(u, t) = (1-t)*F_easy(u) + t*F_hard(u)
# Track solution from t=0 (easy) to t=1 (hard)

from dolfinx import fem

t_param = fem.Constant(mesh, PETSc.ScalarType(0.0))

# Define easy problem (mu=0) and hard problem (mu=1)
def F_easy(u, v):
    return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

def F_hard(u, v):
    return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u**3 * v * ufl.dx  # nonlinear

# Homotopy residual
def F_homotopy(u, v, t):
    return (1 - t) * F_easy(u, v) + t * F_hard(u, v)

u = fem.Function(V)

num_steps = 20
for step, t_val in enumerate(np.linspace(0, 1, num_steps + 1)[1:]):
    print(f"\n=== Continuation step {step}, t = {t_val:.4f} ===")
    t_param.value = t_val

    v = fem.TestFunction(V)
    du = fem.TrialFunction(V)

    F = F_homotopy(u, v, t_param)
    J = ufl.derivative(F, u, du)

    # Newton loop with current t_val
    for it in range(max_iter):
        # ... Newton iteration
        pass
```

**Use when:**
- Target problem is difficult but nearby simpler problems are easy
- Standard Newton diverges from initial guess
- Need robustness over speed

## Deflation Method

Find multiple solutions to the same problem.

**Idea:** After finding solution u1, deflate the residual to exclude it:

```python
F_deflated = F / ||u - u1||^beta

# Now Newton on deflated residual converges to different solution u2
```

```python
# Practical deflation in DOLFINx
from dolfinx import fem
import ufl

solutions_found = []

def deflation_factor(u, u_prev):
    """
    Returns 1 / (eps + ||u - u_prev||)
    for each previously found solution.
    """
    factor = 1.0
    eps = 1e-6
    for u_sol in solutions_found:
        dist = fem.assemble_scalar(fem.form((u - u_sol)**2 * ufl.dx))**0.5
        factor *= 1.0 / (eps + dist)
    return factor

u = fem.Function(V)

# Try multiple random initial guesses
for trial in range(num_trials):
    print(f"\n=== Trial {trial} ===")

    # Random initial guess
    u.x.array[:] = np.random.randn(u.x.array.shape[0]) * 0.1

    v = fem.TestFunction(V)
    du = fem.TrialFunction(V)

    # Original residual
    F = ...

    # Apply deflation
    F_def = F * deflation_factor(u, solutions_found)

    # Newton on deflated problem
    J_def = ufl.derivative(F_def, u, du)

    for it in range(max_iter):
        b = fem.assemble_vector(fem.form(F_def))
        fem.apply_lifting(b, [fem.form(J_def)], [bcs])

        norm_b = b.norm()
        print(f"Iteration {it}: |res| = {norm_b:.6e}")

        if norm_b < 1e-12:
            # Check if this is a new solution
            is_new = True
            for u_old in solutions_found:
                dist = fem.assemble_scalar(fem.form((u - u_old)**2 * ufl.dx))**0.5
                if dist < 1e-4:
                    is_new = False
                    break

            if is_new:
                solutions_found.append(u.x.copy())
                print(f"  Found new solution #{len(solutions_found)}")
            break

        # Newton step (solve J_def * du = -F_def)
        # ...
```

**Converges to:** Different stationary points for each initial guess.

## Trust Region Methods

Restrict Newton step size to a "trust region" where quadratic model is accurate.

```python
# Instead of full Newton step du, solve:
# min ||J*du + F||^2  s.t.  ||du|| <= Delta (trust region radius)
#
# Use Levenberg-Marquardt as simple approximation:
# (J^T*J + mu*I) * du = -J^T*F

from dolfinx import fem
from petsc4py import PETSc

mu = 0.1  # regularization parameter
Delta = 1.0  # trust region radius

u = fem.Function(V)

for it in range(max_iter):
    # Assemble Jacobian
    J_form = fem.form(ufl.derivative(F, u, du))
    A = fem.assemble_matrix(J_form, bcs=bcs)

    # Add regularization: A := J^T*J + mu*I
    # (Simplified: diagonal addition)
    A.diagonalScale(left=None, right=None)  # normalize
    A.shift(mu)

    b = fem.assemble_vector(fem.form(F))

    # Solve regularized system
    du_vec = A.createVecRight()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.solve(-b, du_vec)

    # Check step size and trust region
    step_norm = du_vec.norm()

    if step_norm > Delta:
        # Step too large, decrease mu and recompute
        mu *= 2
        print(f"  Step too large ({step_norm:.6e} > {Delta}), increasing mu to {mu}")
        continue

    # Accept step
    u.x.array[:] += du_vec.array

    # Adapt trust region based on predicted vs actual reduction
    # ... (omitted for brevity)
```

## Quasi-Newton Methods (BFGS, L-BFGS)

Approximate Jacobian using gradient history; useful when Jacobian is expensive to compute.

```python
# L-BFGS: Limited-memory BFGS
# Maintain a few (s, y) pairs: s_k = u_{k+1} - u_k, y_k = grad_k+1 - grad_k
# Use these to approximate J^{-1} * grad without storing full matrix

# DOLFINx + PETSc: use SNES with -snes_type qn -snes_qn_type lbfgs

# Or implement manually (simplified 1-rank BFGS):
from dolfinx import fem
import numpy as np

u = fem.Function(V)
grad = fem.Function(V)  # gradient of scalar functional

# Store one (s, y) pair
s_prev = None
y_prev = None
H_diag = 1.0  # initial Hessian approximation (identity)

for it in range(max_iter):
    # Compute gradient (e.g., from residual)
    grad_form = fem.form(...)
    grad_vec = fem.assemble_vector(grad_form)
    grad.x.array[:] = grad_vec.array

    # Approximate update: du = -H_approx * grad
    # Using BFGS: H_k = (I - s*y^T / y^T*s) * H_{k-1} * (I - y*s^T / y^T*s) + s*s^T / y^T*s

    if s_prev is not None:
        # BFGS recurrence
        sy = np.dot(s_prev, y_prev)
        if abs(sy) > 1e-8:
            # Simplified: du = -(1/rho) * grad, where rho updated via BFGS
            du = -H_diag * grad.x.array
        else:
            du = -grad.x.array
    else:
        du = -grad.x.array

    # Line search
    alpha = 1.0
    u.x.array[:] += alpha * du

    # Update (s, y) pair
    s_curr = alpha * du
    grad_new = ... # compute new gradient
    y_curr = grad_new - grad.x.array

    s_prev = s_curr
    y_prev = y_curr
```

**Convergence:** Superlinear (between linear and quadratic).

**Use when:** Jacobian is expensive or unavailable.

## Convergence Rate Summary

| Method | Rate | Iterations | Cost per Iter |
|--------|------|-----------|---------------|
| **Picard** | Linear (r < 1) | Many | 1 linear solve |
| **Newton** | Quadratic (r ≈ 2) | Few | 1 Jacobian + 1 solve |
| **Modified Newton** | Superlinear | Few-med | K solves, 1 Jacobian |
| **Quasi-Newton** | Superlinear | Med | 1 gradient eval |
| **Trust region** | Superlinear | Few-med | Multiple solves |
| **L-BFGS** | Superlinear | Med | 1 gradient eval |

**Best choice depends on:**
- **Problem structure:** Smooth → Newton; nonsmooth → Picard
- **Computational cost:** Jacobian expensive → quasi-Newton
- **Robustness needed:** Poor initial guess → Picard-to-Newton, continuation
- **Multiple solutions:** deflation
- **Snap-through behavior:** arc-length method
