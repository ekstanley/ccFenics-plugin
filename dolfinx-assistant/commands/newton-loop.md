# Custom Newton Solver Setup

Interactive command to build and execute a custom Newton-Raphson solver for nonlinear PDEs.

## Step 1: Identify Problem Type

**Question:** What type of nonlinear problem are you solving?

- **Hyperelasticity** (large deformation, neo-Hookean/St. Venant-Kirchhoff)
- **Contact mechanics** (gap constraints, friction)
- **Phase-field** (Allen-Cahn, Cahn-Hilliard)
- **Nonlinear diffusion** (u_t - div(a(u) grad u) = 0)
- **Nonlinear Poisson** (k(u) dependent coefficient)
- **Navier-Stokes** (convection, full NS coupling)
- **Other** (describe below)

**Action:** If your problem isn't listed, provide the residual form F(u; v) you want to solve.

---

## Step 2: Define or Confirm Residual Form

**Question:** Do you have a nonlinear residual form F(u; v) defined, or do you need help?

**Option A: Already have form**
- Provide the UFL expression as string, e.g., `"inner(grad(u), grad(v))*dx - f*v*dx + u**3*v*dx"`
- Confirm domain (entire mesh or subdomain)
- Confirm boundary conditions

**Option B: Need to build form**
- Describe the PDE in words
  - Example: "steady-state heat diffusion with temperature-dependent conductivity k(T) = 1 + T^2"
- Provide any known exact form or reference

**Example forms:**

*Nonlinear Poisson with cubic reaction:*
```python
F = inner(grad(u), grad(v))*dx - f*v*dx + u**3*v*dx
```

*Hyperelasticity (neo-Hookean):*
```python
E = 210e9  # Young's modulus (Pa)
nu = 0.3   # Poisson ratio
mu = E / (2*(1 + nu))
lam = E*nu / ((1+nu)*(1-2*nu))

I = Identity(mesh.geometry.dim)
F_def = I + grad(u)         # deformation gradient
C = F_def.T @ F_def         # right Cauchy-Green
J = det(F_def)

# Strain energy density (neo-Hookean)
psi = (mu/2) * (tr(C) - 3 - 2*ln(J)) + (lam/2) * ln(J)**2

# Residual: -dψ/du : dv
F = derivative(psi * dx, u, TestFunction(V))
```

---

## Step 3: Convergence Criteria

**Question:** What convergence tolerance and maximum iterations?

**Typical values:**

| Problem | atol (absolute) | rtol (relative) | max_iter |
|---------|---|---|---|
| **Poisson-like** | 1e-12 | 1e-10 | 30 |
| **Nonlinear diffusion** | 1e-10 | 1e-8 | 50 |
| **Hyperelasticity** | 1e-10 | 1e-8 | 50 |
| **Navier-Stokes** | 1e-8 | 1e-6 | 100 |

**Recommended:**
- Start conservative: atol=1e-12, rtol=1e-10, max_iter=50
- If diverges: increase max_iter or reduce load step
- If slow: relax atol (e.g., 1e-10)

---

## Step 4: Load Stepping

**Question:** Does your problem benefit from load stepping / continuation?

**When to use load stepping:**

- ✅ Solution jumps abruptly at certain parameter values (bifurcation)
- ✅ Newton diverges from arbitrary initial guess (need to approach slowly)
- ✅ Physical constraint limits (e.g., contact forces must be non-negative)
- ✅ Coupled systems with energy stability requirements

**When NOT needed:**

- ✅ Problem is well-posed and Newton converges from zero initial guess
- ✅ Parameter appears linearly (no multiplicity/bifurcation)

**If YES:**
- **Number of steps?** (e.g., 10, 20, 50)
  - Fewer (5-10): faster but may miss singular points
  - More (50+): slower but more robust
- **Strategy?**
  - Linear: λ ∈ [0, 1] in equal steps (simplest)
  - Adaptive: adjust step size if Newton converges slowly/fast

**Example load-stepping form:**

```python
# Define load factor parameter
lambda_param = fem.Constant(mesh, PETSc.ScalarType(0.0))

# Residual with load factor multiplying body force
F = inner(grad(u), grad(v))*dx - lambda_param*f*v*dx + u**3*v*dx
```

---

## Step 5: Generate Newton Loop

**Action:** Choose generation strategy.

### Option A: Automatic Generation (Recommended)

The system generates a standard Newton loop:

```python
# Parameters
atol = 1e-12
rtol = 1e-10
max_iter = 50
load_steps = 10

# Load factors [0.1, 0.2, ..., 1.0]
load_factor_list = np.linspace(0, 1, load_steps + 1)[1:]

# For each load step
for load_factor in load_factor_list:
    lambda_param.value = load_factor
    print(f"\n=== Load step {load_factor:.2f} ===")

    # Newton iteration
    for it in range(max_iter):
        # Assemble residual
        b = fem.assemble_vector(fem.form(F))
        fem.apply_lifting(b, [fem.form(J)], [bcs])
        b.ghostUpdate(...)

        # Check convergence
        norm_b = b.norm()
        print(f"  Iter {it}: |F| = {norm_b:.6e}")

        if norm_b < atol or (it > 0 and norm_b < rtol * norm_b_prev):
            print("  Converged!")
            break

        norm_b_prev = norm_b

        # Solve Newton step
        A = fem.assemble_matrix(fem.form(J), bcs=bcs)
        A.assemble()

        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.pc.setType("hypre")

        du = A.createVecRight()
        ksp.solve(-b, du)

        u.x.array[:] += du.array

    else:
        print("  Warning: did not converge")
```

### Option B: Custom Variants

**Damped Newton** (if oscillating):
```python
# After computing du:
u.x.array[:] += 0.5 * du.array  # damping factor 0.5
```

**Modified Newton** (reuse Jacobian):
```python
# Assemble Jacobian once every K steps
if (it + 1) % 5 == 0:
    A = fem.assemble_matrix(fem.form(J), bcs=bcs)
    A.assemble()
    ksp.setOperators(A)
```

**Line search** (if sensitive):
```python
# After computing du, implement Armijo backtracking
# See custom-newton-loops/SKILL.md for full code
```

---

## Step 6: Execute and Monitor Convergence

**Action:** Run the Newton loop.

```python
run_custom_code(code=<generated_newton_code>)
```

**What to monitor:**

```
Iteration 0: |F| = 1.234e+00  ← Initial residual
Iteration 1: |F| = 1.200e-01  ← 10x decrease (good)
Iteration 2: |F| = 1.500e-02  ← 8x decrease (quadratic convergence starting)
Iteration 3: |F| = 2.000e-04  ← ~100x decrease (quadratic)
Iteration 4: |F| = 5.000e-09  ← Tiny, converged ✓
```

**Convergence quality:**

| Pattern | Meaning | Action |
|---|---|---|
| Steady 10x decrease | Linear convergence (OK but slow) | May need better preconditioner or damping |
| Exponential decrease (100x, 1e6x) | Quadratic convergence (ideal) | ✓ Continue |
| Sudden drop to near-zero | Abrupt convergence (may indicate ill-posedness) | Check solution physically |
| Stagnation (ratio > 0.9) | Convergence stalled | Increase damping, reduce load step, or restart |
| Divergence (increasing residual) | Newton diverged | Reduce step size, better initial guess, check Jacobian |

---

## Step 7: Handle Convergence Failure

**If Newton diverges or stagnates:**

### Diagnosis

**Symptom:** Residual increases or oscillates

```
Iteration 0: |F| = 1.000e+00
Iteration 1: |F| = 5.000e-01  ← Good
Iteration 2: |F| = 3.000e-01  ← Good
Iteration 3: |F| = 5.000e-01  ← Increased! (bad)
Iteration 4: |F| = 8.000e-01  ← Diverging
```

**Causes and fixes:**

| Cause | Fix |
|---|---|
| **Initial guess too far** | Use smaller load step (reduce from 0.1 to 0.05) |
| **Jacobian inaccurate** | Use analytical Jacobian, not finite difference |
| **Step too large** | Add damping: `u += 0.5*du` instead of `u += du` |
| **Problem is singular** | Add regularization (e.g., small penalty term) |
| **Solver fails silently** | Check KSP convergence: `print(ksp.converged)` |
| **Preconditioner too weak** | Use `-pc_type hypre` or `-pc_type gamg` |

### Adaptive Damping

```python
# Start with damping factor
alpha = 0.5

for it in range(max_iter):
    # Assemble, check convergence...

    # Compute Newton step
    du = solve(A, -b)
    norm_du = du.norm()

    # Try step
    u.x.array[:] += alpha * du.array

    # Recompute residual
    b_new = assemble_vector(fem.form(F))
    norm_b_new = b_new.norm()

    # If residual increased, backtrack
    while norm_b_new > norm_b_prev and alpha > 1e-8:
        u.x.array[:] -= alpha * du.array  # undo
        alpha *= 0.5
        u.x.array[:] += alpha * du.array  # try smaller step
        b_new = assemble_vector(fem.form(F))
        norm_b_new = b_new.norm()

    # Increase damping if converging well
    if norm_b_new < 0.1 * norm_b_prev:
        alpha = min(alpha * 1.2, 1.0)

    norm_b_prev = norm_b_new
```

---

## Step 8: Visualize and Post-Process

**Action:** Plot solution and convergence history.

```python
# Plot solution
plot_solution('u_final')

# Plot convergence history (residual vs iteration)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Residual norm (log scale)
ax1.semilogy(residual_history, 'o-', linewidth=2, markersize=6)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Residual norm |F(u)|')
ax1.set_title('Newton Convergence')
ax1.grid(True, which='both', alpha=0.3)

# Update norm
ax2.semilogy(update_history, 'o-', linewidth=2, markersize=6, color='orange')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Update norm |du|')
ax2.set_title('Step Size')
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('newton_convergence.png')
plt.show()

# Compute error (if reference solution available)
L2_error = compute_error(exact=u_ref, norm_type="L2", function_name="u")
print(f"L2 error vs reference: {L2_error['error_value']:.6e}")
```

---

## Summary Checklist

- [ ] Problem type identified (hyperelasticity, nonlinear diffusion, etc.)
- [ ] Residual form F(u; v) defined or generated
- [ ] Jacobian J = dF/du confirmed (via `derivative`)
- [ ] Convergence criteria chosen (atol, rtol, max_iter)
- [ ] Load stepping enabled if needed (λ from 0 to 1)
- [ ] Newton loop generated and executed
- [ ] Convergence monitored (residual decreasing quadratically)
- [ ] Solution visualized and validated
- [ ] Error computed against reference (if available)

---

## Common Newton Loop Pitfalls

| Issue | Fix |
|---|---|
| **Jacobian computed incorrectly** | Use `J = ufl.derivative(F, u, du)` not hand-coded |
| **BCs not applied to Jacobian** | Include `bcs=bcs` in matrix assembly |
| **Initial guess zero but problem nonlinear** | Use `u.x.array[:] = small_perturbation` |
| **Memory bloat in long loops** | Destroy PETSc objects: `A.destroy()`, `ksp.destroy()` |
| **Load step too large** | Reduce number of steps or use adaptive stepping |
| **Solution register not found** | Register with session: `session.solutions['u'] = u` |

---

## See Also

- **SKILL:** `custom-newton-loops/SKILL.md` — Full implementation guide
- **Reference:** `custom-newton-loops/references/newton-strategies.md` — Arc-length, Picard, quasi-Newton
- **Tool:** `solve_nonlinear()` MCP tool — For simpler cases (built-in SNES)
- **Command:** `/setup-matrix-free` — Configure preconditioner for large systems
