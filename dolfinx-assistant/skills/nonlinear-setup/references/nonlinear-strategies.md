# Advanced Nonlinear Strategies Reference

## Arc-Length Method (Riks Method)

For problems with limit points (snap-through buckling, softening materials) where load-controlled Newton fails.

### Concept

Instead of controlling load λ, parameterize the solution path by arc-length s. Both u and λ are unknowns.

### Augmented system

```
F(u, λ) = 0          (equilibrium)
c(u, λ, s) = 0       (constraint: arc-length)
```

The constraint equation:
```
||Δu||² + (Δλ)² = (Δs)²
```

### DOLFINx implementation sketch

Requires `run_custom_code` — not built into the MCP solver tools.

```python
# Pseudo-code
ds = initial_arc_length
for step in range(max_steps):
    # Predictor: tangent direction
    K = assemble_jacobian(u, lam)
    dF_dlam = assemble_load_vector()
    du_t = solve(K, dF_dlam)

    # Corrector: Newton on augmented system
    for newton_iter in range(max_newton):
        R = residual(u, lam)
        du_R = solve(K, -R)
        du_lam = solve(K, dF_dlam)

        # Solve constraint for dlam
        dlam = -(dot(du_pred, du_R) + ...) / (dot(du_pred, du_lam) + ...)
        du = du_R + dlam * du_lam

        u += du
        lam += dlam

        if norm(R) < tol:
            break
```

## Picard Iteration (Fixed-Point)

Simpler than Newton but only linearly convergent. Useful as a starting strategy before switching to Newton.

### Pattern

For F(u) = 0, linearize by evaluating nonlinear coefficients at the previous iterate:

```
Step n: Solve A(u^{n-1}) u^n = b(u^{n-1})
```

### Navier-Stokes example

Full problem: (u·∇)u + ∇p - ν∇²u = f

Picard linearization: (u^{n-1}·∇)u^n + ∇p^n - ν∇²u^n = f

This is a linear Stokes-like problem at each step.

### Picard-to-Newton switching

1. Run 3-5 Picard iterations (cheap, robust, gets close)
2. Switch to Newton (quadratic convergence from good starting point)

## Continuation Methods

### Natural parameter continuation

Follow solution branch by slowly varying a parameter:
```
λ₁ = 0.0 → λ₂ = 0.1 → ... → λ_N = 1.0
```

Use solution at λ_k as initial guess for λ_{k+1}.

### Pseudo-arclength continuation

Like arc-length but with a different constraint. Better for tracking solution branches near turning points.

### Deflation

Find multiple solutions by "deflating" known solutions from the problem. After finding u₁, modify F:

```
G(u) = F(u) / ||u - u₁||^p
```

Newton on G avoids reconverging to u₁ and finds new solutions.

## Bifurcation Detection

### Indicators

- Determinant of tangent stiffness changes sign → bifurcation point
- Eigenvalue of K crosses zero → stability change
- Newton convergence degrades near bifurcation

### Branch switching

At a bifurcation point, perturb the solution along the bifurcation eigenvector to switch to the other branch.

## Damped Newton Variants

### Backtracking line search

After computing Newton direction d:
```
α = 1.0
while ||F(u + α*d)|| > (1 - c*α) * ||F(u)||:
    α *= 0.5
u += α * d
```

PETSc's `newtonls` does this automatically.

### Trust region

Restrict Newton step to a trust region radius δ:
```
min ||F(u) + J*d||  subject to  ||d|| ≤ δ
```

Adjust δ based on predicted vs actual reduction. PETSc's `newtontr` implements this.
