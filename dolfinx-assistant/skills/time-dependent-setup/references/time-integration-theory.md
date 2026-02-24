# Time Integration Theory Reference

## Theta-Method Family

The general theta-method for u_t = F(u):

```
(u^{n+1} - u^n) / dt = θ F(u^{n+1}) + (1-θ) F(u^n)
```

| θ | Name | Order | Stability |
|---|------|-------|-----------|
| 0.0 | Forward Euler | 1 | Conditional: dt ≤ 2/λ_max |
| 0.5 | Crank-Nicolson | 2 | A-stable, may oscillate |
| 1.0 | Backward Euler | 1 | L-stable, strongly damping |

### Weak Form for Heat Equation

PDE: ∂u/∂t - ∇·(κ∇u) = f

**Backward Euler** (θ=1):
```
a(u, v) = inner(u, v)*dx + dt*κ*inner(grad(u), grad(v))*dx
L(v) = inner(u_n, v)*dx + dt*inner(f, v)*dx
```

**Crank-Nicolson** (θ=0.5):
```
a(u, v) = inner(u, v)*dx + 0.5*dt*κ*inner(grad(u), grad(v))*dx
L(v) = inner(u_n, v)*dx - 0.5*dt*κ*inner(grad(u_n), grad(v))*dx + dt*inner(f, v)*dx
```

## BDF Methods (Backward Differentiation Formulas)

### BDF2

```
(3u^{n+1} - 4u^n + u^{n-1}) / (2*dt) = F(u^{n+1})
```

Second-order, A-stable, requires two previous solutions.

### BDF3-BDF6

Higher order but stability regions shrink. BDF methods above order 6 are not A-stable.

## Stability Regions

### A-stability
A method is A-stable if its stability region contains the entire left half-plane. All implicit methods in the theta-family with θ ≥ 0.5 are A-stable.

### L-stability
A-stable + strong damping of stiff components. Backward Euler is L-stable. Crank-Nicolson is NOT L-stable (oscillates on stiff problems).

**Practical implication**: For very stiff problems (large eigenvalue spread), backward Euler is safer than Crank-Nicolson despite lower order.

## Error Estimates

### Spatial error (fixed dt)
Same as steady-state: O(h^{k+1}) in L2 for degree-k elements on smooth solutions.

### Temporal error (fixed h)
- Backward Euler: O(dt)
- Crank-Nicolson: O(dt²)
- BDF2: O(dt²)

### Combined error
```
||u - u_h^n|| ≤ C₁ h^{k+1} + C₂ dt^p
```

where p is the time integration order. Balance errors by choosing dt ~ h^{(k+1)/p}.

### Practical balance

| Element | Backward Euler dt | Crank-Nicolson dt |
|---------|-------------------|-------------------|
| P1 (k=1) | dt ~ h² | dt ~ h |
| P2 (k=2) | dt ~ h³ | dt ~ h^{3/2} |

## Energy Stability Analysis

For the heat equation with backward Euler:

```
||u^{n+1}||² + dt ||∇u^{n+1}||² ≤ ||u^n||² + dt ||f||²
```

This discrete energy inequality guarantees:
1. Solution norm doesn't grow without forcing
2. Energy dissipation is monotone
3. Scheme is unconditionally stable

Check this numerically by computing `inner(u, u)*dx` at each time step.
