# Library of Manufactured Solutions for Common PDEs

## Poisson Equation: -∇²u = f

### 2D: Trigonometric Solution

**Domain**: Ω = [0,1]²

**u_exact**:
```
u(x,y) = sin(π*x) * sin(π*y)
```

**Verification**:
```
∂u/∂x = π*cos(π*x)*sin(π*y)
∂²u/∂x² = -π²*sin(π*x)*sin(π*y)

∂u/∂y = sin(π*x)*π*cos(π*y)
∂²u/∂y² = -π²*sin(π*x)*sin(π*y)

∇²u = -2π²*sin(π*x)*sin(π*y)
```

**Source term f**:
```
f(x,y) = 2π²*sin(π*x)*sin(π*y)
```

**Boundary conditions**: u = 0 on ∂Ω (automatically satisfied by u_exact)

**Strength**: Simple, smooth, zero BCs (tests Dirichlet)

**Expected convergence** (2D refinement h = 1/N, N = 8,16,32,...):
- P1: L2 ~ O(h²) = O(1/N²), H1 ~ O(h) = O(1/N)
- P2: L2 ~ O(h³) = O(1/N³), H1 ~ O(h²) = O(1/N²)
- P3: L2 ~ O(h⁴), H1 ~ O(h³)

---

### 2D: Non-Homogeneous Dirichlet

**u_exact** (matches non-zero BC):
```
u(x,y) = sin(π*x)*sin(π*y) + (1-x)*(1-y)
```

**∇²u**:
```
∇²(sin(π*x)*sin(π*y)) = -2π²*sin(π*x)*sin(π*y)
∇²((1-x)*(1-y)) = 0  (linear, hence harmonic)

∇²u = -2π²*sin(π*x)*sin(π*y)
```

**Source term f**:
```
f(x,y) = 2π²*sin(π*x)*sin(π*y)
```

**BCs**:
```
u(0,y) = sin(π*y) + (1-y)  ← set via apply_boundary_condition
u(1,y) = (1-y)
u(x,0) = sin(π*x) + (1-x)
u(x,1) = sin(π*x)
```

**Note**: u is smooth across entire domain, but BCs are non-homogeneous. Tests BC handling.

---

### 3D: Trigonometric

**Domain**: Ω = [0,1]³

**u_exact**:
```
u(x,y,z) = sin(π*x) * sin(π*y) * sin(π*z)
```

**∇²u**: (by symmetry)
```
∇²u = -3π²*sin(π*x)*sin(π*y)*sin(π*z)
```

**Source term f**:
```
f(x,y,z) = 3π²*sin(π*x)*sin(π*y)*sin(π*z)
```

**BCs**: u = 0 on ∂Ω

**Strength**: Separable, extends easily to 3D

---

### Polynomial Solution (Higher-Order Verification)

**u_exact** (quartic polynomial):
```
u(x,y) = x²*(1-x)² * y²*(1-y)²
```

**Verification**:
```
u ≥ 0 everywhere, u = 0 on boundary
u is smooth (C∞)

∂u/∂x = 2*x*(1-x)² * y²*(1-y)² + x²*2*(1-x)*(-1) * y²*(1-y)²
      = 2*x*(1-2x)*(1-x)*y²*(1-y)²

∂²u/∂x² = [complex but computable]
Similarly for ∂²u/∂y²
```

**Recommendation**: Use symbolic math (SymPy) to derive f:
```python
import sympy as sp

x, y = sp.symbols('x y')
u = x**2 * (1-x)**2 * y**2 * (1-y)**2

f = -sp.diff(u, x, 2) - sp.diff(u, y, 2)
f = sp.simplify(f)
print(f"f = {f}")  # Prints expanded form
```

**Strength**: Tests higher-order elements (P2, P3) since u ∈ P4

---

## Heat Equation: ∂u/∂t - κ∇²u = f

**Domain**: Ω = [0,1]² × [0, T_final]

### Exponential Decay (Smooth in Time)

**u_exact(x,y,t)**:
```
u(x,y,t) = exp(-λ*t) * sin(π*x) * sin(π*y)
```

where λ is a decay rate (e.g., λ = π²).

**Derivation of f**:
```
∂u/∂t = -λ*exp(-λ*t)*sin(π*x)*sin(π*y)

∇²u = -2π²*exp(-λ*t)*sin(π*x)*sin(π*y)

∂u/∂t - κ∇²u = -λ*exp(-λ*t)*sin(π*x)*sin(π*y) + 2κπ²*exp(-λ*t)*sin(π*x)*sin(π*y)
               = exp(-λ*t)*sin(π*x)*sin(π*y) * (2κπ² - λ)
```

**Source term f**:
```
f(x,y,t) = exp(-λ*t)*sin(π*x)*sin(π*y) * (2κπ² - λ)
```

**With κ = 1 and λ = π²**:
```
f(x,y,t) = exp(-π²*t)*sin(π*x)*sin(π*y) * (2π² - π²) = π²*exp(-π²*t)*sin(π*x)*sin(π*y)
```

**Initial condition**:
```
u(x,y,0) = sin(π*x)*sin(π*y)
```

**BCs**: u = 0 on ∂Ω for all t

**Convergence** (Implicit Euler + P1):
- Spatial: L2 ~ O(h²)
- Temporal: O(Δt) (first order in time)
- Global (space+time): limited by temporal (O(Δt))

**With Crank-Nicolson + P1**:
- Spatial: L2 ~ O(h²)
- Temporal: O(Δt²) (second order in time)
- Global: O(h² + Δt²)

---

### Polynomial in Space, Exponential in Time

**u_exact(x,y,t)**:
```
u(x,y,t) = exp(-2π²*t) * x*(1-x)*y*(1-y)
```

**Derivation**:
```
∂u/∂t = -2π²*exp(-2π²*t)*x*(1-x)*y*(1-y)

∂²u/∂x² = exp(-2π²*t)*[2*(1-2x)*y*(1-y)]
∂²u/∂y² = exp(-2π²*t)*[x*(1-x)*2*(1-2y)]

∇²u = exp(-2π²*t)*[2*(1-2x)*y*(1-y) + x*(1-x)*2*(1-2y)]

f = ∂u/∂t - κ∇²u  (with κ=1)
  = exp(-2π²*t) * [...calculations...]
```

**Strength**: Polynomial spatial part is easier for P1 elements to represent (less oscillation)

---

## Linear Elasticity: -∇·σ = f

### 2D: Polynomial Displacement

**Domain**: Ω = [0,1]² (plane strain)

**u_exact** (displacement vector):
```
u_x(x,y) = x²*(1-x)*y*(1-y)
u_y(x,y) = x*(1-x)*y²*(1-y)
```

**Strain**:
```
ε_xx = ∂u_x/∂x
ε_yy = ∂u_y/∂y
ε_xy = 0.5*(∂u_x/∂y + ∂u_y/∂x)
```

Compute each derivative explicitly (tedious but straightforward).

**Stress** (with Lamé parameters λ, μ):
```
σ_xx = λ*(ε_xx + ε_yy) + 2μ*ε_xx
σ_yy = λ*(ε_xx + ε_yy) + 2μ*ε_yy
σ_xy = 2μ*ε_xy
```

**Body force f = -∇·σ**:
```
f_x = -(∂σ_xx/∂x + ∂σ_xy/∂y)
f_y = -(∂σ_xy/∂x + ∂σ_yy/∂y)
```

**Recommendation**: Use SymPy to compute f symbolically:
```python
import sympy as sp

x, y, lam, mu = sp.symbols('x y lambda mu')

u_x = x**2 * (1-x) * y * (1-y)
u_y = x * (1-x) * y**2 * (1-y)

eps_xx = sp.diff(u_x, x)
eps_yy = sp.diff(u_y, y)
eps_xy = 0.5 * (sp.diff(u_x, y) + sp.diff(u_y, x))

sig_xx = lam*(eps_xx + eps_yy) + 2*mu*eps_xx
sig_yy = lam*(eps_xx + eps_yy) + 2*mu*eps_yy
sig_xy = 2*mu*eps_xy

f_x = -sp.diff(sig_xx, x) - sp.diff(sig_xy, y)
f_y = -sp.diff(sig_xy, x) - sp.diff(sig_yy, y)

print(f"f_x = {sp.simplify(f_x)}")
print(f"f_y = {sp.simplify(f_y)}")
```

**BCs**: u = u_exact on ∂Ω (full Dirichlet)

**Convergence** (P1, 2D refinement):
- L2 ~ O(h²) for each component
- H1 ~ O(h)

---

### 2D: Trigonometric Displacement

**u_exact**:
```
u_x(x,y) = sin(π*x)*sin(π*y)
u_y(x,y) = cos(π*x)*cos(π*y)
```

**Strain**:
```
ε_xx = π*cos(π*x)*sin(π*y)
ε_yy = -π*sin(π*x)*cos(π*y)
ε_xy = 0.5*(π*cos(π*x)*cos(π*y) - π*sin(π*x)*sin(π*y))
```

**Stress** (linear combination of ε):
```
σ_xx = λ(ε_xx + ε_yy) + 2μ*ε_xx
σ_yy = λ(ε_xx + ε_yy) + 2μ*ε_yy
σ_xy = 2μ*ε_xy
```

**Body force f = -∇·σ**: (use SymPy or compute by hand)

**Strength**: Smooth trigonometric terms, clear periodicity

---

## Stokes Flow: -∇·σ(u,p) = f, ∇·u = 0

### 2D: Polynomial Velocity (Divergence-Free)

**Velocity u_exact** (constructed to be divergence-free):
```
u_x(x,y) = sin(π*x)*cos(π*y)
u_y(x,y) = -cos(π*x)*sin(π*y)

Verify: ∇·u = ∂u_x/∂x + ∂u_y/∂y = π*cos(π*x)*cos(π*y) - π*cos(π*x)*cos(π*y) = 0 ✓
```

**Pressure u_exact** (any smooth function; affects f but not velocity):
```
p(x,y) = sin(π*x)*sin(π*y)
```

**Stress σ = -p*I + μ(∇u + ∇u^T)**:
```
σ_xx = -p + 2μ*∂u_x/∂x = -sin(π*x)*sin(π*y) + 2μ*π*cos(π*x)*cos(π*y)
σ_xy = μ*(∂u_x/∂y + ∂u_y/∂x) = μ*[-π*sin(π*x)*sin(π*y) - π*sin(π*x)*sin(π*y)]
```

**Body force f = -∇·σ** (computed from above; tedious)

**Continuity**: ∇·u = 0 (automatically by construction)

---

## Helmholtz Equation: -∇²u - k²u = f

### 2D: Trigonometric

**Domain**: Ω = [0,1]², wavenumber k = π

**u_exact**:
```
u(x,y) = sin(π*x)*sin(π*y)
```

**Derivation**:
```
∇²u = -2π²*sin(π*x)*sin(π*y)
-k²u = -π²*sin(π*x)*sin(π*y)

-∇²u - k²u = 2π²*sin(π*x)*sin(π*y) + π²*sin(π*x)*sin(π*y) = 3π²*sin(π*x)*sin(π*y)
```

**Source term f**:
```
f(x,y) = 3π²*sin(π*x)*sin(π*y)
```

**BCs**: u = 0 on ∂Ω

**Convergence** (P1, 2D):
- L2 ~ O(h²)
- H1 ~ O(h)

---

## Advection-Diffusion: ∂u/∂t + a·∇u - ν∇²u = f

### 1D Steady Advection-Diffusion

**Domain**: Ω = [0,1], advection velocity a = 1, diffusivity ν

**u_exact** (with absorbing BC at x=1):
```
u(x) = (1 - exp(P*x)) / (1 - exp(P))
```

where Péclet number P = a*L/ν = 1/ν.

**Verification**:
```
u'(x) = P*exp(P*x) / (1 - exp(P))
u''(x) = P²*exp(P*x) / (1 - exp(P))

a*u' - ν*u'' = exp(P*x) / (1 - exp(P)) * [P - ν*P²]
```

**Source f**: (if P = 1/ν, then a*P - ν*P² = 1 - 1 = 0, so f = 0)

**With f = 0**:
```
f(x) = 0
```

**BCs**:
```
u(0) = 0  (inlet)
u(1) = 1  (outlet or absorbing)
```

**Strength**: Tests advection-dominated regime (small ν → high P)

---

## Convergence Rate Summary Table

| PDE | Element | Dim | u_exact | L2 Rate | H1 Rate | Notes |
|-----|---------|-----|---------|---------|---------|-------|
| Poisson | P1 | 2D | sin(πx)sin(πy) | 2 | 1 | Classic |
| Poisson | P2 | 2D | sin(πx)sin(πy) | 3 | 2 | Higher order |
| Heat (IE+P1) | P1 | 2D+time | exp(-t)sin(πx)sin(πy) | 2(space), 1(time) | 1 | Time dominates |
| Heat (CN+P1) | P1 | 2D+time | exp(-t)sin(πx)sin(πy) | 2(space), 2(time) | 1 | Balanced |
| Elasticity | P1 | 2D | poly | 2 | 1 | Per component |
| Helmholtz | P1 | 2D | sin(πx)sin(πy) | 2 | 1 | No phase error |
| DG-P1 | DG | 2D | sin(πx)sin(πy) | 2 | 1 | Interior penalty |
| Mixed Poisson | RT-DG | 2D | sin(πx)sin(πy) | 2 | 1 | Flux-based |

---

## Red Flags: When Rates Go Wrong

| Observed Rate | Expected | Likely Cause |
|---------------|----------|-------------|
| 0.5 | 2 | Source f computed wrong; or BC mismatch |
| 1.0 | 2 | Pre-asymptotic regime (mesh too coarse); element order mismatch |
| 1.5 | 2 | Partial assembly bug; some elements refined, others not |
| 2.5 | 2 | Superconvergence (rare, OK); or error norm too weak |
| Negative | anything | Solver diverging; solution not improving; code bug |
| NaN | anything | Source f causes overflow; or BC conflict; check exp(), log() |

---

## Tools for Deriving f Automatically

Use **SymPy** to avoid hand errors:

```python
import sympy as sp

# Define domain symbols and solution
x, y = sp.symbols('x y', real=True)
u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)

# Compute Laplacian
laplacian_u = sp.diff(u, x, 2) + sp.diff(u, y, 2)

# Source term for Poisson: -∇²u = f
f = -laplacian_u

# Simplify
f = sp.simplify(f)

# Convert to MCP format
print(f"set_material_properties name=f value='{f}'")
```

Output (example): `set_material_properties name=f value='2*pi**2*sin(pi*x)*sin(pi*y)'`

Use **SymPy** whenever deriving f to avoid algebraic mistakes.
