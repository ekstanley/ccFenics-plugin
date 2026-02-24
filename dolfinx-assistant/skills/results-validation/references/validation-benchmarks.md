# Validation Benchmarks Reference

## Manufactured Solutions Library

### 2D Scalar (Poisson, Diffusion)

**Smooth sinusoidal** (most common):
- Exact: `sin(pi*x[0])*sin(pi*x[1])`
- Source: `2*pi**2*sin(pi*x[0])*sin(pi*x[1])`
- Domain: Unit square [0,1]²
- BCs: Homogeneous Dirichlet (u=0 on all boundaries)
- Expected L2 rate: k+1, H1 rate: k

**Polynomial**:
- Exact: `x[0]*(1-x[0])*x[1]*(1-x[1])`
- Source: `2*x[1]*(1-x[1]) + 2*x[0]*(1-x[0])`
- Domain: Unit square
- BCs: Homogeneous Dirichlet
- Note: P2+ elements recover this exactly (it's degree 4 in each variable)

**Exponential peak**:
- Exact: `exp(-50*((x[0]-0.5)**2 + (x[1]-0.5)**2))`
- Source: compute via symbolic differentiation
- Tests: gradient resolution near the peak
- Warning: needs fine mesh near center

### 2D Vector (Elasticity)

**Beam bending** (plane stress):
- Exact displacement: derived from Euler-Bernoulli theory
- Domain: Rectangle [0, L] × [0, H]
- BCs: Fixed left end, traction on right
- Validates: Correct stress/strain computation

### 3D Scalar

**3D sinusoidal**:
- Exact: `sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])`
- Source: `3*pi**2*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])`
- Domain: Unit cube
- BCs: Homogeneous Dirichlet

## Classic Benchmarks

### Lid-Driven Cavity (Stokes/Navier-Stokes)

- Domain: Unit square
- BCs: u=(1,0) on top, u=(0,0) on other walls
- Reference: Ghia et al. (1982) — velocity profiles along centerlines
- Re=100: mild, Re=1000: challenging, Re=10000: very challenging
- Check: velocity at center of primary vortex

### Cook's Membrane (Elasticity)

- Domain: Tapered quadrilateral
- Load: Uniform traction on right edge
- Reference: vertical displacement at tip
- Tests: sensitivity to element type (locking detection)

### Channel Flow (Stokes)

- Domain: Rectangle [0, L] × [0, H]
- BCs: Parabolic inlet, zero-traction outlet, no-slip walls
- Exact: Poiseuille flow u = 4*U_max*y*(H-y)/H², v = 0, p = linear
- Tests: correct pressure drop, velocity profile

### NAFEMS Benchmarks (Structural)

- LE1: Thick cylinder under pressure — stress at inner surface
- LE10: Thick plate with hole — stress concentration
- FV32: Cantilevered beam — tip displacement

## Convergence Rate Verification Protocol

### Step-by-Step

1. Pick 4-5 mesh sizes: h = 1/8, 1/16, 1/32, 1/64, 1/128
2. Solve on each mesh with the same element
3. Compute e_h = ||u_exact - u_h|| in L2 and H1
4. For consecutive meshes: rate = log(e_{h1}/e_{h2}) / log(h1/h2)
5. Check asymptotic rate (finest meshes) against theory

### Expected Rates Table

| Element | L2 Rate | H1 Rate | L∞ Rate |
|---------|---------|---------|---------|
| P1 | 2 | 1 | 2 |
| P2 | 3 | 2 | 3 |
| P3 | 4 | 3 | 4 |
| DG0 | 1 | 0.5 | 1 |
| DG1 | 2 | 1 | 2 |
| RT1 | 1 (flux) | 1 (div) | — |

### Troubleshooting Bad Rates

| Observed | Expected | Likely Cause |
|----------|----------|-------------|
| Rate ≈ 0 | k+1 | Bug in formulation or BC application |
| Rate ≈ 1 | 2 (P1) | Corner singularity reducing global rate |
| Rate ≈ k | k+1 | Solution not smooth enough for superconvergence |
| Rate > k+1 | k+1 | Superconvergence (usually OK, verify on different meshes) |
| Rate oscillates | Steady | Pre-asymptotic regime, use finer meshes |
