# Cylindrical Coordinates Reference

**Version**: 0.1.0

---

## Coordinate System

**Cylindrical coordinates** (r, θ, z):
- **r ≥ 0**: Radial distance from z-axis
- **θ ∈ [0, 2π)**: Azimuthal angle
- **z**: Height (axial)

**Relation to Cartesian** (x, y, z):
```
x = r·cos(θ)
y = r·sin(θ)
z = z

r = √(x² + y²)
θ = atan2(y, x)
```

**In 2D axisymmetric problems**:
- Solve on **meridional plane** (r, z) with r ≥ 0
- 3D solution obtained by rotating 2D solution by θ ∈ [0, 2π]

---

## Differential Operators in Cylindrical

### Gradient

**Scalar field** u(r, θ, z):
```
∇u = (∂u/∂r, 1/r · ∂u/∂θ, ∂u/∂z)
    = (∂u/∂r)·e_r + (1/r)·(∂u/∂θ)·e_θ + (∂u/∂z)·e_z
```

**In 2D axisymmetric** (no θ-dependence):
```
∇u = (∂u/∂r, ∂u/∂z)  ← Just r and z components
```

**Vector field** u = (u_r, u_θ, u_z):
```
∇u = [[∂u_r/∂r,          1/r·∂u_r/∂θ - u_θ/r,  ∂u_r/∂z],
      [∂u_θ/∂r,          1/r·∂u_θ/∂θ + u_r/r,  ∂u_θ/∂z],
      [∂u_z/∂r,          1/r·∂u_z/∂θ,           ∂u_z/∂z]]

For axisymmetric (u_θ = 0, no θ-dep):
∇u = [[∂u_r/∂r,        -u_θ/r,      ∂u_r/∂z],     For u_θ=0:
      [∂u_θ/∂r,        u_r/r,       ∂u_θ/∂z],  → [[∂u_r/∂r,    0,      ∂u_r/∂z],
      [∂u_z/∂r,        1/r·∂u_z/∂θ, ∂u_z/∂z]]      [0,          0,      0],
                                                     [∂u_z/∂r,    0,      ∂u_z/∂z]]
```

### Divergence

**Vector field** u = (u_r, u_θ, u_z):
```
∇·u = 1/r · ∂(r·u_r)/∂r + 1/r · ∂u_θ/∂θ + ∂u_z/∂z
    = (∂u_r/∂r + u_r/r) + (1/r)·∂u_θ/∂θ + ∂u_z/∂z

For axisymmetric:
∇·u = ∂u_r/∂r + u_r/r + ∂u_z/∂z
```

### Curl

**Vector field** u = (u_r, u_θ, u_z):
```
∇×u = (1/r·∂u_z/∂θ - ∂u_θ/∂z,
       ∂u_r/∂z - ∂u_z/∂r,
       1/r·(∂(r·u_θ)/∂r - ∂u_r/∂θ))

For axisymmetric (no θ-dependence, u_θ=0):
∇×u = (0,
       ∂u_r/∂z - ∂u_z/∂r,
       1/r · ∂(r·u_θ)/∂r) = (0, ∂u_r/∂z - ∂u_z/∂r, 0)
```

### Laplacian

**Scalar** u(r, θ, z):
```
∇²u = 1/r · ∂(r·∂u/∂r)/∂r + 1/r² · ∂²u/∂θ² + ∂²u/∂z²
    = ∂²u/∂r² + 1/r · ∂u/∂r + 1/r² · ∂²u/∂θ² + ∂²u/∂z²

For axisymmetric (no θ-dep):
∇²u = ∂²u/∂r² + 1/r · ∂u/∂r + ∂²u/∂z²
```

---

## Strain Tensor in Cylindrical

### Small Strain (Linear Elasticity)

**Definition**: ε_ij = 1/2 · (∂u_i/∂x_j + ∂u_j/∂x_i)

**In cylindrical** for u = (u_r, u_θ, u_z):

```
ε_rr = ∂u_r/∂r
ε_θθ = 1/r · ∂u_θ/∂θ + u_r/r
ε_zz = ∂u_z/∂z
ε_rθ = 1/2 · (∂u_θ/∂r + 1/r · ∂u_r/∂θ - u_θ/r)
ε_rz = 1/2 · (∂u_r/∂z + ∂u_z/∂r)
ε_θz = 1/2 · (∂u_θ/∂z + 1/r · ∂u_z/∂θ)
```

**For axisymmetric** (no θ-dependence, u_θ = 0):

```
ε_rr = ∂u_r/∂r
ε_θθ = u_r/r           ← Azimuthal strain from circumferential stretch
ε_zz = ∂u_z/∂z
ε_rz = 1/2 · (∂u_r/∂z + ∂u_z/∂r)
ε_rθ = 0
ε_θz = 0
```

**Matrix form** (axisymmetric):
```
ε = [[∂u_r/∂r,      0,       1/2·∂u_r/∂z],
     [0,             u_r/r,   0],
     [1/2·∂u_z/∂r,  0,       ∂u_z/∂z]]

Trace: tr(ε) = ∂u_r/∂r + u_r/r + ∂u_z/∂z  (volumetric strain)
```

### Volumetric Strain

```
Δ = div(u) = ∂u_r/∂r + u_r/r + ∂u_z/∂z
```

---

## Stress-Strain Relations (Isotropic Material)

### Hooke's Law

**Linear isotropic elasticity**:
```
σ = 2μ·ε + λ·tr(ε)·I

where μ = shear modulus, λ = Lame parameter
```

**Component form** (axisymmetric):
```
σ_rr = 2μ·ε_rr + λ·tr(ε)
     = 2μ·∂u_r/∂r + λ·(∂u_r/∂r + u_r/r + ∂u_z/∂z)

σ_θθ = 2μ·ε_θθ + λ·tr(ε)
     = 2μ·u_r/r + λ·(∂u_r/∂r + u_r/r + ∂u_z/∂z)

σ_zz = 2μ·ε_zz + λ·tr(ε)
     = 2μ·∂u_z/∂z + λ·(∂u_r/∂r + u_r/r + ∂u_z/∂z)

σ_rz = 2μ·ε_rz = μ·(∂u_r/∂z + ∂u_z/∂r)

σ_rθ = 0, σ_θz = 0 (axisymmetric)
```

### Material Parameters

Common relations:
```
μ = E / (2(1+ν))           (shear modulus from Young's modulus E, Poisson ratio ν)
λ = E·ν / ((1+ν)(1-2ν))    (Lame parameter)

Bulk modulus: K = λ + 2μ/3
Wave speeds: c_p = √((λ + 2μ)/ρ), c_s = √(μ/ρ)
```

---

## Equilibrium Equations in Cylindrical

**Cauchy's equation of motion**: ∇·σ + ρ·f = ρ·a

**In static case** (a = 0):
```
∇·σ + ρ·f = 0

Component-wise:
∂σ_rr/∂r + 1/r·∂σ_rθ/∂θ + ∂σ_rz/∂z + (σ_rr - σ_θθ)/r + ρ·f_r = 0
∂σ_rθ/∂r + 1/r·∂σ_θθ/∂θ + ∂σ_θz/∂z + 2σ_rθ/r + ρ·f_θ = 0
∂σ_rz/∂r + 1/r·∂σ_θz/∂θ + ∂σ_zz/∂z + σ_rz/r + ρ·f_z = 0
```

**For axisymmetric** (no θ-dependence, σ_rθ = σ_θz = 0, f_θ = 0):
```
∂σ_rr/∂r + ∂σ_rz/∂z + (σ_rr - σ_θθ)/r + ρ·f_r = 0
∂σ_rz/∂r + ∂σ_zz/∂z + σ_rz/r + ρ·f_z = 0
```

---

## Integration Measure

### Volume Integral in Cylindrical

```
∫∫∫ f(r, θ, z) dV = ∫₀^2π ∫ ∫ f(r, θ, z) · r dr dθ dz

For axisymmetric (f independent of θ):
                 = 2π · ∫ ∫ f(r, z) · r dr dz
```

**In FEM (2D meridional solve)**:
```
Physical integral = 2π × (Numerical integral in (r,z) with measure r·dr·dz)

In UFL: multiply by x[0] (which is r)
```

### Surface Integral (Axial Boundary)

On boundary z = const (flat face perpendicular to z-axis):
```
∫∫_S f dS = ∫₀^2π ∫₀^r_max f(r, θ) · r dr dθ
          = 2π · ∫₀^r_max f(r) · r dr

In UFL: multiply by x[0] when integrating over lateral boundaries
```

### Surface Integral (Lateral Boundary)

On boundary r = const (cylindrical surface):
```
∫∫_S f dS = ∫₀^2π ∫₀^L f(r, θ, z) · r dθ dz
          = 2π·r · ∫₀^L f(r, z) dz

In UFL: already accounts for arc length ≈ r·dθ; use normal measure ds
```

---

## Common Axisymmetric PDEs

### Axisymmetric Poisson

```
-∇²u = f

Cylindrical: -1/r · d/dr(r · du/dr) - d²u/dz² = f

Weak form: ∫ r·∇u·∇v dr dz = ∫ r·f·v dr dz
```

### Axisymmetric Heat Conduction

```
ρ·c·∂T/∂t - κ∇²T = Q

Cylindrical: ρ·c·∂T/∂t - κ/r · d/dr(r·dT/dr) - κ·d²T/dz² = Q

Weak form: ∫ r·ρ·c·∂T/∂t·v dr dz + ∫ r·κ·∇T·∇v dr dz = ∫ r·Q·v dr dz
```

### Axisymmetric Elasticity

```
∇·σ + ρ·f = 0

with σ = 2μ·ε(u) + λ·div(u)·I, ε = sym(∇u)

Special: azimuthal strain ε_θθ = u_r/r contributes to stress

Weak form: ∫ r·[2μ·ε(u):ε(v) + λ·div(u)·div(v)] dr dz = ∫ r·ρ·f·v dr dz
         + ∫ r·[term from u_r/r strain] dr dz
```

### Axisymmetric Stokes

```
-μ∇²u + ∇p = f
div(u) = 0

Weak form (velocity-pressure):
∫ r·μ·∇u:∇v dr dz - ∫ r·p·div(v) dr dz = ∫ r·f·v dr dz
∫ r·div(u)·q dr dz = 0

With u_θ = 0, u = [u_r, u_z]
```

---

## Special Points and Boundaries

### Axis Singularity (r = 0)

**In 2D meridional (r, z) domain**:
- r = 0 is a line (the z-axis)
- All solutions must be regular there: u(0, z) < ∞, ∇u(0, z) < ∞

**In weak form**:
```
∫ r·f(r,z) dr dz → as r→0, integrand ~ r, so contribution → 0

Consequence: Natural BC at r=0; typically u_r(0,z) = 0 (symmetry)
```

**No explicit Dirichlet BC needed** unless problem physics demands it.

### Outer Boundary (r = r_max)

Standard Dirichlet, Neumann, or Robin BCs apply:
- Dirichlet: u(r_max, z) = u_D
- Neumann: du/dr(r_max, z) = g
- Robin: du/dr + α·u = h

### Axis-Perpendicular Boundaries (z = const)

Standard BCs; integral includes r weighting.

---

## Asymptotic Behavior at Axis

### Scalar Field

For Poisson (-∇²u = f):
```
u(r, z) = u₀(z) + O(r²)  as r→0

du/dr(r, z) = r · d²u₀/dz²(z) / 2 + O(r³)  as r→0

Example: u = r · f(z) satisfies -∇²u = f if f is independent of r
```

### Vector Field (Elasticity)

For axisymmetric elasticity:
```
u_r(r, z) = α(z)·r + O(r³)  as r→0
u_z(r, z) = β(z) + O(r²)    as r→0

Azimuthal strain: ε_θθ = u_r/r = α(z) + O(r²)

Stress divergence remains finite at r=0
```

---

## Verification Checks

### Dimensions

All terms in PDE should have same dimension:
```
[∂²u/∂r²] = [∂²u/∂z²] = [1/r · ∂u/∂r]  ← All have dim(u)/length²
```

### Regularity

Solution must be regular (single-valued) at r=0:
```
u(0, z) is well-defined
∇u(0, z) is O(1) or smaller in magnitude
```

### Material Symmetry

In axisymmetric domain, material properties can depend on r and z but not on θ.

---

## Computational Considerations

### Mesh Near Axis

- Include elements near r = 0 (no singularity to resolve)
- Ratio r_min/r_max should not be too small (no ill-conditioning from very thin elements)
- Uniform mesh in r often works well

### Integration Accuracy

- Factor of r in integrand means accuracy near r = 0 is less critical (integrand → 0 as r → 0)
- Quadrature rules automatically handle r weighting

### Solver Conditioning

- Axisymmetric problems typically better-conditioned than 3D equivalents (smaller system)
- Penalty-type BCs (Nitsche) at r = 0 should use γ consistent with element size

---

## See Also

- **axisymmetric-formulations/SKILL.md** — Practical implementation in DOLFINx
- **/ufl-form-authoring** — UFL syntax for building forms in cylindrical coordinates
- **/pde-cookbook** — Axisymmetric PDE recipes
