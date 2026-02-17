---
name: fem-element-selection
description: |
  Helps select the right finite element type and function space for DOLFINx problems.
  Use when the user asks which element to use, about element types, function space selection,
  Taylor-Hood, P1 vs P2, Nedelec, Raviart-Thomas, or inf-sup stability.
---

# Element Selection Guide

## Quick Reference

| Problem type | Element | `create_function_space` params |
|---|---|---|
| Scalar (Poisson, heat) | Lagrange P1 | `family="Lagrange", degree=1` |
| Scalar (high accuracy) | Lagrange P2 | `family="Lagrange", degree=2` |
| Elasticity (2D) | Lagrange P1 vector | `family="Lagrange", degree=1, shape=[2]` |
| Elasticity (3D) | Lagrange P1 vector | `family="Lagrange", degree=1, shape=[3]` |
| Stokes velocity | Lagrange P2 vector | `family="Lagrange", degree=2, shape=[2]` |
| Stokes pressure | Lagrange P1 | `family="Lagrange", degree=1` |
| DG transport | DG P0/P1 | `family="DG", degree=0` |
| Electromagnetics (H(curl)) | Nedelec | `family="N1curl", degree=1` |
| Mixed flow (H(div)) | Raviart-Thomas | `family="RT", degree=1` |

## Decision Tree

### 1. What quantities are you solving for?

- **Single scalar** (temperature, pressure, concentration) -> Lagrange scalar
- **Vector field** (displacement, velocity) -> Lagrange vector (add `shape=[gdim]`)
- **Mixed system** (velocity + pressure) -> Mixed space via `create_mixed_space`
- **Flux/current** (H(div) or H(curl)) -> Raviart-Thomas or Nedelec

### 2. What polynomial degree?

| Degree | L2 convergence rate | Pros | Cons |
|---|---|---|---|
| P1 (degree=1) | O(h^2) | Fewest DOFs, fast | Less accurate |
| P2 (degree=2) | O(h^3) | Good accuracy | ~4x DOFs vs P1 (2D) |
| P3 (degree=3) | O(h^4) | High accuracy | ~9x DOFs vs P1 (2D) |
| P4 (degree=4) | O(h^5) | Very high accuracy | Expensive, rarely needed |

Rule of thumb: Start with P1 for development, P2 for production accuracy.

### 3. Continuous or Discontinuous?

- **Continuous Galerkin (CG)**: `family="Lagrange"` -- standard choice, values shared at nodes
- **Discontinuous Galerkin (DG)**: `family="DG"` -- values discontinuous across elements, needed for transport/advection with shocks

## Mixed Element Spaces

For coupled problems (Stokes, Darcy, poroelasticity):

```
create_function_space(family="Lagrange", degree=2, shape=[2], name="V")  # velocity
create_function_space(family="Lagrange", degree=1, name="Q")              # pressure
create_mixed_space(space_names=["V", "Q"], name="W")
```

### Inf-Sup Stability (LBB Condition)

Mixed spaces must satisfy the inf-sup condition. Stable pairs:

| Velocity | Pressure | Name | Stable? |
|---|---|---|---|
| P2 | P1 | Taylor-Hood | Yes |
| P1+bubble | P1 | MINI | Yes |
| P1 | P1 | -- | **No** (needs stabilization) |
| P2 | P0 | -- | **No** |

Always use Taylor-Hood (P2/P1) unless you have a specific reason not to.

## Special Element Families

### Nedelec (H(curl))
For electromagnetic problems (curl-conforming):
```
create_function_space(family="N1curl", degree=1, name="V")
```

### Raviart-Thomas (H(div))
For flux-conservative problems (div-conforming):
```
create_function_space(family="RT", degree=1, name="V")
```

### BDM (Brezzi-Douglas-Marini)
Higher-order H(div) elements:
```
create_function_space(family="BDM", degree=1, name="V")
```

## Common Pitfalls

- **Volumetric locking**: P1 elements for nearly incompressible elasticity (nu > 0.49). Fix: use P2 or mixed u/p formulation.
- **Inf-sup violation**: P1/P1 for Stokes. Fix: use P2/P1 Taylor-Hood.
- **Oscillations**: CG for convection-dominated problems. Fix: use DG or add stabilization.
- **Wrong shape**: Forgetting `shape=[gdim]` for vector problems. Elasticity and flow need vector spaces.
