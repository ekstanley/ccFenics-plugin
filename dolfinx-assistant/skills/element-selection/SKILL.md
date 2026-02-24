---
name: element-selection
description: >
  This skill should be used when the user asks "which element should I use",
  "element type for my problem", "P1 vs P2", "Taylor-Hood", "Raviart-Thomas",
  "Nedelec", "function space selection", "inf-sup stability", or needs guidance
  choosing finite element families and polynomial degrees for DOLFINx simulations.
version: 0.1.0
---

# Finite Element Selection Guide

Match the right element family and polynomial degree to the physics. Wrong choices cause spurious oscillations, locking, or inf-sup instability.

## Decision Flow

1. Identify the PDE type (scalar, vector, mixed, curl-conforming)
2. Check regularity requirements (H1, H(div), H(curl), L2)
3. Pick the element family
4. Choose polynomial degree based on accuracy needs and cost

## Element Families by Problem Type

### Scalar Elliptic (Poisson, Diffusion, Helmholtz)

| Element | DOLFINx Family | Degree | When to Use |
|---------|---------------|--------|-------------|
| Lagrange | `"Lagrange"` | 1-3 | Default for scalar problems. P1 = fast, P2 = accurate |
| DG | `"DG"` | 0-2 | Discontinuous coefficients, advection-dominated, sharp fronts |

**Rule of thumb**: Start with P1 Lagrange. Move to P2 if you need better accuracy or the solution has steep gradients.

### Elasticity and Solid Mechanics

| Element | DOLFINx Family | Shape | When to Use |
|---------|---------------|-------|-------------|
| Vector Lagrange | `"Lagrange"`, shape=[gdim] | P1/P2 | Standard elasticity. P2 avoids volumetric locking |
| DG Vector | `"DG"`, shape=[gdim] | P0/P1 | Contact, fracture, discontinuous displacement |

**Critical**: For nearly incompressible materials (Poisson ratio > 0.49), use P2 elements or a mixed formulation. P1 will lock.

### Incompressible Flow (Stokes, Navier-Stokes)

Must satisfy the inf-sup (LBB) condition. Velocity and pressure spaces can't be chosen independently.

| Pair Name | Velocity | Pressure | Stability |
|-----------|----------|----------|-----------|
| Taylor-Hood | P2 Lagrange, shape=[gdim] | P1 Lagrange | Stable, widely used |
| MINI | P1 + bubble | P1 | Cheaper, less accurate |
| P2-P0 | P2 Lagrange | P0 DG | Stable but poor pressure |
| Scott-Vogelius | Pk, k≥4 on barycentrically refined | Pk-1 DG | Exactly divergence-free |

**Default recommendation**: Taylor-Hood (P2/P1). It's stable, well-tested, and works on triangles and tetrahedra.

### Mixed Formulations (Darcy, Mixed Poisson)

| Element | DOLFINx Family | Space | When to Use |
|---------|---------------|-------|-------------|
| Raviart-Thomas | `"RT"` | H(div) | Flux variable in mixed Poisson, Darcy flow |
| Brezzi-Douglas-Marini | `"BDM"` | H(div) | Higher accuracy flux, more DOFs |
| DG | `"DG"` | L2 | Pressure/scalar in mixed formulation |

**Pairing rule**: RT_k pairs with DG_{k-1}. BDM_k pairs with DG_{k-1}.

### Electromagnetics (Maxwell)

| Element | DOLFINx Family | Space | When to Use |
|---------|---------------|-------|-------------|
| Nedelec (1st kind) | `"N1curl"` | H(curl) | Electric field, curl-curl problems |
| Nedelec (2nd kind) | `"N2curl"` | H(curl) | Higher accuracy, more DOFs |

### Time-Dependent Problems

Use the same element as the steady-state version of your PDE. Time integration is independent of spatial discretization.

## Degree Selection

| Degree | Convergence Rate (L2) | Cost per DOF | Best For |
|--------|----------------------|--------------|----------|
| 1 | O(h²) | Low | Large problems, quick estimates |
| 2 | O(h³) | Medium | Production runs, good accuracy/cost ratio |
| 3 | O(h⁴) | High | High accuracy on coarse meshes |
| 4+ | O(h^{k+1}) | Very high | Spectral-like accuracy, smooth solutions |

**Practical advice**: P2 hits the sweet spot for most engineering problems. P1 if memory-constrained. P3+ only for smooth solutions where you can use coarser meshes.

## Common Mistakes

- **P1/P1 for Stokes**: Violates inf-sup. Produces spurious pressure oscillations.
- **P1 for near-incompressible elasticity**: Volumetric locking. Use P2 or mixed formulation.
- **High-degree on rough solutions**: Wastes DOFs. Use low-order with mesh refinement instead.
- **Lagrange for curl problems**: Wrong function space. Use Nedelec elements for H(curl).
- **Forgetting shape parameter**: Vector problems need `shape=[gdim]` in `create_function_space`.

## Reference Material

For detailed element properties, convergence proofs, and advanced pairings, see `references/element-details.md`.
