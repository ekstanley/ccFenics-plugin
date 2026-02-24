# Element Details Reference

## Lagrange Elements

Standard continuous piecewise polynomial elements. The workhorse of FEM.

**DOLFINx usage**: `create_function_space(name, family="Lagrange", degree=k)`

### Properties by Degree

| Degree | Nodes per Triangle | Nodes per Tet | Continuity | Notes |
|--------|-------------------|---------------|------------|-------|
| 1 | 3 (vertices) | 4 | C0 | Linear interpolation, cheapest |
| 2 | 6 (vertices + edges) | 10 | C0 | Quadratic, good accuracy/cost |
| 3 | 10 | 20 | C0 | Cubic, smooth solutions |

### Convergence Rates (quasi-uniform mesh, smooth solution)

| Degree k | L2 error | H1 error | L∞ error |
|----------|----------|----------|----------|
| 1 | O(h²) | O(h) | O(h²) |
| 2 | O(h³) | O(h²) | O(h³) |
| 3 | O(h⁴) | O(h³) | O(h⁴) |

## DG (Discontinuous Galerkin) Elements

No inter-element continuity. Each cell has independent DOFs.

**DOLFINx usage**: `create_function_space(name, family="DG", degree=k)`

### When DG Wins

- Advection-dominated problems (upwinding is natural)
- Discontinuous coefficients (no Gibbs phenomenon)
- hp-adaptivity (different degrees per cell)
- Conservation (element-wise conserved)

### Cost

More DOFs than CG for the same mesh. DG0 on triangles = 1 DOF/cell. DG1 = 3 DOFs/cell. CG1 = ~1 DOF/vertex (shared).

## Raviart-Thomas (RT)

H(div)-conforming. Normal component is continuous across facets.

**DOLFINx usage**: `create_function_space(name, family="RT", degree=k)`

### Properties

- RT_1: lowest order, 1 DOF per facet (2D), 1 DOF per face (3D)
- Pairs with DG_{k-1} for mixed formulations
- Exactly preserves div(σ) = f pointwise

## Nedelec (N1curl)

H(curl)-conforming. Tangential component is continuous across facets.

**DOLFINx usage**: `create_function_space(name, family="N1curl", degree=k)`

### Properties

- Edge elements: DOFs associated with edges
- Required for electromagnetics (Maxwell equations)
- Do NOT use Lagrange for curl-curl problems

## Inf-Sup Stable Pairs for Stokes/Navier-Stokes

The velocity-pressure pair must satisfy the inf-sup condition. Violating it produces spurious pressure modes.

### Stable Pairs

| Name | Velocity | Pressure | Pros | Cons |
|------|----------|----------|------|------|
| Taylor-Hood | P2 | P1 | Simple, robust, well-tested | Not divergence-free |
| MINI | P1+bubble | P1 | Fewer DOFs | Less accurate |
| Scott-Vogelius | P4+ | P3 DG | Exactly div-free | Needs barycentric refinement |
| Crouzeix-Raviart | P2 (nonconforming) | P1 DG | Good for some problems | Nonconforming analysis needed |

### Unstable Pairs (DO NOT USE)

- P1/P1: Checkerboard pressure oscillations
- P1/P0: Only conditionally stable on specific meshes
- Equal-order without stabilization: Always unstable
