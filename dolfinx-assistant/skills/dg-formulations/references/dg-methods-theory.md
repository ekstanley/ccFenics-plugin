# DG Methods: Theory and Derivation

**Version**: 0.1.0

---

## Interior Penalty Method Derivation

### Starting Problem (Poisson)

```
-∇²u = f   in Ω
  u = 0    on ∂Ω
```

### Standard Galerkin Approach (H1)

Multiply by test v ∈ H1₀, integrate by parts:

```
∫ ∇u·∇v dx = ∫ f·v dx
Ω           Ω
```

### Discontinuous Galerkin Approach

In DG, allow jumps. Element-by-element integration by parts on each K:

```
∑ₖ [∫ₖ ∇u·∇v dx - ∫∂K ∇u·n·v ds] = ∑ₖ ∫ₖ f·v dx
```

Rearrange boundary terms (interior and exterior):

```
∑ₖ ∫ₖ ∇u·∇v dx - ∑ᶠⁱⁿᵗ ∫ᶠ (∇u(+)·n - ∇u(-)·n)·v̄ ds - ∑ᶠᵇᵈ ∫ᶠ ∇u·n·v ds = ∑ₖ ∫ₖ f·v dx
```

where:
- `f^int` = interior facets
- `f^bd` = boundary facets
- `v̄` = average value on interior facet
- `∇u(+)` = gradient on "+" side, `∇u(-)` = gradient on "-" side

### Interior Penalty Discretization

Enforce weak continuity by penalizing jumps:

```
∫ₖ ∇u·∇v dx
+ ∑ᶠⁱⁿᵗ [- ⟨∇u⟩·[v]·n - [u]·⟨∇v⟩·n·ds + (α/hf) [u]·[v] ds]
+ ∑ᶠᵇᵈ [(α/hf) u·v - ∇u·n·v - u·∇v·n] ds
= ∫ f·v dx
```

Where:
- `[u] = u(+) - u(-)`  (jump)
- `⟨∇u⟩ = (∇u(+) + ∇u(-))/2`  (average)
- `α = penalty parameter ≈ C·degree²`
- `hf = element size at facet`

### Bilinear Form (SIPG)

**Symmetric**: Swapping u ↔ v gives same form.

```
a(u, v) = ∫Ω ∇u·∇v dx
        + ∑ᶠⁱⁿᵗ [- ⟨∇u⟩·[v] - [u]·⟨∇v⟩ + (α/hf)[u][v]] dS
        + ∑ᶠᵇᵈ [- ∇u·n·v - u·∇v·n + (α/hf)u·v] ds
```

### Key Properties

| Property | SIPG Form | Proof Sketch |
|---|---|---|
| **Symmetry** | a(u,v) = a(v,u) | Asymmetry terms cancel |
| **Coercivity** | a(u,u) ≥ C\|u\|²_{DG} | Penalty dominates at boundary |
| **Consistency** | a(u_exact, v) - L(v) = 0 (for smooth u_exact) | Interior penalty vanishes for smooth u |
| **Convergence** | \|u - u_h\|_{DG} ≤ C·h^p | Standard error analysis |

---

## Penalty Parameter Theory

### Coercivity Condition

For the bilinear form to be coercive (stable):

```
a(v, v) ≥ γ·||v||²_{DG}   for all v ∈ V_h
```

This requires α to be bounded below:

```
α ≥ C·(degree + 1)²  (3D: C·(degree+1)² + O(degree))
```

### Practical Values

| degree | Minimum α | Recommended α |
|---|---|---|
| 1 | 2 | 10 |
| 2 | 9 | 20 |
| 3 | 16 | 30-40 |

**Formula**: `α = C·degree·(degree + 1)` with `C ∈ [5, 20]`.

### Trade-offs

- **α too small** → Coercivity lost; solver diverges
- **α too large** → Problem well-conditioned but round-off errors increase
- **Sweet spot** → α ≈ 10-20 for most problems

---

## Upwind Numerical Flux

### Scalar Advection Problem

```
∂u/∂t + ∇·(b·u) = 0   (conservative form)
```

### Element-Wise Integration

On element K with velocity b:

```
∫ₖ [∂u/∂t + ∇·(b·u)] v dx = 0

∫ₖ ∂u/∂t·v dx + ∫ₖ ∇·(b·u)·v dx = 0

∫ₖ ∂u/∂t·v dx - ∫ₖ b·u·∇v dx + ∫∂K b·u·v·n ds = 0
```

### Interior Facet Term

On interior facet, flux is discontinuous. Replace with numerical flux u*:

```
∫ᶠ b·u*·v·n ds   (where u* is upwind-biased approximation)
```

Upwind choice (CFL stability):

```
u* = u(-)  if b·n > 0  (flow into element)
u* = u(+)  if b·n < 0  (flow out of element)
```

Conveniently written:

```
u* = u(+)·χ(b·n<0) + u(-)·χ(b·n>0)
   = (u(+) + u(-)) / 2 - 0.5·|b·n|·[u]  (Lax-Friedrichs variant)
```

### Boundary Flux

- **Inflow boundary (b·n < 0)**: Impose u = u_D (Dirichlet)
- **Outflow boundary (b·n > 0)**: Natural (use interior value u)

---

## Lax-Friedrichs Flux

### Definition

For conservation law with flux f(u):

```
∂u/∂t + ∇·f(u) = 0
```

Numerical flux on interior facet:

```
F_LF = (f(u(+)) + f(u(-))) / 2 - λ_max·(u(+) - u(-)) / 2
```

where `λ_max = max(df/du)` (max wave speed).

### For Linear Advection

```
f(u) = b·u   ⟹   λ_max = |b·n|

F_LF = (b·u(+) + b·u(-)) / 2 - (|b·n| / 2)·[u]
```

### Comparison to Upwind

| Property | Upwind | Lax-Friedrichs |
|---|---|---|
| Consistency | Exact | Approximate (adds diffusion) |
| Stability | Requires CFL | Less CFL-restrictive |
| Accuracy | Lower (1st order) | Higher (2nd order, less diffusion) |

---

## DG Error Estimates

### L² Error Bound (Poisson)

For smooth exact solution u, DG solution u_h with degree p:

```
||u - u_h||_{L²} ≤ C·(h^(p+1) + α^(-1/2)·h^p + h_avg^(-1/2)·h^(p+1/2))
                 ≤ C·h^(p+1)    (for α, h_avg = O(1))
```

**Optimal rate**: O(h^(p+1)) in L².

### Energy Norm Error

```
||u - u_h||_{DG} ≤ C·h^p·||u||_{H^(p+1)}
```

where DG norm includes L² plus jump penalization.

### Convergence Table (Poisson with SIPG)

| Mesh | h | L² Error | Rate | H¹ Error | Rate |
|---|---|---|---|---|---|
| Lv 1 | 1/2 | 1.2e-2 | - | 0.43 | - |
| Lv 2 | 1/4 | 3.0e-3 | 2.0 | 0.22 | 1.0 |
| Lv 3 | 1/8 | 7.5e-4 | 2.0 | 0.11 | 1.0 |

(For degree p=1; p=2 shows rate p+1=3 in L²)

---

## hp-Adaptivity with DG

### Hanging Nodes and DG

**Key advantage**: DG **naturally handles hanging nodes** (no special constraint)

- Element can have different degree p locally
- No continuity constraints across boundaries
- Standard assembly works (no mortar spaces needed)

### Error Estimation

Two approaches:

1. **Residual-based**: Estimate via element residual + jump penalization
2. **Gradient recovery**: Compare DG gradient to recovered continuous gradient

### Refinement Indicator

```
η_K = ||f + ∇·grad(u_h)||²_K + ∑_f h_f⁻¹·||[grad(u_h)]||²_f
```

Refine elements with η_K > θ·max(η).

---

## Ghost Penalties for CutFEM

### Problem

In unfitted methods (domain boundary not aligned with mesh), element interiors may be only partially inside domain. This can cause instability.

### Solution: Ghost Penalty

Add penalty on "ghost" facets (element edges on boundary side):

```
G_ghost = γ·h·∑ᶠ⁽¹⁾ ||[∇u·n]||²_f dS   (f^(1) = interior element boundaries)
```

where `γ = O(1)` and `h = element size`.

### Effect

- Stabilizes form for partially-cut elements
- Adds "internal" penalty analogous to interior penalty
- Minimal impact on accuracy (high-order reconstruction still possible)

---

## Comparison: CG vs DG

| Aspect | Continuous Galerkin (CG) | Discontinuous Galerkin (DG) |
|---|---|---|
| **Space** | H¹ (continuous) | L² broken (discontinuous) |
| **DOF count** | O(h⁻ᵈ) | O(p·h⁻ᵈ) (p=degree) |
| **Bilinear form** | Simple (∫∇u·∇v) | Complex (interior penalty) |
| **Penalty param** | None | Needs α tuning |
| **Advection** | Requires stabilization (SUPG) | Stable with upwind flux |
| **Jumps allowed** | No (enforced by space) | Yes (penalized) |
| **hp-adaptivity** | Hanging nodes problematic | Natural |
| **Memory** | Lower | Higher |
| **Assembly** | Faster (volume only) | Slower (volume + dS terms) |
| **Typical use** | Elliptic problems | Hyperbolic, advection-dominated |

---

## Boundary Conditions in DG

### Strong vs Weak Dirichlet

| Aspect | Strong (CG) | Weak (DG) |
|---|---|---|
| **Implementation** | Modify system (eliminate DOFs) | Add penalty terms to form |
| **Accuracy** | Exact on boundary | Penalized error O(α⁻¹) |
| **Robustness** | Less robust for complex BCs | More robust (continuous with problem change) |
| **Assembly** | Simpler | More terms in linear form |

### Weak Dirichlet in DG (on boundary tag)

```
Penalty bilinear:   (α/h)·u·v ds(tag)
Asymmetry bilinear: -∇u·n·v - u·∇v·n ds(tag)

Penalty linear:     (α/h)·u_D·v ds(tag)
Asymmetry linear:   u_D·∇v·n ds(tag)
```

**Verification**: Test on known exact solution; error should decay as O(h^p).

---

## Practical Implementation Checklist

- [ ] Create DG space (family="DG")
- [ ] Compute or set penalty parameter α ≈ 10-20
- [ ] Mark boundaries if Dirichlet BCs present
- [ ] Build bilinear form with interior penalty + boundary penalty terms
- [ ] Build linear form with source + boundary source
- [ ] Define variational form
- [ ] Choose iterative solver with ILU or Jacobi preconditioner (faster than direct for large systems)
- [ ] Solve and check convergence on refined meshes
- [ ] Validate: ||u - u_h||_L² should decrease at rate O(h^(p+1))

---

## References

- **Interior Penalty Methods**: Arnold et al. (2002) "Unified Analysis of DG Methods for Elliptic Problems"
- **Upwind Fluxes**: Cockburn (1998) "An Introduction to the Discontinuous Galerkin Method"
- **hp-Adaptivity**: Houston et al. (2009) "Adaptive DGFEM with automatic error control"

---

## See Also

- **dg-formulations/SKILL.md** — Practical DG setup and examples
- **/ufl-form-authoring** — Jump and average operator details
