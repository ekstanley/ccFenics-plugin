# Weak Form Derivations for 15 PDEs

**Version**: 0.1.0

Each PDE: strong form → multiply by test v → integrate by parts → identify bilinear + linear forms.

---

## 1. Poisson Equation

**Strong**: -∇²u = f

**Multiply by v, integrate**: ∫ -∇²u·v dx = ∫ f·v dx

**IBP on Laplacian**: ∫ ∇u·∇v dx - ∫_{∂Ω} ∇u·n·v ds = ∫ f·v dx

**Weak form**: ∫ ∇u·∇v dx = ∫ f·v dx + ∫_{∂Ω} g·v ds (where g = ∇u·n on Neumann boundary)

**Bilinear**: a(u,v) = ∫ ∇u·∇v dx
**Linear**: L(v) = ∫ f·v dx + ∫ g·v ds

---

## 2. Heat Equation

**Strong**: ∂u/∂t - κ∇²u = f

**Backward Euler** (implicit): (u^{n+1} - u^n)/Δt - κ∇²u^{n+1} = f^{n+1}

**Multiply by v, integrate**: ∫ (u^{n+1} - u^n)·v/Δt dx - κ·∫ ∇²u^{n+1}·v dx = ∫ f·v dx

**IBP on Laplacian**: ∫ (u^{n+1}·v)/Δt dx + κ·∫ ∇u^{n+1}·∇v dx = ∫ f·v dx + ∫ (u^n·v)/Δt dx

**Bilinear**: a(u,v) = ∫ (u·v)/Δt + κ·∇u·∇v dx
**Linear**: L(v) = ∫ f·v + (u^n·v)/Δt dx

---

## 3. Linear Elasticity

**Strong**: ∇·σ + f = 0, σ = 2μ·ε(u) + λ·tr(ε)·I, ε = sym(∇u)

**Multiply by test v (vector), integrate**: ∫ (∇·σ)·v + f·v dx = 0

**IBP on divergence**: -∫ σ:∇v dx + ∫_{∂Ω} σ·n·v ds + ∫ f·v dx = 0

**Substitute σ**: ∫ [2μ·ε(u):ε(v) + λ·tr(ε(u))·tr(ε(v))] dx = ∫ f·v dx + ∫ t·v ds (where t = σ·n)

**Bilinear**: a(u,v) = ∫ [2μ·ε(u):ε(v) + λ·tr(ε(u))·tr(ε(v))] dx
**Linear**: L(v) = ∫ f·v dx + ∫ t·v ds

---

## 4. Stokes Flow

**Strong**: -μ∇²u + ∇p = f, ∇·u = 0

**Weak form** (multiply first by v, second by q):
- ∫ (-μ∇²u)·v + (∇p)·v dx + ∫ f·v dx = 0
- ∫ (∇·u)·q dx = 0

**IBP**:
- μ·∫ ∇u:∇v dx - ∫ p·div(v) dx + ∫ f·v dx = 0
- -∫ div(u)·q dx = 0

**Bilinear**: a(u,v,p,q) = μ·∫ ∇u:∇v - p·div(v) - q·div(u) dx
**Linear**: L(v,q) = ∫ f·v dx

---

## 5. Navier-Stokes

**Strong**: (u·∇)u - μ∇²u + ∇p = f, ∇·u = 0

**Multiply by v, q**:
- ∫ (u·∇)u·v - μ∇²u·v + (∇p)·v + f·v dx = 0
- ∫ div(u)·q dx = 0

**IBP**:
- ∫ (u·∇)u·v dx + μ·∇u:∇v - p·div(v) + f·v dx = 0

**Bilinear** (nonlinear): a(u,v,p,q) = ∫ [(u·∇)u·v + μ·∇u:∇v] - p·div(v) - q·div(u) dx
**Linear**: L(v,q) = ∫ f·v dx

---

## 6. Helmholtz

**Strong**: -∇²u - k²u = f

**Multiply by v, integrate**: ∫ (-∇²u - k²u)·v dx = ∫ f·v dx

**IBP**: ∫ ∇u·∇v dx - k²·∫ u·v dx = ∫ f·v dx + ∫_{∂Ω} (∇u·n)·v ds

**Bilinear**: a(u,v) = ∫ ∇u·∇v - k²·u·v dx
**Linear**: L(v) = ∫ f·v dx + ∫ g·v ds (where g = ∇u·n on boundary)

---

## 7. Advection-Diffusion

**Strong**: ∂u/∂t + b·∇u - ε∇²u = f

**Steady form**: b·∇u - ε∇²u = f

**Multiply by v, integrate**: ∫ b·∇u·v - ε∇²u·v dx = ∫ f·v dx

**IBP on Laplacian**: ∫ b·∇u·v + ε·∇u·∇v dx = ∫ f·v dx + ∫ ε·(∇u·n)·v ds

**Bilinear**: a(u,v) = ∫ b·∇u·v + ε·∇u·∇v dx
**Linear**: L(v) = ∫ f·v dx + ∫ (ε·∇u·n)·v ds

---

## 8. Reaction-Diffusion

**Strong**: ∂u/∂t - D∇²u + R(u) = 0

**Backward Euler**: (u^{n+1} - u^n)/Δt - D∇²u^{n+1} + R(u^{n+1}) = 0

**Multiply by v, integrate**: ∫ (u^{n+1} - u^n)·v/Δt - D∇²u^{n+1}·v + R(u^{n+1})·v dx = 0

**IBP**: ∫ (u^{n+1}·v)/Δt + D·∇u^{n+1}·∇v + R(u^{n+1})·v dx = ∫ (u^n·v)/Δt dx

**Bilinear**: a(u,v) = ∫ (u·v)/Δt + D·∇u·∇v + R'(u)·u·v dx (tangent stiffness, nonlinear)
**Linear**: L(v) = ∫ (u^n·v)/Δt dx

---

## 9. Wave Equation

**Strong**: ∂²u/∂t² - c²∇²u = 0

**Leapfrog** (explicit): (u^{n+1} - 2u^n + u^{n-1})/Δt² = c²∇²u^n

**Rearrange**: u^{n+1} = 2u^n - u^{n-1} + (c·Δt)²∇²u^n

**Multiply by v, integrate**: ∫ (u^{n+1} - 2u^n + u^{n-1})·v/Δt² + c²∇²u^n·v dx = 0

(Typically solved via L² projection or explicit update, not variational assembly)

---

## 10. Biharmonic (Fourth-Order)

**Strong**: ∇⁴u = f, u = u_D, ∇u·n = 0 on ∂Ω

**Via mixed method** (σ = ∇²u):
- ∇·σ + f = 0  →  -∫ ∇σ·∇v dx + ∫ f·v dx = 0
- σ - ∇²u = 0  →  ∫ σ·τ - u·div(div(τ)) dx = 0

**Combined weak form**:
∫ σ·τ + u·div(div(τ)) dx = 0 (first equation)
∫ σ·v + ∇σ·∇v dx = ∫ f·v dx (second equation, after IBP on ∇σ)

**Bilinear**: a(u,σ;v,τ) = ∫ σ·τ + u·div(div(τ)) + σ·v + ∇σ·∇v dx (saddle-point)
**Linear**: L(v,τ) = ∫ f·v dx

---

## 11. Hyperelasticity

**Strong**: ∇·P + ρf = 0, where P = ∂W/∂F (first Piola-Kirchhoff stress)

W(F) = (μ/2)·(J^{-2/3}·tr(F^T·F) - 3) + (λ/2)·(J-1)², J = det(F)

**Weak form** (multiply by v, integrate):
∫ P·∇v dx = ∫ ρf·v dx

**Variational** (at equilibrium, extremize total energy):
δ ∫ [W(∇u) - ρf·u] dx = 0

→ ∫ (∂W/∂∇u):∇v dx = ∫ ρf·v dx  (residual form)

**Nonlinear bilinear**: F(u; v) = ∫ P(∇u):∇v - ρf·v dx = 0
**Tangent stiffness** (for Newton): K(u;w,v) = ∂F/∂u = ∫ S:∇w:∇v dx (fourth-order stiffness tensor)

---

## 12. Mixed Poisson

**Strong**: σ + ∇u = 0, ∇·σ = -f

**Weak form** (multiply first by τ, second by v):
- ∫ σ·τ dx - ∫ u·div(τ) dx + ∫ u_D·τ·n ds = 0
- ∫ div(σ)·v dx = -∫ f·v dx

**Rearrange second**: -∫ σ·∇v dx = -∫ f·v dx

**Combined**:
∫ σ·τ + u·div(τ) + div(σ)·v dx = ∫ u_D·τ·n + f·v dx (saddle-point)

**Bilinear**: a(σ,u;τ,v) = ∫ σ·τ + u·div(τ) + div(σ)·v dx
**Linear**: L(τ,v) = ∫ u_D·τ·n ds + ∫ f·v dx

---

## 13. Maxwell Equations

**Strong** (frequency domain, e^{-i·ω·t} convention):
∇×E - i·ω·μ₀·H = 0
∇×H + i·ω·ε₀·E = J_s

**Eliminate H**: ∇×(∇×E) + ω²·μ₀·ε₀·E = -i·ω·μ₀·J_s

**Weak form** (multiply by v, integrate):
∫ (∇×E)·(∇×v) + ω²·μ₀·ε₀·E·v dx = -i·ω·μ₀·∫ J_s·v dx

(Using ∫ ∇×A·∇×B = ∫ ∇²A·∇²B after IBP, but curl-curl is standard)

**Bilinear**: a(E,v) = ∫ (1/μ₀)·∇×E·∇×v + ω²ε₀·E·v dx
**Linear**: L(v) = -i·ω·μ₀·∫ J_s·v dx

---

## 14. Cahn-Hilliard

**Strong** (fourth-order): ∂c/∂t = ∇·(M·∇(∂W/∂c - λ∇²c))

W(c) = (c²/4)·(1-c)², ∂W/∂c = (c/2)·(1-c)·(1+c)

**Decouple via convex splitting**:
- Explicit: ∂W/∂c^n
- Implicit: -λ∇²c^{n+1}

**Multiply by v, integrate**:
∫ (c^{n+1} - c^n)/Δt·v dx = ∫ ∇·(M·∇(∂W/∂c^n - λ∇²c^{n+1}))·v dx

**IBP on divergence**:
-∫ (M·∇(∂W/∂c^n - λ∇²c^{n+1}))·∇v dx = ∫ (c^{n+1} - c^n)/Δt·v dx

**IBP on ∇²c^{n+1}**:
∫ (c^{n+1}·v)/Δt + M·λ·∇c^{n+1}·∇v dx = ∫ (c^n·v)/Δt + M·(∂W/∂c^n)·v dx

**Bilinear**: a(c,v) = ∫ (c·v)/Δt + M·λ·∇c·∇v dx
**Linear**: L(v) = ∫ (c^n·v)/Δt + M·(∂W/∂c^n)·v dx

---

## 15. Singular Poisson

**Strong**: -∇²u = f in Ω, du/dn = 0 on ∂Ω

**Condition**: ∫_Ω f dx = 0 (compatibility)

**Weak form** (standard):
∫ ∇u·∇v dx = ∫ f·v dx

**Problem**: Rank deficiency (constant u is in nullspace). System singular.

**Solution**: Attach constant nullspace to linear system or impose ∫_Ω u dx = 0 (fix one DOF).

**Bilinear**: a(u,v) = ∫ ∇u·∇v dx
**Linear**: L(v) = ∫ f·v dx
**Solver**: Direct with nullspace, or iterative with nullspace mode

---

## Summary: IBP Pattern

**Standard IBP** (Poisson-like):
```
∫ ∇²u·v dx = -∫ ∇u·∇v dx + ∫_{∂Ω} (∇u·n)·v ds
```

**Time discretization** (Backward Euler):
```
∂u/∂t → (u^{n+1} - u^n)/Δt  (in integral, move u^n to RHS)
```

**Nonlinear terms**:
```
Linearize around current iterate or move to RHS
```

**Mixed formulations**:
```
Use saddle-point structure: ∫ a(u,σ)·b(σ,v) dx
```

---

## Consistency Check

For each weak form, verify:
1. **Linearity**: a(αu₁+βu₂, v) = α·a(u₁,v) + β·a(u₂,v) ✓
2. **Symmetry** (if expected): a(u,v) = a(v,u) ✓
3. **Boundary terms**: Accounted for on all ∂Ω regions ✓
4. **Dimension**: All terms have same [dimension] ✓

---

## See Also

- **pde-cookbook/SKILL.md** — Full recipes for all 15 PDEs
- **/ufl-form-authoring** — UFL syntax for implementing weak forms
