# PDE Cookbook: 15 Problem Recipes

**Triggers**: "solve PDE", "Poisson", "elasticity", "Stokes", "heat equation", "advection", "Helmholtz", "biharmonic", "mixed"

**Version**: 0.1.0

**Team**: Alpha (Formulation Experts)

## Quick Reference: All 15 PDEs

Each recipe includes: strong form, weak form (UFL), element choice, recommended BCs, solver, and MCP tool sequence.

---

## 1. Poisson Equation

**Strong form**: -∇²u = f in Ω, u = u_D on ∂Ω

**Weak form**: ∫ ∇u·∇v dx = ∫ f·v dx + ∫_∂Ω_N g·v ds

**Elements**: Lagrange degree 1-2 (P1/P2)

**BCs**: Dirichlet + Neumann

**Solver**: Direct (LU)

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)
mark_boundaries(markers=[{"tag": 1, "condition": "True"}], name="bnd")
set_material_properties(name="f", value="1.0")
define_variational_form(bilinear="inner(grad(u), grad(v))*dx",
                        linear="f*v*dx", trial_space="V", test_space="V")
apply_boundary_condition(value=0.0, boundary_tag=1, function_space="V")
solve(solver_type="direct", solution_name="u")
```

---

## 2. Heat Equation (Time-Dependent Diffusion)

**Strong form**: ∂u/∂t - κ∇²u = f, u = u_D on ∂Ω, u(x,0) = u₀(x)

**Weak form** (Backward Euler): (u^{n+1} - u^n)/Δt·v + κ·∇u^{n+1}·∇v = f·v

**Elements**: Lagrange degree 1

**BCs**: Dirichlet (time-varying possible)

**Solver**: Direct per timestep

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="kappa", value="1.0")
dt = 0.01
define_variational_form(
    bilinear="inner(u, v)*dx + dt*kappa*inner(grad(u), grad(v))*dx",
    linear="u_n*v*dx",  # u_n from previous timestep
    trial_space="V", test_space="V")
apply_boundary_condition(value=0.0, boundary="True", function_space="V")
interpolate(expression="sin(pi*x[0])", target_space="V")  # Initial condition
solve_time_dependent(t_end=1.0, dt=dt, solution_name="u_final")
```

---

## 3. Linear Elasticity

**Strong form**: ∇·σ + f = 0, σ = 2μ·ε(u) + λ·div(u)·I, u = u_D on ∂Ω_D

**Weak form**: ∫ [2μ·ε(u):ε(v) + λ·div(u)·div(v)] dx = ∫ f·v dx

**Elements**: Lagrange degree 1 vector (P1-P1 stable for 2D/3D)

**BCs**: Dirichlet (component-wise possible)

**Solver**: Direct or iterative (AMG preconditioner good)

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1, shape=[2])  # shape=[2] for 2D vector field, shape=[3] for 3D — REQUIRED for vector problems
set_material_properties(name="mu", value="1.0")
set_material_properties(name="lambda", value="1.0")
set_material_properties(name="f", value="[0.0, -10.0]")  # Body force
define_variational_form(
    bilinear="2*mu*inner(sym(grad(u)), sym(grad(v)))*dx + lambda*tr(sym(grad(u)))*tr(sym(grad(v)))*dx",
    linear="inner(f, v)*dx",
    trial_space="V", test_space="V")
apply_boundary_condition(value=[0.0, 0.0], boundary="True", function_space="V")
solve(solver_type="direct", solution_name="u_disp")
```

---

## 4. Stokes Flow (Incompressible)

**Strong form**: -μ∇²u + ∇p = f, ∇·u = 0, u = u_D on ∂Ω

**Weak form** (velocity-pressure): ∫ μ·∇u:∇v dx - ∫ p·div(v) dx - ∫ q·div(u) dx = ∫ f·v dx

**Elements**: Taylor-Hood (P2-P1): Lagrange degree 2 for velocity, degree 1 for pressure

**BCs**: Dirichlet velocity (no pressure BC)

**Solver**: Direct or iterative (with Schur complement preconditioner)

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V_u", family="Lagrange", degree=2, shape=[2])
create_function_space(name="V_p", family="Lagrange", degree=1)
create_mixed_space(name="W", subspaces=["V_u", "V_p"])
set_material_properties(name="mu", value="1.0")
set_material_properties(name="f", value="[0.0, 0.0]")
define_variational_form(
    bilinear="mu*inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx",
    linear="inner(f, v)*dx",
    trial_space="W", test_space="W")
apply_boundary_condition(value=[0.0, 0.0], boundary="True", function_space="W", sub_space=0)
solve(solver_type="direct", solution_name="u_p")
```

---

## 5. Navier-Stokes (Steady Incompressible)

**Strong form**: (u·∇)u - μ∇²u + ∇p = f, ∇·u = 0

**Weak form**: ∫ (u·∇)u·v dx + μ·∇u:∇v - p·div(v) - q·div(u) = ∫ f·v

**Nonlinear bilinear term**: (u·∇)u → need `solve_nonlinear()`

**Elements**: Taylor-Hood (P2-P1)

**BCs**: Dirichlet velocity

**Solver**: Newton with iterative inner solver

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V_u", family="Lagrange", degree=2, shape=[2])
create_function_space(name="V_p", family="Lagrange", degree=1)
create_mixed_space(name="W", subspaces=["V_u", "V_p"])
create_function(name="u_guess", function_space="W", expression="0.0")  # Initial guess
set_material_properties(name="mu", value="0.1")
# Residual form (u, p, v, q)
residual = "inner((u[0]*grad(u))[0:2], v)*dx + mu*inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx"
solve_nonlinear(residual=residual, unknown="u_guess", solution_name="u_p_newton")
```

---

## 6. Helmholtz (Acoustic Scattering)

**Strong form**: -∇²u - k²u = f, u = 0 on ∂Ω (or impedance BC)

**Weak form**: ∫ ∇u·∇v - k²u·v dx = ∫ f·v dx

**Elements**: Lagrange degree 1-2 (need ~10 DOF per wavelength λ = 2π/k)

**BCs**: Dirichlet or impedance (du/dn + i·k·u = g)

**Solver**: Direct or iterative (GMRES with ILU)

```python
create_unit_square(name="mesh", nx=30, ny=30)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="k", value="1.0")  # Wave number
set_material_properties(name="f", value="1.0")
define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx - k**2*inner(u, v)*dx",
    linear="f*v*dx",
    trial_space="V", test_space="V")
apply_boundary_condition(value=0.0, boundary="True", function_space="V")
solve(solver_type="direct", solution_name="u_helm")
```

---

## 7. Advection-Diffusion

**Strong form**: ∂u/∂t + b·∇u - ε∇²u = f

**Weak form** (with stabilization): ∫ ε·∇u·∇v + b·∇u·v + τ·(ε∇²u + b·∇u - f)·(b·∇v) dx = ∫ f·v dx

Simplified (no stabilization): ∫ ε·∇u·∇v + b·∇u·v dx = ∫ f·v dx

**Elements**: Lagrange degree 1 (or DG for better advection)

**BCs**: Dirichlet inflow, natural outflow

**Solver**: Direct or iterative

```python
create_unit_square(name="mesh", nx=30, ny=30)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="eps", value="0.01")
set_material_properties(name="b_x", value="1.0")
set_material_properties(name="b_y", value="0.0")
define_variational_form(
    bilinear="eps*inner(grad(u), grad(v))*dx + (b_x*grad(u)[0] + b_y*grad(u)[1])*v*dx",
    linear="0.0*v*dx",
    trial_space="V", test_space="V")
apply_boundary_condition(value=1.0, boundary="x[0] < 1e-14", function_space="V")  # Inflow
solve(solver_type="direct", solution_name="u_adv")
```

---

## 8. Reaction-Diffusion

**Strong form**: ∂u/∂t - D∇²u + R(u) = 0, u(x,0) = u₀

R(u) = nonlinear reaction (e.g., λu(1-u))

**Weak form**: (u^{n+1} - u^n)/Δt·v + D·∇u^{n+1}·∇v + R(u^{n+1})·v = 0

**Elements**: Lagrange degree 1

**Nonlinear**: Need `solve_nonlinear()` per timestep

**Solver**: Newton + iterative inner

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="D", value="0.1")
set_material_properties(name="lambda", value="1.0")
dt = 0.01
# Reaction-diffusion with FitzHugh-Nagumo-like nonlinearity
define_variational_form(
    bilinear="inner(u, v)*dx + dt*D*inner(grad(u), grad(v))*dx",
    linear="u_n*v*dx - dt*lambda*u_n*(1-u_n)*v*dx",  # Simplified
    trial_space="V", test_space="V")
solve_time_dependent(t_end=1.0, dt=dt, solution_name="u_react")
```

---

## 9. Wave Equation

**Strong form**: ∂²u/∂t² - c²∇²u = 0

**Weak form** (explicit time-stepping): (u^{n+1} - 2u^n + u^{n-1})/Δt²·v + c²·∇u^n·∇v = 0

**Elements**: Lagrange degree 1

**Time-stepping**: Explicit (CFL constraint: Δt ≤ h/c)

**Solver**: Direct per timestep

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="c", value="1.0")
dt = 0.01  # Must satisfy CFL
# Use custom code for explicit time-stepping (not directly supported by solve_time_dependent for wave)
run_custom_code(code="""
# Explicit wave equation time-stepping
""")
```

---

## 10. Biharmonic (Fourth-Order)

**Strong form**: ∇⁴u = f, u = u_D, ∇u·n = 0 on ∂Ω

**Weak form** (via mixed method, σ = ∇²u): ∫ σ·τ + u·div(div(τ)) dx = 0, ∫ σ·v + ∇²v·u dx = ∫ f·v

**Elements**: Mixed: Lagrange degree 2 for u, degree 1 for σ (or use higher-degree continuous)

**BCs**: Essential (both u and ∇u·n); or use Morley element for plate problems

**Solver**: Direct

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V_u", family="Lagrange", degree=2)
create_function_space(name="V_sigma", family="Lagrange", degree=1)
create_mixed_space(name="W", subspaces=["V_u", "V_sigma"])
set_material_properties(name="f", value="1.0")
define_variational_form(
    bilinear="inner(sigma, tau)*dx + inner(u, div(div(tau)))*dx + inner(sigma, v)*dx + inner(u, div(div(v)))*dx",
    linear="f*v*dx",
    trial_space="W", test_space="W")
apply_boundary_condition(value=0.0, boundary="True", function_space="W", sub_space=0)
solve(solver_type="direct", solution_name="u_biharm")
```

---

## 11. Hyperelasticity (Neo-Hookean)

**Strong form**: ∇·P + ρf = 0, P = ∂W/∂F (first Piola-Kirchhoff), W = neohookean energy

**Energy**: W = (μ/2)(J^{-2/3}·tr(F^T·F) - 3) + (λ/2)(J-1)²

**Nonlinear**: ∂²W/∂u² defines tangent stiffness for Newton

**Elements**: Lagrange degree 1 vector

**Solver**: Newton with iterative inner

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1, shape=[2])
set_material_properties(name="mu", value="1.0")
set_material_properties(name="lambda", value="1.0")
create_function(name="u_def", function_space="V", expression="0.0")  # Deformation
# Residual (hyperelastic): F = I + grad(u), J = det(F), etc.
residual = "... (complex expression with det, inv, etc.)"
solve_nonlinear(residual=residual, unknown="u_def", solution_name="u_hyperelastic")
```

---

## 12. Mixed Poisson (Flux Variable)

**Strong form**: σ + ∇u = 0, ∇·σ = -f, u = u_D on ∂Ω

**Weak form**: ∫ σ·τ + u·div(τ) dx = ∫ u_D·τ·n ds, ∫ div(σ)·v dx = ∫ f·v dx

**Elements**: Raviart-Thomas (RT) for σ (degree p), DG degree p-1 for u

**BCs**: Essential on u; natural (flux) on σ

**Solver**: Direct or iterative (with Schur)

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V_sigma", family="RT", degree=0)  # "RT" = Raviart-Thomas. DOLFINx also accepts "Raviart-Thomas" as the family string.
create_function_space(name="V_u", family="DG", degree=0)      # DG degree 0
create_mixed_space(name="W", subspaces=["V_sigma", "V_u"])
set_material_properties(name="f", value="1.0")
define_variational_form(
    bilinear="inner(sigma, tau)*dx + inner(u, div(tau))*dx + inner(div(sigma), v)*dx",
    linear="inner(f, v)*dx",
    trial_space="W", test_space="W")
# BC on u (weakly via form)
solve(solver_type="direct", solution_name="sigma_u")
```

---

## 13. Maxwell Equations (Frequency Domain)

**Strong form**: ∇×E - i·ω·μ₀·H = 0, ∇×H + i·ω·ε₀·E = J

Eliminate H: ∇×(1/μ₀·∇×E) + ω²·ε₀·E = -i·ω·J

**Weak form**: ∫ 1/μ₀·∇×E·∇×v + ω²ε₀·E·v dx = ∫ -i·ω·J·v dx

**Elements**: Nedelec (N1curl) for E

**BCs**: Dirichlet on tangential E, or impedance

**Solver**: Direct or iterative (complex system)

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V_E", family="N1curl", degree=1)  # Nedelec
set_material_properties(name="omega", value="1.0")
set_material_properties(name="eps0", value="1.0")
set_material_properties(name="mu0", value="1.0")
set_material_properties(name="J", value="[1.0, 0.0]")  # Current source
define_variational_form(
    bilinear="(1.0/mu0)*inner(curl(u), curl(v))*dx + (omega**2*eps0)*inner(u, v)*dx",
    linear="-1j*omega*inner(J, v)*dx",
    trial_space="V_E", test_space="V_E")
apply_boundary_condition(value=[0.0, 0.0], boundary="True", function_space="V_E")
solve(solver_type="direct", solution_name="E_field")
```

---

## 14. Cahn-Hilliard (Phase Field)

**Strong form**: ∂c/∂t = -∇²(∂W/∂c - λ∇²c) = 0

W(c) = (c²/4)·(1-c)² (double-well potential)

**Decoupled form** (via convex splitting):
- ∂c/∂t = ∇·(M·∇(W'(c^n) - λ∇²c^{n+1}))

**Elements**: Lagrange degree 1

**Time-stepping**: Semi-implicit (W' explicit, Laplacian implicit)

**Solver**: Direct per timestep

```python
create_unit_square(name="mesh", nx=50, ny=50)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="M", value="1.0")  # Mobility
set_material_properties(name="lam", value="0.1")  # Interface param
dt = 0.001
define_variational_form(
    bilinear="M*inner(grad(u), grad(v))*dx + lam*inner(grad(grad(u)), grad(grad(v)))*dx",
    linear="c_n*v*dx - dt*M*inner(grad(c_n**3 - c_n), grad(v))*dx",
    trial_space="V", test_space="V")
solve_time_dependent(t_end=1.0, dt=dt, solution_name="c_final")
```

---

## 15. Singular Poisson (Pure Neumann with Nullspace)

**Strong form**: -∇²u = f in Ω, du/dn = 0 on ∂Ω, ∫_Ω f dx = 0 (compatibility)

**Problem**: System is singular (constant mode in nullspace)

**Solution**: Attach constant nullspace to system

**Elements**: Lagrange degree 1

**Solver**: Direct (handles singular systems) or iterative with nullspace mode

```python
create_unit_square(name="mesh", nx=20, ny=20)
create_function_space(name="V", family="Lagrange", degree=1)
set_material_properties(name="f", value="1.0")  # Note: ∫f dx must = 0
define_variational_form(
    bilinear="inner(grad(u), grad(v))*dx",
    linear="f*v*dx",
    trial_space="V", test_space="V")
# PURE NEUMANN: No apply_boundary_condition() call needed.
# The system is singular — nullspace_mode="constant" handles this.
# CRITICAL: The source term must satisfy compatibility: ∫f dx = 0
# If ∫f dx ≠ 0, the system has no solution.
solve(solver_type="direct", solution_name="u_singular", nullspace_mode="constant")
```

---

## Recipe Summary Table

| # | PDE | Type | Elements | Key Feature | Solver |
|---|---|---|---|---|---|
| 1 | Poisson | Elliptic | P1/P2 | Linear | Direct |
| 2 | Heat | Parabolic | P1 | Time-dependent | Time-step |
| 3 | Elasticity | Elliptic vector | P1 vector | Stress-strain | Direct/AMG |
| 4 | Stokes | Elliptic mixed | P2-P1 | Incompressibility | Direct/Schur |
| 5 | Navier-Stokes | Nonlinear mixed | P2-P1 | Advective term | Newton |
| 6 | Helmholtz | Elliptic | P1 | Wave number | Direct/GMRES |
| 7 | Advection-Diffusion | Mixed | P1 | Stabilization | Direct |
| 8 | Reaction-Diffusion | Nonlinear parabolic | P1 | Nonlinear reaction | Newton/time-step |
| 9 | Wave | Hyperbolic | P1 | CFL condition | Explicit |
| 10 | Biharmonic | Elliptic 4th-order | P2 mixed | High-order | Direct |
| 11 | Hyperelasticity | Nonlinear elliptic | P1 vector | Large deformation | Newton |
| 12 | Mixed Poisson | Mixed elliptic | RT-DG | Flux variable | Direct/Schur |
| 13 | Maxwell | Elliptic vector | Nedelec | Curl-curl | Direct |
| 14 | Cahn-Hilliard | Nonlinear parabolic | P1 | Phase field | Newton/time-step |
| 15 | Singular Poisson | Elliptic singular | P1 | Nullspace | Direct+nullspace |

---

## General Workflow

1. **Identify PDE type** from above catalog (or closest match)
2. **Choose elements** (standard recommendation provided)
3. **Create mesh and spaces** via create_unit_square / create_function_space
4. **Mark boundaries** for different BC regions
5. **Set material properties** via set_material_properties
6. **Define variational form** (weak form provided)
7. **Apply boundary conditions** (type per problem)
8. **Solve** (direct, iterative, nonlinear, or time-dependent)
9. **Validate** (compute_error, convergence study)

---

## See Also

- **weak-form-derivations.md** — Full derivations for all 15 PDEs
- **/ufl-form-authoring** — UFL syntax for building forms
- **/advanced-boundary-conditions** — BC patterns for each problem type
- **command: compose-form.md** — Interactive form builder
