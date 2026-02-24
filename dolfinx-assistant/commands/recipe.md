# PDE Recipe Lookup

**Purpose**: Quick lookup and execution of standard PDE formulations with optimal element choices and boundary condition patterns.

---

## Step 1: Select Your PDE

Which equation are you solving? (Pick one)

- **Poisson** (steady diffusion/electrostatics)
- **Heat Equation** (time-dependent diffusion)
- **Elasticity** (linear or hyperelastic)
- **Stokes** (incompressible creeping flow)
- **Navier-Stokes** (incompressible viscous flow)
- **Helmholtz** (frequency domain waves/acoustics)
- **Advection-Diffusion** (transport with diffusion)
- **Reaction-Diffusion** (pattern formation, Turing)
- **Wave Equation** (time-dependent hyperbolic)
- **Biharmonic** (plate bending, higher-order)
- **Hyperelasticity** (finite deformation elasticity)
- **Mixed Poisson** (mixed form with flux)
- **Maxwell** (electromagnetics, H(curl))
- **Cahn-Hilliard** (phase field, spinodal decomposition)
- **Singular Poisson** (pure Neumann, nullspace)

---

## Step 2: Recipe Card

### Poisson: -∇·(κ∇u) = f

**Strong form**:
```
-∇·(κ∇u) = f  in Ω
u = u_D       on Γ_D (Dirichlet)
-κ∇u·n = q_N  on Γ_N (Neumann)
```

**Weak form**:
```
∫ κ∇u·∇v dx = ∫ f*v dx + ∫ q_N*v ds
```

**Element recommendation**: Lagrange P1 or P2 (u is scalar, continuous)

**Material properties**:
- κ (diffusivity): scalar or function of position
- f (source): scalar or function of position

**Boundary condition types**:
- Dirichlet: `apply_boundary_condition value=u_D boundary='condition'`
- Neumann: Automatic via `linear='q_N*v*ds'` in `define_variational_form`
- Robin: `apply_boundary_condition value='alpha*u + beta' boundary='...'`

**MCP workflow**:
```bash
create_mesh name=mesh ...
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh
set_material_properties name=kappa value=1.0
set_material_properties name=f value=1.0
apply_boundary_condition value=0 boundary='x[0]==0' function_space=V
apply_boundary_condition value=1 boundary='x[0]==1' function_space=V
define_variational_form bilinear='kappa*inner(grad(u), grad(v))*dx' linear='f*v*dx'
solve solution_name=u_solution
plot_solution function_name=u_solution
```

**Expected solution**: Smooth scalar field, max/min set by BCs and f

---

### Heat Equation: ∂u/∂t - κ∇²u = f

**Strong form** (unsteady):
```
∂u/∂t - κ∇²u = f  in Ω × (0,T)
u(x,0) = u_0(x)   (initial condition)
u = u_D(t)        on Γ_D × (0,T)
-κ∇u·n = q_N(t)  on Γ_N × (0,T)
```

**Weak form** (Implicit Euler in time):
```
∫ (u^{n+1} - u^n)/Δt * v dx + ∫ κ∇u^{n+1}·∇v dx = ∫ f^{n+1}*v dx
```

**Element recommendation**: Lagrange P1 (spatial), Implicit Euler or Crank-Nicolson (temporal)

**MCP workflow** (time loop):
```bash
create_mesh name=mesh ...
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh
set_material_properties name=kappa value=1.0
set_material_properties name=f value='sin(pi*x[0])'
apply_boundary_condition value=0 boundary='x[1]==0' function_space=V  # BC
define_variational_form bilinear='inner(u, v)*dx + dt*kappa*inner(grad(u), grad(v))*dx' \
                         linear='inner(u_old, v)*dx + dt*f*v*dx'
# Loop:
for t in range(n_steps):
    solve solution_name=u_n1
    # u_n = u_n1 for next iteration
    # Can export or accumulate solutions
solve_time_dependent t_end=1.0 dt=0.01 time_scheme=backward_euler
```

**Expected solution**: Smooth initial condition decaying (if f small)

---

### Linear Elasticity: -∇·σ(u) = f

**Strong form**:
```
-∇·σ = f_body   in Ω    (balance of momentum)
σ = λ(∇·u)I + 2μ ε(u)   (Hooke's law)
ε(u) = (∇u + ∇u^T)/2   (strain)

u = u_D         on Γ_D
σ·n = t_N       on Γ_N
```

**Weak form**:
```
∫ σ(u):ε(v) dx = ∫ f_body·v dx + ∫ t_N·v ds
```

**Element recommendation**: Lagrange P1 or P2 vector (u is 2D or 3D vector, continuous)

**Material properties**:
- λ, μ (Lamé parameters): Can derive from E (Young), ν (Poisson)
  - μ = E / (2*(1+ν))
  - λ = E*ν / ((1+ν)*(1-2*ν))
- f_body (body force): vector function

**Boundary condition types**:
- Dirichlet (displacement): `apply_boundary_condition value=[u_x, u_y] boundary='...' function_space=V`
- Component-wise: `apply_boundary_condition value=0 boundary='x[0]==0' sub_space=0 function_space=V` (fix x-component)
- Neumann (traction): Automatic via `linear='inner(t_N, v)*ds'`

**MCP workflow**:
```bash
create_mesh name=mesh shape=unit_square nx=20 ny=20
create_function_space name=V family=Lagrange degree=1 shape=[2] mesh_name=mesh
set_material_properties name=E value=1.0   # Young's modulus
set_material_properties name=nu value=0.3  # Poisson ratio
set_material_properties name=mu value='E/(2*(1+nu))'
set_material_properties name=lambda value='E*nu/((1+nu)*(1-2*nu))'
# Fixed left, traction on right
apply_boundary_condition value=[0, 0] boundary='x[0]==0' function_space=V
apply_boundary_condition value='[0.1, 0]' boundary='x[0]==1' function_space=V
define_variational_form bilinear='lambda*inner(div(u), div(v))*dx + 2*mu*inner(sym(grad(u)), sym(grad(v)))*dx' \
                         linear='inner(f_body, v)*dx'
solve solution_name=u_displacement
plot_solution function_name=u_displacement component=0  # Plot x-displacement
```

**Expected solution**: Displacement field u, typically with max ~load magnitude / stiffness

---

### Stokes Flow: -∇·σ(u,p) = f, ∇·u = 0

**Strong form** (creeping flow, inertia negligible):
```
-∇·σ + ∇p = f_body   (momentum)
∇·u = 0              (continuity)
σ = -p*I + μ(∇u + ∇u^T)  (Newtonian fluid)

u = u_D    on Γ_D
p = p_D    on Γ_p (if needed)
σ·n = 0    on Γ_stress-free
```

**Weak form** (mixed u-p):
```
∫ μ∇u:∇v dx - ∫ p (∇·v) dx = ∫ f·v dx
∫ q (∇·u) dx = 0  (divergence-free constraint)
```

**Element recommendation**: **Taylor-Hood P2-P1** (u: vector P2, p: scalar P1)
- Satisfies inf-sup condition
- Robust for all Reynolds numbers

**Material properties**:
- μ (dynamic viscosity)
- f_body (body force, e.g., buoyancy)

**MCP workflow**:
```bash
create_mesh name=mesh shape=rectangle dimensions={'width': 4, 'height': 1} nx=40 ny=10
create_function_space name=V_velocity family=Lagrange degree=2 shape=[2] mesh_name=mesh
create_function_space name=V_pressure family=Lagrange degree=1 mesh_name=mesh
create_mixed_space name=V subspaces=['V_velocity', 'V_pressure']
set_material_properties name=mu value=0.001
apply_boundary_condition value=[1.0, 0] boundary='x[0]==0' sub_space=0 function_space=V  # Inlet: u_x=1, u_y=0
apply_boundary_condition value=[0, 0] boundary='(x[1]==0 or x[1]==1)' sub_space=0 function_space=V  # No-slip
define_variational_form bilinear='mu*inner(grad(u), grad(v))*dx - p*div(v)*dx + q*div(u)*dx' \
                         linear='inner(f_body, v)*dx'
solve solution_name=stokes_solution
```

**Expected solution**: Velocity field u (smooth, no-slip on walls), pressure p (linear drop along flow)

---

### Helmholtz Equation: -∇²u - k²u = f

**Strong form** (time-harmonic waves):
```
-∇²u - k²u = f  in Ω   (k = ω/c = wavenumber)
u = u_D         on Γ_D (Dirichlet)
∂u/∂n - ik*u = 0  on Γ_absorbing (absorbing BC for outgoing waves)
```

**Weak form**:
```
∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
```

**Element recommendation**: Lagrange P1 or P2

**Parameter**: k (wavenumber) = 2π/λ or ω/c

**MCP workflow**:
```bash
create_mesh name=mesh shape=unit_square nx=20 ny=20
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh
set_material_properties name=k value=10  # Wavenumber
set_material_properties name=f value='0'  # Or point source
apply_boundary_condition value=0 boundary='(x[0]==0 or x[0]==1 or x[1]==0 or x[1]==1)' function_space=V
define_variational_form bilinear='inner(grad(u), grad(v))*dx - k**2*inner(u, v)*dx' \
                         linear='f*v*dx'
solve solution_name=u_acoustic
plot_solution function_name=u_acoustic plot_type=contour
```

**Expected solution**: Oscillatory field with wavelength λ = 2π/k

---

### Wave Equation: ∂²u/∂t² - c²∇²u = f

**Strong form** (time-dependent hyperbolic):
```
∂²u/∂t² - c²∇²u = f  in Ω × (0,T)
u(x,0) = u_0(x)
∂u/∂t(x,0) = v_0(x)  (initial velocity)
```

**Weak form** (Newmark or explicit RK):
```
∫ ∂²u/∂t² * v dx + c² ∫ ∇u·∇v dx = ∫ f*v dx
```

**Element recommendation**: Lagrange P1 (spatial), Newmark (temporal, conditionally stable)

**MCP workflow**: (Explicit time-stepping)
```bash
create_mesh name=mesh ...
create_function_space name=V family=Lagrange degree=1 mesh_name=mesh
set_material_properties name=c value=1.0
apply_boundary_condition value=0 boundary='...' function_space=V
# Manual loop (not yet automated in MCP):
solve_time_dependent t_end=1.0 dt=0.01 time_scheme=newmark  # If available
# Else use run_custom_code for custom time loop
```

**Expected solution**: Wave propagation, conserves energy (sum of kinetic + potential)

---

## Step 3: Domain and Mesh Parameters

Ask or provide:
- **Domain shape**: unit_square, rectangle, unit_cube, box, custom (Gmsh)
- **Mesh resolution**: nx, ny, nz (number of cells in each direction)
- **Material parameters**: κ, μ, E, ν, f, etc.
- **Boundary conditions**: Values on Γ_D, Γ_N, etc.

---

## Step 4: Execute Recipe

**Boilerplate for each recipe**:

```bash
# 1. Create mesh
create_mesh name=my_mesh shape={shape} nx={nx} ny={ny}

# 2. Create function space (scalar for Poisson, vector for elasticity, mixed for Stokes)
create_function_space name=V family={family} degree={degree} shape={shape} mesh_name=my_mesh

# 3. Set material properties
set_material_properties name={param} value={value}

# 4. Apply boundary conditions
apply_boundary_condition value={value} boundary={condition} function_space=V

# 5. Define variational form
define_variational_form bilinear='{bilinear_form}' linear='{linear_form}'

# 6. Solve
solve solution_name=u_solution

# 7. Post-process
plot_solution function_name=u_solution plot_type=contour
evaluate_solution function_name=u_solution points=[[0.5, 0.5], ...]
compute_functionals expressions=['inner(u, u)*dx', 'inner(grad(u), grad(u))*dx']
```

---

## Step 5: Visualize and Diagnose

```bash
# Plot scalar solution
plot_solution function_name=u_solution plot_type=contour colormap=viridis

# Plot vector solution (magnitude and components)
plot_solution function_name=u_velocity component=0  # x-component
plot_solution function_name=u_velocity              # magnitude

# Evaluate at points
evaluate_solution function_name=u_solution points=[[0.1, 0.1], [0.5, 0.5]]

# Compute energy or flux
compute_functionals expressions=['inner(grad(u), grad(u))*dx', 'u*u*dx']

# Export for external viz
export_solution filename=solution.xdmf format=xdmf functions=['u_solution']
```

---

## Step 6: Next Steps

Once recipe solve succeeds:

1. **Convergence study**: Run MMS to verify O(h^{p+1}) rates
   - See `commands/setup-mms.md`

2. **Parameter sweep**: Vary material properties (E, κ, μ) and observe solution sensitivity

3. **Mesh refinement**: Increase nx, ny to check stability and convergence

4. **Multi-physics**: Couple with another domain (e.g., thermal load → mechanical)
   - See `skills/multi-physics-coupling/`

5. **Nonlinear version**: Replace bilinear form with residual for Newton's method
   - Use `solve_nonlinear` instead of `solve`

6. **Time-dependent**: Convert to unsteady variant (heat, wave) with `solve_time_dependent`

---

## Quick Reference: Element Families

| PDE | Variable | Scalar/Vector | Family | Degree | Why |
|-----|----------|---------------|--------|--------|-----|
| Poisson | u (scalar) | Scalar | Lagrange | 1–2 | Continuous, standard |
| Elasticity | u (displacement) | Vector | Lagrange | 1–2 | Continuous, symmetric strain |
| Stokes | u, p | Vector + Scalar | Lagrange-DG | 2, 1 | Taylor-Hood stable (inf-sup) |
| Helmholtz | u (scalar) | Scalar | Lagrange | 1–2 | Standard, like Poisson |
| Heat (spatial) | T (temperature) | Scalar | Lagrange | 1–2 | Same as Poisson |
| Mixed Poisson | u (flux), p (pressure) | Vector + Scalar | RT-DG | 1, 0 | H(div) for flux continuity |
| Maxwell | E (E-field) | Vector | Nedelec (N1curl) | 1–2 | H(curl) for E-field |

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Solver diverges | Incompatible BCs or wrong bilinear form | Check form signature, verify BCs set correctly |
| Solution is zero | All BCs are zero + no source | Check f value, verify BC values |
| Solution is constant | Neumann problem without null-space handling | Use `nullspace_mode='constant'` in `solve` |
| NaN in solution | Source f too large or BC conflict | Check material property values, plot initial state |
| Slow convergence | Mesh too coarse or solver tolerance | Refine mesh (increase nx, ny), lower rtol in solve |
| Memory error | Mesh or element degree too high | Use coarser mesh or lower degree element |

