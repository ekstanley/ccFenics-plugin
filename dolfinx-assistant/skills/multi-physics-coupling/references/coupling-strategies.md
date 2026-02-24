# Multi-Physics Coupling Strategies Reference

## Monolithic vs Partitioned Coupling

### Monolithic Coupling
**Approach**: Assemble and solve single large system with all fields coupled.

**Advantages**:
- Strong coupling: stable for tightly-coupled problems
- Quadratic convergence (Newton-like)
- No sub-iteration needed

**Disadvantages**:
- Large system matrix: memory and compute intensive
- Cannot reuse legacy single-physics solvers
- Harder to debug (multiple fields in residual)
- Requires carefully tuned preconditioners

**When to use**: Tightly-coupled, expensive domain changes (ALE, phase change), or when partitioned coupling diverges.

**Implementation** (DOLFINx):
```python
# Mixed function space: u_fluid, p_fluid, u_struct
V_mono = dolfinx.fem.FunctionSpace(mesh, dolfinx.fem.MixedElement([...]))

# Single residual F(w; v) = 0 with w = (u_f, p_f, u_s)
# Assemble and solve
```

---

### Partitioned Coupling
**Approach**: Solve each physics separately, exchange data at interface, iterate until convergence.

**Advantages**:
- Modular: reuse single-physics solvers
- Flexible: different mesh/discr. per domain
- Easier debugging
- Allows different time steps per physics

**Disadvantages**:
- Linear convergence (slower than monolithic)
- May diverge if coupling strong
- Requires iteration loop
- More data transfer overhead

**When to use**: Loosely-coupled problems, legacy solvers, or when domain separation is natural.

**Convergence classification**:
- **Weak coupling** (e.g., one-way thermal→mechanical): Single pass is often sufficient
- **Moderate coupling** (e.g., thermal-elasticity with feedback): 5-20 iterations
- **Strong coupling** (e.g., FSI with incompressible fluid): 50+ iterations or use relaxation

---

## Staggered vs Iterative Schemes

### Staggered Scheme (Implicit in time, decoupled in space)
For time-dependent multi-physics at step n → n+1:

```
t_n → t_{n+1}

FOR k = 1..N_stagger:
  1. Solve Physics A at t_{n+1} using Physics B data from k-1
  2. Solve Physics B at t_{n+1} using Physics A data from step 1

Return u_A(t_{n+1}), u_B(t_{n+1})
```

**Example: Thermal-flow at t_{n+1}**
```
1. Solve heat equation: ρc_p (T^{n+1} - T^n)/Δt + κ∇²T^{n+1} = u^n · ∇T^n
2. Solve Stokes: -∇·σ(u) = f(T^{n+1})
```

**Advantages**: Implicit in time (stable), each solve linear/cheaper

**Disadvantages**: Decoupling error per step (order depends on scheme)

---

### Iterative (Gauss-Seidel) Coupling
For each coupling iteration k at fixed t:

```
FOR k = 1..N_iter:
  1. Solve A with B data from k-1
  2. Solve B with A data from k
  3. Check ||u_k - u_{k-1}|| < tol
     IF converged: break
     ELSE: k ← k+1
```

**Convergence rate**: Depends on coupling strength (eigenvalue of iteration matrix)
- If |λ| < 1: converges linearly with rate |λ|
- If |λ| ≈ 1: very slow (need relaxation)

---

## Relaxation and Acceleration

### Underrelaxation (Damping)
```
u^{k+1} = u^k + ω(u_new^k - u^k)     with ω ∈ (0, 1]
```

**Effect**: Reduces step size, improves stability for strong coupling

**Choose ω**:
- ω = 1.0: No relaxation (fixed-point)
- ω = 0.5: Moderate damping
- ω < 0.2: Heavy damping (very slow but stable)

**Heuristic**: Start with ω = 0.1, increase if converging monotonically

---

### Aitken Acceleration (Dynamic Relaxation)
Adjust ω automatically based on residual sequence.

**Algorithm**:
```
Given r^{k-1}, r^k (residuals), ω^{k-1}:

Δr = r^k - r^{k-1}
ω^k = ω^{k-1} * (ω^{k-1} - 1) / (1 - 2*ω^{k-1} + ω^{k-1}*||Δr||²/||r^k||²)

Clamp ω^k ∈ [0.01, 0.9] for stability
```

**Effect**: Converges in 5-10 iterations instead of 50+ for moderately coupled problems

**Implementation** (DOLFINx):
```python
import numpy as np

u_old = u.copy()
u_new = solve_domain_A(u_old)
residual_old = np.linalg.norm((u_new - u_old).vector)

for k in range(20):
    u_old = u_new.copy()
    u_new = solve_domain_A(u_new)
    residual_new = np.linalg.norm((u_new - u_old).vector)

    if residual_new < tol:
        break

    # Aitken: compute omega_k
    denom = residual_new**2 - 2*residual_new*residual_old + residual_old**2
    if abs(denom) > 1e-14:
        omega = (residual_new - residual_old) / denom * (omega - 1.0)
        omega = np.clip(omega, 0.01, 0.9)

    u_new = u_old + omega * (u_new - u_old)
    residual_old = residual_new
```

---

## Interface Operators

### Point-to-Point Interpolation (Default in MCP)
Direct evaluation and transfer of solution values.

**Use when**: Meshes are conforming or nearly aligned at interface

**DOLFINx MCP call**:
```bash
interpolate target=u_target source_function=u_source source_mesh=mesh_source
```

---

### Mortar Method (Advanced)
Introduce Lagrange multiplier on interface to enforce weak continuity.

**Weak form**: Find (u_A, u_B, λ) s.t. for all (v_A, v_B, μ):
```
a_A(u_A, v_A) + (λ, v_A|_Γ)_Γ = L_A(v_A)
a_B(u_B, v_B) - (λ, v_B|_Γ)_Γ = L_B(v_B)
(μ, u_A - u_B)_Γ = 0                      [interface constraint]
```

**When to use**: Non-conforming meshes, large aspect ratio differences, or when you need weak continuity

**Implementation**: Requires custom UFL and assembly (use `run_custom_code`)

---

### L2-Projection onto Interface
Project solution to interface-aligned space before transfer.

**Example**: Average interface values from both domains
```python
# Extract interface dofs
interface_dofs_A = dolfinx.fem.locate_dofs_topological(space_A, fdim=1, entities=interface_facets)
interface_dofs_B = dolfinx.fem.locate_dofs_topological(space_B, fdim=1, entities=interface_facets)

# Average
u_interface = 0.5 * (u_A.x.array[interface_dofs_A] + u_B.x.array[interface_dofs_B])

# Project back
u_A.x.array[interface_dofs_A] = u_interface
u_B.x.array[interface_dofs_B] = u_interface
```

---

## Common Multi-Physics Combinations

### 1. Thermal-Elasticity (One-way coupling: Thermal → Mechanical)

**Problem**:
- Heat eq: ∇²T = f in Ω
- Elasticity: -∇·σ = f_mech + σ_thermal in Ω

where σ_thermal = λ tr(ε_th) I + 2μ ε_th, and ε_th = α(T - T_ref)I

**Coupling strength**: Weak (T drives ε_th, but ε_th doesn't affect T)

**Scheme**:
1. Solve heat → T
2. Interpolate T onto mechanical mesh
3. Compute σ_thermal and solve elasticity
4. Done (no iteration needed)

**Expected iterations**: 1 pass

---

### 2. Thermoconvection (Two-way: Thermal ↔ Fluid)

**Problem**:
- Heat eq: ρc_p (∂T/∂t + u·∇T) - κ∇²T = Q
- Stokes/NS: -∇·σ(u) = -ρgβ(T - T_ref) e_z

**Coupling strength**: Moderate to strong (buoyancy drives flow, convection enhances cooling)

**Scheme** (time-stepping with staggered sub-iteration):
```
For t_n → t_{n+1}:

  FOR k = 1..5:
    1. Solve Stokes at t_{n+1} with buoyancy from T^{(k-1)}
    2. Solve heat at t_{n+1} with advection from u^{(k)}

  t_{n+1} ← t_n + Δt
```

**Expected iterations**: 3-5 per time step

---

### 3. Fluid-Structure Interaction (Bidirectional: Fluid ↔ Structure)

**Problem**:
- Stokes/NS: -∇·σ_f(u,p) = f in Ω_f
- Elasticity: -∇·σ_s(u_s) = f_s in Ω_s
- Interface: u_f = ∂u_s/∂t (continuity), σ_f·n = σ_s·n (traction)

**Coupling strength**: Strong (flow resistance couples to structural motion)

**Scheme** (monolithic for tight coupling, partitioned for modularity):

**Partitioned (Gauss-Seidel)**:
```
FOR k = 1..20:
  1. Solve fluid with u_f|_Γ = ∂u_s^{(k-1)}/∂t
  2. Extract traction τ = σ_f·n from interface
  3. Solve structure with Neumann BC τ on interface
  4. Update u_s
```

**Expected iterations**: 15-30 (strong coupling, use relaxation)

---

### 4. Electrochemistry (Reaction ↔ Diffusion ↔ Potential)

**Problem**:
- Concentration: ∂c/∂t - D∇²c = -k·c in Ω (reaction term)
- Potential: -∇·(σ∇φ) = F(c) in Ω (c-dependent conductivity or source)
- Coupling: k = f(φ), σ = g(c)

**Coupling strength**: Moderate (explicit time-stepping decouples)

**Scheme** (time-stepping):
```
FOR t_n → t_{n+1}:
  1. Evaluate k, σ at c^n, φ^n
  2. Solve concentration with implicit Euler + k^n
  3. Solve potential with σ^n
  4. t ← t + Δt
```

**Expected iterations**: 1 per time step (explicit coupling)

---

## Convergence Criteria for Iterative Coupling

### Residual-based
```
||u^{k} - u^{k-1}||_L2 / ||u^{k}||_L2 < tol_rel
```

Compute using:
```bash
compute_functionals expressions=['inner(u_new - u_old, u_new - u_old)*dx', 'inner(u_new, u_new)*dx']
```

### Relative energy error
```
|E^{k} - E^{k-1}| / |E^{k}| < tol_energy
```

where E = kinetic + potential energy (problem-dependent)

### Absolute residual
```
||F(u^{k})||_L2 < tol_abs
```

where F is the coupled residual (strong form)

### Practical choice
- **Loose coupling** (weak interaction): use 1e-3 relative residual, 3-5 iterations
- **Moderate coupling**: use 1e-4, 10-15 iterations
- **Strong coupling**: use 1e-5 + Aitken relaxation, 20-50 iterations

---

## Entity Maps and Mesh Relationships

**Purpose**: Track which cells/facets in submesh correspond to parent mesh.

**Auto-created by `create_submesh`**: Returns entity_map with:
- `parent_cell_map`: submesh cell k → parent mesh cell parent_cell_map[k]
- `parent_facet_map`: submesh facet j → parent mesh facet parent_facet_map[j]

**Use case**:
- Transfer boundary tags from parent to submesh
- Identify interface facets (boundary of submesh that's interior to parent)
- Map post-processing results back to parent mesh

**Example**:
```python
# Find interface: facets in submesh that touch parent interior
interface_facets = []
for j in range(submesh.topology.num_entities(fdim)):
    parent_j = entity_map.parent_facet_map[j]
    parent_cells = mesh.topology.incident_entities(tdim, parent_j)
    if len(parent_cells) == 2:  # Interior facet in parent
        interface_facets.append(j)
```

---

## Performance Scaling

| Coupling Type | N_iter (typical) | Memory | CPU (relative) | Stability |
|---------------|------------------|--------|----------------|-----------|
| Monolithic | 1 | 4x | 100x (direct solve) | Excellent |
| Partitioned (no relax) | 20-50 | 2x | 20x (iterative) | Moderate |
| Partitioned + Aitken | 5-10 | 2x | 5x (iterative) | Good |
| Staggered (implicit) | 1/step | 1x | 1x | Good (time-dependent) |
| One-way | 1 | 1x | 1x | Excellent |

