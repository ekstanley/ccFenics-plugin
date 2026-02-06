"""Prompt templates for guided FEniCSx workflows."""

from __future__ import annotations

from .._app import mcp


@mcp.prompt()
def setup_poisson() -> str:
    """Guide through setting up and solving a Poisson equation (-div(grad(u)) = f)."""
    return """You are helping the user solve a Poisson equation using DOLFINx.

The Poisson equation is: -div(grad(u)) = f  on  Omega
with Dirichlet BCs:       u = g           on  dOmega

Follow these steps:

1. **Create the mesh**: Ask about domain shape and resolution.
   Use create_unit_square for [0,1]^2 domains.

2. **Create function space**: Ask about element type and degree.
   Lagrange degree 1 is simplest; degree 2 gives better accuracy.

3. **Define the source term f**: Use set_material_properties.
   Ask the user for f(x,y). Example: f = 2*pi^2*sin(pi*x)*sin(pi*y)

4. **Define variational form**: The weak form is:
   - Bilinear: inner(grad(u), grad(v)) * dx
   - Linear: f * v * dx

5. **Apply boundary conditions**: Ask which boundaries and what values.
   For homogeneous Dirichlet: value=0.0, boundary="True"

6. **Solve**: Use direct solver for small problems.

7. **Post-process**: Compute error if exact solution is known.
   Export to XDMF for visualization.

At each step, explain what is happening physically and mathematically."""


@mcp.prompt()
def debug_convergence() -> str:
    """Help debug a solver that failed to converge."""
    return """You are helping debug a DOLFINx solver convergence failure.

Diagnostic steps:

1. **Check session state**: Use get_session_state to review:
   - Are boundary conditions applied? (essential for well-posedness)
   - Is there at least one BC? (Neumann-only problems need special handling)
   - Do forms reference the correct spaces?

2. **Review the variational form**:
   - Is the bilinear form symmetric and coercive? (required for CG)
   - Does the linear form include all source terms?
   - Are material properties defined before forms that reference them?

3. **Check boundary conditions**:
   - Do BCs cover enough of the boundary?
   - Are BC values compatible with the solution space?

4. **Solver settings**:
   - Direct solver (LU) should always converge for well-posed problems
   - If iterative solver fails, try: increasing max_iter, loosening tolerances,
     changing preconditioner (try "lu" or "hypre")
   - For indefinite systems, use "gmres" instead of "cg"

5. **Common fixes**:
   - Missing BCs -> add boundary conditions
   - Wrong element -> try higher degree or different family
   - Ill-conditioned -> try direct solver first, then iterative with good PC"""


@mcp.prompt()
def setup_elasticity() -> str:
    """Guide through setting up a linear elasticity problem."""
    return """You are helping the user solve a linear elasticity problem using DOLFINx.

The equations: -div(sigma(u)) = f  on Omega
where sigma(u) = lambda * tr(epsilon(u)) * I + 2 * mu * epsilon(u)
and epsilon(u) = 0.5 * (grad(u) + grad(u).T)

Follow these steps:

1. **Create the mesh**: Ask about geometry and resolution.
   Use create_mesh with appropriate shape (rectangle, box).

2. **Create vector function space**: family="Lagrange", degree=1, shape=[gdim]
   where gdim is 2 for 2D or 3 for 3D.

3. **Define material properties**: Ask for Young's modulus (E) and Poisson's ratio (nu).
   Compute Lame parameters: lambda_ = E*nu/((1+nu)*(1-2*nu)), mu = E/(2*(1+nu))
   Use set_material_properties for each.

4. **Define variational form**:
   - Bilinear: inner(sigma(u), epsilon(v)) * dx
   - Where sigma(u) = lambda_*nabla_div(u)*Identity(gdim) + 2*mu*sym(grad(u))
   - Linear: dot(f, v) * dx + dot(t, v) * ds  (body force + traction)

5. **Apply boundary conditions**: Fix displacement on clamped faces.
   Apply traction via Neumann BC in the linear form.

6. **Solve**: Direct solver works well for small-medium problems.

7. **Post-process**: Compute von Mises stress, export displacement field."""


@mcp.prompt()
def setup_stokes() -> str:
    """Guide through setting up a Stokes flow problem."""
    return """You are helping the user solve a Stokes flow problem using DOLFINx.

The equations: -div(2*mu*epsilon(u)) + grad(p) = f  (momentum)
               div(u) = 0                           (incompressibility)

This is a saddle-point problem requiring a mixed function space (velocity + pressure).

Follow these steps:

1. **Create the mesh**: Ask about domain and resolution.

2. **Create function spaces**:
   - Velocity: family="Lagrange", degree=2, shape=[gdim]  (Taylor-Hood)
   - Pressure: family="Lagrange", degree=1
   - Mixed: create_mixed_space combining both

3. **Define variational form** (monolithic):
   - Bilinear: inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx
   - Linear: inner(f, v)*dx
   - Where (u,p) are trial functions and (v,q) are test functions

4. **Apply boundary conditions**:
   - No-slip walls: u = [0,0] on walls
   - Inlet velocity: u = [profile, 0] on inlet
   - Pressure: p = 0 at outlet (or use do-nothing BC)

5. **Solve**: Use direct solver with system_kind="nest" for block structure.
   For iterative: use GMRES with block preconditioner.

6. **Post-process**: Plot velocity magnitude, pressure field, streamlines."""


@mcp.prompt()
def setup_navier_stokes() -> str:
    """Guide through setting up an incompressible Navier-Stokes problem."""
    return """You are helping the user solve incompressible Navier-Stokes using DOLFINx.

The equations: du/dt + (u . grad)u - nu*div(grad(u)) + grad(p) = f
               div(u) = 0

This requires time-stepping and nonlinear solving.

Follow these steps:

1. **Create the mesh**: Ask about domain, resolution, and Reynolds number.

2. **Create function spaces** (Taylor-Hood P2/P1):
   - Velocity: family="Lagrange", degree=2, shape=[gdim]
   - Pressure: family="Lagrange", degree=1

3. **Set material properties**: viscosity nu = 1/Re

4. **Time discretization**: Choose backward Euler or Crank-Nicolson.
   Semi-implicit: linearize convection as (u_n . grad)u

5. **Define variational form**:
   - Bilinear: inner(u/dt, v)*dx + inner(grad(u), grad(v))*nu*dx
     + inner(dot(u_n, nabla_grad(u)), v)*dx - p*div(v)*dx - q*div(u)*dx
   - Linear: inner(u_n/dt, v)*dx + inner(f, v)*dx

6. **Apply boundary conditions**: No-slip, inlet, outlet.

7. **Time loop**: Use solve_time_dependent with appropriate dt and t_end.

8. **Post-process**: Animate velocity field, compute drag/lift coefficients."""


@mcp.prompt()
def convergence_study() -> str:
    """Automate a mesh convergence study."""
    return """You are running a mesh convergence study using DOLFINx.

A convergence study verifies that the numerical solution approaches the exact
solution as the mesh is refined. For element degree p, the expected L2 error
convergence rate is O(h^(p+1)).

Follow these steps:

1. **Define the problem**: Get the exact solution and source term.

2. **Choose mesh sizes**: Use a geometric sequence, e.g., nx = [4, 8, 16, 32, 64].
   The mesh size h ~ 1/nx.

3. **For each mesh size**:
   a. reset_session (clean state)
   b. create_mesh with current nx
   c. create_function_space
   d. Define forms, BCs, material properties
   e. solve
   f. compute_error against exact solution
   g. Record (h, error) pair

4. **Compute convergence rate**: rate = log(e1/e2) / log(h1/h2)
   For P1 elements, expect rate ~ 2.0 in L2 norm.
   For P2 elements, expect rate ~ 3.0 in L2 norm.

5. **Report**: Present a table of (h, error, rate) and verify the rate
   matches the expected O(h^(p+1)) behavior.

If the rate is significantly lower than expected, investigate:
- Boundary condition accuracy
- Source term interpolation quality
- Mesh quality issues"""
