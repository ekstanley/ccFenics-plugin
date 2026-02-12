Solve a Poisson equation (-div(grad(u)) = f) on a unit square using DOLFINx MCP tools.

Arguments: mesh_size (default 32), degree (default 1)

Execute the following steps using MCP tools, reporting results after each step:

1. Create mesh: `create_unit_square(nx=$ARGUMENTS.mesh_size or 32, ny=$ARGUMENTS.mesh_size or 32, name="poisson_mesh")`

2. Create function space: `create_function_space(family="Lagrange", degree=$ARGUMENTS.degree or 1, name="V")`

3. Set source term: `set_material_properties(name="f", value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])")`

4. Define variational form: `define_variational_form(bilinear="inner(grad(u), grad(v)) * dx", linear="f * v * dx", name="poisson")`

5. Apply boundary condition: `apply_boundary_condition(value="0.0", boundary="True", name="bc_zero")`

6. Solve: `solve(form_name="poisson", solver_type="lu", name="u_h")`

7. Compute error: `compute_error(solution_name="u_h", exact_solution="sin(pi*x[0])*sin(pi*x[1])", error_type="L2")`

8. Export: `export_solution(solution_name="u_h", filename="poisson_solution", format="vtk")`

After completion, report:
- Mesh size and DOF count
- Solver convergence status
- L2 error (expected: ~proportional to h^(p+1))
- Output file location
