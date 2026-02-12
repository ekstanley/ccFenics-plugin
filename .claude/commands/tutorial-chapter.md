Walk through a DOLFINx tutorial chapter step-by-step using MCP tools.

Arguments: chapter identifier (e.g., "2.3", "heat", "nonlinear-poisson", "robin")

Map the argument to the appropriate skill and execute it interactively:

| Identifier | Skill | Topic |
|---|---|---|
| 1.1, poisson | fem-workflow-poisson | Basic Poisson equation |
| 1.2, complex-poisson | fem-workflow-complex-poisson | Complex-valued Poisson |
| 1.3, nitsche | fem-workflow-nitsche | Nitsche method |
| 1.4, membrane | fem-workflow-membrane | Membrane on curved domain |
| 2.1, 2.2, heat, heat-equation | fem-workflow-heat-equation | Heat equation |
| 2.3, nonlinear, nonlinear-poisson | fem-workflow-nonlinear-poisson | Nonlinear Poisson |
| 2.4, elasticity | fem-workflow-elasticity | Linear elasticity |
| 2.5, navier-stokes, ns | fem-workflow-navier-stokes | Navier-Stokes (IPCS) |
| 2.6, hyperelasticity | fem-workflow-hyperelasticity | Hyperelasticity |
| 2.7, helmholtz | fem-workflow-helmholtz | Helmholtz equation |
| 2.8, adaptive, amr | fem-workflow-adaptive-refinement | Adaptive refinement |
| 2.9, singular | fem-workflow-singular-poisson | Singular Poisson |
| 3.1, mixed-bcs | fem-workflow-mixed-bcs | Mixed BCs |
| 3.2, multiple-dirichlet | fem-workflow-multiple-dirichlet | Multiple Dirichlet BCs |
| 3.3, materials, subdomains | fem-workflow-material-subdomains | Material subdomains |
| 3.4, robin | fem-workflow-robin-bc | Robin BCs |
| 3.5, component-bc | fem-workflow-component-bc | Component-wise BCs |
| 3.6, electromagnetics, maxwell | fem-workflow-electromagnetics | Electromagnetics |
| 4.1, mixed-poisson, rt | fem-workflow-mixed-poisson | Mixed Poisson (RT) |
| 4.2, 4.3, solver-config | fem-workflow-solver-config | Solver configuration |
| 4.5, custom-newton | fem-workflow-custom-newton | Custom Newton solver |

If no argument is provided, list all available chapters and ask the user to choose one.

For each chapter:
1. Explain the problem being solved and its physical context
2. Walk through each tool call step-by-step
3. Explain what each step does and why
4. Execute the tool calls and show results
5. Suggest parameter variations the user can explore
