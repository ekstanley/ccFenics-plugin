---
description: Export solution to files and generate plots
allowed-tools: Read, Write
---

Export the current simulation results to files for external visualization and archiving.

## Process

1. **Identify solutions**: Call `get_session_state` to list all available solutions. If $ARGUMENTS specifies a solution name, use it. Otherwise export all solutions.

2. **Generate plots**: For each solution, call `plot_solution` with:
   - Contour plot (default colormap "viridis")
   - Mesh overlay if mesh is coarse enough to see (< 5000 cells)
   - Save to workspace as `{solution_name}_contour.png`

3. **Export to XDMF**: Call `export_solution` with:
   - Format: "xdmf" (compatible with ParaView)
   - Filename: `{solution_name}.xdmf`

4. **Export to VTK**: Call `export_solution` with:
   - Format: "vtk"
   - Filename: `{solution_name}.vtk`

5. **Summary data**: Call `get_solver_diagnostics` and write a JSON summary file containing:
   - Solution name, norm, min/max values
   - Mesh info
   - Solver convergence data
   - Export file paths

6. **Bundle**: Call `bundle_workspace_files` to create a ZIP archive of all exported files.

Report what was exported and the total file sizes.
