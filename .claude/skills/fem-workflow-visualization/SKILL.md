---
name: fem-workflow-visualization
description: |
  Guides the user through visualizing and exporting FEM solutions using DOLFINx MCP tools.
  Use when the user asks about plotting solutions, exporting to VTK/XDMF, visualizing results,
  or saving simulation output for ParaView.
---

# Visualization & Export Workflow

Export FEM solutions for visualization in ParaView, or inspect values programmatically.

## Step-by-Step Tool Sequence

### 1. Solve a Problem First

Ensure a solution exists in the session (e.g., from `solve()` or `solve_nonlinear()`).

### 2. Export Solution to File

Use `export_solution` to write VTK or XDMF files:

```
export_solution(
    solution_name="u_h",
    filename="solution",
    format="vtk"
)
```

Supported formats:
- **VTK** (`.pvd`/`.vtu`): ParaView native format, good for single snapshots
- **XDMF** (`.xdmf`/`.h5`): HDF5-backed, efficient for time series and large meshes

### 3. Plot Solution (In-Container)

Use `plot_solution` for a quick matplotlib visualization:

```
plot_solution(
    solution_name="u_h",
    filename="plot.png",
    title="Poisson Solution"
)
```

This generates a PNG image inside the Docker container.

### 4. Inspect Values at Points

Use `evaluate_solution` to query specific coordinates:

```
evaluate_solution(
    solution_name="u_h",
    points=[[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]]
)
```

Returns the solution value at each point (handles point location automatically).

### 5. Export Time Series

For time-dependent problems, export at multiple timesteps:

```
solve_time_dependent(
    t_end=1.0, dt=0.1,
    output_times=[0.1, 0.5, 1.0]
)
export_solution(solution_name="u_h", filename="heat", format="xdmf")
```

## File Format Guide

| Format | Extension | Best For | Tool |
|---|---|---|---|
| VTK | `.pvd`/`.vtu` | Single snapshots, small meshes | ParaView |
| XDMF | `.xdmf`/`.h5` | Time series, large meshes | ParaView, h5py |
| PNG | `.png` | Quick inspection | Any image viewer |

## ParaView Tips

After exporting:
1. Open the `.pvd` or `.xdmf` file in ParaView
2. Apply a "Warp By Scalar" filter for 2D scalar fields to see elevation
3. Use "Glyph" filter for vector fields (elasticity, flow)
4. Use "Contour" filter for isosurfaces in 3D
