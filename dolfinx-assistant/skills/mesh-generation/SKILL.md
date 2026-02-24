---
name: mesh-generation
description: >
  This skill should be used when the user asks about "mesh generation", "Gmsh",
  "mesh import", "mesh refinement", "mesh size", "mesh quality", "structured mesh",
  "unstructured mesh", "boundary layers", "mesh grading", "how many elements",
  "mesh too coarse", "mesh too fine", or needs guidance on creating and managing
  meshes for DOLFINx simulations.
version: 0.1.0
---

# Mesh Generation Guide

The mesh determines solution accuracy, solver performance, and computational cost. A bad mesh wastes time or produces wrong results. This guide covers built-in mesh creation, Gmsh import, and refinement strategies.

## Built-In Meshes

DOLFINx provides simple domain meshers through MCP tools. Use these for standard geometries and testing.

### Available Shapes

| Shape | Tool | Parameters | Cell Types |
|-------|------|-----------|------------|
| Unit square | `create_mesh` shape="unit_square" | nx, ny | triangle, quadrilateral |
| Unit cube | `create_mesh` shape="unit_cube" | nx, ny, nz | tetrahedron, hexahedron |
| Rectangle | `create_mesh` shape="rectangle" | nx, ny, dimensions={width, height} | triangle, quadrilateral |
| Box | `create_mesh` shape="box" | nx, ny, nz, dimensions={x, y, z} | tetrahedron, hexahedron |

### Mesh Sizing Guidelines

| Application | Recommended N | Total Cells (2D) | Notes |
|-------------|--------------|-------------------|-------|
| Quick test | 8-16 | 128-512 | Fast iteration, rough accuracy |
| Development | 32-64 | 2K-8K | Good balance for debugging |
| Production (2D) | 64-256 | 8K-130K | Publication-quality results |
| Production (3D) | 16-64 | 24K-1.5M | Memory becomes the constraint |

**Rule of thumb**: Start with N=16 for setup and debugging. Increase to N=64+ for final results. Always run a convergence study before trusting results.

### Triangle vs Quadrilateral

| Property | Triangles | Quadrilaterals |
|----------|-----------|---------------|
| Mesh flexibility | Better for complex geometry | Needs structured grids |
| Convergence | Standard rates | Can be superconvergent |
| Locking resistance | Good with P2 | Naturally better |
| DOLFINx default | Yes | Supported |

**Default recommendation**: Triangles in 2D, tetrahedra in 3D. They're more flexible and well-tested.

## Gmsh Import

For complex geometries, create meshes in Gmsh and import with `create_custom_mesh`.

### Workflow

1. Create geometry in Gmsh (GUI or Python API)
2. Export as `.msh` format (version 2 or 4)
3. Upload the `.msh` file
4. Import: `create_custom_mesh(name="my_mesh", filename="path/to/mesh.msh")`
5. The tool auto-detects cell tags and facet tags from the Gmsh physical groups

### Gmsh Physical Groups

Gmsh physical groups become MeshTags in DOLFINx:

- **Physical surfaces (2D) / volumes (3D)**: Cell tags for material subdomains
- **Physical lines (2D) / surfaces (3D)**: Facet tags for boundary conditions

Name your physical groups descriptively. The tag integers map directly to `boundary_tag` in `apply_boundary_condition`.

### Common Gmsh Pitfalls

- **Missing physical groups**: DOLFINx ignores untagged entities. Always assign physical groups.
- **Mixed element types**: DOLFINx requires a single cell type per mesh. Don't mix triangles and quads.
- **Very small elements**: Gmsh can create tiny elements at intersections. Set minimum element size.
- **2D mesh in 3D space**: If your 2D mesh lives in a 3D coordinate system, make sure geometric dimension matches.

## Mesh Refinement Strategies

### Uniform Refinement

Use `refine_mesh` to uniformly subdivide all cells. Each refinement roughly quadruples cell count (2D) or octuples it (3D).

| Level | 2D Cells (from N=8) | 3D Cells (from N=4) |
|-------|---------------------|---------------------|
| 0 | 128 | 384 |
| 1 | 512 | 3,072 |
| 2 | 2,048 | 24,576 |
| 3 | 8,192 | 196,608 |

**Use for**: Convergence studies, global accuracy improvement.

### Adaptive Refinement (via custom code)

DOLFINx supports adaptive mesh refinement through cell marking. The workflow:

1. Solve on initial mesh
2. Compute error estimator per cell
3. Mark cells with high error
4. Refine marked cells
5. Transfer solution to new mesh
6. Re-solve

Common error estimators:
- **Residual-based**: Cheap, works for elliptic PDEs
- **Recovery-based (ZZ)**: Better accuracy, needs gradient recovery
- **Goal-oriented**: Targets specific output quantity

### When to Refine

- **Near boundaries**: Boundary layers in flow, stress concentrations
- **Near singularities**: Re-entrant corners, point loads
- **Near interfaces**: Material discontinuities, phase boundaries
- **In regions of interest**: If you care about specific locations, refine there

## Mesh Quality Assessment

Always check mesh quality before solving.

### Mesh Quality Metrics

Call `compute_mesh_quality()` and evaluate:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Quality ratio (min/max vol) | > 0.1 | 0.01 - 0.1 | < 0.01 |
| All volumes positive | Required | — | Inverted elements = invalid |
| Std/mean volume | < 0.5 | 0.5 - 1.0 | > 1.0 |

### Quality Thresholds

- **quality_ratio > 0.5**: Excellent (uniform mesh, good conditioning)
- **quality_ratio 0.1-0.5**: Acceptable (graded mesh, manageable)
- **quality_ratio < 0.1**: Poor — strongly consider remeshing

Use these metrics to predict solver performance:
- Better quality → fewer iterations, faster convergence
- Poor quality → ill-conditioned system, slow solver

### Fixing Bad Meshes

- **Inverted elements**: Re-mesh entirely. Can't fix inverted cells.
- **Highly non-uniform**: Adjust Gmsh mesh size fields or use grading.
- **Too few elements**: Increase N or refine specific regions.
- **Too many elements**: Coarsen or use adaptive approach.

## Submesh Extraction

Use `create_submesh` to extract a region from a tagged mesh:

1. Tag cells with `manage_mesh_tags` or import from Gmsh physical groups
2. Extract: `create_submesh(name="subdomain", tags_name="cell_tags", tag_values=[1])`
3. Create function spaces on the submesh
4. Solve on the subdomain independently

**Use cases**: Domain decomposition, multi-physics coupling, localized analysis.

## Reference Material

For Gmsh scripting examples and advanced meshing techniques, see `references/gmsh-patterns.md`.
