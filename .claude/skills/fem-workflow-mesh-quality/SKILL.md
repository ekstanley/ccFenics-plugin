---
name: fem-workflow-mesh-quality
description: |
  Guides the user through analyzing mesh quality and inspecting mesh properties using DOLFINx MCP tools.
  Use when the user asks about mesh quality, element quality, mesh statistics, aspect ratio,
  or diagnosing mesh-related solver issues.
---

# Mesh Quality Analysis Workflow

Inspect mesh properties and diagnose quality issues that affect solver performance.

## Step-by-Step Tool Sequence

### 1. Create or Load a Mesh

```
create_unit_square(name="mesh", nx=16, ny=16)
```

### 2. Get Basic Mesh Info

Use `get_mesh_info` to inspect mesh properties:

```
get_mesh_info(mesh_name="mesh")
```

Returns: cell type, number of cells, number of vertices, geometric/topological dimensions, bounding box.

### 3. Compute Mesh Quality Metrics

Use `compute_mesh_quality` for detailed quality analysis:

```
compute_mesh_quality(mesh_name="mesh")
```

Returns per-element quality metrics:
- **Aspect ratio**: Ratio of circumradius to inradius (ideal = 1 for equilateral triangle)
- **Minimum angle**: Smallest angle in each element (should be > 20 degrees)
- **Skewness**: How far from ideal shape (0 = perfect, 1 = degenerate)

### 4. Diagnose Solver Issues

Poor mesh quality manifests as:
- **Ill-conditioned stiffness matrix** → slow iterative convergence
- **Large aspect ratios** → inaccurate gradient approximation
- **Degenerate elements** → singular Jacobian, solver failure

### 5. Remediation

If quality is poor:
- Increase mesh resolution in problem areas
- Use adaptive refinement (`mark_boundaries` + `create_submesh`)
- Switch to quadrilateral elements for rectangular domains

## Quality Thresholds

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Aspect ratio | < 2.0 | 2.0 - 5.0 | > 5.0 |
| Min angle (deg) | > 30 | 20 - 30 | < 20 |
| Skewness | < 0.25 | 0.25 - 0.75 | > 0.75 |

## Impact on Solver

| Issue | Symptom | Fix |
|---|---|---|
| High aspect ratio | Slow CG convergence | Refine in stretched direction |
| Tiny elements next to large | Stiff system | Grade mesh size smoothly |
| Near-degenerate triangles | NaN in solution | Re-mesh or use quads |
