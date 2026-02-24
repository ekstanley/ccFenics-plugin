---
description: Audit mesh quality before solving
allowed-tools: Read
---

Run a mesh quality audit on the current active mesh (or the mesh specified in $ARGUMENTS).

## Checks to Perform

1. **Basic info**: Call `get_mesh_info` — report cell type, number of cells, vertices, dimension.

2. **Quality metrics**: Call `compute_mesh_quality` — report min/max/mean volume, standard deviation, quality ratio.

3. **Assessment**:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Quality ratio (min/max vol) | > 0.1 | 0.01 - 0.1 | < 0.01 |
| All volumes positive | Yes | — | Any negative = invalid |
| Std/mean volume | < 0.5 | 0.5 - 1.0 | > 1.0 |

4. **Recommendations**:
   - If quality is poor: suggest mesh refinement or re-meshing
   - If highly non-uniform: warn about solver conditioning
   - If any negative volumes: STOP — mesh is invalid, must regenerate

Present results as a concise quality report card with pass/fail indicators.
