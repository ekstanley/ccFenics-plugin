---
description: Import a mesh from a Gmsh .msh file
allowed-tools: Read, Glob
---

Guide the user through importing a Gmsh mesh file into DOLFINx.

## Step 1: Locate the Mesh File

If $ARGUMENTS contains a filename, use it. Otherwise:

1. Check `/workspace/` for `.msh` files using `list_workspace_files(pattern="*.msh")`
2. If found, present options and ask which to import
3. If not found, ask the user to upload or specify the file path

## Step 2: Import

Call `create_custom_mesh`:
```
create_custom_mesh(name="imported_mesh", filename="/workspace/mesh.msh")
```

## Step 3: Inspect

After import, run diagnostics:

1. `get_mesh_info(name="imported_mesh")` — report cell type, count, vertices, dimension
2. `compute_mesh_quality(mesh_name="imported_mesh")` — check element quality

Present a summary:
```
Imported: mesh.msh
  Cells: 4,521 triangles
  Vertices: 2,315
  Dimension: 2D
  Quality ratio: 0.12 (acceptable)
  Cell tags: imported_mesh_cell_tags (tags: 1, 2, 3)
  Facet tags: imported_mesh_facet_tags (tags: 1, 2, 3, 4)
```

## Step 4: Tag Mapping

If the mesh has physical groups (cell_tags or facet_tags), present them:

```
Boundary Tags:
  Tag 1: left boundary (23 facets)
  Tag 2: right boundary (23 facets)
  Tag 3: top boundary (30 facets)
  Tag 4: bottom boundary (30 facets)

Material Tags:
  Tag 1: steel region (2,100 cells)
  Tag 2: aluminum region (2,421 cells)
```

Ask the user to confirm the mapping or provide names for each tag.

## Step 5: Next Steps

Suggest:
- Create a function space: `/sim-setup`
- Check mesh quality in detail: `/check-mesh`
- Extract a subdomain: describe `create_submesh` usage
- Apply BCs using the imported facet tags

Remind the user that boundary conditions can reference tags directly:
```
apply_boundary_condition(value=0.0, boundary_tag=1)
```
