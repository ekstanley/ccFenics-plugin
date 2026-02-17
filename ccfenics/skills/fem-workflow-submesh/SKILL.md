---
name: fem-workflow-submesh
description: |
  Guides the user through creating submeshes and extracting subdomains using DOLFINx MCP tools.
  Use when the user asks about submeshes, extracting a subdomain, domain decomposition,
  or working with subsets of the mesh.
---

# Submesh & Subdomain Extraction Workflow

Extract subdomains from a parent mesh for localized analysis.

## Step-by-Step Tool Sequence

### 1. Create Parent Mesh and Tags

```
create_unit_square(name="mesh", nx=32, ny=32)
```

### 2. Tag Subdomains

Use `manage_mesh_tags` to mark cells by region:

```
manage_mesh_tags(
    action="create",
    name="subdomain_tags",
    markers=[
        {"tag": 1, "condition": "x[0] < 0.5"},
        {"tag": 2, "condition": "x[0] >= 0.5"}
    ],
    dimension=2
)
```

### 3. Extract Submesh

Use `create_submesh` to extract a subdomain:

```
create_submesh(
    name="left_domain",
    parent_mesh="mesh",
    tags_name="subdomain_tags",
    tag=1
)
```

This creates a new mesh containing only the cells tagged with the specified tag. An entity map is automatically created to map between parent and child entities.

### 4. Solve on Submesh

Create function spaces and solve on the extracted submesh:

```
create_function_space(name="V_sub", family="Lagrange", degree=1, mesh="left_domain")
# ... define forms, BCs, solve on the submesh
```

### 5. Transfer Data

Use entity maps to transfer data between parent and child meshes for multi-domain coupling.

## Use Cases

| Use Case | Approach |
|---|---|
| Multi-material | Tag cells by material, solve with piecewise coefficients |
| Local refinement | Extract region of interest, solve at higher resolution |
| Contact problems | Extract surfaces for contact detection |
| Domain decomposition | Split into subdomains for parallel or iterative solving |

## Important Notes

- Submeshes inherit the parent mesh's geometric dimension
- Entity maps track the parent-child cell correspondence
- Boundary facets of the submesh include both external boundaries and internal interfaces
