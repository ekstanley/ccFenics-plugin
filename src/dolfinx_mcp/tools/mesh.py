"""Mesh creation and inspection tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import DOLFINxAPIError, DOLFINxMCPError, DuplicateNameError, PostconditionError, PreconditionError, handle_tool_errors
from ..session import EntityMapInfo, MeshInfo, MeshTagsInfo, SessionState

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


@mcp.tool()
@handle_tool_errors
async def create_unit_square(
    name: str,
    nx: int = 8,
    ny: int = 8,
    cell_type: str = "triangle",
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a unit square mesh [0,1]x[0,1].

    Args:
        name: Unique name for this mesh.
        nx: Number of cells in x direction.
        ny: Number of cells in y direction.
        cell_type: Element type -- "triangle" or "quadrilateral".
    """
    # Preconditions
    if not name:
        raise PreconditionError("Mesh name must be non-empty.")
    if nx <= 0:
        raise PreconditionError(f"nx must be > 0, got {nx}.")
    if ny <= 0:
        raise PreconditionError(f"ny must be > 0, got {ny}.")
    _VALID_CELL_TYPES_2D = {"triangle", "quadrilateral"}
    if cell_type not in _VALID_CELL_TYPES_2D:
        raise PreconditionError(
            f"Invalid cell_type '{cell_type}'. Must be one of {sorted(_VALID_CELL_TYPES_2D)}."
        )

    from mpi4py import MPI
    import dolfinx.mesh

    session = _get_session(ctx)

    if name in session.meshes:
        raise DuplicateNameError(
            f"Mesh '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    cell_types = {
        "triangle": dolfinx.mesh.CellType.triangle,
        "quadrilateral": dolfinx.mesh.CellType.quadrilateral,
    }

    try:
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, nx, ny, cell_types[cell_type]
        )
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(f"Failed to create mesh: {exc}") from exc

    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)

    info = MeshInfo(
        name=name,
        mesh=mesh,
        cell_type=cell_type,
        num_cells=topology.index_map(topology.dim).size_local,
        num_vertices=topology.index_map(0).size_local,
        gdim=mesh.geometry.dim,
        tdim=topology.dim,
    )

    # Postcondition: mesh must have cells and vertices
    if info.num_cells <= 0 or info.num_vertices <= 0:
        raise PostconditionError(
            f"Mesh creation produced {info.num_cells} cells and {info.num_vertices} vertices; "
            "expected both > 0."
        )

    session.meshes[name] = info
    session.active_mesh = name

    if __debug__:
        session.check_invariants()

    result = info.summary()
    result["active"] = True
    logger.info("Created unit square mesh '%s' (%dx%d %s)", name, nx, ny, cell_type)
    return result


@mcp.tool()
@handle_tool_errors
async def get_mesh_info(
    name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Get information about a mesh.

    Args:
        name: Mesh name. Defaults to the active mesh.
    """
    import numpy as np

    session = _get_session(ctx)
    info = session.get_mesh(name)
    mesh = info.mesh

    # Compute bounding box
    coords = mesh.geometry.x
    bbox_min = coords.min(axis=0).tolist()
    bbox_max = coords.max(axis=0).tolist()

    # Postcondition: bounding box coordinates must be finite
    if not np.isfinite(coords).all():
        raise PostconditionError(
            "Bounding box contains NaN/Inf values.",
            suggestion="Check mesh geometry for degenerate elements.",
        )

    result = info.summary()
    result["bounding_box"] = {"min": bbox_min, "max": bbox_max}
    result["active"] = (info.name == session.active_mesh)
    return result

@mcp.tool()
@handle_tool_errors
async def create_mesh(
    name: str,
    shape: str,
    nx: int = 8,
    ny: int = 8,
    nz: int = 8,
    cell_type: str = "triangle",
    dimensions: dict | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a mesh with specified shape and parameters.

    Args:
        name: Unique name for this mesh.
        shape: Shape type -- "unit_square", "unit_cube", "rectangle", or "box".
        nx: Number of cells in x direction.
        ny: Number of cells in y direction.
        nz: Number of cells in z direction (for 3D shapes).
        cell_type: Element type -- "triangle"/"quadrilateral" for 2D, "tetrahedron"/"hexahedron" for 3D.
        dimensions: Optional dimensions dict. For rectangle: {"width": w, "height": h}. For box: {"x": x, "y": y, "z": z}.
    """
    # Precondition: validate cell_type early (before lazy imports)
    _VALID_CELL_TYPES = {"triangle", "quadrilateral", "tetrahedron", "hexahedron"}
    if cell_type not in _VALID_CELL_TYPES:
        raise PreconditionError(
            f"Invalid cell_type '{cell_type}'. Must be one of {sorted(_VALID_CELL_TYPES)}."
        )

    # Precondition: validate shape early (before lazy imports)
    _VALID_SHAPES = {"unit_square", "unit_cube", "rectangle", "box"}
    if shape not in _VALID_SHAPES:
        raise PreconditionError(
            f"Invalid shape '{shape}'. Must be one of {sorted(_VALID_SHAPES)}."
        )
    if not name:
        raise PreconditionError("Mesh name must be non-empty.")
    if nx <= 0:
        raise PreconditionError(f"nx must be > 0, got {nx}.")
    if ny <= 0:
        raise PreconditionError(f"ny must be > 0, got {ny}.")
    if shape in ("unit_cube", "box") and nz <= 0:
        raise PreconditionError(f"nz must be > 0 for 3D shapes, got {nz}.")

    from mpi4py import MPI
    import dolfinx.mesh

    session = _get_session(ctx)

    if name in session.meshes:
        raise DuplicateNameError(
            f"Mesh '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    try:
        if shape == "unit_square":
            cell_types = {
                "triangle": dolfinx.mesh.CellType.triangle,
                "quadrilateral": dolfinx.mesh.CellType.quadrilateral,
            }
            if cell_type not in cell_types:
                raise DOLFINxAPIError(
                    f"Unsupported cell type '{cell_type}' for unit_square.",
                    suggestion=f"Use one of: {list(cell_types.keys())}",
                )
            mesh = dolfinx.mesh.create_unit_square(
                MPI.COMM_WORLD, nx, ny, cell_types[cell_type]
            )

        elif shape == "unit_cube":
            cell_types = {
                "tetrahedron": dolfinx.mesh.CellType.tetrahedron,
                "hexahedron": dolfinx.mesh.CellType.hexahedron,
            }
            if cell_type not in cell_types:
                raise DOLFINxAPIError(
                    f"Unsupported cell type '{cell_type}' for unit_cube.",
                    suggestion=f"Use one of: {list(cell_types.keys())}",
                )
            mesh = dolfinx.mesh.create_unit_cube(
                MPI.COMM_WORLD, nx, ny, nz, cell_types[cell_type]
            )

        elif shape == "rectangle":
            if dimensions is None:
                raise DOLFINxAPIError(
                    "Rectangle shape requires 'dimensions' dict with 'width' and 'height'.",
                    suggestion="Example: dimensions={'width': 2.0, 'height': 1.0}",
                )
            width = dimensions.get("width")
            height = dimensions.get("height")
            if width is None or height is None:
                raise DOLFINxAPIError(
                    "Rectangle dimensions must include 'width' and 'height'.",
                    suggestion="Example: dimensions={'width': 2.0, 'height': 1.0}",
                )
            cell_types = {
                "triangle": dolfinx.mesh.CellType.triangle,
                "quadrilateral": dolfinx.mesh.CellType.quadrilateral,
            }
            if cell_type not in cell_types:
                raise DOLFINxAPIError(
                    f"Unsupported cell type '{cell_type}' for rectangle.",
                    suggestion=f"Use one of: {list(cell_types.keys())}",
                )
            mesh = dolfinx.mesh.create_rectangle(
                MPI.COMM_WORLD, [[0, 0], [width, height]], [nx, ny], cell_types[cell_type]
            )

        elif shape == "box":
            if dimensions is None:
                raise DOLFINxAPIError(
                    "Box shape requires 'dimensions' dict with 'x', 'y', and 'z'.",
                    suggestion="Example: dimensions={'x': 2.0, 'y': 1.0, 'z': 1.0}",
                )
            x_dim = dimensions.get("x")
            y_dim = dimensions.get("y")
            z_dim = dimensions.get("z")
            if x_dim is None or y_dim is None or z_dim is None:
                raise DOLFINxAPIError(
                    "Box dimensions must include 'x', 'y', and 'z'.",
                    suggestion="Example: dimensions={'x': 2.0, 'y': 1.0, 'z': 1.0}",
                )
            cell_types = {
                "tetrahedron": dolfinx.mesh.CellType.tetrahedron,
                "hexahedron": dolfinx.mesh.CellType.hexahedron,
            }
            if cell_type not in cell_types:
                raise DOLFINxAPIError(
                    f"Unsupported cell type '{cell_type}' for box.",
                    suggestion=f"Use one of: {list(cell_types.keys())}",
                )
            mesh = dolfinx.mesh.create_box(
                MPI.COMM_WORLD, [[0, 0, 0], [x_dim, y_dim, z_dim]], [nx, ny, nz], cell_types[cell_type]
            )

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(f"Failed to create mesh: {exc}") from exc

    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)

    info = MeshInfo(
        name=name,
        mesh=mesh,
        cell_type=cell_type,
        num_cells=topology.index_map(topology.dim).size_local,
        num_vertices=topology.index_map(0).size_local,
        gdim=mesh.geometry.dim,
        tdim=topology.dim,
    )

    # Postcondition: mesh must have cells and vertices
    if info.num_cells <= 0 or info.num_vertices <= 0:
        raise PostconditionError(
            f"Mesh creation produced {info.num_cells} cells and {info.num_vertices} vertices; "
            "expected both > 0."
        )

    session.meshes[name] = info
    session.active_mesh = name

    if __debug__:
        session.check_invariants()

    result = info.summary()
    result["active"] = True
    result["shape"] = shape
    logger.info("Created %s mesh '%s' (%s)", shape, name, cell_type)
    return result


def _make_marker_fn(condition: str):
    """Create a boundary marker function from a condition string.

    Args:
        condition: Python expression using 'x' (coordinate array) and 'np' (numpy).
                  Example: "x[0] < 1e-14" or "np.isclose(x[1], 1.0)"
    """
    import numpy as np

    if condition.strip() == "True":
        def marker(x):
            return np.full(x.shape[1], True)
        return marker

    def marker(x):
        ns = {"x": x, "np": np, "pi": np.pi, "__builtins__": {}}
        return eval(condition, ns)  # noqa: S307
    return marker


@mcp.tool()
@handle_tool_errors
async def mark_boundaries(
    markers: list[dict],
    name: str = "boundary_tags",
    mesh_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Mark boundary facets based on geometric conditions.

    Args:
        markers: List of marker specifications. Each dict has "tag" (int) and "condition" (str).
                Example: [{"tag": 1, "condition": "x[0] < 1e-14"}, {"tag": 2, "condition": "x[0] > 1.0 - 1e-14"}]
        name: Name for the boundary tags object.
        mesh_name: Mesh name. Defaults to the active mesh.
    """
    # Preconditions
    if not markers:
        raise PreconditionError("markers list must be non-empty.")
    for m in markers:
        if not isinstance(m.get("tag"), int) or m["tag"] < 0:
            raise PreconditionError(
                f"Each marker 'tag' must be a non-negative int, got {m.get('tag')}."
            )
        if not m.get("condition"):
            raise PreconditionError("Each marker must have a non-empty 'condition' string.")

    import dolfinx.mesh
    import numpy as np

    session = _get_session(ctx)

    if name in session.mesh_tags:
        raise DuplicateNameError(
            f"MeshTags '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    mesh_info = session.get_mesh(mesh_name)
    mesh = mesh_info.mesh

    fdim = mesh.topology.dim - 1

    # Collect all boundary facets
    all_facet_indices = []
    all_tag_values = []

    try:
        for marker_spec in markers:
            tag = marker_spec.get("tag")
            condition = marker_spec.get("condition")

            if tag is None or condition is None:
                raise DOLFINxAPIError(
                    "Each marker must have 'tag' (int) and 'condition' (str).",
                    suggestion="Example: {'tag': 1, 'condition': 'x[0] < 1e-14'}",
                )

            marker_fn = _make_marker_fn(condition)
            facet_indices = dolfinx.mesh.locate_entities_boundary(mesh, fdim, marker_fn)

            all_facet_indices.append(facet_indices)
            all_tag_values.append(np.full(len(facet_indices), tag, dtype=np.int32))

        # Concatenate all facets and tags
        if all_facet_indices:
            facet_indices = np.concatenate(all_facet_indices)
            tag_values = np.concatenate(all_tag_values)
        else:
            facet_indices = np.array([], dtype=np.int32)
            tag_values = np.array([], dtype=np.int32)

        # Create MeshTags
        meshtags = dolfinx.mesh.meshtags(mesh, fdim, facet_indices, tag_values)

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(f"Failed to create boundary markers: {exc}") from exc

    # Store in session
    tags_info = MeshTagsInfo(
        name=name,
        tags=meshtags,
        mesh_name=mesh_info.name,
        dimension=fdim,
        unique_tags=[int(t) for t in np.unique(tag_values)],
    )
    session.mesh_tags[name] = tags_info

    # Postcondition: boundary marking must produce at least one tag
    if not tags_info.unique_tags:
        raise PostconditionError(
            "Boundary marking produced no tagged facets.",
            suggestion="Check boundary conditions match the mesh geometry.",
        )

    # Count tags
    tag_counts = {}
    for tag in np.unique(tag_values):
        tag_counts[int(tag)] = int(np.sum(tag_values == tag))

    if __debug__:
        session.check_invariants()

    logger.info("Created boundary tags '%s' on mesh '%s'", name, mesh_info.name)
    return {
        "name": name,
        "mesh": mesh_info.name,
        "dim": fdim,
        "num_facets": len(facet_indices),
        "tag_counts": tag_counts,
    }


@mcp.tool()
@handle_tool_errors
async def refine_mesh(
    name: str | None = None,
    new_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Uniformly refine a mesh.

    Args:
        name: Mesh name to refine. Defaults to the active mesh.
        new_name: Name for the refined mesh. Defaults to "{name}_refined".
    """
    import dolfinx.mesh

    session = _get_session(ctx)

    mesh_info = session.get_mesh(name)
    refined_name = new_name or f"{mesh_info.name}_refined"

    if refined_name in session.meshes:
        raise DuplicateNameError(
            f"Mesh '{refined_name}' already exists.",
            suggestion=f"Use a different name or remove '{refined_name}' first.",
        )

    try:
        refined_mesh = dolfinx.mesh.refine(mesh_info.mesh)
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(f"Failed to refine mesh: {exc}") from exc

    topology = refined_mesh.topology
    topology.create_connectivity(topology.dim, 0)

    refined_info = MeshInfo(
        name=refined_name,
        mesh=refined_mesh,
        cell_type=mesh_info.cell_type,
        num_cells=topology.index_map(topology.dim).size_local,
        num_vertices=topology.index_map(0).size_local,
        gdim=refined_mesh.geometry.dim,
        tdim=topology.dim,
    )

    session.meshes[refined_name] = refined_info
    session.active_mesh = refined_name

    result = refined_info.summary()
    result["active"] = True
    result["original_mesh"] = mesh_info.name
    result["original_cells"] = mesh_info.num_cells
    result["original_vertices"] = mesh_info.num_vertices
    result["refinement_factor"] = refined_info.num_cells / mesh_info.num_cells

    # Postcondition: uniform refinement must increase cell count
    if refined_info.num_cells <= mesh_info.num_cells:
        raise PostconditionError(
            f"refine_mesh(): refined mesh has {refined_info.num_cells} cells, "
            f"expected more than original {mesh_info.num_cells}"
        )

    if __debug__:
        session.check_invariants()

    logger.info(
        "Refined mesh '%s' -> '%s' (%d -> %d cells)",
        mesh_info.name, refined_name, mesh_info.num_cells, refined_info.num_cells
    )
    return result


@mcp.tool()
@handle_tool_errors
async def create_custom_mesh(
    name: str,
    filename: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Import mesh from a Gmsh .msh file.

    Args:
        name: Unique name for this mesh.
        filename: Path to the .msh file.
    """
    # Preconditions
    if not name:
        raise PreconditionError("Mesh name must be non-empty.")
    if not filename:
        raise PreconditionError("filename must be non-empty.")

    from mpi4py import MPI
    import dolfinx.mesh

    session = _get_session(ctx)

    if name in session.meshes:
        raise DuplicateNameError(
            f"Mesh '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    try:
        import gmsh
    except ImportError as exc:
        raise DOLFINxAPIError(
            "Gmsh not available. Install with 'pip install gmsh'.",
            suggestion="Install gmsh to import .msh files.",
        ) from exc

    try:
        from dolfinx.io.gmsh import model_to_mesh

        gmsh.initialize()
        try:
            gmsh.open(filename)
            mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0)
        finally:
            gmsh.finalize()

        mesh = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        facet_tags = mesh_data.facet_tags

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(f"Failed to import mesh from '{filename}': {exc}") from exc

    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)

    # Determine cell type
    cell_type_map = {
        dolfinx.mesh.CellType.triangle: "triangle",
        dolfinx.mesh.CellType.quadrilateral: "quadrilateral",
        dolfinx.mesh.CellType.tetrahedron: "tetrahedron",
        dolfinx.mesh.CellType.hexahedron: "hexahedron",
    }
    cell_type = cell_type_map.get(mesh.topology.cell_type, "unknown")

    info = MeshInfo(
        name=name,
        mesh=mesh,
        cell_type=cell_type,
        num_cells=topology.index_map(topology.dim).size_local,
        num_vertices=topology.index_map(0).size_local,
        gdim=mesh.geometry.dim,
        tdim=topology.dim,
    )

    # Postcondition: mesh must have cells and vertices
    if info.num_cells <= 0 or info.num_vertices <= 0:
        raise PostconditionError(
            f"Mesh creation produced {info.num_cells} cells and {info.num_vertices} vertices; "
            "expected both > 0."
        )

    session.meshes[name] = info
    session.active_mesh = name

    # Store cell tags if they exist
    result = info.summary()
    result["active"] = True
    result["filename"] = filename

    if cell_tags is not None:
        import numpy as np
        unique_cell_tags = np.unique(cell_tags.values).tolist()
        cell_tags_name = f"{name}_cell_tags"
        session.mesh_tags[cell_tags_name] = MeshTagsInfo(
            name=cell_tags_name,
            tags=cell_tags,
            mesh_name=name,
            dimension=topology.dim,
            unique_tags=[int(t) for t in unique_cell_tags],
        )
        result["cell_tags"] = cell_tags_name

    if facet_tags is not None:
        import numpy as np
        unique_facet_tags = np.unique(facet_tags.values).tolist()
        facet_tags_name = f"{name}_facet_tags"
        session.mesh_tags[facet_tags_name] = MeshTagsInfo(
            name=facet_tags_name,
            tags=facet_tags,
            mesh_name=name,
            dimension=topology.dim - 1,
            unique_tags=[int(t) for t in unique_facet_tags],
        )
        result["facet_tags"] = facet_tags_name

    if __debug__:
        session.check_invariants()

    logger.info("Imported mesh '%s' from '%s' (%s)", name, filename, cell_type)
    return result


@mcp.tool()
@handle_tool_errors
async def create_submesh(
    name: str,
    tags_name: str,
    tag_values: list[int],
    parent_mesh: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Extract a submesh from tagged entities.

    Args:
        name: Unique name for the submesh.
        tags_name: Name of the MeshTags to use.
        tag_values: List of tag values to include in the submesh.
        parent_mesh: Parent mesh name. Defaults to the mesh associated with tags_name.
    """
    # Preconditions
    if not name:
        raise PreconditionError("Submesh name must be non-empty.")
    if not tag_values:
        raise PreconditionError("tag_values must be non-empty.")
    if not all(isinstance(v, int) for v in tag_values):
        raise PreconditionError("All tag_values must be integers.")

    import dolfinx.mesh
    import numpy as np

    session = _get_session(ctx)

    if name in session.meshes:
        raise DuplicateNameError(
            f"Mesh '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    tags_info = session.get_mesh_tags(tags_name)
    mesh_name = parent_mesh or tags_info.mesh_name
    mesh_info = session.get_mesh(mesh_name)
    mesh = mesh_info.mesh
    tags = tags_info.tags

    # Find entities matching tag values
    try:
        entity_mask = np.isin(tags.values, tag_values)
        entities = tags.indices[entity_mask]

        if len(entities) == 0:
            raise DOLFINxAPIError(
                f"No entities found with tag values {tag_values} in '{tags_name}'.",
                suggestion=f"Available tags: {tags_info.unique_tags}",
            )

        submesh, entity_map = dolfinx.mesh.create_submesh(
            mesh, tags_info.dimension, entities
        )

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(f"Failed to create submesh: {exc}") from exc

    topology = submesh.topology
    topology.create_connectivity(topology.dim, 0)

    submesh_info = MeshInfo(
        name=name,
        mesh=submesh,
        cell_type=mesh_info.cell_type,
        num_cells=topology.index_map(topology.dim).size_local,
        num_vertices=topology.index_map(0).size_local,
        gdim=submesh.geometry.dim,
        tdim=topology.dim,
    )

    # Postcondition: submesh cannot have more cells than parent
    if submesh_info.num_cells > mesh_info.num_cells:
        raise PostconditionError(
            f"Submesh has {submesh_info.num_cells} cells, exceeding parent's {mesh_info.num_cells}."
        )

    session.meshes[name] = submesh_info
    session.active_mesh = name

    # Store entity map
    entity_map_name = f"{name}_entity_map"
    entity_map_info = EntityMapInfo(
        name=entity_map_name,
        entity_map=entity_map,
        parent_mesh=mesh_name,
        child_mesh=name,
        dimension=tags_info.dimension,
    )
    session.entity_maps[entity_map_name] = entity_map_info

    result = submesh_info.summary()
    result["active"] = True
    result["parent_mesh"] = mesh_name
    result["tags_name"] = tags_name
    result["tag_values"] = tag_values
    result["entity_map"] = entity_map_name
    result["num_extracted_entities"] = len(entities)

    if __debug__:
        session.check_invariants()

    logger.info(
        "Created submesh '%s' from '%s' with tags %s (%d entities)",
        name, mesh_name, tag_values, len(entities)
    )
    return result


@mcp.tool()
@handle_tool_errors
async def manage_mesh_tags(
    action: str,
    name: str,
    mesh_name: str | None = None,
    dimension: int | None = None,
    values: list[dict] | None = None,
    tags_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create or query mesh tags.

    Args:
        action: Action to perform -- "create" or "query".
        name: Name for the mesh tags (for create) or mesh name (for query).
        mesh_name: Mesh name (for create). Defaults to active mesh.
        dimension: Dimension of entities to tag (for create).
        values: List of {"entities": list[int], "tag": int} dicts (for create).
        tags_name: Name of tags to query (for query).
    """
    # Precondition: validate action before lazy imports
    if action not in ("create", "query"):
        raise PreconditionError(
            f"action must be 'create' or 'query', got '{action}'."
        )

    import dolfinx.mesh
    import numpy as np

    session = _get_session(ctx)

    if action == "create":
        if name in session.mesh_tags:
            raise DuplicateNameError(
                f"MeshTags '{name}' already exists.",
                suggestion=f"Use a different name or remove '{name}' first.",
            )

        if dimension is None:
            raise DOLFINxAPIError(
                "Missing 'dimension' parameter for create action.",
                suggestion="Specify dimension (0=vertices, 1=edges, 2=facets, 3=cells).",
            )

        if values is None:
            raise DOLFINxAPIError(
                "Missing 'values' parameter for create action.",
                suggestion="Provide list of {'entities': [int], 'tag': int} dicts.",
            )

        mesh_info = session.get_mesh(mesh_name)
        mesh = mesh_info.mesh

        # Build entity-tag arrays
        all_entities = []
        all_tags = []

        try:
            for item in values:
                entities = item.get("entities")
                tag = item.get("tag")

                if entities is None or tag is None:
                    raise DOLFINxAPIError(
                        "Each value dict must have 'entities' (list[int]) and 'tag' (int).",
                        suggestion="Example: {'entities': [0, 1, 2], 'tag': 1}",
                    )

                all_entities.extend(entities)
                all_tags.extend([tag] * len(entities))

            entity_indices = np.array(all_entities, dtype=np.int32)
            tag_values = np.array(all_tags, dtype=np.int32)

            meshtags = dolfinx.mesh.meshtags(mesh, dimension, entity_indices, tag_values)

        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(f"Failed to create mesh tags: {exc}") from exc

        # Store in session
        unique_tags = np.unique(tag_values).tolist()
        tags_info = MeshTagsInfo(
            name=name,
            tags=meshtags,
            mesh_name=mesh_info.name,
            dimension=dimension,
            unique_tags=[int(t) for t in unique_tags],
        )
        session.mesh_tags[name] = tags_info

        # Postcondition: tag creation must produce at least one tag
        if not tags_info.unique_tags:
            raise PostconditionError(
                "Mesh tag creation produced no tagged entities.",
                suggestion="Check entity indices and tag values are valid.",
            )

        # Count tags
        tag_counts = {}
        for tag in unique_tags:
            tag_counts[int(tag)] = int(np.sum(tag_values == tag))

        if __debug__:
            session.check_invariants()

        logger.info("Created mesh tags '%s' on mesh '%s' (dim=%d)", name, mesh_info.name, dimension)
        return {
            "name": name,
            "mesh": mesh_info.name,
            "dimension": dimension,
            "num_entities": len(entity_indices),
            "unique_tags": [int(t) for t in unique_tags],
            "tag_counts": tag_counts,
        }

    elif action == "query":
        query_name = tags_name or name
        tags_info = session.get_mesh_tags(query_name)
        tags = tags_info.tags

        # Count occurrences of each tag
        tag_counts = {}
        for tag in tags_info.unique_tags:
            tag_counts[tag] = int(np.sum(tags.values == tag))

        logger.info("Queried mesh tags '%s'", query_name)
        return {
            "name": query_name,
            "mesh": tags_info.mesh_name,
            "dimension": tags_info.dimension,
            "num_entities": len(tags.indices),
            "unique_tags": tags_info.unique_tags,
            "tag_counts": tag_counts,
        }


@mcp.tool()
@handle_tool_errors
async def compute_mesh_quality(
    mesh_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Compute mesh quality metrics (cell volumes).

    Calculates min, max, mean, and standard deviation of cell volumes
    for the specified mesh. All volumes should be positive for a valid mesh.

    Args:
        mesh_name: Name of the mesh. If None, uses the active mesh.

    Returns:
        dict with min_volume, max_volume, mean_volume, std_volume, num_cells,
        and quality_ratio (min/max).
    """
    import dolfinx.mesh
    import numpy as np

    session = _get_session(ctx)
    mesh_info = session.get_mesh(mesh_name)
    msh = mesh_info.mesh

    try:
        # Compute cell volumes using DOLFINx geometry API
        tdim = msh.topology.dim
        num_cells = msh.topology.index_map(tdim).size_local

        # Create entity-to-geometry connectivity
        msh.topology.create_connectivity(tdim, 0)

        # Get geometry and compute volumes per cell
        geometry = msh.geometry.x
        cell_volumes = np.zeros(num_cells)

        for cell_idx in range(num_cells):
            # Get vertex coordinates for this cell
            cell_entities = msh.geometry.dofmap[cell_idx]
            coords = geometry[cell_entities]

            if tdim == 2:
                # Triangle area via cross product
                if len(cell_entities) == 3:
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                    cell_volumes[cell_idx] = 0.5 * abs(
                        v1[0] * v2[1] - v1[1] * v2[0]
                    )
                else:
                    # Quadrilateral: split into two triangles
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                    v3 = coords[3] - coords[0]
                    cell_volumes[cell_idx] = 0.5 * (
                        abs(v1[0] * v2[1] - v1[1] * v2[0])
                        + abs(v2[0] * v3[1] - v2[1] * v3[0])
                    )
            elif tdim == 3:
                # Tetrahedron volume
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                v3 = coords[3] - coords[0]
                cell_volumes[cell_idx] = abs(
                    np.dot(v1, np.cross(v2, v3))
                ) / 6.0
            elif tdim == 1:
                # Line segment length
                cell_volumes[cell_idx] = np.linalg.norm(coords[1] - coords[0])

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to compute mesh quality for '{mesh_info.name}': {exc}",
            suggestion="Check mesh integrity and topology.",
        ) from exc

    # Postcondition: all metrics must be finite
    if not np.isfinite(cell_volumes).all():
        raise PostconditionError(
            "Mesh quality metrics contain NaN/Inf.",
            suggestion="Check mesh geometry for degenerate cells.",
        )

    # Postcondition: all cell volumes must be > 0
    if not (cell_volumes > 0).all():
        raise PostconditionError(
            "Some cell volumes are non-positive, indicating degenerate cells.",
            suggestion="Check mesh for inverted or zero-area elements.",
        )

    min_vol = float(np.min(cell_volumes))
    max_vol = float(np.max(cell_volumes))
    mean_vol = float(np.mean(cell_volumes))
    std_vol = float(np.std(cell_volumes))
    quality_ratio = min_vol / max_vol if max_vol > 0 else 0.0

    if __debug__:
        session.check_invariants()

    logger.info(
        "Mesh quality for '%s': vol_range=[%.2e, %.2e], ratio=%.4f",
        mesh_info.name, min_vol, max_vol, quality_ratio,
    )
    return {
        "mesh_name": mesh_info.name,
        "num_cells": int(num_cells),
        "min_volume": round(min_vol, 10),
        "max_volume": round(max_vol, 10),
        "mean_volume": round(mean_vol, 10),
        "std_volume": round(std_vol, 10),
        "quality_ratio": round(quality_ratio, 6),
    }
