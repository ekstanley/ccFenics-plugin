"""Function space creation tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import DOLFINxAPIError, DOLFINxMCPError, DuplicateNameError, PostconditionError, PreconditionError, handle_tool_errors
from ..session import FunctionSpaceInfo, SessionState

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


@mcp.tool()
@handle_tool_errors
async def create_function_space(
    name: str,
    family: str = "Lagrange",
    degree: int = 1,
    shape: list[int] | None = None,
    mesh_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a finite element function space on a mesh.

    Args:
        name: Unique name for this function space.
        family: Finite element family (e.g. "Lagrange", "DG", "N1curl", "RT").
        degree: Polynomial degree.
        shape: Component shape for vector/tensor spaces (e.g. [2] for 2D vector).
            Omit for scalar spaces.
        mesh_name: Which mesh to build on. Defaults to the active mesh.
    """
    # Preconditions
    if not name:
        raise PreconditionError("Function space name must be non-empty.")
    if not family:
        raise PreconditionError("Element family must be non-empty.")
    if degree < 0:
        raise PreconditionError(f"degree must be >= 0, got {degree}.")
    if degree > 10:
        raise PreconditionError(f"degree {degree} exceeds sanity limit of 10.")

    import dolfinx.fem

    session = _get_session(ctx)

    if name in session.function_spaces:
        raise DuplicateNameError(
            f"Function space '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    mesh_info = session.get_mesh(mesh_name)
    mesh = mesh_info.mesh

    try:
        if shape is not None:
            V = dolfinx.fem.functionspace(mesh, (family, degree, tuple(shape)))
        else:
            V = dolfinx.fem.functionspace(mesh, (family, degree))
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to create function space: {exc}",
            suggestion=f"Check element family '{family}' and degree {degree} are valid.",
        ) from exc

    num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # Postcondition: function space must have DOFs
    if num_dofs <= 0:
        raise PostconditionError(
            f"Function space created with {num_dofs} DOFs; expected > 0."
        )

    fs_info = FunctionSpaceInfo(
        name=name,
        space=V,
        mesh_name=mesh_info.name,
        element_family=family,
        element_degree=degree,
        num_dofs=num_dofs,
        shape=tuple(shape) if shape else None,
    )

    session.function_spaces[name] = fs_info

    if __debug__:
        session.check_invariants()

    logger.info(
        "Created function space '%s' (%s degree %d, %d DOFs)",
        name, family, degree, num_dofs,
    )
    return fs_info.summary()


@mcp.tool()
@handle_tool_errors
async def create_mixed_space(
    name: str,
    subspaces: list[str],
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a mixed finite element function space from existing spaces.

    Args:
        name: Unique name for this mixed function space.
        subspaces: List of existing function space names to combine.
            All subspaces must be defined on the same mesh.
    """
    # Precondition: validate subspace count before lazy imports
    if not subspaces or len(subspaces) < 2:
        raise PreconditionError(
            "Mixed space requires at least 2 subspaces.",
            suggestion="Provide a list with at least 2 function space names.",
        )

    import basix.ufl
    import dolfinx.fem

    session = _get_session(ctx)

    if name in session.function_spaces:
        raise DuplicateNameError(
            f"Function space '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    # Resolve all subspaces
    subspace_infos = [session.get_space(s) for s in subspaces]

    # Verify all on same mesh
    mesh_names = {info.mesh_name for info in subspace_infos}
    if len(mesh_names) > 1:
        raise DOLFINxAPIError(
            f"All subspaces must be on the same mesh. Found meshes: {mesh_names}",
            suggestion="Ensure all function spaces use the same mesh.",
        )

    mesh_name = subspace_infos[0].mesh_name
    mesh_info = session.get_mesh(mesh_name)
    mesh = mesh_info.mesh

    # Extract UFL elements and create mixed element
    try:
        elements = [space_info.space.ufl_element() for space_info in subspace_infos]
        mixed_el = basix.ufl.mixed_element(elements)
        W = dolfinx.fem.functionspace(mesh, mixed_el)
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to create mixed function space: {exc}",
            suggestion="Check that all subspaces are compatible for mixing.",
        ) from exc

    num_dofs = W.dofmap.index_map.size_local * W.dofmap.index_map_bs

    # Postcondition: mixed space must have DOFs
    if num_dofs <= 0:
        raise PostconditionError(
            f"Mixed function space created with {num_dofs} DOFs; expected > 0."
        )

    # Calculate max degree from subspaces
    max_degree = max(info.element_degree for info in subspace_infos)

    fs_info = FunctionSpaceInfo(
        name=name,
        space=W,
        mesh_name=mesh_name,
        element_family="Mixed",
        element_degree=max_degree,
        num_dofs=num_dofs,
        shape=None,
    )

    session.function_spaces[name] = fs_info

    if __debug__:
        session.check_invariants()

    logger.info(
        "Created mixed function space '%s' from %d subspaces (%d DOFs)",
        name, len(subspaces), num_dofs,
    )
    return {
        **fs_info.summary(),
        "subspaces": subspaces,
    }
