"""Function space creation tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import get_session, mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    DuplicateNameError,
    PostconditionError,
    PreconditionError,
    handle_tool_errors,
)
from ..session import FunctionSpaceInfo
from ._validators import require_nonempty

logger = logging.getLogger(__name__)


@mcp.tool()
@handle_tool_errors
async def create_function_space(
    name: str,
    family: str = "Lagrange",
    degree: int = 1,
    shape: list[int] | None = None,
    variant: str | None = None,
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
        variant: Lagrange element variant (e.g. "equispaced", "gll_warped").
            Only applicable for Lagrange-family elements.
        mesh_name: Which mesh to build on. Defaults to the active mesh.

    Returns:
        dict with name, mesh_name, element_family, element_degree, num_dofs,
        and optionally shape and variant.
    """
    # Preconditions
    require_nonempty(name, "Function space name")
    require_nonempty(family, "Element family")
    if degree < 0:
        raise PreconditionError(f"degree must be >= 0, got {degree}.")
    if degree > 10:
        raise PreconditionError(f"degree {degree} exceeds sanity limit of 10.")
    # PS-6: validate shape components early (before dolfinx call)
    if shape is not None:
        if not shape:
            raise PreconditionError("shape must be non-empty if provided.")
        if not all(isinstance(d, int) and d > 0 for d in shape):
            raise PreconditionError(
                f"All shape dimensions must be positive integers, got {shape}.",
            )
    # Validate variant
    _VALID_VARIANTS = frozenset({
        "equispaced", "gll_warped", "gll_isaac", "gll",
        "chebyshev_warped", "chebyshev_isaac",
        "gl_warped", "gl_isaac", "legendre", "bernstein",
    })
    if variant is not None and variant not in _VALID_VARIANTS:
        raise PreconditionError(
            f"variant must be one of {sorted(_VALID_VARIANTS)} or None, got '{variant}'.",
            suggestion="Common variants: 'equispaced', 'gll_warped', 'legendre'.",
        )

    import dolfinx.fem

    session = get_session(ctx)

    if name in session.function_spaces:
        raise DuplicateNameError(
            f"Function space '{name}' already exists.",
            suggestion=f"Use a different name or remove '{name}' first.",
        )

    mesh_info = session.get_mesh(mesh_name)
    mesh = mesh_info.mesh

    try:
        if variant is not None:
            import basix
            import basix.ufl
            variant_enum = getattr(basix.LagrangeVariant, variant, None)
            if variant_enum is None:
                raise DOLFINxAPIError(
                    f"Unknown Lagrange variant '{variant}' in basix.",
                    suggestion=f"Valid variants: {sorted(_VALID_VARIANTS)}",
                )
            element = basix.ufl.element(
                family, mesh.basix_cell(), degree,
                lagrange_variant=variant_enum,
                shape=tuple(shape) if shape else (),
            )
            V = dolfinx.fem.functionspace(mesh, element)
        elif shape is not None:
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
    session._space_id_to_name[id(fs_info.space)] = name

    if __debug__:
        session.check_invariants()

    logger.info(
        "Created function space '%s' (%s degree %d, %d DOFs%s)",
        name, family, degree, num_dofs,
        f", variant={variant}" if variant else "",
    )
    result = fs_info.summary()
    if variant is not None:
        result["variant"] = variant
    return result


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

    Returns:
        dict with name, mesh_name, element_family ("Mixed"), element_degree
        (max degree of subspaces), num_dofs, and subspaces (list of names).
    """
    # Precondition: validate subspace count before lazy imports
    if not subspaces or len(subspaces) < 2:
        raise PreconditionError(
            "Mixed space requires at least 2 subspaces.",
            suggestion="Provide a list with at least 2 function space names.",
        )

    import basix.ufl
    import dolfinx.fem

    session = get_session(ctx)

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
    session._space_id_to_name[id(fs_info.space)] = name

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
