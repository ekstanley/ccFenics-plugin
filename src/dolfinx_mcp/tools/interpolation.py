"""Interpolation and discrete operator tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import DOLFINxAPIError, PreconditionError, handle_tool_errors
from ..session import SessionState

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


def _eval_interp_expression(expr: str, x):
    """Evaluate an interpolation expression at coordinate arrays.

    SECURITY: This uses Python's eval intentionally. DOLFINx interpolation
    requires callable expressions. Mitigations:
    1. __builtins__ set to empty dict (no system access)
    2. Restricted namespace with only safe functions
    3. Docker container isolation (--network none, non-root, --rm)
    """
    import numpy as np

    ns = {
        "x": x,
        "np": np,
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "log": np.log,
        "__builtins__": {},
    }
    result = eval(expr, ns)  # noqa: S307 -- restricted namespace, Docker-sandboxed
    if isinstance(result, (int, float)):
        return np.full(x.shape[1], float(result))
    return result


@mcp.tool()
@handle_tool_errors
async def interpolate(
    target: str,
    expression: str | None = None,
    source_function: str | None = None,
    source_mesh: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Interpolate an expression or function into a target function.

    Args:
        target: Name of the target function to interpolate into.
        expression: Python/numpy expression to interpolate (e.g., "sin(pi*x[0])").
            Use x[0], x[1], x[2] for coordinates. Mutually exclusive with source_function.
        source_function: Name of source function to interpolate from.
            If on same mesh, does direct interpolation. If source_mesh is provided,
            does cross-mesh interpolation.
        source_mesh: Name of the mesh for source_function (for cross-mesh interpolation).
    """
    import numpy as np
    import dolfinx.fem

    session = _get_session(ctx)

    # Get target function
    target_info = session.get_function(target)
    target_func = target_info.function

    # Validate arguments
    if expression is not None and source_function is not None:
        raise DOLFINxAPIError(
            "Cannot specify both 'expression' and 'source_function'.",
            suggestion="Use either expression-based or function-based interpolation, not both.",
        )

    if expression is None and source_function is None:
        raise DOLFINxAPIError(
            "Must specify either 'expression' or 'source_function'.",
            suggestion="Provide an expression string or source function name.",
        )

    # Expression-based interpolation
    if expression is not None:
        try:
            target_func.interpolate(
                lambda x: _eval_interp_expression(expression, x)
            )
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to interpolate expression '{expression}': {exc}",
                suggestion="Check expression syntax. Use x[0], x[1], x[2] for coordinates.",
            ) from exc

        # Postcondition: interpolation result must be finite
        if not np.isfinite(target_func.x.array).all():
            raise DOLFINxAPIError(
                "Interpolation produced NaN/Inf values.",
                suggestion="Check source expression or function for validity.",
            )

        # Compute statistics
        l2_norm = float(np.linalg.norm(target_func.x.array))
        min_val = float(np.min(target_func.x.array))
        max_val = float(np.max(target_func.x.array))

        if __debug__:
            session.check_invariants()

        logger.info("Interpolated expression into '%s'", target)
        return {
            "target": target,
            "source": "expression",
            "expression": expression,
            "l2_norm": l2_norm,
            "min_value": min_val,
            "max_value": max_val,
        }

    # Function-based interpolation
    source_info = session.get_function(source_function)
    source_func = source_info.function

    # Same-mesh interpolation
    if source_mesh is None:
        try:
            target_func.interpolate(source_func)
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to interpolate from '{source_function}': {exc}",
                suggestion="Ensure source and target spaces are compatible.",
            ) from exc

        # Postcondition: interpolation result must be finite
        if not np.isfinite(target_func.x.array).all():
            raise DOLFINxAPIError(
                "Interpolation produced NaN/Inf values.",
                suggestion="Check source expression or function for validity.",
            )

        # Compute statistics
        l2_norm = float(np.linalg.norm(target_func.x.array))
        min_val = float(np.min(target_func.x.array))
        max_val = float(np.max(target_func.x.array))

        if __debug__:
            session.check_invariants()

        logger.info("Interpolated '%s' into '%s'", source_function, target)
        return {
            "target": target,
            "source": source_function,
            "interpolation_type": "same_mesh",
            "l2_norm": l2_norm,
            "min_value": min_val,
            "max_value": max_val,
        }

    # Cross-mesh interpolation
    try:
        # Get source mesh
        source_mesh_info = session.get_mesh(source_mesh)
        target_space_info = session.get_space(target_info.space_name)
        target_space = target_space_info.space

        # Create interpolation data
        interpolation_data = dolfinx.fem.create_interpolation_data(
            target_space.mesh._cpp_object,
            target_space.element,
            source_mesh_info.mesh._cpp_object,
        )

        # Perform interpolation
        target_func.interpolate(source_func, cells=interpolation_data)
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed cross-mesh interpolation: {exc}",
            suggestion="Ensure meshes are compatible and source function is defined appropriately.",
        ) from exc

    # Postcondition: interpolation result must be finite
    if not np.isfinite(target_func.x.array).all():
        raise DOLFINxAPIError(
            "Interpolation produced NaN/Inf values.",
            suggestion="Check source expression or function for validity.",
        )

    # Compute statistics
    l2_norm = float(np.linalg.norm(target_func.x.array))
    min_val = float(np.min(target_func.x.array))
    max_val = float(np.max(target_func.x.array))

    if __debug__:
        session.check_invariants()

    logger.info(
        "Cross-mesh interpolated '%s' (mesh: %s) into '%s'",
        source_function, source_mesh, target,
    )
    return {
        "target": target,
        "source": source_function,
        "source_mesh": source_mesh,
        "interpolation_type": "cross_mesh",
        "l2_norm": l2_norm,
        "min_value": min_val,
        "max_value": max_val,
    }


@mcp.tool()
@handle_tool_errors
async def create_discrete_operator(
    operator_type: str,
    source_space: str,
    target_space: str,
    name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a discrete differential operator matrix.

    Args:
        operator_type: Type of operator: "gradient", "curl", or "interpolation".
        source_space: Name of the source function space.
        target_space: Name of the target function space.
        name: Name to store the operator. Auto-generated if omitted.
    """
    # Precondition: validate operator_type before expensive imports
    if operator_type not in ("gradient", "curl", "interpolation"):
        raise PreconditionError(
            f"operator_type must be 'gradient', 'curl', or 'interpolation', got '{operator_type}'."
        )

    import dolfinx.fem

    session = _get_session(ctx)

    # Resolve function spaces
    source_info = session.get_space(source_space)
    target_info = session.get_space(target_space)

    V_source = source_info.space
    V_target = target_info.space

    # Generate name if not provided
    if name is None:
        name = f"op_{operator_type}_{len(session.ufl_symbols)}"

    # Create the operator
    try:
        if operator_type == "gradient":
            operator = dolfinx.fem.discrete_gradient(V_source, V_target)
        elif operator_type == "curl":
            operator = dolfinx.fem.discrete_curl(V_source, V_target)
        elif operator_type == "interpolation":
            operator = dolfinx.fem.petsc.interpolation_matrix(V_source, V_target)
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to create {operator_type} operator: {exc}",
            suggestion="Check that source and target spaces are compatible for this operator.",
        ) from exc

    # Get matrix dimensions
    rows, cols = operator.getSize()
    nnz = operator.getInfo()["nz_used"]

    # Store in session (using ufl_symbols dict with prefix for now)
    operator_key = f"_operator_{name}"
    session.ufl_symbols[operator_key] = operator

    if __debug__:
        session.check_invariants()

    logger.info(
        "Created %s operator '%s': %dx%d matrix, %d NNZ",
        operator_type, name, rows, cols, int(nnz),
    )
    return {
        "name": name,
        "operator_type": operator_type,
        "source_space": source_space,
        "target_space": target_space,
        "matrix_size": {"rows": int(rows), "cols": int(cols)},
        "nnz": int(nnz),
    }
