"""Interpolation and discrete operator tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    PostconditionError,
    PreconditionError,
    SolverError,
    handle_tool_errors,
)
from ..session import SessionState

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


def _eval_interp_expression(expr: str, x: Any) -> Any:
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

    Returns:
        dict with target (str), source (str), l2_norm (float), min_value (float),
        max_value (float), and optionally interpolation_type ("same_mesh" or
        "cross_mesh"), source_mesh (str), and expression (str).
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
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to interpolate expression '{expression}': {exc}",
                suggestion="Check expression syntax. Use x[0], x[1], x[2] for coordinates.",
            ) from exc

        # Postcondition: interpolation result must be finite
        if not np.isfinite(target_func.x.array).all():
            raise PostconditionError(
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
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to interpolate from '{source_function}': {exc}",
                suggestion="Ensure source and target spaces are compatible.",
            ) from exc

        # Postcondition: interpolation result must be finite
        if not np.isfinite(target_func.x.array).all():
            raise PostconditionError(
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
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed cross-mesh interpolation: {exc}",
            suggestion="Ensure meshes are compatible and source function is defined appropriately.",
        ) from exc

    # Postcondition: interpolation result must be finite
    if not np.isfinite(target_func.x.array).all():
        raise PostconditionError(
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

    Returns:
        dict with name (str), operator_type (str), source_space (str),
        target_space (str), matrix_size (dict with rows/cols), and nnz (int).
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
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to create {operator_type} operator: {exc}",
            suggestion="Check that source and target spaces are compatible for this operator.",
        ) from exc

    # Get matrix dimensions
    rows, cols = operator.getSize()
    nnz = operator.getInfo()["nz_used"]

    # Postcondition: operator matrix must have positive dimensions
    if rows <= 0 or cols <= 0:
        raise PostconditionError(
            f"Discrete operator has invalid dimensions: {rows}x{cols}; expected both > 0."
        )

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


@mcp.tool()
@handle_tool_errors
async def project(
    name: str,
    target_space: str,
    expression: str | None = None,
    source_function: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """L2-project an expression or function onto a target function space.

    Solves the mass matrix system M*u = b where M = inner(u,v)*dx and
    b = inner(source, v)*dx. This is useful when interpolation is not
    available (e.g., projecting expressions that cannot be point-evaluated).

    Exactly one of ``expression`` or ``source_function`` must be provided.

    Args:
        name: Name for the projected function result.
        target_space: Name of the target function space.
        expression: UFL expression string to project.
        source_function: Name of an existing function to project.

    Returns:
        dict with name, l2_norm, min_value, max_value of the projection.
    """
    # Preconditions: validate before lazy imports
    if not name or not name.strip():
        raise PreconditionError("name must be non-empty.")
    if expression is not None and source_function is not None:
        raise PreconditionError(
            "Exactly one of 'expression' or 'source_function' must be provided, not both."
        )
    if expression is None and source_function is None:
        raise PreconditionError(
            "Exactly one of 'expression' or 'source_function' must be provided."
        )

    import dolfinx.fem
    import dolfinx.fem.petsc
    import numpy as np
    import ufl

    session = _get_session(ctx)

    # Resolve target space via accessor
    space_info = session.get_space(target_space)
    V = space_info.space

    # Build test/trial functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Build the source term for the RHS
    if source_function is not None:
        fn_info = session.get_function(source_function)
        source = fn_info.function
    else:
        # Evaluate UFL expression string
        from ..ufl_context import build_namespace, safe_evaluate

        ufl_namespace = build_namespace(session)
        try:
            source = safe_evaluate(expression, ufl_namespace)
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Expression evaluation failed: {exc}",
                suggestion="Check UFL expression syntax.",
            ) from exc

    # Assemble and solve: M*u_h = b
    a_form = ufl.inner(u, v) * ufl.dx
    L_form = ufl.inner(source, v) * ufl.dx

    try:
        problem = dolfinx.fem.petsc.LinearProblem(
            a_form, L_form, bcs=[],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise SolverError(
            f"L2 projection solve failed: {exc}",
            suggestion="Check expression/function compatibility with target space.",
        ) from exc

    # Postcondition: result must be finite
    if not np.isfinite(uh.x.array).all():
        raise PostconditionError(
            "Projection produced NaN/Inf values.",
            suggestion="Check source expression/function and target space compatibility.",
        )

    # Compute statistics
    l2_norm = float(np.linalg.norm(uh.x.array))
    min_val = float(np.min(uh.x.array))
    max_val = float(np.max(uh.x.array))

    # Postcondition: L2 norm must be non-negative
    if l2_norm < 0:
        raise PostconditionError(f"L2 norm must be non-negative, got {l2_norm}.")

    # Store result as function in session
    from ..session import FunctionInfo
    session.functions[name] = FunctionInfo(
        name=name,
        function=uh,
        space_name=target_space,
        description=f"L2 projection of {'expression' if expression else source_function}",
    )

    if __debug__:
        session.check_invariants()

    logger.info(
        "Projected %s into '%s' (space: %s), L2 norm=%.6e",
        expression or source_function, name, target_space, l2_norm,
    )
    return {
        "name": name,
        "l2_norm": round(l2_norm, 8),
        "min_value": round(min_val, 8),
        "max_value": round(max_val, 8),
    }
