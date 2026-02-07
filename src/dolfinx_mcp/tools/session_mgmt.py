"""Session management tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import DOLFINxAPIError, DOLFINxMCPError, PostconditionError, PreconditionError, handle_tool_errors
from ..session import SessionState

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


@mcp.tool()
@handle_tool_errors
async def get_session_state(
    ctx: Context = None,
) -> dict[str, Any]:
    """Get the current session state: all meshes, spaces, functions, BCs, forms, solutions.

    Use this to inspect what objects are available and their properties.

    Returns:
        dict with active_mesh (str or null), meshes (dict of mesh summaries),
        function_spaces (dict), functions (dict), boundary_conditions (dict),
        forms (dict), solutions (dict), mesh_tags (dict), entity_maps (dict),
        and ufl_symbols (list of registered symbol names).
    """
    session = _get_session(ctx)

    if __debug__:
        session.check_invariants()

    return session.overview()


@mcp.tool()
@handle_tool_errors
async def reset_session(
    ctx: Context = None,
) -> dict[str, Any]:
    """Clear all session state: meshes, spaces, functions, BCs, forms, solutions.

    This resets the session to a clean state, removing all registered objects.

    Returns:
        dict with status ("reset") and message (confirmation string).
    """
    session = _get_session(ctx)
    session.cleanup()

    if __debug__:
        session.check_invariants()

    return {"status": "reset", "message": "Session state cleared"}


@mcp.tool()
@handle_tool_errors
async def run_custom_code(
    code: str,
    capture_output: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Execute arbitrary Python code in the session context.

    SECURITY NOTE: This executes user-provided code with full Python access.
    Safe only because it runs inside a Docker container with:
    - Network isolation (--network none)
    - Non-root user
    - Ephemeral container (--rm)
    - No host filesystem access

    The execution namespace includes:
    - session: The current SessionState object
    - dolfinx, ufl, numpy, mpi4py: Scientific computing modules
    - All session-registered functions, meshes, spaces, etc.

    Args:
        code: Python code to execute
        capture_output: If True, capture and return stdout/stderr

    Returns:
        dict with "output" (captured text) and "error" (if any)
    """
    if not code or not code.strip():
        raise PreconditionError("Code string must be non-empty.")

    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout

    session = _get_session(ctx)

    # Lazy import scientific modules
    try:
        import dolfinx
        import numpy as np
        import ufl
        from mpi4py import MPI
    except ImportError as e:
        return {"output": "", "error": f"Import error: {e}"}

    # Build execution namespace
    exec_ns = {
        "session": session,
        "dolfinx": dolfinx,
        "ufl": ufl,
        "np": np,
        "MPI": MPI,
        "__builtins__": __builtins__,
    }

    # Add all session-registered objects
    exec_ns.update(session.functions)
    exec_ns.update(session.meshes)
    exec_ns.update(session.function_spaces)

    # Execute code
    output_text = ""
    error_text = None

    try:
        if capture_output:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_ns)
            output_text = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                error_text = stderr_capture.getvalue()
        else:
            exec(code, exec_ns)
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"

    if __debug__:
        session.check_invariants()

    return {"output": output_text, "error": error_text}


@mcp.tool()
@handle_tool_errors
async def assemble(
    target: str,
    form: str,
    name: str | None = None,
    apply_bcs: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Manually assemble a UFL form into a scalar, vector, or matrix.

    Args:
        target: Assembly target type: "scalar", "vector", or "matrix"
        form: UFL expression string to assemble
        name: Optional name for the assembled object (for vector/matrix)
        apply_bcs: If True, apply boundary conditions (for vector/matrix)

    Returns:
        dict with assembly results depending on target type:
        - scalar: {"value": float}
        - vector: {"norm": float, "size": int}
        - matrix: {"dims": [int, int], "nnz": int}
    """
    # Precondition: validate target type before expensive imports
    if target not in ("scalar", "vector", "matrix"):
        raise PreconditionError(
            f"target must be 'scalar', 'vector', or 'matrix', got '{target}'."
        )

    import dolfinx.fem
    import dolfinx.fem.petsc

    from ..ufl_context import build_namespace, safe_evaluate

    session = _get_session(ctx)

    # Build UFL namespace from session
    ufl_namespace = build_namespace(session)

    # Evaluate form expression
    try:
        ufl_form = safe_evaluate(form, ufl_namespace)
    except DOLFINxMCPError:
        raise
    except Exception as e:
        raise DOLFINxAPIError(
            f"Form evaluation failed: {e}",
            suggestion="Check UFL expression syntax and available symbols.",
        ) from e

    # Assemble based on target type
    try:
        if target == "scalar":
            import numpy as np
            fem_form = dolfinx.fem.form(ufl_form)
            scalar_val = float(dolfinx.fem.assemble_scalar(fem_form))
            if not np.isfinite(scalar_val):
                raise DOLFINxAPIError(
                    "Assembly produced NaN/Inf scalar value.",
                    suggestion="Check form expression and boundary conditions.",
                )
            if __debug__:
                session.check_invariants()
            return {"value": scalar_val}

        elif target == "vector":
            fem_form = dolfinx.fem.form(ufl_form)
            vec = dolfinx.fem.petsc.assemble_vector(fem_form)
            vec.ghostUpdate(
                addv=dolfinx.cpp.la.InsertMode.add,
                mode=dolfinx.cpp.la.ScatterMode.reverse,
            )

            if apply_bcs:
                # Apply BCs from session
                bcs = list(session.bcs.values())
                if bcs:
                    dolfinx.fem.petsc.set_bc(vec, bcs)

            vec.ghostUpdate(
                addv=dolfinx.cpp.la.InsertMode.insert,
                mode=dolfinx.cpp.la.ScatterMode.forward,
            )

            norm = vec.norm()
            size = vec.getSize()

            if __debug__:
                session.check_invariants()
            return {"norm": float(norm), "size": int(size)}

        elif target == "matrix":
            fem_form = dolfinx.fem.form(ufl_form)
            bcs = list(session.bcs.values()) if apply_bcs else []
            mat = dolfinx.fem.petsc.assemble_matrix(fem_form, bcs=bcs)
            mat.assemble()

            dims = mat.getSize()
            nnz = mat.getInfo()["nz_used"]

            if __debug__:
                session.check_invariants()
            return {"dims": list(dims), "nnz": int(nnz)}

    except DOLFINxMCPError:
        raise
    except Exception as e:
        raise DOLFINxAPIError(
            f"Assembly failed: {e}",
            suggestion="Check form expression and boundary conditions.",
        ) from e


_VALID_OBJECT_TYPES = frozenset({
    "mesh", "space", "function", "bc", "form",
    "solution", "mesh_tags", "entity_map",
})

# Mapping from object_type to (registry_attr, display_name)
_REGISTRY_MAP = {
    "function": ("functions", "Function"),
    "bc": ("bcs", "Boundary condition"),
    "form": ("forms", "Form"),
    "solution": ("solutions", "Solution"),
    "mesh_tags": ("mesh_tags", "Mesh tags"),
    "entity_map": ("entity_maps", "Entity map"),
}


@mcp.tool()
@handle_tool_errors
async def remove_object(
    name: str,
    object_type: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Remove a named object from the session.

    For meshes, this cascades: all dependent spaces, functions, BCs,
    solutions, mesh tags, and entity maps are also removed. For spaces,
    dependent functions, BCs, and solutions are removed. Leaf types
    (function, bc, form, solution, mesh_tags, entity_map) are removed
    directly.

    Args:
        name: Name of the object to remove.
        object_type: One of: mesh, space, function, bc, form, solution,
            mesh_tags, entity_map.

    Returns:
        dict with removed object name, type, and cascade information.
    """
    # Preconditions: validate before any imports or session access
    if not name or not name.strip():
        raise PreconditionError("name must be non-empty.")
    if object_type not in _VALID_OBJECT_TYPES:
        raise PreconditionError(
            f"object_type must be one of {sorted(_VALID_OBJECT_TYPES)}, "
            f"got '{object_type}'."
        )

    session = _get_session(ctx)

    removed = {"name": name, "object_type": object_type}

    try:
        if object_type == "mesh":
            # Cascade deletion via existing session method
            session.remove_mesh(name)
            removed["cascade"] = True

        elif object_type == "space":
            if name not in session.function_spaces:
                raise DOLFINxAPIError(
                    f"Function space '{name}' not found.",
                    suggestion="Check available spaces with get_session_state.",
                )
            session._remove_space_dependents(name)
            del session.function_spaces[name]
            removed["cascade"] = True

        else:
            # Leaf types: direct deletion
            registry_attr, display_name = _REGISTRY_MAP[object_type]
            registry = getattr(session, registry_attr)
            if name not in registry:
                raise DOLFINxAPIError(
                    f"{display_name} '{name}' not found.",
                    suggestion="Check available objects with get_session_state.",
                )
            del registry[name]
            removed["cascade"] = False

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to remove {object_type} '{name}': {exc}",
            suggestion="Check the object exists and try again.",
        ) from exc

    # Postcondition: object no longer in its registry
    registry_attr = {
        "mesh": "meshes", "space": "function_spaces",
    }.get(object_type, _REGISTRY_MAP.get(object_type, (None,))[0])
    if registry_attr and name in getattr(session, registry_attr):
        raise PostconditionError(
            f"remove_object(): '{name}' still present in {registry_attr} after removal."
        )

    if __debug__:
        session.check_invariants()

    logger.info("Removed %s '%s'", object_type, name)
    return removed
