"""Session management tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import get_session, mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    FileIOError,
    PostconditionError,
    PreconditionError,
    handle_tool_errors,
)
from ._validators import require_nonempty

logger = logging.getLogger(__name__)


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
    session = get_session(ctx)

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
    session = get_session(ctx)
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
    require_nonempty(code, "Code string")

    import io
    from contextlib import redirect_stderr, redirect_stdout

    session = get_session(ctx)

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

    session = get_session(ctx)

    # Build UFL namespace from session
    ufl_namespace = build_namespace(session)

    # Add trial/test functions if a function space is available
    import ufl
    if session.function_spaces:
        space_info = next(iter(session.function_spaces.values()))
        ufl_namespace["u"] = ufl.TrialFunction(space_info.space)
        ufl_namespace["v"] = ufl.TestFunction(space_info.space)

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
            from petsc4py import PETSc

            fem_form = dolfinx.fem.form(ufl_form)
            vec = dolfinx.fem.petsc.assemble_vector(fem_form)
            vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )

            if apply_bcs:
                # Apply BCs from session (extract raw DirichletBC from BCInfo)
                raw_bcs = [bc_info.bc for bc_info in session.bcs.values()]
                if raw_bcs:
                    dolfinx.fem.petsc.set_bc(vec, raw_bcs)

            vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD,
            )

            norm = vec.norm()
            size = vec.getSize()

            if __debug__:
                session.check_invariants()
            return {"norm": float(norm), "size": int(size)}

        elif target == "matrix":
            fem_form = dolfinx.fem.form(ufl_form)
            raw_bcs = [bc_info.bc for bc_info in session.bcs.values()] if apply_bcs else []
            mat = dolfinx.fem.petsc.assemble_matrix(fem_form, bcs=raw_bcs)
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
    require_nonempty(name, "name")
    if object_type not in _VALID_OBJECT_TYPES:
        raise PreconditionError(
            f"object_type must be one of {sorted(_VALID_OBJECT_TYPES)}, "
            f"got '{object_type}'."
        )

    session = get_session(ctx)

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
            session._space_id_to_name.pop(id(session.function_spaces[name].space), None)
            del session.function_spaces[name]
            # INV-9: clear forms referencing deleted space as trial space
            stale_forms = [
                fname for fname, finfo in session.forms.items()
                if finfo.trial_space_name == name
            ]
            for fname in stale_forms:
                del session.forms[fname]
            # INV-8: forms require at least one function_space
            if not session.function_spaces:
                session.forms.clear()
                session.ufl_symbols.clear()
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


# MIME type mapping for auto-detection in read_workspace_file
_BINARY_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".xdmf": "application/octet-stream",
    ".h5": "application/octet-stream",
}

_TEXT_EXTENSIONS = {
    ".pvd": "application/xml",
    ".vtu": "application/xml",
    ".vtk": "application/xml",
    ".csv": "text/csv",
    ".json": "application/json",
    ".txt": "text/plain",
}

_MAX_FILE_SIZE = 10_485_760  # 10 MB


@mcp.tool()
@handle_tool_errors
async def read_workspace_file(
    file_path: str,
    encoding: str = "auto",
    ctx: Context = None,
) -> dict[str, Any]:
    """Read a file from the /workspace/ directory with automatic encoding detection.

    Use this to retrieve images (PNG, JPEG) as base64 or text files (VTK, CSV, JSON)
    as plain text. Useful for MCP clients that cannot directly access Docker volume
    mounts.

    Args:
        file_path: Path to the file, relative to /workspace/ or absolute.
            Examples: "solution.png", "/workspace/output.vtu"
        encoding: How to encode the file content in the response.
            "auto" (default) detects from extension: base64 for images/binary,
            text for VTK/CSV/JSON/txt.
            "base64" forces base64 encoding.
            "text" forces UTF-8 text encoding.

    Returns:
        dict with file_path (str), encoding ("base64" or "text"),
        content (str), file_size_bytes (int), and mime_type (str).
    """
    import base64
    import os
    from pathlib import Path

    # PRE-1: file_path non-empty
    require_nonempty(file_path, "file_path")

    # PRE-2: encoding is valid
    valid_encodings = ("auto", "base64", "text")
    if encoding not in valid_encodings:
        raise PreconditionError(
            f"encoding must be one of {valid_encodings}, got '{encoding}'."
        )

    # Resolve path: treat relative paths as relative to /workspace/
    resolved = (
        Path("/workspace") / file_path
        if not file_path.startswith("/")
        else Path(file_path)
    )

    resolved = resolved.resolve()

    # PRE-3: path traversal check — must be within /workspace/
    workspace_root = Path("/workspace").resolve()
    if not str(resolved).startswith(str(workspace_root) + os.sep) and resolved != workspace_root:
        raise FileIOError(
            f"Path '{file_path}' resolves outside /workspace/.",
            suggestion=(
                "Provide a path within /workspace/"
                " (e.g. 'solution.png' or '/workspace/output.vtu')."
            ),
        )

    # PRE-4: file exists
    if not resolved.is_file():
        raise FileIOError(
            f"File not found: {resolved}",
            suggestion="Check the file path with get_session_state or export_solution first.",
        )

    # PRE-5: file size limit
    file_size = resolved.stat().st_size
    if file_size > _MAX_FILE_SIZE:
        raise PreconditionError(
            f"File size {file_size} bytes exceeds 10MB limit.",
            suggestion=(
                "Use export_solution to write a smaller file,"
                " or access the Docker volume directly."
            ),
        )

    # Determine encoding and MIME type
    ext = resolved.suffix.lower()

    if encoding == "auto":
        if ext in _BINARY_EXTENSIONS:
            actual_encoding = "base64"
            mime_type = _BINARY_EXTENSIONS[ext]
        elif ext in _TEXT_EXTENSIONS:
            actual_encoding = "text"
            mime_type = _TEXT_EXTENSIONS[ext]
        else:
            actual_encoding = "text"
            mime_type = "text/plain"
    else:
        actual_encoding = encoding
        if ext in _BINARY_EXTENSIONS:
            mime_type = _BINARY_EXTENSIONS[ext]
        elif ext in _TEXT_EXTENSIONS:
            mime_type = _TEXT_EXTENSIONS[ext]
        else:
            mime_type = "application/octet-stream" if actual_encoding == "base64" else "text/plain"

    # Read file content
    try:
        if actual_encoding == "base64":
            raw = resolved.read_bytes()
            content = base64.b64encode(raw).decode("ascii")
        else:
            content = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        raise FileIOError(
            f"Failed to read file '{resolved}': {exc}",
            suggestion="Check file permissions and encoding.",
        ) from exc

    # POST-2: content non-empty
    if not content:
        raise PostconditionError(
            "File content is empty.",
            suggestion="The file exists but contains no data.",
        )

    # POST-4: base64 is decodable (always-on for base64)
    if actual_encoding == "base64":
        try:
            decoded = base64.b64decode(content)
        except Exception as err:
            raise PostconditionError(
                "Generated base64 content is not decodable."
            ) from err
        # POST-5: decoded size matches file size
        if len(decoded) != file_size:
            raise PostconditionError(
                f"Base64 decoded size ({len(decoded)}) does not match file size ({file_size})."
            )

    session = get_session(ctx)
    if __debug__:
        session.check_invariants()

    logger.info("Read workspace file '%s' (%s, %d bytes)", resolved, actual_encoding, file_size)
    return {
        "file_path": str(resolved),
        "encoding": actual_encoding,
        "content": content,
        "file_size_bytes": file_size,
        "mime_type": mime_type,
    }
