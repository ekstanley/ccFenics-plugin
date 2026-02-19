"""Post-processing tools: error computation, solution export.

NOTE ON EVAL USAGE: Exact solution expressions are Python/numpy syntax
evaluated in restricted namespaces (no builtins). The Docker container
provides the security boundary.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import Context

from .._app import get_session, mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    FileIOError,
    FunctionNotFoundError,
    PostconditionError,
    PreconditionError,
    handle_tool_errors,
)
from ..eval_helpers import eval_numpy_expression
from ._validators import require_nonempty, require_positive

logger = logging.getLogger(__name__)


def _prepare_point_eval(mesh: Any, points_list: list) -> tuple:
    """Validate points, pad to 3D, build BB tree, find colliding cells.

    Returns:
        (points_array, colliding_cells) where points_array has shape (3, N).
    """
    import numpy as np

    import dolfinx.geometry

    points_array = np.array(points_list, dtype=np.float64)
    if not np.isfinite(points_array).all():
        raise PreconditionError("All point coordinates must be finite (no NaN/Inf).")
    if points_array.shape[1] == 2:
        points_array = np.column_stack([points_array, np.zeros(len(points_array))])
    elif points_array.shape[1] != 3:
        raise DOLFINxAPIError(
            f"Points must be 2D or 3D, got shape {points_array.shape}",
            suggestion="Provide points as [[x1, y1], [x2, y2], ...] or [[x1, y1, z1], ...]",
        )
    points_array = points_array.T
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points_array.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, candidates, points_array.T,
    )
    return points_array, colliding_cells


def _extract_mixed_subspace(uh: Any, V: Any, sub_index: int) -> tuple:
    """Extract sub-function from mixed space.

    Returns:
        (uh_sub, V_sub) -- the collapsed sub-function and sub-space.
    """
    import dolfinx.fem

    V_sub, sub_map = V.sub(sub_index).collapse()
    uh_sub = dolfinx.fem.Function(V_sub)
    uh_sub.x.array[:] = uh.x.array[sub_map]
    return uh_sub, V_sub


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect stdout (including C-level fd 1) to /dev/null.

    VTK/PyVista initialization emits text to stdout which corrupts
    the MCP stdio JSON-RPC transport. This guard suppresses all
    stdout writes during VTK operations.

    Gracefully degrades to no-op if stdout has no file descriptor
    (e.g., in pytest capture mode).
    """
    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, io.UnsupportedOperation):
        yield
        return

    saved_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stdout_fd)
        yield
    finally:
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)
        os.close(devnull)


def _validate_output_path(path: str) -> str:
    """Validate and resolve an output file path within /workspace.

    Args:
        path: Raw file path from tool arguments.

    Returns:
        Resolved absolute path within /workspace.

    Raises:
        FileIOError: If the resolved path escapes /workspace.
    """
    resolved = os.path.realpath(os.path.join("/workspace", path))
    if not resolved.startswith("/workspace"):
        raise FileIOError(
            f"Output path must be within /workspace, got '{path}'.",
            suggestion="Use a simple filename like 'result.xdmf' or "
            "a relative path within /workspace.",
        )
    return resolved


@mcp.tool()
@handle_tool_errors
async def compute_error(
    exact: str,
    norm_type: str = "L2",
    function_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Compute the error between a computed solution and an exact solution.

    Args:
        exact: Exact solution as a math expression string.
            Use x[0], x[1] for spatial coordinates, pi for constants.
            Example: "sin(pi*x[0])*sin(pi*x[1])"
        norm_type: Error norm type -- "L2" or "H1".
        function_name: Name of the computed solution. Defaults to the last solution.

    Returns:
        dict with norm_type (str), error_value (float), and
        function_name (str, the computed solution used).
    """
    # Preconditions
    require_nonempty(exact, "exact expression")
    if norm_type.upper() not in ("L2", "H1"):
        raise PreconditionError(
            f"norm_type must be 'L2' or 'H1', got '{norm_type}'."
        )

    import dolfinx.fem
    import numpy as np
    import ufl

    session = get_session(ctx)

    # Find the computed solution
    if function_name is not None:
        fn_info = session.get_function(function_name)
        uh = fn_info.function
    else:
        last_sol = session.get_last_solution()
        uh = last_sol.function

    V = uh.function_space

    # Interpolate exact solution into the same space
    u_exact = dolfinx.fem.Function(V)
    try:
        u_exact.interpolate(
            lambda x: eval_numpy_expression(exact, x)
        )
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to interpolate exact solution: {exc}",
            suggestion="Check expression syntax. Use x[0], x[1] for coordinates.",
        ) from exc

    # Compute error
    error_func = dolfinx.fem.Function(V)
    error_func.x.array[:] = uh.x.array - u_exact.x.array
    dx = ufl.Measure("dx", domain=V.mesh)

    if norm_type.upper() == "L2":
        error_form = dolfinx.fem.form(ufl.inner(error_func, error_func) * dx)
        error_val = float(np.sqrt(abs(dolfinx.fem.assemble_scalar(error_form))))
    elif norm_type.upper() == "H1":
        error_form = dolfinx.fem.form(
            (ufl.inner(error_func, error_func)
             + ufl.inner(ufl.grad(error_func), ufl.grad(error_func))) * dx
        )
        error_val = float(np.sqrt(abs(dolfinx.fem.assemble_scalar(error_form))))

    import math
    if not math.isfinite(error_val):
        raise PostconditionError(
            f"Error norm must be finite, got {error_val}."
        )
    if error_val < 0:
        raise PostconditionError(f"Error norm must be non-negative, got {error_val}.")

    if __debug__:
        session.check_invariants()

    logger.info("Computed %s error: %.6e", norm_type, error_val)
    return {
        "norm_type": norm_type.upper(),
        "error_value": error_val,
        "function_name": function_name or session.get_last_solution().name,
    }


@mcp.tool()
@handle_tool_errors
async def export_solution(
    filename: str,
    format: str = "xdmf",
    functions: list[str] | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Export solution functions to a file for visualization.

    Args:
        filename: Output filename (written to /workspace/).
        format: File format -- "xdmf" or "vtk".
        functions: Names of functions to export. Defaults to all solutions.

    Returns:
        dict with file_path (str), format (str), file_size_bytes (int),
        and functions_exported (list of function names).
    """
    # Preconditions: validate inputs before expensive imports
    fmt = format.lower()
    if fmt not in ("xdmf", "vtk"):
        raise PreconditionError(f"format must be 'xdmf' or 'vtk', got '{format}'.")

    # Validate and resolve output path within /workspace
    filepath = _validate_output_path(filename)

    import dolfinx.io
    from mpi4py import MPI

    session = get_session(ctx)

    # Resolve functions to export
    if functions is not None:
        funcs_to_export = {}
        for fname in functions:
            if fname in session.functions:
                funcs_to_export[fname] = session.functions[fname].function
            elif fname in session.solutions:
                funcs_to_export[fname] = session.solutions[fname].function
            else:
                raise FunctionNotFoundError(
                    f"Function '{fname}' not found.",
                    suggestion="Check available functions with get_session_state.",
                )
    else:
        funcs_to_export = {
            name: sol.function for name, sol in session.solutions.items()
        }
        if not funcs_to_export:
            raise DOLFINxAPIError(
                "No functions to export.",
                suggestion="Solve the problem first, or specify function names.",
            )

    if fmt == "xdmf":
        if not filepath.endswith(".xdmf"):
            filepath += ".xdmf"
        try:
            with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filepath, "w") as xdmf:
                first_func = next(iter(funcs_to_export.values()))
                xdmf.write_mesh(first_func.function_space.mesh)
                for fname, func in funcs_to_export.items():
                    func.name = fname
                    xdmf.write_function(func)
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(f"Failed to write XDMF file: {exc}") from exc

    elif fmt == "vtk":
        if not filepath.endswith(".pvd"):
            filepath += ".pvd"
        try:
            with dolfinx.io.VTKFile(MPI.COMM_WORLD, filepath, "w") as vtk:
                for fname, func in funcs_to_export.items():
                    func.name = fname
                    vtk.write_function(func)
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(f"Failed to write VTK file: {exc}") from exc

    file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

    # Postcondition: exported file must be non-empty
    if file_size <= 0:
        raise PostconditionError(
            f"Export produced empty file at '{filepath}'.",
            suggestion="Check that functions contain data and output path is writable.",
        )

    if __debug__:
        session.check_invariants()

    logger.info("Exported %d functions to %s (%s)", len(funcs_to_export), filepath, fmt)
    return {
        "file_path": filepath,
        "format": fmt,
        "file_size_bytes": file_size,
        "functions_exported": list(funcs_to_export.keys()),
    }


@mcp.tool()
@handle_tool_errors
async def evaluate_solution(
    points: list[list[float]],
    function_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Evaluate a solution function at specified points.

    Args:
        points: List of coordinates where to evaluate. Each point is a list [x, y] or [x, y, z].
        function_name: Name of the function to evaluate. Defaults to the last solution.

    Returns:
        dict with function_name (str), num_points (int), and evaluations
        (list of dicts, each with point and value; value is null if outside mesh).
    """
    # Preconditions
    if not points:
        raise PreconditionError("points list must be non-empty.")

    import numpy as np

    session = get_session(ctx)

    # Find the function to evaluate
    if function_name is not None:
        uh = session.get_function(function_name).function
    else:
        uh = session.get_last_solution().function

    mesh = uh.function_space.mesh
    points_array, colliding_cells = _prepare_point_eval(mesh, points)

    # Evaluate at each point
    results = []
    for i, point in enumerate(points):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            # Point is inside mesh, evaluate at first colliding cell
            point_3d = points_array[:, i]
            value = uh.eval(point_3d, cells[0])
            # Postcondition: evaluated values must be finite
            if not np.isfinite(value).all():
                raise PostconditionError(
                    f"evaluate_solution(): non-finite value at point {point}"
                )
            results.append({
                "point": point,
                "value": value.tolist() if hasattr(value, 'tolist') else float(value),
            })
        else:
            # Point is outside mesh
            results.append({
                "point": point,
                "value": None,
            })

    if __debug__:
        session.check_invariants()

    logger.info("Evaluated solution at %d points", len(points))
    return {
        "function_name": function_name or session.get_last_solution().name,
        "num_points": len(points),
        "evaluations": results,
    }


@mcp.tool()
@handle_tool_errors
async def compute_functionals(
    expressions: list[str],
    ctx: Context = None,
) -> dict[str, Any]:
    """Compute functionals by integrating UFL expressions over the domain.

    Args:
        expressions: List of UFL expression strings to integrate.
            Example: ["u*u*dx", "inner(grad(u), grad(u))*dx"]

    Returns:
        dict with num_functionals (int) and functionals (list of dicts,
        each with expression (str) and value (float)).
    """
    # Precondition: expressions must be non-empty
    if not expressions:
        raise PreconditionError("expressions list must be non-empty.")
    # P3: each expression must be a non-empty string
    for i, expr in enumerate(expressions):
        if not isinstance(expr, str) or not expr.strip():
            raise PreconditionError(f"expressions[{i}] must be a non-empty string.")

    import dolfinx.fem

    from ..ufl_context import build_namespace, safe_evaluate

    session = get_session(ctx)

    if not session.solutions and not session.functions:
        raise DOLFINxAPIError(
            "No functions available for functional computation.",
            suggestion="Create a solution or function first.",
        )

    # Build UFL namespace with available functions
    namespace = build_namespace(session)

    results = []
    for expr_str in expressions:
        try:
            # Evaluate expression in UFL namespace
            expr = safe_evaluate(expr_str, namespace)

            # Compile and assemble
            form = dolfinx.fem.form(expr)
            value = dolfinx.fem.assemble_scalar(form)

            # Postcondition: assembled value must be finite
            import math
            fval = float(value)
            if not math.isfinite(fval):
                raise PostconditionError(
                    f"Functional '{expr_str}' produced non-finite value: {fval}."
                )

            results.append({
                "expression": expr_str,
                "value": fval,
            })
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to compute functional '{expr_str}': {exc}",
                suggestion="Check expression syntax. Available: dx, ds, inner, grad, etc.",
            ) from exc

    if __debug__:
        session.check_invariants()

    logger.info("Computed %d functionals", len(expressions))
    return {
        "num_functionals": len(expressions),
        "functionals": results,
    }


@mcp.tool()
@handle_tool_errors
async def query_point_values(
    points: list[list[float]],
    function_name: str | None = None,
    tolerance: float = 1e-12,
    ctx: Context = None,
) -> dict[str, Any]:
    """Query solution values at points with detailed geometric information.

    Args:
        points: List of coordinates where to query. Each point is a list [x, y] or [x, y, z].
        function_name: Name of the function to query. Defaults to the last solution.
        tolerance: Geometric tolerance for point location.

    Returns:
        dict with function_name (str), num_points (int), tolerance (float),
        and queries (list of dicts, each with point, value, and cell_index;
        value and cell_index are null if outside mesh).
    """
    # Preconditions
    if not points:
        raise PreconditionError("points list must be non-empty.")
    require_positive(tolerance, "tolerance")

    import numpy as np

    session = get_session(ctx)

    # Find the function to evaluate
    if function_name is not None:
        uh = session.get_function(function_name).function
    else:
        uh = session.get_last_solution().function

    mesh = uh.function_space.mesh
    points_array, colliding_cells = _prepare_point_eval(mesh, points)

    # Evaluate at each point with cell information
    results = []
    for i, point in enumerate(points):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            # Point is inside mesh, evaluate at first colliding cell
            point_3d = points_array[:, i]
            value = uh.eval(point_3d, cells[0])
            # Postcondition: evaluated values must be finite
            if not np.isfinite(value).all():
                raise PostconditionError(
                    f"query_point_values(): non-finite value at point {point}"
                )
            results.append({
                "point": point,
                "value": value.tolist() if hasattr(value, 'tolist') else float(value),
                "cell_index": int(cells[0]),
            })
        else:
            # Point is outside mesh
            results.append({
                "point": point,
                "value": None,
                "cell_index": None,
            })

    if __debug__:
        session.check_invariants()

    logger.info("Queried solution at %d points (tolerance=%.2e)", len(points), tolerance)
    return {
        "function_name": function_name or session.get_last_solution().name,
        "num_points": len(points),
        "tolerance": tolerance,
        "queries": results,
    }


@mcp.tool()
@handle_tool_errors
async def plot_solution(
    function_name: str | None = None,
    plot_type: str = "contour",
    colormap: str = "viridis",
    show_mesh: bool = False,
    output_file: str | None = None,
    component: int | None = None,
    return_base64: bool = False,
    ctx: Context = None,
) -> dict[str, Any]:
    """Generate a 2D visualization plot of a solution function.

    Supports both scalar and vector function spaces. For vector fields,
    plots the magnitude by default, or a specific component if specified.

    Args:
        function_name: Name of the function to plot. Defaults to the last solution.
        plot_type: Type of plot -- "contour" or "warp".
        colormap: Matplotlib colormap name (e.g., "viridis", "plasma", "coolwarm").
        show_mesh: Whether to show mesh edges on the plot.
        output_file: Output filename. Defaults to "/workspace/plot.png".
        component: For vector fields, which component to plot (0-indexed).
            If None (default), plots the magnitude for vector fields.
            Ignored for scalar fields.
        return_base64: If True, include the PNG image as a base64-encoded string
            in the response under the "image_base64" key.

    Returns:
        dict with file_path (str), plot_type (str), file_size_bytes (int),
        is_vector (bool), and optionally component_plotted, num_components,
        and image_base64.
    """
    # Preconditions: validate inputs before expensive imports
    if plot_type not in ("contour", "warp"):
        raise PreconditionError(f"plot_type must be 'contour' or 'warp', got '{plot_type}'.")

    # PRE-V1: component validation
    if component is not None and (not isinstance(component, int) or component < 0):
        raise PreconditionError(
            f"component must be a non-negative integer, got {component!r}."
        )

    # Validate output path within /workspace (early, before session/imports)
    output_path = _validate_output_path(output_file or "plot.png")

    session = get_session(ctx)

    # Find the function to plot
    mixed_info = None
    if function_name is not None:
        func_info = session.get_function(function_name)
    else:
        func_info = session.get_last_solution()
    uh = func_info.function
    V = uh.function_space

    # Detect mixed space via session registry (not DOLFINx API, to avoid mock issues)
    is_mixed = (
        func_info.space_name in session.function_spaces
        and session.function_spaces[func_info.space_name].element_family == "Mixed"
    )

    # Try to import pyvista
    try:
        import numpy as np
        import pyvista
        from dolfinx.plot import vtk_mesh
    except ImportError as exc:
        raise DOLFINxAPIError(
            "PyVista is not available for plotting.",
            suggestion="Install pyvista in the container or use export_solution + external tools.",
        ) from exc

    # Handle mixed function spaces: extract sub-function for plotting
    # vtk_mesh() only supports CG/DG Lagrange, not mixed spaces
    if is_mixed:
        n_sub = V.num_sub_spaces
        sub_idx = component if component is not None else 0
        if sub_idx >= n_sub:
            raise PreconditionError(
                f"component={sub_idx} out of range for mixed space with {n_sub} sub-spaces.",
                suggestion=f"Valid sub-space indices: 0 to {n_sub - 1}.",
            )
        uh, V = _extract_mixed_subspace(uh, V, sub_idx)
        component = None  # Already extracted, don't re-index
        mixed_info = f"Extracted sub-space {sub_idx} from mixed space for plotting."
        logger.info(mixed_info)

    # PRE-DG0: DG0 has cell-wise constants -- PyVista warp needs vertex data
    if plot_type == "warp" and func_info.space_name in session.function_spaces:
        sp = session.function_spaces[func_info.space_name]
        if sp.element_degree == 0 and sp.element_family in (
            "DG", "Discontinuous Lagrange",
        ):
            raise PreconditionError(
                "Warp plots not supported for DG0 (piecewise constant) functions. "
                "DG0 has cell-wise constant values with no vertex data for warping.",
                suggestion="Use plot_type='contour' instead, or project to CG1 first.",
            )

    # Detect vector vs scalar field
    # DOLFINx 0.10+: _BasixElement uses reference_value_shape, not value_shape
    value_shape = V.ufl_element().reference_value_shape
    is_vector = len(value_shape) > 0
    num_components = value_shape[0] if is_vector else 1

    # PRE-V2: component range check for vector fields
    if component is not None and is_vector and component >= num_components:
        raise PreconditionError(
            f"component={component} out of range for {num_components}-component vector field.",
            suggestion=f"Valid components: 0 to {num_components - 1}.",
        )

    # Enable headless rendering
    pyvista.OFF_SCREEN = True

    try:
        # Suppress VTK/PyVista stdout to protect stdio JSON-RPC transport
        with _suppress_stdout():
            # Create VTK mesh and assign scalar data
            topology, cell_types, geometry = vtk_mesh(V)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

            if is_vector:
                # Reshape DOF array into (num_points, num_components)
                num_points = geometry.shape[0]
                values = uh.x.array.real.reshape(num_points, num_components)

                if component is not None:
                    plot_data = values[:, component]
                    component_plotted = component
                else:
                    # Magnitude: sqrt(sum of squares)
                    plot_data = np.sqrt(np.sum(values**2, axis=1))
                    component_plotted = "magnitude"
            else:
                plot_data = uh.x.array.real
                component_plotted = None

            # POST-V3: scalar data shape check (always-on, not debug-only)
            if plot_data.ndim != 1:
                raise PostconditionError(
                    f"Expected 1D plot data, got shape {plot_data.shape}.",
                    suggestion="Internal error in plot data extraction. Report as bug.",
                )

            grid.point_data["solution"] = plot_data

            # Create plotter
            plotter = pyvista.Plotter(off_screen=True)

            if plot_type == "contour":
                plotter.add_mesh(
                    grid, scalars="solution", cmap=colormap, show_edges=show_mesh,
                )
            elif plot_type == "warp":
                warped = grid.warp_by_scalar("solution", factor=0.1)
                plotter.add_mesh(
                    warped, scalars="solution", cmap=colormap, show_edges=show_mesh,
                )

            # Save screenshot
            plotter.screenshot(output_path)
            plotter.close()

        # Postcondition: plot file must exist after rendering
        if not os.path.exists(output_path):
            raise PostconditionError(
                f"Plot file not created at '{output_path}'.",
                suggestion="Check output directory exists and is writable.",
            )

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        if __debug__:
            session.check_invariants()

        logger.info("Generated %s plot at %s (%d bytes)", plot_type, output_path, file_size)
        result: dict[str, Any] = {
            "file_path": output_path,
            "plot_type": plot_type,
            "file_size_bytes": file_size,
            "is_vector": is_vector,
        }

        if mixed_info is not None:
            result["mixed_space_info"] = mixed_info

        if is_vector:
            result["component_plotted"] = component_plotted
            result["num_components"] = num_components

        # Optionally return base64-encoded PNG
        if return_base64:
            import base64
            with open(output_path, "rb") as f:
                raw = f.read()
            b64_content = base64.b64encode(raw).decode("ascii")
            # POST-B1/B2: base64 content is non-empty and decodable
            if not b64_content:
                raise PostconditionError("Base64 encoding produced empty content.")
            result["image_base64"] = b64_content

        return result

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to generate plot: {exc}",
            suggestion="Check that the function is compatible with visualization.",
        ) from exc
