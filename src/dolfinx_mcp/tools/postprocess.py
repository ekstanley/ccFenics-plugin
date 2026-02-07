"""Post-processing tools: error computation, solution export.

NOTE ON EVAL USAGE: Exact solution expressions are Python/numpy syntax
evaluated in restricted namespaces (no builtins). The Docker container
provides the security boundary.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import DOLFINxAPIError, DOLFINxMCPError, FunctionNotFoundError, PostconditionError, PreconditionError, handle_tool_errors
from ..session import SessionState

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


def _eval_exact_expression(expr: str, x) -> Any:
    """Evaluate an exact solution expression at coordinate arrays.

    SECURITY: Restricted namespace, Docker-sandboxed.
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
    ns.setdefault("__builtins__", {})
    result = eval(expr, ns)  # noqa: S307 -- restricted namespace, Docker-sandboxed
    if isinstance(result, (int, float)):
        return np.full(x.shape[1], float(result))
    return result


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
    """
    # Preconditions
    if not exact or not exact.strip():
        raise PreconditionError("exact expression must be non-empty.")
    if norm_type.upper() not in ("L2", "H1"):
        raise PreconditionError(
            f"norm_type must be 'L2' or 'H1', got '{norm_type}'."
        )

    import dolfinx.fem
    import numpy as np
    import ufl

    session = _get_session(ctx)

    # Find the computed solution
    if function_name is not None:
        if function_name not in session.functions:
            raise FunctionNotFoundError(
                f"Function '{function_name}' not found.",
                suggestion="Check available functions with get_session_state.",
            )
        uh = session.functions[function_name].function
        space_name = session.functions[function_name].space_name
    else:
        # Use the most recently stored solution
        if not session.solutions:
            raise DOLFINxAPIError(
                "No solutions available. Run solve first.",
                suggestion="Use the solve tool to compute a solution.",
            )
        last_sol = list(session.solutions.values())[-1]
        uh = last_sol.function
        space_name = last_sol.space_name

    V = uh.function_space

    # Interpolate exact solution into the same space
    u_exact = dolfinx.fem.Function(V)
    try:
        u_exact.interpolate(
            lambda x: _eval_exact_expression(exact, x)
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

    if norm_type.upper() == "L2":
        error_form = dolfinx.fem.form(ufl.inner(error_func, error_func) * ufl.dx)
        error_val = float(np.sqrt(abs(dolfinx.fem.assemble_scalar(error_form))))
    elif norm_type.upper() == "H1":
        error_form = dolfinx.fem.form(
            (ufl.inner(error_func, error_func)
             + ufl.inner(ufl.grad(error_func), ufl.grad(error_func))) * ufl.dx
        )
        error_val = float(np.sqrt(abs(dolfinx.fem.assemble_scalar(error_form))))

    import math
    if not math.isfinite(error_val):
        raise PostconditionError(
            f"Error norm must be finite, got {error_val}."
        )
    if error_val < 0:
        raise PostconditionError(f"Error norm must be non-negative, got {error_val}.")

    logger.info("Computed %s error: %.6e", norm_type, error_val)
    return {
        "norm_type": norm_type.upper(),
        "error_value": error_val,
        "function_name": function_name or list(session.solutions.keys())[-1],
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
    """
    # Precondition: validate format before expensive imports
    fmt = format.lower()
    if fmt not in ("xdmf", "vtk"):
        raise PreconditionError(f"format must be 'xdmf' or 'vtk', got '{format}'.")

    from mpi4py import MPI
    import dolfinx.io

    session = _get_session(ctx)

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

    # Ensure output directory exists
    output_dir = "/workspace"
    filepath = os.path.join(output_dir, filename)

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
    """
    # Preconditions
    if not points:
        raise PreconditionError("points list must be non-empty.")

    import dolfinx.geometry
    import numpy as np

    session = _get_session(ctx)

    # Find the function to evaluate
    if function_name is not None:
        if function_name not in session.functions:
            raise FunctionNotFoundError(
                f"Function '{function_name}' not found.",
                suggestion="Check available functions with get_session_state.",
            )
        uh = session.functions[function_name].function
    else:
        # Use the most recently stored solution
        if not session.solutions:
            raise DOLFINxAPIError(
                "No solutions available. Run solve first.",
                suggestion="Use the solve tool to compute a solution.",
            )
        last_sol = list(session.solutions.values())[-1]
        uh = last_sol.function

    mesh = uh.function_space.mesh

    # Convert points to numpy array and pad to 3D if needed
    points_array = np.array(points, dtype=np.float64)

    # Validate all points are finite
    if not np.isfinite(points_array).all():
        raise PreconditionError("All point coordinates must be finite (no NaN/Inf).")

    if points_array.shape[1] == 2:
        # Pad with zeros for z-coordinate
        points_array = np.column_stack([points_array, np.zeros(len(points_array))])
    elif points_array.shape[1] != 3:
        raise DOLFINxAPIError(
            f"Points must be 2D or 3D, got shape {points_array.shape}",
            suggestion="Provide points as [[x1, y1], [x2, y2], ...] or [[x1, y1, z1], ...]",
        )

    # Transpose to shape (3, N) as required by DOLFINx
    points_array = points_array.T

    # Build bounding box tree and find colliding cells
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points_array.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points_array.T)

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

    logger.info("Evaluated solution at %d points", len(points))
    return {
        "function_name": function_name or list(session.solutions.keys())[-1],
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
    """
    # Precondition: expressions must be non-empty
    if not expressions:
        raise PreconditionError("expressions list must be non-empty.")

    import dolfinx.fem

    from ..ufl_context import build_namespace, safe_evaluate

    session = _get_session(ctx)

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
    """
    # Preconditions
    if not points:
        raise PreconditionError("points list must be non-empty.")
    if tolerance <= 0:
        raise PreconditionError(f"tolerance must be > 0, got {tolerance}.")

    import dolfinx.geometry
    import numpy as np

    session = _get_session(ctx)

    # Find the function to evaluate
    if function_name is not None:
        if function_name not in session.functions:
            raise FunctionNotFoundError(
                f"Function '{function_name}' not found.",
                suggestion="Check available functions with get_session_state.",
            )
        uh = session.functions[function_name].function
    else:
        # Use the most recently stored solution
        if not session.solutions:
            raise DOLFINxAPIError(
                "No solutions available. Run solve first.",
                suggestion="Use the solve tool to compute a solution.",
            )
        last_sol = list(session.solutions.values())[-1]
        uh = last_sol.function

    mesh = uh.function_space.mesh

    # Convert points to numpy array and pad to 3D if needed
    points_array = np.array(points, dtype=np.float64)

    # Validate all points are finite
    if not np.isfinite(points_array).all():
        raise PreconditionError("All point coordinates must be finite (no NaN/Inf).")

    if points_array.shape[1] == 2:
        # Pad with zeros for z-coordinate
        points_array = np.column_stack([points_array, np.zeros(len(points_array))])
    elif points_array.shape[1] != 3:
        raise DOLFINxAPIError(
            f"Points must be 2D or 3D, got shape {points_array.shape}",
            suggestion="Provide points as [[x1, y1], [x2, y2], ...] or [[x1, y1, z1], ...]",
        )

    # Transpose to shape (3, N) as required by DOLFINx
    points_array = points_array.T

    # Build bounding box tree and find colliding cells
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points_array.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points_array.T)

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

    logger.info("Queried solution at %d points (tolerance=%.2e)", len(points), tolerance)
    return {
        "function_name": function_name or list(session.solutions.keys())[-1],
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
    ctx: Context = None,
) -> dict[str, Any]:
    """Generate a 2D visualization plot of a solution function.

    Args:
        function_name: Name of the function to plot. Defaults to the last solution.
        plot_type: Type of plot -- "contour" or "warp".
        colormap: Matplotlib colormap name (e.g., "viridis", "plasma", "coolwarm").
        show_mesh: Whether to show mesh edges on the plot.
        output_file: Output filename. Defaults to "/workspace/plot.png".
    """
    # Precondition: validate plot_type before expensive imports
    if plot_type not in ("contour", "warp"):
        raise PreconditionError(f"plot_type must be 'contour' or 'warp', got '{plot_type}'.")

    session = _get_session(ctx)

    # Find the function to plot
    if function_name is not None:
        if function_name not in session.functions:
            raise FunctionNotFoundError(
                f"Function '{function_name}' not found.",
                suggestion="Check available functions with get_session_state.",
            )
        uh = session.functions[function_name].function
    else:
        # Use the most recently stored solution
        if not session.solutions:
            raise DOLFINxAPIError(
                "No solutions available. Run solve first.",
                suggestion="Use the solve tool to compute a solution.",
            )
        last_sol = list(session.solutions.values())[-1]
        uh = last_sol.function

    V = uh.function_space

    # Try to import pyvista
    try:
        import pyvista
        from dolfinx.plot import vtk_mesh
    except ImportError as exc:
        raise DOLFINxAPIError(
            "PyVista is not available for plotting.",
            suggestion="Install pyvista in the container or use export_solution + external tools.",
        ) from exc

    # Enable headless rendering
    pyvista.OFF_SCREEN = True

    try:
        # Create VTK mesh
        topology, cell_types, geometry = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["solution"] = uh.x.array.real

        # Create plotter
        plotter = pyvista.Plotter(off_screen=True)

        if plot_type == "contour":
            plotter.add_mesh(grid, scalars="solution", cmap=colormap, show_edges=show_mesh)
        elif plot_type == "warp":
            warped = grid.warp_by_scalar("solution", factor=0.1)
            plotter.add_mesh(warped, scalars="solution", cmap=colormap, show_edges=show_mesh)

        # Save screenshot
        output_path = output_file or "/workspace/plot.png"
        plotter.screenshot(output_path)
        plotter.close()

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        logger.info("Generated %s plot at %s (%d bytes)", plot_type, output_path, file_size)
        return {
            "file_path": output_path,
            "plot_type": plot_type,
            "file_size_bytes": file_size,
        }

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to generate plot: {exc}",
            suggestion="Check that the function is compatible with visualization.",
        ) from exc
