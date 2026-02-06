"""Linear and nonlinear PDE solver tools."""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import DOLFINxAPIError, PreconditionError, SolverError, handle_tool_errors
from ..session import SessionState, SolutionInfo

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


@mcp.tool()
@handle_tool_errors
async def solve(
    solver_type: str = "direct",
    ksp_type: str | None = None,
    pc_type: str | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int = 1000,
    petsc_options: dict[str, Any] | None = None,
    solution_name: str = "u_h",
    ctx: Context = None,
) -> dict[str, Any]:
    """Solve the current variational problem.

    Requires that variational forms (bilinear + linear) and boundary
    conditions have been defined.

    Args:
        solver_type: "direct" for LU factorization, "iterative" for Krylov solver.
        ksp_type: PETSc KSP type (e.g. "cg", "gmres"). Only for iterative solver.
            Defaults to "preonly" for direct, "cg" for iterative.
        pc_type: PETSc preconditioner type (e.g. "lu", "ilu", "hypre").
            Defaults to "lu" for direct, "hypre" for iterative.
        rtol: Relative tolerance (iterative only).
        atol: Absolute tolerance (iterative only).
        max_iter: Maximum iterations (iterative only).
        petsc_options: Additional PETSc options as key-value pairs.
        solution_name: Name for the solution function.
    """
    import dolfinx.fem.petsc
    import numpy as np

    session = _get_session(ctx)

    # Validate forms exist
    if "bilinear" not in session.forms:
        raise DOLFINxAPIError(
            "No bilinear form defined.",
            suggestion="Use define_variational_form first.",
        )
    if "linear" not in session.forms:
        raise DOLFINxAPIError(
            "No linear form defined.",
            suggestion="Use define_variational_form first.",
        )

    a_form = session.forms["bilinear"].form
    L_form = session.forms["linear"].form

    # Collect boundary conditions
    bcs = [bc_info.bc for bc_info in session.bcs.values()]

    # Build PETSc options
    opts: dict[str, Any] = {}
    if solver_type == "direct":
        opts["ksp_type"] = ksp_type or "preonly"
        opts["pc_type"] = pc_type or "lu"
    elif solver_type == "iterative":
        opts["ksp_type"] = ksp_type or "cg"
        opts["pc_type"] = pc_type or "hypre"
        opts["ksp_rtol"] = rtol
        opts["ksp_atol"] = atol
        opts["ksp_max_it"] = max_iter
    else:
        raise DOLFINxAPIError(
            f"Unknown solver_type '{solver_type}'.",
            suggestion="Use 'direct' or 'iterative'.",
        )

    if petsc_options:
        opts.update(petsc_options)

    # Solve
    t0 = time.perf_counter()
    try:
        # DOLFINx v0.10: LinearProblem accepts petsc_options dict directly
        problem = dolfinx.fem.petsc.LinearProblem(
            a_form, L_form, bcs=bcs, petsc_options=opts
        )
        uh = problem.solve()
    except Exception as exc:
        raise SolverError(
            f"Solver failed: {exc}",
            suggestion="Check boundary conditions cover the problem. "
            "For iterative solvers, try increasing max_iter or loosening tolerances.",
        ) from exc
    wall_time = time.perf_counter() - t0

    # Postcondition: solution must be finite
    if not np.isfinite(uh.x.array).all():
        raise SolverError(
            "Solution contains NaN or Inf values.",
            suggestion="Check boundary conditions, mesh quality, and solver parameters.",
        )

    # Extract convergence info
    solver = problem.solver
    converged_reason = solver.getConvergedReason()
    iterations = solver.getIterationNumber()
    residual_norm = solver.getResidualNorm() if hasattr(solver, "getResidualNorm") else 0.0

    converged = converged_reason > 0

    # Compute L2 norm of solution
    from dolfinx.fem import assemble_scalar, form as compile_form
    import ufl

    V = uh.function_space
    l2_norm_form = compile_form(ufl.inner(uh, uh) * ufl.dx)
    l2_norm = float(np.sqrt(abs(assemble_scalar(l2_norm_form))))

    # Postcondition: L2 norm must be non-negative
    assert l2_norm >= 0, f"L2 norm negative: {l2_norm}"

    # Identify space name
    space_name = None
    for sname, sinfo in session.function_spaces.items():
        if sinfo.space is V:
            space_name = sname
            break
    if space_name is None:
        space_name = "unknown"

    # Store solution
    sol_info = SolutionInfo(
        name=solution_name,
        function=uh,
        space_name=space_name,
        converged=converged,
        iterations=iterations,
        residual_norm=float(residual_norm),
        wall_time=wall_time,
    )
    session.solutions[solution_name] = sol_info

    # Also register as a function for post-processing
    from ..session import FunctionInfo
    session.functions[solution_name] = FunctionInfo(
        name=solution_name,
        function=uh,
        space_name=space_name,
        description="Solution of variational problem",
    )

    # Debug-mode postcondition: verify session invariants
    if __debug__:
        session.check_invariants()

    logger.info(
        "Solved: converged=%s, iterations=%d, residual=%.2e, wall_time=%.3fs",
        converged, iterations, residual_norm, wall_time,
    )

    return {
        "converged": converged,
        "converged_reason": int(converged_reason),
        "iterations": iterations,
        "residual_norm": float(residual_norm),
        "wall_time": round(wall_time, 4),
        "solution_name": solution_name,
        "solution_norm_L2": round(l2_norm, 8),
    }


def _build_petsc_opts(
    solver_type: str,
    ksp_type: str | None,
    pc_type: str | None,
    petsc_options: dict[str, Any] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Build PETSc options dictionary for solvers.

    Args:
        solver_type: "direct" or "iterative"
        ksp_type: PETSc KSP type override
        pc_type: PETSc preconditioner type override
        petsc_options: Additional PETSc options
        rtol: Relative tolerance (iterative only)
        atol: Absolute tolerance (iterative only)
        max_iter: Maximum iterations (iterative only)

    Returns:
        Dictionary of PETSc options
    """
    opts: dict[str, Any] = {}
    if solver_type == "direct":
        opts["ksp_type"] = ksp_type or "preonly"
        opts["pc_type"] = pc_type or "lu"
    elif solver_type == "iterative":
        opts["ksp_type"] = ksp_type or "cg"
        opts["pc_type"] = pc_type or "hypre"
        opts["ksp_rtol"] = rtol
        opts["ksp_atol"] = atol
        opts["ksp_max_it"] = max_iter
    else:
        raise DOLFINxAPIError(
            f"Unknown solver_type '{solver_type}'.",
            suggestion="Use 'direct' or 'iterative'.",
        )

    if petsc_options:
        opts.update(petsc_options)

    return opts


@mcp.tool()
@handle_tool_errors
async def solve_time_dependent(
    t_end: float,
    dt: float,
    t_start: float = 0.0,
    time_scheme: str = "backward_euler",
    output_times: list[float] | None = None,
    solver_type: str = "direct",
    ksp_type: str | None = None,
    pc_type: str | None = None,
    petsc_options: dict[str, Any] | None = None,
    solution_name: str = "u_h",
    ctx: Context = None,
) -> dict[str, Any]:
    """Solve a time-dependent variational problem.

    Requires bilinear and linear forms with time-dependent terms. The user
    should define u_n (previous timestep solution) in the session namespace.

    Args:
        t_end: End time for simulation
        dt: Time step size
        t_start: Start time (default 0.0)
        time_scheme: Time integration scheme (currently only "backward_euler")
        output_times: Specific times to record snapshots (optional)
        solver_type: "direct" or "iterative"
        ksp_type: PETSc KSP type override
        pc_type: PETSc preconditioner type override
        petsc_options: Additional PETSc options
        solution_name: Name for final solution function

    Returns:
        Dictionary with steps_completed, final_time, snapshots, solution_name
    """
    # Preconditions
    if dt <= 0:
        raise PreconditionError(f"dt must be > 0, got {dt}.")
    if t_end <= t_start:
        raise PreconditionError(f"t_end ({t_end}) must be > t_start ({t_start}).")

    import dolfinx.fem.petsc
    import numpy as np

    session = _get_session(ctx)

    # Validate forms exist
    if "bilinear" not in session.forms:
        raise DOLFINxAPIError(
            "No bilinear form defined.",
            suggestion="Use define_variational_form first.",
        )
    if "linear" not in session.forms:
        raise DOLFINxAPIError(
            "No linear form defined.",
            suggestion="Use define_variational_form first.",
        )

    if time_scheme != "backward_euler":
        raise DOLFINxAPIError(
            f"Unsupported time scheme '{time_scheme}'.",
            suggestion="Currently only 'backward_euler' is supported.",
        )

    a_form = session.forms["bilinear"].form
    L_form = session.forms["linear"].form
    bcs = [bc_info.bc for bc_info in session.bcs.values()]

    # Build solver options
    opts = _build_petsc_opts(solver_type, ksp_type, pc_type, petsc_options)

    # Time integration loop
    t = t_start
    step = 0
    snapshots = []
    t0_total = time.perf_counter()

    logger.info(
        "Starting time integration: t_start=%.3f, t_end=%.3f, dt=%.3e",
        t_start, t_end, dt
    )

    while t < t_end - dt / 2:
        t += dt
        step += 1

        try:
            problem = dolfinx.fem.petsc.LinearProblem(
                a_form, L_form, bcs=bcs, petsc_options=opts
            )
            uh = problem.solve()
        except Exception as exc:
            raise SolverError(
                f"Time step {step} at t={t:.3e} failed: {exc}",
                suggestion="Check boundary conditions and forms for time-dependent problem.",
            ) from exc

        # Update previous solution (if u_n exists)
        if "u_n" in session.functions:
            session.functions["u_n"].function.x.array[:] = uh.x.array

        # Record snapshot if requested
        if output_times and any(abs(t - ot) < dt / 2 for ot in output_times):
            snapshots.append({"time": round(t, 10), "step": step})
            logger.debug("Recorded snapshot at t=%.3e (step %d)", t, step)

    wall_time_total = time.perf_counter() - t0_total

    # Store final solution
    from dolfinx.fem import assemble_scalar, form as compile_form
    import ufl

    V = uh.function_space
    l2_norm_form = compile_form(ufl.inner(uh, uh) * ufl.dx)
    l2_norm = float(np.sqrt(abs(assemble_scalar(l2_norm_form))))

    # Identify space name
    space_name = None
    for sname, sinfo in session.function_spaces.items():
        if sinfo.space is V:
            space_name = sname
            break
    if space_name is None:
        space_name = "unknown"

    sol_info = SolutionInfo(
        name=solution_name,
        function=uh,
        space_name=space_name,
        converged=True,
        iterations=step,
        residual_norm=0.0,
        wall_time=wall_time_total,
    )
    session.solutions[solution_name] = sol_info

    # Also register as function
    from ..session import FunctionInfo
    session.functions[solution_name] = FunctionInfo(
        name=solution_name,
        function=uh,
        space_name=space_name,
        description=f"Time-dependent solution at t={t:.3e}",
    )

    logger.info(
        "Time integration completed: steps=%d, final_t=%.3f, wall_time=%.3fs",
        step, t, wall_time_total
    )

    return {
        "steps_completed": step,
        "final_time": round(t, 10),
        "snapshots": snapshots,
        "solution_name": solution_name,
        "solution_norm_L2": round(l2_norm, 8),
        "wall_time": round(wall_time_total, 4),
    }


@mcp.tool()
@handle_tool_errors
async def get_solver_diagnostics(
    ctx: Context = None,
) -> dict[str, Any]:
    """Get diagnostics from the last solver run.

    Returns information about the most recent solution including solver type,
    convergence, iterations, residual norm, wall time, and solution properties.

    Returns:
        Dictionary with solver diagnostics and solution information
    """
    import numpy as np
    from dolfinx.fem import assemble_scalar, form as compile_form
    import ufl

    session = _get_session(ctx)

    if not session.solutions:
        raise DOLFINxAPIError(
            "No solutions available.",
            suggestion="Run solve() or solve_time_dependent() first.",
        )

    # Get the last solution
    last_solution_name = list(session.solutions.keys())[-1]
    sol_info = session.solutions[last_solution_name]

    # Compute number of DOFs
    V = sol_info.function.function_space
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # Compute L2 norm
    l2_norm_form = compile_form(ufl.inner(sol_info.function, sol_info.function) * ufl.dx)
    l2_norm = float(np.sqrt(abs(assemble_scalar(l2_norm_form))))

    return {
        "solution_name": sol_info.name,
        "space_name": sol_info.space_name,
        "converged": sol_info.converged,
        "iterations": sol_info.iterations,
        "residual_norm": float(sol_info.residual_norm),
        "wall_time": round(sol_info.wall_time, 4),
        "solution_norm_L2": round(l2_norm, 8),
        "num_dofs": num_dofs,
    }
