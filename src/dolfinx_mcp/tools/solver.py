"""Linear and nonlinear PDE solver tools."""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    FunctionNotFoundError,
    PostconditionError,
    PreconditionError,
    SolverError,
    handle_tool_errors,
)
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

    Returns:
        dict with converged (bool), converged_reason (int), iterations (int),
        residual_norm (float), wall_time (float, seconds), solution_name (str),
        and solution_norm_L2 (float).
    """
    # Precondition: validate solver_type before imports
    if solver_type not in ("direct", "iterative"):
        raise PreconditionError(
            f"solver_type must be 'direct' or 'iterative', got '{solver_type}'."
        )

    import dolfinx.fem.petsc
    import numpy as np

    session = _get_session(ctx)

    # Retrieve forms via accessor (postcondition-checked)
    a_form = session.get_form("bilinear", suggestion="Use define_variational_form first.").ufl_form
    L_form = session.get_form("linear", suggestion="Use define_variational_form first.").ufl_form

    # Collect boundary conditions
    bcs = [bc_info.bc for bc_info in session.bcs.values()]

    # Build PETSc options
    opts = _build_petsc_opts(solver_type, ksp_type, pc_type, petsc_options, rtol, atol, max_iter)

    # Solve
    t0 = time.perf_counter()
    try:
        problem = dolfinx.fem.petsc.LinearProblem(
            a_form, L_form, bcs=bcs,
            petsc_options=opts, petsc_options_prefix="solve",
        )
        uh = problem.solve()
    except DOLFINxMCPError:
        raise
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
    from ..utils import compute_l2_norm
    l2_norm = compute_l2_norm(uh)

    # Postcondition: L2 norm must be non-negative
    if l2_norm < 0:
        raise PostconditionError(f"L2 norm must be non-negative, got {l2_norm}.")

    # Identify space name
    space_name = session.find_space_name(uh.function_space)

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
    if time_scheme != "backward_euler":
        raise PreconditionError(
            f"time_scheme must be 'backward_euler', got '{time_scheme}'."
        )

    import dolfinx.fem.petsc
    import numpy as np

    session = _get_session(ctx)

    # Retrieve forms via accessor (postcondition-checked)
    a_form = session.get_form("bilinear", suggestion="Use define_variational_form first.").ufl_form
    L_form = session.get_form("linear", suggestion="Use define_variational_form first.").ufl_form
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
                a_form, L_form, bcs=bcs,
                petsc_options=opts, petsc_options_prefix="ts",
            )
            uh = problem.solve()
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise SolverError(
                f"Time step {step} at t={t:.3e} failed: {exc}",
                suggestion="Check boundary conditions and forms for time-dependent problem.",
            ) from exc

        # Postcondition: solution must be finite at each timestep
        if not np.isfinite(uh.x.array).all():
            raise SolverError(
                f"Solution at step {step} (t={t:.3e}) contains NaN/Inf.",
                suggestion="Check BCs, forms, and time step size.",
            )

        # Update previous solution (if u_n exists)
        if "u_n" in session.functions:
            session.functions["u_n"].function.x.array[:] = uh.x.array

        # Record snapshot if requested
        if output_times and any(abs(t - ot) < dt / 2 for ot in output_times):
            snapshots.append({"time": round(t, 10), "step": step})
            logger.debug("Recorded snapshot at t=%.3e (step %d)", t, step)

    wall_time_total = time.perf_counter() - t0_total

    # Store final solution
    from ..utils import compute_l2_norm
    l2_norm = compute_l2_norm(uh)

    # Identify space name
    space_name = session.find_space_name(uh.function_space)

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

    if __debug__:
        session.check_invariants()

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
    from ..utils import compute_l2_norm

    session = _get_session(ctx)

    # Get the last solution
    sol_info = session.get_last_solution()

    # Compute number of DOFs
    V = sol_info.function.function_space
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # Compute L2 norm
    l2_norm = compute_l2_norm(sol_info.function)

    # Postcondition: L2 norm must be non-negative
    if l2_norm < 0:
        raise PostconditionError(f"L2 norm must be non-negative, got {l2_norm}.")

    if __debug__:
        session.check_invariants()

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


_VALID_SNES_TYPES = frozenset({"newtonls", "newtontr", "nrichardson"})


@mcp.tool()
@handle_tool_errors
async def solve_nonlinear(
    residual: str,
    unknown: str,
    jacobian: str | None = None,
    snes_type: str = "newtonls",
    ksp_type: str = "preonly",
    pc_type: str = "lu",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int = 50,
    petsc_options: dict[str, Any] | None = None,
    solution_name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Solve a nonlinear variational problem using Newton's method.

    Requires a mutable Function (the unknown) already in the session, typically
    created via interpolate with a zero initial guess. The solver modifies this
    function in-place.

    Args:
        residual: UFL residual form F(u;v) as a string.
        unknown: Name of a Function in the session (mutable unknown).
        jacobian: Optional explicit Jacobian J(u;du,v) as a UFL string.
            If None, auto-derived via ufl.derivative.
        snes_type: PETSc SNES solver type ("newtonls", "newtontr", "nrichardson").
        ksp_type: Inner linear solver KSP type (default "preonly").
        pc_type: Inner preconditioner type (default "lu").
        rtol: Relative tolerance for Newton solver.
        atol: Absolute tolerance for Newton solver.
        max_iter: Maximum Newton iterations.
        petsc_options: Additional PETSc options as key-value pairs.
        solution_name: Name to register the result under (defaults to unknown).

    Returns:
        dict with converged (bool), iterations (int), residual_norm (float),
        wall_time (float), solution_name (str), solution_norm_L2 (float).
    """
    from ..ufl_context import _check_forbidden

    # PRE-1: residual non-empty
    if not residual or not residual.strip():
        raise PreconditionError("residual must be a non-empty string.")

    # PRE-2: residual passes token blocklist
    _check_forbidden(residual)

    # PRE-3: jacobian passes token blocklist if provided
    if jacobian is not None:
        if not jacobian.strip():
            raise PreconditionError("jacobian must be non-empty if provided.")
        _check_forbidden(jacobian)

    # PRE-5: snes_type is valid
    if snes_type not in _VALID_SNES_TYPES:
        raise PreconditionError(
            f"snes_type must be one of {sorted(_VALID_SNES_TYPES)}, got '{snes_type}'."
        )

    # PRE-6: numeric parameters are positive
    if max_iter <= 0:
        raise PreconditionError(f"max_iter must be > 0, got {max_iter}.")
    if rtol <= 0:
        raise PreconditionError(f"rtol must be > 0, got {rtol}.")
    if atol <= 0:
        raise PreconditionError(f"atol must be > 0, got {atol}.")

    session = _get_session(ctx)

    # PRE-4: unknown exists in session.functions
    if unknown not in session.functions:
        available = list(session.functions.keys())
        raise FunctionNotFoundError(
            f"Unknown function '{unknown}' not found. Available: {available}",
            suggestion="Create the unknown function first via interpolate "
            "(e.g. interpolate(name='u', expression='0.0',"
            " function_space='V')).",
        )

    import dolfinx.fem
    import dolfinx.fem.petsc
    import numpy as np
    import ufl

    from ..ufl_context import build_namespace, safe_evaluate

    u_func = session.functions[unknown].function
    space_name = session.functions[unknown].space_name
    V = u_func.function_space

    # Build UFL namespace with session symbols + the unknown and test function
    ns = build_namespace(session)
    ns["u"] = u_func
    ns["v"] = ufl.TestFunction(V)
    ns["du"] = ufl.TrialFunction(V)

    # Evaluate residual form F(u; v)
    F_ufl = safe_evaluate(residual, ns)

    # Derive or evaluate Jacobian J(u; du, v)
    if jacobian is None:
        J_ufl = ufl.derivative(F_ufl, u_func, ufl.TrialFunction(V))
    else:
        J_ufl = safe_evaluate(jacobian, ns)

    # Collect boundary conditions
    bcs = [bc_info.bc for bc_info in session.bcs.values()]

    # Build PETSc options dict for the NonlinearProblem
    petsc_opts: dict[str, Any] = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "snes_type": snes_type,
        "snes_rtol": str(rtol),
        "snes_atol": str(atol),
        "snes_max_it": str(max_iter),
    }
    if petsc_options:
        petsc_opts.update(petsc_options)

    # Create nonlinear problem and solve
    t0 = time.perf_counter()
    try:
        problem = dolfinx.fem.petsc.NonlinearProblem(
            F_ufl, u_func,
            petsc_options_prefix="nlsolve_",
            bcs=bcs, J=J_ufl,
            petsc_options=petsc_opts,
        )
        problem.solve()

        # Extract convergence info from the underlying SNES
        snes = problem.solver
        converged = snes.getConvergedReason() > 0
        n_iters = snes.getIterationNumber()
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise SolverError(
            f"Nonlinear solver failed: {exc}",
            suggestion="Check residual form, boundary conditions, and initial guess. "
            "Try a better initial guess or smaller load step.",
        ) from exc
    wall_time = time.perf_counter() - t0

    # POST-1: solution must be finite
    if not np.isfinite(u_func.x.array).all():
        raise SolverError(
            "Nonlinear solution contains NaN or Inf values.",
            suggestion="Check initial guess, boundary conditions, and residual form. "
            "Consider load stepping for large deformations.",
        )

    # Compute L2 norm
    from ..utils import compute_l2_norm
    l2_norm = compute_l2_norm(u_func)

    # POST-2: L2 norm must be non-negative
    if l2_norm < 0:
        raise PostconditionError(f"L2 norm must be non-negative, got {l2_norm}.")

    # Get residual norm from SNES if available
    residual_norm = 0.0
    import contextlib
    with contextlib.suppress(Exception):
        residual_norm = float(snes.getFunctionNorm())

    # Register solution
    sol_name = solution_name or unknown
    sol_info = SolutionInfo(
        name=sol_name,
        function=u_func,
        space_name=space_name,
        converged=bool(converged),
        iterations=int(n_iters),
        residual_norm=residual_norm,
        wall_time=wall_time,
    )
    session.solutions[sol_name] = sol_info

    # Also register/update as a function
    from ..session import FunctionInfo
    session.functions[sol_name] = FunctionInfo(
        name=sol_name,
        function=u_func,
        space_name=space_name,
        description="Nonlinear solver solution",
    )

    # POST-3: session invariants
    if __debug__:
        session.check_invariants()

    logger.info(
        "Nonlinear solve: converged=%s, iterations=%d, residual=%.2e, wall_time=%.3fs",
        converged, n_iters, residual_norm, wall_time,
    )

    return {
        "converged": bool(converged),
        "iterations": int(n_iters),
        "residual_norm": float(residual_norm),
        "wall_time": round(wall_time, 4),
        "solution_name": sol_name,
        "solution_norm_L2": round(l2_norm, 8),
    }
