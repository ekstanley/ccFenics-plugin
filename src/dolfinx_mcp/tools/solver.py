"""Linear and nonlinear PDE solver tools."""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp.server.fastmcp import Context

from .._app import get_session, mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    FunctionNotFoundError,
    PostconditionError,
    PreconditionError,
    SolverError,
    handle_tool_errors,
)
from ..session import SolutionInfo
from ._validators import require_nonempty, require_positive

logger = logging.getLogger(__name__)


_VALID_NULLSPACE_MODES = frozenset({"constant", "rigid_body"})


def _validate_solver_tolerances(rtol: float, atol: float, max_iter: int) -> None:
    """Validate iterative solver tolerance parameters."""
    require_positive(rtol, "rtol")
    require_positive(atol, "atol")
    require_positive(max_iter, "max_iter")


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
    nullspace_mode: str | None = None,
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
        nullspace_mode: Nullspace handling for singular systems.
            "constant" — attach a constant nullspace (pure Neumann scalar problems).
            "rigid_body" — attach rigid body modes (pure Neumann elasticity).
            None — no nullspace (default).

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
    # Precondition: validate nullspace_mode
    if nullspace_mode is not None and nullspace_mode not in _VALID_NULLSPACE_MODES:
        raise PreconditionError(
            f"nullspace_mode must be one of {sorted(_VALID_NULLSPACE_MODES)} or None, "
            f"got '{nullspace_mode}'."
        )
    # S1: solution_name must be non-empty
    require_nonempty(solution_name, "solution_name")
    # S2: validate iterative solver parameters
    if solver_type == "iterative":
        _validate_solver_tolerances(rtol, atol, max_iter)

    import dolfinx.fem.petsc
    import numpy as np

    session = get_session(ctx)

    # Retrieve forms via accessor (postcondition-checked)
    a_form = session.get_form("bilinear", suggestion="Use define_variational_form first.").ufl_form
    L_form = session.get_form("linear", suggestion="Use define_variational_form first.").ufl_form

    # Collect boundary conditions
    bcs = [bc_info.bc for bc_info in session.bcs.values()]

    # Build PETSc options
    opts = _build_petsc_opts(solver_type, ksp_type, pc_type, petsc_options, rtol, atol, max_iter)

    # Auto-MUMPS for mixed-space direct solves (saddle-point systems need it)
    if solver_type == "direct" and "pc_factor_mat_solver_type" not in opts:
        a_info = session.forms.get("bilinear")
        if (a_info is not None
                and a_info.trial_space_name
                and a_info.trial_space_name in session.function_spaces
                and session.function_spaces[a_info.trial_space_name].element_family == "Mixed"):
            opts["pc_factor_mat_solver_type"] = "mumps"
            logger.info("Auto-selected MUMPS for mixed-space direct solve")

    # Solve
    t0 = time.perf_counter()
    try:
        problem = dolfinx.fem.petsc.LinearProblem(
            a_form, L_form, bcs=bcs,
            petsc_options=opts, petsc_options_prefix="solve",
        )

        # Attach nullspace if requested (singular systems like pure Neumann)
        if nullspace_mode is not None:
            from petsc4py import PETSc

            A = problem.A
            if nullspace_mode == "constant":
                nullspace = PETSc.NullSpace().create(constant=True)
            elif nullspace_mode == "rigid_body":
                # Build rigid body modes for vector problems
                V = list(session.function_spaces.values())[-1].space
                rb_vectors = _build_rigid_body_modes(V)
                nullspace = PETSc.NullSpace().create(
                    constant=False, vectors=rb_vectors,
                )
            A.setNullSpace(nullspace)
            nullspace.remove(problem.b)
            logger.info("Attached %s nullspace to solver matrix", nullspace_mode)

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

    # Postcondition: L2 norm must be non-negative (mathematically guaranteed)
    if __debug__:
        if l2_norm < 0:
            raise PostconditionError(f"L2 norm must be non-negative, got {l2_norm}.")

    # Identify space name
    space_name = session.find_space_name(uh.function_space)

    # Store solution (cache l2_norm to avoid recomputation in diagnostics)
    sol_info = SolutionInfo(
        name=solution_name,
        function=uh,
        space_name=space_name,
        converged=converged,
        iterations=iterations,
        residual_norm=float(residual_norm),
        wall_time=wall_time,
        l2_norm=l2_norm,
    )
    session.solutions[solution_name] = sol_info

    # Also register as a function for post-processing (allow re-solve overwrite)
    if solution_name in session.functions:
        del session.functions[solution_name]
    session.register_function(
        solution_name, uh, space_name,
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


def _build_rigid_body_modes(V: Any) -> list[Any]:
    """Build rigid body mode vectors for a vector function space.

    Used for elasticity nullspace (3 modes in 2D: 2 translations + 1 rotation;
    6 modes in 3D: 3 translations + 3 rotations).

    Args:
        V: dolfinx FunctionSpace (vector-valued)

    Returns:
        List of PETSc Vec objects representing rigid body modes
    """
    import dolfinx.fem
    from petsc4py import PETSc

    gdim = V.mesh.geometry.dim
    dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    def _create_vec(func: dolfinx.fem.Function) -> PETSc.Vec:
        vec = func.x.petsc_vec.copy()
        vec.array[:] = func.x.array[:]
        norm = vec.norm()
        if norm > 0:
            vec.scale(1.0 / norm)
        return vec

    modes = []
    # Translation modes
    for i in range(gdim):
        f = dolfinx.fem.Function(V)
        f.x.array[:] = 0.0
        bs = V.dofmap.index_map_bs
        f.x.array[i::bs] = 1.0
        modes.append(_create_vec(f))

    # Rotation modes
    x = V.tabulate_dof_coordinates()[:dofs // V.dofmap.index_map_bs]
    if gdim == 2:
        # Single rotation in 2D: (-y, x)
        f = dolfinx.fem.Function(V)
        f.x.array[:] = 0.0
        bs = V.dofmap.index_map_bs
        f.x.array[0::bs] = -x[:, 1]
        f.x.array[1::bs] = x[:, 0]
        modes.append(_create_vec(f))
    elif gdim == 3:
        for axis in range(3):
            f = dolfinx.fem.Function(V)
            f.x.array[:] = 0.0
            bs = V.dofmap.index_map_bs
            i, j = (axis + 1) % 3, (axis + 2) % 3
            f.x.array[i::bs] = -x[:, j]
            f.x.array[j::bs] = x[:, i]
            modes.append(_create_vec(f))

    # Orthogonalize via Gram-Schmidt
    for i in range(len(modes)):
        for j in range(i):
            dot = modes[i].dot(modes[j])
            modes[i].axpy(-dot, modes[j])
        norm = modes[i].norm()
        if norm > 0:
            modes[i].scale(1.0 / norm)

    return modes


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
    import math

    # Preconditions
    # S3: all time parameters must be finite
    if not math.isfinite(dt):
        raise PreconditionError(f"dt must be finite, got {dt}.")
    if not math.isfinite(t_end):
        raise PreconditionError(f"t_end must be finite, got {t_end}.")
    if not math.isfinite(t_start):
        raise PreconditionError(f"t_start must be finite, got {t_start}.")
    if dt <= 0:
        raise PreconditionError(f"dt must be > 0, got {dt}.")
    if t_end <= t_start:
        raise PreconditionError(f"t_end ({t_end}) must be > t_start ({t_start}).")
    if time_scheme != "backward_euler":
        raise PreconditionError(
            f"time_scheme must be 'backward_euler', got '{time_scheme}'."
        )
    # S1: solution_name must be non-empty
    require_nonempty(solution_name, "solution_name")
    if output_times is not None:
        for i, t in enumerate(output_times):
            if not isinstance(t, (int, float)) or not math.isfinite(t):
                raise PreconditionError(f"output_times[{i}] must be a finite number, got {t}.")
        if any(t < t_start or t > t_end for t in output_times):
            raise PreconditionError(
                f"All output_times must be in [{t_start}, {t_end}].",
                suggestion=f"Got times outside range: "
                f"{[t for t in output_times if t < t_start or t > t_end]}",
            )

    max_steps = int((t_end - t_start) / dt)
    if max_steps > 1_000_000:
        raise PreconditionError(
            f"Time integration would require {max_steps} steps. Maximum is 1,000,000.",
            suggestion="Increase dt or reduce the time range.",
        )

    import dolfinx.fem.petsc
    import numpy as np

    session = get_session(ctx)

    # Retrieve forms via accessor (postcondition-checked)
    a_form = session.get_form("bilinear", suggestion="Use define_variational_form first.").ufl_form
    L_form = session.get_form("linear", suggestion="Use define_variational_form first.").ufl_form
    bcs = [bc_info.bc for bc_info in session.bcs.values()]

    # Build solver options
    opts = _build_petsc_opts(solver_type, ksp_type, pc_type, petsc_options)

    # S4: warn if u_n not in session (time-stepping won't update previous solution)
    if "u_n" not in session.functions:
        logger.warning(
            "No 'u_n' function in session. Time-stepping will not update previous solution."
        )

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

    # Postcondition: time loop must reach t_end (within one dt)
    if __debug__:
        if abs(t - t_end) > dt:
            raise PostconditionError(
                f"Time loop ended at t={t:.6e}, expected t_end={t_end:.6e} "
                f"(gap > dt={dt:.6e}).",
                suggestion="Check time step size and loop condition.",
            )

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
        l2_norm=l2_norm,
    )
    session.solutions[solution_name] = sol_info

    # Also register as function (allow re-solve overwrite)
    if solution_name in session.functions:
        del session.functions[solution_name]
    session.register_function(
        solution_name, uh, space_name,
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
    session = get_session(ctx)

    # Get the last solution
    sol_info = session.get_last_solution()

    # Compute number of DOFs
    V = sol_info.function.function_space
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # Use cached L2 norm (computed at solve time, avoids form recompilation)
    l2_norm = sol_info.l2_norm

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
    require_nonempty(residual, "residual")

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
    _validate_solver_tolerances(rtol, atol, max_iter)

    session = get_session(ctx)

    # PRE-4: unknown exists in session.functions
    if unknown not in session.functions:
        available = list(session.functions.keys())
        raise FunctionNotFoundError(
            f"Unknown function '{unknown}' not found. Available: {available}",
            suggestion="Create the unknown function first via project "
            "(e.g. project(name='u', target_space='V', expression='0.0')).",
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

    # POST-2: L2 norm must be non-negative (mathematically guaranteed)
    if __debug__:
        if l2_norm < 0:
            raise PostconditionError(f"L2 norm must be non-negative, got {l2_norm}.")

    # Get residual norm from SNES if available
    residual_norm = 0.0
    import contextlib
    with contextlib.suppress(Exception):
        residual_norm = float(snes.getFunctionNorm())

    # Register solution (cache l2_norm to avoid recomputation)
    sol_name = solution_name or unknown
    sol_info = SolutionInfo(
        name=sol_name,
        function=u_func,
        space_name=space_name,
        converged=bool(converged),
        iterations=int(n_iters),
        residual_norm=residual_norm,
        wall_time=wall_time,
        l2_norm=l2_norm,
    )
    session.solutions[sol_name] = sol_info

    # Also register/update as a function (allow re-solve overwrite)
    if sol_name in session.functions:
        del session.functions[sol_name]
    session.register_function(
        sol_name, u_func, space_name,
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


_VALID_EPS_TYPES = frozenset({"krylovschur", "lanczos", "arnoldi", "lapack", "power"})
_VALID_WHICH_EIGENVALUES = frozenset({
    "smallest_magnitude", "largest_magnitude",
    "smallest_real", "largest_real",
    "target_magnitude", "target_real",
})


@mcp.tool()
@handle_tool_errors
async def solve_eigenvalue(
    stiffness_form: str,
    mass_form: str,
    num_eigenvalues: int = 6,
    which: str = "smallest_magnitude",
    target: float | None = None,
    eps_type: str = "krylovschur",
    function_space: str | None = None,
    solution_prefix: str = "eig",
    ctx: Context = None,
) -> dict[str, Any]:
    """Solve a generalized eigenvalue problem A*x = lambda*B*x.

    Assembles stiffness (A) and mass (B) matrices from UFL form strings
    and uses SLEPc's EPS solver to find eigenvalues and eigenvectors.

    Args:
        stiffness_form: UFL bilinear form string for the stiffness matrix A
            (e.g. "inner(grad(u), grad(v))*dx").
        mass_form: UFL bilinear form string for the mass matrix B
            (e.g. "inner(u, v)*dx").
        num_eigenvalues: Number of eigenvalues to compute (1-1000).
        which: Which eigenvalues to target. One of: smallest_magnitude,
            largest_magnitude, smallest_real, largest_real,
            target_magnitude, target_real.
        target: Target value for spectral transformation (required when
            which starts with "target_").
        eps_type: SLEPc eigensolver type. One of: krylovschur, lanczos,
            arnoldi, lapack, power.
        function_space: Function space name. Defaults to the active space.
        solution_prefix: Prefix for eigenvector names (e.g. "eig" produces
            "eig_0", "eig_1", ...).

    Returns:
        dict with num_converged, eigenvalues (list of {index, real, imag}),
        eigenvectors (list of names), wall_time, and solver_type.
    """
    # PRE-1: stiffness_form non-empty
    require_nonempty(stiffness_form, "stiffness_form")
    # PRE-2: mass_form non-empty
    require_nonempty(mass_form, "mass_form")
    # PRE-3: num_eigenvalues in [1, 1000]
    if not 1 <= num_eigenvalues <= 1000:
        raise PreconditionError(
            f"num_eigenvalues must be in [1, 1000], got {num_eigenvalues}.",
        )
    # PRE-4: which valid
    if which not in _VALID_WHICH_EIGENVALUES:
        raise PreconditionError(
            f"which must be one of {sorted(_VALID_WHICH_EIGENVALUES)}, got '{which}'.",
        )
    # PRE-5: target required for target_ modes
    if which.startswith("target_") and target is None:
        raise PreconditionError(
            f"target value is required when which='{which}'.",
            suggestion="Provide a target eigenvalue for spectral transformation.",
        )
    # PRE-6: eps_type valid
    if eps_type not in _VALID_EPS_TYPES:
        raise PreconditionError(
            f"eps_type must be one of {sorted(_VALID_EPS_TYPES)}, got '{eps_type}'.",
        )

    # Eager forbidden-token check on expression strings
    from ..ufl_context import _check_forbidden
    _check_forbidden(stiffness_form)
    _check_forbidden(mass_form)

    import dolfinx.fem
    import dolfinx.fem.petsc
    import numpy as np
    import ufl

    session = get_session(ctx)

    # Resolve function space
    space_name = function_space
    if space_name is None:
        # Use first available function space
        if not session.function_spaces:
            raise PreconditionError(
                "No function spaces defined. Create one first.",
                suggestion="Use create_function_space() before solve_eigenvalue().",
            )
        space_name = next(iter(session.function_spaces))
    space_info = session.get_space(space_name)
    V = space_info.space

    # Build UFL namespace and evaluate forms
    from ..ufl_context import build_namespace, safe_evaluate

    ns = build_namespace(session, space_info.mesh_name)
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    ns["u"] = u_trial
    ns["v"] = v_test

    a_form = safe_evaluate(stiffness_form, ns)
    b_form = safe_evaluate(mass_form, ns)

    t0 = time.perf_counter()

    try:
        # Collect BCs for this function space
        bcs_list = [
            bc_info.bc for bc_info in session.bcs.values()
            if bc_info.space_name == space_name
        ]

        # Compile and assemble matrices with BCs applied.
        # assemble_matrix(bcs=...) correctly zeros rows/columns for constrained
        # DOFs and sets diagonal to 1.0, preserving matrix symmetry.
        # This creates spurious eigenvalue=1.0 (diag(A)/diag(B) = 1) which
        # we filter out below.
        a_compiled = dolfinx.fem.form(a_form)
        b_compiled = dolfinx.fem.form(b_form)

        A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=bcs_list)
        A.assemble()
        B = dolfinx.fem.petsc.assemble_matrix(b_compiled, bcs=bcs_list)
        B.assemble()

        # Request extra eigenvalues to compensate for BC-induced spurious
        # eigenvalue at 1.0. We'll filter it out after solving.
        request_nev = num_eigenvalues + (1 if bcs_list else 0)

        # Create SLEPc EPS solver
        from slepc4py import SLEPc

        eps = SLEPc.EPS().create(A.getComm())
        eps.setOperators(A, B)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        # Configure solver type
        eps_type_map = {
            "krylovschur": SLEPc.EPS.Type.KRYLOVSCHUR,
            "lanczos": SLEPc.EPS.Type.LANCZOS,
            "arnoldi": SLEPc.EPS.Type.ARNOLDI,
            "lapack": SLEPc.EPS.Type.LAPACK,
            "power": SLEPc.EPS.Type.POWER,
        }
        eps.setType(eps_type_map[eps_type])

        # Configure which eigenvalues to find
        which_map = {
            "smallest_magnitude": SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
            "largest_magnitude": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
            "smallest_real": SLEPc.EPS.Which.SMALLEST_REAL,
            "largest_real": SLEPc.EPS.Which.LARGEST_REAL,
            "target_magnitude": SLEPc.EPS.Which.TARGET_MAGNITUDE,
            "target_real": SLEPc.EPS.Which.TARGET_REAL,
        }
        eps.setWhichEigenpairs(which_map[which])

        if target is not None:
            eps.setTarget(target)
            st = eps.getST()
            st.setType(SLEPc.ST.Type.SINVERT)

        eps.setDimensions(nev=request_nev)
        eps.setFromOptions()
        eps.solve()

    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise SolverError(
            f"Eigenvalue solver failed: {exc}",
            suggestion="Check stiffness and mass forms. Ensure the problem is well-posed.",
        ) from exc

    wall_time = time.perf_counter() - t0

    # Extract results
    nconv = eps.getConverged()

    # POST-1: at least one eigenvalue must converge
    if nconv <= 0:
        raise SolverError(
            f"Eigenvalue solver converged 0 eigenvalues (requested {num_eigenvalues}).",
            suggestion="Increase matrix size, check forms, or try a different eps_type.",
        )

    eigenvalues = []
    eigenvector_names = []

    # When BCs are applied, assemble_matrix sets diagonal=1 for constrained
    # DOFs in both A and B, creating a spurious eigenvalue at exactly 1.0.
    # We detect and skip these (tolerance for float comparison).
    _BC_SPURIOUS_TOL = 1e-10
    out_idx = 0

    for i in range(nconv):
        if out_idx >= num_eigenvalues:
            break

        # Get eigenvalue
        eigval = eps.getEigenvalue(i)

        # Skip BC-induced spurious eigenvalue = 1.0
        if bcs_list and abs(eigval.real - 1.0) < _BC_SPURIOUS_TOL and abs(eigval.imag) < _BC_SPURIOUS_TOL:
            continue

        eigenvalues.append({
            "index": out_idx,
            "real": float(eigval.real),
            "imag": float(eigval.imag),
        })

        # Get eigenvector
        vr = dolfinx.fem.Function(V)
        vi = dolfinx.fem.Function(V)
        eps.getEigenvector(i, vr.x.petsc_vec, vi.x.petsc_vec)
        vr.x.scatter_forward()

        # POST-2: eigenvector must be finite
        if not np.isfinite(vr.x.array).all():
            raise PostconditionError(
                f"Eigenvector {out_idx} contains NaN or Inf values."
            )

        # Register eigenvector in session
        vec_name = f"{solution_prefix}_{out_idx}"
        if vec_name in session.functions:
            del session.functions[vec_name]
        session.register_function(
            vec_name, vr, space_name,
            description=f"Eigenvector {out_idx} (eigenvalue={eigval.real:.6g})",
        )
        eigenvector_names.append(vec_name)
        out_idx += 1

    # POST-3: eigenvalues list non-empty
    if not eigenvalues:
        raise PostconditionError("Eigenvalues list is empty despite nconv > 0.")

    # INV-1: session invariants
    if __debug__:
        session.check_invariants()

    # Clean up
    eps.destroy()

    logger.info(
        "Eigenvalue solve: %d converged, eps_type=%s, which=%s, wall_time=%.3fs",
        nconv, eps_type, which, wall_time,
    )

    return {
        "num_converged": nconv,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvector_names,
        "wall_time": round(wall_time, 4),
        "solver_type": eps_type,
    }
