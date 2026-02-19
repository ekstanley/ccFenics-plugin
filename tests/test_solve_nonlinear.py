"""Tests for solve_nonlinear tool -- preconditions, postconditions, error paths."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from conftest import (
    assert_error_type,
    assert_no_error,
    make_mesh_info,
    make_mock_ctx,
    make_space_info,
)

from dolfinx_mcp.session import (
    FunctionInfo,
    SessionState,
)


def _populated_session() -> SessionState:
    """Session with mesh, space, and a mutable unknown function."""
    s = SessionState()
    s.meshes["m1"] = make_mesh_info("m1")
    s.function_spaces["V"] = make_space_info("V")
    mock_func = MagicMock()
    mock_func.function_space = s.function_spaces["V"].space
    s.functions["u"] = FunctionInfo(
        name="u", function=mock_func, space_name="V",
        description="Initial guess",
    )
    s.active_mesh = "m1"
    return s


def _build_ns_mock():
    """Build a minimal namespace mock for build_namespace."""
    return {
        "__builtins__": {},
        "inner": MagicMock(),
        "grad": MagicMock(),
        "dx": MagicMock(),
        "ds": MagicMock(),
        "x": MagicMock(),
    }


def _setup_solver_mocks(n_iters=3, converged=True, finite=True, l2_norm=1.234):
    """Create standard mocks for solve_nonlinear happy path."""
    mock_dolfinx = MagicMock()
    mock_np = MagicMock()
    mock_ufl = MagicMock()

    # Configure NonlinearProblem mock -- new API uses problem.solver (SNES)
    mock_problem = mock_dolfinx.fem.petsc.NonlinearProblem.return_value
    mock_snes = mock_problem.solver
    mock_snes.getConvergedReason.return_value = 2 if converged else -1
    mock_snes.getIterationNumber.return_value = n_iters
    mock_snes.getFunctionNorm.return_value = 1e-11

    # Configure np.isfinite
    mock_np.isfinite.return_value.all.return_value = finite

    modules = {
        "dolfinx": mock_dolfinx,
        "dolfinx.fem": mock_dolfinx.fem,
        "dolfinx.fem.petsc": mock_dolfinx.fem.petsc,
        "numpy": mock_np,
        "ufl": mock_ufl,
    }
    return modules, l2_norm


# ─── Precondition tests ───────────────────────────────────────────────


class TestSolveNonlinearPreconditions:

    @pytest.mark.asyncio
    async def test_rejects_empty_residual(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(residual="", unknown="u", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_whitespace_residual(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="   ", unknown="u", ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_forbidden_token_in_residual(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="import os; inner(grad(u),grad(v))*dx",
            unknown="u", ctx=ctx,
        )
        assert_error_type(result, "INVALID_UFL_EXPRESSION")

    @pytest.mark.asyncio
    async def test_rejects_forbidden_token_in_jacobian(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            jacobian="__import__('os').system('rm -rf /')",
            ctx=ctx,
        )
        assert_error_type(result, "INVALID_UFL_EXPRESSION")

    @pytest.mark.asyncio
    async def test_rejects_empty_jacobian(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            jacobian="  ",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_unknown_function_not_found(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="nonexistent",
            ctx=ctx,
        )
        assert_error_type(result, "FUNCTION_NOT_FOUND")

    @pytest.mark.asyncio
    async def test_rejects_invalid_snes_type(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            snes_type="invalid_solver",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_negative_max_iter(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            max_iter=-1,
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_zero_max_iter(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            max_iter=0,
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_negative_rtol(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            rtol=-1e-10,
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_rejects_negative_atol(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)
        result = await solve_nonlinear(
            residual="inner(grad(u),grad(v))*dx",
            unknown="u",
            atol=-1e-12,
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ─── Postcondition tests ──────────────────────────────────────────────


class TestSolveNonlinearPostconditions:

    @pytest.mark.asyncio
    async def test_nan_solution_raises_solver_error(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)

        modules, _ = _setup_solver_mocks(finite=False)

        with (
            patch.dict(sys.modules, modules),
            patch(
                "dolfinx_mcp.ufl_context.build_namespace",
                return_value=_build_ns_mock(),
            ),
            patch(
                "dolfinx_mcp.ufl_context.safe_evaluate",
                return_value=MagicMock(),
            ),
        ):
            result = await solve_nonlinear(
                residual="inner(grad(u),grad(v))*dx",
                unknown="u",
                ctx=ctx,
            )

        assert_error_type(result, "SOLVER_ERROR")
        assert "NaN" in result["message"] or "Inf" in result["message"]

    @pytest.mark.asyncio
    async def test_negative_l2_norm_raises_postcondition_error(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)

        modules, _ = _setup_solver_mocks()

        with (
            patch.dict(sys.modules, modules),
            patch(
                "dolfinx_mcp.ufl_context.build_namespace",
                return_value=_build_ns_mock(),
            ),
            patch(
                "dolfinx_mcp.ufl_context.safe_evaluate",
                return_value=MagicMock(),
            ),
            patch(
                "dolfinx_mcp.utils.compute_l2_norm",
                return_value=-1.0,
            ),
        ):
            result = await solve_nonlinear(
                residual="inner(grad(u),grad(v))*dx",
                unknown="u",
                ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")


# ─── Error path tests ─────────────────────────────────────────────────


class TestSolveNonlinearErrorPaths:

    @pytest.mark.asyncio
    async def test_dolfinx_api_error_wrapped(self):
        """DOLFINx exceptions during solve are wrapped as SolverError."""
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)

        modules, _ = _setup_solver_mocks()
        # Make NonlinearProblem raise
        # Access the mock via the same attribute chain the code uses
        mock_dolfinx = modules["dolfinx"]
        mock_dolfinx.fem.petsc.NonlinearProblem.side_effect = RuntimeError(
            "PETSc error: DIVERGED"
        )

        with (
            patch.dict(sys.modules, modules),
            patch(
                "dolfinx_mcp.ufl_context.build_namespace",
                return_value=_build_ns_mock(),
            ),
            patch(
                "dolfinx_mcp.ufl_context.safe_evaluate",
                return_value=MagicMock(),
            ),
        ):
            result = await solve_nonlinear(
                residual="inner(grad(u),grad(v))*dx",
                unknown="u",
                ctx=ctx,
            )

        assert_error_type(result, "SOLVER_ERROR")
        assert "DIVERGED" in result["message"]


# ─── Happy path tests ─────────────────────────────────────────────────


class TestSolveNonlinearHappyPath:

    @pytest.mark.asyncio
    async def test_successful_solve_returns_expected_keys(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)

        modules, l2 = _setup_solver_mocks(n_iters=3, converged=True)

        with (
            patch.dict(sys.modules, modules),
            patch(
                "dolfinx_mcp.ufl_context.build_namespace",
                return_value=_build_ns_mock(),
            ),
            patch(
                "dolfinx_mcp.ufl_context.safe_evaluate",
                return_value=MagicMock(),
            ),
            patch(
                "dolfinx_mcp.utils.compute_l2_norm",
                return_value=l2,
            ),
        ):
            result = await solve_nonlinear(
                residual="inner(grad(u),grad(v))*dx",
                unknown="u",
                ctx=ctx,
            )

        assert_no_error(result)
        assert result["converged"] is True
        assert result["iterations"] == 3
        assert result["solution_name"] == "u"
        assert result["solution_norm_L2"] == l2
        assert "wall_time" in result
        assert "residual_norm" in result

    @pytest.mark.asyncio
    async def test_custom_solution_name(self):
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = _populated_session()
        ctx = make_mock_ctx(session)

        modules, _ = _setup_solver_mocks(n_iters=2)

        with (
            patch.dict(sys.modules, modules),
            patch(
                "dolfinx_mcp.ufl_context.build_namespace",
                return_value=_build_ns_mock(),
            ),
            patch(
                "dolfinx_mcp.ufl_context.safe_evaluate",
                return_value=MagicMock(),
            ),
            patch(
                "dolfinx_mcp.utils.compute_l2_norm",
                return_value=0.5,
            ),
        ):
            result = await solve_nonlinear(
                residual="inner(grad(u),grad(v))*dx",
                unknown="u",
                solution_name="my_solution",
                ctx=ctx,
            )

        assert result["solution_name"] == "my_solution"
        assert "my_solution" in session.solutions
        assert "my_solution" in session.functions

    @pytest.mark.asyncio
    async def test_all_valid_snes_types_accepted(self):
        """All three SNES types pass precondition validation."""
        from dolfinx_mcp.tools.solver import solve_nonlinear

        for snes in ("newtonls", "newtontr", "nrichardson"):
            session = _populated_session()
            ctx = make_mock_ctx(session)

            modules, _ = _setup_solver_mocks(n_iters=1)

            with (
                patch.dict(sys.modules, modules),
                patch(
                    "dolfinx_mcp.ufl_context.build_namespace",
                    return_value=_build_ns_mock(),
                ),
                patch(
                    "dolfinx_mcp.ufl_context.safe_evaluate",
                    return_value=MagicMock(),
                ),
                patch(
                    "dolfinx_mcp.utils.compute_l2_norm",
                    return_value=0.1,
                ),
            ):
                result = await solve_nonlinear(
                    residual="inner(grad(u),grad(v))*dx",
                    unknown="u",
                    snes_type=snes,
                    ctx=ctx,
                )
            assert "error" not in result, (
                f"snes_type={snes} should be accepted"
            )
