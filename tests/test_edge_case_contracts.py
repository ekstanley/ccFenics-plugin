"""Tests for DbC edge case fixes (E1-E7, PS-1 through S4).

Host-side tests using mocked DOLFINx objects. Each test verifies that the
precondition/postcondition catches invalid input and returns a structured
error dict via @handle_tool_errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dolfinx_mcp.session import (
    FunctionSpaceInfo,
    MeshInfo,
    SessionState,
)


def _mock_ctx(session: SessionState):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = session
    return ctx


def _make_mesh_info(name: str = "m1", tdim: int = 2) -> MeshInfo:
    return MeshInfo(
        name=name, mesh=MagicMock(), cell_type="triangle",
        num_cells=100, num_vertices=64, gdim=2, tdim=tdim,
    )


def _make_space_info(
    name: str = "V", mesh_name: str = "m1", num_sub_spaces: int = 0,
) -> FunctionSpaceInfo:
    mock_space = MagicMock()
    mock_space.num_sub_spaces = num_sub_spaces
    return FunctionSpaceInfo(
        name=name, space=mock_space, mesh_name=mesh_name,
        element_family="Lagrange", element_degree=1, num_dofs=64,
    )


def _base_session(tdim: int = 2, num_sub_spaces: int = 0) -> SessionState:
    s = SessionState()
    s.meshes["m1"] = _make_mesh_info("m1", tdim=tdim)
    s.function_spaces["V"] = _make_space_info("V", num_sub_spaces=num_sub_spaces)
    s.active_mesh = "m1"
    return s


# ---------------------------------------------------------------------------
# E1: apply_boundary_condition -- sub_space upper bound
# ---------------------------------------------------------------------------


class TestApplyBCEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_sub_space_out_of_range(self):
        """E1: sub_space=2 on a 2-subspace mixed space should fail."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = _base_session(num_sub_spaces=2)
        ctx = _mock_ctx(session)

        result = await apply_boundary_condition(
            value=0.0,
            boundary="np.isclose(x[0], 0.0)",
            function_space="V",
            sub_space=2,
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "out of range" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_sub_space_on_scalar_space(self):
        """E1: sub_space specified on a space with no sub-spaces."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = _base_session(num_sub_spaces=0)
        ctx = _mock_ctx(session)

        result = await apply_boundary_condition(
            value=0.0,
            boundary="np.isclose(x[0], 0.0)",
            function_space="V",
            sub_space=0,
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "no sub-spaces" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_nan_bc_value(self):
        """PS-1: value=NaN should fail."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await apply_boundary_condition(
            value=float("nan"),
            boundary="np.isclose(x[0], 0.0)",
            function_space="V",
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_inf_bc_value(self):
        """PS-1: value=inf should fail."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await apply_boundary_condition(
            value=float("inf"),
            boundary="np.isclose(x[0], 0.0)",
            function_space="V",
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_both_boundary_and_boundary_tag(self):
        """PS-3: specifying both boundary and boundary_tag should fail."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await apply_boundary_condition(
            value=0.0,
            boundary="np.isclose(x[0], 0.0)",
            boundary_tag=1,
            function_space="V",
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "both" in result["message"].lower() or "Cannot" in result["message"]


# ---------------------------------------------------------------------------
# E2: solve_time_dependent -- output_times + step cap
# ---------------------------------------------------------------------------


class TestSolveTimeDependentEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_output_times_outside_range(self):
        """E2: output_times outside [t_start, t_end] should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        # Add required forms and BCs for the solver
        session.forms["bilinear"] = MagicMock()
        session.forms["linear"] = MagicMock()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=1.0, dt=0.1,
            output_times=[0.5, 1.5],
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "output_times" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_non_finite_output_time(self):
        """E2: NaN in output_times should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        session.forms["bilinear"] = MagicMock()
        session.forms["linear"] = MagicMock()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=1.0, dt=0.1,
            output_times=[float("nan")],
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_excessive_step_count(self):
        """E2: dt so small it would exceed 1M steps should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        session.forms["bilinear"] = MagicMock()
        session.forms["linear"] = MagicMock()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=100.0, dt=1e-8,
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "1,000,000" in result["message"] or "steps" in result["message"]


# ---------------------------------------------------------------------------
# E3: manage_mesh_tags -- dimension validation
# ---------------------------------------------------------------------------


class TestManageMeshTagsEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_dimension_exceeding_mesh_tdim(self):
        """E3: dimension=3 on a 2D mesh should fail."""
        from dolfinx_mcp.tools.mesh import manage_mesh_tags

        session = _base_session(tdim=2)
        ctx = _mock_ctx(session)

        result = await manage_mesh_tags(
            action="create",
            name="tags",
            dimension=3,
            values=[{"entities": [0], "tag": 1}],
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "dimension" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_rejects_negative_dimension(self):
        """E3: dimension=-1 should fail."""
        from dolfinx_mcp.tools.mesh import manage_mesh_tags

        session = _base_session(tdim=2)
        ctx = _mock_ctx(session)

        result = await manage_mesh_tags(
            action="create",
            name="tags",
            dimension=-1,
            values=[{"entities": [0], "tag": 1}],
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"


# ---------------------------------------------------------------------------
# E6: set_material_properties -- constant finiteness
# ---------------------------------------------------------------------------


class TestSetMaterialEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_nan_material_constant(self):
        """E6: value=NaN should fail."""
        from dolfinx_mcp.tools.problem import set_material_properties

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await set_material_properties(
            name="kappa", value=float("nan"), ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_inf_material_constant(self):
        """E6: value=inf should fail."""
        from dolfinx_mcp.tools.problem import set_material_properties

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await set_material_properties(
            name="kappa", value=float("inf"), ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]


# ---------------------------------------------------------------------------
# E7: mark_boundaries -- duplicate tag validation
# ---------------------------------------------------------------------------


class TestMarkBoundariesEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_duplicate_marker_tags(self):
        """E7: duplicate tag values across markers should fail."""
        from dolfinx_mcp.tools.mesh import mark_boundaries

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await mark_boundaries(
            markers=[
                {"tag": 1, "condition": "np.isclose(x[0], 0.0)"},
                {"tag": 1, "condition": "np.isclose(x[0], 1.0)"},
            ],
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "Duplicate" in result["message"]


# ---------------------------------------------------------------------------
# PS-5: define_variational_form -- cross-mesh trial/test spaces
# ---------------------------------------------------------------------------


class TestDefineVariationalFormEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_cross_mesh_trial_test_space(self):
        """PS-5: trial and test spaces on different meshes should fail."""
        from dolfinx_mcp.tools.problem import define_variational_form

        session = SessionState()
        session.meshes["m1"] = _make_mesh_info("m1")
        session.meshes["m2"] = _make_mesh_info("m2")
        session.function_spaces["V1"] = _make_space_info("V1", mesh_name="m1")
        session.function_spaces["V2"] = _make_space_info("V2", mesh_name="m2")
        session.active_mesh = "m1"
        ctx = _mock_ctx(session)

        result = await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="f * v * dx",
            trial_space="V1",
            test_space="V2",
            ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "mesh" in result["message"].lower()


# ---------------------------------------------------------------------------
# PS-6: create_function_space -- shape validation
# ---------------------------------------------------------------------------


class TestCreateFunctionSpaceEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_negative_shape_element(self):
        """PS-6: negative shape dimensions should fail."""
        from dolfinx_mcp.tools.spaces import create_function_space

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await create_function_space(
            name="V_bad", shape=[-1], ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "positive" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_shape(self):
        """PS-6: empty shape list should fail."""
        from dolfinx_mcp.tools.spaces import create_function_space

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await create_function_space(
            name="V_bad", shape=[], ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "non-empty" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_zero_shape_element(self):
        """PS-6: zero in shape dimensions should fail."""
        from dolfinx_mcp.tools.spaces import create_function_space

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await create_function_space(
            name="V_bad", shape=[0], ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "positive" in result["message"].lower()


# ---------------------------------------------------------------------------
# S1: solve / solve_time_dependent -- empty solution_name
# ---------------------------------------------------------------------------


class TestSolverNameEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_empty_solution_name_solve(self):
        """S1: empty solution_name in solve should fail."""
        from dolfinx_mcp.tools.solver import solve

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve(solution_name="", ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "solution_name" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_empty_solution_name_time_dependent(self):
        """S1: empty solution_name in solve_time_dependent should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=1.0, dt=0.1, solution_name="", ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "solution_name" in result["message"]


# ---------------------------------------------------------------------------
# S2: solve -- iterative solver parameter validation
# ---------------------------------------------------------------------------


class TestSolverParamEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_negative_rtol(self):
        """S2: negative rtol for iterative solver should fail."""
        from dolfinx_mcp.tools.solver import solve

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve(solver_type="iterative", rtol=-1e-10, ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "rtol" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_zero_max_iter(self):
        """S2: zero max_iter for iterative solver should fail."""
        from dolfinx_mcp.tools.solver import solve

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve(solver_type="iterative", max_iter=0, ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "max_iter" in result["message"]


# ---------------------------------------------------------------------------
# S3: solve_time_dependent -- finiteness checks
# ---------------------------------------------------------------------------


class TestTimeDependentFiniteness:

    @pytest.mark.asyncio
    async def test_rejects_inf_dt(self):
        """S3: dt=inf should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=1.0, dt=float("inf"), ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_nan_t_end(self):
        """S3: t_end=NaN should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=float("nan"), dt=0.1, ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_nan_t_start(self):
        """S3: t_start=NaN should fail."""
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await solve_time_dependent(
            t_end=1.0, dt=0.1, t_start=float("nan"), ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "finite" in result["message"]


# ---------------------------------------------------------------------------
# P3: compute_functionals -- empty expression validation
# ---------------------------------------------------------------------------


class TestComputeFunctionalsEdgeCases:

    @pytest.mark.asyncio
    async def test_rejects_empty_expression_in_list(self):
        """P3: empty string in expressions list should fail."""
        from dolfinx_mcp.tools.postprocess import compute_functionals

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await compute_functionals(
            expressions=["u*u*dx", ""], ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "non-empty" in result["message"]

    @pytest.mark.asyncio
    async def test_rejects_whitespace_expression(self):
        """P3: whitespace-only expression should fail."""
        from dolfinx_mcp.tools.postprocess import compute_functionals

        session = _base_session()
        ctx = _mock_ctx(session)

        result = await compute_functionals(
            expressions=["   "], ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "non-empty" in result["message"]
