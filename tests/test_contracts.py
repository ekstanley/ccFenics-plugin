"""Contract violation tests for Design-by-Contract enforcement.

Tests dataclass __post_init__ validators, SessionState.check_invariants(),
and tool precondition checks. No Docker or DOLFINx required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dolfinx_mcp.errors import InvariantError, PreconditionError
from dolfinx_mcp.session import (
    BCInfo,
    EntityMapInfo,
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    MeshTagsInfo,
    SessionState,
    SolutionInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh_info(name: str = "m1") -> MeshInfo:
    return MeshInfo(
        name=name,
        mesh=MagicMock(),
        cell_type="triangle",
        num_cells=100,
        num_vertices=64,
        gdim=2,
        tdim=2,
    )


def _make_space_info(name: str = "V", mesh_name: str = "m1") -> FunctionSpaceInfo:
    return FunctionSpaceInfo(
        name=name,
        space=MagicMock(),
        mesh_name=mesh_name,
        element_family="Lagrange",
        element_degree=1,
        num_dofs=64,
    )


def _make_function_info(name: str = "f", space_name: str = "V") -> FunctionInfo:
    return FunctionInfo(
        name=name,
        function=MagicMock(),
        space_name=space_name,
    )


def _make_bc_info(name: str = "bc0", space_name: str = "V") -> BCInfo:
    return BCInfo(
        name=name,
        bc=MagicMock(),
        space_name=space_name,
        num_dofs=10,
    )


def _make_solution_info(name: str = "u_h", space_name: str = "V") -> SolutionInfo:
    return SolutionInfo(
        name=name,
        function=MagicMock(),
        space_name=space_name,
        converged=True,
        iterations=5,
        residual_norm=1e-10,
        wall_time=0.5,
    )


def _populated_session() -> SessionState:
    """Create a valid, populated session for invariant testing."""
    s = SessionState()
    s.meshes["m1"] = _make_mesh_info("m1")
    s.function_spaces["V"] = _make_space_info("V", "m1")
    s.functions["f"] = _make_function_info("f", "V")
    s.bcs["bc0"] = _make_bc_info("bc0", "V")
    s.solutions["u_h"] = _make_solution_info("u_h", "V")
    s.active_mesh = "m1"
    return s


# ---------------------------------------------------------------------------
# Phase 2: Dataclass __post_init__ precondition tests (8 tests)
# ---------------------------------------------------------------------------


class TestDataclassContracts:
    def test_mesh_info_rejects_zero_cells(self):
        with pytest.raises(AssertionError, match="num_cells must be > 0"):
            MeshInfo(
                name="bad",
                mesh=MagicMock(),
                cell_type="triangle",
                num_cells=0,
                num_vertices=10,
                gdim=2,
                tdim=2,
            )

    def test_mesh_info_rejects_invalid_gdim(self):
        with pytest.raises(AssertionError, match="gdim must be 1, 2, or 3"):
            MeshInfo(
                name="bad",
                mesh=MagicMock(),
                cell_type="triangle",
                num_cells=10,
                num_vertices=10,
                gdim=0,
                tdim=1,
            )

    def test_mesh_info_rejects_tdim_gt_gdim(self):
        with pytest.raises(AssertionError, match="tdim.*must be <= gdim"):
            MeshInfo(
                name="bad",
                mesh=MagicMock(),
                cell_type="triangle",
                num_cells=10,
                num_vertices=10,
                gdim=2,
                tdim=3,
            )

    def test_space_info_rejects_negative_degree(self):
        with pytest.raises(AssertionError, match="degree must be >= 0"):
            FunctionSpaceInfo(
                name="V",
                space=MagicMock(),
                mesh_name="m1",
                element_family="Lagrange",
                element_degree=-1,
                num_dofs=10,
            )

    def test_space_info_rejects_zero_dofs(self):
        with pytest.raises(AssertionError, match="num_dofs must be > 0"):
            FunctionSpaceInfo(
                name="V",
                space=MagicMock(),
                mesh_name="m1",
                element_family="Lagrange",
                element_degree=1,
                num_dofs=0,
            )

    def test_bc_info_rejects_zero_dofs(self):
        with pytest.raises(AssertionError, match="num_dofs must be > 0"):
            BCInfo(
                name="bc",
                bc=MagicMock(),
                space_name="V",
                num_dofs=0,
            )

    def test_solution_info_rejects_negative_iterations(self):
        with pytest.raises(AssertionError, match="iterations must be >= 0"):
            SolutionInfo(
                name="u",
                function=MagicMock(),
                space_name="V",
                converged=True,
                iterations=-1,
                residual_norm=0.0,
                wall_time=0.0,
            )

    def test_entity_map_rejects_empty_names(self):
        with pytest.raises(AssertionError, match="parent_mesh must be non-empty"):
            EntityMapInfo(
                name="em",
                entity_map=MagicMock(),
                parent_mesh="",
                child_mesh="c",
                dimension=1,
            )


# ---------------------------------------------------------------------------
# Phase 3: SessionState invariant tests (5 tests)
# ---------------------------------------------------------------------------


class TestSessionInvariants:
    def test_check_invariants_valid_session(self):
        session = _populated_session()
        # Should not raise
        session.check_invariants()

    def test_check_invariants_dangling_active_mesh(self):
        session = SessionState()
        session.active_mesh = "deleted_mesh"
        with pytest.raises(InvariantError, match="active_mesh.*not in meshes"):
            session.check_invariants()

    def test_check_invariants_dangling_space_mesh_ref(self):
        session = SessionState()
        # Space references mesh "m1" which does not exist
        session.function_spaces["V"] = _make_space_info("V", "m1")
        with pytest.raises(InvariantError, match="non-existent mesh"):
            session.check_invariants()

    def test_check_invariants_dangling_function_space_ref(self):
        session = SessionState()
        session.meshes["m1"] = _make_mesh_info("m1")
        # Function references space "V" which does not exist
        session.functions["f"] = _make_function_info("f", "V")
        with pytest.raises(InvariantError, match="non-existent space"):
            session.check_invariants()

    def test_remove_mesh_postconditions(self):
        session = _populated_session()
        session.remove_mesh("m1")
        # After removal, no references to "m1" should remain
        assert "m1" not in session.meshes
        assert all(fs.mesh_name != "m1" for fs in session.function_spaces.values())
        assert all(mt.mesh_name != "m1" for mt in session.mesh_tags.values())
        # Empty session should pass invariants
        session.check_invariants()


# ---------------------------------------------------------------------------
# Phase 4: Tool precondition tests (5 tests, mocked context)
# ---------------------------------------------------------------------------


def _mock_ctx(session: SessionState):
    """Create a mock MCP Context with the given session."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = session
    return ctx


class TestToolPreconditions:
    @pytest.mark.asyncio
    async def test_create_unit_square_rejects_zero_nx(self):
        from dolfinx_mcp.tools.mesh import create_unit_square

        session = SessionState()
        ctx = _mock_ctx(session)
        result = await create_unit_square(name="m", nx=0, ny=8, ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_solve_time_dependent_rejects_negative_dt(self):
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = SessionState()
        ctx = _mock_ctx(session)
        result = await solve_time_dependent(t_end=1.0, dt=-0.1, ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_set_material_reserved_name(self):
        from dolfinx_mcp.tools.problem import set_material_properties

        session = SessionState()
        session.meshes["m1"] = _make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = _mock_ctx(session)
        result = await set_material_properties(name="grad", value=1.0, ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_evaluate_solution_rejects_empty_points(self):
        from dolfinx_mcp.tools.postprocess import evaluate_solution

        session = SessionState()
        ctx = _mock_ctx(session)
        result = await evaluate_solution(points=[], ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_create_function_space_rejects_negative_degree(self):
        from dolfinx_mcp.tools.spaces import create_function_space

        session = SessionState()
        session.meshes["m1"] = _make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = _mock_ctx(session)
        result = await create_function_space(name="V", degree=-1, ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"
