"""Shared test fixtures for DOLFINx MCP tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dolfinx_mcp.session import (
    BCInfo,
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    SessionState,
    SolutionInfo,
)


@pytest.fixture
def session() -> SessionState:
    """Fresh session state for each test."""
    return SessionState()


@pytest.fixture
def mock_ctx(session: SessionState):
    """Mock MCP Context wired to the given session."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = session
    return ctx


@pytest.fixture
def populated_session() -> SessionState:
    """Session with mesh 'm1', space 'V', function 'f', BC 'bc0', solution 'u_h'."""
    s = SessionState()
    s.meshes["m1"] = MeshInfo(
        name="m1", mesh=MagicMock(), cell_type="triangle",
        num_cells=100, num_vertices=64, gdim=2, tdim=2,
    )
    s.function_spaces["V"] = FunctionSpaceInfo(
        name="V", space=MagicMock(), mesh_name="m1",
        element_family="Lagrange", element_degree=1, num_dofs=64,
    )
    s.functions["f"] = FunctionInfo(
        name="f", function=MagicMock(), space_name="V",
    )
    s.bcs["bc0"] = BCInfo(
        name="bc0", bc=MagicMock(), space_name="V", num_dofs=10,
    )
    s.solutions["u_h"] = SolutionInfo(
        name="u_h", function=MagicMock(), space_name="V",
        converged=True, iterations=5, residual_norm=1e-10, wall_time=0.5,
    )
    s.active_mesh = "m1"
    return s


@pytest.fixture
def mock_ctx_populated(populated_session: SessionState):
    """Mock MCP Context wired to a populated session."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = populated_session
    return ctx
