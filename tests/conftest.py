"""Shared test fixtures and factory helpers for DOLFINx MCP tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dolfinx_mcp.session import (
    BCInfo,
    EntityMapInfo,
    FormInfo,
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    MeshTagsInfo,
    SessionState,
    SolutionInfo,
)

# ---------------------------------------------------------------------------
# Factory helpers -- canonical versions used across all test files
# ---------------------------------------------------------------------------


def make_mesh_info(name="m1", **kwargs):
    """Create a MeshInfo with sensible defaults. Override any field via kwargs."""
    defaults = {
        "name": name,
        "mesh": MagicMock(),
        "cell_type": "triangle",
        "num_cells": 100,
        "num_vertices": 64,
        "gdim": 2,
        "tdim": 2,
    }
    defaults.update(kwargs)
    return MeshInfo(**defaults)


def make_space_info(name="V", mesh_name="m1", **kwargs):
    """Create a FunctionSpaceInfo with sensible defaults."""
    defaults = {
        "name": name,
        "space": MagicMock(),
        "mesh_name": mesh_name,
        "element_family": "Lagrange",
        "element_degree": 1,
        "num_dofs": 64,
    }
    defaults.update(kwargs)
    return FunctionSpaceInfo(**defaults)


def make_function_info(name="f", space_name="V", **kwargs):
    """Create a FunctionInfo with sensible defaults."""
    defaults = {
        "name": name,
        "function": MagicMock(),
        "space_name": space_name,
    }
    defaults.update(kwargs)
    return FunctionInfo(**defaults)


def make_bc_info(name="bc0", space_name="V", **kwargs):
    """Create a BCInfo with sensible defaults."""
    defaults = {
        "name": name,
        "bc": MagicMock(),
        "space_name": space_name,
        "num_dofs": 10,
    }
    defaults.update(kwargs)
    return BCInfo(**defaults)


def make_solution_info(name="u_h", space_name="V", **kwargs):
    """Create a SolutionInfo with sensible defaults."""
    defaults = {
        "name": name,
        "function": MagicMock(),
        "space_name": space_name,
        "converged": True,
        "iterations": 5,
        "residual_norm": 1e-10,
        "wall_time": 0.5,
    }
    defaults.update(kwargs)
    return SolutionInfo(**defaults)


def make_form_info(name="bilinear", **kwargs):
    """Create a FormInfo with sensible defaults."""
    defaults = {
        "name": name,
        "form": MagicMock(),
        "ufl_form": MagicMock(),
    }
    defaults.update(kwargs)
    return FormInfo(**defaults)


def make_mesh_tags_info(name="tags0", mesh_name="m1", dimension=1, **kwargs):
    """Create a MeshTagsInfo with sensible defaults."""
    defaults = {
        "name": name,
        "tags": MagicMock(),
        "mesh_name": mesh_name,
        "dimension": dimension,
        "unique_tags": [1, 2, 3],
    }
    defaults.update(kwargs)
    return MeshTagsInfo(**defaults)


def make_entity_map_info(
    name="emap0", parent_mesh="m1", child_mesh="m1_sub", dimension=2, **kwargs
):
    """Create an EntityMapInfo with sensible defaults."""
    defaults = {
        "name": name,
        "entity_map": MagicMock(),
        "parent_mesh": parent_mesh,
        "child_mesh": child_mesh,
        "dimension": dimension,
    }
    defaults.update(kwargs)
    return EntityMapInfo(**defaults)


# ---------------------------------------------------------------------------
# Session and context helpers
# ---------------------------------------------------------------------------


def make_mock_ctx(session: SessionState):
    """Create a mock MCP Context wired to the given session."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = session
    return ctx


def make_session_with_mesh(mesh_name="m1") -> SessionState:
    """Create a session with a single mesh and active_mesh set."""
    s = SessionState()
    s.meshes[mesh_name] = make_mesh_info(mesh_name)
    s.active_mesh = mesh_name
    return s


def make_populated_session() -> SessionState:
    """Create a session with mesh, space, function, BC, and solution."""
    s = SessionState()
    s.meshes["m1"] = make_mesh_info("m1")
    s.function_spaces["V"] = make_space_info("V", "m1")
    s.functions["f"] = make_function_info("f", "V")
    s.bcs["bc0"] = make_bc_info("bc0", "V")
    s.solutions["u_h"] = make_solution_info("u_h", "V")
    s.active_mesh = "m1"
    return s


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_no_error(result: dict) -> None:
    """Assert tool result has no error."""
    assert "error" not in result, f"Tool returned error: {result}"


def assert_solve_success(result: dict) -> None:
    """Assert solver tool succeeded."""
    assert_no_error(result)
    assert result.get("converged") is True
    assert result.get("solution_norm_L2", 0) > 0


def assert_error_type(result: dict, expected: str) -> None:
    """Assert tool result has specific error type."""
    assert "error" in result, f"Expected error but got: {result}"
    assert result["error"] == expected, (
        f"Expected error '{expected}' but got '{result['error']}'"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session() -> SessionState:
    """Fresh session state for each test."""
    return SessionState()


@pytest.fixture
def mock_ctx(session: SessionState):
    """Mock MCP Context wired to the given session."""
    return make_mock_ctx(session)


@pytest.fixture
def populated_session() -> SessionState:
    """Session with mesh 'm1', space 'V', function 'f', BC 'bc0', solution 'u_h'."""
    return make_populated_session()


@pytest.fixture
def mock_ctx_populated(populated_session: SessionState):
    """Mock MCP Context wired to a populated session."""
    return make_mock_ctx(populated_session)
