"""Unit tests for SessionState -- no Docker or DOLFINx required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dolfinx_mcp.errors import (
    FunctionSpaceNotFoundError,
    MeshNotFoundError,
    NoActiveMeshError,
)
from dolfinx_mcp.session import (
    BCInfo,
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    SessionState,
    SolutionInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh_info(name: str) -> MeshInfo:
    return MeshInfo(
        name=name,
        mesh=MagicMock(),
        cell_type="triangle",
        num_cells=100,
        num_vertices=64,
        gdim=2,
        tdim=2,
    )


def _make_space_info(name: str, mesh_name: str) -> FunctionSpaceInfo:
    return FunctionSpaceInfo(
        name=name,
        space=MagicMock(),
        mesh_name=mesh_name,
        element_family="Lagrange",
        element_degree=1,
        num_dofs=64,
    )


def _make_function_info(name: str, space_name: str) -> FunctionInfo:
    return FunctionInfo(
        name=name,
        function=MagicMock(),
        space_name=space_name,
    )


def _make_bc_info(name: str, space_name: str) -> BCInfo:
    return BCInfo(
        name=name,
        bc=MagicMock(),
        space_name=space_name,
        num_dofs=10,
    )


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


class TestSessionInit:
    def test_empty_session(self, session: SessionState):
        assert session.active_mesh is None
        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0

    def test_overview_empty(self, session: SessionState):
        ov = session.overview()
        assert ov["active_mesh"] is None
        assert ov["meshes"] == {}


# ---------------------------------------------------------------------------
# Tests: mesh accessors
# ---------------------------------------------------------------------------


class TestMeshAccessors:
    def test_get_mesh_no_active(self, session: SessionState):
        with pytest.raises(NoActiveMeshError):
            session.get_mesh()

    def test_get_mesh_not_found(self, session: SessionState):
        with pytest.raises(MeshNotFoundError, match="nonexistent"):
            session.get_mesh("nonexistent")

    def test_get_mesh_by_name(self, session: SessionState):
        info = _make_mesh_info("m1")
        session.meshes["m1"] = info
        assert session.get_mesh("m1") is info

    def test_get_mesh_default_active(self, session: SessionState):
        info = _make_mesh_info("m1")
        session.meshes["m1"] = info
        session.active_mesh = "m1"
        assert session.get_mesh() is info


# ---------------------------------------------------------------------------
# Tests: space accessors
# ---------------------------------------------------------------------------


class TestSpaceAccessors:
    def test_get_space_not_found(self, session: SessionState):
        with pytest.raises(FunctionSpaceNotFoundError):
            session.get_space("V")

    def test_get_only_space_none(self, session: SessionState):
        with pytest.raises(FunctionSpaceNotFoundError, match="No function"):
            session.get_only_space()

    def test_get_only_space_one(self, session: SessionState):
        s = _make_space_info("V", "m1")
        session.function_spaces["V"] = s
        assert session.get_only_space() is s

    def test_get_only_space_multiple(self, session: SessionState):
        session.function_spaces["V1"] = _make_space_info("V1", "m1")
        session.function_spaces["V2"] = _make_space_info("V2", "m1")
        with pytest.raises(FunctionSpaceNotFoundError, match="Multiple"):
            session.get_only_space()


# ---------------------------------------------------------------------------
# Tests: cascade deletion
# ---------------------------------------------------------------------------


class TestCascadeDeletion:
    def test_remove_mesh_cascades(self, session: SessionState):
        session.meshes["m1"] = _make_mesh_info("m1")
        session.function_spaces["V"] = _make_space_info("V", "m1")
        session.functions["f"] = _make_function_info("f", "V")
        session.bcs["bc0"] = _make_bc_info("bc0", "V")
        session.active_mesh = "m1"

        session.remove_mesh("m1")

        assert "m1" not in session.meshes
        assert "V" not in session.function_spaces
        assert "f" not in session.functions
        assert "bc0" not in session.bcs
        assert session.active_mesh is None

    def test_remove_mesh_not_found(self, session: SessionState):
        with pytest.raises(MeshNotFoundError):
            session.remove_mesh("missing")

    def test_remove_mesh_preserves_other(self, session: SessionState):
        session.meshes["m1"] = _make_mesh_info("m1")
        session.meshes["m2"] = _make_mesh_info("m2")
        session.function_spaces["V1"] = _make_space_info("V1", "m1")
        session.function_spaces["V2"] = _make_space_info("V2", "m2")
        session.active_mesh = "m2"

        session.remove_mesh("m1")

        assert "m2" in session.meshes
        assert "V2" in session.function_spaces
        assert session.active_mesh == "m2"


# ---------------------------------------------------------------------------
# Tests: overview
# ---------------------------------------------------------------------------


class TestOverview:
    def test_overview_populated(self, session: SessionState):
        session.meshes["m1"] = _make_mesh_info("m1")
        session.function_spaces["V"] = _make_space_info("V", "m1")
        session.active_mesh = "m1"

        ov = session.overview()
        assert ov["active_mesh"] == "m1"
        assert "m1" in ov["meshes"]
        assert "V" in ov["function_spaces"]
        assert ov["meshes"]["m1"]["cell_type"] == "triangle"


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_clears_all(self, session: SessionState):
        session.meshes["m1"] = _make_mesh_info("m1")
        session.function_spaces["V"] = _make_space_info("V", "m1")
        session.active_mesh = "m1"
        session.ufl_symbols["f"] = MagicMock()

        session.cleanup()

        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0
        assert len(session.ufl_symbols) == 0
        assert session.active_mesh is None
