"""Unit tests for error hierarchy and handle_tool_errors decorator."""

from __future__ import annotations

import pytest

from dolfinx_mcp.errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    DuplicateNameError,
    FileIOError,
    FunctionNotFoundError,
    FunctionSpaceNotFoundError,
    InvalidUFLExpressionError,
    MeshNotFoundError,
    NoActiveMeshError,
    SolverError,
    handle_tool_errors,
)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_base_error(self):
        e = DOLFINxMCPError("test message")
        assert str(e) == "test message"
        assert e.error_code == "DOLFINX_MCP_ERROR"

    def test_base_with_suggestion(self):
        e = DOLFINxMCPError("msg", suggestion="try this")
        d = e.to_dict()
        assert d["error"] == "DOLFINX_MCP_ERROR"
        assert d["message"] == "msg"
        assert d["suggestion"] == "try this"

    def test_no_active_mesh(self):
        e = NoActiveMeshError("no mesh")
        assert e.error_code == "NO_ACTIVE_MESH"
        assert "Create a mesh" in e.suggestion

    def test_mesh_not_found(self):
        e = MeshNotFoundError("mesh 'foo' not found")
        assert e.error_code == "MESH_NOT_FOUND"

    def test_solver_error(self):
        e = SolverError("diverged")
        assert e.error_code == "SOLVER_ERROR"

    def test_duplicate_name(self):
        e = DuplicateNameError("already exists")
        assert e.error_code == "DUPLICATE_NAME"

    def test_all_subclasses_have_error_code(self):
        subclasses = [
            NoActiveMeshError,
            MeshNotFoundError,
            FunctionSpaceNotFoundError,
            FunctionNotFoundError,
            InvalidUFLExpressionError,
            SolverError,
            DuplicateNameError,
            DOLFINxAPIError,
            FileIOError,
        ]
        for cls in subclasses:
            assert hasattr(cls, "error_code")
            assert cls.error_code != "DOLFINX_MCP_ERROR"

    def test_to_dict_without_suggestion(self):
        e = DOLFINxAPIError("api broke")
        d = e.to_dict()
        assert "error" in d
        assert "message" in d
        assert "suggestion" not in d  # empty string => excluded


# ---------------------------------------------------------------------------
# handle_tool_errors decorator
# ---------------------------------------------------------------------------


class TestHandleToolErrors:
    @pytest.mark.asyncio
    async def test_passes_through_on_success(self):
        @handle_tool_errors
        async def good_tool():
            return {"status": "ok"}

        result = await good_tool()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_catches_mcp_error(self):
        @handle_tool_errors
        async def failing_tool():
            raise MeshNotFoundError("mesh 'foo' not found")

        result = await failing_tool()
        assert result["error"] == "MESH_NOT_FOUND"
        assert "foo" in result["message"]

    @pytest.mark.asyncio
    async def test_catches_generic_exception(self):
        @handle_tool_errors
        async def crashing_tool():
            raise RuntimeError("something broke")

        result = await crashing_tool()
        assert result["error"] == "INTERNAL_ERROR"
        assert "something broke" in result["message"]

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        @handle_tool_errors
        async def my_tool():
            return {}

        assert my_tool.__name__ == "my_tool"
