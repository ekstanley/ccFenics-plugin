"""Tests for dolfinx_mcp_jupyter.display -- contract and type guards."""

from __future__ import annotations

import pytest

from dolfinx_mcp_jupyter.display import render_result


class TestDisplayContracts:
    """Type guard at the render_result boundary."""

    def test_render_result_not_list(self):
        with pytest.raises(TypeError, match="Expected list of result items"):
            render_result({"type": "text", "data": "hello"}, "tool")  # type: ignore[arg-type]

    def test_render_result_string(self):
        with pytest.raises(TypeError, match="Expected list of result items"):
            render_result("not a list", "tool")  # type: ignore[arg-type]

    def test_render_result_none(self):
        with pytest.raises(TypeError, match="Expected list of result items"):
            render_result(None, "tool")  # type: ignore[arg-type]
