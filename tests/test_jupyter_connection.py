"""Tests for dolfinx_mcp_jupyter.connection -- initial state and config."""

from __future__ import annotations

from dolfinx_mcp_jupyter.config import MCPConfig
from dolfinx_mcp_jupyter.connection import MCPConnection


class TestMCPConnectionInit:
    """MCPConnection should start in a disconnected state."""

    def test_not_connected(self):
        conn = MCPConnection()
        assert not conn.connected

    def test_empty_tools(self):
        conn = MCPConnection()
        assert conn.list_tools() == {}

    def test_custom_config(self):
        config = MCPConfig(transport="streamable-http", server_url="http://host:9000/mcp")
        conn = MCPConnection(config)
        assert conn.config.transport == "streamable-http"
        assert conn.config.server_url == "http://host:9000/mcp"
        assert not conn.connected

    def test_default_config(self):
        conn = MCPConnection()
        assert conn.config.transport == "stdio"
        assert conn.config.server_command == "python -m dolfinx_mcp"


class TestMCPConnectionContracts:
    """Contract violation tests for MCPConnection."""

    def test_call_tool_empty_name(self):
        import asyncio

        import pytest

        conn = MCPConnection()
        with pytest.raises(ValueError, match="Tool name must not be empty"):
            asyncio.get_event_loop().run_until_complete(conn.call_tool(""))

    def test_call_tool_whitespace_name(self):
        import asyncio

        import pytest

        conn = MCPConnection()
        with pytest.raises(ValueError, match="Tool name must not be empty"):
            asyncio.get_event_loop().run_until_complete(conn.call_tool("   "))
