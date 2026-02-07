"""Tests for dolfinx_mcp_jupyter.config."""

from __future__ import annotations

import os
from unittest.mock import patch

from dolfinx_mcp_jupyter.config import MCPConfig


class TestMCPConfigDefaults:
    """MCPConfig should provide sensible defaults."""

    def test_default_transport(self):
        config = MCPConfig()
        assert config.transport == "stdio"

    def test_default_server_command(self):
        config = MCPConfig()
        assert config.server_command == "python -m dolfinx_mcp"

    def test_default_server_url(self):
        config = MCPConfig()
        assert config.server_url == "http://localhost:8000/mcp"

    def test_default_timeout(self):
        config = MCPConfig()
        assert config.timeout == 300.0


class TestMCPConfigEnvOverride:
    """MCPConfig should read from environment variables."""

    def test_transport_from_env(self):
        with patch.dict(os.environ, {"DOLFINX_MCP_TRANSPORT": "streamable-http"}):
            config = MCPConfig()
            assert config.transport == "streamable-http"

    def test_command_from_env(self):
        with patch.dict(os.environ, {"DOLFINX_MCP_COMMAND": "docker run dolfinx-mcp"}):
            config = MCPConfig()
            assert config.server_command == "docker run dolfinx-mcp"

    def test_url_from_env(self):
        with patch.dict(os.environ, {"DOLFINX_MCP_URL": "http://mcp:9000/mcp"}):
            config = MCPConfig()
            assert config.server_url == "http://mcp:9000/mcp"


class TestMCPConfigExplicit:
    """Explicit constructor arguments should override defaults."""

    def test_explicit_transport(self):
        config = MCPConfig(transport="sse")
        assert config.transport == "sse"

    def test_explicit_timeout(self):
        config = MCPConfig(timeout=60.0)
        assert config.timeout == 60.0


class TestMCPConfigContracts:
    """Contract violation tests -- invalid inputs must raise ValueError."""

    def test_invalid_transport(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid transport"):
            MCPConfig(transport="invalid")

    def test_negative_timeout(self):
        import pytest

        with pytest.raises(ValueError, match="Timeout must be positive"):
            MCPConfig(timeout=-1.0)

    def test_zero_timeout(self):
        import pytest

        with pytest.raises(ValueError, match="Timeout must be positive"):
            MCPConfig(timeout=0.0)

    def test_empty_server_command(self):
        import pytest

        with pytest.raises(ValueError, match="server_command must not be empty"):
            MCPConfig(server_command="")

    def test_empty_server_url(self):
        import pytest

        with pytest.raises(ValueError, match="server_url must not be empty"):
            MCPConfig(server_url="")

    def test_whitespace_only_command(self):
        import pytest

        with pytest.raises(ValueError, match="server_command must not be empty"):
            MCPConfig(server_command="   ")
