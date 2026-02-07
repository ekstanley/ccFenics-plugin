"""Configuration for DOLFINx MCP Jupyter integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MCPConfig:
    """Configuration for connecting to a DOLFINx MCP server.

    Reads defaults from environment variables where available.

    Attributes:
        transport: ``"stdio"`` (spawn subprocess) or ``"streamable-http"``
            (connect to URL).
        server_command: Command to spawn the MCP server (stdio mode).
        server_url: URL of the MCP server (HTTP mode).
        timeout: Timeout in seconds for individual tool calls.
    """

    transport: str = field(
        default_factory=lambda: os.environ.get("DOLFINX_MCP_TRANSPORT", "stdio"),
    )
    server_command: str = field(
        default_factory=lambda: os.environ.get(
            "DOLFINX_MCP_COMMAND", "python -m dolfinx_mcp"
        ),
    )
    server_url: str = field(
        default_factory=lambda: os.environ.get(
            "DOLFINX_MCP_URL", "http://localhost:8000/mcp"
        ),
    )
    timeout: float = 300.0

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        valid_transports = {"stdio", "streamable-http", "sse"}
        if self.transport not in valid_transports:
            msg = (
                f"Invalid transport {self.transport!r}. "
                f"Must be one of: {', '.join(sorted(valid_transports))}"
            )
            raise ValueError(msg)
        if self.timeout <= 0:
            msg = f"Timeout must be positive, got {self.timeout}"
            raise ValueError(msg)
        if not self.server_command or not self.server_command.strip():
            raise ValueError("server_command must not be empty")
        if not self.server_url or not self.server_url.strip():
            raise ValueError("server_url must not be empty")
