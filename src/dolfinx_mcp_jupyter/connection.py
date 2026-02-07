"""MCP client connection manager for DOLFINx MCP server."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent

from .config import MCPConfig

logger = logging.getLogger(__name__)


class MCPConnection:
    """Manages a persistent connection to a DOLFINx MCP server.

    Supports two transports:
    - ``stdio``: spawns the server as a subprocess, communicates via pipes.
    - ``streamable-http``: connects to an HTTP endpoint.

    The connection is lazy -- it is established on the first ``call_tool``
    if ``connect()`` has not been called explicitly.
    """

    def __init__(self, config: MCPConfig | None = None) -> None:
        self.config = config or MCPConfig()
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: dict[str, dict[str, Any]] = {}

    @property
    def connected(self) -> bool:
        """Whether a live MCP session exists."""
        return self._session is not None

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        if self._session is not None:
            logger.info("Already connected")
            return

        self._exit_stack = AsyncExitStack()

        if self.config.transport == "stdio":
            parts = self.config.server_command.split()
            server_params = StdioServerParameters(
                command=parts[0],
                args=parts[1:] if len(parts) > 1 else [],
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params, errlog=sys.stderr),
            )
        elif self.config.transport in ("streamable-http", "sse"):
            if self.config.transport == "streamable-http":
                from mcp.client.streamable_http import streamablehttp_client

                ctx = streamablehttp_client(self.config.server_url)
            else:
                from mcp.client.sse import sse_client

                ctx = sse_client(self.config.server_url)

            transport_tuple = await self._exit_stack.enter_async_context(ctx)
            read_stream, write_stream = transport_tuple[0], transport_tuple[1]
        else:
            msg = f"Unknown transport: {self.config.transport}"
            raise ValueError(msg)

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream),
        )
        await self._session.initialize()

        # Cache tool list
        tools_result = await self._session.list_tools()
        self._tools = {
            t.name: {"description": t.description or "", "inputSchema": t.inputSchema}
            for t in tools_result.tools
        }
        logger.info("Connected to DOLFINx MCP server (%d tools)", len(self._tools))

    async def disconnect(self) -> None:
        """Close the connection and clean up resources."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None
        self._tools = {}
        logger.info("Disconnected from DOLFINx MCP server")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Call an MCP tool and return parsed content items.

        Returns a list of dicts, each representing one content item:
        - ``{"type": "text", "data": <parsed JSON or raw text>}``
        - ``{"type": "image", "data": <base64 str>, "mimeType": <str>}``
        - ``{"type": "error", "message": <str>}``
        """
        if not name or not name.strip():
            msg = "Tool name must not be empty"
            raise ValueError(msg)

        if self._session is None:
            await self.connect()

        if self._session is None:
            msg = "Connection failed: session not established after connect()"
            raise RuntimeError(msg)

        result: CallToolResult = await self._session.call_tool(
            name=name,
            arguments=arguments or {},
        )

        parsed: list[dict[str, Any]] = []

        if result.isError:
            error_text = ""
            for item in result.content:
                if isinstance(item, TextContent):
                    error_text += item.text
            parsed.append({"type": "error", "message": error_text})
            return parsed

        for item in result.content:
            if isinstance(item, TextContent):
                try:
                    parsed.append({"type": "text", "data": json.loads(item.text)})
                except (json.JSONDecodeError, ValueError):
                    parsed.append({"type": "text", "data": item.text})
            elif isinstance(item, ImageContent):
                parsed.append({
                    "type": "image",
                    "data": item.data,
                    "mimeType": item.mimeType,
                })
            else:
                parsed.append({"type": "text", "data": str(item)})

        return parsed

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """Return cached tool metadata. Must be connected first."""
        return dict(self._tools)
