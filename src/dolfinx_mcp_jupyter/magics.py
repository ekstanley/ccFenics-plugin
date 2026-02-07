"""IPython magic commands for DOLFINx MCP integration.

Provides line and cell magics for connecting to a DOLFINx MCP server
and calling tools from Jupyter notebook cells.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

from .config import MCPConfig
from .connection import MCPConnection
from .display import render_result, render_tools_table


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous magic context.

    Jupyter runs its own event loop, so we need ``nest_asyncio`` to allow
    nested ``run_until_complete`` calls.
    """
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
        return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _parse_kwargs(parts: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` tokens into a dict with type coercion.

    Coercion order: bool -> int -> float -> JSON -> string.
    """
    kwargs: dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, _, value = part.partition("=")

        # Bool
        if value.lower() in ("true", "false"):
            kwargs[key] = value.lower() == "true"
        # Int
        elif value.lstrip("-").isdigit():
            kwargs[key] = int(value)
        else:
            # Float
            try:
                kwargs[key] = float(value)
            except ValueError:
                # JSON (lists, dicts)
                try:
                    kwargs[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Plain string
                    kwargs[key] = value

    return kwargs


@magics_class
class DOLFINxMagics(Magics):
    """IPython magics for interacting with a DOLFINx MCP server."""

    def __init__(self, shell: Any, **kwargs: Any) -> None:
        super().__init__(shell, **kwargs)
        self._connection = MCPConnection()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @line_magic
    def dolfinx_connect(self, line: str = "") -> None:  # noqa: C901
        """Connect to the DOLFINx MCP server.

        Usage::

            %dolfinx_connect
            %dolfinx_connect --transport streamable-http --url http://host:8000/mcp
            %dolfinx_connect --command "docker run --rm -i dolfinx-mcp"
        """
        parts = shlex.split(line) if line.strip() else []
        config = MCPConfig()

        i = 0
        while i < len(parts):
            if parts[i] == "--transport" and i + 1 < len(parts):
                config.transport = parts[i + 1]
                i += 2
            elif parts[i] == "--url" and i + 1 < len(parts):
                config.server_url = parts[i + 1]
                i += 2
            elif parts[i] == "--command" and i + 1 < len(parts):
                config.server_command = parts[i + 1]
                i += 2
            else:
                i += 1

        self._connection = MCPConnection(config)
        _run_async(self._connection.connect())

        n_tools = len(self._connection.list_tools())
        print(f"Connected ({config.transport}). {n_tools} tools available.")

    @line_magic
    def dolfinx_disconnect(self, line: str = "") -> None:
        """Disconnect from the DOLFINx MCP server."""
        _run_async(self._connection.disconnect())
        print("Disconnected.")

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    @line_magic
    def dolfinx_tools(self, line: str = "") -> None:
        """List available DOLFINx MCP tools.

        Usage::

            %dolfinx_tools
        """
        if not self._connection.connected:
            _run_async(self._connection.connect())
        render_tools_table(self._connection.list_tools())

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    @line_magic
    def dolfinx(self, line: str) -> None:
        """Call a single DOLFINx MCP tool.

        Usage::

            %dolfinx create_unit_square name=mesh1 nx=16 ny=16
            %dolfinx solve solver_type=direct
            %dolfinx plot_solution
            %dolfinx get_session_state
        """
        if not line.strip():
            print("Usage: %dolfinx <tool_name> [key=value ...]")
            return

        parts = shlex.split(line)
        tool_name = parts[0]
        kwargs = _parse_kwargs(parts[1:])

        if not self._connection.connected:
            _run_async(self._connection.connect())

        items = _run_async(self._connection.call_tool(tool_name, kwargs))
        render_result(items, tool_name)

        # Store in shell namespace for programmatic access
        self.shell.user_ns["_dolfinx_result"] = items

    @cell_magic
    def dolfinx_workflow(self, line: str, cell: str) -> None:
        """Execute a multi-step DOLFINx workflow.

        Each line in the cell body is a tool call (same syntax as ``%dolfinx``).
        Lines starting with ``#`` are comments and are skipped.

        Usage::

            %%dolfinx_workflow
            create_unit_square name=mesh nx=16 ny=16
            create_function_space name=V family=Lagrange degree=1
            set_material_properties name=f value="-6.0"
            define_variational_form bilinear="inner(grad(u), grad(v)) * dx" linear="f * v * dx"
            apply_boundary_condition value=0.0 boundary="True"
            solve solver_type=direct
            plot_solution
        """
        if not self._connection.connected:
            _run_async(self._connection.connect())

        all_results: list[list[dict[str, Any]]] = []

        for step, raw_line in enumerate(cell.strip().splitlines(), 1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = shlex.split(stripped)
            tool_name = parts[0]
            kwargs = _parse_kwargs(parts[1:])

            print(f"[{step}] {tool_name}...")
            items = _run_async(self._connection.call_tool(tool_name, kwargs))
            all_results.append(items)

            # Check for errors
            has_error = any(it.get("type") == "error" for it in items)
            render_result(items, tool_name)

            if has_error:
                print(f"Workflow stopped at step {step} due to error.")
                break

        self.shell.user_ns["_dolfinx_results"] = all_results
