"""Entry point for ``python -m dolfinx_mcp``.

Parses CLI arguments and sets environment variables *before* importing the
server module so that ``_app.py`` can read host/port at FastMCP construction
time.
"""

from __future__ import annotations

import argparse
import os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DOLFINx MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host for HTTP transports (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for HTTP transports (default: 8000)",
    )
    return parser.parse_args()


args = _parse_args()
os.environ["DOLFINX_MCP_TRANSPORT"] = args.transport
os.environ["DOLFINX_MCP_HOST"] = args.host
os.environ["DOLFINX_MCP_PORT"] = str(args.port)

from dolfinx_mcp.server import main  # noqa: E402

main()
