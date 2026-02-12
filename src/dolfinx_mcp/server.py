"""DOLFINx MCP server entry point.

Configures logging, imports all tool/prompt/resource modules to trigger
decorator registration, then starts the MCP transport (stdio or HTTP).
"""

from __future__ import annotations

import os

from .logging_config import configure_logging

# Configure logging BEFORE any other imports that might log
configure_logging()

# Import the FastMCP instance
from ._app import mcp  # noqa: E402

# Import prompt and resource modules
from .prompts import templates  # noqa: E402, F401
from .resources import providers  # noqa: E402, F401

# Import tool modules to trigger @mcp.tool() registration
from .tools import (  # noqa: E402, F401
    interpolation,
    mesh,
    postprocess,
    problem,
    session_mgmt,
    solver,
    spaces,
)


def main() -> None:
    """Start the DOLFINx MCP server."""
    transport = os.environ.get("DOLFINX_MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)  # type: ignore[arg-type]  # env var is validated by FastMCP
