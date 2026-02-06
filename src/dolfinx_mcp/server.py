"""DOLFINx MCP server entry point.

Configures logging, imports all tool/prompt/resource modules to trigger
decorator registration, then starts the MCP stdio transport.
"""

from __future__ import annotations

from .logging_config import configure_logging

# Configure logging BEFORE any other imports that might log
configure_logging()

# Import the FastMCP instance
from ._app import mcp  # noqa: E402

# Import tool modules to trigger @mcp.tool() registration
from .tools import mesh  # noqa: E402, F401
from .tools import spaces  # noqa: E402, F401
from .tools import problem  # noqa: E402, F401
from .tools import solver  # noqa: E402, F401
from .tools import postprocess  # noqa: E402, F401
from .tools import interpolation  # noqa: E402, F401
from .tools import session_mgmt  # noqa: E402, F401

# Import prompt and resource modules
from .prompts import templates  # noqa: E402, F401
from .resources import providers  # noqa: E402, F401


def main() -> None:
    """Start the DOLFINx MCP server with stdio transport."""
    mcp.run(transport="stdio")
