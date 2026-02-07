"""FastMCP application instance and lifespan management.

This module exists to break circular imports. Tool modules import ``mcp``
from here; server.py imports tool modules to trigger decorator registration.

Import DAG:
    session.py, errors.py, ufl_context.py, logging_config.py  (leaves)
        ^
    _app.py  (this file -- imports session.py)
        ^
    tools/*.py  (import _app.py + leaves)
        ^
    server.py  (imports _app.py + all tool modules)
        ^
    __main__.py
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .session import SessionState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[SessionState]:
    """Lifespan context manager -- creates and tears down session state."""
    session = SessionState()
    logger.info("DOLFINx MCP session initialized")
    try:
        yield session
    finally:
        session.cleanup()
        logger.info("DOLFINx MCP session shut down")


mcp = FastMCP(
    "dolfinx-mcp",
    lifespan=app_lifespan,
    host=os.environ.get("DOLFINX_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("DOLFINX_MCP_PORT", "8000")),
)
