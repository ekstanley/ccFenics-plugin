"""Logging configuration for DOLFINx MCP server.

CRITICAL: stdout is reserved for JSON-RPC (MCP stdio transport).
ALL logging MUST go to stderr. Any stdout write breaks the transport.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure all logging to write to stderr only.

    Suppresses verbose third-party loggers that would clutter output.
    """
    # Root handler -> stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress verbose third-party loggers
    for name in ("petsc4py", "mpi4py", "gmsh", "ufl", "basix", "ffcx"):
        logging.getLogger(name).setLevel(logging.WARNING)
