"""Tests for dolfinx_mcp.logging_config -- stderr-only logging setup."""

from __future__ import annotations

import logging
import sys

from dolfinx_mcp.logging_config import configure_logging


class TestConfigureLogging:
    """Verify logging configuration invariants."""

    def test_handler_is_stderr(self):
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        handler = root.handlers[-1]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr

    def test_default_level_info(self):
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_custom_level_debug(self):
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_suppresses_third_party(self):
        configure_logging()
        for name in ("petsc4py", "gmsh", "ufl", "basix", "ffcx", "mpi4py"):
            logger = logging.getLogger(name)
            assert logger.level >= logging.WARNING

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        dummy = logging.StreamHandler()
        root.addHandler(dummy)
        handler_count_before = len(root.handlers)

        configure_logging()

        # After configure, root should have exactly 1 handler (the new stderr one)
        assert len(root.handlers) == 1
        assert root.handlers[0] is not dummy
