"""Shared test fixtures for DOLFINx MCP tests."""

from __future__ import annotations

import pytest

from dolfinx_mcp.session import SessionState


@pytest.fixture
def session() -> SessionState:
    """Fresh session state for each test."""
    return SessionState()
