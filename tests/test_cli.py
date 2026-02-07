"""Tests for dolfinx_mcp.__main__ -- CLI argument parsing via subprocess.

The __main__.py module has module-level side effects (calls main() on import),
so we test via subprocess to avoid blocking the test runner.
"""

from __future__ import annotations

import subprocess
import sys


class TestCLIHelp:
    """Verify CLI --help output documents all arguments."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "dolfinx_mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--transport" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_help_shows_transport_choices(self):
        result = subprocess.run(
            [sys.executable, "-m", "dolfinx_mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "stdio" in result.stdout
        assert "streamable-http" in result.stdout
        assert "sse" in result.stdout

    def test_help_shows_default_port(self):
        result = subprocess.run(
            [sys.executable, "-m", "dolfinx_mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "8000" in result.stdout


class TestCLIValidation:
    """Verify CLI rejects invalid arguments."""

    def test_invalid_transport_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "dolfinx_mcp", "--transport", "invalid"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_invalid_port_type_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "dolfinx_mcp", "--port", "not_a_number"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_unknown_flag_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "dolfinx_mcp", "--nonexistent"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
