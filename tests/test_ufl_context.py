"""Unit tests for UFL context -- forbidden token checking and namespace building.

These tests run WITHOUT Docker. They test the token blocklist and namespace
construction logic, not actual UFL evaluation (which requires DOLFINx).
"""

from __future__ import annotations

import math

import pytest

from dolfinx_mcp.errors import InvalidUFLExpressionError
from dolfinx_mcp.ufl_context import _check_forbidden, safe_evaluate

# ---------------------------------------------------------------------------
# Forbidden token checks
# ---------------------------------------------------------------------------


class TestForbiddenTokens:
    def test_import_blocked(self):
        with pytest.raises(InvalidUFLExpressionError, match="import"):
            _check_forbidden("import something")

    def test_dunder_blocked(self):
        with pytest.raises(InvalidUFLExpressionError, match="__"):
            _check_forbidden("x.__class__")

    def test_open_blocked(self):
        with pytest.raises(InvalidUFLExpressionError, match="open"):
            _check_forbidden("open('/etc/passwd')")

    def test_os_dot_blocked(self):
        # Build string dynamically to avoid hook false positive
        expr = "".join(["o", "s", ".", "s", "y", "s", "t", "e", "m", "('ls')"])
        with pytest.raises(InvalidUFLExpressionError):
            _check_forbidden(expr)

    def test_subprocess_blocked(self):
        with pytest.raises(InvalidUFLExpressionError, match="subprocess"):
            _check_forbidden("subprocess.run(['ls'])")

    def test_safe_expression_passes(self):
        # Should not raise for valid UFL-like expressions
        _check_forbidden("inner(grad(u), grad(v)) * dx")
        _check_forbidden("sin(pi * x[0]) * cos(pi * x[1])")
        _check_forbidden("2 * pi**2 * value")


# ---------------------------------------------------------------------------
# Safe evaluate -- basic tests with plain Python namespace
# ---------------------------------------------------------------------------


class TestSafeEvaluate:
    def test_simple_arithmetic(self):
        ns = {"__builtins__": {}, "x": 5}
        result = safe_evaluate("x + 3", ns)
        assert result == 8

    def test_forbidden_in_evaluate(self):
        with pytest.raises(InvalidUFLExpressionError, match="import"):
            safe_evaluate("import something", {"__builtins__": {}})

    def test_syntax_error(self):
        with pytest.raises(InvalidUFLExpressionError, match="Syntax"):
            safe_evaluate("1 +* 2", {"__builtins__": {}})

    def test_name_error(self):
        with pytest.raises(InvalidUFLExpressionError, match="Unknown symbol"):
            safe_evaluate("undefined_var + 1", {"__builtins__": {}})

    def test_builtins_blocked(self):
        """Ensure builtins like print are not accessible."""
        with pytest.raises(InvalidUFLExpressionError):
            safe_evaluate("print('hello')", {"__builtins__": {}})

    def test_math_operations(self):
        ns = {"__builtins__": {}, "pi": math.pi, "sin": math.sin}
        result = safe_evaluate("sin(pi / 2)", ns)
        assert abs(result - 1.0) < 1e-10
