"""Shared expression evaluation helpers for DOLFINx MCP tools.

All expression evaluation uses restricted namespaces intentionally --
DOLFINx interpolation and UFL expressions are Python syntax. Security
is enforced by:
1. Token blocklist via _check_forbidden (blocks import, __, exec, etc.)
2. Empty __builtins__ dict (no system access)
3. Docker container isolation (--network none, non-root, --rm)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def eval_numpy_expression(expr: str, x: Any) -> Any:
    """Evaluate a numpy expression at coordinate arrays.

    Used for interpolation, exact solutions, and material properties.
    The expression can use x[0], x[1], x[2] for coordinates and
    common math functions (sin, cos, exp, sqrt, abs, log).

    Args:
        expr: Python/numpy expression string.
        x: Coordinate array of shape (gdim, N).

    Returns:
        Array of shape (N,) or scalar broadcast to that shape.
    """
    import numpy as np

    from .ufl_context import _check_forbidden

    _check_forbidden(expr)

    ns: dict[str, Any] = {
        "x": x,
        "np": np,
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "log": np.log,
        "__builtins__": {},
    }
    result = eval(expr, ns)  # noqa: S307 -- restricted namespace, Docker-sandboxed
    if isinstance(result, (int, float)):
        return np.full(x.shape[1], float(result))
    result = np.asarray(result)
    n_points = x.shape[1]
    if result.shape != (n_points,) and result.shape != ():
        try:
            result = np.broadcast_to(result, (n_points,)).copy()
        except ValueError as exc:
            from .errors import PostconditionError
            raise PostconditionError(
                f"Expression produced shape {result.shape}, expected ({n_points},).",
                suggestion="Expression must return a scalar or array matching "
                "the number of mesh points.",
            ) from exc
    return result


def make_boundary_marker(condition: str) -> Callable[[Any], Any]:
    """Create a boundary marker callable from a condition string.

    Security: _check_forbidden is called EAGERLY at creation time,
    rejecting malicious expressions before the callable is stored.

    Args:
        condition: Python expression using 'x' (coordinate array)
                  and 'np' (numpy). Example: "np.isclose(x[0], 0.0)"

    Returns:
        Callable[[ndarray], ndarray] suitable for DOLFINx boundary location.
    """
    import numpy as np

    from .ufl_context import _check_forbidden

    _check_forbidden(condition)

    if condition.strip() == "True":
        def marker(x: Any) -> Any:
            return np.full(x.shape[1], True)
        return marker

    def marker(x: Any) -> Any:
        ns: dict[str, Any] = {"x": x, "np": np, "pi": np.pi, "__builtins__": {}}
        result = eval(condition, ns)  # noqa: S307 -- restricted namespace, Docker-sandboxed
        result = np.asarray(result)
        if result.dtype.kind != 'b':
            result = result.astype(bool)
        return result
    return marker
