"""Safe UFL expression evaluation for DOLFINx MCP server.

UFL (Unified Form Language) expressions are Python syntax -- there is no
separate parser. We evaluate user-supplied expression strings in a restricted
namespace that includes UFL operators, numpy constants, and session-registered
symbols. The security boundary is the Docker container (--network none,
non-root, --rm).
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any

from .errors import InvalidUFLExpressionError, PostconditionError

if TYPE_CHECKING:
    from .session import SessionState

logger = logging.getLogger(__name__)

# Tokens that must never appear in user-supplied UFL expressions.
_FORBIDDEN_TOKENS = frozenset({
    "import",
    "__",
    "exec",
    "eval",
    "compile",
    "open",
    "os.",
    "sys.",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir(",
    "breakpoint",
})

# Regex to match forbidden tokens as whole words or prefixes.
_FORBIDDEN_RE = re.compile(
    "|".join(re.escape(tok) for tok in _FORBIDDEN_TOKENS),
    re.IGNORECASE,
)


def _check_forbidden(expr_str: str) -> None:
    """Raise if expression contains forbidden tokens."""
    match = _FORBIDDEN_RE.search(expr_str)
    if match:
        raise InvalidUFLExpressionError(
            f"Expression contains forbidden token: '{match.group()}'",
            suggestion="UFL expressions should only contain mathematical operations. "
            "Remove any system/import/exec calls.",
        )



# Module-level cache for static UFL namespace entries (lazy-initialized).
_STATIC_UFL_NS: dict[str, Any] | None = None


def _build_static_ns() -> dict[str, Any]:
    """Build the static UFL namespace once (lazy, on first call).

    Contains all UFL operators, math functions, constructors, and constants
    that never change between calls. Only mesh-specific geometry and session
    data are added per call in build_namespace().
    """
    import ufl
    import numpy as np

    return {
        "__builtins__": {},
        # UFL differential operators
        "grad": ufl.grad,
        "div": ufl.div,
        "curl": ufl.curl,
        "nabla_grad": ufl.nabla_grad,
        "nabla_div": ufl.nabla_div,
        # UFL algebra
        "inner": ufl.inner,
        "dot": ufl.dot,
        "cross": ufl.cross,
        "outer": ufl.outer,
        "tr": ufl.tr,
        "det": ufl.det,
        "dev": ufl.dev,
        "sym": ufl.sym,
        "skew": ufl.skew,
        "transpose": ufl.transpose,
        "Identity": ufl.Identity,
        "as_tensor": ufl.as_tensor,
        "as_vector": ufl.as_vector,
        "as_matrix": ufl.as_matrix,
        # UFL DG operators (jump/average across facets)
        "jump": ufl.jump,
        "avg": ufl.avg,
        "cell_avg": ufl.cell_avg,
        "facet_avg": ufl.facet_avg,
        # UFL math functions
        "sqrt": ufl.sqrt,
        "exp": ufl.exp,
        "ln": ufl.ln,
        "sin": ufl.sin,
        "cos": ufl.cos,
        "tan": ufl.tan,
        "asin": ufl.asin,
        "acos": ufl.acos,
        "atan": ufl.atan,
        "atan2": getattr(ufl, "atan_2", ufl.atan2),
        "abs": ufl.algebra.Abs,
        "sign": ufl.sign,
        "conditional": ufl.conditional,
        "le": ufl.le,
        "ge": ufl.ge,
        "lt": ufl.lt,
        "gt": ufl.gt,
        "eq": ufl.eq,
        "ne": ufl.ne,
        "Max": ufl.max_value,
        "Min": ufl.min_value,
        "max_value": ufl.max_value,
        "min_value": ufl.min_value,
        # UFL measures (base, without subdomain_data)
        "dx": ufl.dx,
        # UFL special forms
        "Dx": ufl.Dx,
        "variable": ufl.variable,
        "diff": ufl.diff,
        "split": ufl.split,
        "as_ufl": ufl.as_ufl,
        # UFL function constructors
        "TrialFunction": ufl.TrialFunction,
        "TestFunction": ufl.TestFunction,
        "TrialFunctions": ufl.TrialFunctions,
        "TestFunctions": ufl.TestFunctions,
        # UFL geometry constructors (unbound -- need mesh argument when called)
        "SpatialCoordinate": ufl.SpatialCoordinate,
        "FacetNormal": ufl.FacetNormal,
        "CellDiameter": ufl.CellDiameter,
        # Constants
        "pi": math.pi,
        "e": math.e,
        "np": np,
    }


def build_namespace(session: SessionState, mesh_name: str | None = None) -> dict[str, Any]:
    """Build a restricted namespace for UFL expression evaluation.

    Includes:
    - UFL operators (grad, div, inner, dot, dx, ds, dS, etc.)
    - numpy/math constants (pi, e, sin, cos, exp, sqrt, abs, ln)
    - Spatial coordinate x, facet normal n, cell diameter h
    - All session-registered functions, spaces, and UFL symbols
    """
    import ufl

    global _STATIC_UFL_NS
    if _STATIC_UFL_NS is None:
        _STATIC_UFL_NS = _build_static_ns()

    mesh_info = session.get_mesh(mesh_name)
    mesh = mesh_info.mesh

    # Start from cached static namespace (operators, math, constructors)
    ns = _STATIC_UFL_NS.copy()

    # Mesh-specific geometry
    ns["x"] = ufl.SpatialCoordinate(mesh)
    ns["n"] = ufl.FacetNormal(mesh)
    ns["h"] = ufl.CellDiameter(mesh)

    # Override bare dx with domain-bound measure (fixes multi-mesh ambiguity)
    ns["dx"] = ufl.Measure("dx", domain=mesh)

    # Look up facet tags for this mesh to enable ds(tag) and dS(tag)
    mesh_name_resolved = mesh_name or session.active_mesh
    boundary_tags = None
    interior_tags = None
    fdim = mesh.topology.dim - 1

    # O(1) cache lookup for boundary tags
    cached_bt = session._boundary_tag_cache.get(mesh_name_resolved)
    if cached_bt and cached_bt in session.mesh_tags:
        bt_info = session.mesh_tags[cached_bt]
        if bt_info.dimension == fdim:
            boundary_tags = bt_info.tags

    # O(1) cache lookup for interior tags
    cached_it = session._interior_tag_cache.get(mesh_name_resolved)
    if cached_it and cached_it in session.mesh_tags:
        it_info = session.mesh_tags[cached_it]
        if it_info.dimension == fdim:
            interior_tags = it_info.tags

    # Fallback scan for uncached tags (e.g. from manage_mesh_tags or custom code)
    if boundary_tags is None or interior_tags is None:
        for _tag_name, tag_info in session.mesh_tags.items():
            if tag_info.mesh_name == mesh_name_resolved and tag_info.dimension == fdim:
                if tag_info.name.startswith("interior_"):
                    if interior_tags is None:
                        interior_tags = tag_info.tags
                        session._interior_tag_cache[mesh_name_resolved] = tag_info.name
                elif boundary_tags is None:
                    boundary_tags = tag_info.tags
                    session._boundary_tag_cache[mesh_name_resolved] = tag_info.name
            if boundary_tags is not None and interior_tags is not None:
                break

    if boundary_tags is not None:
        ns["ds"] = ufl.Measure("ds", domain=mesh, subdomain_data=boundary_tags)
    else:
        ns["ds"] = ufl.ds

    if interior_tags is not None:
        ns["dS"] = ufl.Measure("dS", domain=mesh, subdomain_data=interior_tags)
    else:
        ns["dS"] = ufl.dS

    # Inject session-registered UFL symbols (materials, coefficients)
    ns.update(session.ufl_symbols)

    # Inject session functions (so forms can reference solution fields etc.)
    for fname, finfo in session.functions.items():
        ns[fname] = finfo.function

    if __debug__:
        _required = {"dx", "ds", "inner", "grad", "x", "jump", "avg"}
        missing = _required - ns.keys()
        if missing:
            raise PostconditionError(
                f"build_namespace(): missing required keys {missing}"
            )
    return ns


def safe_evaluate(expr_str: str, namespace: dict[str, Any]) -> Any:
    """Evaluate a UFL expression string in a restricted namespace.

    Uses eval() intentionally -- UFL is Python syntax and there is no
    alternative parser. Security is enforced by:
    1. Token blocklist (_check_forbidden)
    2. Empty __builtins__ dict (no access to system functions)
    3. Docker container isolation (--network none, non-root, --rm)

    Args:
        expr_str: Python/UFL expression string (e.g. "inner(grad(u), grad(v)) * dx")
        namespace: Pre-built namespace from build_namespace()

    Returns:
        Evaluated UFL form, expression, or Python value.

    Raises:
        InvalidUFLExpressionError: on forbidden tokens or evaluation failure.
    """
    _check_forbidden(expr_str)

    # Ensure builtins are blocked
    namespace.setdefault("__builtins__", {})

    try:
        # SECURITY: eval() is required here -- UFL has no separate parser.
        # The namespace is restricted (no builtins) and forbidden tokens are
        # pre-filtered. The Docker container provides the security boundary.
        result = eval(expr_str, namespace)  # noqa: S307 -- intentional, see docstring
    except SyntaxError as exc:
        raise InvalidUFLExpressionError(
            f"Syntax error in expression: {exc}",
            suggestion="Check parentheses and UFL operator syntax.",
        ) from exc
    except NameError as exc:
        raise InvalidUFLExpressionError(
            f"Unknown symbol in expression: {exc}",
            suggestion="Ensure all variables (u, v, f, etc.) are defined before use.",
        ) from exc
    except Exception as exc:
        raise InvalidUFLExpressionError(
            f"Failed to evaluate expression '{expr_str}': {exc}",
        ) from exc

    if result is None:
        raise InvalidUFLExpressionError(
            f"Expression '{expr_str}' evaluated to None",
            suggestion="Ensure the expression produces a value, not a statement.",
        )
    return result
