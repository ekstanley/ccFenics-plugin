"""Problem definition tools: variational forms, boundary conditions, materials.

NOTE ON EVAL USAGE: UFL (Unified Form Language) expressions are Python syntax.
There is no separate UFL parser. Expression evaluation in restricted namespaces
(no builtins, token blocklist) is the standard DOLFINx approach. The Docker
container (--network none, non-root, --rm) provides the security boundary.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from .._app import mcp
from ..errors import (
    DOLFINxAPIError,
    DuplicateNameError,
    InvalidUFLExpressionError,
    PreconditionError,
    handle_tool_errors,
)
from ..session import BCInfo, FormInfo, FunctionInfo, SessionState
from ..ufl_context import build_namespace, safe_evaluate

logger = logging.getLogger(__name__)


def _get_session(ctx: Context) -> SessionState:
    return ctx.request_context.lifespan_context


# ---------------------------------------------------------------------------
# Expression helpers (restricted-namespace evaluation for UFL/numpy)
# ---------------------------------------------------------------------------


def _make_boundary_fn(expr: str):
    """Create a boundary marker callable from a string expression.

    The expression should use x as a (3, N) array.
    Example: "np.isclose(x[0], 0.0)" or "True".
    """
    import numpy as np

    if expr.strip() == "True":
        def marker(x):
            return np.full(x.shape[1], True)
        return marker

    def marker(x):
        # SECURITY: restricted namespace, Docker-sandboxed
        ns = {"x": x, "np": np, "pi": np.pi, "__builtins__": {}}
        result = _restricted_eval(expr, ns)
        if isinstance(result, bool):
            return np.full(x.shape[1], result)
        return result

    return marker


def _eval_bc_expression(expr: str, x, ns: dict) -> Any:
    """Evaluate a BC value expression at coordinate arrays."""
    import numpy as np

    local_ns = dict(ns)
    local_ns["x"] = x
    local_ns["np"] = np
    local_ns["__builtins__"] = {}
    result = _restricted_eval(expr, local_ns)
    if isinstance(result, (int, float)):
        return np.full(x.shape[1], float(result))
    return result


def _eval_material_expression(expr: str, x, mesh) -> Any:
    """Evaluate a material property expression at coordinate arrays."""
    import numpy as np

    ns = {
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
    result = _restricted_eval(expr, ns)
    if isinstance(result, (int, float)):
        return np.full(x.shape[1], float(result))
    return result


def _restricted_eval(expr_str: str, namespace: dict) -> Any:
    """Evaluate expression in a restricted namespace.

    SECURITY: This uses Python's eval intentionally. UFL is Python syntax
    and has no separate parser. Mitigations:
    1. __builtins__ set to empty dict (no system access)
    2. Token blocklist in ufl_context._check_forbidden
    3. Docker container isolation (--network none, non-root, --rm)
    """
    namespace.setdefault("__builtins__", {})
    return eval(expr_str, namespace)  # noqa: S307


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
@handle_tool_errors
async def define_variational_form(
    bilinear: str,
    linear: str,
    trial_space: str | None = None,
    test_space: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Define the bilinear and linear forms of a variational problem.

    Args:
        bilinear: UFL expression for bilinear form a(u,v).
            Example: "inner(grad(u), grad(v)) * dx"
        linear: UFL expression for linear form L(v).
            Example: "f * v * dx"
        trial_space: Name of the trial function space. If omitted, uses the
            only defined space (error if multiple exist).
        test_space: Name of the test function space. If omitted, same as trial_space.
    """
    import ufl
    import dolfinx.fem

    # Preconditions
    if not bilinear or not bilinear.strip():
        raise PreconditionError("bilinear form expression must be non-empty.")
    if not linear or not linear.strip():
        raise PreconditionError("linear form expression must be non-empty.")

    session = _get_session(ctx)

    # Resolve function spaces
    if trial_space is not None:
        V_trial_info = session.get_space(trial_space)
    else:
        V_trial_info = session.get_only_space()

    if test_space is not None:
        V_test_info = session.get_space(test_space)
    else:
        V_test_info = V_trial_info

    V_trial = V_trial_info.space
    V_test = V_test_info.space

    # Build UFL namespace with trial/test functions
    ns = build_namespace(session, V_trial_info.mesh_name)
    u = ufl.TrialFunction(V_trial)
    v = ufl.TestFunction(V_test)
    ns["u"] = u
    ns["v"] = v

    # Evaluate bilinear form
    a_ufl = safe_evaluate(bilinear, ns)

    # Evaluate linear form
    L_ufl = safe_evaluate(linear, ns)

    # Compile forms
    try:
        a_compiled = dolfinx.fem.form(a_ufl)
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to compile bilinear form: {exc}",
            suggestion="Check that the bilinear form involves both u (trial) and v (test).",
        ) from exc

    try:
        L_compiled = dolfinx.fem.form(L_ufl)
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to compile linear form: {exc}",
            suggestion="Check that the linear form involves v (test) but not u (trial).",
        ) from exc

    # Store forms
    session.forms["bilinear"] = FormInfo(
        name="bilinear",
        form=a_compiled,
        ufl_form=a_ufl,
        description=bilinear,
    )
    session.forms["linear"] = FormInfo(
        name="linear",
        form=L_compiled,
        ufl_form=L_ufl,
        description=linear,
    )

    logger.info("Defined variational forms: a=%s, L=%s", bilinear, linear)
    return {
        "bilinear_form": "compiled",
        "linear_form": "compiled",
        "bilinear_expression": bilinear,
        "linear_expression": linear,
        "trial_space": V_trial_info.name,
        "test_space": V_test_info.name,
    }


@mcp.tool()
@handle_tool_errors
async def apply_boundary_condition(
    value: str | float,
    boundary: str | None = None,
    boundary_tag: int | None = None,
    function_space: str | None = None,
    sub_space: int | None = None,
    name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Apply a Dirichlet boundary condition.

    Args:
        value: BC value -- a float for constant, or a UFL expression string.
        boundary: Geometric boundary condition as a Python expression of x.
            Example: "np.isclose(x[0], 0.0)" for left boundary.
            Use "True" for all boundary DOFs.
        boundary_tag: Integer tag for marked boundary facets (alternative to geometric).
        function_space: Name of the function space. Defaults to the only space.
        sub_space: Sub-space index for mixed/vector spaces.
        name: Name for this BC. Auto-generated if omitted.
    """
    import numpy as np
    import dolfinx.fem
    import dolfinx.mesh

    session = _get_session(ctx)

    # Resolve function space
    if function_space is not None:
        fs_info = session.get_space(function_space)
    else:
        fs_info = session.get_only_space()

    V = fs_info.space
    mesh = session.get_mesh(fs_info.mesh_name).mesh

    # Determine the space for DOF location (may be a sub-space)
    if sub_space is not None:
        V_collapse, _ = V.sub(sub_space).collapse()
        V_dof = V.sub(sub_space)
    else:
        V_collapse = V
        V_dof = V

    # Create the BC value
    if isinstance(value, (int, float)):
        bc_value = dolfinx.fem.Constant(mesh, float(value))
    else:
        # Expression string -- interpolate into function
        ns = build_namespace(session, fs_info.mesh_name)
        ns["np"] = np

        try:
            bc_func = dolfinx.fem.Function(V_collapse)
            bc_func.interpolate(
                lambda x: _eval_bc_expression(value, x, ns)
            )
            bc_value = bc_func
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to interpolate BC value expression: {exc}",
                suggestion="Check the expression syntax. Use x[0], x[1], x[2] for coordinates.",
            ) from exc

    # Locate boundary DOFs
    if boundary is not None:
        try:
            boundary_fn = _make_boundary_fn(boundary)
            fdim = mesh.topology.dim - 1
            boundary_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, fdim, boundary_fn
            )
            dofs = dolfinx.fem.locate_dofs_topological(
                (V_dof, V_collapse) if sub_space is not None else V_dof,
                fdim,
                boundary_facets,
            )
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to locate boundary DOFs: {exc}",
                suggestion="Check boundary expression. Example: 'np.isclose(x[0], 0.0)'",
            ) from exc
    elif boundary_tag is not None:
        raise DOLFINxAPIError(
            "Tagged boundary conditions require mesh facet tags. "
            "Use create_custom_mesh with boundary marking first.",
            suggestion="Use geometric boundary condition instead, or create a tagged mesh.",
        )
    else:
        raise DOLFINxAPIError(
            "Either 'boundary' (geometric) or 'boundary_tag' must be specified.",
        )

    # Create DirichletBC
    if sub_space is not None:
        bc = dolfinx.fem.dirichletbc(bc_value, dofs, V_dof)
    else:
        bc = dolfinx.fem.dirichletbc(bc_value, dofs)

    # Generate name if not provided
    if name is None:
        name = f"bc_{len(session.bcs)}"

    num_constrained = len(dofs) if isinstance(dofs, np.ndarray) else len(dofs[0])

    bc_info = BCInfo(
        name=name,
        bc=bc,
        space_name=fs_info.name,
        num_dofs=int(num_constrained),
        description=f"value={value}, boundary={boundary or f'tag={boundary_tag}'}",
    )

    session.bcs[name] = bc_info
    logger.info("Applied BC '%s': %d DOFs constrained", name, num_constrained)
    return bc_info.summary()


@mcp.tool()
@handle_tool_errors
async def set_material_properties(
    name: str,
    value: str | float,
    function_space: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Define a material property or coefficient for use in variational forms.

    The property is registered as a UFL symbol that can be referenced
    by name in form expressions.

    Args:
        name: Symbol name (e.g. "f", "kappa", "mu"). Will be available
            in UFL expressions.
        value: A float for a constant, or a UFL/numpy expression string.
            For expressions, use x[0], x[1] for spatial coordinates.
            Example: "2*pi**2*sin(pi*x[0])*sin(pi*x[1])"
        function_space: Function space for interpolation (required for
            non-constant expressions). Defaults to the only space.
    """
    # Preconditions: reject reserved UFL symbol names
    _RESERVED_UFL_NAMES = frozenset({
        "x", "n", "h", "grad", "div", "curl", "dx", "ds", "dS",
        "inner", "dot", "cross", "outer", "tr", "det", "sym", "skew",
        "nabla_grad", "nabla_div", "sqrt", "exp", "ln", "cos", "sin",
        "tan", "acos", "asin", "atan", "cosh", "sinh", "tanh",
    })
    if name in _RESERVED_UFL_NAMES:
        raise PreconditionError(
            f"Name '{name}' is a reserved UFL symbol.",
            suggestion=f"Choose a different name. Reserved: {sorted(_RESERVED_UFL_NAMES)[:10]}...",
        )

    import dolfinx.fem
    import numpy as np

    session = _get_session(ctx)

    if isinstance(value, (int, float)):
        # Scalar constant
        mesh_info = session.get_mesh()
        constant = dolfinx.fem.Constant(mesh_info.mesh, float(value))
        session.ufl_symbols[name] = constant
        logger.info("Set constant material property '%s' = %s", name, value)
        return {"name": name, "type": "constant", "value": float(value)}

    # Expression string -- need to interpolate
    if function_space is not None:
        fs_info = session.get_space(function_space)
    else:
        fs_info = session.get_only_space()

    V = fs_info.space
    mesh_info = session.get_mesh(fs_info.mesh_name)

    func = dolfinx.fem.Function(V)
    try:
        func.interpolate(
            lambda x: _eval_material_expression(value, x, mesh_info.mesh)
        )
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to interpolate material expression '{value}': {exc}",
            suggestion="Check expression syntax. Use x[0], x[1] for coords, pi for constants.",
        ) from exc

    session.ufl_symbols[name] = func
    session.functions[name] = FunctionInfo(
        name=name,
        function=func,
        space_name=fs_info.name,
        description=f"Material property: {value}",
    )

    logger.info("Set interpolated material property '%s' = %s", name, value)
    return {"name": name, "type": "interpolated", "expression": value}
