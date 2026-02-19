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

from .._app import get_session, mcp
from ..errors import (
    DOLFINxAPIError,
    DOLFINxMCPError,
    InvalidUFLExpressionError,
    PostconditionError,
    PreconditionError,
    handle_tool_errors,
)
from ..eval_helpers import eval_numpy_expression, make_boundary_marker
from ..session import BCInfo, FormInfo, FunctionSpaceInfo
from ..ufl_context import build_namespace, safe_evaluate
from ._validators import require_finite, require_nonempty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BC expression helper (kept local -- distinct from eval_helpers.py because
# it merges an external UFL namespace with numpy overrides)
# ---------------------------------------------------------------------------


_BC_EXPR_NS: dict | None = None


def _get_bc_expr_ns() -> dict:
    """Return cached numpy-override namespace for BC expressions."""
    global _BC_EXPR_NS
    if _BC_EXPR_NS is None:
        import numpy as np

        _BC_EXPR_NS = {
            "np": np, "pi": np.pi, "e": np.e,
            "sin": np.sin, "cos": np.cos, "exp": np.exp,
            "sqrt": np.sqrt, "abs": np.abs, "log": np.log,
            "tan": np.tan, "__builtins__": {},
        }
    return _BC_EXPR_NS


def _eval_bc_expression(expr: str, x: Any, ns: dict[str, Any]) -> Any:
    """Evaluate a BC value expression at coordinate arrays.

    Overrides UFL math functions with numpy equivalents because BC
    interpolation operates on coordinate arrays, not symbolic UFL expressions.

    SECURITY: _check_forbidden is called eagerly, __builtins__ is empty,
    and the Docker container provides the final security boundary.
    """
    import numpy as np

    from ..ufl_context import _check_forbidden

    _check_forbidden(expr)

    local_ns = {**ns, **_get_bc_expr_ns(), "x": x}
    result = eval(expr, local_ns)  # noqa: S307 -- restricted namespace, Docker-sandboxed
    if isinstance(result, (int, float)):
        return np.full(x.shape[1], float(result))
    return result


def _validate_bc_value(value: str | float | list[float]) -> None:
    """Validate boundary condition value (scalar, vector, or expression)."""
    if isinstance(value, (int, float)):
        require_finite(value, "BC value")
    elif isinstance(value, list):
        if len(value) == 0:
            raise PreconditionError(
                "BC list value must be non-empty.",
                suggestion="Provide at least one component value, e.g. [1.0, 0.0].",
            )
        if not all(isinstance(v, (int, float)) for v in value):
            raise PreconditionError(
                "All elements in BC list value must be numeric.",
                suggestion="Provide a list of floats, e.g. [1.0, 0.0].",
            )
        for i, v in enumerate(value):
            require_finite(v, f"BC element [{i}]")


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

    Returns:
        dict with bilinear_form ("compiled"), linear_form ("compiled"),
        bilinear_expression (str), linear_expression (str),
        trial_space (str), and test_space (str).
    """
    # Preconditions
    require_nonempty(bilinear, "bilinear form expression")
    require_nonempty(linear, "linear form expression")

    session = get_session(ctx)

    # Resolve function spaces (before lazy imports for host-side preconditions)
    if trial_space is not None:
        V_trial_info = session.get_space(trial_space)
    else:
        V_trial_info = session.get_only_space()

    V_test_info = (
        session.get_space(test_space) if test_space is not None else V_trial_info
    )

    # PS-5: trial and test spaces must be on the same mesh
    if V_trial_info.mesh_name != V_test_info.mesh_name:
        raise PreconditionError(
            f"Trial space '{V_trial_info.name}' is on mesh '{V_trial_info.mesh_name}' "
            f"but test space '{V_test_info.name}' is on mesh '{V_test_info.mesh_name}'.",
            suggestion="Both trial and test spaces must be on the same mesh.",
        )

    import dolfinx.fem
    import ufl

    V_trial = V_trial_info.space
    V_test = V_test_info.space

    # Build UFL namespace with trial/test functions
    ns = build_namespace(session, V_trial_info.mesh_name)
    u = ufl.TrialFunction(V_trial)
    v = ufl.TestFunction(V_test)
    ns["u"] = u
    ns["v"] = v

    # Detect mixed space for enhanced error messages
    is_mixed = V_trial_info.element_family == "Mixed"

    # Evaluate bilinear form with mixed-space-aware error handling
    try:
        a_ufl = safe_evaluate(bilinear, ns)
    except InvalidUFLExpressionError as exc:
        if is_mixed and "is not defined" in str(exc):
            raise InvalidUFLExpressionError(
                str(exc),
                suggestion=(
                    "For mixed function spaces, decompose trial/test functions "
                    "using split(): e.g. split(u)[0] for velocity, split(u)[1] "
                    "for pressure, split(v)[0] and split(v)[1] for test functions. "
                    "Do NOT use separate variable names like 'p' or 'q'."
                ),
            ) from exc
        raise

    # Evaluate linear form with mixed-space-aware error handling
    try:
        L_ufl = safe_evaluate(linear, ns)
    except InvalidUFLExpressionError as exc:
        if is_mixed and "is not defined" in str(exc):
            raise InvalidUFLExpressionError(
                str(exc),
                suggestion=(
                    "For mixed function spaces, decompose trial/test functions "
                    "using split(): e.g. split(u)[0] for velocity, split(u)[1] "
                    "for pressure, split(v)[0] and split(v)[1] for test functions."
                ),
            ) from exc
        raise

    # Compile forms
    try:
        a_compiled = dolfinx.fem.form(a_ufl)
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        suggestion = "Check that the bilinear form involves both u (trial) and v (test)."
        if is_mixed:
            suggestion += (
                " For saddle-point systems (Stokes), use MUMPS solver: "
                "petsc_options={'pc_factor_mat_solver_type': 'mumps'}."
            )
        raise DOLFINxAPIError(
            f"Failed to compile bilinear form: {exc}",
            suggestion=suggestion,
        ) from exc

    try:
        L_compiled = dolfinx.fem.form(L_ufl)
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to compile linear form: {exc}",
            suggestion="Check that the linear form involves v (test) but not u (trial).",
        ) from exc

    # Postcondition: compiled forms must not be None
    if a_compiled is None or L_compiled is None:
        raise PostconditionError(
            "Form compilation returned None.",
            suggestion="Check UFL expressions produce valid forms.",
        )

    # Store forms (trial_space_name enables solver auto-config for mixed spaces)
    session.forms["bilinear"] = FormInfo(
        name="bilinear",
        form=a_compiled,
        ufl_form=a_ufl,
        description=bilinear,
        trial_space_name=V_trial_info.name,
    )
    session.forms["linear"] = FormInfo(
        name="linear",
        form=L_compiled,
        ufl_form=L_ufl,
        description=linear,
        trial_space_name=V_trial_info.name,
    )

    if __debug__:
        session.check_invariants()

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
    value: str | float | list[float],
    boundary: str | None = None,
    boundary_tag: int | None = None,
    function_space: str | None = None,
    sub_space: int | None = None,
    name: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Apply a Dirichlet boundary condition.

    Args:
        value: BC value -- a float for constant, a list of floats for
            vector-valued BCs (e.g., [1.0, 0.0] for 2D), or a UFL
            expression string.
        boundary: Geometric boundary condition as a Python expression of x.
            Example: "np.isclose(x[0], 0.0)" for left boundary.
            Use "True" for all boundary DOFs.
        boundary_tag: Integer tag for marked boundary facets (alternative to geometric).
        function_space: Name of the function space. Defaults to the only space.
        sub_space: Sub-space index for mixed/vector spaces.
        name: Name for this BC. Auto-generated if omitted.

    Returns:
        dict with name, space_name, num_dofs (constrained DOF count),
        and description (summary of value and boundary).
    """
    if sub_space is not None and sub_space < 0:
        raise PreconditionError(f"sub_space must be >= 0, got {sub_space}.")

    _validate_bc_value(value)

    # PS-3: boundary and boundary_tag are mutually exclusive
    if boundary is not None and boundary_tag is not None:
        raise PreconditionError(
            "Cannot specify both 'boundary' (geometric) and 'boundary_tag' (tagged). Use one.",
            suggestion="Use 'boundary' for geometric or 'boundary_tag' for tagged boundaries.",
        )

    session = get_session(ctx)

    # Resolve function space (no dolfinx import needed)
    if function_space is not None:
        fs_info = session.get_space(function_space)
    else:
        fs_info = session.get_only_space()

    V = fs_info.space

    # Precondition: sub_space must be within range
    if sub_space is not None:
        num_subs = getattr(V, "num_sub_spaces", 0)
        if num_subs == 0:
            raise PreconditionError(
                f"sub_space={sub_space} specified but space '{fs_info.name}' has no sub-spaces.",
                suggestion="Remove sub_space parameter, or use a mixed/vector function space.",
            )
        if sub_space >= num_subs:
            raise PreconditionError(
                f"sub_space={sub_space} out of range. Space '{fs_info.name}' has "
                f"{num_subs} sub-spaces (0..{num_subs - 1}).",
            )

    import dolfinx.fem
    import dolfinx.mesh
    import numpy as np

    mesh = session.get_mesh(fs_info.mesh_name).mesh

    # Determine the space for DOF location (may be a sub-space)
    if sub_space is not None:
        V_collapse, _ = V.sub(sub_space).collapse()
        V_dof = V.sub(sub_space)
    else:
        V_collapse = V
        V_dof = V

    # Create the BC value
    # When sub_space is set, DOF location returns a pair (parent, collapsed);
    # the Constant overload of dirichletbc expects a plain ndarray.
    # So for sub_space + scalar, we interpolate into a Function instead.
    if isinstance(value, list):
        # Vector-valued BC: validate dimension and interpolate constant vector
        value_shape = V_collapse.ufl_element().reference_value_shape
        if not value_shape:
            raise PreconditionError(
                f"List value provided but space '{fs_info.name}' is scalar (no vector components).",
                suggestion="Use a float for scalar spaces, or use a vector function space.",
            )
        expected_dim = value_shape[0]
        if len(value) != expected_dim:
            raise PreconditionError(
                f"List has {len(value)} components but space expects {expected_dim}.",
                suggestion=f"Provide a list with exactly {expected_dim} values.",
            )
        bc_func = dolfinx.fem.Function(V_collapse)
        vec_values = np.array(value, dtype=np.float64)
        bc_func.interpolate(
            lambda x, _v=vec_values: np.tile(_v.reshape(-1, 1), (1, x.shape[1]))
        )
        if not np.isfinite(bc_func.x.array).all():
            raise PostconditionError(
                "Vector BC produced non-finite values after interpolation.",
                suggestion="Check that all list components are finite.",
            )
        bc_value = bc_func
    elif isinstance(value, (int, float)) and sub_space is None:
        bc_value = dolfinx.fem.Constant(mesh, float(value))
    elif isinstance(value, (int, float)):
        # sub_space is set -- use Function for compatibility with dofs pair
        bc_func = dolfinx.fem.Function(V_collapse)
        bc_func.x.array[:] = float(value)
        bc_value = bc_func
    else:
        # Expression string -- interpolate into function
        ns = build_namespace(session, fs_info.mesh_name)
        ns["np"] = np

        try:
            bc_func = dolfinx.fem.Function(V_collapse)
            bc_func.interpolate(
                lambda x: _eval_bc_expression(value, x, ns)
            )
            # PS-2: interpolated BC must be finite
            if not np.isfinite(bc_func.x.array).all():
                raise PostconditionError(
                    "BC expression produced non-finite values after interpolation.",
                    suggestion="Check expression produces finite values over the boundary.",
                )
            bc_value = bc_func
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to interpolate BC value expression: {exc}",
                suggestion="Check the expression syntax. Use x[0], x[1], x[2] for coordinates.",
            ) from exc

    # Locate boundary DOFs
    if boundary is not None:
        try:
            boundary_fn = make_boundary_marker(boundary)
            fdim = mesh.topology.dim - 1
            boundary_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, fdim, boundary_fn
            )
            dofs = dolfinx.fem.locate_dofs_topological(
                (V_dof, V_collapse) if sub_space is not None else V_dof,
                fdim,
                boundary_facets,
            )
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to locate boundary DOFs: {exc}",
                suggestion="Check boundary expression. Example: 'np.isclose(x[0], 0.0)'",
            ) from exc
    elif boundary_tag is not None:
        # Look up facet tags for this mesh
        mesh_name = fs_info.mesh_name
        tag_info = None

        # O(1) cache lookup
        cached = session._boundary_tag_cache.get(mesh_name)
        if cached and cached in session.mesh_tags:
            tag_info = session.mesh_tags[cached]

        # Fallback scan
        if tag_info is None:
            fdim = mesh.topology.dim - 1
            for _tn, _ti in session.mesh_tags.items():
                if _ti.mesh_name == mesh_name and _ti.dimension == fdim:
                    tag_info = _ti
                    break

        # PRE: mesh tags must exist
        if tag_info is None:
            raise PreconditionError(
                f"No boundary tags found for mesh '{mesh_name}'.",
                suggestion="Call mark_boundaries() first to tag boundary facets, "
                "or use a geometric 'boundary' condition instead.",
            )

        # PRE: requested tag must exist in available tags
        if boundary_tag not in tag_info.unique_tags:
            raise PreconditionError(
                f"Boundary tag {boundary_tag} not found. "
                f"Available tags: {tag_info.unique_tags}.",
                suggestion=f"Valid tags for mesh '{mesh_name}': {tag_info.unique_tags}.",
            )

        try:
            fdim = mesh.topology.dim - 1
            tagged_facets = tag_info.tags.find(boundary_tag)
            dofs = dolfinx.fem.locate_dofs_topological(
                (V_dof, V_collapse) if sub_space is not None else V_dof,
                fdim,
                tagged_facets,
            )
        except DOLFINxMCPError:
            raise
        except Exception as exc:
            raise DOLFINxAPIError(
                f"Failed to locate DOFs for boundary tag {boundary_tag}: {exc}",
                suggestion="Check that the tag corresponds to valid boundary facets.",
            ) from exc
    else:
        raise DOLFINxAPIError(
            "Either 'boundary' (geometric) or 'boundary_tag' must be specified.",
        )

    # Create DirichletBC
    # DOLFINx 0.10+: Function overload takes (func, dofs) without V for
    # full-space BCs, or (func, dofs_pair, V_sub) for sub-space BCs;
    # Constant overload takes (constant, dofs, V).
    try:
        if isinstance(bc_value, dolfinx.fem.Function) and sub_space is not None:
            bc = dolfinx.fem.dirichletbc(bc_value, dofs, V_dof)
        elif isinstance(bc_value, dolfinx.fem.Function):
            bc = dolfinx.fem.dirichletbc(bc_value, dofs)
        elif sub_space is not None:
            bc = dolfinx.fem.dirichletbc(bc_value, dofs, V_dof)
        else:
            bc = dolfinx.fem.dirichletbc(bc_value, dofs, V)
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to create DirichletBC: {exc}",
            suggestion="Check BC value and boundary specification.",
        ) from exc

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

    if __debug__:
        session.check_invariants()

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

    Returns:
        dict with name, type ("constant" or "interpolated"), and either
        value (float, for constants) or expression (str, for interpolated).
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

    if isinstance(value, (int, float)):
        require_finite(value, "Material property value")

    import dolfinx.fem
    import numpy as np

    session = get_session(ctx)

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

    # Auto-coerce numeric strings to Constants (avoids shape mismatch on vector spaces)
    import math
    try:
        numeric_val = float(value)
        if math.isfinite(numeric_val):
            constant = dolfinx.fem.Constant(mesh_info.mesh, numeric_val)
            session.ufl_symbols[name] = constant
            logger.info("Set constant material property '%s' = %s (coerced from string)", name, value)
            return {"name": name, "type": "constant", "value": numeric_val}
    except (ValueError, TypeError):
        pass  # Not numeric, proceed with interpolation

    # Detect vector space: scalar expressions need a scalar space
    value_shape = V.ufl_element().reference_value_shape
    is_vector_space = len(value_shape) > 0
    actual_space_name = fs_info.name

    if is_vector_space:
        # Auto-create scalar CG space for scalar material expression
        scalar_space_name = f"_scalar_{fs_info.name}"
        if scalar_space_name in session.function_spaces:
            V_scalar = session.function_spaces[scalar_space_name].space
        else:
            mesh = mesh_info.mesh
            degree = fs_info.element_degree
            V_scalar = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
            session.function_spaces[scalar_space_name] = FunctionSpaceInfo(
                name=scalar_space_name,
                space=V_scalar,
                mesh_name=fs_info.mesh_name,
                element_family="Lagrange",
                element_degree=degree,
                num_dofs=V_scalar.dofmap.index_map.size_global,
            )
        func = dolfinx.fem.Function(V_scalar)
        actual_space_name = scalar_space_name
        logger.info(
            "Auto-created scalar space for material '%s' on vector space '%s'",
            name, fs_info.name,
        )
    else:
        func = dolfinx.fem.Function(V)

    try:
        func.interpolate(
            lambda x: eval_numpy_expression(value, x)
        )
    except DOLFINxMCPError:
        raise
    except Exception as exc:
        raise DOLFINxAPIError(
            f"Failed to interpolate material expression '{value}': {exc}",
            suggestion="Check expression syntax. Use x[0], x[1] for coords, pi for constants.",
        ) from exc

    # Postcondition: interpolated material property must be finite
    if not np.isfinite(func.x.array).all():
        raise PostconditionError(
            f"Material property '{name}' contains NaN/Inf after interpolation.",
            suggestion="Check expression produces finite values over the domain.",
        )

    session.ufl_symbols[name] = func
    if name in session.functions:
        del session.functions[name]
    session.register_function(
        name, func, actual_space_name,
        description=f"Material property: {value}",
    )

    if __debug__:
        session.check_invariants()

    logger.info("Set interpolated material property '%s' = %s", name, value)
    return {"name": name, "type": "interpolated", "expression": value}
