"""Docker integration tests: tutorial workflow coverage.

Run with:
    docker run --rm dolfinx-mcp:latest python -m pytest tests/test_tutorial_workflows.py -v

Tutorial coverage tests follow each DOLFINx tutorial chapter's workflow through
MCP tools, verifying end-to-end functionality.

Groups:
    T1 (2): Part 1 -- Fundamentals (Poisson, membrane)
    T2 (3): Part 2 -- Time-Dependent & Nonlinear (heat, elasticity, nonlinear Poisson)
    T3 (5): Part 3 -- Boundary Conditions (mixed, multiple Dirichlet, subdomains,
            Robin, component-wise)
    T4 (4): Part 4 -- Advanced Topics (convergence study, Helmholtz, mixed Poisson,
            singular Poisson)
    T5 (5): Part 5 -- Remaining Tutorial Coverage (Nitsche, hyperelasticity,
            electromagnetics, adaptive refinement, Stokes flow)
    T6 (6): Part 6 -- Official Demo Coverage (Allen-Cahn, biharmonic, DG Poisson,
            Lagrange variants, Laplacian eigenvalue, EM waveguide eigenvalue)

All gaps resolved in v0.7.0:
    G1: ds(tag) subdomain data -- build_namespace now attaches subdomain_data from mesh_tags
    G2: split() for mixed spaces -- ufl.split added to namespace
    G3: Nullspace support -- solve() now accepts nullspace_mode parameter

Official demo gaps closed in v0.8.0:
    DG operators (jump, avg), Lagrange element variants, SLEPc eigenvalue solver,
    Cahn-Hilliard/biharmonic skills.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

# Skip entire module if DOLFINx is not available (e.g. running outside Docker)
dolfinx = pytest.importorskip("dolfinx")

from dolfinx_mcp.session import SessionState  # noqa: E402


class _FakeContext:
    """Minimal mock of mcp.server.fastmcp.Context for testing tools directly."""

    def __init__(self, session: SessionState):
        self.request_context = MagicMock()
        self.request_context.lifespan_context = session


@pytest.fixture
def session() -> SessionState:
    return SessionState()


@pytest.fixture
def ctx(session: SessionState) -> _FakeContext:
    return _FakeContext(session)


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------


async def _setup_poisson_mms(
    session: SessionState, ctx: _FakeContext, N: int = 8, degree: int = 1
) -> tuple[dict, str]:
    """Poisson with manufactured solution sin(pi*x[0])*sin(pi*x[1]).

    Returns (solve_result, exact_expression_string).
    """
    from dolfinx_mcp.tools.mesh import create_unit_square
    from dolfinx_mcp.tools.problem import (
        apply_boundary_condition,
        define_variational_form,
        set_material_properties,
    )
    from dolfinx_mcp.tools.solver import solve
    from dolfinx_mcp.tools.spaces import create_function_space

    await create_unit_square(name="mesh", nx=N, ny=N, ctx=ctx)
    await create_function_space(name="V", family="Lagrange", degree=degree, ctx=ctx)
    await set_material_properties(
        name="f",
        value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])",
        ctx=ctx,
    )
    await define_variational_form(
        bilinear="inner(grad(u), grad(v)) * dx",
        linear="f * v * dx",
        ctx=ctx,
    )
    await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)
    result = await solve(solver_type="direct", ctx=ctx)
    return result, "sin(pi*x[0])*sin(pi*x[1])"


async def _setup_elasticity_2d(
    session: SessionState, ctx: _FakeContext
) -> dict:
    """2D linear elasticity on a beam with vector space shape=[2].

    Returns the solve result dict.
    """
    from dolfinx_mcp.tools.mesh import create_mesh
    from dolfinx_mcp.tools.problem import (
        apply_boundary_condition,
        define_variational_form,
        set_material_properties,
    )
    from dolfinx_mcp.tools.solver import solve
    from dolfinx_mcp.tools.spaces import create_function_space

    await create_mesh(
        name="mesh", shape="rectangle", nx=8, ny=4,
        dimensions={"width": 2.0, "height": 1.0}, ctx=ctx,
    )
    await create_function_space(
        name="V", family="Lagrange", degree=1, shape=[2], ctx=ctx,
    )
    # Lame parameters as scalar constants
    await set_material_properties(name="lambda_", value=1.0, ctx=ctx)
    await set_material_properties(name="mu", value=1.0, ctx=ctx)

    # Elasticity: sigma(u):eps(v) = f.v
    await define_variational_form(
        bilinear=(
            "inner(lambda_*nabla_div(u)*Identity(2) "
            "+ 2*mu*sym(grad(u)), sym(grad(v)))*dx"
        ),
        linear="inner(as_vector([0.0, -1.0]), v)*dx",
        ctx=ctx,
    )
    # Fixed left boundary (both components via sub_space)
    await apply_boundary_condition(
        value=0.0, boundary="np.isclose(x[0], 0.0)", sub_space=0, ctx=ctx,
    )
    await apply_boundary_condition(
        value=0.0, boundary="np.isclose(x[0], 0.0)", sub_space=1, ctx=ctx,
    )
    return await solve(solver_type="direct", ctx=ctx)


async def _setup_heat_equation(
    session: SessionState, ctx: _FakeContext,
    dt: float = 0.1, t_end: float = 0.3,
) -> dict:
    """Heat equation with backward Euler. Returns time-dependent result."""
    from dolfinx_mcp.tools.mesh import create_unit_square
    from dolfinx_mcp.tools.problem import (
        apply_boundary_condition,
        define_variational_form,
        set_material_properties,
    )
    from dolfinx_mcp.tools.solver import solve_time_dependent
    from dolfinx_mcp.tools.spaces import create_function_space

    await create_unit_square(name="mesh", nx=4, ny=4, ctx=ctx)
    await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)
    await set_material_properties(name="u_n", value="0*x[0]", ctx=ctx)
    await set_material_properties(name="f", value="1.0 + 0*x[0]", ctx=ctx)

    await define_variational_form(
        bilinear=f"(u*v + {dt}*inner(grad(u), grad(v))) * dx",
        linear=f"(u_n + {dt}*f) * v * dx",
        ctx=ctx,
    )
    await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

    return await solve_time_dependent(t_end=t_end, dt=dt, ctx=ctx)


# ---------------------------------------------------------------------------
# Group T1: Part 1 -- Fundamentals (2 tests)
# ---------------------------------------------------------------------------


class TestPart1Fundamentals:
    """Tutorial Part 1: Poisson and membrane fundamentals."""

    @pytest.mark.asyncio
    async def test_poisson_manufactured_solution(self, session, ctx):
        """T1.1: Ch1.1 -- Full Poisson pipeline with manufactured solution.

        Tools: create_unit_square -> create_function_space -> set_material_properties
        -> define_variational_form -> apply_boundary_condition -> solve -> compute_error

        MMS: f = 2*pi^2*sin(pi*x)*sin(pi*y), exact = sin(pi*x)*sin(pi*y)
        """
        from dolfinx_mcp.tools.postprocess import compute_error

        result, exact_expr = await _setup_poisson_mms(session, ctx, N=8, degree=1)

        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])

        # L2 error < 0.1 on coarse mesh (8x8, P1)
        error_result = await compute_error(
            exact=exact_expr, norm_type="L2", ctx=ctx,
        )
        assert "error" not in error_result
        assert error_result["error_value"] < 0.1
        assert error_result["error_value"] >= 0
        assert math.isfinite(error_result["error_value"])

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_membrane_deflection(self, session, ctx):
        """T1.2: Ch1.2 -- Membrane deflection with Gaussian load on rectangle.

        Tools: create_mesh(rectangle) -> create_function_space ->
        set_material_properties -> define_variational_form ->
        apply_boundary_condition -> solve -> evaluate_solution
        """
        from dolfinx_mcp.tools.mesh import create_mesh
        from dolfinx_mcp.tools.postprocess import evaluate_solution
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_mesh(
            name="mesh", shape="rectangle", nx=8, ny=8,
            dimensions={"width": 1.0, "height": 1.0}, ctx=ctx,
        )
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )
        # Gaussian load centered at (0.5, 0.5)
        await set_material_properties(
            name="p",
            value="4*exp(-50*((x[0]-0.5)**2 + (x[1]-0.5)**2))",
            ctx=ctx,
        )
        await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="p * v * dx",
            ctx=ctx,
        )
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True

        # Max deflection at center should be positive
        eval_result = await evaluate_solution(
            points=[[0.5, 0.5]], ctx=ctx,
        )
        assert "error" not in eval_result
        center_val = eval_result["evaluations"][0]["value"]
        if isinstance(center_val, list):
            center_val = center_val[0]
        assert center_val > 0
        assert math.isfinite(center_val)

        session.check_invariants()


# ---------------------------------------------------------------------------
# Group T2: Part 2 -- Time-Dependent & Nonlinear (3 tests)
# ---------------------------------------------------------------------------


class TestPart2TimeDependentNonlinear:
    """Tutorial Part 2: Time-dependent and nonlinear problems."""

    @pytest.mark.asyncio
    async def test_heat_equation_backward_euler(self, session, ctx):
        """T2.1: Ch2.1 -- Heat equation with backward Euler time integration.

        Tools: create_unit_square -> create_function_space ->
        set_material_properties (u_n, f) -> define_variational_form ->
        apply_boundary_condition -> solve_time_dependent
        """
        result = await _setup_heat_equation(session, ctx, dt=0.1, t_end=0.3)

        assert "error" not in result
        assert result["steps_completed"] > 0
        assert abs(result["final_time"] - 0.3) < 0.1
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_linear_elasticity_vector_space(self, session, ctx):
        """T2.2: Ch2.5 -- Linear elasticity with vector function space.

        Covers: Vector function space (shape=[2]), UFL operators (sym,
        nabla_div, Identity, as_vector), component-wise BCs via sub_space.
        """
        result = await _setup_elasticity_2d(session, ctx)

        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_nonlinear_poisson_qu_coefficient(self, session, ctx):
        """T2.3: Ch2.4 -- Nonlinear Poisson with q(u) = 1 + u^2 coefficient.

        Residual: (1 + u**2)*inner(grad(u),grad(v))*dx - f*v*dx
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve_nonlinear
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )
        await set_material_properties(
            name="f", value="1.0 + 0*x[0]", ctx=ctx,
        )
        # Mutable unknown with zero initial guess
        await set_material_properties(
            name="u", value="0.0 + 0*x[0]", ctx=ctx,
        )
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        result = await solve_nonlinear(
            residual="(1 + u**2)*inner(grad(u), grad(v))*dx - f*v*dx",
            unknown="u",
            ctx=ctx,
        )

        assert "error" not in result
        assert result["converged"] is True
        assert result["iterations"] >= 1
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])
        assert math.isfinite(result["residual_norm"])

        session.check_invariants()


# ---------------------------------------------------------------------------
# Group T3: Part 3 -- Boundary Conditions (5 tests)
# ---------------------------------------------------------------------------


class TestPart3BoundaryConditions:
    """Tutorial Part 3: Various boundary condition types."""

    @pytest.mark.asyncio
    async def test_mixed_dirichlet_neumann(self, session, ctx):
        """T3.1: Ch3.1 -- Mixed Dirichlet + Neumann BCs.

        Dirichlet u=0 on x[0]=0, Neumann g*v*ds(2) on x[0]=1.
        Uses ds(tag) with subdomain_data from mark_boundaries.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square, mark_boundaries
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        await mark_boundaries(
            markers=[
                {"tag": 1, "condition": "np.isclose(x[0], 0.0)"},
                {"tag": 2, "condition": "np.isclose(x[0], 1.0)"},
            ],
            ctx=ctx,
        )

        await set_material_properties(
            name="f", value="1.0 + 0*x[0]", ctx=ctx,
        )
        await set_material_properties(
            name="g", value="1.0 + 0*x[0]", ctx=ctx,
        )

        # Neumann term in linear form using ds(2)
        await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="f*v*dx + g*v*ds(2)",
            ctx=ctx,
        )
        await apply_boundary_condition(
            value=0.0, boundary="np.isclose(x[0], 0.0)", ctx=ctx,
        )

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_multiple_dirichlet_bcs(self, session, ctx):
        """T3.2: Ch3.2 -- Multiple Dirichlet BCs with different values.

        Laplace equation: u=0 on x[0]=0, u=1 on x[0]=1, f=0.
        Exact solution: u = x[0] (linear, exactly representable by P1).
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.postprocess import evaluate_solution
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        # Zero source (Laplace equation)
        await set_material_properties(name="f", value=0.0, ctx=ctx)

        await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="f * v * dx",
            ctx=ctx,
        )
        # u=0 on left, u=1 on right
        await apply_boundary_condition(
            value=0.0, boundary="np.isclose(x[0], 0.0)", ctx=ctx,
        )
        await apply_boundary_condition(
            value=1.0, boundary="np.isclose(x[0], 1.0)", ctx=ctx,
        )

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0

        # Evaluate at center -- exact u = x[0] = 0.5
        eval_result = await evaluate_solution(
            points=[[0.5, 0.5]], ctx=ctx,
        )
        assert "error" not in eval_result
        center_val = eval_result["evaluations"][0]["value"]
        if isinstance(center_val, list):
            center_val = center_val[0]
        assert abs(center_val - 0.5) < 0.05

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_material_subdomains_conditional(self, session, ctx):
        """T3.3: Ch3.3 -- Material subdomains using UFL conditional.

        Piecewise diffusivity: k=1 for x<0.5, k=10 for x>=0.5.
        Tests UFL conditional/lt operators in variational form.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )
        await set_material_properties(
            name="f", value="1.0 + 0*x[0]", ctx=ctx,
        )

        await define_variational_form(
            bilinear=(
                "conditional(lt(x[0], 0.5), 1.0, 10.0) "
                "* inner(grad(u), grad(v)) * dx"
            ),
            linear="f * v * dx",
            ctx=ctx,
        )
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_robin_boundary_condition(self, session, ctx):
        """T3.4: Ch3.4 -- Robin BC via form modification with ds(tag).

        Bilinear adds alpha*u*v*ds(2), linear adds alpha*g_robin*v*ds(2).
        Uses ds(tag) with subdomain_data from mark_boundaries.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square, mark_boundaries
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        await mark_boundaries(
            markers=[
                {"tag": 1, "condition": "np.isclose(x[0], 0.0)"},
                {"tag": 2, "condition": "np.isclose(x[0], 1.0)"},
            ],
            ctx=ctx,
        )

        await set_material_properties(
            name="f", value="1.0 + 0*x[0]", ctx=ctx,
        )
        await set_material_properties(name="alpha", value=10.0, ctx=ctx)
        await set_material_properties(name="g_robin", value=1.0, ctx=ctx)

        # Robin terms in both bilinear and linear forms
        await define_variational_form(
            bilinear="inner(grad(u), grad(v))*dx + alpha*u*v*ds(2)",
            linear="f*v*dx + alpha*g_robin*v*ds(2)",
            ctx=ctx,
        )
        await apply_boundary_condition(
            value=0.0, boundary="np.isclose(x[0], 0.0)", ctx=ctx,
        )

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert math.isfinite(result["solution_norm_L2"])

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_component_wise_bc(self, session, ctx):
        """T3.5: Ch3.5 -- Component-wise BCs on vector function space.

        Vector elasticity: fix x-component=0 on left, y-component=0 on bottom.
        Tests sub_space parameter in apply_boundary_condition.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, shape=[2], ctx=ctx,
        )
        await set_material_properties(name="lambda_", value=1.0, ctx=ctx)
        await set_material_properties(name="mu", value=1.0, ctx=ctx)

        await define_variational_form(
            bilinear=(
                "inner(lambda_*nabla_div(u)*Identity(2) "
                "+ 2*mu*sym(grad(u)), sym(grad(v)))*dx"
            ),
            linear="inner(as_vector([0.0, -1.0]), v)*dx",
            ctx=ctx,
        )
        # Fix x-component on left
        await apply_boundary_condition(
            value=0.0, boundary="np.isclose(x[0], 0.0)",
            sub_space=0, ctx=ctx,
        )
        # Fix y-component on bottom
        await apply_boundary_condition(
            value=0.0, boundary="np.isclose(x[1], 0.0)",
            sub_space=1, ctx=ctx,
        )

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0

        # Verify 2 BCs registered
        assert len(session.bcs) == 2

        session.check_invariants()


# ---------------------------------------------------------------------------
# Group T4: Part 4 -- Advanced Topics (4 tests)
# ---------------------------------------------------------------------------


class TestPart4AdvancedTopics:
    """Tutorial Part 4: Convergence, Helmholtz, mixed Poisson, singular."""

    @pytest.mark.asyncio
    async def test_convergence_rate_study(self, session, ctx):
        """T4.1: Ch4.4 -- Convergence rate study at 3 mesh resolutions.

        Loop over N=[4,8,16]: solve Poisson MMS, compute L2 error.
        Verify error decreases and rate ~ 2 for P1 elements.
        """
        from dolfinx_mcp.tools.postprocess import compute_error
        from dolfinx_mcp.tools.session_mgmt import reset_session

        errors = []
        mesh_sizes = [4, 8, 16]

        for N in mesh_sizes:
            result, exact_expr = await _setup_poisson_mms(
                session, ctx, N=N, degree=1,
            )
            assert "error" not in result
            assert result["converged"] is True

            error_result = await compute_error(
                exact=exact_expr, norm_type="L2", ctx=ctx,
            )
            assert "error" not in error_result
            errors.append(error_result["error_value"])

            # Reset for next iteration
            await reset_session(ctx=ctx)

        # Errors must decrease with refinement
        assert errors[1] < errors[0], f"Error did not decrease: {errors}"
        assert errors[2] < errors[1], f"Error did not decrease: {errors}"

        # Convergence rate ~ 2 for P1 (log ratio between successive h-halvings)
        rates = [
            math.log(errors[i] / errors[i + 1]) / math.log(2.0)
            for i in range(len(errors) - 1)
        ]
        for rate in rates:
            assert 1.5 < rate < 2.5, (
                f"Convergence rate {rate:.2f} outside expected range [1.5, 2.5]"
            )

    @pytest.mark.asyncio
    async def test_helmholtz_equation(self, session, ctx):
        """T4.2: Ch4.5 -- Helmholtz equation (Laplacian + reaction term).

        Form: (inner(grad(u),grad(v)) - k^2*u*v)*dx = f*v*dx
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        await set_material_properties(name="k", value=1.0, ctx=ctx)
        await set_material_properties(
            name="f", value="1.0 + 0*x[0]", ctx=ctx,
        )

        await define_variational_form(
            bilinear="(inner(grad(u), grad(v)) - k**2 * u * v) * dx",
            linear="f * v * dx",
            ctx=ctx,
        )
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_mixed_poisson_rt_dg(self, session, ctx):
        """T4.3: Ch4.1 -- Mixed Poisson with RT/DG elements.

        Creates RT(1) and DG(0) spaces, forms mixed space.
        Uses split() for sub-component access in mixed forms.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space, create_mixed_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V_rt", family="RT", degree=1, ctx=ctx,
        )
        await create_function_space(
            name="V_dg", family="DG", degree=0, ctx=ctx,
        )

        mix_result = await create_mixed_space(
            name="W", subspaces=["V_rt", "V_dg"], ctx=ctx,
        )
        assert "error" not in mix_result

        await set_material_properties(
            name="f", value=1.0, ctx=ctx,
        )

        # Mixed form uses split() to access sub-components
        form_result = await define_variational_form(
            bilinear=(
                "(inner(split(u)[0], split(v)[0]) "
                "+ split(u)[1]*div(split(v)[0]) "
                "+ div(split(u)[0])*split(v)[1]) * dx"
            ),
            linear="-f * split(v)[1] * dx",
            trial_space="W",
            test_space="W",
            ctx=ctx,
        )
        assert "error" not in form_result

        # Saddle-point system requires MUMPS for LU factorization
        result = await solve(
            solver_type="direct",
            petsc_options={"pc_factor_mat_solver_type": "mumps"},
            ctx=ctx,
        )
        assert "error" not in result
        assert result["converged"] is True

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_singular_poisson_pure_neumann(self, session, ctx):
        """T4.4: Singular Poisson with pure Neumann BCs.

        Pure Neumann (no Dirichlet BCs), zero-mean source f.
        Uses nullspace_mode="constant" to handle the singular system.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        # Zero-mean source (compatible with pure Neumann)
        await set_material_properties(
            name="f",
            value="sin(2*pi*x[0])*sin(2*pi*x[1])",
            ctx=ctx,
        )

        await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="f * v * dx",
            ctx=ctx,
        )

        # Pure Neumann -- attach constant nullspace for singular system
        result = await solve(
            solver_type="iterative", ksp_type="cg", pc_type="hypre",
            max_iter=1000, nullspace_mode="constant", ctx=ctx,
        )
        assert "error" not in result
        assert result["converged"] is True

        session.check_invariants()


# ---------------------------------------------------------------------------
# T5: Remaining Tutorial Coverage
# ---------------------------------------------------------------------------


class TestT5RemainingTutorialCoverage:
    """T5: Tests for tutorial chapters not covered in T1-T4.

    T5.1: Nitsche's method for weak Dirichlet BCs
    T5.2: Hyperelasticity (nonlinear large deformation)
    T5.3: Electromagnetics (Nedelec N1curl elements)
    T5.4: Adaptive mesh refinement (AMR)
    T5.5: Stokes flow (Taylor-Hood mixed elements)
    """

    @pytest.mark.asyncio
    async def test_nitsche_weak_dirichlet(self, session, ctx):
        """T5.1: Nitsche's method for imposing Dirichlet BCs weakly.

        Instead of strong Dirichlet BC imposition, use penalty terms on the
        boundary via ds. This tests FacetNormal (n), CellDiameter (h), and
        the ds measure — core Nitsche ingredients from the tutorial.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import define_variational_form
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        # Nitsche bilinear form: a(u,v) = inner(grad(u), grad(v))*dx
        #   - inner(dot(grad(u), n), v)*ds   (consistency)
        #   - inner(u, dot(grad(v), n))*ds    (symmetry / adjoint-consistency)
        #   + (alpha/h)*inner(u, v)*ds        (penalty)
        # Linear form: L(v) = f*v*dx
        #   - inner(g, dot(grad(v), n))*ds    (symmetry)
        #   + (alpha/h)*inner(g, v)*ds        (penalty)
        # where g = 0 (homogeneous Dirichlet), f = 1, alpha = penalty param

        alpha = 10.0
        bilinear = (
            "inner(grad(u), grad(v)) * dx"
            " - inner(dot(grad(u), n), v) * ds"
            " - inner(u, dot(grad(v), n)) * ds"
            f" + ({alpha}/h) * inner(u, v) * ds"
        )
        # g = 0 (homogeneous Dirichlet), so boundary terms with g vanish.
        # Only the volume source f*v*dx remains.
        linear = "1.0 * v * dx"

        form_result = await define_variational_form(
            bilinear=bilinear, linear=linear, ctx=ctx,
        )
        assert "error" not in form_result

        # No strong Dirichlet BCs -- Nitsche handles them weakly
        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_hyperelasticity_nonlinear(self, session, ctx):
        """T5.2: Hyperelasticity-inspired strong nonlinearity.

        Tests solve_nonlinear with an exponential coefficient (akin to
        neo-Hookean strain energy). Stronger nonlinearity than T2.3's
        q(u)=1+u^2, exercising Newton convergence for stiff problems.

        Residual: exp(u)*inner(grad(u), grad(v))*dx - f*v*dx
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve_nonlinear
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )

        # Body force
        await set_material_properties(name="f", value=1.0, ctx=ctx)

        # Mutable unknown with zero initial guess
        await set_material_properties(
            name="u_h", value="0.0 + 0*x[0]", ctx=ctx,
        )

        # Homogeneous Dirichlet on all boundaries
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        # Exponential nonlinearity: exp(u) * inner(grad(u), grad(v)) * dx
        # The solver maps unknown="u_h" to 'u' in the namespace
        residual = "exp(u)*inner(grad(u), grad(v))*dx - f*v*dx"

        result = await solve_nonlinear(
            residual=residual,
            unknown="u_h",
            snes_type="newtonls",
            ksp_type="preonly",
            pc_type="lu",
            max_iter=50,
            ctx=ctx,
        )
        assert "error" not in result
        assert result["converged"] is True
        assert result["iterations"] >= 1
        assert result["solution_norm_L2"] > 0

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_electromagnetics_nedelec(self, session, ctx):
        """T5.3: Electromagnetics with Nedelec (N1curl) edge elements.

        Tests H(curl) function space creation and curl operator in forms.
        Solves a simple curl-curl problem: curl(curl(E)) + E = f.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=4, ny=4, ctx=ctx)

        # N1curl = Nedelec first-kind edge elements for H(curl) space
        await create_function_space(
            name="V", family="N1curl", degree=1, ctx=ctx,
        )

        # curl-curl + mass:  inner(curl(u), curl(v))*dx + inner(u, v)*dx = inner(f, v)*dx
        # In 2D, curl of H(curl) element is a scalar, so inner() works.
        bilinear = "inner(curl(u), curl(v)) * dx + inner(u, v) * dx"
        linear = "inner(as_vector([1.0, 0.0]), v) * dx"

        form_result = await define_variational_form(
            bilinear=bilinear, linear=linear, ctx=ctx,
        )
        assert "error" not in form_result

        # No BCs (natural BCs for N1curl are tangential component = 0 on boundary)
        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_adaptive_mesh_refinement(self, session, ctx):
        """T5.4: Adaptive mesh refinement (AMR) loop.

        Solve on coarse mesh, refine, solve on fine mesh.
        Tests refine_mesh + re-solve pipeline (core AMR workflow).
        """
        from dolfinx_mcp.tools.mesh import create_unit_square, refine_mesh
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        # --- Coarse solve ---
        await create_unit_square(name="coarse", nx=4, ny=4, ctx=ctx)
        await create_function_space(
            name="V_coarse", family="Lagrange", degree=1, ctx=ctx,
        )
        await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="1.0 * v * dx",
            ctx=ctx,
        )
        await apply_boundary_condition(
            value=0.0, boundary="True", ctx=ctx,
        )
        coarse_result = await solve(
            solver_type="direct", solution_name="u_coarse", ctx=ctx,
        )
        assert "error" not in coarse_result
        assert coarse_result["converged"] is True

        coarse_norm = coarse_result["solution_norm_L2"]

        # Refine the coarse mesh (sets refined mesh as active)
        refine_result = await refine_mesh(
            name="coarse", new_name="fine", ctx=ctx,
        )
        assert "error" not in refine_result
        assert refine_result["refinement_factor"] > 1.0

        # --- Fine solve ---
        # Clear stale BCs/forms from coarse solve (they reference the old mesh)
        session.bcs.clear()
        session.forms.clear()

        await create_function_space(
            name="V_fine", family="Lagrange", degree=1, ctx=ctx,
        )
        # Specify trial_space explicitly (multiple spaces exist now)
        await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="1.0 * v * dx",
            trial_space="V_fine",
            ctx=ctx,
        )
        await apply_boundary_condition(
            value=0.0, boundary="True",
            function_space="V_fine", ctx=ctx,
        )
        fine_result = await solve(
            solver_type="direct", solution_name="u_fine", ctx=ctx,
        )
        assert "error" not in fine_result
        assert fine_result["converged"] is True

        # Both solves should produce positive norms
        assert coarse_norm > 0
        assert fine_result["solution_norm_L2"] > 0

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_stokes_flow_taylor_hood(self, session, ctx):
        """T5.5: Stokes flow with Taylor-Hood (P2/P1) mixed elements.

        Tests mixed function space creation and saddle-point solve.
        This is the Navier-Stokes proxy from the tutorial (Re=0 case).

        Solves: -div(grad(u)) + grad(p) = f, div(u) = 0
        Using Taylor-Hood elements (P2 velocity, P1 pressure).
        Uses split() to decompose mixed trial/test functions.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import (
            create_function_space,
            create_mixed_space,
        )

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)

        # Taylor-Hood: P2 velocity (vector), P1 pressure (scalar)
        await create_function_space(
            name="V", family="Lagrange", degree=2, shape=[2], ctx=ctx,
        )
        await create_function_space(
            name="Q", family="Lagrange", degree=1, ctx=ctx,
        )
        await create_mixed_space(
            name="W", subspaces=["V", "Q"], ctx=ctx,
        )

        # Stokes weak form on mixed space using split():
        # split(u)[0] = velocity trial, split(u)[1] = pressure trial
        # split(v)[0] = velocity test,  split(v)[1] = pressure test
        # a = inner(grad(vel_u), grad(vel_v))*dx
        #   - press_u * div(vel_v) * dx
        #   - press_v * div(vel_u) * dx
        # L = inner(f, vel_v) * dx
        bilinear = (
            "(inner(grad(split(u)[0]), grad(split(v)[0]))"
            " - split(u)[1] * div(split(v)[0])"
            " - split(v)[1] * div(split(u)[0])) * dx"
        )
        linear = "inner(as_vector([1.0, 0.0]), split(v)[0]) * dx"

        form_result = await define_variational_form(
            bilinear=bilinear,
            linear=linear,
            trial_space="W",
            test_space="W",
            ctx=ctx,
        )
        assert "error" not in form_result

        # No-slip BC on entire boundary (velocity = 0, sub_space=0)
        await apply_boundary_condition(
            value=0.0, boundary="True",
            function_space="W", sub_space=0, ctx=ctx,
        )

        # Saddle-point system requires MUMPS
        result = await solve(
            solver_type="direct",
            petsc_options={"pc_factor_mat_solver_type": "mumps"},
            ctx=ctx,
        )
        assert "error" not in result
        assert result["converged"] is True

        session.check_invariants()


# ---------------------------------------------------------------------------
# T6: Official Demo Coverage (v0.8.0)
# ---------------------------------------------------------------------------


class TestT6OfficialDemoCoverage:
    """Tests for official DOLFINx demo capabilities added in v0.8.0.

    T6.1: Allen-Cahn steady-state (nonlinear phase-field proxy)
    T6.2: Biharmonic mixed formulation (fourth-order PDE)
    T6.3: DG Poisson with SIPG (jump/avg operators)
    T6.4: Lagrange GLL variant (element variant parameter)
    T6.5: Laplacian eigenvalue (SLEPc EPS solver)
    T6.6: EM waveguide eigenvalue (Nedelec curl-curl)
    """

    @pytest.mark.asyncio
    async def test_t6_1_allen_cahn_steady_state(self, session, ctx):
        """Allen-Cahn: -eps^2*lap(u) + u^3 - u = 0 with tanh-profile BCs."""
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve_nonlinear
        from dolfinx_mcp.tools.spaces import create_function_space

        # Use finer mesh for better Newton convergence with sharp interface
        await create_unit_square(name="mesh", nx=32, ny=32, ctx=ctx)
        await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)

        # Initial guess: tanh-profile approximation for better convergence
        await set_material_properties(
            name="u", value="np.tanh((x[0] - 0.5) / 0.1)",
            function_space="V", ctx=ctx,
        )

        # Dirichlet BCs: u = -1 at x=0, u = 1 at x=1
        await apply_boundary_condition(
            value=-1.0, boundary="np.isclose(x[0], 0.0)", ctx=ctx,
        )
        await apply_boundary_condition(
            value=1.0, boundary="np.isclose(x[0], 1.0)", ctx=ctx,
        )

        # Solve: residual = eps^2*inner(grad(u),grad(v))*dx + (u^3 - u)*v*dx
        # Use eps^2 = 0.04 (eps=0.2) for easier convergence while still
        # demonstrating the phase-field interface
        result = await solve_nonlinear(
            residual="0.04*inner(grad(u), grad(v))*dx + (u**3 - u)*v*dx",
            unknown="u",
            ctx=ctx,
        )
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_t6_2_biharmonic_mixed(self, session, ctx):
        """Biharmonic: nabla^4 u = f via mixed (sigma, u) formulation."""
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import (
            create_function_space,
            create_mixed_space,
        )

        await create_unit_square(name="mesh", nx=16, ny=16, ctx=ctx)

        # Two P2 spaces for sigma and u
        await create_function_space(
            name="V_sigma", family="Lagrange", degree=2, ctx=ctx,
        )
        await create_function_space(
            name="V_u", family="Lagrange", degree=2, ctx=ctx,
        )
        await create_mixed_space(name="W", subspaces=["V_sigma", "V_u"], ctx=ctx)

        await set_material_properties(name="f", value=1.0, ctx=ctx)

        # Mixed bilinear form using split()
        bilinear = (
            "(split(u)[0]*split(v)[0]"
            " - inner(grad(split(u)[1]), grad(split(v)[0]))"
            " + inner(grad(split(u)[0]), grad(split(v)[1])))*dx"
        )
        linear = "f * split(v)[1] * dx"

        form_result = await define_variational_form(
            bilinear=bilinear,
            linear=linear,
            trial_space="W",
            test_space="W",
            ctx=ctx,
        )
        assert "error" not in form_result

        # BC: u = 0 on boundary (sub_space=1 for the u-component)
        await apply_boundary_condition(
            value=0.0, boundary="True",
            function_space="W", sub_space=1, ctx=ctx,
        )

        # Saddle-point system needs MUMPS
        result = await solve(
            solver_type="direct",
            petsc_options={"pc_factor_mat_solver_type": "mumps"},
            ctx=ctx,
        )
        assert "error" not in result
        assert result["converged"] is True

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_t6_3_dg_poisson_sipg(self, session, ctx):
        """DG Poisson with SIPG: jump/avg operators on interior facets."""
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(name="V", family="DG", degree=1, ctx=ctx)
        await set_material_properties(name="f", value=1.0, ctx=ctx)

        # SIPG bilinear form with jump/avg operators
        # Volume + consistency + symmetry + penalty + Nitsche boundary terms
        alpha = "10.0"
        bilinear = (
            "inner(grad(u), grad(v))*dx"
            " - inner(avg(nabla_grad(u)), jump(v, n))*dS"
            " - inner(jump(u, n), avg(nabla_grad(v)))*dS"
            f" + ({alpha})/avg(h) * inner(jump(u), jump(v))*dS"
            f" + ({alpha})/h * inner(u, v)*ds"
            " - inner(dot(grad(u), n), v)*ds"
            " - inner(u, dot(grad(v), n))*ds"
        )
        linear = "f*v*dx"

        form_result = await define_variational_form(
            bilinear=bilinear, linear=linear, ctx=ctx,
        )
        assert "error" not in form_result

        # No Dirichlet BCs (imposed weakly via Nitsche terms)
        result = await solve(solver_type="direct", ctx=ctx)
        assert "error" not in result
        assert result["converged"] is True

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_t6_4_lagrange_gll_variant(self, session, ctx):
        """Lagrange element with GLL variant (equispaced nodes -> GLL nodes)."""
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=4, ny=4, ctx=ctx)

        result = await create_function_space(
            name="V_gll", family="Lagrange", degree=3,
            variant="gll_warped", ctx=ctx,
        )
        assert "error" not in result
        assert result["num_dofs"] > 0
        assert result["variant"] == "gll_warped"

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_t6_5_laplacian_eigenvalue(self, session, ctx):
        """Laplacian eigenvalues on unit square: lambda_1 = 2*pi^2."""
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import apply_boundary_condition
        from dolfinx_mcp.tools.solver import solve_eigenvalue
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=16, ny=16, ctx=ctx)
        await create_function_space(
            name="V", family="Lagrange", degree=1, ctx=ctx,
        )
        await apply_boundary_condition(
            value=0.0, boundary="True", ctx=ctx,
        )

        result = await solve_eigenvalue(
            stiffness_form="inner(grad(u), grad(v))*dx",
            mass_form="inner(u, v)*dx",
            num_eigenvalues=4,
            function_space="V",
            ctx=ctx,
        )
        assert "error" not in result
        assert result["num_converged"] >= 4

        # Check first eigenvalue: lambda_1 = 2*pi^2 ~ 19.739
        lambda_exact = 2 * math.pi**2
        lambda_computed = result["eigenvalues"][0]["real"]
        relative_error = abs(lambda_computed - lambda_exact) / lambda_exact
        assert relative_error < 0.05, (
            f"First eigenvalue {lambda_computed:.4f} deviates from exact "
            f"{lambda_exact:.4f} by {relative_error:.2%}"
        )

        # Check eigenvectors stored in session
        assert "eig_0" in session.functions
        assert len(result["eigenvectors"]) >= 4

        session.check_invariants()

    @pytest.mark.asyncio
    async def test_t6_6_em_waveguide_eigenvalue(self, session, ctx):
        """EM waveguide: curl-curl eigenvalue with Nedelec elements.

        Uses natural BCs (PMC boundary) -- no explicit Dirichlet BCs needed
        since N1curl spaces require vector-valued BCs which need topological
        DOF location. The curl-curl operator is well-posed with natural BCs.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.solver import solve_eigenvalue
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(
            name="V", family="N1curl", degree=1, ctx=ctx,
        )

        result = await solve_eigenvalue(
            stiffness_form="inner(curl(u), curl(v))*dx",
            mass_form="inner(u, v)*dx",
            num_eigenvalues=6,
            which="smallest_magnitude",
            function_space="V",
            ctx=ctx,
        )
        assert "error" not in result
        assert result["num_converged"] >= 1

        # Eigenvalues should have non-negative real parts (curl-curl is PSD)
        for ev in result["eigenvalues"]:
            assert ev["real"] >= -1e-10, (
                f"Eigenvalue {ev['index']} has negative real part: {ev['real']}"
            )

        session.check_invariants()
