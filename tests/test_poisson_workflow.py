"""Integration test: full Poisson workflow inside Docker.

Run with:
    docker run --rm dolfinx-mcp:latest python -m pytest tests/test_poisson_workflow.py -v

This test exercises the complete Poisson equation solve:
    -div(grad(u)) = f on [0,1]^2
    u = 0 on boundary
    f = 2*pi^2*sin(pi*x)*sin(pi*y)
    exact: u = sin(pi*x)*sin(pi*y)
"""

from __future__ import annotations

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


class TestPoissonWorkflow:
    """End-to-end Poisson equation solve on unit square."""

    @pytest.mark.asyncio
    async def test_full_poisson(self, session: SessionState, ctx: _FakeContext):
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.postprocess import compute_error
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            define_variational_form,
            set_material_properties,
        )
        from dolfinx_mcp.tools.session_mgmt import get_session_state
        from dolfinx_mcp.tools.solver import solve
        from dolfinx_mcp.tools.spaces import create_function_space

        # Step 1: Create mesh
        result = await create_unit_square(name="mesh", nx=32, ny=32, ctx=ctx)
        assert result["name"] == "mesh"
        assert result["num_cells"] == 2048
        assert result["active"] is True

        # Step 2: Create P2 function space
        result = await create_function_space(
            name="V", family="Lagrange", degree=2, ctx=ctx
        )
        assert result["name"] == "V"
        assert result["element_degree"] == 2
        assert result["num_dofs"] > 0

        # Step 3: Define source term
        result = await set_material_properties(
            name="f",
            value="2*pi**2*sin(pi*x[0])*sin(pi*x[1])",
            ctx=ctx,
        )
        assert result["name"] == "f"
        assert result["type"] == "interpolated"

        # Step 4: Define variational form
        result = await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx",
            linear="f * v * dx",
            ctx=ctx,
        )
        assert result["bilinear_form"] == "compiled"
        assert result["linear_form"] == "compiled"

        # Step 5: Apply homogeneous Dirichlet BC on all boundaries
        result = await apply_boundary_condition(
            value=0.0,
            boundary="True",
            ctx=ctx,
        )
        assert result["num_dofs"] > 0

        # Step 6: Solve
        result = await solve(solver_type="direct", ctx=ctx)
        assert result["converged"] is True

        # Step 7: Compute L2 error
        result = await compute_error(
            exact="sin(pi*x[0])*sin(pi*x[1])",
            norm_type="L2",
            ctx=ctx,
        )
        error = result["error_value"]
        # P2 on 32x32 mesh should give L2 error < 1e-4
        assert error < 1e-4, f"L2 error too large: {error}"

        # Step 8: Check session state
        state = await get_session_state(ctx=ctx)
        assert state["active_mesh"] == "mesh"
        assert "V" in state["function_spaces"]
        assert "u_h" in state["solutions"]
