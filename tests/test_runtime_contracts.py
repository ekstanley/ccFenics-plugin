"""Docker integration tests: runtime contracts with real DOLFINx objects.

Run with:
    docker run --rm dolfinx-mcp:latest python -m pytest tests/test_runtime_contracts.py -v

Phase 7 tests exercise postconditions, debug invariants, and session cascade
operations that require real DOLFINx objects -- contracts that cannot be tested
without the FEniCSx runtime.

Groups:
    A (7): Positive-path -- runtime contracts pass during valid operations
    B (2): Negative-path -- postconditions detect and reject bad data
    C (3): Session operations -- cascade deletion and cleanup with real objects
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

# Skip entire module if DOLFINx is not available (e.g. running outside Docker)
dolfinx = pytest.importorskip("dolfinx")

from dolfinx_mcp.session import SessionState


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


async def _setup_poisson(session: SessionState, ctx: _FakeContext) -> dict:
    """Set up and solve standard Poisson problem on 8x8 P1 mesh.

    Returns the solve result dict. Uses a coarse mesh (8x8, degree 1) for
    speed -- sufficient for contract verification, not accuracy testing.
    """
    from dolfinx_mcp.tools.mesh import create_unit_square
    from dolfinx_mcp.tools.spaces import create_function_space
    from dolfinx_mcp.tools.problem import (
        apply_boundary_condition,
        define_variational_form,
        set_material_properties,
    )
    from dolfinx_mcp.tools.solver import solve

    await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
    await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)
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
    return await solve(solver_type="direct", ctx=ctx)


# ---------------------------------------------------------------------------
# Group A: Positive-Path Runtime Contract Verification (7 tests)
# ---------------------------------------------------------------------------


class TestPositivePathContracts:
    """Verify runtime postconditions and debug invariants pass during valid operations."""

    @pytest.mark.asyncio
    async def test_solve_postconditions_pass(self, session, ctx):
        """A1: solve() postconditions -- solution finite, L2 >= 0, debug invariant.

        Exercises: solver.py:113 (finite check), solver.py:135 (L2 >= 0),
        solver.py:170 (debug invariant).
        """
        result = await _setup_poisson(session, ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])

    @pytest.mark.asyncio
    async def test_solver_diagnostics_l2_norm(self, session, ctx):
        """A2: get_solver_diagnostics() L2 >= 0 postcondition.

        Exercises: solver.py:440 (L2 non-negative postcondition).
        """
        from dolfinx_mcp.tools.solver import get_solver_diagnostics

        await _setup_poisson(session, ctx)
        result = await get_solver_diagnostics(ctx=ctx)
        assert "error" not in result
        assert result["solution_norm_L2"] >= 0
        assert math.isfinite(result["solution_norm_L2"])

    @pytest.mark.asyncio
    async def test_compute_error_postconditions_pass(self, session, ctx):
        """A3: compute_error() finite + non-negative postconditions.

        Exercises: postprocess.py:134 (finite check), postprocess.py:138 (>= 0).
        """
        from dolfinx_mcp.tools.postprocess import compute_error

        await _setup_poisson(session, ctx)
        result = await compute_error(
            exact="sin(pi*x[0])*sin(pi*x[1])",
            norm_type="L2",
            ctx=ctx,
        )
        assert "error" not in result
        assert result["error_value"] >= 0
        assert math.isfinite(result["error_value"])

    @pytest.mark.asyncio
    async def test_interpolate_expression_postcondition_pass(self, session, ctx):
        """A4: interpolate() finiteness postcondition + debug invariant.

        Exercises: interpolation.py:106 (finite check), interpolation.py:117
        (debug invariant).
        """
        from dolfinx_mcp.tools.interpolation import interpolate

        await _setup_poisson(session, ctx)
        # Interpolate a new expression into the existing 'f' function
        result = await interpolate(
            target="f",
            expression="sin(pi*x[0])",
            ctx=ctx,
        )
        assert "error" not in result
        assert math.isfinite(result["l2_norm"])
        assert result["min_value"] >= -1.0 - 1e-10  # sin range check
        assert result["max_value"] <= 1.0 + 1e-10

    @pytest.mark.asyncio
    async def test_assemble_scalar_finite(self, session, ctx):
        """A5: assemble(target='scalar') NaN/Inf postcondition.

        Exercises: session_mgmt.py:185 (np.isfinite scalar check).
        """
        from dolfinx_mcp.tools.session_mgmt import assemble

        await _setup_poisson(session, ctx)
        result = await assemble(
            target="scalar",
            form="inner(u_h, u_h)*dx",
            ctx=ctx,
        )
        assert "error" not in result
        assert "value" in result
        assert math.isfinite(result["value"])
        assert result["value"] >= 0  # inner product is non-negative

    @pytest.mark.asyncio
    async def test_compute_functionals_finite(self, session, ctx):
        """A6: compute_functionals() per-functional finite check.

        Exercises: postprocess.py:373 (math.isfinite per functional).
        """
        from dolfinx_mcp.tools.postprocess import compute_functionals

        await _setup_poisson(session, ctx)
        result = await compute_functionals(
            expressions=["inner(u_h, u_h)*dx"],
            ctx=ctx,
        )
        assert "error" not in result
        assert result["num_functionals"] == 1
        assert math.isfinite(result["functionals"][0]["value"])

    @pytest.mark.asyncio
    async def test_full_workflow_debug_invariants_pass(self, session, ctx):
        """A7: Full workflow exercises all 19 debug invariant checks.

        Exercises every ``if __debug__: session.check_invariants()`` path:
        create_unit_square, create_function_space, set_material_properties,
        define_variational_form, apply_boundary_condition, solve,
        interpolate, and session state query. If any invariant check fails,
        InvariantError propagates and the test fails.
        """
        from dolfinx_mcp.tools.interpolation import interpolate
        from dolfinx_mcp.tools.session_mgmt import assemble, get_session_state

        # Full Poisson setup + solve (exercises invariants in mesh, space,
        # material, form, BC, and solver paths)
        result = await _setup_poisson(session, ctx)
        assert "error" not in result

        # Exercise interpolation path invariants
        result = await interpolate(
            target="f",
            expression="sin(pi*x[0])*cos(pi*x[1])",
            ctx=ctx,
        )
        assert "error" not in result

        # Exercise assembly path
        result = await assemble(
            target="scalar",
            form="inner(u_h, u_h)*dx",
            ctx=ctx,
        )
        assert "error" not in result

        # Verify session state consistency
        state = await get_session_state(ctx=ctx)
        assert state["active_mesh"] == "mesh"
        assert "V" in state["function_spaces"]
        assert "u_h" in state["solutions"]

        # Explicit invariant check -- raises InvariantError if broken
        session.check_invariants()


# ---------------------------------------------------------------------------
# Group B: Negative-Path Contract Violation Testing (2 tests)
# ---------------------------------------------------------------------------


class TestNegativePathContracts:
    """Verify postconditions detect and reject invalid data."""

    @pytest.mark.asyncio
    async def test_interpolate_nan_rejected(self, session, ctx):
        """B1: NaN interpolation fires finiteness postcondition.

        Exercises: interpolation.py:108 -- ``not np.isfinite(...)`` fires
        PostconditionError("Interpolation produced NaN/Inf values.").
        handle_tool_errors catches it, returns error dict.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.spaces import create_function_space
        from dolfinx_mcp.tools.problem import set_material_properties
        from dolfinx_mcp.tools.interpolation import interpolate

        await create_unit_square(name="mesh", nx=4, ny=4, ctx=ctx)
        await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)
        # Create a target function (expression-based so it registers in functions)
        await set_material_properties(name="g", value="0*x[0]", ctx=ctx)

        # Inject NaN via expression -- np is available in the eval namespace
        result = await interpolate(
            target="g",
            expression="np.nan + 0*x[0]",
            ctx=ctx,
        )
        assert result["error"] == "POSTCONDITION_VIOLATED"
        assert "NaN" in result["message"] or "Inf" in result["message"]

    @pytest.mark.asyncio
    async def test_compute_error_nan_exact_rejected(self, session, ctx):
        """B2: NaN exact solution fires finite postcondition in compute_error.

        Exercises: postprocess.py:134 -- ``math.isfinite(error_val)`` fires
        PostconditionError("Error norm must be finite").
        handle_tool_errors catches it, returns error dict.
        """
        from dolfinx_mcp.tools.postprocess import compute_error

        await _setup_poisson(session, ctx)

        # Exact solution evaluates to all NaN -> error function all NaN ->
        # assemble_scalar returns NaN -> math.isfinite fires
        result = await compute_error(
            exact="np.nan + 0*x[0]",
            norm_type="L2",
            ctx=ctx,
        )
        assert result["error"] == "POSTCONDITION_VIOLATED"
        assert "finite" in result["message"].lower()


# ---------------------------------------------------------------------------
# Group C: Session Operations with Real DOLFINx Objects (3 tests)
# ---------------------------------------------------------------------------


class TestSessionOperations:
    """Verify cleanup, cascade deletion, and reset with real DOLFINx objects."""

    @pytest.mark.asyncio
    async def test_cleanup_postconditions_real_objects(self, session, ctx):
        """C1: cleanup() 12 postcondition assertions with populated session.

        Exercises: session.py:462-486 -- all 12 registry-empty assertions.
        After full Poisson setup, cleanup() must clear every registry.
        If any registry is non-empty, PostconditionError fires.
        """
        await _setup_poisson(session, ctx)

        # Verify session is populated before cleanup
        assert len(session.meshes) > 0
        assert len(session.function_spaces) > 0
        assert len(session.functions) > 0
        assert len(session.solutions) > 0
        assert len(session.bcs) > 0
        assert len(session.forms) > 0

        # cleanup() runs all 12 postcondition checks internally
        session.cleanup()

        # Verify all registries empty (redundant with postconditions, but explicit)
        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0
        assert len(session.functions) == 0
        assert len(session.bcs) == 0
        assert len(session.forms) == 0
        assert len(session.solutions) == 0
        assert len(session.mesh_tags) == 0
        assert len(session.entity_maps) == 0
        assert len(session.ufl_symbols) == 0
        assert session.active_mesh is None

        # Invariant check on empty state (trivially consistent)
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_remove_mesh_cascade_real_objects(self, session, ctx):
        """C2: remove_mesh() cascade + no-dangling postconditions.

        Exercises: session.py:384-406 -- postcondition checks after cascade
        deletion of mesh and all dependents (spaces, functions, BCs).
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.spaces import create_function_space
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            set_material_properties,
        )

        await create_unit_square(name="mesh", nx=4, ny=4, ctx=ctx)
        await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)
        await set_material_properties(name="g", value="x[0]", ctx=ctx)
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        # Verify populated state
        assert "mesh" in session.meshes
        assert "V" in session.function_spaces
        assert "g" in session.functions
        assert len(session.bcs) == 1

        # Cascade delete -- exercises postcondition checks at session.py:384-406
        session.remove_mesh("mesh")

        # Verify cascade removed everything dependent on the mesh
        assert "mesh" not in session.meshes
        assert len(session.function_spaces) == 0
        assert len(session.functions) == 0
        assert len(session.bcs) == 0
        assert session.active_mesh is None

        # Invariant check passes on post-cascade state
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_reset_session_with_invariant_check(self, session, ctx):
        """C3: reset_session() tool exercises debug invariant after cleanup.

        Exercises: session_mgmt.py:46-47 -- ``if __debug__: session.check_invariants()``
        on a real populated-then-cleared session.
        """
        from dolfinx_mcp.tools.session_mgmt import reset_session

        await _setup_poisson(session, ctx)

        # Verify session is populated
        assert len(session.solutions) > 0

        # reset_session calls cleanup() then check_invariants() under __debug__
        result = await reset_session(ctx=ctx)
        assert result["status"] == "reset"

        # Verify session is empty after reset
        assert len(session.meshes) == 0
        assert len(session.solutions) == 0
        assert session.active_mesh is None
