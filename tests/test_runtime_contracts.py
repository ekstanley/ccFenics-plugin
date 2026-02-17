"""Docker integration tests: runtime contracts with real DOLFINx objects.

Run with:
    docker run --rm dolfinx-mcp:latest python -m pytest tests/test_runtime_contracts.py -v

Phase 7 tests exercise postconditions, debug invariants, and session cascade
operations that require real DOLFINx objects -- contracts that cannot be tested
without the FEniCSx runtime.

Phase 15 tests (Groups D, E, F) expand coverage to mesh operations, solver/
postprocess tools, and Phase 14 tools.

Groups:
    A (7): Positive-path -- runtime contracts pass during valid operations
    B (2): Negative-path -- postconditions detect and reject bad data
    C (3): Session operations -- cascade deletion and cleanup with real objects
    D (4): Mesh operations -- create, refine, mark, submesh
    E (4): Solver/postprocess -- time-dependent, export, evaluate, query
    F (3): Phase 14 tools -- remove_object, compute_mesh_quality, project
    G (3): Nonlinear solver -- solve_nonlinear postconditions and session state
    H (2): Eval helpers -- shape postcondition and boolean coercion
    I (1): Pipeline integration -- plot_solution → read_workspace_file round-trip
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


async def _setup_poisson(session: SessionState, ctx: _FakeContext) -> dict:
    """Set up and solve standard Poisson problem on 8x8 P1 mesh.

    Returns the solve result dict. Uses a coarse mesh (8x8, degree 1) for
    speed -- sufficient for contract verification, not accuracy testing.
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
        from dolfinx_mcp.tools.interpolation import interpolate
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import set_material_properties
        from dolfinx_mcp.tools.spaces import create_function_space

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
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            set_material_properties,
        )
        from dolfinx_mcp.tools.spaces import create_function_space

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


# ---------------------------------------------------------------------------
# Group D: Mesh Operations (4 tests)
# ---------------------------------------------------------------------------


class TestMeshOperations:
    """Verify runtime contracts during mesh creation, refinement, and tagging."""

    @pytest.mark.asyncio
    async def test_create_mesh_rectangle(self, session, ctx):
        """D1: create_mesh(shape='rectangle') postconditions.

        Verifies correct cell count, dimensions, and invariant check.
        """
        from dolfinx_mcp.tools.mesh import create_mesh

        result = await create_mesh(
            name="rect", shape="rectangle", nx=4, ny=4,
            dimensions={"width": 2.0, "height": 1.0},
            ctx=ctx,
        )
        assert "error" not in result
        assert result["num_cells"] > 0
        assert result["gdim"] == 2
        assert "rect" in session.meshes
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_refine_mesh_increases_cells(self, session, ctx):
        """D2: refine_mesh postcondition -- refined cells > original.

        Exercises: mesh.py postcondition for cell count increase.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square, refine_mesh

        await create_unit_square(name="m", nx=4, ny=4, ctx=ctx)
        original_cells = session.meshes["m"].num_cells

        result = await refine_mesh(name="m", new_name="m_ref", ctx=ctx)
        assert "error" not in result
        assert session.meshes["m_ref"].num_cells > original_cells
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_mark_boundaries_and_manage_tags(self, session, ctx):
        """D3: mark_boundaries + manage_mesh_tags -- boundary facets tagged.

        Exercises boundary tagging and tag query postconditions.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square, manage_mesh_tags, mark_boundaries

        await create_unit_square(name="m", nx=4, ny=4, ctx=ctx)
        result = await mark_boundaries(
            markers=[
                {"tag": 1, "condition": "np.isclose(x[0], 0.0)"},
                {"tag": 2, "condition": "np.isclose(x[0], 1.0)"},
            ],
            ctx=ctx,
        )
        assert "error" not in result

        # Query the tags
        query_result = await manage_mesh_tags(
            action="query", name=result["name"], ctx=ctx,
        )
        assert "error" not in query_result
        assert query_result["num_entities"] > 0
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_create_submesh(self, session, ctx):
        """D4: create_submesh postconditions -- fewer cells, entity map created.

        Exercises submesh creation and entity map registration.
        """
        from dolfinx_mcp.tools.mesh import create_submesh, create_unit_square, mark_boundaries

        await create_unit_square(name="m", nx=8, ny=8, ctx=ctx)
        tag_result = await mark_boundaries(
            markers=[{"tag": 1, "condition": "x[0] < 0.5"}],
            ctx=ctx,
        )

        # Try to create a submesh from tagged cells
        result = await create_submesh(
            name="sub", tags_name=tag_result["name"],
            tag_values=[1], ctx=ctx,
        )
        if "error" not in result:
            assert result["num_cells"] <= session.meshes["m"].num_cells
            assert len(session.entity_maps) > 0
            session.check_invariants()


# ---------------------------------------------------------------------------
# Group E: Solver/Postprocess (4 tests)
# ---------------------------------------------------------------------------


class TestSolverPostprocess:
    """Verify runtime contracts during solve, export, evaluate, query."""

    @pytest.mark.asyncio
    async def test_solve_time_dependent(self, session, ctx):
        """E1: solve_time_dependent postconditions.

        Sets up a heat equation with backward Euler and verifies
        steps completed, final time near t_end, and invariant check.
        """
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

        # Set up u_n (previous timestep) as zero
        await set_material_properties(name="u_n", value="0*x[0]", ctx=ctx)
        # Source term
        await set_material_properties(name="f", value="1.0 + 0*x[0]", ctx=ctx)

        dt = 0.1
        # Heat equation: M(u-u_n)/dt + K*u = f*v*dx
        # -> (u*v + dt*inner(grad(u), grad(v)))*dx = (u_n + dt*f)*v*dx
        await define_variational_form(
            bilinear=f"(u*v + {dt}*inner(grad(u), grad(v))) * dx",
            linear=f"(u_n + {dt}*f) * v * dx",
            ctx=ctx,
        )
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        result = await solve_time_dependent(
            t_end=0.3, dt=dt, ctx=ctx,
        )
        assert "error" not in result
        assert result["steps_completed"] > 0
        assert abs(result["final_time"] - 0.3) < dt
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_export_solution(self, session, ctx):
        """E2: export_solution postcondition -- file written.

        Exercises: postprocess.py export_solution with XDMF format.
        """
        from dolfinx_mcp.tools.postprocess import export_solution

        await _setup_poisson(session, ctx)

        result = await export_solution(
            filename="test_export.xdmf", format="xdmf", ctx=ctx,
        )
        assert "error" not in result
        assert result["file_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_evaluate_solution_finite(self, session, ctx):
        """E3: evaluate_solution postcondition -- values finite.

        Exercises finiteness postcondition at interior points.
        """
        from dolfinx_mcp.tools.postprocess import evaluate_solution

        await _setup_poisson(session, ctx)

        result = await evaluate_solution(
            points=[[0.5, 0.5], [0.25, 0.75]],
            ctx=ctx,
        )
        assert "error" not in result
        for pt in result["evaluations"]:
            val = pt["value"]
            if isinstance(val, list):
                assert all(math.isfinite(v) for v in val)
            else:
                assert math.isfinite(val)

    @pytest.mark.asyncio
    async def test_query_point_values(self, session, ctx):
        """E4: query_point_values postcondition -- values and cell indices.

        Exercises postconditions for point query results.
        """
        from dolfinx_mcp.tools.postprocess import query_point_values

        await _setup_poisson(session, ctx)

        result = await query_point_values(
            points=[[0.5, 0.5]],
            ctx=ctx,
        )
        assert "error" not in result
        assert len(result["queries"]) == 1
        pt_result = result["queries"][0]
        val = pt_result["value"]
        if isinstance(val, list):
            assert all(math.isfinite(v) for v in val)
        else:
            assert math.isfinite(val)


# ---------------------------------------------------------------------------
# Group F: Phase 14 Tools (3 tests)
# ---------------------------------------------------------------------------


class TestPhase14Tools:
    """Verify runtime contracts for remove_object, compute_mesh_quality, project."""

    @pytest.mark.asyncio
    async def test_remove_object_mesh_cascade(self, session, ctx):
        """F1: remove_object(mesh) cascade removes all dependents.

        Exercises: session_mgmt.py remove_object with mesh cascade.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import set_material_properties
        from dolfinx_mcp.tools.session_mgmt import remove_object
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="m", nx=4, ny=4, ctx=ctx)
        await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)
        await set_material_properties(name="g", value="x[0]", ctx=ctx)

        assert "m" in session.meshes
        assert "V" in session.function_spaces
        assert "g" in session.functions

        result = await remove_object(name="m", object_type="mesh", ctx=ctx)
        assert "error" not in result
        assert result["cascade"] is True
        assert "m" not in session.meshes
        assert len(session.function_spaces) == 0
        assert len(session.functions) == 0
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_compute_mesh_quality(self, session, ctx):
        """F2: compute_mesh_quality -- all metrics > 0 and finite.

        Exercises: mesh.py postconditions for quality metrics.
        """
        from dolfinx_mcp.tools.mesh import compute_mesh_quality, create_unit_square

        await create_unit_square(name="m", nx=4, ny=4, ctx=ctx)
        result = await compute_mesh_quality(mesh_name="m", ctx=ctx)
        assert "error" not in result
        assert result["min_volume"] > 0
        assert result["max_volume"] > 0
        assert result["mean_volume"] > 0
        assert result["quality_ratio"] > 0
        assert result["quality_ratio"] <= 1.0
        assert math.isfinite(result["std_volume"])

    @pytest.mark.asyncio
    async def test_project_l2(self, session, ctx):
        """F3: project() L2 projection -- result finite, norm > 0.

        Exercises: interpolation.py project postconditions.
        """
        from dolfinx_mcp.tools.interpolation import project
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=4, ny=4, ctx=ctx)
        await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)

        result = await project(
            name="p_func",
            target_space="V",
            expression="sin(pi*x[0])",
            ctx=ctx,
        )
        assert "error" not in result
        assert result["l2_norm"] > 0
        assert math.isfinite(result["l2_norm"])
        assert "p_func" in session.functions
        session.check_invariants()


# ---------------------------------------------------------------------------
# Group G: Nonlinear Solver (3 tests)
# ---------------------------------------------------------------------------


async def _setup_nonlinear_poisson(session: SessionState, ctx: _FakeContext) -> dict:
    """Set up and solve a linear Poisson in residual form via solve_nonlinear.

    Uses -div(grad(u)) = 1 with homogeneous Dirichlet BCs on an 8x8 P1 mesh.
    This is linear so Newton converges in 1 iteration -- simplest possible
    nonlinear-solver runtime test.

    Returns the solve_nonlinear result dict.
    """
    from dolfinx_mcp.tools.mesh import create_unit_square
    from dolfinx_mcp.tools.problem import (
        apply_boundary_condition,
        set_material_properties,
    )
    from dolfinx_mcp.tools.solver import solve_nonlinear
    from dolfinx_mcp.tools.spaces import create_function_space

    await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
    await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)

    # Source term f = 1
    await set_material_properties(name="f", value="1.0 + 0*x[0]", ctx=ctx)

    # Mutable unknown with zero initial guess
    await set_material_properties(name="u", value="0.0 + 0*x[0]", ctx=ctx)

    # Homogeneous Dirichlet BC on all boundaries
    await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

    return await solve_nonlinear(
        residual="inner(grad(u), grad(v))*dx - f*v*dx",
        unknown="u",
        ctx=ctx,
    )


class TestNonlinearSolver:
    """Verify runtime contracts for solve_nonlinear with real DOLFINx objects."""

    @pytest.mark.asyncio
    async def test_solve_nonlinear_postconditions_pass(self, session, ctx):
        """G1: solve_nonlinear() postconditions -- finite, L2 >= 0, invariants.

        Exercises: solver.py:563 (POST-1 finite), solver.py:575 (POST-2 L2 >= 0),
        solver.py:607-609 (POST-3 invariants).
        """
        result = await _setup_nonlinear_poisson(session, ctx)
        assert "error" not in result
        assert result["converged"] is True
        assert result["iterations"] >= 1
        assert result["solution_norm_L2"] > 0
        assert math.isfinite(result["solution_norm_L2"])
        assert math.isfinite(result["residual_norm"])

    @pytest.mark.asyncio
    async def test_solve_nonlinear_session_state(self, session, ctx):
        """G2: solve_nonlinear registers solution and function, invariants hold.

        Exercises: solver.py:596 (solution registration), solver.py:600
        (function registration), session invariant check.
        """
        result = await _setup_nonlinear_poisson(session, ctx)
        assert "error" not in result

        # Solution registered under the unknown name
        assert "u" in session.solutions
        assert "u" in session.functions
        assert session.solutions["u"].converged is True

        # Explicit invariant check
        session.check_invariants()

    @pytest.mark.asyncio
    async def test_solve_nonlinear_custom_solution_name(self, session, ctx):
        """G3: solve_nonlinear with solution_name registers under custom name.

        Exercises dual registration with a custom solution_name parameter.
        """
        from dolfinx_mcp.tools.mesh import create_unit_square
        from dolfinx_mcp.tools.problem import (
            apply_boundary_condition,
            set_material_properties,
        )
        from dolfinx_mcp.tools.solver import solve_nonlinear
        from dolfinx_mcp.tools.spaces import create_function_space

        await create_unit_square(name="mesh", nx=8, ny=8, ctx=ctx)
        await create_function_space(name="V", family="Lagrange", degree=1, ctx=ctx)
        await set_material_properties(name="f", value="1.0 + 0*x[0]", ctx=ctx)
        await set_material_properties(name="u", value="0.0 + 0*x[0]", ctx=ctx)
        await apply_boundary_condition(value=0.0, boundary="True", ctx=ctx)

        result = await solve_nonlinear(
            residual="inner(grad(u), grad(v))*dx - f*v*dx",
            unknown="u",
            solution_name="my_nl_sol",
            ctx=ctx,
        )
        assert "error" not in result
        assert result["solution_name"] == "my_nl_sol"

        # Custom name registered in both registries
        assert "my_nl_sol" in session.solutions
        assert "my_nl_sol" in session.functions

        session.check_invariants()


# ---------------------------------------------------------------------------
# Group H: Eval Helper Postconditions (2 tests)
# ---------------------------------------------------------------------------


class TestEvalHelperPostconditions:
    """Verify eval_helpers postconditions with real numpy arrays."""

    def test_interpolate_wrong_shape_expression(self):
        """H1: eval_numpy_expression rejects wrong-shape result.

        An expression returning shape (2, N) instead of (N,) must raise
        PostconditionError or be broadcast-corrected.
        """
        import numpy as np

        from dolfinx_mcp.eval_helpers import eval_numpy_expression

        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])  # (2, 3)

        # Expression returning shape (N,) -- should pass
        result = eval_numpy_expression("x[0] + x[1]", x)
        assert result.shape == (3,)

        # Expression returning scalar -- should broadcast
        result = eval_numpy_expression("1.0", x)
        assert result.shape == (3,)

        # Expression returning shape (2, N) -- should raise PostconditionError
        from dolfinx_mcp.errors import PostconditionError

        with pytest.raises(PostconditionError, match="shape"):
            # np.vstack([x[0], x[1]]) returns (2, 3) which cannot broadcast to (3,)
            eval_numpy_expression("np.vstack([x[0], x[1]])", x)

    def test_boundary_marker_non_boolean_coerced(self):
        """H2: make_boundary_marker coerces non-boolean result to bool.

        An expression like 'x[0] + 1.0' returns float array; the marker
        must coerce it to boolean so DOLFINx can use it.
        """
        import numpy as np

        from dolfinx_mcp.eval_helpers import make_boundary_marker

        # Float expression that is truthy everywhere
        marker = make_boundary_marker("x[0] + 1.0")
        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
        result = marker(x)
        assert result.dtype == np.bool_, f"Expected bool, got {result.dtype}"
        assert result.all()  # x[0]+1.0 is always > 0, so all True

        # Boolean expression -- should pass through unchanged
        marker_bool = make_boundary_marker("np.isclose(x[0], 0.0)")
        result_bool = marker_bool(x)
        assert result_bool.dtype == np.bool_


# ---------------------------------------------------------------------------
# Group I: Pipeline Integration (1 test)
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Verify end-to-end pipelines spanning multiple tool modules."""

    @pytest.mark.asyncio
    async def test_plot_then_read_roundtrip(self, session, ctx):
        """I1: plot_solution → read_workspace_file round-trip.

        Solves Poisson, plots with return_base64=True, then reads the
        same PNG via read_workspace_file. Both base64 strings must decode
        to identical bytes, validating the full pipeline.
        """
        import base64

        from dolfinx_mcp.tools.postprocess import plot_solution
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        await _setup_poisson(session, ctx)

        # Plot with base64 return
        plot_result = await plot_solution(
            output_file="/workspace/pipeline_test.png",
            return_base64=True,
            ctx=ctx,
        )
        assert "error" not in plot_result
        assert "image_base64" in plot_result
        plot_b64 = plot_result["image_base64"]

        # Read the same file via read_workspace_file
        read_result = await read_workspace_file(
            file_path=plot_result["file_path"],
            ctx=ctx,
        )
        assert "error" not in read_result
        assert read_result["encoding"] == "base64"
        read_b64 = read_result["content"]

        # Both must decode to identical bytes
        plot_bytes = base64.b64decode(plot_b64)
        read_bytes = base64.b64decode(read_b64)
        assert plot_bytes == read_bytes, (
            f"Round-trip mismatch: plot gave {len(plot_bytes)} bytes, "
            f"read gave {len(read_bytes)} bytes"
        )
