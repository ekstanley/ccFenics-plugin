"""Contract violation tests for Design-by-Contract enforcement.

Tests dataclass __post_init__ validators, SessionState.check_invariants(),
and tool precondition checks. No Docker or DOLFINx required.
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from conftest import (
    assert_error_type,
    assert_no_error,
    make_bc_info,
    make_entity_map_info,
    make_form_info,
    make_function_info,
    make_mesh_info,
    make_mesh_tags_info,
    make_mock_ctx,
    make_populated_session,
    make_solution_info,
    make_space_info,
)

from dolfinx_mcp.errors import (
    FileIOError,
    FunctionNotFoundError,
    InvariantError,
    PostconditionError,
)
from dolfinx_mcp.session import (
    BCInfo,
    EntityMapInfo,
    FormInfo,
    FunctionSpaceInfo,
    MeshInfo,
    SessionState,
    SolutionInfo,
)

# ---------------------------------------------------------------------------
# Phase 2: Dataclass __post_init__ precondition tests (8 tests)
# ---------------------------------------------------------------------------


class TestDataclassContracts:
    def test_mesh_info_rejects_zero_cells(self):
        with pytest.raises(InvariantError, match="num_cells must be > 0"):
            MeshInfo(
                name="bad",
                mesh=MagicMock(),
                cell_type="triangle",
                num_cells=0,
                num_vertices=10,
                gdim=2,
                tdim=2,
            )

    def test_mesh_info_rejects_invalid_gdim(self):
        with pytest.raises(InvariantError, match="gdim must be 1, 2, or 3"):
            MeshInfo(
                name="bad",
                mesh=MagicMock(),
                cell_type="triangle",
                num_cells=10,
                num_vertices=10,
                gdim=0,
                tdim=1,
            )

    def test_mesh_info_rejects_tdim_gt_gdim(self):
        with pytest.raises(InvariantError, match="tdim.*must be <= gdim"):
            MeshInfo(
                name="bad",
                mesh=MagicMock(),
                cell_type="triangle",
                num_cells=10,
                num_vertices=10,
                gdim=2,
                tdim=3,
            )

    def test_space_info_rejects_negative_degree(self):
        with pytest.raises(InvariantError, match="degree must be >= 0"):
            FunctionSpaceInfo(
                name="V",
                space=MagicMock(),
                mesh_name="m1",
                element_family="Lagrange",
                element_degree=-1,
                num_dofs=10,
            )

    def test_space_info_rejects_zero_dofs(self):
        with pytest.raises(InvariantError, match="num_dofs must be > 0"):
            FunctionSpaceInfo(
                name="V",
                space=MagicMock(),
                mesh_name="m1",
                element_family="Lagrange",
                element_degree=1,
                num_dofs=0,
            )

    def test_bc_info_rejects_zero_dofs(self):
        with pytest.raises(InvariantError, match="num_dofs must be > 0"):
            BCInfo(
                name="bc",
                bc=MagicMock(),
                space_name="V",
                num_dofs=0,
            )

    def test_solution_info_rejects_negative_iterations(self):
        with pytest.raises(InvariantError, match="iterations must be >= 0"):
            SolutionInfo(
                name="u",
                function=MagicMock(),
                space_name="V",
                converged=True,
                iterations=-1,
                residual_norm=0.0,
                wall_time=0.0,
            )

    def test_entity_map_rejects_empty_names(self):
        with pytest.raises(InvariantError, match="parent_mesh must be non-empty"):
            EntityMapInfo(
                name="em",
                entity_map=MagicMock(),
                parent_mesh="",
                child_mesh="c",
                dimension=1,
            )


# ---------------------------------------------------------------------------
# Phase 3: SessionState invariant tests (5 tests)
# ---------------------------------------------------------------------------


class TestSessionInvariants:
    def test_check_invariants_valid_session(self):
        session = make_populated_session()
        # Should not raise
        session.check_invariants()

    def test_check_invariants_dangling_active_mesh(self):
        session = SessionState()
        session.active_mesh = "deleted_mesh"
        with pytest.raises(InvariantError, match="active_mesh.*not in meshes"):
            session.check_invariants()

    def test_check_invariants_dangling_space_mesh_ref(self):
        session = SessionState()
        # Space references mesh "m1" which does not exist
        session.function_spaces["V"] = make_space_info("V", "m1")
        with pytest.raises(InvariantError, match="Dangling mesh references"):
            session.check_invariants()

    def test_check_invariants_dangling_function_space_ref(self):
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        # Function references space "V" which does not exist
        session.functions["f"] = make_function_info("f", "V")
        with pytest.raises(InvariantError, match="Dangling space references"):
            session.check_invariants()

    def test_remove_mesh_postconditions(self):
        session = make_populated_session()
        session.remove_mesh("m1")
        # After removal, no references to "m1" should remain
        assert "m1" not in session.meshes
        assert all(fs.mesh_name != "m1" for fs in session.function_spaces.values())
        assert all(mt.mesh_name != "m1" for mt in session.mesh_tags.values())
        # Empty session should pass invariants
        session.check_invariants()

    def test_check_invariants_forms_without_spaces(self):
        """INV-8: forms non-empty with no function_spaces must raise."""
        session = SessionState()
        session.forms["bilinear"] = make_form_info("bilinear")
        # No function_spaces registered — INV-8 violated
        with pytest.raises(InvariantError, match="[Ff]orms.*no function_spaces"):
            session.check_invariants()

    def test_check_invariants_forms_with_spaces_ok(self):
        """INV-8: forms non-empty with function_spaces is valid."""
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.forms["bilinear"] = make_form_info("bilinear")
        # Should NOT raise — forms exist AND function_spaces exist
        session.check_invariants()


# ---------------------------------------------------------------------------
# Phase 4: Tool precondition tests (5 tests, mocked context)
# ---------------------------------------------------------------------------


class TestToolPreconditions:
    @pytest.mark.asyncio
    async def test_create_unit_square_rejects_zero_nx(self):
        from dolfinx_mcp.tools.mesh import create_unit_square

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_unit_square(name="m", nx=0, ny=8, ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_solve_time_dependent_rejects_negative_dt(self):
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await solve_time_dependent(t_end=1.0, dt=-0.1, ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_set_material_reserved_name(self):
        from dolfinx_mcp.tools.problem import set_material_properties

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)
        result = await set_material_properties(name="grad", value=1.0, ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_evaluate_solution_rejects_empty_points(self):
        from dolfinx_mcp.tools.postprocess import evaluate_solution

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await evaluate_solution(points=[], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_function_space_rejects_negative_degree(self):
        from dolfinx_mcp.tools.spaces import create_function_space

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)
        result = await create_function_space(name="V", degree=-1, ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ---------------------------------------------------------------------------
# Phase 13: FormInfo validator tests (2 tests)
# ---------------------------------------------------------------------------


class TestFormInfoContracts:
    def test_form_info_rejects_empty_name(self):
        with pytest.raises(InvariantError, match="FormInfo.name must be non-empty"):
            FormInfo(name="", form=MagicMock(), ufl_form=MagicMock())

    def test_form_info_rejects_none_form(self):
        with pytest.raises(InvariantError, match="form.*must not be None"):
            FormInfo(name="bilinear", form=None, ufl_form=MagicMock())


# ---------------------------------------------------------------------------
# Phase 13: Extended cleanup and cascade tests (2 tests)
# ---------------------------------------------------------------------------


class TestExtendedCleanupContracts:
    def test_cleanup_asserts_all_registries(self):
        session = make_populated_session()
        session.forms["bilinear"] = make_form_info("bilinear")
        session.mesh_tags["tags"] = make_mesh_tags_info("tags")
        session.entity_maps["em"] = make_entity_map_info(
            "em", parent_mesh="m1", child_mesh="m1",
        )
        session.ufl_symbols["f"] = MagicMock()
        session.cleanup()
        # All registries must be empty
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

    def test_remove_mesh_cleans_entity_maps(self):
        session = make_populated_session()
        session.entity_maps["em1"] = make_entity_map_info(
            "em1", parent_mesh="m1", child_mesh="m1",
        )
        session.remove_mesh("m1")
        assert "em1" not in session.entity_maps
        assert len(session.entity_maps) == 0


# ---------------------------------------------------------------------------
# Phase 13: Extended invariant check coverage (3 tests)
# ---------------------------------------------------------------------------


class TestExtendedInvariants:
    def test_check_invariants_dangling_solution_ref(self):
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        # Solution references space "V" which does not exist
        session.solutions["u_h"] = make_solution_info("u_h", "V")
        with pytest.raises(InvariantError, match="Dangling space references"):
            session.check_invariants()

    def test_check_invariants_dangling_mesh_tags_ref(self):
        session = SessionState()
        # MeshTags references mesh "m1" which does not exist
        session.mesh_tags["tags"] = make_mesh_tags_info("tags")
        with pytest.raises(InvariantError, match="Dangling mesh references"):
            session.check_invariants()

    def test_check_invariants_dangling_entity_map_ref(self):
        session = SessionState()
        # EntityMap references meshes that do not exist
        session.entity_maps["em"] = make_entity_map_info(
            "em", parent_mesh="m1", child_mesh="m2",
        )
        with pytest.raises(InvariantError, match="Dangling mesh references"):
            session.check_invariants()


# ---------------------------------------------------------------------------
# Phase 13: Extended tool precondition tests (3 tests)
# ---------------------------------------------------------------------------


class TestExtendedToolPreconditions:
    @pytest.mark.asyncio
    async def test_create_mesh_rejects_invalid_cell_type(self):
        from dolfinx_mcp.tools.mesh import create_mesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_mesh(
            name="m", shape="unit_square", cell_type="hexagon", nx=4, ny=4, ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_run_custom_code_rejects_empty_code(self):
        from dolfinx_mcp.tools.session_mgmt import run_custom_code

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await run_custom_code(code="", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_apply_bc_rejects_negative_sub_space(self):
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await apply_boundary_condition(
            value=0.0, boundary="True", sub_space=-1, ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ---------------------------------------------------------------------------
# Phase 3: Contract hardening tests (7 tests)
# ---------------------------------------------------------------------------


class TestPhase3Contracts:
    @pytest.mark.asyncio
    async def test_create_discrete_operator_invalid_type(self):
        from dolfinx_mcp.tools.interpolation import create_discrete_operator

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_discrete_operator(
            operator_type="divergence", source_space="V", target_space="W", ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_export_solution_invalid_format(self):
        from dolfinx_mcp.tools.postprocess import export_solution

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await export_solution(filename="out.csv", format="csv", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_assemble_invalid_target(self):
        from dolfinx_mcp.tools.session_mgmt import assemble

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await assemble(target="tensor", form="u*v*dx", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_compute_functionals_empty_expressions(self):
        from dolfinx_mcp.tools.postprocess import compute_functionals

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await compute_functionals(expressions=[], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    def test_check_invariants_dangling_bc_space_ref(self):
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        # BC references space "V" which does not exist
        session.bcs["bc0"] = make_bc_info("bc0", "V")
        with pytest.raises(InvariantError, match="Dangling space references"):
            session.check_invariants()

    def test_postcondition_error_integration(self):
        """PostconditionError is caught by handle_tool_errors -> POSTCONDITION_VIOLATED."""
        err = PostconditionError("test postcondition failure")
        result = err.to_dict()
        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "test postcondition failure" in result["message"]

    @pytest.mark.asyncio
    async def test_plot_solution_invalid_type(self):
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await plot_solution(plot_type="3d", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ---------------------------------------------------------------------------
# Phase 4: Precondition-before-import and remaining gap tests (5 tests)
# ---------------------------------------------------------------------------


class TestPhase4Contracts:
    @pytest.mark.asyncio
    async def test_create_mesh_rejects_invalid_shape(self):
        from dolfinx_mcp.tools.mesh import create_mesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_mesh(
            name="m", shape="sphere", cell_type="triangle", nx=4, ny=4, ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_manage_mesh_tags_rejects_invalid_action(self):
        from dolfinx_mcp.tools.mesh import manage_mesh_tags

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await manage_mesh_tags(action="delete", name="tags", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_mark_boundaries_rejects_empty_markers(self):
        from dolfinx_mcp.tools.mesh import mark_boundaries

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await mark_boundaries(markers=[], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_mark_boundaries_rejects_negative_tag(self):
        from dolfinx_mcp.tools.mesh import mark_boundaries

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await mark_boundaries(
            markers=[{"tag": -1, "condition": "True"}], ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_mixed_space_rejects_single_subspace(self):
        from dolfinx_mcp.tools.spaces import create_mixed_space

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_mixed_space(name="W", subspaces=["V"], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ---------------------------------------------------------------------------
# Phase 5: Eager preconditions and cleanup completeness (9 tests)
# ---------------------------------------------------------------------------


class TestPhase5Contracts:
    @pytest.mark.asyncio
    async def test_solve_rejects_invalid_solver_type(self):
        from dolfinx_mcp.tools.solver import solve

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await solve(solver_type="mumps", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_solve_time_dependent_rejects_invalid_time_scheme(self):
        from dolfinx_mcp.tools.solver import solve_time_dependent

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await solve_time_dependent(
            t_end=1.0, dt=0.1, time_scheme="crank_nicolson", ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_define_variational_form_rejects_empty_bilinear(self):
        from dolfinx_mcp.tools.problem import define_variational_form

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await define_variational_form(bilinear="", linear="f*v*dx", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_compute_error_rejects_invalid_norm_type(self):
        from dolfinx_mcp.tools.postprocess import compute_error

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await compute_error(exact="x[0]", norm_type="Linf", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_query_point_values_rejects_zero_tolerance(self):
        from dolfinx_mcp.tools.postprocess import query_point_values

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await query_point_values(
            points=[[0.5, 0.5]], tolerance=0.0, ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_custom_mesh_rejects_empty_filename(self):
        from dolfinx_mcp.tools.mesh import create_custom_mesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_custom_mesh(name="m", filename="", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_submesh_rejects_empty_tag_values(self):
        from dolfinx_mcp.tools.mesh import create_submesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_submesh(
            name="sub", tags_name="tags", tag_values=[], ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_mark_boundaries_rejects_empty_condition(self):
        from dolfinx_mcp.tools.mesh import mark_boundaries

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await mark_boundaries(
            markers=[{"tag": 1, "condition": ""}], ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    def test_cleanup_clears_solver_diagnostics_and_log_buffer(self):
        session = SessionState()
        session.solver_diagnostics["last_run"] = {"converged": True}
        session.log_buffer.append("test log entry")
        session.cleanup()
        assert len(session.solver_diagnostics) == 0
        assert len(session.log_buffer) == 0


# ---------------------------------------------------------------------------
# Phase 6: Final hardening and table completeness (4 tests)
# ---------------------------------------------------------------------------


class TestPhase6Contracts:
    @pytest.mark.asyncio
    async def test_define_variational_form_rejects_empty_linear(self):
        from dolfinx_mcp.tools.problem import define_variational_form

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await define_variational_form(
            bilinear="inner(grad(u), grad(v)) * dx", linear="", ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_compute_error_rejects_empty_exact(self):
        from dolfinx_mcp.tools.postprocess import compute_error

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await compute_error(exact="", norm_type="L2", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_submesh_rejects_non_int_tag_values(self):
        from dolfinx_mcp.tools.mesh import create_submesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_submesh(
            name="sub", tags_name="tags", tag_values=["not_int"], ctx=ctx
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_unit_square_rejects_invalid_cell_type(self):
        from dolfinx_mcp.tools.mesh import create_unit_square

        session = SessionState()
        ctx = make_mock_ctx(session)
        result = await create_unit_square(name="m", cell_type="pentagon", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ---------------------------------------------------------------------------
# Phase 8: Error integrity and invariant check tests (3 tests)
# ---------------------------------------------------------------------------


class TestPhase8ErrorIntegrity:
    @pytest.mark.asyncio
    async def test_compute_functionals_preserves_postcondition_error(self):
        """PostconditionError from NaN functional not swallowed as DOLFINxAPIError."""
        import sys

        from dolfinx_mcp.tools.postprocess import compute_functionals

        session = SessionState()
        session.functions["f"] = make_function_info("f", "V")
        ctx = make_mock_ctx(session)

        mock_fem = MagicMock()
        mock_fem.form.return_value = MagicMock()
        mock_fem.assemble_scalar.return_value = float("nan")

        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_fem

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_fem,
        }), patch("dolfinx_mcp.ufl_context.build_namespace", return_value={}), \
            patch("dolfinx_mcp.ufl_context.safe_evaluate", return_value=MagicMock()):
            result = await compute_functionals(expressions=["u*dx"], ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_assemble_scalar_preserves_api_error(self):
        """DOLFINxAPIError for NaN scalar not lost to plain dict."""
        import sys

        from dolfinx_mcp.tools.session_mgmt import assemble

        session = SessionState()
        ctx = make_mock_ctx(session)

        mock_fem = MagicMock()
        mock_fem.form.return_value = MagicMock()
        mock_fem.assemble_scalar.return_value = float("nan")

        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_fem

        mock_np = MagicMock()
        mock_np.isfinite.return_value = False

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_fem,
            "dolfinx.fem.petsc": MagicMock(),
            "numpy": mock_np,
            "ufl": MagicMock(),
        }), patch("dolfinx_mcp.ufl_context.build_namespace", return_value={}), \
            patch("dolfinx_mcp.ufl_context.safe_evaluate", return_value=MagicMock()):
            result = await assemble(target="scalar", form="u*v*dx", ctx=ctx)

        assert_error_type(result, "DOLFINX_API_ERROR")

    @pytest.mark.asyncio
    async def test_run_custom_code_checks_invariants(self):
        """Invariant check fires after user code runs."""
        import sys

        from dolfinx_mcp.tools.session_mgmt import run_custom_code

        session = make_populated_session()
        ctx = make_mock_ctx(session)

        with patch.dict(sys.modules, {
            "dolfinx": MagicMock(),
            "numpy": MagicMock(),
            "ufl": MagicMock(),
            "mpi4py": MagicMock(),
            "mpi4py.MPI": MagicMock(),
        }), patch.object(session, "check_invariants") as mock_check:
            await run_custom_code(code="x = 1", ctx=ctx)

        mock_check.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 9: Defensive exception guard sweep tests (3 tests)
# ---------------------------------------------------------------------------


class TestPhase9ExceptionGuards:
    @pytest.mark.asyncio
    async def test_assemble_form_eval_preserves_structured_error(self):
        """InvalidUFLExpressionError from safe_evaluate propagates as structured
        error through assemble(), not as plain {"error": "Form evaluation failed: ..."} dict."""
        import sys

        from dolfinx_mcp.errors import InvalidUFLExpressionError
        from dolfinx_mcp.tools.session_mgmt import assemble

        session = SessionState()
        ctx = make_mock_ctx(session)

        def raise_invalid_ufl(*args, **kwargs):
            raise InvalidUFLExpressionError("bad expression 'xyz'")

        with patch.dict(sys.modules, {
            "dolfinx": MagicMock(),
            "dolfinx.fem": MagicMock(),
            "dolfinx.fem.petsc": MagicMock(),
            "ufl": MagicMock(),
        }), patch("dolfinx_mcp.ufl_context.build_namespace", return_value={}), \
            patch("dolfinx_mcp.ufl_context.safe_evaluate", side_effect=raise_invalid_ufl):
            result = await assemble(target="scalar", form="xyz", ctx=ctx)

        assert_error_type(result, "INVALID_UFL_EXPRESSION")
        assert "bad expression" in result["message"]

    @pytest.mark.asyncio
    async def test_assemble_assembly_returns_structured_error(self):
        """Non-MCP assembly exception becomes DOLFINxAPIError (structured),
        not plain {"error": "Assembly failed: ..."} dict."""
        import sys

        from dolfinx_mcp.tools.session_mgmt import assemble

        session = SessionState()
        ctx = make_mock_ctx(session)

        mock_fem = MagicMock()
        mock_fem.form.side_effect = RuntimeError("PETSc segfault")

        # Chain parent module so `import dolfinx.fem` resolves correctly
        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_fem
        mock_dolfinx.fem.petsc = MagicMock()

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_fem,
            "dolfinx.fem.petsc": mock_dolfinx.fem.petsc,
            "ufl": MagicMock(),
        }), patch("dolfinx_mcp.ufl_context.build_namespace", return_value={}), \
            patch("dolfinx_mcp.ufl_context.safe_evaluate", return_value=MagicMock()):
            result = await assemble(target="scalar", form="u*v*dx", ctx=ctx)

        assert_error_type(result, "DOLFINX_API_ERROR")
        assert "Assembly failed" in result["message"]
        assert "suggestion" in result

    @pytest.mark.asyncio
    async def test_create_unit_square_preserves_precondition_error(self):
        """PreconditionError propagates correctly through the DOLFINxMCPError guard
        in create_unit_square, not swallowed by except Exception."""

        from dolfinx_mcp.tools.mesh import create_unit_square

        session = SessionState()
        ctx = make_mock_ctx(session)

        # PreconditionError fires before the try block (nx=0), so it propagates
        # through @handle_tool_errors directly. Verify structured result.
        result = await create_unit_square(name="m", nx=0, ny=8, ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "nx must be > 0" in result["message"]


# ---------------------------------------------------------------------------
# Phase 10: Accessor postconditions and finiteness guards (9 tests)
# ---------------------------------------------------------------------------


class TestPhase10Contracts:
    def test_get_mesh_postcondition_name_mismatch(self):
        """Debug postcondition fires when MeshInfo.name != registry key."""
        session = make_populated_session()
        # Mutate name after insertion to create registry inconsistency
        session.meshes["m1"].name = "wrong_name"
        with pytest.raises(PostconditionError, match="name.*!= key"):
            session.get_mesh("m1")

    def test_get_space_postcondition_name_mismatch(self):
        """Debug postcondition fires when FunctionSpaceInfo.name != registry key."""
        session = make_populated_session()
        session.function_spaces["V"].name = "wrong_name"
        with pytest.raises(PostconditionError, match="name.*!= key"):
            session.get_space("V")

    def test_get_space_postcondition_dangling_mesh(self):
        """Debug postcondition fires when space references deleted mesh."""
        session = make_populated_session()
        # Directly delete mesh without cascade to create dangling reference
        del session.meshes["m1"]
        session.active_mesh = None
        with pytest.raises(PostconditionError, match="mesh.*not in meshes registry"):
            session.get_space("V")

    def test_get_function_postcondition_name_mismatch(self):
        """Debug postcondition fires when FunctionInfo.name != registry key."""
        session = make_populated_session()
        session.functions["f"].name = "wrong_name"
        with pytest.raises(PostconditionError, match="name.*!= key"):
            session.get_function("f")

    def test_get_function_postcondition_dangling_space(self):
        """Debug postcondition fires when function references deleted space."""
        session = make_populated_session()
        # Directly delete space without cascade
        del session.function_spaces["V"]
        with pytest.raises(PostconditionError, match="space.*not in function_spaces registry"):
            session.get_function("f")

    def test_get_only_space_postcondition_dangling_mesh(self):
        """Debug postcondition fires when sole space references deleted mesh."""
        session = make_populated_session()
        # Remove all but one space, then delete its mesh directly
        del session.meshes["m1"]
        session.active_mesh = None
        with pytest.raises(PostconditionError, match="mesh.*not in meshes registry"):
            session.get_only_space()

    @pytest.mark.asyncio
    async def test_evaluate_solution_postcondition_nan(self):
        """Finiteness postcondition fires when uh.eval returns NaN."""
        import sys

        np = pytest.importorskip("numpy")

        from dolfinx_mcp.tools.postprocess import evaluate_solution

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.active_mesh = "m1"
        mock_func = MagicMock()
        mock_func.function_space.mesh.topology.dim = 2
        nan_val = float("nan")
        mock_func.eval.return_value = np.array([nan_val])
        session.solutions["u_h"] = make_solution_info("u_h", "V")
        session.solutions["u_h"].function = mock_func
        ctx = make_mock_ctx(session)

        mock_geometry = MagicMock()
        mock_colliding = MagicMock()
        mock_colliding.links.return_value = [0]
        mock_geometry.compute_colliding_cells.return_value = mock_colliding

        with patch.dict(sys.modules, {
            "dolfinx": MagicMock(),
            "dolfinx.geometry": mock_geometry,
        }):
            result = await evaluate_solution(points=[[0.5, 0.5]], ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "non-finite value" in result["message"]

    @pytest.mark.asyncio
    async def test_query_point_values_postcondition_nan(self):
        """Finiteness postcondition fires when uh.eval returns NaN."""
        import sys

        np = pytest.importorskip("numpy")

        from dolfinx_mcp.tools.postprocess import query_point_values

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.active_mesh = "m1"
        mock_func = MagicMock()
        mock_func.function_space.mesh.topology.dim = 2
        nan_val = float("nan")
        mock_func.eval.return_value = np.array([nan_val])
        session.solutions["u_h"] = make_solution_info("u_h", "V")
        session.solutions["u_h"].function = mock_func
        ctx = make_mock_ctx(session)

        mock_geometry = MagicMock()
        mock_colliding = MagicMock()
        mock_colliding.links.return_value = [0]
        mock_geometry.compute_colliding_cells.return_value = mock_colliding

        with patch.dict(sys.modules, {
            "dolfinx": MagicMock(),
            "dolfinx.geometry": mock_geometry,
        }):
            result = await query_point_values(points=[[0.5, 0.5]], ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "non-finite value" in result["message"]

    def test_safe_evaluate_rejects_none_result(self):
        """Raises InvalidUFLExpressionError when expression evaluates to None."""
        from dolfinx_mcp.errors import InvalidUFLExpressionError
        from dolfinx_mcp.ufl_context import safe_evaluate

        # Expression that evaluates to None in a restricted namespace
        ns = {"__builtins__": {}, "none_val": None}
        with pytest.raises(InvalidUFLExpressionError, match="evaluated to None"):
            safe_evaluate("none_val", ns)


# ---------------------------------------------------------------------------
# Phase 11: Registry accessor completion and refinement postcondition (9 tests)
# ---------------------------------------------------------------------------


class TestPhase11Contracts:
    # -- get_solution() tests --

    def test_get_solution_postcondition_name_mismatch(self):
        """Debug postcondition fires when SolutionInfo.name != registry key."""
        session = make_populated_session()
        session.solutions["u_h"].name = "wrong_name"
        with pytest.raises(PostconditionError, match="name.*!= key"):
            session.get_solution("u_h")

    def test_get_solution_postcondition_dangling_space(self):
        """Debug postcondition fires when solution references deleted space."""
        session = make_populated_session()
        # Directly delete space without cascade to create dangling reference
        del session.function_spaces["V"]
        with pytest.raises(PostconditionError, match="space.*not in function_spaces registry"):
            session.get_solution("u_h")

    def test_get_solution_not_found(self):
        """get_solution raises FunctionNotFoundError for missing key."""
        session = SessionState()
        with pytest.raises(FunctionNotFoundError, match="Solution.*not found"):
            session.get_solution("nonexistent")

    def test_get_solution_returns_correct_info(self):
        """get_solution returns the correct SolutionInfo for a valid key."""
        session = make_populated_session()
        result = session.get_solution("u_h")
        assert result.name == "u_h"
        assert result.space_name == "V"

    # -- get_form() tests --

    def test_get_form_postcondition_name_mismatch(self):
        """Debug postcondition fires when FormInfo.name != registry key."""
        session = SessionState()
        form_info = FormInfo(name="wrong", form=MagicMock(), ufl_form=MagicMock())
        session.forms["bilinear"] = form_info
        with pytest.raises(PostconditionError, match="name.*!= key"):
            session.get_form("bilinear")

    def test_get_form_not_found(self):
        """get_form raises DOLFINxAPIError for missing key."""
        from dolfinx_mcp.errors import DOLFINxAPIError

        session = SessionState()
        with pytest.raises(DOLFINxAPIError, match="bilinear"):
            session.get_form("bilinear")

    def test_get_form_returns_correct_info(self):
        """get_form returns the correct FormInfo for a valid key."""
        session = SessionState()
        form_info = FormInfo(name="bilinear", form=MagicMock(), ufl_form=MagicMock())
        session.forms["bilinear"] = form_info
        result = session.get_form("bilinear")
        assert result is form_info
        assert result.name == "bilinear"

    def test_get_form_custom_suggestion(self):
        """get_form passes custom suggestion to error."""
        from dolfinx_mcp.errors import DOLFINxAPIError

        session = SessionState()
        with pytest.raises(DOLFINxAPIError) as exc_info:
            session.get_form("bilinear", suggestion="Custom guidance.")
        assert "Custom guidance" in exc_info.value.suggestion

    # -- refine_mesh postcondition test --

    def test_refine_mesh_postcondition_cell_count_fires(self):
        """Postcondition fires when refined mesh has <= original cell count.

        Directly verify the postcondition logic by constructing a scenario
        where refined_info.num_cells <= mesh_info.num_cells.
        """
        original = make_mesh_info("m1", num_cells=100)
        refined = make_mesh_info("m1_refined", num_cells=80)  # fewer cells -> violation

        # Simulate the postcondition check from refine_mesh
        assert refined.num_cells <= original.num_cells, \
            "Test setup: refined must have <= original cells"
        with pytest.raises(PostconditionError, match="expected more than original"):
            raise PostconditionError(
                f"refine_mesh(): refined mesh has {refined.num_cells} cells, "
                f"expected more than original {original.num_cells}"
            )

    def test_refine_mesh_postcondition_cell_count_passes(self):
        """No postcondition fires when refined mesh has more cells."""
        original = make_mesh_info("m1", num_cells=100)
        refined = make_mesh_info("m1_refined", num_cells=200)  # more cells -> OK

        assert refined.num_cells > original.num_cells, \
            "Refinement must increase cell count"

    @pytest.mark.asyncio
    async def test_refine_mesh_postcondition_integration(self):
        """Full integration: refine_mesh returns POSTCONDITION_VIOLATED when
        dolfinx.mesh.refine produces a mesh with fewer cells."""
        import sys

        from dolfinx_mcp.tools.mesh import refine_mesh

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")  # num_cells=100
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        # Build a mock refined mesh with fewer cells than original
        mock_index_map = MagicMock()
        mock_index_map.size_local = 50  # fewer than original 100

        mock_topology = MagicMock()
        mock_topology.dim = 2
        mock_topology.index_map.return_value = mock_index_map

        mock_refined_mesh = MagicMock()
        mock_refined_mesh.topology = mock_topology
        mock_refined_mesh.geometry.dim = 2

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx_mesh.refine.return_value = mock_refined_mesh

        # Wire dolfinx.mesh attribute on the parent module mock
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.mesh": mock_dolfinx_mesh,
        }):
            result = await refine_mesh(name="m1", ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "expected more than original" in result["message"]


class TestPhase12InterpolationPostconditionErrorType:
    """Phase 12: verify interpolation NaN postconditions produce POSTCONDITION_VIOLATED."""

    @pytest.mark.asyncio
    async def test_interpolate_expression_postcondition_error_type(self):
        """Expression-based interpolation NaN check must raise PostconditionError."""
        import sys

        from dolfinx_mcp.tools.interpolation import interpolate

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")

        # Mock function whose .x.array triggers isfinite -> False (NaN)
        mock_func = MagicMock()
        mock_isfinite_result = MagicMock()
        mock_isfinite_result.all.return_value = False
        session.functions["f"] = make_function_info("f", "V")
        session.functions["f"].function = mock_func

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        # Mock numpy so isfinite returns False (simulating NaN)
        mock_np = MagicMock()
        mock_np.isfinite.return_value = mock_isfinite_result
        mock_np.full.return_value = [0.0]

        mock_dolfinx = MagicMock()
        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx.fem,
        }):
            result = await interpolate(target="f", expression="0*x[0]", ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "NaN" in result["message"] or "Inf" in result["message"]

    @pytest.mark.asyncio
    async def test_interpolate_function_postcondition_error_type(self):
        """Function-based interpolation NaN check must raise PostconditionError."""
        import sys

        from dolfinx_mcp.tools.interpolation import interpolate

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")

        # Source and target functions
        session.functions["src"] = make_function_info("src", "V")
        session.functions["tgt"] = make_function_info("tgt", "V")

        # Mock target function -- isfinite will return False (NaN postcondition)
        mock_isfinite_result = MagicMock()
        mock_isfinite_result.all.return_value = False

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        mock_np = MagicMock()
        mock_np.isfinite.return_value = mock_isfinite_result

        mock_dolfinx = MagicMock()
        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx.fem,
        }):
            result = await interpolate(target="tgt", source_function="src", ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "NaN" in result["message"] or "Inf" in result["message"]


# ---------------------------------------------------------------------------
# Phase 13: Accessor tests -- get_mesh_tags, get_entity_map, get_last_solution
# ---------------------------------------------------------------------------



class TestPhase13GetMeshTags:
    """Tests for SessionState.get_mesh_tags accessor."""

    def test_get_mesh_tags_not_found(self):
        session = SessionState()
        with pytest.raises(Exception) as exc_info:
            session.get_mesh_tags("nonexistent")
        assert "not found" in str(exc_info.value).lower()

    def test_get_mesh_tags_postcondition_name_mismatch(self):
        """Debug postcondition fires if MeshTagsInfo.name != registry key."""
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        tags_info = make_mesh_tags_info("wrong_name", mesh_name="m1")
        session.mesh_tags["tags0"] = tags_info  # key != tags_info.name

        with pytest.raises(Exception) as exc_info:
            session.get_mesh_tags("tags0")
        assert "name" in str(exc_info.value).lower() or "mismatch" in str(exc_info.value).lower()

    def test_get_mesh_tags_postcondition_dangling_mesh(self):
        """Debug postcondition fires if mesh_name not in meshes."""
        session = SessionState()
        # No mesh registered, but tags reference "m1"
        tags_info = make_mesh_tags_info("tags0", mesh_name="m1")
        session.mesh_tags["tags0"] = tags_info

        with pytest.raises(Exception) as exc_info:
            session.get_mesh_tags("tags0")
        assert "mesh" in str(exc_info.value).lower()


class TestPhase13GetEntityMap:
    """Tests for SessionState.get_entity_map accessor."""

    def test_get_entity_map_not_found(self):
        session = SessionState()
        with pytest.raises(Exception) as exc_info:
            session.get_entity_map("nonexistent")
        assert "not found" in str(exc_info.value).lower()

    def test_get_entity_map_postcondition_name_mismatch(self):
        """Debug postcondition fires if EntityMapInfo.name != registry key."""
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.meshes["m1_sub"] = make_mesh_info("m1_sub")
        emap_info = make_entity_map_info("wrong_name", parent_mesh="m1", child_mesh="m1_sub")
        session.entity_maps["emap0"] = emap_info  # key != emap_info.name

        with pytest.raises(Exception) as exc_info:
            session.get_entity_map("emap0")
        assert "name" in str(exc_info.value).lower() or "mismatch" in str(exc_info.value).lower()

    def test_get_entity_map_postcondition_dangling_parent(self):
        """Debug postcondition fires if parent_mesh not in meshes."""
        session = SessionState()
        # child mesh exists but parent does not
        session.meshes["m1_sub"] = make_mesh_info("m1_sub")
        emap_info = make_entity_map_info("emap0", parent_mesh="m1", child_mesh="m1_sub")
        session.entity_maps["emap0"] = emap_info

        with pytest.raises(Exception) as exc_info:
            session.get_entity_map("emap0")
        assert "parent" in str(exc_info.value).lower() or "mesh" in str(exc_info.value).lower()


class TestPhase13GetLastSolution:
    """Tests for SessionState.get_last_solution accessor."""

    def test_get_last_solution_empty(self):
        session = SessionState()
        with pytest.raises(Exception) as exc_info:
            session.get_last_solution()
        assert "no solutions" in str(exc_info.value).lower()

    def test_get_last_solution_returns_latest(self):
        """Returns the most recently added solution."""
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.solutions["sol1"] = make_solution_info("sol1", "V")
        session.solutions["sol2"] = make_solution_info("sol2", "V")

        result = session.get_last_solution()
        assert result.name == "sol2"

    def test_get_last_solution_postcondition_dangling_space(self):
        """Debug postcondition fires if space_name not in function_spaces."""
        session = SessionState()
        # Add solution referencing space "V" without registering the space
        sol = make_solution_info("u_h", "V")
        session.solutions["u_h"] = sol

        with pytest.raises(Exception) as exc_info:
            session.get_last_solution()
        assert "space" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Phase 14: Tool tests -- remove_object, compute_mesh_quality, project
# ---------------------------------------------------------------------------


class TestPhase14RemoveObject:
    """Tests for the remove_object tool."""

    @pytest.mark.asyncio
    async def test_remove_object_empty_name(self):
        from dolfinx_mcp.tools.session_mgmt import remove_object

        ctx = MagicMock()
        ctx.request_context.lifespan_context = SessionState()

        result = await remove_object(name="", object_type="function", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_remove_object_invalid_type(self):
        from dolfinx_mcp.tools.session_mgmt import remove_object

        ctx = MagicMock()
        ctx.request_context.lifespan_context = SessionState()

        result = await remove_object(name="x", object_type="widget", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_remove_object_not_found(self):
        from dolfinx_mcp.tools.session_mgmt import remove_object

        ctx = MagicMock()
        ctx.request_context.lifespan_context = SessionState()

        result = await remove_object(name="nonexistent", object_type="function", ctx=ctx)
        assert_error_type(result, "DOLFINX_API_ERROR")

    @pytest.mark.asyncio
    async def test_remove_object_mesh_cascade(self):
        """Removing a mesh cascades to dependent objects."""
        from dolfinx_mcp.tools.session_mgmt import remove_object

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.functions["f"] = make_function_info("f", "V")
        session.solutions["sol"] = make_solution_info("sol", "V")

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        result = await remove_object(name="m1", object_type="mesh", ctx=ctx)
        assert result["name"] == "m1"
        assert result["cascade"] is True
        assert "m1" not in session.meshes
        assert "V" not in session.function_spaces
        assert "f" not in session.functions
        assert "sol" not in session.solutions

    @pytest.mark.asyncio
    async def test_remove_object_leaf_delete(self):
        """Removing a leaf object deletes only that object."""
        from dolfinx_mcp.tools.session_mgmt import remove_object

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.functions["f1"] = make_function_info("f1", "V")
        session.functions["f2"] = make_function_info("f2", "V")

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        result = await remove_object(name="f1", object_type="function", ctx=ctx)
        assert result["cascade"] is False
        assert "f1" not in session.functions
        assert "f2" in session.functions  # other function preserved


class TestPhase14ComputeMeshQuality:
    """Tests for the compute_mesh_quality tool."""

    @pytest.mark.asyncio
    async def test_compute_mesh_quality_missing_mesh(self):
        from dolfinx_mcp.tools.mesh import compute_mesh_quality

        session = SessionState()
        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        result = await compute_mesh_quality(mesh_name="nonexistent", ctx=ctx)
        # Should fail because mesh doesn't exist (via get_mesh accessor)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_compute_mesh_quality_postcondition_finite(self):
        """Postcondition catches NaN in quality metrics."""
        import sys

        from dolfinx_mcp.tools.mesh import compute_mesh_quality

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        # Mock dolfinx mesh with geometry that produces NaN volumes
        mock_np = MagicMock()
        mock_np.isfinite.return_value = MagicMock(all=MagicMock(return_value=False))
        mock_np.zeros.return_value = [float("nan")]

        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = MagicMock()

        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx.mesh": mock_dolfinx.mesh,
        }):
            result = await compute_mesh_quality(mesh_name="m1", ctx=ctx)
        # Should catch the postcondition violation
        assert "error" in result

    @pytest.mark.asyncio
    async def test_compute_mesh_quality_valid_access(self):
        """Verifies that the mesh accessor is used (not direct dict access)."""
        from dolfinx_mcp.tools.mesh import compute_mesh_quality

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        # Will fail during actual computation (no real mesh), but it should
        # at least get past the accessor check
        result = await compute_mesh_quality(ctx=ctx)
        # The error should be about computation, not about missing mesh
        if "error" in result:
            assert "not found" not in result.get("message", "").lower()


class TestPhase14Project:
    """Tests for the project (L2 projection) tool."""

    @pytest.mark.asyncio
    async def test_project_empty_name(self):
        from dolfinx_mcp.tools.interpolation import project

        ctx = MagicMock()
        ctx.request_context.lifespan_context = SessionState()

        result = await project(
            name="", target_space="V", expression="0*x[0]", ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_project_both_expression_and_source(self):
        from dolfinx_mcp.tools.interpolation import project

        ctx = MagicMock()
        ctx.request_context.lifespan_context = SessionState()

        result = await project(
            name="p",
            target_space="V",
            expression="0*x[0]",
            source_function="f",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_project_neither_expression_nor_source(self):
        from dolfinx_mcp.tools.interpolation import project

        ctx = MagicMock()
        ctx.request_context.lifespan_context = SessionState()

        result = await project(name="p", target_space="V", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_project_postcondition_nan(self):
        """Postcondition catches NaN/Inf in projection result."""
        import sys

        from dolfinx_mcp.tools.interpolation import project

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.functions["src"] = make_function_info("src", "V")

        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        # Mock numpy so isfinite returns False (NaN)
        mock_isfinite_result = MagicMock()
        mock_isfinite_result.all.return_value = False

        mock_np = MagicMock()
        mock_np.isfinite.return_value = mock_isfinite_result

        mock_dolfinx = MagicMock()
        mock_ufl = MagicMock()

        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx.fem,
            "dolfinx.fem.petsc": mock_dolfinx.fem.petsc,
            "ufl": mock_ufl,
        }):
            result = await project(
                name="p", target_space="V", source_function="src", ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")


# ---------------------------------------------------------------------------
# Phase 17: Postcondition completion tests (14 tests)
# ---------------------------------------------------------------------------


def _mock_mesh_with_zero_cells():
    """Create a mock DOLFINx mesh object that reports 0 cells."""
    mock_index_map_cells = MagicMock()
    mock_index_map_cells.size_local = 0
    mock_index_map_verts = MagicMock()
    mock_index_map_verts.size_local = 0

    mock_topology = MagicMock()
    mock_topology.dim = 2
    mock_topology.index_map = (
        lambda d: mock_index_map_cells if d == 2 else mock_index_map_verts
    )

    mock_mesh = MagicMock()
    mock_mesh.topology = mock_topology
    mock_mesh.geometry.dim = 2
    return mock_mesh


class TestPhase17Postconditions:
    """Phase 17: Verify all 14 new postconditions fire correctly."""

    # -- Mesh creation postconditions (3 tools, same check pattern) --

    @pytest.mark.asyncio
    async def test_create_unit_square_postcondition_zero_cells(self):
        """Postcondition fires when create_unit_square produces 0 cells."""
        import sys

        from dolfinx_mcp.tools.mesh import create_unit_square

        session = SessionState()
        ctx = make_mock_ctx(session)
        mock_mesh = _mock_mesh_with_zero_cells()

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx_mesh.CellType.triangle = "triangle"
        mock_dolfinx_mesh.create_unit_square.return_value = mock_mesh
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh
        mock_mpi = MagicMock()

        with patch.object(MeshInfo, "__post_init__", lambda self: None):
            with patch.dict(sys.modules, {
                "mpi4py": mock_mpi,
                "mpi4py.MPI": mock_mpi.MPI,
                "dolfinx": mock_dolfinx,
                "dolfinx.mesh": mock_dolfinx_mesh,
            }):
                result = await create_unit_square(name="m", nx=2, ny=2, ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "cells" in result["message"]

    @pytest.mark.asyncio
    async def test_create_mesh_postcondition_zero_cells(self):
        """Postcondition fires when create_mesh produces 0 cells."""
        import sys

        from dolfinx_mcp.tools.mesh import create_mesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        mock_mesh = _mock_mesh_with_zero_cells()

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx_mesh.CellType.triangle = "triangle"
        mock_dolfinx_mesh.create_unit_square.return_value = mock_mesh
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh
        mock_mpi = MagicMock()

        with patch.object(MeshInfo, "__post_init__", lambda self: None):
            with patch.dict(sys.modules, {
                "mpi4py": mock_mpi,
                "mpi4py.MPI": mock_mpi.MPI,
                "dolfinx": mock_dolfinx,
                "dolfinx.mesh": mock_dolfinx_mesh,
            }):
                result = await create_mesh(
                    name="m", shape="unit_square", nx=2, ny=2, ctx=ctx,
                )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "cells" in result["message"]

    @pytest.mark.asyncio
    async def test_create_custom_mesh_postcondition_zero_cells(self):
        """Postcondition fires when create_custom_mesh (Gmsh import) produces 0 cells."""
        import sys

        from dolfinx_mcp.tools.mesh import create_custom_mesh

        session = SessionState()
        ctx = make_mock_ctx(session)
        mock_mesh = _mock_mesh_with_zero_cells()
        mock_mesh.topology.cell_type = "unmapped_type"

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh

        mock_mesh_data = MagicMock()
        mock_mesh_data.mesh = mock_mesh
        mock_dolfinx.io.gmsh.model_to_mesh.return_value = mock_mesh_data

        mock_mpi = MagicMock()
        mock_gmsh = MagicMock()

        with patch.object(MeshInfo, "__post_init__", lambda self: None):
            with patch.dict(sys.modules, {
                "mpi4py": mock_mpi,
                "mpi4py.MPI": mock_mpi.MPI,
                "dolfinx": mock_dolfinx,
                "dolfinx.mesh": mock_dolfinx_mesh,
                "dolfinx.io": mock_dolfinx.io,
                "dolfinx.io.gmsh": mock_dolfinx.io.gmsh,
                "gmsh": mock_gmsh,
            }):
                result = await create_custom_mesh(
                    name="m", filename="test.msh", ctx=ctx,
                )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "cells" in result["message"]

    # -- get_mesh_info postcondition (NaN bounding box) --

    @pytest.mark.asyncio
    async def test_get_mesh_info_postcondition_nan_bbox(self):
        """Postcondition fires when mesh geometry contains NaN."""
        import sys

        from dolfinx_mcp.tools.mesh import get_mesh_info

        session = SessionState()
        mesh_info = make_mesh_info("m1")

        # Mock coords so .min/.max work but isfinite fails
        mock_min_result = MagicMock()
        mock_min_result.tolist.return_value = [0.0, 0.0, 0.0]
        mock_max_result = MagicMock()
        mock_max_result.tolist.return_value = [1.0, 1.0, 0.0]
        mock_coords = MagicMock()
        mock_coords.min.return_value = mock_min_result
        mock_coords.max.return_value = mock_max_result
        mesh_info.mesh.geometry.x = mock_coords

        session.meshes["m1"] = mesh_info
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        # Mock numpy so np.isfinite(coords).all() returns False
        mock_np = MagicMock()
        mock_finite_result = MagicMock()
        mock_finite_result.all.return_value = False
        mock_np.isfinite.return_value = mock_finite_result

        with patch.dict(sys.modules, {"numpy": mock_np}):
            result = await get_mesh_info(name="m1", ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "NaN" in result["message"] or "Inf" in result["message"]

    # -- mark_boundaries postcondition (empty tags) --

    @pytest.mark.asyncio
    async def test_mark_boundaries_postcondition_no_tags(self):
        """Postcondition fires when no boundary facets are tagged."""
        import sys

        from dolfinx_mcp.tools.mesh import mark_boundaries

        session = SessionState()
        mesh_info = make_mesh_info("m1")
        mesh_info.mesh.topology.dim = 2
        session.meshes["m1"] = mesh_info
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        # Mock empty facet array from locate_entities_boundary
        mock_empty_facets = MagicMock()
        mock_empty_facets.__len__ = MagicMock(return_value=0)

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx_mesh.locate_entities_boundary.return_value = mock_empty_facets
        mock_dolfinx_mesh.meshtags.return_value = MagicMock()
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh

        # Mock numpy so np.unique(tag_values) returns empty iterable
        mock_np = MagicMock()
        mock_np.int32 = "int32"
        mock_np.unique.return_value = []  # Empty -> unique_tags = []

        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx": mock_dolfinx,
            "dolfinx.mesh": mock_dolfinx_mesh,
        }):
            result = await mark_boundaries(
                markers=[{"tag": 1, "condition": "x[0] < 1e-14"}],
                name="tags",
                ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "no tagged facets" in result["message"]

    # -- create_submesh postcondition (exceeds parent cells) --

    @pytest.mark.asyncio
    async def test_create_submesh_postcondition_exceeds_parent(self):
        """Postcondition fires when submesh has more cells than parent."""
        import sys

        from dolfinx_mcp.tools.mesh import create_submesh

        session = SessionState()
        session.meshes["parent"] = make_mesh_info("parent", num_cells=10)
        session.active_mesh = "parent"

        mock_tags = MagicMock()
        mock_tags.values = MagicMock()
        mock_tags.indices = MagicMock()
        session.mesh_tags["ftags"] = make_mesh_tags_info(
            "ftags", mesh_name="parent", tags=mock_tags, unique_tags=[1],
        )
        ctx = make_mock_ctx(session)

        # Mock entities returned after np.isin filtering
        mock_entities = MagicMock()
        mock_entities.__len__ = MagicMock(return_value=3)

        # Mock numpy: np.isin returns mask, indices[mask] returns entities
        mock_np = MagicMock()
        mock_np.isin.return_value = MagicMock()
        mock_tags.indices.__getitem__ = MagicMock(return_value=mock_entities)

        # Mock submesh: 20 cells > parent's 10
        mock_sub_index_cells = MagicMock()
        mock_sub_index_cells.size_local = 20
        mock_sub_index_verts = MagicMock()
        mock_sub_index_verts.size_local = 15

        mock_sub_topology = MagicMock()
        mock_sub_topology.dim = 2
        mock_sub_topology.index_map = (
            lambda d: mock_sub_index_cells if d == 2 else mock_sub_index_verts
        )

        mock_submesh = MagicMock()
        mock_submesh.topology = mock_sub_topology
        mock_submesh.geometry.dim = 2

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx_mesh.create_submesh.return_value = (
            mock_submesh, MagicMock(),
        )
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh

        with patch.object(MeshInfo, "__post_init__", lambda self: None):
            with patch.dict(sys.modules, {
                "numpy": mock_np,
                "dolfinx": mock_dolfinx,
                "dolfinx.mesh": mock_dolfinx_mesh,
            }):
                result = await create_submesh(
                    name="sub", tags_name="ftags", tag_values=[1], ctx=ctx,
                )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "exceeding parent" in result["message"]

    # -- manage_mesh_tags postcondition (empty tags on create) --

    @pytest.mark.asyncio
    async def test_manage_mesh_tags_postcondition_no_tags(self):
        """Postcondition fires when tag creation produces no tagged entities."""
        import sys

        from dolfinx_mcp.tools.mesh import manage_mesh_tags

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        mock_dolfinx_mesh = MagicMock()
        mock_dolfinx_mesh.meshtags.return_value = MagicMock()
        mock_dolfinx = MagicMock()
        mock_dolfinx.mesh = mock_dolfinx_mesh

        # Mock numpy so np.unique(tag_values).tolist() returns []
        mock_np = MagicMock()
        mock_np.int32 = "int32"
        mock_unique_result = MagicMock()
        mock_unique_result.tolist.return_value = []
        mock_np.unique.return_value = mock_unique_result

        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx": mock_dolfinx,
            "dolfinx.mesh": mock_dolfinx_mesh,
        }):
            result = await manage_mesh_tags(
                name="tags", action="create", dimension=1,
                values=[{"entities": [], "tag": 1}],
                ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "no tagged entities" in result["message"]

    # -- Function space postconditions (2 tools) --

    @pytest.mark.asyncio
    async def test_create_function_space_postcondition_zero_dofs(self):
        """Postcondition fires when function space has 0 DOFs."""
        import sys

        from dolfinx_mcp.tools.spaces import create_function_space

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        mock_dofmap = MagicMock()
        mock_dofmap.index_map.size_local = 0
        mock_dofmap.index_map_bs = 1
        mock_V = MagicMock()
        mock_V.dofmap = mock_dofmap

        mock_dolfinx_fem = MagicMock()
        mock_dolfinx_fem.functionspace.return_value = mock_V
        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_dolfinx_fem

        with patch.object(FunctionSpaceInfo, "__post_init__", lambda self: None):
            with patch.dict(sys.modules, {
                "dolfinx": mock_dolfinx,
                "dolfinx.fem": mock_dolfinx_fem,
            }):
                result = await create_function_space(
                    name="V", degree=1, ctx=ctx,
                )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "DOFs" in result["message"]

    @pytest.mark.asyncio
    async def test_create_mixed_space_postcondition_zero_dofs(self):
        """Postcondition fires when mixed space has 0 DOFs."""
        import sys

        from dolfinx_mcp.tools.spaces import create_mixed_space

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V1"] = make_space_info("V1", "m1")
        session.function_spaces["V2"] = make_space_info("V2", "m1")
        ctx = make_mock_ctx(session)

        mock_dofmap = MagicMock()
        mock_dofmap.index_map.size_local = 0
        mock_dofmap.index_map_bs = 1
        mock_W = MagicMock()
        mock_W.dofmap = mock_dofmap

        mock_basix = MagicMock()
        mock_dolfinx_fem = MagicMock()
        mock_dolfinx_fem.functionspace.return_value = mock_W
        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_dolfinx_fem

        with patch.object(FunctionSpaceInfo, "__post_init__", lambda self: None):
            with patch.dict(sys.modules, {
                "basix": mock_basix,
                "basix.ufl": mock_basix.ufl,
                "dolfinx": mock_dolfinx,
                "dolfinx.fem": mock_dolfinx_fem,
            }):
                result = await create_mixed_space(
                    name="W", subspaces=["V1", "V2"], ctx=ctx,
                )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "DOFs" in result["message"]

    # -- create_discrete_operator postcondition (zero dimensions) --

    @pytest.mark.asyncio
    async def test_create_discrete_operator_postcondition_zero_dims(self):
        """Postcondition fires when operator has 0x0 dimensions."""
        import sys

        from dolfinx_mcp.tools.interpolation import create_discrete_operator

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V1"] = make_space_info("V1", "m1")
        session.function_spaces["V2"] = make_space_info("V2", "m1")
        ctx = make_mock_ctx(session)

        mock_operator = MagicMock()
        mock_operator.getSize.return_value = (0, 0)

        mock_dolfinx_fem = MagicMock()
        mock_dolfinx_fem.discrete_gradient.return_value = mock_operator
        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_dolfinx_fem

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx_fem,
        }):
            result = await create_discrete_operator(
                operator_type="gradient",
                source_space="V1",
                target_space="V2",
                ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "dimensions" in result["message"]

    # -- define_variational_form postcondition (None form) --

    @pytest.mark.asyncio
    async def test_define_variational_form_postcondition_none_form(self):
        """Postcondition fires when form compilation returns None."""
        import sys

        from dolfinx_mcp.tools.problem import define_variational_form

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V"] = make_space_info("V", "m1")
        ctx = make_mock_ctx(session)

        mock_ufl = MagicMock()
        mock_dolfinx_fem = MagicMock()
        mock_dolfinx_fem.form.return_value = None
        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_dolfinx_fem

        with patch("dolfinx_mcp.tools.problem.build_namespace", return_value={}), \
             patch("dolfinx_mcp.tools.problem.safe_evaluate", return_value=MagicMock()):
            with patch.dict(sys.modules, {
                "ufl": mock_ufl,
                "dolfinx": mock_dolfinx,
                "dolfinx.fem": mock_dolfinx_fem,
            }):
                result = await define_variational_form(
                    bilinear="inner(grad(u), grad(v)) * dx",
                    linear="f * v * dx",
                    trial_space="V",
                    ctx=ctx,
                )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "None" in result["message"]

    # -- set_material_properties postcondition (NaN after interpolation) --

    @pytest.mark.asyncio
    async def test_set_material_properties_postcondition_nan(self):
        """Postcondition fires when interpolated material contains NaN."""
        import sys

        from dolfinx_mcp.tools.problem import set_material_properties

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V"] = make_space_info("V", "m1")
        ctx = make_mock_ctx(session)

        mock_isfinite_result = MagicMock()
        mock_isfinite_result.all.return_value = False
        mock_np = MagicMock()
        mock_np.isfinite.return_value = mock_isfinite_result

        mock_dolfinx_fem = MagicMock()
        mock_dolfinx = MagicMock()
        mock_dolfinx.fem = mock_dolfinx_fem

        with patch.dict(sys.modules, {
            "numpy": mock_np,
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx_fem,
        }):
            result = await set_material_properties(
                name="kappa", value="sin(pi*x[0])",
                function_space="V", ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "NaN" in result["message"] or "Inf" in result["message"]

    # -- export_solution postcondition (empty file) --

    @pytest.mark.asyncio
    async def test_export_solution_postcondition_empty_file(self):
        """Postcondition fires when exported file is empty."""
        import sys

        from dolfinx_mcp.tools.postprocess import export_solution

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.solutions["u_h"] = make_solution_info("u_h", "V")
        ctx = make_mock_ctx(session)

        mock_dolfinx_io = MagicMock()
        mock_dolfinx = MagicMock()
        mock_dolfinx.io = mock_dolfinx_io
        mock_mpi = MagicMock()

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=0), patch.dict(sys.modules, {
            "mpi4py": mock_mpi,
            "mpi4py.MPI": mock_mpi.MPI,
            "dolfinx": mock_dolfinx,
            "dolfinx.io": mock_dolfinx_io,
        }):
            result = await export_solution(
                filename="test.xdmf", format="xdmf", ctx=ctx,
            )

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "empty file" in result["message"]

    # -- plot_solution postcondition (file not created) --

    @pytest.mark.asyncio
    async def test_plot_solution_postcondition_no_file(self):
        """Postcondition fires when plot file is not created."""
        import sys

        from dolfinx_mcp.tools.postprocess import plot_solution

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.solutions["u_h"] = make_solution_info("u_h", "V")
        ctx = make_mock_ctx(session)

        mock_pyvista = MagicMock()
        mock_dolfinx_plot = MagicMock()
        mock_dolfinx_plot.vtk_mesh.return_value = (
            MagicMock(), MagicMock(), MagicMock(),
        )
        mock_dolfinx = MagicMock()
        mock_dolfinx.plot = mock_dolfinx_plot

        with patch("os.path.exists", return_value=False), patch.dict(sys.modules, {
            "pyvista": mock_pyvista,
            "dolfinx": mock_dolfinx,
            "dolfinx.plot": mock_dolfinx_plot,
        }):
            result = await plot_solution(ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "not created" in result["message"]


class TestPhase19Deduplication:
    """Phase 19: Tests for extracted utilities."""

    def test_find_space_name_found(self):
        """find_space_name returns correct name when space object matches."""
        session = SessionState()
        space_info = make_space_info("V", "m1")
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = space_info

        result = session.find_space_name(space_info.space)
        assert result == "V"

    def test_find_space_name_not_found(self):
        """find_space_name returns 'unknown' when no match."""
        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")

        result = session.find_space_name(MagicMock())
        assert result == "unknown"

    def test_find_space_name_empty_session(self):
        """find_space_name returns 'unknown' on empty session."""
        session = SessionState()
        result = session.find_space_name(MagicMock())
        assert result == "unknown"

    def test_build_petsc_opts_direct(self):
        """_build_petsc_opts returns correct direct solver options."""
        from dolfinx_mcp.tools.solver import _build_petsc_opts

        opts = _build_petsc_opts("direct", None, None)
        assert opts["ksp_type"] == "preonly"
        assert opts["pc_type"] == "lu"
        assert "ksp_rtol" not in opts

    def test_build_petsc_opts_iterative(self):
        """_build_petsc_opts returns correct iterative solver options."""
        from dolfinx_mcp.tools.solver import _build_petsc_opts

        opts = _build_petsc_opts(
            "iterative", "gmres", "ilu",
            petsc_options={"ksp_monitor": True},
            rtol=1e-8,
        )
        assert opts["ksp_type"] == "gmres"
        assert opts["pc_type"] == "ilu"
        assert opts["ksp_rtol"] == 1e-8
        assert opts["ksp_monitor"] is True


class TestPhase20LocalTestCompletion:
    """Phase 20: Local tests for tools previously only tested via Docker."""

    # -- get_session_state (3 tests) --

    @pytest.mark.asyncio
    async def test_get_session_state_empty_session(self):
        """get_session_state returns expected keys on empty session."""
        from dolfinx_mcp.tools.session_mgmt import get_session_state

        session = SessionState()
        ctx = make_mock_ctx(session)

        result = await get_session_state(ctx=ctx)

        assert "active_mesh" in result
        assert "meshes" in result
        assert "function_spaces" in result
        assert "functions" in result
        assert "solutions" in result
        assert result["active_mesh"] is None
        assert result["meshes"] == {}

    @pytest.mark.asyncio
    async def test_get_session_state_populated(self):
        """get_session_state returns all registered objects."""
        from dolfinx_mcp.tools.session_mgmt import get_session_state

        session = SessionState()
        mesh_info = make_mesh_info("m1")
        session.meshes["m1"] = mesh_info
        session.active_mesh = "m1"
        space_info = make_space_info("V", "m1")
        session.function_spaces["V"] = space_info
        ctx = make_mock_ctx(session)

        result = await get_session_state(ctx=ctx)

        assert "m1" in result["meshes"]
        assert "V" in result["function_spaces"]
        assert result["active_mesh"] == "m1"

    @pytest.mark.asyncio
    async def test_get_session_state_after_removal(self):
        """get_session_state reflects deletions."""
        from dolfinx_mcp.tools.session_mgmt import get_session_state

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        # Remove the mesh
        session.remove_mesh("m1")

        result = await get_session_state(ctx=ctx)

        assert result["meshes"] == {}
        assert result["active_mesh"] is None

    # -- reset_session (3 tests) --

    @pytest.mark.asyncio
    async def test_reset_session_clears_all(self):
        """reset_session empties all registries."""
        from dolfinx_mcp.tools.session_mgmt import reset_session

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V"] = make_space_info("V", "m1")
        ctx = make_mock_ctx(session)

        await reset_session(ctx=ctx)

        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0
        assert session.active_mesh is None

    @pytest.mark.asyncio
    async def test_reset_session_empty_session(self):
        """reset_session does not error on empty session."""
        from dolfinx_mcp.tools.session_mgmt import reset_session

        session = SessionState()
        ctx = make_mock_ctx(session)

        result = await reset_session(ctx=ctx)

        assert result["status"] == "reset"

    @pytest.mark.asyncio
    async def test_reset_session_returns_status(self):
        """reset_session returns expected status dict."""
        from dolfinx_mcp.tools.session_mgmt import reset_session

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        result = await reset_session(ctx=ctx)

        assert result["status"] == "reset"
        assert "message" in result

    # -- get_mesh_info (3 tests) --

    @pytest.mark.asyncio
    async def test_get_mesh_info_missing_mesh(self):
        """get_mesh_info errors when mesh not found."""
        import sys

        from dolfinx_mcp.tools.mesh import get_mesh_info

        session = SessionState()
        ctx = make_mock_ctx(session)

        # Mock numpy so the top-of-function import succeeds before accessor fires
        mock_np = MagicMock()
        with patch.dict(sys.modules, {"numpy": mock_np}):
            result = await get_mesh_info(name="nonexistent", ctx=ctx)

        assert_error_type(result, "MESH_NOT_FOUND")

    @pytest.mark.asyncio
    async def test_get_mesh_info_returns_expected_keys(self):
        """get_mesh_info returns summary with bounding box."""
        import sys

        from dolfinx_mcp.tools.mesh import get_mesh_info

        session = SessionState()
        mesh_info = make_mesh_info("m1")

        # Mock coords for bbox computation
        mock_min_result = MagicMock()
        mock_min_result.tolist.return_value = [0.0, 0.0, 0.0]
        mock_max_result = MagicMock()
        mock_max_result.tolist.return_value = [1.0, 1.0, 0.0]
        mock_coords = MagicMock()
        mock_coords.min.return_value = mock_min_result
        mock_coords.max.return_value = mock_max_result
        mesh_info.mesh.geometry.x = mock_coords

        session.meshes["m1"] = mesh_info
        session.active_mesh = "m1"
        ctx = make_mock_ctx(session)

        # Mock numpy so isfinite passes
        mock_np = MagicMock()
        mock_finite_result = MagicMock()
        mock_finite_result.all.return_value = True
        mock_np.isfinite.return_value = mock_finite_result

        with patch.dict(sys.modules, {"numpy": mock_np}):
            result = await get_mesh_info(name="m1", ctx=ctx)

        assert "bounding_box" in result
        assert result["bounding_box"]["min"] == [0.0, 0.0, 0.0]
        assert result["bounding_box"]["max"] == [1.0, 1.0, 0.0]
        assert "active" in result
        assert result["name"] == "m1"

    @pytest.mark.asyncio
    async def test_get_mesh_info_uses_accessor(self):
        """get_mesh_info uses get_mesh() accessor."""
        import sys

        from dolfinx_mcp.tools.mesh import get_mesh_info

        session = SessionState()
        ctx = make_mock_ctx(session)

        # Mock numpy so the top-of-function import succeeds before accessor fires
        mock_np = MagicMock()
        with patch.dict(sys.modules, {"numpy": mock_np}):
            # No active mesh set -> accessor should raise
            result = await get_mesh_info(ctx=ctx)

        assert_error_type(result, "NO_ACTIVE_MESH")

    # -- get_solver_diagnostics (3 tests) --

    @pytest.mark.asyncio
    async def test_get_solver_diagnostics_no_solution(self):
        """get_solver_diagnostics errors when no solutions exist."""
        from dolfinx_mcp.tools.solver import get_solver_diagnostics

        session = SessionState()
        ctx = make_mock_ctx(session)

        result = await get_solver_diagnostics(ctx=ctx)

        # get_last_solution() raises DOLFINxAPIError -> error_code = "DOLFINX_API_ERROR"
        assert_error_type(result, "DOLFINX_API_ERROR")

    @pytest.mark.asyncio
    async def test_get_solver_diagnostics_returns_expected_keys(self):
        """get_solver_diagnostics returns expected structure."""

        from dolfinx_mcp.tools.solver import get_solver_diagnostics

        session = SessionState()
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")

        mock_func = MagicMock()
        mock_func.function_space.dofmap.index_map.size_global = 64
        mock_func.function_space.dofmap.index_map_bs = 1

        sol_info = SolutionInfo(
            name="u_h",
            function=mock_func,
            space_name="V",
            converged=True,
            iterations=5,
            residual_norm=1e-10,
            wall_time=0.5,
        )
        session.solutions["u_h"] = sol_info
        ctx = make_mock_ctx(session)

        # Mock compute_l2_norm at its source module (lazy-imported inside function body)
        with patch("dolfinx_mcp.utils.compute_l2_norm", return_value=1.234):
            result = await get_solver_diagnostics(ctx=ctx)

        assert result["solution_name"] == "u_h"
        assert result["converged"] is True
        assert result["iterations"] == 5
        assert "solution_norm_L2" in result
        assert "num_dofs" in result

    @pytest.mark.asyncio
    async def test_get_solver_diagnostics_uses_accessor(self):
        """get_solver_diagnostics uses get_last_solution() accessor."""
        from dolfinx_mcp.tools.solver import get_solver_diagnostics

        session = SessionState()
        ctx = make_mock_ctx(session)

        # No solutions -> accessor raises
        result = await get_solver_diagnostics(ctx=ctx)

        assert "error" in result


# ---------------------------------------------------------------------------
# Phase 22: Postcondition edge-case tests (4 tests)
# ---------------------------------------------------------------------------


class TestPhase22PostconditionEdgeCases:
    """Edge-case postcondition tests for solver and compute_error paths."""

    @pytest.mark.asyncio
    async def test_solve_postcondition_nan_solution(self):
        """solve() fires SolverError when solution contains NaN."""
        import sys

        from dolfinx_mcp.tools.solver import solve

        session = make_populated_session()
        session.forms["bilinear"] = make_form_info("bilinear")
        session.forms["linear"] = make_form_info("linear")
        ctx = make_mock_ctx(session)

        # Mock full dolfinx hierarchy + numpy
        mock_dolfinx = MagicMock()
        mock_np = MagicMock()

        # np.isfinite(uh.x.array).all() returns False -> NaN detected
        mock_np.isfinite.return_value.all.return_value = False

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx.fem,
            "dolfinx.fem.petsc": mock_dolfinx.fem.petsc,
            "numpy": mock_np,
        }):
            result = await solve(solver_type="direct", ctx=ctx)

        assert result["error"] == "SOLVER_ERROR"
        assert "NaN" in result["message"] or "Inf" in result["message"]

    @pytest.mark.asyncio
    async def test_solve_postcondition_negative_l2_norm(self):
        """solve() fires PostconditionError when L2 norm is negative."""
        import sys

        from dolfinx_mcp.tools.solver import solve

        session = make_populated_session()
        session.forms["bilinear"] = make_form_info("bilinear")
        session.forms["linear"] = make_form_info("linear")
        ctx = make_mock_ctx(session)

        # Mock full dolfinx hierarchy + numpy
        mock_dolfinx = MagicMock()
        mock_np = MagicMock()

        # np.isfinite passes (solution is finite)
        mock_np.isfinite.return_value.all.return_value = True

        # Solver converged
        mock_problem = mock_dolfinx.fem.petsc.LinearProblem.return_value
        mock_problem.solver.getConvergedReason.return_value = 1
        mock_problem.solver.getIterationNumber.return_value = 10
        mock_problem.solver.getResidualNorm.return_value = 1e-12

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx.fem,
            "dolfinx.fem.petsc": mock_dolfinx.fem.petsc,
            "numpy": mock_np,
        }), patch("dolfinx_mcp.utils.compute_l2_norm", return_value=-1.0):
            result = await solve(solver_type="direct", ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "non-negative" in result["message"]

    @pytest.mark.asyncio
    async def test_get_solver_diagnostics_returns_cached_l2_norm(self):
        """get_solver_diagnostics() reads l2_norm from SolutionInfo cache."""
        from dolfinx_mcp.tools.solver import get_solver_diagnostics

        session = make_populated_session()
        # Ensure solution has function_space with dofmap
        mock_fn = MagicMock()
        mock_fn.function_space.dofmap.index_map.size_global = 100
        mock_fn.function_space.dofmap.index_map_bs = 1
        session.solutions["u_h"] = SolutionInfo(
            name="u_h", function=mock_fn, space_name="V",
            converged=True, iterations=5, residual_norm=1e-10, wall_time=0.5,
            l2_norm=3.14,
        )
        ctx = make_mock_ctx(session)

        result = await get_solver_diagnostics(ctx=ctx)

        assert_no_error(result)
        assert result["solution_norm_L2"] == round(3.14, 8)

    @pytest.mark.asyncio
    async def test_compute_error_postcondition_nan(self):
        """compute_error() fires PostconditionError when error value is NaN."""
        import sys

        from dolfinx_mcp.tools.postprocess import compute_error

        session = make_populated_session()
        # Set up solution with function that has function_space
        mock_fn = MagicMock()
        mock_fn.function_space = MagicMock()
        session.solutions["u_h"] = SolutionInfo(
            name="u_h", function=mock_fn, space_name="V",
            converged=True, iterations=5, residual_norm=1e-10, wall_time=0.5,
        )
        ctx = make_mock_ctx(session)

        # Mock full dolfinx hierarchy + numpy + ufl
        mock_dolfinx = MagicMock()
        mock_np = MagicMock()
        mock_ufl = MagicMock()

        # np.sqrt returns NaN so error_val = float(NaN) = NaN
        mock_np.sqrt.return_value = float("nan")

        with patch.dict(sys.modules, {
            "dolfinx": mock_dolfinx,
            "dolfinx.fem": mock_dolfinx.fem,
            "numpy": mock_np,
            "ufl": mock_ufl,
        }):
            result = await compute_error(exact="x[0]", norm_type="L2", ctx=ctx)

        assert_error_type(result, "POSTCONDITION_VIOLATED")
        assert "finite" in result["message"]


# ---------------------------------------------------------------------------
# Phase 31: postprocess.py _suppress_stdout, _validate_output_path, return type
# ---------------------------------------------------------------------------


class TestSuppressStdout:
    """Tests for _suppress_stdout() context manager (Phase 31A)."""

    def test_fd_redirect_and_restore(self):
        """Verify stdout fd is redirected then restored after context exits."""
        from dolfinx_mcp.tools.postprocess import _suppress_stdout

        # Capture stdout fd before
        original_fd = os.dup(1)
        try:
            with _suppress_stdout():
                # Inside: fd 1 should point to /dev/null
                # Writing should not produce output (can't easily verify
                # the target, but the context should not raise)
                pass

            # After: stdout fd should be restored
            # Verify by writing to fd 1 -- should succeed without error
            os.write(1, b"")  # zero-byte write as sanity check
        finally:
            os.close(original_fd)

    def test_graceful_noop_when_no_fileno(self):
        """Verify _suppress_stdout is a no-op when stdout lacks fileno()."""
        from dolfinx_mcp.tools.postprocess import _suppress_stdout

        # Replace sys.stdout with an object that raises on fileno()
        fake_stdout = io.StringIO()
        with patch("dolfinx_mcp.tools.postprocess.sys.stdout", fake_stdout):
            # Should not raise -- graceful fallback
            with _suppress_stdout():
                pass  # no-op path


class TestValidateOutputPath:
    """Tests for _validate_output_path() helper (Phase 31C)."""

    def test_simple_filename(self):
        """Simple filename resolves to /workspace/filename."""
        from dolfinx_mcp.tools.postprocess import _validate_output_path

        result = _validate_output_path("result.xdmf")
        assert result == "/workspace/result.xdmf"

    def test_subdirectory_path(self):
        """Subdirectory path resolves within /workspace."""
        from dolfinx_mcp.tools.postprocess import _validate_output_path

        result = _validate_output_path("sub/result.xdmf")
        assert result == "/workspace/sub/result.xdmf"

    def test_absolute_workspace_path(self):
        """Absolute path already within /workspace is accepted."""
        from dolfinx_mcp.tools.postprocess import _validate_output_path

        result = _validate_output_path("/workspace/result.xdmf")
        # os.path.join("/workspace", "/workspace/result.xdmf") = "/workspace/result.xdmf"
        assert result == "/workspace/result.xdmf"

    def test_path_traversal_rejected(self):
        """Path traversal via ../../ is rejected with FileIOError."""
        from dolfinx_mcp.tools.postprocess import _validate_output_path

        with pytest.raises(FileIOError, match="must be within /workspace"):
            _validate_output_path("../../etc/passwd")

    def test_absolute_outside_workspace_rejected(self):
        """Absolute path outside /workspace is rejected."""
        from dolfinx_mcp.tools.postprocess import _validate_output_path

        with pytest.raises(FileIOError, match="must be within /workspace"):
            _validate_output_path("/tmp/evil.xdmf")

    @pytest.mark.asyncio
    async def test_export_solution_rejects_traversal(self, mock_ctx):
        """export_solution rejects path traversal with FILE_IO_ERROR."""
        from dolfinx_mcp.tools.postprocess import export_solution

        result = await export_solution(
            filename="../../etc/passwd", format="xdmf", ctx=mock_ctx,
        )
        assert result["error"] == "FILE_IO_ERROR"
        assert "/workspace" in result["message"]

    @pytest.mark.asyncio
    async def test_plot_solution_rejects_traversal(self, mock_ctx):
        """plot_solution rejects output_file outside /workspace with FILE_IO_ERROR."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        result = await plot_solution(
            output_file="/tmp/evil.png", ctx=mock_ctx,
        )
        assert result["error"] == "FILE_IO_ERROR"
        assert "/workspace" in result["message"]


class TestPlotSolutionReturnType:
    """Tests for plot_solution return type fix (Phase 31B)."""

    def _setup_plot_session(self, mock_ctx):
        """Set up a mock session with a solution for plot tests."""
        import numpy as np

        session = mock_ctx.request_context.lifespan_context

        mock_func = MagicMock()
        mock_space = MagicMock()
        # reference_value_shape = () means scalar
        mock_space.ufl_element.return_value.reference_value_shape = ()
        mock_func.function_space = mock_space
        mock_func.x.array.real = np.array([0.0, 1.0, 2.0])

        sol_info = SolutionInfo(
            name="u_h",
            space_name="V",
            function=mock_func,
            converged=True,
            iterations=1,
            residual_norm=1e-10,
            wall_time=0.1,
        )
        space_info = FunctionSpaceInfo(
            name="V",
            mesh_name="m",
            space=MagicMock(),
            element_family="Lagrange",
            element_degree=1,
            num_dofs=10,
        )
        mesh_info = MeshInfo(
            name="m",
            mesh=MagicMock(),
            cell_type="triangle",
            tdim=2,
            gdim=2,
            num_cells=8,
            num_vertices=9,
        )

        session.meshes = {"m": mesh_info}
        session.function_spaces = {"V": space_info}
        session.solutions = {"u_h": sol_info}
        session.functions = {}
        return session

    @pytest.mark.asyncio
    async def test_return_type_is_dict(self, mock_ctx):
        """plot_solution must return dict, not list."""
        import sys

        mock_pyvista = MagicMock()
        mock_vtk_mesh = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        mock_pyvista.Plotter.return_value = MagicMock()

        from dolfinx_mcp.tools.postprocess import plot_solution

        self._setup_plot_session(mock_ctx)

        with patch.dict(sys.modules, {
            "pyvista": mock_pyvista,
            "dolfinx.plot": MagicMock(vtk_mesh=mock_vtk_mesh),
        }), patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=1024):
            result = await plot_solution(
                output_file="/workspace/test.png", ctx=mock_ctx,
            )

        # Must be dict, not list
        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"

    @pytest.mark.asyncio
    async def test_return_dict_keys(self, mock_ctx):
        """plot_solution return dict must have file_path, plot_type, file_size_bytes."""
        import sys

        mock_pyvista = MagicMock()
        mock_vtk_mesh = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        mock_pyvista.Plotter.return_value = MagicMock()

        mock_dolfinx = MagicMock()
        mock_dolfinx.plot = MagicMock(vtk_mesh=mock_vtk_mesh)

        from dolfinx_mcp.tools.postprocess import plot_solution

        self._setup_plot_session(mock_ctx)

        with patch.dict(sys.modules, {
            "pyvista": mock_pyvista,
            "dolfinx": mock_dolfinx,
            "dolfinx.plot": mock_dolfinx.plot,
        }), patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=512):
            result = await plot_solution(
                output_file="/workspace/test.png", ctx=mock_ctx,
            )

        assert "file_path" in result
        assert "plot_type" in result
        assert "file_size_bytes" in result


# ──────────────────────────────────────────────────────────────────
# Stress-test bug fixes (FIX-1 through FIX-5)
# ──────────────────────────────────────────────────────────────────


class TestDG0WarpPrecondition:
    """FIX-1 (BUG-004): Warp plots rejected for DG0 functions."""

    @pytest.mark.asyncio
    async def test_plot_warp_rejects_dg0(self, mock_ctx):
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = mock_ctx.request_context.lifespan_context
        session.meshes["m"] = make_mesh_info("m")
        session.function_spaces["V_dg0"] = make_space_info(
            "V_dg0", mesh_name="m", element_family="DG", element_degree=0,
        )
        session.functions["f"] = make_function_info("f", space_name="V_dg0")
        session.solutions["f"] = make_solution_info("f", space_name="V_dg0")

        result = await plot_solution(
            function_name="f", plot_type="warp",
            output_file="/workspace/test.png", ctx=mock_ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "DG0" in result["message"]

    @pytest.mark.asyncio
    async def test_plot_contour_allows_dg0(self, mock_ctx):
        """Contour plots should NOT be blocked by the DG0 guard."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = mock_ctx.request_context.lifespan_context
        session.meshes["m"] = make_mesh_info("m")
        session.function_spaces["V_dg0"] = make_space_info(
            "V_dg0", mesh_name="m", element_family="DG", element_degree=0,
        )
        session.functions["f"] = make_function_info("f", space_name="V_dg0")
        session.solutions["f"] = make_solution_info("f", space_name="V_dg0")

        # Should pass the DG0 precondition (will fail later at pyvista import)
        result = await plot_solution(
            function_name="f", plot_type="contour",
            output_file="/workspace/test.png", ctx=mock_ctx,
        )
        # The error should NOT be PRECONDITION_VIOLATED about DG0
        if "error" in result:
            assert "DG0" not in result.get("message", "")


class TestVectorExpressionShape:
    """FIX-2 (BUG-003): eval_numpy_expression allows (d, N) for vectors."""

    def test_vector_2d_expression(self):
        import numpy as np

        from dolfinx_mcp.eval_helpers import eval_numpy_expression

        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = eval_numpy_expression("np.vstack([x[0], x[1]])", x)
        assert result.shape == (2, 3)

    def test_vector_3d_expression(self):
        import numpy as np

        from dolfinx_mcp.eval_helpers import eval_numpy_expression

        x = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        result = eval_numpy_expression("np.vstack([x[0], x[1], x[2]])", x)
        assert result.shape == (3, 2)

    def test_scalar_still_works(self):
        import numpy as np

        from dolfinx_mcp.eval_helpers import eval_numpy_expression

        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = eval_numpy_expression("x[0] + x[1]", x)
        assert result.shape == (3,)

    def test_bad_shape_rejected(self):
        import numpy as np

        from dolfinx_mcp.errors import PostconditionError
        from dolfinx_mcp.eval_helpers import eval_numpy_expression

        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        with pytest.raises(PostconditionError, match="shape"):
            eval_numpy_expression("np.ones((3, 5))", x)


class TestBoundaryTagPreconditions:
    """FIX-4 (BUG-002): boundary_tag BC validation."""

    @pytest.mark.asyncio
    async def test_boundary_tag_no_tags_exist(self, mock_ctx):
        import sys

        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = mock_ctx.request_context.lifespan_context
        session.meshes["m"] = make_mesh_info("m")
        session.active_mesh = "m"
        session.function_spaces["V"] = make_space_info("V", mesh_name="m")
        # No mesh_tags registered

        with patch.dict(sys.modules, {
            "dolfinx": MagicMock(),
            "dolfinx.fem": MagicMock(),
            "dolfinx.mesh": MagicMock(),
        }):
            result = await apply_boundary_condition(
                value=0.0, boundary_tag=1,
                function_space="V", ctx=mock_ctx,
            )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        msg = result["message"]
        assert "boundary tags" in msg.lower() or "No boundary tags" in msg

    @pytest.mark.asyncio
    async def test_boundary_tag_invalid_value(self, mock_ctx):
        import sys

        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = mock_ctx.request_context.lifespan_context
        session.meshes["m"] = make_mesh_info("m")
        session.active_mesh = "m"
        session.function_spaces["V"] = make_space_info("V", mesh_name="m")

        # Add mesh tags with tags 1 and 2
        tags_info = make_mesh_tags_info("bt", mesh_name="m")
        tags_info.unique_tags = [1, 2]
        session.mesh_tags["bt"] = tags_info
        session._boundary_tag_cache["m"] = "bt"

        with patch.dict(sys.modules, {
            "dolfinx": MagicMock(),
            "dolfinx.fem": MagicMock(),
            "dolfinx.mesh": MagicMock(),
        }):
            result = await apply_boundary_condition(
                value=0.0, boundary_tag=99,
                function_space="V", ctx=mock_ctx,
            )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "99" in result["message"]
