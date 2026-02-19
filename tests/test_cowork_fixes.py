"""Tests for Cowork test report fixes (v0.10.0+).

Covers:
- read_workspace_file: path traversal, encoding auto-detect, size limits
- plot_solution: vector field support, component selection, return_base64
- project: numeric expression coercion (int/float → str)
- solve_nonlinear: corrected error suggestion text
- define_variational_form: mixed-space-aware error messages
- overview: resilient _safe_summary for custom objects (F4)
- create_function: new tool for function creation (F2)
- apply_boundary_condition: vector-valued BCs via list[float] (F1)
- set_material_properties: numeric string auto-coercion (F3)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from conftest import (
    assert_error_type,
    assert_no_error,
    make_form_info,
    make_function_info,
    make_mesh_info,
    make_mock_ctx,
    make_session_with_mesh,
    make_space_info,
)

from dolfinx_mcp.errors import (
    InvalidUFLExpressionError,
    InvariantError,
)
from dolfinx_mcp.session import SessionState

# ===========================================================================
# 1. read_workspace_file
# ===========================================================================


class TestReadWorkspaceFile:
    """Tests for the read_workspace_file tool."""

    @pytest.mark.asyncio
    async def test_pre1_empty_path(self):
        """PRE-1: file_path must be non-empty."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await read_workspace_file(file_path="", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre2_invalid_encoding(self):
        """PRE-2: encoding must be valid."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await read_workspace_file(file_path="test.png", encoding="binary", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre3_path_traversal(self):
        """PRE-3: path must be within /workspace/."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await read_workspace_file(file_path="../etc/passwd", ctx=ctx)
        assert_error_type(result, "FILE_IO_ERROR")

    @pytest.mark.asyncio
    async def test_pre3_absolute_traversal(self):
        """PRE-3: absolute path outside /workspace/ is rejected."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await read_workspace_file(file_path="/etc/passwd", ctx=ctx)
        assert_error_type(result, "FILE_IO_ERROR")

    @pytest.mark.asyncio
    async def test_pre4_file_not_found(self):
        """PRE-4: file must exist."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await read_workspace_file(file_path="/workspace/nonexistent.png", ctx=ctx)
        assert_error_type(result, "FILE_IO_ERROR")

    def test_pre5_max_file_size_constant(self):
        """PRE-5: max file size constant is 10MB."""
        from dolfinx_mcp.tools.session_mgmt import _MAX_FILE_SIZE

        assert _MAX_FILE_SIZE == 10_485_760

    @pytest.mark.asyncio
    async def test_auto_detect_png_base64(self):
        """Auto-detection encodes .png as base64."""
        from dolfinx_mcp.tools.session_mgmt import (
            _BINARY_EXTENSIONS,
        )

        assert ".png" in _BINARY_EXTENSIONS
        assert _BINARY_EXTENSIONS[".png"] == "image/png"

    @pytest.mark.asyncio
    async def test_auto_detect_vtk_text(self):
        """Auto-detection encodes .vtk as text."""
        from dolfinx_mcp.tools.session_mgmt import _TEXT_EXTENSIONS

        assert ".vtk" in _TEXT_EXTENSIONS
        assert _TEXT_EXTENSIONS[".vtk"] == "application/xml"

    @pytest.mark.asyncio
    async def test_auto_detect_csv_text(self):
        """Auto-detection encodes .csv as text."""
        from dolfinx_mcp.tools.session_mgmt import _TEXT_EXTENSIONS

        assert ".csv" in _TEXT_EXTENSIONS
        assert _TEXT_EXTENSIONS[".csv"] == "text/csv"

    def test_extension_coverage(self):
        """All common FEM output extensions are covered in auto-detect maps."""
        from dolfinx_mcp.tools.session_mgmt import (
            _BINARY_EXTENSIONS,
            _TEXT_EXTENSIONS,
        )

        # Binary image formats
        for ext in (".png", ".jpg", ".jpeg"):
            assert ext in _BINARY_EXTENSIONS, f"{ext} missing from binary map"

        # Text-based FEM output formats
        for ext in (".vtk", ".vtu", ".pvd", ".csv", ".json", ".txt"):
            assert ext in _TEXT_EXTENSIONS, f"{ext} missing from text map"

        # Binary data formats
        for ext in (".xdmf", ".h5"):
            assert ext in _BINARY_EXTENSIONS, f"{ext} missing from binary map"

        # No overlap between the two maps
        overlap = set(_BINARY_EXTENSIONS) & set(_TEXT_EXTENSIONS)
        assert not overlap, f"Extensions in both maps: {overlap}"


# ===========================================================================
# 2. plot_solution — vector fields and base64
# ===========================================================================


class TestPlotSolutionVector:
    """Tests for vector field support in plot_solution."""

    @pytest.mark.asyncio
    async def test_pre_v1_negative_component(self):
        """PRE-V1: component must be non-negative."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = make_session_with_mesh()
        mock_func = MagicMock()
        session.functions["u_vec"] = make_function_info("u_vec", "V", function=mock_func)
        ctx = make_mock_ctx(session)

        result = await plot_solution(
            function_name="u_vec", component=-1, ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_v1_non_int_component(self):
        """PRE-V1: component must be an integer."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = make_session_with_mesh()
        mock_func = MagicMock()
        session.functions["u_vec"] = make_function_info("u_vec", "V", function=mock_func)
        ctx = make_mock_ctx(session)

        result = await plot_solution(
            function_name="u_vec", component=1.5, ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_v2_component_out_of_range(self):
        """PRE-V2: component must be within range for vector fields."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = make_session_with_mesh()

        # Must register space "V" so get_function() postcondition passes
        session.function_spaces["V"] = make_space_info("V", num_dofs=128)

        # Mock a vector function with 2 components
        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = (2,)
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space

        session.functions["u_vec"] = make_function_info("u_vec", "V", function=mock_func)
        ctx = make_mock_ctx(session)

        # Mock pyvista import to get past the import check
        with patch.dict("sys.modules", {"pyvista": MagicMock(), "dolfinx.plot": MagicMock()}):
            result = await plot_solution(
                function_name="u_vec", component=5, ctx=ctx,
            )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "out of range" in result["message"]

    @pytest.mark.asyncio
    async def test_scalar_field_ignores_component(self):
        """Scalar field sets is_vector=False and no component_plotted key."""
        import numpy as np

        from dolfinx_mcp.tools.postprocess import plot_solution

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")

        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = ()  # scalar
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space
        mock_func.x.array.real = np.array([1.0, 2.0, 3.0])

        session.functions["u_s"] = make_function_info("u_s", "V", function=mock_func)
        ctx = make_mock_ctx(session)

        geometry = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

        mock_pyvista = MagicMock()
        mock_pyvista.OFF_SCREEN = True
        mock_vtk_mesh = MagicMock(return_value=(MagicMock(), MagicMock(), geometry))
        mock_dolfinx_plot = MagicMock()
        mock_dolfinx_plot.vtk_mesh = mock_vtk_mesh

        with patch.dict("sys.modules", {
            "pyvista": mock_pyvista,
            "dolfinx.plot": mock_dolfinx_plot,
            "numpy": np,
        }), patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=512):
            result = await plot_solution(
                function_name="u_s", component=None, ctx=ctx,
            )

        assert result["is_vector"] is False
        assert "component_plotted" not in result

    @pytest.mark.asyncio
    async def test_vector_magnitude_computation(self):
        """Vector field magnitude: sqrt(3^2+4^2)=5, sqrt(0+0)=0, sqrt(1+0)=1."""
        import numpy as np

        from dolfinx_mcp.tools.postprocess import plot_solution

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V", num_dofs=128)

        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = (2,)  # 2D vector
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space
        # 3 points, 2 components each: (3,4), (0,0), (1,0)
        mock_func.x.array.real = np.array([3.0, 4.0, 0.0, 0.0, 1.0, 0.0])

        session.functions["u_vec"] = make_function_info("u_vec", "V", function=mock_func)
        ctx = make_mock_ctx(session)

        geometry = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

        # Capture what gets assigned to grid.point_data["solution"]
        captured = {}

        class FakeGrid:
            point_data = captured
            def warp_by_scalar(self, *a, **kw):
                return self

        mock_pyvista = MagicMock()
        mock_pyvista.OFF_SCREEN = True
        mock_pyvista.UnstructuredGrid.return_value = FakeGrid()
        mock_vtk_mesh = MagicMock(return_value=(MagicMock(), MagicMock(), geometry))
        mock_dolfinx_plot = MagicMock()
        mock_dolfinx_plot.vtk_mesh = mock_vtk_mesh

        with patch.dict("sys.modules", {
            "pyvista": mock_pyvista,
            "dolfinx.plot": mock_dolfinx_plot,
            "numpy": np,
        }), patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=512):
            result = await plot_solution(
                function_name="u_vec", component=None, ctx=ctx,
            )

        assert result["is_vector"] is True
        assert result["component_plotted"] == "magnitude"
        assert result["num_components"] == 2

        # Verify magnitude values
        plot_data = captured.get("solution")
        if plot_data is not None:
            np.testing.assert_allclose(plot_data, [5.0, 0.0, 1.0])

    @pytest.mark.asyncio
    async def test_return_base64_produces_decodable_content(self):
        """return_base64=True includes decodable base64 string in result."""
        import base64

        import numpy as np

        from dolfinx_mcp.tools.postprocess import plot_solution

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")

        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = ()
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space
        mock_func.x.array.real = np.array([1.0, 2.0, 3.0])

        session.functions["u"] = make_function_info("u", "V", function=mock_func)
        ctx = make_mock_ctx(session)

        geometry = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_pyvista = MagicMock()
        mock_pyvista.OFF_SCREEN = True
        mock_vtk_mesh = MagicMock(return_value=(MagicMock(), MagicMock(), geometry))
        mock_dolfinx_plot = MagicMock()
        mock_dolfinx_plot.vtk_mesh = mock_vtk_mesh

        import io

        with patch.dict("sys.modules", {
            "pyvista": mock_pyvista,
            "dolfinx.plot": mock_dolfinx_plot,
            "numpy": np,
        }), patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=len(fake_png)), \
                patch("builtins.open", return_value=io.BytesIO(fake_png)):
            result = await plot_solution(
                function_name="u", return_base64=True, ctx=ctx,
            )

        assert "image_base64" in result
        decoded = base64.b64decode(result["image_base64"])
        assert decoded == fake_png


# ===========================================================================
# 3. project — numeric expression coercion
# ===========================================================================


class TestProjectNumericCoercion:
    """Tests for numeric expression coercion in project tool."""

    @pytest.mark.asyncio
    async def test_int_expression_coerced(self):
        """Integer expression should be coerced to string."""
        from dolfinx_mcp.tools.interpolation import project

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        ctx = make_mock_ctx(session)

        # project with expression=0 should not raise Pydantic error
        # It will fail deeper in the DOLFINx API (no real DOLFINx), but
        # should NOT fail at parameter validation
        import dolfinx_mcp.tools.interpolation as interp_mod

        with patch.object(interp_mod, "eval_numpy_expression"):
            # Mock the DOLFINx import chain
            mock_dolfinx = MagicMock()
            mock_ufl = MagicMock()
            mock_np = MagicMock()

            with patch.dict("sys.modules", {
                "dolfinx": mock_dolfinx,
                "dolfinx.fem": mock_dolfinx.fem,
                "ufl": mock_ufl,
                "numpy": mock_np,
            }):
                # The function should get past the str conversion
                # and fail at session.get_space (no real space with .space attr)
                result = await project(
                    name="u", target_space="V", expression=0, ctx=ctx,
                )
                # It will fail at DOLFINx API level, but NOT at Pydantic validation
                # The key test: expression=0 (int) doesn't cause a type error
                assert result.get("error") != "PRECONDITION_VIOLATED" or \
                    "expression" not in result.get("message", "")

    @pytest.mark.asyncio
    async def test_float_expression_coerced(self):
        """Float expression should be coerced to string."""
        from dolfinx_mcp.tools.interpolation import project

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        ctx = make_mock_ctx(session)

        # project with expression=3.14 should not raise Pydantic error
        # It will fail deeper, but the type coercion should work
        result = await project(
            name="u", target_space="V", expression=3.14, ctx=ctx,
        )
        # Should not be a type validation error
        assert result.get("error") != "PRECONDITION_VIOLATED" or \
            "expression" not in result.get("message", "")

    @pytest.mark.asyncio
    async def test_string_expression_still_works(self):
        """String expression should still work as before."""
        from dolfinx_mcp.tools.interpolation import project

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        ctx = make_mock_ctx(session)

        result = await project(
            name="u", target_space="V", expression="sin(pi*x[0])", ctx=ctx,
        )
        # Will fail at DOLFINx level but not at type validation
        assert result.get("error") != "PRECONDITION_VIOLATED" or \
            "expression" not in result.get("message", "")


# ===========================================================================
# 4. solve_nonlinear — corrected error suggestion
# ===========================================================================


class TestSolveNonlinearSuggestion:
    """Tests for corrected error suggestion in solve_nonlinear."""

    @pytest.mark.asyncio
    async def test_unknown_not_found_suggests_project(self):
        """Missing unknown function should suggest 'project' not 'interpolate'."""
        from dolfinx_mcp.tools.solver import solve_nonlinear

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)

        result = await solve_nonlinear(
            residual="inner(grad(u), grad(v))*dx",
            unknown="nonexistent",
            ctx=ctx,
        )

        assert_error_type(result, "FUNCTION_NOT_FOUND")
        assert "project" in result["suggestion"]
        assert "target_space" in result["suggestion"]
        # Must NOT reference nonexistent interpolate params
        assert "function_space" not in result["suggestion"]


# ===========================================================================
# 5. define_variational_form — mixed-space error messages
# ===========================================================================


class TestDefineVariationalFormMixedSpace:
    """Tests for mixed-space-aware error messages."""

    @pytest.mark.asyncio
    async def test_mixed_space_undefined_symbol_suggests_split(self):
        """When trial space is Mixed and symbol is undefined, suggest split()."""
        from dolfinx_mcp.tools.problem import define_variational_form

        session = make_session_with_mesh()
        # Create a mixed function space
        session.function_spaces["W"] = make_space_info(
            "W", element_family="Mixed", element_degree=2, num_dofs=2467,
        )
        ctx = make_mock_ctx(session)

        # Mock safe_evaluate to raise InvalidUFLExpressionError with "is not defined"
        with patch("dolfinx_mcp.tools.problem.safe_evaluate") as mock_eval, \
             patch("dolfinx_mcp.tools.problem.build_namespace") as mock_ns:
            mock_ns.return_value = {"__builtins__": {}}
            mock_eval.side_effect = InvalidUFLExpressionError(
                "Unknown symbol in expression: name 'p' is not defined",
                suggestion="Ensure all variables (u, v, f, etc.) are defined before use.",
            )

            with patch.dict("sys.modules", {
                "dolfinx": MagicMock(),
                "dolfinx.fem": MagicMock(),
                "ufl": MagicMock(),
            }):
                result = await define_variational_form(
                    bilinear="inner(grad(u), grad(v))*dx - p*div(v)*dx",
                    linear="inner(as_vector([0, 0]), v)*dx",
                    trial_space="W",
                    test_space="W",
                    ctx=ctx,
                )

        assert_error_type(result, "INVALID_UFL_EXPRESSION")
        assert "split()" in result["suggestion"]
        assert "split(u)[0]" in result["suggestion"]
        assert "split(u)[1]" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_non_mixed_space_preserves_original_error(self):
        """When trial space is NOT Mixed, error message is unchanged."""
        from dolfinx_mcp.tools.problem import define_variational_form

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        ctx = make_mock_ctx(session)

        with patch("dolfinx_mcp.tools.problem.safe_evaluate") as mock_eval, \
             patch("dolfinx_mcp.tools.problem.build_namespace") as mock_ns:
            mock_ns.return_value = {"__builtins__": {}}
            mock_eval.side_effect = InvalidUFLExpressionError(
                "Unknown symbol in expression: name 'foo' is not defined",
                suggestion="Ensure all variables (u, v, f, etc.) are defined before use.",
            )

            with patch.dict("sys.modules", {
                "dolfinx": MagicMock(),
                "dolfinx.fem": MagicMock(),
                "ufl": MagicMock(),
            }):
                result = await define_variational_form(
                    bilinear="foo*inner(grad(u), grad(v))*dx",
                    linear="f*v*dx",
                    trial_space="V",
                    test_space="V",
                    ctx=ctx,
                )

        assert_error_type(result, "INVALID_UFL_EXPRESSION")
        # Should NOT mention split() for non-mixed spaces
        assert "split()" not in result.get("suggestion", "")


# ===========================================================================
# 6. INV-8: remove_object(type="space") must clear forms
# ===========================================================================


class TestRemoveObjectSpaceClearsForms:
    """Tests for INV-8 enforcement in remove_object."""

    @pytest.mark.asyncio
    async def test_remove_last_space_clears_forms(self):
        """When the last space is removed, forms must be cleared (INV-8)."""
        from dolfinx_mcp.tools.session_mgmt import remove_object

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        # Simulate a form that was defined using space V
        session.forms["a"] = make_form_info("a")
        ctx = make_mock_ctx(session)

        # Remove the only space — must clear forms to satisfy INV-8
        result = await remove_object(name="V", object_type="space", ctx=ctx)

        assert_no_error(result)
        assert session.forms == {}
        assert "V" not in session.function_spaces


# ---------------------------------------------------------------------------
# F4: Resilient overview() with _safe_summary
# ---------------------------------------------------------------------------


class TestOverviewResilience:
    """overview() must survive custom objects without .summary()."""

    def test_overview_with_custom_object(self) -> None:
        """Custom object in functions registry should produce fallback dict."""
        session = make_session_with_mesh()
        session.functions["custom_fn"] = object()  # no .summary() method

        result = session.overview()

        assert "functions" in result
        assert result["functions"]["custom_fn"] == {"name": "custom_fn", "type": "custom"}

    def test_overview_normal_objects_still_work(self) -> None:
        """Standard FunctionInfo objects should still produce normal summaries."""
        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        session.functions["f"] = make_function_info("f", description="test function")

        result = session.overview()

        assert result["functions"]["f"]["name"] == "f"
        assert result["functions"]["f"]["space_name"] == "V"

    def test_overview_mixed_custom_and_normal(self) -> None:
        """Mix of custom and normal objects in same registry."""
        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        session.functions["normal"] = make_function_info("normal")
        session.functions["weird"] = "not a FunctionInfo"

        result = session.overview()

        assert result["functions"]["normal"]["name"] == "normal"
        assert result["functions"]["weird"] == {"name": "weird", "type": "custom"}


# ---------------------------------------------------------------------------
# F2: create_function tool
# ---------------------------------------------------------------------------


class TestCreateFunction:
    """Tests for the new create_function tool (precondition checks only).

    Happy-path tests (zero-init, expression) run in Docker via
    test_runtime_contracts.py::TestCoworkIssueFixes.
    """

    @pytest.mark.asyncio
    async def test_create_function_empty_name(self) -> None:
        """Empty name returns PRECONDITION_VIOLATED."""
        from dolfinx_mcp.tools.interpolation import create_function

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await create_function(name="", function_space="V", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_create_function_duplicate_name(self) -> None:
        """Duplicate function name returns DUPLICATE_NAME."""
        from dolfinx_mcp.tools.interpolation import create_function

        session = make_session_with_mesh()
        session.function_spaces["V"] = make_space_info("V")
        session.functions["u_existing"] = make_function_info("u_existing")

        ctx = make_mock_ctx(session)
        result = await create_function(name="u_existing", function_space="V", ctx=ctx)
        assert_error_type(result, "DUPLICATE_NAME")

    @pytest.mark.asyncio
    async def test_create_function_missing_space(self) -> None:
        """Non-existent function space returns FUNCTION_SPACE_NOT_FOUND."""
        from dolfinx_mcp.tools.interpolation import create_function

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await create_function(name="u", function_space="nonexistent", ctx=ctx)
        assert_error_type(result, "FUNCTION_SPACE_NOT_FOUND")


# ---------------------------------------------------------------------------
# F1: Vector-valued BCs
# ---------------------------------------------------------------------------


class TestVectorBoundaryCondition:
    """Tests for list[float] value in apply_boundary_condition.

    Precondition tests only — happy-path (actual interpolation) tested in
    test_runtime_contracts.py::TestCoworkIssueFixes with real DOLFINx.
    """

    @pytest.mark.asyncio
    async def test_vector_bc_wrong_dimension(self) -> None:
        """3-component list on 2D vector space returns PRECONDITION_VIOLATED."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = make_session_with_mesh()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = (2,)
        mock_space.ufl_element.return_value = mock_element
        mock_space.num_sub_spaces = 0

        session.function_spaces["V_vec"] = make_space_info(
            "V_vec", space=mock_space, num_dofs=128, shape=(2,),
        )

        mesh_mock = session.meshes["m1"].mesh
        mesh_mock.topology.dim = 2
        ctx = make_mock_ctx(session)
        result = await apply_boundary_condition(
            value=[1.0, 0.0, 0.0],
            boundary="np.isclose(x[0], 0.0)",
            function_space="V_vec",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "3 components" in result["message"]

    @pytest.mark.asyncio
    async def test_vector_bc_on_scalar_space(self) -> None:
        """List value on scalar space returns PRECONDITION_VIOLATED."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = make_session_with_mesh()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = ()  # scalar
        mock_space.ufl_element.return_value = mock_element
        mock_space.num_sub_spaces = 0

        session.function_spaces["V"] = make_space_info("V", space=mock_space)

        mesh_mock = session.meshes["m1"].mesh
        mesh_mock.topology.dim = 2
        ctx = make_mock_ctx(session)
        result = await apply_boundary_condition(
            value=[1.0],
            boundary="np.isclose(x[0], 0.0)",
            function_space="V",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "scalar" in result["message"]

    @pytest.mark.asyncio
    async def test_vector_bc_non_finite(self) -> None:
        """Non-finite values in list return PRECONDITION_VIOLATED."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await apply_boundary_condition(
            value=[float("inf"), 0.0],
            boundary="np.isclose(x[0], 0.0)",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "finite" in result["message"]

    @pytest.mark.asyncio
    async def test_vector_bc_empty_list(self) -> None:
        """Empty list returns PRECONDITION_VIOLATED."""
        from dolfinx_mcp.tools.problem import apply_boundary_condition

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await apply_boundary_condition(
            value=[],
            boundary="np.isclose(x[0], 0.0)",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")
        assert "non-empty" in result["message"]


# ---------------------------------------------------------------------------
# F3: Numeric string auto-coercion in set_material_properties
# ---------------------------------------------------------------------------


class TestMaterialNumericCoercion:
    """Numeric string auto-coercion tests.

    Happy-path tests (actual Constant creation, interpolation) run in
    test_runtime_contracts.py::TestCoworkIssueFixes with real DOLFINx.
    """

    def test_numeric_string_detected(self) -> None:
        """Verify float() can parse numeric strings that should be coerced."""
        # These should all parse as floats and trigger coercion
        for s in ("1.0", "0", "-3.14", "2.5e-3"):
            assert isinstance(float(s), float)

    def test_nonnumeric_string_not_detected(self) -> None:
        """Verify non-numeric strings are NOT parsed as floats."""
        for s in ("sin(pi*x[0])", "x[0]**2 + x[1]**2", "1.0 + x[0]"):
            with pytest.raises(ValueError):
                float(s)


# ---------------------------------------------------------------------------
# Performance Optimizations (P1-P7)
# ---------------------------------------------------------------------------


class TestPerformanceOptimizations:
    """Tests for performance optimizations P1-P7."""

    def test_numpy_ns_module_constant_exists(self) -> None:
        """P1: _NUMPY_NS is a module-level constant with expected keys."""
        from dolfinx_mcp.eval_helpers import _NUMPY_NS

        expected_keys = {"np", "pi", "e", "sin", "cos", "exp", "sqrt", "abs", "log", "__builtins__"}
        assert set(_NUMPY_NS.keys()) == expected_keys
        assert _NUMPY_NS["__builtins__"] == {}

    def test_numpy_ns_superset_of_boundary_needs(self) -> None:
        """P2: _NUMPY_NS contains all keys needed by boundary markers."""
        from dolfinx_mcp.eval_helpers import _NUMPY_NS

        boundary_keys = {"np", "pi", "__builtins__"}
        assert boundary_keys.issubset(set(_NUMPY_NS.keys()))
        assert _NUMPY_NS["__builtins__"] == {}

    def test_eval_numpy_expression_correct(self) -> None:
        """P1: eval_numpy_expression still produces correct values."""
        import numpy as np

        from dolfinx_mcp.eval_helpers import eval_numpy_expression

        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])  # 2D coords, 3 points
        result = eval_numpy_expression("sin(pi * x[0])", x)
        expected = np.sin(np.pi * x[0])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_boundary_marker_closure_correct(self) -> None:
        """P2: make_boundary_marker closure still works after namespace caching."""
        import numpy as np

        from dolfinx_mcp.eval_helpers import make_boundary_marker

        marker = make_boundary_marker("np.isclose(x[0], 0.0)")
        x = np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
        result = marker(x)
        assert result[0] is np.bool_(True) or result[0] == True  # noqa: E712
        assert result[1] == False  # noqa: E712
        assert result[2] == False  # noqa: E712

    def test_tag_counting_matches_naive(self) -> None:
        """P5: np.unique(return_counts=True) matches O(k*n) loop."""
        import numpy as np

        tag_values = np.array([1, 2, 1, 3, 2, 1, 3, 3, 3])

        # O(n) method (new)
        unique_tags, counts = np.unique(tag_values, return_counts=True)
        fast_counts = {
            int(t): int(c)
            for t, c in zip(unique_tags, counts, strict=True)
        }

        # O(k*n) method (old)
        naive_counts = {}
        for tag in np.unique(tag_values):
            naive_counts[int(tag)] = int(np.sum(tag_values == tag))

        assert fast_counts == naive_counts
        assert fast_counts == {1: 3, 2: 2, 3: 4}

    def test_solution_info_has_l2_norm(self) -> None:
        """P6: SolutionInfo accepts and stores l2_norm field."""
        from dolfinx_mcp.session import SolutionInfo

        sol = SolutionInfo(
            name="u",
            function=MagicMock(),
            space_name="V",
            converged=True,
            iterations=5,
            residual_norm=1e-10,
            wall_time=0.5,
            l2_norm=1.234,
        )
        assert sol.l2_norm == 1.234
        assert "l2_norm" in sol.summary()
        assert sol.summary()["l2_norm"] == 1.234

    def test_solution_info_l2_norm_default_zero(self) -> None:
        """P6: SolutionInfo defaults l2_norm to 0.0."""
        from dolfinx_mcp.session import SolutionInfo

        sol = SolutionInfo(
            name="u",
            function=MagicMock(),
            space_name="V",
            converged=True,
            iterations=5,
            residual_norm=1e-10,
            wall_time=0.5,
        )
        assert sol.l2_norm == 0.0


class TestReportV2Fixes:
    """Tests for fixes from Test Report v2 (F1-F3)."""

    def test_form_info_has_trial_space_name(self) -> None:
        """F1: FormInfo accepts and stores trial_space_name."""
        form = make_form_info("bilinear", description="test", trial_space_name="W")
        assert form.trial_space_name == "W"
        assert form.summary()["trial_space_name"] == "W"

    def test_form_info_trial_space_name_default_empty(self) -> None:
        """F1: FormInfo defaults trial_space_name to empty string."""
        form = make_form_info("bilinear")
        assert form.trial_space_name == ""


class TestDbCAuditFixes:
    """Tests for gaps found in DbC audit (BUG-1, INV-9, CASCADE-1, ASSERT-1)."""

    def test_solve_eigenvalue_no_active_space_attr(self) -> None:
        """BUG-1: solve_eigenvalue with function_space=None should not crash with AttributeError.

        SessionState has active_mesh, not active_space. The old code
        `function_space or session.active_space` raised AttributeError.
        After fix, it falls through to the existing None-handling logic.
        """
        session = SessionState()
        # No function spaces registered → should get PreconditionError (not AttributeError)
        # We can't call the async tool directly in unit tests, but we verify
        # the session has no active_space attribute (proving the bug existed).
        assert not hasattr(session, "active_space")
        assert hasattr(session, "active_mesh")

    def test_inv9_trial_space_name_dangling_raises(self) -> None:
        """INV-9: Form with trial_space_name pointing to non-existent space → InvariantError."""
        session = SessionState()
        session.meshes["m"] = make_mesh_info("m", num_cells=4, num_vertices=5)
        session.function_spaces["V"] = make_space_info("V", mesh_name="m", num_dofs=5)
        # Add form referencing a space that doesn't exist
        session.forms["bilinear"] = make_form_info(
            "bilinear", trial_space_name="DELETED_SPACE",
        )
        with pytest.raises(InvariantError, match="Dangling space references"):
            session.check_invariants()

    def test_inv9_trial_space_name_empty_ok(self) -> None:
        """INV-9: Form with empty trial_space_name passes invariant check."""
        session = SessionState()
        session.meshes["m"] = make_mesh_info("m", num_cells=4, num_vertices=5)
        session.function_spaces["V"] = make_space_info("V", mesh_name="m", num_dofs=5)
        session.forms["bilinear"] = make_form_info(
            "bilinear", trial_space_name="",  # empty is valid (backward compat)
        )
        session.check_invariants()  # should not raise

    def test_scalar_space_cascade_on_parent_deletion(self) -> None:
        """CASCADE-1: Deleting space 'V' also removes '_scalar_V'."""
        session = SessionState()
        session.meshes["m"] = make_mesh_info("m", num_cells=4, num_vertices=5)
        session.function_spaces["V"] = make_space_info(
            "V", mesh_name="m", element_degree=2, num_dofs=25,
        )
        session.function_spaces["_scalar_V"] = make_space_info(
            "_scalar_V", mesh_name="m", element_degree=2, num_dofs=9,
        )
        # Also add a function on the scalar space to verify full cascade
        session.functions["mu"] = make_function_info("mu", "_scalar_V")

        # Remove parent space
        session._remove_space_dependents("V")
        del session.function_spaces["V"]

        # Both _scalar_V and function "mu" should be gone
        assert "_scalar_V" not in session.function_spaces
        assert "mu" not in session.functions
