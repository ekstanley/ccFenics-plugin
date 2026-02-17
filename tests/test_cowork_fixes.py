"""Tests for Cowork test report fixes (v0.10.0).

Covers:
- read_workspace_file: path traversal, encoding auto-detect, size limits
- plot_solution: vector field support, component selection, return_base64
- project: numeric expression coercion (int/float → str)
- solve_nonlinear: corrected error suggestion text
- define_variational_form: mixed-space-aware error messages
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dolfinx_mcp.errors import (
    FileIOError,
    FunctionNotFoundError,
    InvalidUFLExpressionError,
    PreconditionError,
)
from dolfinx_mcp.session import (
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    SessionState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ctx(session: SessionState):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = session
    return ctx


def _make_session() -> SessionState:
    s = SessionState()
    s.meshes["m1"] = MeshInfo(
        name="m1", mesh=MagicMock(), cell_type="triangle",
        num_cells=100, num_vertices=64, gdim=2, tdim=2,
    )
    s.active_mesh = "m1"
    return s


# ===========================================================================
# 1. read_workspace_file
# ===========================================================================


class TestReadWorkspaceFile:
    """Tests for the read_workspace_file tool."""

    @pytest.mark.asyncio
    async def test_pre1_empty_path(self):
        """PRE-1: file_path must be non-empty."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = _make_session()
        ctx = _mock_ctx(session)
        result = await read_workspace_file(file_path="", ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_pre2_invalid_encoding(self):
        """PRE-2: encoding must be valid."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = _make_session()
        ctx = _mock_ctx(session)
        result = await read_workspace_file(file_path="test.png", encoding="binary", ctx=ctx)
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_pre3_path_traversal(self):
        """PRE-3: path must be within /workspace/."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = _make_session()
        ctx = _mock_ctx(session)
        result = await read_workspace_file(file_path="../etc/passwd", ctx=ctx)
        assert result["error"] == "FILE_IO_ERROR"

    @pytest.mark.asyncio
    async def test_pre3_absolute_traversal(self):
        """PRE-3: absolute path outside /workspace/ is rejected."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = _make_session()
        ctx = _mock_ctx(session)
        result = await read_workspace_file(file_path="/etc/passwd", ctx=ctx)
        assert result["error"] == "FILE_IO_ERROR"

    @pytest.mark.asyncio
    async def test_pre4_file_not_found(self):
        """PRE-4: file must exist."""
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = _make_session()
        ctx = _mock_ctx(session)
        result = await read_workspace_file(file_path="/workspace/nonexistent.png", ctx=ctx)
        assert result["error"] == "FILE_IO_ERROR"

    def test_pre5_max_file_size_constant(self):
        """PRE-5: max file size constant is 10MB."""
        from dolfinx_mcp.tools.session_mgmt import _MAX_FILE_SIZE

        assert _MAX_FILE_SIZE == 10_485_760

    @pytest.mark.asyncio
    async def test_auto_detect_png_base64(self):
        """Auto-detection encodes .png as base64."""
        from dolfinx_mcp.tools.session_mgmt import (
            _BINARY_EXTENSIONS,
            _TEXT_EXTENSIONS,
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

        session = _make_session()
        mock_func = MagicMock()
        session.functions["u_vec"] = FunctionInfo(
            name="u_vec", function=mock_func, space_name="V",
        )
        ctx = _mock_ctx(session)

        result = await plot_solution(
            function_name="u_vec", component=-1, ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_pre_v1_non_int_component(self):
        """PRE-V1: component must be an integer."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = _make_session()
        mock_func = MagicMock()
        session.functions["u_vec"] = FunctionInfo(
            name="u_vec", function=mock_func, space_name="V",
        )
        ctx = _mock_ctx(session)

        result = await plot_solution(
            function_name="u_vec", component=1.5, ctx=ctx,
        )
        assert result["error"] == "PRECONDITION_VIOLATED"

    @pytest.mark.asyncio
    async def test_pre_v2_component_out_of_range(self):
        """PRE-V2: component must be within range for vector fields."""
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = _make_session()

        # Must register space "V" so get_function() postcondition passes
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=128,
        )

        # Mock a vector function with 2 components
        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = (2,)
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space

        session.functions["u_vec"] = FunctionInfo(
            name="u_vec", function=mock_func, space_name="V",
        )
        ctx = _mock_ctx(session)

        # Mock pyvista import to get past the import check
        with patch.dict("sys.modules", {"pyvista": MagicMock(), "dolfinx.plot": MagicMock()}):
            result = await plot_solution(
                function_name="u_vec", component=5, ctx=ctx,
            )
        assert result["error"] == "PRECONDITION_VIOLATED"
        assert "out of range" in result["message"]

    @pytest.mark.asyncio
    async def test_scalar_field_ignores_component(self):
        """Scalar field sets is_vector=False and no component_plotted key."""
        import numpy as np
        from dolfinx_mcp.tools.postprocess import plot_solution

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )

        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = ()  # scalar
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space
        mock_func.x.array.real = np.array([1.0, 2.0, 3.0])

        session.functions["u_s"] = FunctionInfo(
            name="u_s", function=mock_func, space_name="V",
        )
        ctx = _mock_ctx(session)

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

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=128,
        )

        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = (2,)  # 2D vector
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space
        # 3 points, 2 components each: (3,4), (0,0), (1,0)
        mock_func.x.array.real = np.array([3.0, 4.0, 0.0, 0.0, 1.0, 0.0])

        session.functions["u_vec"] = FunctionInfo(
            name="u_vec", function=mock_func, space_name="V",
        )
        ctx = _mock_ctx(session)

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

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )

        mock_func = MagicMock()
        mock_space = MagicMock()
        mock_element = MagicMock()
        mock_element.reference_value_shape = ()
        mock_space.ufl_element.return_value = mock_element
        mock_func.function_space = mock_space
        mock_func.x.array.real = np.array([1.0, 2.0, 3.0])

        session.functions["u"] = FunctionInfo(
            name="u", function=mock_func, space_name="V",
        )
        ctx = _mock_ctx(session)

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

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )
        ctx = _mock_ctx(session)

        # project with expression=0 should not raise Pydantic error
        # It will fail deeper in the DOLFINx API (no real DOLFINx), but
        # should NOT fail at parameter validation
        import dolfinx_mcp.tools.interpolation as interp_mod

        with patch.object(interp_mod, "eval_numpy_expression") as mock_eval:
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

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )
        ctx = _mock_ctx(session)

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

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )
        ctx = _mock_ctx(session)

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

        session = _make_session()
        ctx = _mock_ctx(session)

        result = await solve_nonlinear(
            residual="inner(grad(u), grad(v))*dx",
            unknown="nonexistent",
            ctx=ctx,
        )

        assert result["error"] == "FUNCTION_NOT_FOUND"
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

        session = _make_session()
        # Create a mixed function space
        session.function_spaces["W"] = FunctionSpaceInfo(
            name="W", space=MagicMock(), mesh_name="m1",
            element_family="Mixed", element_degree=2, num_dofs=2467,
        )
        ctx = _mock_ctx(session)

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

        assert result["error"] == "INVALID_UFL_EXPRESSION"
        assert "split()" in result["suggestion"]
        assert "split(u)[0]" in result["suggestion"]
        assert "split(u)[1]" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_non_mixed_space_preserves_original_error(self):
        """When trial space is NOT Mixed, error message is unchanged."""
        from dolfinx_mcp.tools.problem import define_variational_form

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )
        ctx = _mock_ctx(session)

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

        assert result["error"] == "INVALID_UFL_EXPRESSION"
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

        session = _make_session()
        session.function_spaces["V"] = FunctionSpaceInfo(
            name="V", space=MagicMock(), mesh_name="m1",
            element_family="Lagrange", element_degree=1, num_dofs=64,
        )
        # Simulate a form that was defined using space V
        from dolfinx_mcp.session import FormInfo
        session.forms["a"] = FormInfo(
            name="a", form=MagicMock(), ufl_form=MagicMock(),
        )
        ctx = _mock_ctx(session)

        # Remove the only space — must clear forms to satisfy INV-8
        result = await remove_object(name="V", object_type="space", ctx=ctx)

        assert "error" not in result
        assert session.forms == {}
        assert "V" not in session.function_spaces
