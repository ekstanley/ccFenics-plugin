"""Tests for workspace data transfer tools (v0.10.2).

Covers:
- list_workspace_files: pattern validation, traversal prevention
- bundle_workspace_files: empty input, non-zip, traversal, size limit
- generate_report: empty title, traversal, helper functions
- _html_table: HTML generation and escaping
- _embed_image: file embedding with size limits
- _build_report_html: report assembly with session data
"""

from __future__ import annotations

import pytest
from conftest import (
    assert_error_type,
    make_mock_ctx,
    make_populated_session,
    make_session_with_mesh,
)

from dolfinx_mcp.session import SessionState

# ===========================================================================
# 1. list_workspace_files
# ===========================================================================


class TestListWorkspaceFiles:
    """Tests for the list_workspace_files tool."""

    @pytest.mark.asyncio
    async def test_pre1_empty_pattern(self):
        """PRE-1: pattern must be non-empty."""
        from dolfinx_mcp.tools.session_mgmt import list_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await list_workspace_files(pattern="", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre1_whitespace_pattern(self):
        """PRE-1: whitespace-only pattern is rejected."""
        from dolfinx_mcp.tools.session_mgmt import list_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await list_workspace_files(pattern="   ", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_traversal_dotdot(self):
        """Pattern containing '..' is rejected to prevent path traversal."""
        from dolfinx_mcp.tools.session_mgmt import list_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await list_workspace_files(pattern="../*", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_traversal_embedded_dotdot(self):
        """Pattern with embedded '..' is also rejected."""
        from dolfinx_mcp.tools.session_mgmt import list_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await list_workspace_files(pattern="subdir/../../../etc/*", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_empty_workspace(self, tmp_path, monkeypatch):
        """Empty or nonexistent workspace returns zero files."""
        from dolfinx_mcp.tools.session_mgmt import list_workspace_files

        # On host, /workspace/ doesn't exist, so glob returns nothing
        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await list_workspace_files(pattern="*.nonexistent_ext_xyz", ctx=ctx)
        # Either returns empty list or FILE_IO_ERROR if /workspace/ doesn't exist
        if "error" not in result:
            assert result["num_files"] == 0
            assert result["total_size_bytes"] == 0
            assert result["files"] == []


# ===========================================================================
# 2. bundle_workspace_files
# ===========================================================================


class TestBundleWorkspaceFiles:
    """Tests for the bundle_workspace_files tool."""

    @pytest.mark.asyncio
    async def test_pre1_empty_file_paths(self):
        """PRE-1: file_paths must be non-empty."""
        from dolfinx_mcp.tools.session_mgmt import bundle_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await bundle_workspace_files(file_paths=[], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre3_non_zip_extension(self):
        """PRE-3: archive_name must end with .zip."""
        from dolfinx_mcp.tools.session_mgmt import bundle_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await bundle_workspace_files(
            file_paths=["test.txt"],
            archive_name="bundle.tar.gz",
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_empty_entry(self):
        """Empty string in file_paths is rejected."""
        from dolfinx_mcp.tools.session_mgmt import bundle_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await bundle_workspace_files(file_paths=[""], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_whitespace_entry(self):
        """Whitespace-only entry in file_paths is rejected."""
        from dolfinx_mcp.tools.session_mgmt import bundle_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await bundle_workspace_files(file_paths=["   "], ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre_path_traversal(self):
        """Path traversal outside /workspace/ is rejected."""
        from dolfinx_mcp.tools.session_mgmt import bundle_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await bundle_workspace_files(
            file_paths=["../etc/passwd"],
            ctx=ctx,
        )
        assert_error_type(result, "FILE_IO_ERROR")

    @pytest.mark.asyncio
    async def test_pre_archive_name_traversal(self):
        """archive_name with path traversal is rejected (Fix 1)."""
        from dolfinx_mcp.tools.session_mgmt import bundle_workspace_files

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await bundle_workspace_files(
            file_paths=["test.txt"],
            archive_name="../evil.zip",
            ctx=ctx,
        )
        assert_error_type(result, "FILE_IO_ERROR")

    def test_bundle_size_constant(self):
        """MAX_BUNDLE_SIZE is 50 MB."""
        from dolfinx_mcp.tools.session_mgmt import _MAX_BUNDLE_SIZE

        assert _MAX_BUNDLE_SIZE == 52_428_800


# ===========================================================================
# 3. generate_report
# ===========================================================================


class TestGenerateReport:
    """Tests for the generate_report tool."""

    @pytest.mark.asyncio
    async def test_pre1_empty_title(self):
        """PRE-1: title must be non-empty."""
        from dolfinx_mcp.tools.session_mgmt import generate_report

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await generate_report(title="", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre2_empty_output_file(self):
        """PRE-2: output_file must be non-empty."""
        from dolfinx_mcp.tools.session_mgmt import generate_report

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await generate_report(output_file="", ctx=ctx)
        assert_error_type(result, "PRECONDITION_VIOLATED")

    @pytest.mark.asyncio
    async def test_pre2_output_file_traversal(self):
        """PRE-2: output_file with path traversal is rejected."""
        from dolfinx_mcp.tools.session_mgmt import generate_report

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await generate_report(
            output_file="../report.html",
            ctx=ctx,
        )
        assert_error_type(result, "FILE_IO_ERROR")

    @pytest.mark.asyncio
    async def test_pre3_plot_path_traversal(self):
        """PRE-3: plot_files paths must stay within /workspace/."""
        from dolfinx_mcp.tools.session_mgmt import generate_report

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await generate_report(
            plot_files=["../etc/passwd"],
            ctx=ctx,
        )
        assert_error_type(result, "FILE_IO_ERROR")

    @pytest.mark.asyncio
    async def test_pre3_plot_file_not_found(self):
        """PRE-3: nonexistent plot file is rejected."""
        from dolfinx_mcp.tools.session_mgmt import generate_report

        session = make_session_with_mesh()
        ctx = make_mock_ctx(session)
        result = await generate_report(
            plot_files=["/workspace/nonexistent.png"],
            ctx=ctx,
        )
        assert_error_type(result, "PRECONDITION_VIOLATED")


# ===========================================================================
# 4. _html_table helper
# ===========================================================================


class TestHtmlTable:
    """Tests for the _html_table private helper."""

    def test_basic_table(self):
        """Generates a valid HTML table from headers and rows."""
        from dolfinx_mcp.tools.session_mgmt import _html_table

        result = _html_table(["Name", "Value"], [["alpha", "1"], ["beta", "2"]])
        assert "<table>" in result
        assert "<th>Name</th>" in result
        assert "<th>Value</th>" in result
        assert "<td>alpha</td>" in result
        assert "<td>2</td>" in result
        assert "</table>" in result

    def test_html_escaping(self):
        """HTML special characters are escaped to prevent XSS."""
        from dolfinx_mcp.tools.session_mgmt import _html_table

        result = _html_table(
            ["Col"],
            [["<script>alert('xss')</script>"]],
        )
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_empty_rows(self):
        """Table with headers but no rows produces valid HTML."""
        from dolfinx_mcp.tools.session_mgmt import _html_table

        result = _html_table(["A", "B"], [])
        assert "<thead>" in result
        assert "<tbody>" in result
        assert "</table>" in result


# ===========================================================================
# 5. _embed_image helper
# ===========================================================================


class TestEmbedImage:
    """Tests for the _embed_image private helper."""

    def test_missing_file(self):
        """Returns empty string for nonexistent file."""
        from dolfinx_mcp.tools.session_mgmt import _embed_image

        result = _embed_image("/nonexistent/path/to/image.png")
        assert result == ""

    def test_oversized_file(self, tmp_path):
        """Returns empty string when file exceeds max_bytes."""
        from dolfinx_mcp.tools.session_mgmt import _embed_image

        img = tmp_path / "big.png"
        img.write_bytes(b"\x00" * 200)
        result = _embed_image(str(img), max_bytes=50)
        assert result == ""

    def test_valid_png(self, tmp_path):
        """Embeds a valid PNG file as a base64 data URI."""
        from dolfinx_mcp.tools.session_mgmt import _embed_image

        img = tmp_path / "test.png"
        # Write minimal bytes (not a real PNG, but _embed_image doesn't validate)
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        result = _embed_image(str(img))
        assert result.startswith('<img src="data:image/png;base64,')
        assert 'alt="test.png"' in result

    def test_jpeg_mime_type(self, tmp_path):
        """JPEG files get the correct MIME type."""
        from dolfinx_mcp.tools.session_mgmt import _embed_image

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        result = _embed_image(str(img))
        assert "data:image/jpeg;base64," in result

    def test_xss_filename_escaped(self, tmp_path):
        """Filenames with HTML chars are escaped in alt attribute (Fix 2)."""
        from dolfinx_mcp.tools.session_mgmt import _embed_image

        # Filename with double-quote that could break HTML attribute
        img = tmp_path / 'test"onload=alert(1).png'
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)
        result = _embed_image(str(img))
        assert result != ""
        # The double-quote must be escaped
        assert 'alt="test&quot;onload=alert(1).png"' in result
        # Raw unescaped filename must NOT appear
        assert 'alt="test"onload' not in result


# ===========================================================================
# 6. _build_report_html helper
# ===========================================================================


class TestBuildReportHtml:
    """Tests for the _build_report_html private helper."""

    def test_minimal_report(self):
        """Minimal report with all sections disabled produces valid HTML."""
        from dolfinx_mcp.tools.session_mgmt import _build_report_html

        session = SessionState()
        html = _build_report_html(
            session,
            title="Test Report",
            include_plots=False,
            include_solver_info=False,
            include_mesh_info=False,
            include_session_state=False,
            plot_data=[],
        )
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Report</title>" in html
        assert "DOLFINx MCP v" in html
        assert "</html>" in html

    def test_report_with_mesh_section(self):
        """Report includes mesh table when session has meshes."""
        from dolfinx_mcp.tools.session_mgmt import _build_report_html

        session = make_populated_session()
        html = _build_report_html(
            session,
            title="Mesh Report",
            include_plots=False,
            include_solver_info=False,
            include_mesh_info=True,
            include_session_state=False,
            plot_data=[],
        )
        assert "<h2>Mesh Information</h2>" in html
        assert "<table>" in html

    def test_report_with_solver_section(self):
        """Report includes solver table when session has solutions."""
        from dolfinx_mcp.tools.session_mgmt import _build_report_html

        session = make_populated_session()
        html = _build_report_html(
            session,
            title="Solver Report",
            include_plots=False,
            include_solver_info=True,
            include_mesh_info=False,
            include_session_state=False,
            plot_data=[],
        )
        assert "<h2>Solver Diagnostics</h2>" in html

    def test_report_with_plots(self):
        """Report embeds provided plot data."""
        from dolfinx_mcp.tools.session_mgmt import _build_report_html

        session = SessionState()
        plot_data = [("result.png", '<img src="data:image/png;base64,abc">')]
        html = _build_report_html(
            session,
            title="Plot Report",
            include_plots=True,
            include_solver_info=False,
            include_mesh_info=False,
            include_session_state=False,
            plot_data=plot_data,
        )
        assert "<h2>Plots</h2>" in html
        assert "result.png" in html
        assert "data:image/png;base64,abc" in html

    def test_report_session_state_section(self):
        """Report includes session state JSON when enabled."""
        from dolfinx_mcp.tools.session_mgmt import _build_report_html

        session = make_populated_session()
        html = _build_report_html(
            session,
            title="State Report",
            include_plots=False,
            include_solver_info=False,
            include_mesh_info=False,
            include_session_state=True,
            plot_data=[],
        )
        assert "<h2>Session State</h2>" in html
        assert "<pre>" in html

    def test_report_title_escaping(self):
        """HTML special chars in title are escaped."""
        from dolfinx_mcp.tools.session_mgmt import _build_report_html

        session = SessionState()
        html = _build_report_html(
            session,
            title="<script>alert('xss')</script>",
            include_plots=False,
            include_solver_info=False,
            include_mesh_info=False,
            include_session_state=False,
            plot_data=[],
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ===========================================================================
# 7. Shared infrastructure
# ===========================================================================


class TestValidateWorkspacePath:
    """Tests for the shared validate_workspace_path validator."""

    def test_valid_simple_filename(self):
        """Simple filename resolves within /workspace/."""
        from dolfinx_mcp.tools._validators import validate_workspace_path

        result = validate_workspace_path("report.html")
        assert result.startswith("/workspace/")
        assert result.endswith("report.html")

    def test_valid_subdirectory(self):
        """Subdirectory paths resolve within /workspace/."""
        from dolfinx_mcp.tools._validators import validate_workspace_path

        result = validate_workspace_path("results/output.vtk")
        assert result.startswith("/workspace/")

    def test_traversal_rejected(self):
        """Path traversal outside /workspace/ raises FileIOError."""
        from dolfinx_mcp.errors import FileIOError
        from dolfinx_mcp.tools._validators import validate_workspace_path

        with pytest.raises(FileIOError, match="within /workspace"):
            validate_workspace_path("../etc/passwd")

    def test_absolute_traversal_rejected(self):
        """Absolute path outside /workspace/ raises FileIOError."""
        from dolfinx_mcp.errors import FileIOError
        from dolfinx_mcp.tools._validators import validate_workspace_path

        with pytest.raises(FileIOError, match="within /workspace"):
            validate_workspace_path("/etc/passwd")

    def test_workspace_boundary_strict(self):
        """Path like /workspace2/... is rejected (Fix 3)."""
        from dolfinx_mcp.errors import FileIOError
        from dolfinx_mcp.tools._validators import validate_workspace_path

        # /workspace2 starts with "/workspace" but is NOT inside /workspace/
        with pytest.raises(FileIOError, match="within /workspace"):
            validate_workspace_path("/workspace2/evil.txt")
