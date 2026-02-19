"""Unit tests for SessionState -- no Docker or DOLFINx required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import (
    make_bc_info,
    make_entity_map_info,
    make_form_info,
    make_function_info,
    make_mesh_info,
    make_mesh_tags_info,
    make_solution_info,
    make_space_info,
)

from dolfinx_mcp.errors import (
    DOLFINxAPIError,
    FunctionSpaceNotFoundError,
    InvariantError,
    MeshNotFoundError,
    NoActiveMeshError,
)
from dolfinx_mcp.session import (
    BCInfo,
    EntityMapInfo,
    FormInfo,
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    MeshTagsInfo,
    SessionState,
    SolutionInfo,
)

# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


class TestSessionInit:
    def test_empty_session(self, session: SessionState):
        assert session.active_mesh is None
        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0

    def test_overview_empty(self, session: SessionState):
        ov = session.overview()
        assert ov["active_mesh"] is None
        assert ov["meshes"] == {}


# ---------------------------------------------------------------------------
# Tests: mesh accessors
# ---------------------------------------------------------------------------


class TestMeshAccessors:
    def test_get_mesh_no_active(self, session: SessionState):
        with pytest.raises(NoActiveMeshError):
            session.get_mesh()

    def test_get_mesh_not_found(self, session: SessionState):
        with pytest.raises(MeshNotFoundError, match="nonexistent"):
            session.get_mesh("nonexistent")

    def test_get_mesh_by_name(self, session: SessionState):
        info = make_mesh_info("m1")
        session.meshes["m1"] = info
        assert session.get_mesh("m1") is info

    def test_get_mesh_default_active(self, session: SessionState):
        info = make_mesh_info("m1")
        session.meshes["m1"] = info
        session.active_mesh = "m1"
        assert session.get_mesh() is info


# ---------------------------------------------------------------------------
# Tests: space accessors
# ---------------------------------------------------------------------------


class TestSpaceAccessors:
    def test_get_space_not_found(self, session: SessionState):
        with pytest.raises(FunctionSpaceNotFoundError):
            session.get_space("V")

    def test_get_only_space_none(self, session: SessionState):
        with pytest.raises(FunctionSpaceNotFoundError, match="No function"):
            session.get_only_space()

    def test_get_only_space_one(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        s = make_space_info("V", "m1")
        session.function_spaces["V"] = s
        assert session.get_only_space() is s

    def test_get_only_space_multiple(self, session: SessionState):
        session.function_spaces["V1"] = make_space_info("V1", "m1")
        session.function_spaces["V2"] = make_space_info("V2", "m1")
        with pytest.raises(FunctionSpaceNotFoundError, match="Multiple"):
            session.get_only_space()


# ---------------------------------------------------------------------------
# Tests: cascade deletion
# ---------------------------------------------------------------------------


class TestCascadeDeletion:
    def test_remove_mesh_cascades(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.functions["f"] = make_function_info("f", "V")
        session.bcs["bc0"] = make_bc_info("bc0", "V")
        session.active_mesh = "m1"

        session.remove_mesh("m1")

        assert "m1" not in session.meshes
        assert "V" not in session.function_spaces
        assert "f" not in session.functions
        assert "bc0" not in session.bcs
        assert session.active_mesh is None

    def test_remove_mesh_not_found(self, session: SessionState):
        with pytest.raises(MeshNotFoundError):
            session.remove_mesh("missing")

    def test_remove_mesh_preserves_other(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.meshes["m2"] = make_mesh_info("m2")
        session.function_spaces["V1"] = make_space_info("V1", "m1")
        session.function_spaces["V2"] = make_space_info("V2", "m2")
        session.active_mesh = "m2"

        session.remove_mesh("m1")

        assert "m2" in session.meshes
        assert "V2" in session.function_spaces
        assert session.active_mesh == "m2"


# ---------------------------------------------------------------------------
# Tests: overview
# ---------------------------------------------------------------------------


class TestOverview:
    def test_overview_populated(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.active_mesh = "m1"

        ov = session.overview()
        assert ov["active_mesh"] == "m1"
        assert "m1" in ov["meshes"]
        assert "V" in ov["function_spaces"]
        assert ov["meshes"]["m1"]["cell_type"] == "triangle"


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_clears_all(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.active_mesh = "m1"
        session.ufl_symbols["f"] = MagicMock()

        session.cleanup()

        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0
        assert len(session.ufl_symbols) == 0
        assert session.active_mesh is None

    def test_cleanup_clears_forms(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.forms["F1"] = make_form_info("F1")
        session.cleanup()
        assert len(session.forms) == 0

    def test_cleanup_clears_solver_diagnostics(self, session: SessionState):
        session.solver_diagnostics["diag1"] = {"status": "ok"}
        session.cleanup()
        assert len(session.solver_diagnostics) == 0

    def test_cleanup_clears_log_buffer(self, session: SessionState):
        session.log_buffer.append("test log")
        session.cleanup()
        assert len(session.log_buffer) == 0


# ---------------------------------------------------------------------------
# Tests: form accessors
# ---------------------------------------------------------------------------


class TestFormAccessors:
    def test_get_form_success(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        form_info = make_form_info("F1")
        session.forms["F1"] = form_info
        assert session.get_form("F1") is form_info

    def test_get_form_not_found(self, session: SessionState):
        with pytest.raises(DOLFINxAPIError, match="missing_form"):
            session.get_form("missing_form")

    def test_register_form(self, session: SessionState):
        form_info = make_form_info("F1", description="test")
        session.forms["F1"] = form_info
        assert "F1" in session.forms
        assert session.forms["F1"].description == "test"

    def test_register_form_duplicate_overwrites(self, session: SessionState):
        f1 = make_form_info("F1")
        f2 = make_form_info("F1")
        session.forms["F1"] = f1
        session.forms["F1"] = f2
        assert session.forms["F1"] is f2


# ---------------------------------------------------------------------------
# Tests: mesh tags accessors
# ---------------------------------------------------------------------------


class TestMeshTagsAccessors:
    def test_get_mesh_tags_success(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        tags_info = make_mesh_tags_info("t1")
        session.mesh_tags["t1"] = tags_info
        assert session.get_mesh_tags("t1") is tags_info

    def test_get_mesh_tags_not_found(self, session: SessionState):
        with pytest.raises(DOLFINxAPIError, match="MeshTags 'missing'"):
            session.get_mesh_tags("missing")


# ---------------------------------------------------------------------------
# Tests: entity map accessors
# ---------------------------------------------------------------------------


class TestEntityMapAccessors:
    def test_get_entity_map_success(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.meshes["m2"] = make_mesh_info("m2")
        em_info = make_entity_map_info("em1", parent_mesh="m1", child_mesh="m2", dimension=1)
        session.entity_maps["em1"] = em_info
        assert session.get_entity_map("em1") is em_info

    def test_get_entity_map_not_found(self, session: SessionState):
        with pytest.raises(DOLFINxAPIError, match="EntityMap 'missing'"):
            session.get_entity_map("missing")


# ---------------------------------------------------------------------------
# Tests: solution accessors
# ---------------------------------------------------------------------------


class TestSolutionAccessors:
    def test_get_last_solution_success(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        sol_info = make_solution_info("u1")
        session.solutions["u1"] = sol_info
        assert session.get_last_solution() is sol_info

    def test_get_last_solution_empty(self, session: SessionState):
        with pytest.raises(DOLFINxAPIError, match="No solutions"):
            session.get_last_solution()

    def test_get_last_solution_returns_last(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        sol1 = make_solution_info("u1")
        sol2 = make_solution_info("u2")
        session.solutions["u1"] = sol1
        session.solutions["u2"] = sol2
        assert session.get_last_solution() is sol2


# ---------------------------------------------------------------------------
# Tests: find_space_name
# ---------------------------------------------------------------------------


class TestFindSpaceName:
    def test_find_space_name_found(self, session: SessionState):
        space_info = make_space_info("V", "m1")
        session.function_spaces["V"] = space_info
        assert session.find_space_name(space_info.space) == "V"

    def test_find_space_name_not_found(self, session: SessionState):
        assert session.find_space_name(MagicMock()) == "unknown"


# ---------------------------------------------------------------------------
# Tests: forms wholesale clear on remove_mesh
# ---------------------------------------------------------------------------


class TestFormsCascade:
    def test_forms_cleared_on_remove_mesh(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.forms["F1"] = make_form_info("F1")
        session.remove_mesh("m1")
        assert len(session.forms) == 0

    def test_ufl_symbols_cleared_on_remove_mesh(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.ufl_symbols["x"] = MagicMock()
        session.remove_mesh("m1")
        assert len(session.ufl_symbols) == 0

    def test_forms_preserved_when_no_spaces_deleted(self, session: SessionState):
        """When remove_mesh deletes a mesh with no spaces, forms survive."""
        session.meshes["m1"] = make_mesh_info("m1")
        session.meshes["m2"] = make_mesh_info("m2")
        session.function_spaces["V"] = make_space_info("V", "m2")
        session.forms["F1"] = make_form_info("F1")
        # m1 has no spaces, so removing it should NOT clear forms
        session.remove_mesh("m1")
        assert "F1" in session.forms


# ---------------------------------------------------------------------------
# Tests: dataclass __post_init__ validation
# ---------------------------------------------------------------------------


class TestDataclassValidation:
    def test_mesh_info_empty_name(self):
        with pytest.raises(InvariantError, match="MeshInfo.name"):
            MeshInfo(name="", mesh=MagicMock(), cell_type="triangle",
                     num_cells=100, num_vertices=64, gdim=2, tdim=2)

    def test_mesh_info_bad_vertices(self):
        with pytest.raises(InvariantError, match="num_vertices"):
            MeshInfo(name="m", mesh=MagicMock(), cell_type="triangle",
                     num_cells=100, num_vertices=0, gdim=2, tdim=2)

    def test_mesh_info_bad_tdim(self):
        with pytest.raises(InvariantError, match="tdim must be"):
            MeshInfo(name="m", mesh=MagicMock(), cell_type="triangle",
                     num_cells=100, num_vertices=64, gdim=2, tdim=0)

    def test_space_info_empty_name(self):
        with pytest.raises(InvariantError, match="FunctionSpaceInfo.name"):
            FunctionSpaceInfo(name="", space=MagicMock(), mesh_name="m1",
                              element_family="Lagrange", element_degree=1, num_dofs=64)

    def test_space_info_empty_mesh(self):
        with pytest.raises(InvariantError, match="mesh_name must be"):
            FunctionSpaceInfo(name="V", space=MagicMock(), mesh_name="",
                              element_family="Lagrange", element_degree=1, num_dofs=64)

    def test_space_info_bad_shape(self):
        with pytest.raises(InvariantError, match="shape must be non-empty"):
            FunctionSpaceInfo(name="V", space=MagicMock(), mesh_name="m1",
                              element_family="Lagrange", element_degree=1,
                              num_dofs=64, shape=())

    def test_space_info_bad_shape_dims(self):
        with pytest.raises(InvariantError, match="shape dims must be > 0"):
            FunctionSpaceInfo(name="V", space=MagicMock(), mesh_name="m1",
                              element_family="Lagrange", element_degree=1,
                              num_dofs=64, shape=(0, 3))

    def test_space_info_shape_in_summary(self):
        si = make_space_info("V", shape=(3,))
        s = si.summary()
        assert "shape" in s
        assert s["shape"] == [3]

    def test_function_info_empty_name(self):
        with pytest.raises(InvariantError, match="FunctionInfo.name"):
            FunctionInfo(name="", function=MagicMock(), space_name="V")

    def test_function_info_empty_space(self):
        with pytest.raises(InvariantError, match="space_name must be"):
            FunctionInfo(name="f", function=MagicMock(), space_name="")

    def test_function_info_summary(self):
        fi = make_function_info("f", description="test func")
        s = fi.summary()
        assert s["description"] == "test func"

    def test_bc_info_empty_name(self):
        with pytest.raises(InvariantError, match="BCInfo.name"):
            BCInfo(name="", bc=MagicMock(), space_name="V", num_dofs=10)

    def test_bc_info_empty_space(self):
        with pytest.raises(InvariantError, match="space_name must be"):
            BCInfo(name="bc", bc=MagicMock(), space_name="", num_dofs=10)

    def test_bc_info_bad_dofs(self):
        with pytest.raises(InvariantError, match="num_dofs must be > 0"):
            BCInfo(name="bc", bc=MagicMock(), space_name="V", num_dofs=0)

    def test_form_info_empty_name(self):
        with pytest.raises(InvariantError, match="FormInfo.name"):
            FormInfo(name="", form=MagicMock(), ufl_form=MagicMock())

    def test_solution_info_empty_name(self):
        with pytest.raises(InvariantError, match="SolutionInfo.name"):
            SolutionInfo(name="", function=MagicMock(), space_name="V",
                         converged=True, iterations=5, residual_norm=1e-10,
                         wall_time=0.5)

    def test_solution_info_empty_space(self):
        with pytest.raises(InvariantError, match="space_name must be"):
            SolutionInfo(name="u", function=MagicMock(), space_name="",
                         converged=True, iterations=5, residual_norm=1e-10,
                         wall_time=0.5)

    def test_solution_info_bad_iterations(self):
        with pytest.raises(InvariantError, match="iterations"):
            SolutionInfo(name="u", function=MagicMock(), space_name="V",
                         converged=True, iterations=-1, residual_norm=1e-10,
                         wall_time=0.5)

    def test_solution_info_bad_residual(self):
        with pytest.raises(InvariantError, match="residual_norm"):
            SolutionInfo(name="u", function=MagicMock(), space_name="V",
                         converged=True, iterations=5, residual_norm=-1.0,
                         wall_time=0.5)

    def test_solution_info_bad_wall_time(self):
        with pytest.raises(InvariantError, match="wall_time"):
            SolutionInfo(name="u", function=MagicMock(), space_name="V",
                         converged=True, iterations=5, residual_norm=1e-10,
                         wall_time=-1.0)

    def test_entity_map_info_empty_name(self):
        with pytest.raises(InvariantError, match="EntityMapInfo.name"):
            EntityMapInfo(name="", entity_map=MagicMock(),
                          parent_mesh="m1", child_mesh="m2", dimension=1)

    def test_entity_map_info_empty_parent(self):
        with pytest.raises(InvariantError, match="parent_mesh must be"):
            EntityMapInfo(name="em1", entity_map=MagicMock(),
                          parent_mesh="", child_mesh="m2", dimension=1)

    def test_entity_map_info_empty_child(self):
        with pytest.raises(InvariantError, match="child_mesh must be"):
            EntityMapInfo(name="em1", entity_map=MagicMock(),
                          parent_mesh="m1", child_mesh="", dimension=1)

    def test_entity_map_info_bad_dimension(self):
        with pytest.raises(InvariantError, match="dimension must be"):
            EntityMapInfo(name="em1", entity_map=MagicMock(),
                          parent_mesh="m1", child_mesh="m2", dimension=-1)

    def test_mesh_tags_info_empty_name(self):
        with pytest.raises(InvariantError, match="MeshTagsInfo.name"):
            MeshTagsInfo(name="", tags=MagicMock(), mesh_name="m1", dimension=1)

    def test_mesh_tags_info_empty_mesh(self):
        with pytest.raises(InvariantError, match="mesh_name must be"):
            MeshTagsInfo(name="t1", tags=MagicMock(), mesh_name="", dimension=1)

    def test_mesh_tags_info_bad_dimension(self):
        with pytest.raises(InvariantError, match="dimension must be"):
            MeshTagsInfo(name="t1", tags=MagicMock(), mesh_name="m1",
                         dimension=-1)


# ---------------------------------------------------------------------------
# Tests: overview includes forms/ufl_symbols
# ---------------------------------------------------------------------------


class TestOverviewCompleteness:
    def test_summary_includes_forms(self, session: SessionState):
        session.meshes["m1"] = make_mesh_info("m1")
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.forms["F1"] = make_form_info("F1")
        ov = session.overview()
        assert "forms" in ov
        assert "F1" in ov["forms"]

    def test_summary_includes_ufl_symbols(self, session: SessionState):
        session.ufl_symbols["x"] = MagicMock()
        ov = session.overview()
        assert "ufl_symbols" in ov
        assert "x" in ov["ufl_symbols"]


# ---------------------------------------------------------------------------
# Tests: golden scenarios (end-to-end workflow)
# ---------------------------------------------------------------------------


class TestGoldenScenarios:
    def test_full_workflow_create_and_destroy(self, session: SessionState):
        """Full workflow: mesh -> space -> func -> BC -> form -> sol -> remove."""
        # Build up
        session.meshes["m1"] = make_mesh_info("m1")
        session.active_mesh = "m1"
        session.function_spaces["V"] = make_space_info("V", "m1")
        session.functions["f"] = make_function_info("f", "V")
        session.bcs["bc0"] = make_bc_info("bc0", "V")
        session.forms["F1"] = make_form_info("F1")
        session.solutions["u1"] = make_solution_info("u1")
        session.ufl_symbols["x"] = MagicMock()
        session.check_invariants()

        # Tear down
        session.remove_mesh("m1")

        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0
        assert len(session.functions) == 0
        assert len(session.bcs) == 0
        assert len(session.forms) == 0
        assert len(session.solutions) == 0
        assert len(session.ufl_symbols) == 0
        assert session.active_mesh is None
        session.check_invariants()

    def test_multi_mesh_partial_removal(self, session: SessionState):
        """Two meshes: remove one, forms survive if other mesh has spaces."""
        session.meshes["m1"] = make_mesh_info("m1")
        session.meshes["m2"] = make_mesh_info("m2")
        session.function_spaces["V1"] = make_space_info("V1", "m1")
        session.function_spaces["V2"] = make_space_info("V2", "m2")
        session.functions["f1"] = make_function_info("f1", "V1")
        session.functions["f2"] = make_function_info("f2", "V2")
        session.forms["F1"] = make_form_info("F1")
        session.active_mesh = "m2"
        session.check_invariants()

        # Remove m1 -- it has spaces, so forms get cleared (wholesale behavior)
        session.remove_mesh("m1")

        assert "m1" not in session.meshes
        assert "V1" not in session.function_spaces
        assert "f1" not in session.functions
        # Forms cleared because dep_spaces for m1 was non-empty
        assert len(session.forms) == 0
        # m2 and its dependents survive
        assert "m2" in session.meshes
        assert "V2" in session.function_spaces
        assert "f2" in session.functions
        session.check_invariants()


class TestRegisterFunction:
    """FIX-5 (BUG-005): register_function API on SessionState."""

    def test_register_function_success(self, session):
        session.meshes["m"] = make_mesh_info("m")
        session.function_spaces["V"] = make_space_info("V", mesh_name="m")

        info = session.register_function("f", MagicMock(), "V", description="test")
        assert "f" in session.functions
        assert info.space_name == "V"
        assert info.description == "test"

    def test_register_function_duplicate_raises(self, session):
        from dolfinx_mcp.errors import DuplicateNameError

        session.meshes["m"] = make_mesh_info("m")
        session.function_spaces["V"] = make_space_info("V", mesh_name="m")
        session.register_function("f", MagicMock(), "V")

        with pytest.raises(DuplicateNameError):
            session.register_function("f", MagicMock(), "V")

    def test_register_function_missing_space_raises(self, session):
        from dolfinx_mcp.errors import FunctionSpaceNotFoundError

        session.meshes["m"] = make_mesh_info("m")
        # No function space registered

        with pytest.raises(FunctionSpaceNotFoundError):
            session.register_function("f", MagicMock(), "nonexistent")

    def test_register_function_empty_name_raises(self, session):
        from dolfinx_mcp.errors import InvariantError

        with pytest.raises(InvariantError):
            session.register_function("", MagicMock(), "V")
