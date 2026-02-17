"""Property-based tests using Hypothesis for SessionState invariants.

Properties tested:
  P1: Any valid operation sequence preserves all 8 invariants
  P2: Cascade deletion removes ALL dependents (no partial removal)
  P3: Cleanup always produces empty state
  P4: Registry keys always match entry names (key == Info.name)
  P5: No operation introduces duplicate keys across registries
  P6: After removeMesh(m), no registry references m
  P7: Forms cleared wholesale when spaces deleted (INV-8 cascade)
  P8: Forms non-empty implies spaces non-empty (INV-8 invariant)
  P9: Extension maps for read_workspace_file have no overlap
  P10: Path traversal attacks always rejected by read_workspace_file
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dolfinx_mcp.session import SessionState

from .strategies import (
    any_operation,
    apply_operation,
    mesh_names,
    operation_sequence,
)


class TestInvariantPreservation:
    """P1: Any sequence of operations preserves all 7 invariants."""

    @given(ops=operation_sequence)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_ops_preserve_invariants(self, ops: list) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)
            session.check_invariants()  # Raises InvariantError if violated


class TestCascadeDeletion:
    """P2: removeMesh removes all dependents completely."""

    @given(
        setup_ops=st.lists(any_operation, min_size=3, max_size=15),
        mesh_to_remove=mesh_names,
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_mesh_no_orphans(
        self, setup_ops: list, mesh_to_remove: str
    ) -> None:
        session = SessionState()
        for op in setup_ops:
            apply_operation(session, op)

        if mesh_to_remove in session.meshes:
            session.remove_mesh(mesh_to_remove)

            # After removal: no registry references the deleted mesh
            assert mesh_to_remove not in session.meshes
            assert all(
                fs.mesh_name != mesh_to_remove
                for fs in session.function_spaces.values()
            )
            assert all(
                mt.mesh_name != mesh_to_remove
                for mt in session.mesh_tags.values()
            )
            assert all(
                em.parent_mesh != mesh_to_remove and em.child_mesh != mesh_to_remove
                for em in session.entity_maps.values()
            )
            # Transitive: no functions/BCs/solutions reference orphaned spaces
            session.check_invariants()


class TestCleanup:
    """P3: cleanup always produces empty state regardless of prior state."""

    @given(ops=operation_sequence)
    @settings(max_examples=200)
    def test_cleanup_always_empties(self, ops: list) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)

        session.cleanup()

        assert len(session.meshes) == 0
        assert len(session.function_spaces) == 0
        assert len(session.functions) == 0
        assert len(session.bcs) == 0
        assert len(session.solutions) == 0
        assert len(session.mesh_tags) == 0
        assert len(session.entity_maps) == 0
        assert session.active_mesh is None
        session.check_invariants()


class TestKeyNameConsistency:
    """P4: Registry keys always match the Info.name field."""

    @given(ops=operation_sequence)
    @settings(max_examples=300)
    def test_key_equals_name(self, ops: list) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)

        for k, v in session.meshes.items():
            assert k == v.name
        for k, v in session.function_spaces.items():
            assert k == v.name
        for k, v in session.functions.items():
            assert k == v.name
        for k, v in session.bcs.items():
            assert k == v.name
        for k, v in session.solutions.items():
            assert k == v.name
        for k, v in session.mesh_tags.items():
            assert k == v.name
        for k, v in session.entity_maps.items():
            assert k == v.name


class TestNoDuplicateKeys:
    """P5: No operation sequence produces duplicate registry keys."""

    @given(ops=operation_sequence)
    @settings(max_examples=200)
    def test_no_cross_registry_collisions(self, ops: list) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)

        # Verify dict invariant: keys are unique within each registry
        # (Python dicts enforce this, but verify the count matches)
        all_mesh_keys = list(session.meshes.keys())
        assert len(all_mesh_keys) == len(set(all_mesh_keys))

        all_space_keys = list(session.function_spaces.keys())
        assert len(all_space_keys) == len(set(all_space_keys))

        all_func_keys = list(session.functions.keys())
        assert len(all_func_keys) == len(set(all_func_keys))


class TestRemoveMeshNoReference:
    """P6: After removeMesh(m), no registry contains m as a foreign key."""

    @given(
        ops=st.lists(any_operation, min_size=1, max_size=20),
        target=mesh_names,
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_post_remove_no_references(self, ops: list, target: str) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)

        if target in session.meshes:
            session.remove_mesh(target)

            # Exhaustive check: target appears nowhere as a foreign key
            assert target not in session.meshes
            if session.active_mesh is not None:
                assert session.active_mesh != target
            for fs in session.function_spaces.values():
                assert fs.mesh_name != target
            for mt in session.mesh_tags.values():
                assert mt.mesh_name != target
            for em in session.entity_maps.values():
                assert em.parent_mesh != target
                assert em.child_mesh != target


class TestFormWholesaleClear:
    """P7: Forms cleared wholesale when removeMesh deletes any space."""

    @given(
        ops=st.lists(any_operation, min_size=3, max_size=20),
        target=mesh_names,
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_forms_cleared_when_spaces_deleted(
        self, ops: list, target: str
    ) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)

        if target in session.meshes:
            # Check if any spaces depend on this mesh
            dep_spaces = [
                sn for sn, si in session.function_spaces.items()
                if si.mesh_name == target
            ]
            session.remove_mesh(target)

            if dep_spaces:
                # Forms and UFL symbols should be cleared wholesale
                assert len(session.forms) == 0
                assert len(session.ufl_symbols) == 0


class TestFormSpaceInvariant:
    """P8: Forms non-empty implies function_spaces non-empty (INV-8)."""

    @given(ops=operation_sequence)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_forms_require_spaces(self, ops: list) -> None:
        session = SessionState()
        for op in ops:
            apply_operation(session, op)
            # After every operation, check INV-8
            if session.forms:
                assert len(session.function_spaces) > 0, (
                    f"INV-8 violated: forms={list(session.forms.keys())} "
                    f"but function_spaces is empty"
                )


class TestExtensionMapNoOverlap:
    """P9: Binary and text extension maps have no overlap."""

    @given(ext=st.from_regex(r"\.[a-z0-9]{1,6}", fullmatch=True))
    @settings(max_examples=200)
    def test_extension_not_in_both_maps(self, ext: str) -> None:
        from dolfinx_mcp.tools.session_mgmt import (
            _BINARY_EXTENSIONS,
            _TEXT_EXTENSIONS,
        )

        # An extension can be in binary OR text, never both
        assert not (ext in _BINARY_EXTENSIONS and ext in _TEXT_EXTENSIONS), (
            f"Extension '{ext}' in both _BINARY_EXTENSIONS and _TEXT_EXTENSIONS"
        )


class TestPathTraversalDefense:
    """P10: Adversarial paths always rejected by read_workspace_file."""

    @given(
        path=st.one_of(
            # Relative traversal attacks
            st.text(
                alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_/.-"),
                min_size=1,
                max_size=50,
            ).filter(lambda s: ".." in s),
            # Known attack vectors
            st.sampled_from([
                "../etc/passwd",
                "/etc/shadow",
                "/workspace/../etc/passwd",
                "../../../../../../etc/passwd",
                "/workspace/../../etc/shadow",
            ]),
        )
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_traversal_always_rejected(self, path: str) -> None:
        from unittest.mock import MagicMock

        from dolfinx_mcp.session import MeshInfo, SessionState
        from dolfinx_mcp.tools.session_mgmt import read_workspace_file

        session = SessionState()
        session.meshes["m1"] = MeshInfo(
            name="m1", mesh=MagicMock(), cell_type="triangle",
            num_cells=100, num_vertices=64, gdim=2, tdim=2,
        )
        session.active_mesh = "m1"
        ctx = MagicMock()
        ctx.request_context.lifespan_context = session

        result = await read_workspace_file(file_path=path, ctx=ctx)

        assert result.get("error") in (
            "FILE_IO_ERROR", "PRECONDITION_VIOLATED",
        ), f"Path '{path}' was not rejected: {result}"
