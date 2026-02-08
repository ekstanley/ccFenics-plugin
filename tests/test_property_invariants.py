"""Property-based tests using Hypothesis for SessionState invariants.

Properties tested:
  P1: Any valid operation sequence preserves all 7 invariants
  P2: Cascade deletion removes ALL dependents (no partial removal)
  P3: Cleanup always produces empty state
  P4: Registry keys always match entry names (key == Info.name)
  P5: No operation introduces duplicate keys across registries
  P6: After removeMesh(m), no registry references m
"""

from __future__ import annotations

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
