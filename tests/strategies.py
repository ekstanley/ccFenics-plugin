"""Hypothesis strategies for SessionState property-based testing.

Provides reusable strategies for generating random operation sequences
on SessionState. Used by test_property_invariants.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import strategies as st

from dolfinx_mcp.session import (
    BCInfo,
    EntityMapInfo,
    FunctionInfo,
    FunctionSpaceInfo,
    MeshInfo,
    MeshTagsInfo,
    SessionState,
    SolutionInfo,
)

# ---------------------------------------------------------------------------
# Name pools -- small enough for exhaustive coverage in model checking,
# large enough to exercise interleaving
# ---------------------------------------------------------------------------

mesh_names = st.sampled_from(["m1", "m2", "m3", "m4"])
space_names = st.sampled_from(["V1", "V2", "V3"])
func_names = st.sampled_from(["f1", "f2", "f3"])
bc_names = st.sampled_from(["bc1", "bc2"])
sol_names = st.sampled_from(["u1", "u2"])
tag_names = st.sampled_from(["t1", "t2"])
emap_names = st.sampled_from(["em1", "em2"])


# ---------------------------------------------------------------------------
# Operation strategies -- each returns a tuple (op_name, *args)
# ---------------------------------------------------------------------------


@st.composite
def register_mesh_op(draw: st.DrawFn) -> tuple[str, str]:
    return ("register_mesh", draw(mesh_names))


@st.composite
def register_space_op(draw: st.DrawFn) -> tuple[str, str, str]:
    return ("register_space", draw(space_names), draw(mesh_names))


@st.composite
def register_function_op(draw: st.DrawFn) -> tuple[str, str, str]:
    return ("register_function", draw(func_names), draw(space_names))


@st.composite
def register_bc_op(draw: st.DrawFn) -> tuple[str, str, str]:
    return ("register_bc", draw(bc_names), draw(space_names))


@st.composite
def register_solution_op(draw: st.DrawFn) -> tuple[str, str, str]:
    return ("register_solution", draw(sol_names), draw(space_names))


@st.composite
def register_tags_op(draw: st.DrawFn) -> tuple[str, str, str]:
    return ("register_tags", draw(tag_names), draw(mesh_names))


@st.composite
def register_entity_map_op(draw: st.DrawFn) -> tuple[str, str, str, str]:
    return ("register_entity_map", draw(emap_names), draw(mesh_names), draw(mesh_names))


@st.composite
def remove_mesh_op(draw: st.DrawFn) -> tuple[str, str]:
    return ("remove_mesh", draw(mesh_names))


@st.composite
def cleanup_op(draw: st.DrawFn) -> tuple[str]:
    _ = draw(st.none())  # Consume from draw to satisfy composite contract
    return ("cleanup",)


# Combined strategy: any single operation
any_operation = st.one_of(
    register_mesh_op(),
    register_space_op(),
    register_function_op(),
    register_bc_op(),
    register_solution_op(),
    register_tags_op(),
    register_entity_map_op(),
    remove_mesh_op(),
    cleanup_op(),
)

# Strategy: sequence of operations (1-30)
operation_sequence = st.lists(any_operation, min_size=1, max_size=30)


# ---------------------------------------------------------------------------
# Operation executor -- silently skips when preconditions are not met
# (expected in random sequences)
# ---------------------------------------------------------------------------


def apply_operation(session: SessionState, op: tuple) -> None:  # noqa: C901
    """Apply a single operation to a session.

    Silently skips if preconditions are not met. This is expected behavior:
    random operation sequences will frequently attempt invalid operations
    (e.g., registering a function space for a mesh that doesn't exist).
    """
    try:
        if op[0] == "register_mesh":
            name = op[1]
            if name not in session.meshes:
                session.meshes[name] = MeshInfo(
                    name=name,
                    mesh=MagicMock(),
                    cell_type="triangle",
                    num_cells=100,
                    num_vertices=64,
                    gdim=2,
                    tdim=2,
                )
                session.active_mesh = name

        elif op[0] == "register_space":
            name, mesh_name = op[1], op[2]
            if mesh_name in session.meshes and name not in session.function_spaces:
                session.function_spaces[name] = FunctionSpaceInfo(
                    name=name,
                    space=MagicMock(),
                    mesh_name=mesh_name,
                    element_family="Lagrange",
                    element_degree=1,
                    num_dofs=64,
                )

        elif op[0] == "register_function":
            name, space_name = op[1], op[2]
            if space_name in session.function_spaces and name not in session.functions:
                session.functions[name] = FunctionInfo(
                    name=name,
                    function=MagicMock(),
                    space_name=space_name,
                )

        elif op[0] == "register_bc":
            name, space_name = op[1], op[2]
            if space_name in session.function_spaces and name not in session.bcs:
                session.bcs[name] = BCInfo(
                    name=name,
                    bc=MagicMock(),
                    space_name=space_name,
                    num_dofs=10,
                )

        elif op[0] == "register_solution":
            name, space_name = op[1], op[2]
            if space_name in session.function_spaces and name not in session.solutions:
                session.solutions[name] = SolutionInfo(
                    name=name,
                    function=MagicMock(),
                    space_name=space_name,
                    converged=True,
                    iterations=5,
                    residual_norm=1e-10,
                    wall_time=0.5,
                )

        elif op[0] == "register_tags":
            name, mesh_name = op[1], op[2]
            if mesh_name in session.meshes and name not in session.mesh_tags:
                session.mesh_tags[name] = MeshTagsInfo(
                    name=name,
                    tags=MagicMock(),
                    mesh_name=mesh_name,
                    dimension=1,
                )

        elif op[0] == "register_entity_map":
            name, parent, child = op[1], op[2], op[3]
            if (
                parent in session.meshes
                and child in session.meshes
                and name not in session.entity_maps
            ):
                session.entity_maps[name] = EntityMapInfo(
                    name=name,
                    entity_map=MagicMock(),
                    parent_mesh=parent,
                    child_mesh=child,
                    dimension=1,
                )

        elif op[0] == "remove_mesh":
            name = op[1]
            if name in session.meshes:
                session.remove_mesh(name)

        elif op[0] == "cleanup":
            session.cleanup()

    except Exception:
        # Invalid preconditions are expected in random sequences; skip silently.
        pass
