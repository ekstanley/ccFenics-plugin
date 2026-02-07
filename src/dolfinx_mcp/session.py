"""Session state management for DOLFINx MCP server.

Holds all DOLFINx objects (meshes, function spaces, functions, BCs, forms,
solutions) in named registries. Provides typed accessors with clear error
messages and cascade deletion when parent objects are removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    DOLFINxAPIError,
    FunctionNotFoundError,
    FunctionSpaceNotFoundError,
    InvariantError,
    MeshNotFoundError,
    NoActiveMeshError,
    PostconditionError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry dataclasses -- wrap DOLFINx objects with metadata
# ---------------------------------------------------------------------------


@dataclass
class MeshInfo:
    name: str
    mesh: Any  # dolfinx.mesh.Mesh
    cell_type: str
    num_cells: int
    num_vertices: int
    gdim: int
    tdim: int

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("MeshInfo.name must be non-empty")
        if self.num_cells <= 0:
            raise InvariantError(f"num_cells must be > 0, got {self.num_cells}")
        if self.num_vertices <= 0:
            raise InvariantError(f"num_vertices must be > 0, got {self.num_vertices}")
        if self.gdim not in (1, 2, 3):
            raise InvariantError(f"gdim must be 1, 2, or 3, got {self.gdim}")
        if self.tdim not in (1, 2, 3):
            raise InvariantError(f"tdim must be 1, 2, or 3, got {self.tdim}")
        if self.tdim > self.gdim:
            raise InvariantError(f"tdim ({self.tdim}) must be <= gdim ({self.gdim})")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cell_type": self.cell_type,
            "num_cells": self.num_cells,
            "num_vertices": self.num_vertices,
            "gdim": self.gdim,
            "tdim": self.tdim,
        }


@dataclass
class FunctionSpaceInfo:
    name: str
    space: Any  # dolfinx.fem.FunctionSpace
    mesh_name: str
    element_family: str
    element_degree: int
    num_dofs: int
    shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("FunctionSpaceInfo.name must be non-empty")
        if not self.mesh_name:
            raise InvariantError("mesh_name must be non-empty")
        if self.element_degree < 0:
            raise InvariantError(f"degree must be >= 0, got {self.element_degree}")
        if self.num_dofs <= 0:
            raise InvariantError(f"num_dofs must be > 0, got {self.num_dofs}")
        if self.shape is not None:
            if not self.shape:
                raise InvariantError("shape must be non-empty if provided")
            if not all(d > 0 for d in self.shape):
                raise InvariantError(f"shape dims must be > 0: {self.shape}")

    def summary(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "mesh_name": self.mesh_name,
            "element_family": self.element_family,
            "element_degree": self.element_degree,
            "num_dofs": self.num_dofs,
        }
        if self.shape is not None:
            result["shape"] = list(self.shape)
        return result


@dataclass
class FunctionInfo:
    name: str
    function: Any  # dolfinx.fem.Function
    space_name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("FunctionInfo.name must be non-empty")
        if not self.space_name:
            raise InvariantError("space_name must be non-empty")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "space_name": self.space_name,
            "description": self.description,
        }


@dataclass
class BCInfo:
    name: str
    bc: Any  # dolfinx.fem.DirichletBC
    space_name: str
    num_dofs: int
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("BCInfo.name must be non-empty")
        if not self.space_name:
            raise InvariantError("space_name must be non-empty")
        if self.num_dofs <= 0:
            raise InvariantError(f"num_dofs must be > 0, got {self.num_dofs}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "space_name": self.space_name,
            "num_dofs": self.num_dofs,
            "description": self.description,
        }


@dataclass
class FormInfo:
    name: str
    form: Any  # dolfinx.fem.Form (compiled)
    ufl_form: Any  # ufl.Form (symbolic)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("FormInfo.name must be non-empty")
        if self.form is None:
            raise InvariantError("form (compiled) must not be None")
        if self.ufl_form is None:
            raise InvariantError("ufl_form (symbolic) must not be None")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
        }


@dataclass
class SolutionInfo:
    name: str
    function: Any  # dolfinx.fem.Function
    space_name: str
    converged: bool
    iterations: int
    residual_norm: float
    wall_time: float

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("SolutionInfo.name must be non-empty")
        if not self.space_name:
            raise InvariantError("space_name must be non-empty")
        if self.iterations < 0:
            raise InvariantError(f"iterations must be >= 0, got {self.iterations}")
        if self.residual_norm < 0.0:
            raise InvariantError(f"residual_norm must be >= 0, got {self.residual_norm}")
        if self.wall_time < 0.0:
            raise InvariantError(f"wall_time must be >= 0, got {self.wall_time}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "space_name": self.space_name,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "wall_time": self.wall_time,
        }


@dataclass
class MeshTagsInfo:
    name: str
    tags: Any  # dolfinx.mesh.MeshTags
    mesh_name: str
    dimension: int
    unique_tags: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("MeshTagsInfo.name must be non-empty")
        if not self.mesh_name:
            raise InvariantError("mesh_name must be non-empty")
        if self.dimension < 0:
            raise InvariantError(f"dimension must be >= 0, got {self.dimension}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mesh_name": self.mesh_name,
            "dimension": self.dimension,
            "unique_tags": self.unique_tags,
        }


@dataclass
class EntityMapInfo:
    name: str
    entity_map: Any  # numpy array or EntityMap from create_submesh
    parent_mesh: str
    child_mesh: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantError("EntityMapInfo.name must be non-empty")
        if not self.parent_mesh:
            raise InvariantError("parent_mesh must be non-empty")
        if not self.child_mesh:
            raise InvariantError("child_mesh must be non-empty")
        if self.dimension < 0:
            raise InvariantError(f"dimension must be >= 0, got {self.dimension}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent_mesh": self.parent_mesh,
            "child_mesh": self.child_mesh,
            "dimension": self.dimension,
        }


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------


class SessionState:
    """Central registry for all DOLFINx objects in the current MCP session.

    Not a dataclass -- has methods, invariants, and cascade logic.
    """

    def __init__(self) -> None:
        self.meshes: dict[str, MeshInfo] = {}
        self.function_spaces: dict[str, FunctionSpaceInfo] = {}
        self.functions: dict[str, FunctionInfo] = {}
        self.bcs: dict[str, BCInfo] = {}
        self.forms: dict[str, FormInfo] = {}
        self.solutions: dict[str, SolutionInfo] = {}
        self.mesh_tags: dict[str, MeshTagsInfo] = {}
        self.entity_maps: dict[str, EntityMapInfo] = {}
        self.active_mesh: str | None = None
        self.ufl_symbols: dict[str, Any] = {}
        self.solver_diagnostics: dict[str, Any] = {}
        self.log_buffer: list[str] = []

    # --- Invariant verification ---

    def check_invariants(self) -> None:
        """Verify all session referential integrity invariants.

        Raises InvariantError if any cross-reference is dangling.
        """
        from .errors import InvariantError

        # INV-1: active_mesh must be None or a valid key
        if self.active_mesh is not None and self.active_mesh not in self.meshes:
            raise InvariantError(
                f"active_mesh '{self.active_mesh}' not in meshes {list(self.meshes.keys())}"
            )

        # INV-2: all function_spaces reference valid meshes
        for name, fs in self.function_spaces.items():
            if fs.mesh_name not in self.meshes:
                raise InvariantError(
                    f"Function space '{name}' references non-existent mesh '{fs.mesh_name}'"
                )

        # INV-3: all functions reference valid function_spaces
        for name, fn in self.functions.items():
            if fn.space_name not in self.function_spaces:
                raise InvariantError(
                    f"Function '{name}' references non-existent space '{fn.space_name}'"
                )

        # INV-4: all BCs reference valid function_spaces
        for name, bc in self.bcs.items():
            if bc.space_name not in self.function_spaces:
                raise InvariantError(
                    f"BC '{name}' references non-existent space '{bc.space_name}'"
                )

        # INV-5: all solutions reference valid function_spaces
        for name, sol in self.solutions.items():
            if sol.space_name not in self.function_spaces:
                raise InvariantError(
                    f"Solution '{name}' references non-existent space '{sol.space_name}'"
                )

        # INV-6: all mesh_tags reference valid meshes
        for name, mt in self.mesh_tags.items():
            if mt.mesh_name not in self.meshes:
                raise InvariantError(
                    f"MeshTags '{name}' references non-existent mesh '{mt.mesh_name}'"
                )

        # INV-7: all entity_maps reference valid meshes
        for name, em in self.entity_maps.items():
            if em.parent_mesh not in self.meshes:
                raise InvariantError(
                    f"EntityMap '{name}' parent_mesh '{em.parent_mesh}' not found"
                )
            if em.child_mesh not in self.meshes:
                raise InvariantError(
                    f"EntityMap '{name}' child_mesh '{em.child_mesh}' not found"
                )

    # --- Typed accessors ---

    def get_mesh(self, name: str | None = None) -> MeshInfo:
        """Return named mesh, or active mesh if name is None."""
        if name is None:
            if self.active_mesh is None:
                raise NoActiveMeshError("No active mesh. Create one first.")
            name = self.active_mesh
        if name not in self.meshes:
            available = list(self.meshes.keys())
            raise MeshNotFoundError(
                f"Mesh '{name}' not found. Available: {available}"
            )
        result = self.meshes[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_mesh(): MeshInfo.name '{result.name}' != registry key '{name}'"
                )
        return result

    def get_space(self, name: str) -> FunctionSpaceInfo:
        if name not in self.function_spaces:
            available = list(self.function_spaces.keys())
            raise FunctionSpaceNotFoundError(
                f"Function space '{name}' not found. Available: {available}"
            )
        result = self.function_spaces[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_space(): FunctionSpaceInfo.name '{result.name}' != registry key '{name}'"
                )
            if result.mesh_name not in self.meshes:
                raise PostconditionError(
                    f"get_space(): mesh '{result.mesh_name}' not in meshes registry"
                )
        return result

    def get_function(self, name: str) -> FunctionInfo:
        if name not in self.functions:
            available = list(self.functions.keys())
            raise FunctionNotFoundError(
                f"Function '{name}' not found. Available: {available}"
            )
        result = self.functions[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_function(): FunctionInfo.name '{result.name}' != registry key '{name}'"
                )
            if result.space_name not in self.function_spaces:
                raise PostconditionError(
                    f"get_function(): space '{result.space_name}' not in function_spaces registry"
                )
        return result

    def get_only_space(self) -> FunctionSpaceInfo:
        """Return the sole function space, or raise if zero or multiple exist."""
        if len(self.function_spaces) == 0:
            raise FunctionSpaceNotFoundError("No function spaces defined.")
        if len(self.function_spaces) == 1:
            result = next(iter(self.function_spaces.values()))
            if __debug__:
                if result.mesh_name not in self.meshes:
                    raise PostconditionError(
                        f"get_only_space(): mesh '{result.mesh_name}' not in meshes registry"
                    )
            return result
        raise FunctionSpaceNotFoundError(
            f"Multiple function spaces exist ({list(self.function_spaces.keys())}). "
            "Specify which one to use."
        )

    def get_solution(self, name: str) -> SolutionInfo:
        """Return named solution, or raise if not found."""
        if name not in self.solutions:
            raise FunctionNotFoundError(
                f"Solution '{name}' not found. Available: {list(self.solutions.keys())}"
            )
        result = self.solutions[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_solution(): SolutionInfo.name '{result.name}' != registry key '{name}'"
                )
            if result.space_name not in self.function_spaces:
                raise PostconditionError(
                    f"get_solution(): space '{result.space_name}' not in function_spaces registry"
                )
        return result

    def get_form(self, name: str, suggestion: str = "") -> FormInfo:
        """Return named form, or raise if not found."""
        if name not in self.forms:
            raise DOLFINxAPIError(
                f"No {name} form defined. Available: {list(self.forms.keys())}",
                suggestion=suggestion or "Use define_variational_form first.",
            )
        result = self.forms[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_form(): FormInfo.name '{result.name}' != registry key '{name}'"
                )
        return result

    def get_mesh_tags(self, name: str) -> MeshTagsInfo:
        """Return named mesh tags, or raise if not found."""
        if name not in self.mesh_tags:
            available = list(self.mesh_tags.keys())
            raise DOLFINxAPIError(
                f"MeshTags '{name}' not found. Available: {available}",
                suggestion="Check available tags with get_session_state.",
            )
        result = self.mesh_tags[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_mesh_tags(): MeshTagsInfo.name '{result.name}' != registry key '{name}'"
                )
            if result.mesh_name not in self.meshes:
                raise PostconditionError(
                    f"get_mesh_tags(): mesh '{result.mesh_name}' not in meshes registry"
                )
        return result

    def get_entity_map(self, name: str) -> EntityMapInfo:
        """Return named entity map, or raise if not found."""
        if name not in self.entity_maps:
            available = list(self.entity_maps.keys())
            raise DOLFINxAPIError(
                f"EntityMap '{name}' not found. Available: {available}",
                suggestion="Check available entity maps with get_session_state.",
            )
        result = self.entity_maps[name]
        if __debug__:
            if result.name != name:
                raise PostconditionError(
                    f"get_entity_map(): EntityMapInfo.name '{result.name}' != registry key '{name}'"
                )
            if result.parent_mesh not in self.meshes:
                raise PostconditionError(
                    f"get_entity_map(): parent_mesh '{result.parent_mesh}' not in meshes registry"
                )
            if result.child_mesh not in self.meshes:
                raise PostconditionError(
                    f"get_entity_map(): child_mesh '{result.child_mesh}' not in meshes registry"
                )
        return result

    def get_last_solution(self) -> SolutionInfo:
        """Return the most recently stored solution, or raise if none exist."""
        if not self.solutions:
            raise DOLFINxAPIError(
                "No solutions available.",
                suggestion="Run solve() or solve_time_dependent() first.",
            )
        name = list(self.solutions.keys())[-1]
        result = self.solutions[name]
        if __debug__:
            if result.space_name not in self.function_spaces:
                raise PostconditionError(
                    f"get_last_solution(): space '{result.space_name}'"
                    " not in function_spaces registry"
                )
        return result

    # --- Cascade deletion ---

    def remove_mesh(self, name: str) -> None:
        """Remove a mesh and all dependent spaces, functions, BCs."""
        if name not in self.meshes:
            raise MeshNotFoundError(f"Mesh '{name}' not found.")

        # Collect dependent spaces
        dep_spaces = [
            s.name for s in self.function_spaces.values() if s.mesh_name == name
        ]
        for sname in dep_spaces:
            self._remove_space_dependents(sname)
            del self.function_spaces[sname]

        # Remove dependent mesh tags
        dep_tags = [k for k, v in self.mesh_tags.items() if v.mesh_name == name]
        for tname in dep_tags:
            del self.mesh_tags[tname]

        # Remove dependent entity maps
        dep_maps = [
            k for k, v in self.entity_maps.items()
            if v.parent_mesh == name or v.child_mesh == name
        ]
        for mname in dep_maps:
            del self.entity_maps[mname]

        del self.meshes[name]
        if self.active_mesh == name:
            self.active_mesh = None

        # Postcondition: no dangling references to deleted mesh
        if name in self.meshes:
            raise PostconditionError(f"remove_mesh(): mesh '{name}' still present after removal")
        if not all(fs.mesh_name != name for fs in self.function_spaces.values()):
            raise PostconditionError(
                f"remove_mesh(): dangling function space references to deleted mesh '{name}'"
            )
        if not all(mt.mesh_name != name for mt in self.mesh_tags.values()):
            raise PostconditionError(
                f"remove_mesh(): dangling mesh tag references to deleted mesh '{name}'"
            )
        if not all(
            em.parent_mesh != name and em.child_mesh != name
            for em in self.entity_maps.values()
        ):
            raise PostconditionError(
                f"remove_mesh(): dangling entity map references to deleted mesh '{name}'"
            )
        if self.active_mesh is not None:
            if self.active_mesh not in self.meshes:
                raise PostconditionError(
                    f"remove_mesh(): active_mesh '{self.active_mesh}' not in meshes registry"
                )

        logger.info("Removed mesh '%s' and %d dependent spaces", name, len(dep_spaces))

    def _remove_space_dependents(self, space_name: str) -> None:
        """Remove functions, BCs, and solutions that depend on a space."""
        for registry in (self.functions, self.bcs, self.solutions):
            to_remove = [
                k for k, v in registry.items() if getattr(v, "space_name", None) == space_name
            ]
            for k in to_remove:
                del registry[k]

        # Postcondition: no remaining references to deleted space
        if not all(
            getattr(v, "space_name", None) != space_name
            for reg in (self.functions, self.bcs, self.solutions)
            for v in reg.values()
        ):
            raise PostconditionError(
                f"_remove_space_dependents(): dangling references to deleted space '{space_name}'"
            )

    # --- Utilities ---

    def find_space_name(self, space_object) -> str:
        """Find the registry name for a function space object, or 'unknown'."""
        for sname, sinfo in self.function_spaces.items():
            if sinfo.space is space_object:
                return sname
        return "unknown"

    # --- Overview ---

    def overview(self) -> dict[str, Any]:
        return {
            "active_mesh": self.active_mesh,
            "meshes": {k: v.summary() for k, v in self.meshes.items()},
            "function_spaces": {k: v.summary() for k, v in self.function_spaces.items()},
            "functions": {k: v.summary() for k, v in self.functions.items()},
            "boundary_conditions": {k: v.summary() for k, v in self.bcs.items()},
            "forms": {k: v.summary() for k, v in self.forms.items()},
            "solutions": {k: v.summary() for k, v in self.solutions.items()},
            "mesh_tags": {k: v.summary() for k, v in self.mesh_tags.items()},
            "entity_maps": {k: v.summary() for k, v in self.entity_maps.items()},
            "ufl_symbols": list(self.ufl_symbols.keys()),
        }

    # --- Cleanup ---

    def cleanup(self) -> None:
        """Drop all references. Called on server shutdown."""
        self.meshes.clear()
        self.function_spaces.clear()
        self.functions.clear()
        self.bcs.clear()
        self.forms.clear()
        self.solutions.clear()
        self.mesh_tags.clear()
        self.entity_maps.clear()
        self.ufl_symbols.clear()
        self.solver_diagnostics.clear()
        self.log_buffer.clear()
        self.active_mesh = None

        # Postcondition: all registries empty
        for _name, _reg in [
            ("meshes", self.meshes),
            ("function_spaces", self.function_spaces),
            ("functions", self.functions),
            ("bcs", self.bcs),
            ("forms", self.forms),
            ("solutions", self.solutions),
            ("mesh_tags", self.mesh_tags),
            ("entity_maps", self.entity_maps),
            ("ufl_symbols", self.ufl_symbols),
            ("solver_diagnostics", self.solver_diagnostics),
            ("log_buffer", self.log_buffer),
        ]:
            if len(_reg) != 0:
                raise PostconditionError(
                    f"cleanup(): {_name} not empty ({len(_reg)} entries)"
                )
        if self.active_mesh is not None:
            raise PostconditionError(f"cleanup(): active_mesh still set to '{self.active_mesh}'")

        logger.info("Session state cleaned up")
