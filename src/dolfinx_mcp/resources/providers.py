"""URI-based resource providers for DOLFINx MCP server."""

from __future__ import annotations

from typing import Any

from .. import __version__
from .._app import mcp


@mcp.resource("dolfinx://capabilities")
def get_capabilities() -> dict[str, Any]:
    """Supported element families, formats, solvers, and features."""
    return {
        "element_families": [
            {
                "name": "Lagrange",
                "description": "Standard continuous Lagrange elements",
                "degrees": "1-5",
            },
            {"name": "DG", "description": "Discontinuous Galerkin elements", "degrees": "0-5"},
            {
                "name": "N1curl",
                "description": "Nedelec edge elements (first kind)",
                "degrees": "1-3",
            },
            {"name": "RT", "description": "Raviart-Thomas face elements", "degrees": "1-3"},
            {"name": "BDM", "description": "Brezzi-Douglas-Marini elements", "degrees": "1-3"},
            {"name": "CR", "description": "Crouzeix-Raviart elements", "degrees": "1"},
        ],
        "mesh_types": [
            "unit_square (triangle, quadrilateral)",
            "unit_cube (tetrahedron, hexahedron)",
            "rectangle",
            "box",
            "custom (Gmsh import)",
        ],
        "solvers": {
            "direct": ["lu", "mumps", "superlu_dist"],
            "iterative_ksp": ["cg", "gmres", "bicgstab", "minres", "richardson"],
            "preconditioners": ["lu", "ilu", "jacobi", "sor", "hypre", "gamg", "none"],
            "time_integration": {
                "methods": ["backward_euler", "forward_euler", "crank_nicolson", "custom"],
                "adaptive_timestepping": "supported",
            },
        },
        "export_formats": ["xdmf", "vtk", "vtkhdf"],
        "norm_types": ["L2", "H1"],
        "version": {
            "dolfinx_mcp": __version__,
            "dolfinx_target": "0.10.0",
        },
    }


@mcp.resource("dolfinx://mesh/schema")
def get_mesh_schema() -> dict[str, Any]:
    """JSON schema describing mesh objects and their properties."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DOLFINx Mesh Schema",
        "description": "Schema for mesh objects in DOLFINx MCP session",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique identifier for the mesh",
            },
            "type": {
                "type": "string",
                "enum": ["unit_square", "unit_cube", "rectangle", "box", "custom"],
                "description": "Mesh generation method",
            },
            "cell_type": {
                "type": "string",
                "enum": ["triangle", "quadrilateral", "tetrahedron", "hexahedron"],
                "description": "Element shape",
            },
            "num_cells": {
                "type": "integer",
                "minimum": 1,
                "description": "Total number of cells in mesh",
            },
            "num_vertices": {
                "type": "integer",
                "minimum": 1,
                "description": "Total number of vertices in mesh",
            },
            "topology_dim": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "Topological dimension (1=interval, 2=surface, 3=volume)",
            },
            "geometry_dim": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "Geometric dimension of embedding space",
            },
            "tags": {
                "type": "object",
                "description": "Named region markers for boundary conditions",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "dim": {"type": "integer", "minimum": 0},
                        "entities": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
        },
        "required": ["name", "type", "cell_type", "num_cells", "num_vertices"],
        "examples": [
            {
                "name": "mesh_01",
                "type": "unit_square",
                "cell_type": "triangle",
                "num_cells": 128,
                "num_vertices": 81,
                "topology_dim": 2,
                "geometry_dim": 2,
                "tags": {
                    "left": {"dim": 1, "entities": [1, 2, 3]},
                    "right": {"dim": 1, "entities": [4, 5, 6]},
                },
            }
        ],
    }


@mcp.resource("dolfinx://solution/schema")
def get_solution_schema() -> dict[str, Any]:
    """JSON schema describing solution objects and their properties."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DOLFINx Solution Schema",
        "description": "Schema for solution objects in DOLFINx MCP session",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique identifier for the solution",
            },
            "mesh": {
                "type": "string",
                "description": "Associated mesh name",
            },
            "function_space": {
                "type": "object",
                "properties": {
                    "element_family": {"type": "string"},
                    "degree": {"type": "integer", "minimum": 0},
                    "dim": {"type": "integer", "minimum": 1},
                },
                "description": "Finite element space definition",
            },
            "problem_type": {
                "type": "string",
                "enum": ["steady", "transient"],
                "description": "Time-dependent or steady-state problem",
            },
            "time_info": {
                "type": "object",
                "properties": {
                    "current_time": {"type": "number"},
                    "timestep": {"type": "number"},
                    "num_steps": {"type": "integer"},
                },
                "description": "Time integration parameters (transient only)",
            },
            "dofs": {
                "type": "integer",
                "minimum": 1,
                "description": "Total degrees of freedom",
            },
            "norm": {
                "type": "object",
                "properties": {
                    "L2": {"type": "number"},
                    "H1": {"type": "number"},
                },
                "description": "Solution norms",
            },
            "exported": {
                "type": "boolean",
                "description": "Whether solution has been exported to file",
            },
        },
        "required": ["name", "mesh", "function_space", "dofs"],
        "examples": [
            {
                "name": "u_solution",
                "mesh": "mesh_01",
                "function_space": {
                    "element_family": "Lagrange",
                    "degree": 1,
                    "dim": 1,
                },
                "problem_type": "steady",
                "dofs": 81,
                "norm": {"L2": 0.123, "H1": 0.456},
                "exported": True,
            }
        ],
    }


@mcp.resource("dolfinx://tags/schema")
def get_tags_schema() -> dict[str, Any]:
    """JSON schema describing mesh tags and region markers."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DOLFINx Mesh Tags Schema",
        "description": "Schema for mesh region markers used in boundary conditions",
        "type": "object",
        "properties": {
            "tag_name": {
                "type": "string",
                "description": "Human-readable identifier for the tagged region",
            },
            "dimension": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": (
                    "Topological dimension of tagged entities"
                    " (0=vertex, 1=edge, 2=face, 3=cell)"
                ),
            },
            "entities": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": "Entity indices belonging to this tag",
            },
            "usage": {
                "type": "string",
                "enum": ["boundary_condition", "subdomain", "material", "interface", "custom"],
                "description": "Intended use of the tag",
            },
        },
        "required": ["tag_name", "dimension", "entities"],
        "notes": [
            "Tags are typically created during mesh generation",
            "Use tags to apply boundary conditions via dolfinx.fem.locate_dofs_topological",
            "Common tag names: 'left', 'right', 'top', 'bottom', 'front', 'back', 'inner', 'outer'",
            "For custom Gmsh meshes, tags correspond to Physical Groups",
        ],
        "examples": [
            {
                "tag_name": "left_boundary",
                "dimension": 1,
                "entities": [0, 1, 2, 3, 4],
                "usage": "boundary_condition",
            },
            {
                "tag_name": "fluid_region",
                "dimension": 2,
                "entities": [10, 11, 12, 13, 14, 15],
                "usage": "subdomain",
            },
        ],
    }


@mcp.resource("dolfinx://log/info")
def get_log_info() -> dict[str, Any]:
    """Logging configuration and level information."""
    return {
        "description": "DOLFINx MCP Server logging configuration",
        "levels": {
            "DEBUG": "Detailed diagnostic information for debugging",
            "INFO": "General informational messages about operations",
            "WARNING": "Warning messages for potentially problematic situations",
            "ERROR": "Error messages for failures that don't stop execution",
            "CRITICAL": "Critical errors that may cause termination",
        },
        "current_level": "INFO",
        "log_destinations": [
            "stderr (console output)",
            "MCP protocol messages",
        ],
        "features": {
            "structured_logging": True,
            "timestamp": True,
            "context_info": "includes session_id and operation context",
        },
        "usage": {
            "view_logs": "Use dolfinx_get_session_state tool to see recent logs",
            "change_level": "Currently not configurable at runtime (set via environment)",
        },
        "best_practices": [
            "DEBUG level for development and troubleshooting",
            "INFO level for normal operation",
            "Check logs after errors for diagnostic information",
            "Log messages include tool names and operation IDs for tracing",
        ],
    }


@mcp.resource("dolfinx://api/reference")
def get_api_reference() -> dict[str, Any]:
    """Quick reference guide for all available MCP tools."""
    return {
        "categories": {
            "mesh_operations": {
                "tools": [
                    {
                        "name": "dolfinx_create_mesh",
                        "purpose": "Generate or import computational meshes",
                        "key_params": ["mesh_type", "nx", "cell_type"],
                        "returns": "mesh_name",
                    },
                    {
                        "name": "dolfinx_get_mesh_info",
                        "purpose": "Query mesh properties and statistics",
                        "key_params": ["mesh_name"],
                        "returns": "cells, vertices, dimension, tags",
                    },
                    {
                        "name": "dolfinx_refine_mesh",
                        "purpose": "Uniform or adaptive mesh refinement",
                        "key_params": ["mesh_name", "refinement_type"],
                        "returns": "refined_mesh_name",
                    },
                ],
            },
            "fem_operations": {
                "tools": [
                    {
                        "name": "dolfinx_create_function_space",
                        "purpose": "Define finite element function space",
                        "key_params": ["mesh_name", "element_family", "degree"],
                        "returns": "space_name",
                    },
                    {
                        "name": "dolfinx_define_variational_form",
                        "purpose": "Set up weak form of PDE",
                        "key_params": ["function_space", "bilinear_form", "linear_form"],
                        "returns": "form_id",
                    },
                ],
            },
            "boundary_conditions": {
                "tools": [
                    {
                        "name": "dolfinx_apply_dirichlet_bc",
                        "purpose": "Apply essential boundary conditions",
                        "key_params": ["function_space", "value", "boundary_tag"],
                        "returns": "bc_id",
                    },
                ],
            },
            "solvers": {
                "tools": [
                    {
                        "name": "dolfinx_solve_linear",
                        "purpose": "Solve linear system Au=b",
                        "key_params": ["form_id", "solver_type", "preconditioner"],
                        "returns": "solution_name",
                    },
                    {
                        "name": "dolfinx_solve_nonlinear",
                        "purpose": "Solve nonlinear problem F(u)=0",
                        "key_params": ["form_id", "method", "tolerance"],
                        "returns": "solution_name",
                    },
                    {
                        "name": "dolfinx_solve_time_dependent",
                        "purpose": "Time integration for transient problems",
                        "key_params": ["form_id", "dt", "num_steps", "theta"],
                        "returns": "solution_name",
                    },
                ],
            },
            "postprocessing": {
                "tools": [
                    {
                        "name": "dolfinx_compute_norm",
                        "purpose": "Calculate L2 or H1 norm of solution",
                        "key_params": ["solution_name", "norm_type"],
                        "returns": "norm_value",
                    },
                    {
                        "name": "dolfinx_export_solution",
                        "purpose": "Save solution to visualization file",
                        "key_params": ["solution_name", "filename", "format"],
                        "returns": "file_path",
                    },
                    {
                        "name": "dolfinx_compute_error",
                        "purpose": "Compare solution with exact/reference",
                        "key_params": ["solution_name", "exact_expr", "norm_type"],
                        "returns": "error_value",
                    },
                ],
            },
            "session_management": {
                "tools": [
                    {
                        "name": "dolfinx_get_session_state",
                        "purpose": "View all meshes, solutions, and logs",
                        "key_params": [],
                        "returns": "session_state",
                    },
                    {
                        "name": "dolfinx_clear_session",
                        "purpose": "Reset session and free resources",
                        "key_params": [],
                        "returns": "status",
                    },
                ],
            },
        },
        "typical_workflow": [
            "1. Create mesh: dolfinx_create_mesh",
            "2. Define function space: dolfinx_create_function_space",
            "3. Set up variational form: dolfinx_define_variational_form",
            "4. Apply boundary conditions: dolfinx_apply_dirichlet_bc",
            "5. Solve: dolfinx_solve_linear or dolfinx_solve_nonlinear",
            "6. Postprocess: dolfinx_compute_norm, dolfinx_export_solution",
            "7. Verify: dolfinx_compute_error (if exact solution known)",
        ],
        "helpful_tips": [
            "Use dolfinx://capabilities resource to see supported element families",
            "Check dolfinx://mesh/schema for mesh object structure",
            "Use get_session_state frequently to track created objects",
            "Export solutions to XDMF for ParaView visualization",
            "Tags are essential for applying boundary conditions",
        ],
    }
