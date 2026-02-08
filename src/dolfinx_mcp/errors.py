"""Structured error hierarchy and tool error handling for DOLFINx MCP server."""

from __future__ import annotations

import functools
import inspect
import logging
import traceback
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class DOLFINxMCPError(Exception):
    """Base error for all DOLFINx MCP operations."""

    error_code: str = "DOLFINX_MCP_ERROR"
    suggestion: str = ""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        if suggestion is not None:
            self.suggestion = suggestion

    def to_dict(self) -> dict[str, str]:
        result: dict[str, str] = {
            "error": self.error_code,
            "message": str(self),
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


# --- Mesh errors ---


class NoActiveMeshError(DOLFINxMCPError):
    error_code = "NO_ACTIVE_MESH"
    suggestion = "Create a mesh first using create_unit_square or another mesh tool."


class MeshNotFoundError(DOLFINxMCPError):
    error_code = "MESH_NOT_FOUND"
    suggestion = "Check available meshes with get_session_state."


# --- Function space / function errors ---


class FunctionSpaceNotFoundError(DOLFINxMCPError):
    error_code = "FUNCTION_SPACE_NOT_FOUND"
    suggestion = "Create a function space first using create_function_space."


class FunctionNotFoundError(DOLFINxMCPError):
    error_code = "FUNCTION_NOT_FOUND"
    suggestion = "Check available functions with get_session_state."


# --- Expression / form errors ---


class InvalidUFLExpressionError(DOLFINxMCPError):
    error_code = "INVALID_UFL_EXPRESSION"
    suggestion = "Check UFL syntax. Example: inner(grad(u), grad(v)) * dx"


# --- Solver errors ---


class SolverError(DOLFINxMCPError):
    error_code = "SOLVER_ERROR"
    suggestion = "Try a different solver or preconditioner, or check boundary conditions."


# --- Name collision ---


class DuplicateNameError(DOLFINxMCPError):
    error_code = "DUPLICATE_NAME"
    suggestion = "Use a different name or remove the existing object first."


# --- DOLFINx API wrapper ---


class DOLFINxAPIError(DOLFINxMCPError):
    error_code = "DOLFINX_API_ERROR"


# --- File I/O ---


class FileIOError(DOLFINxMCPError):
    error_code = "FILE_IO_ERROR"
    suggestion = "Check file path and permissions. Output directory is /workspace."


# --- Contract violations ---


class PreconditionError(DOLFINxMCPError):
    """Raised when a tool receives invalid input parameters."""

    error_code = "PRECONDITION_VIOLATED"
    suggestion = "Check input parameters meet the documented requirements."


class PostconditionError(DOLFINxMCPError):
    """Raised when an operation produces an invalid result."""

    error_code = "POSTCONDITION_VIOLATED"
    suggestion = "Internal error: operation result failed validation. Report as bug."


class InvariantError(DOLFINxMCPError):
    """Raised when session state referential integrity is violated."""

    error_code = "INVARIANT_VIOLATED"
    suggestion = "Internal error: session state is inconsistent. Consider reset_session."


# --- Decorator ---


def handle_tool_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that catches DOLFINxMCPError and generic exceptions,
    returning structured error dicts instead of raising.

    Preserves __signature__ so FastMCP can generate JSON schemas from
    the decorated function's parameters.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except DOLFINxMCPError as exc:
            logger.warning("Tool %s raised %s: %s", fn.__name__, exc.error_code, exc)
            return exc.to_dict()
        except Exception as exc:
            logger.error(
                "Tool %s unexpected error: %s\n%s",
                fn.__name__,
                exc,
                traceback.format_exc(),
            )
            return {
                "error": "INTERNAL_ERROR",
                "message": f"Unexpected error: {exc}",
                "suggestion": "This may be a bug. Check server logs for details.",
            }

    # Preserve signature for FastMCP schema generation.
    # inspect.signature() is needed because plain functions don't have
    # __signature__ as an attribute -- Python generates it on the fly.
    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper
