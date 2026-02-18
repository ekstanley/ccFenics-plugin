"""Shared precondition validators for tool modules."""

from __future__ import annotations

from ..errors import PreconditionError


def require_nonempty(value: str, name: str) -> None:
    """Require a non-empty, non-whitespace string."""
    if not value or not value.strip():
        raise PreconditionError(f"{name} must be non-empty.")


def require_positive(value: int | float, name: str, limit: int | float | None = None) -> None:
    """Require a positive numeric value, optionally bounded."""
    if value <= 0:
        raise PreconditionError(f"{name} must be > 0, got {value}.")
    if limit is not None and value > limit:
        raise PreconditionError(f"{name} {value} exceeds limit {limit}.")


def require_choice(value: str, choices: set[str] | frozenset[str], name: str) -> None:
    """Require value to be in a set of valid choices."""
    if value not in choices:
        raise PreconditionError(
            f"{name} must be one of {sorted(choices)}, got '{value}'."
        )


def require_finite(value: float, name: str) -> None:
    """Require a finite numeric value (not NaN or Inf)."""
    import math
    if not math.isfinite(value):
        raise PreconditionError(f"{name} must be finite, got {value}.")
