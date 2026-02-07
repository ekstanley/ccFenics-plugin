"""Shared utilities for DOLFINx MCP tools.

Extracted from duplicated patterns across tool modules.
"""

from __future__ import annotations


def compute_l2_norm(function) -> float:
    """Compute L2 norm of a DOLFINx function via UFL integral.

    Args:
        function: A dolfinx.fem.Function object.

    Returns:
        L2 norm as a non-negative float.
    """
    import numpy as np
    import ufl
    from dolfinx.fem import assemble_scalar, form as compile_form

    l2_form = compile_form(ufl.inner(function, function) * ufl.dx)
    return float(np.sqrt(abs(assemble_scalar(l2_form))))
