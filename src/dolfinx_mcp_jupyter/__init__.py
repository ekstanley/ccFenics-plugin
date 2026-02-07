"""DOLFINx MCP JupyterLab integration -- IPython magics for FEA via MCP.

Usage in a Jupyter notebook::

    %load_ext dolfinx_mcp_jupyter
    %dolfinx_connect
    %dolfinx_tools
    %dolfinx create_unit_square name=mesh nx=16 ny=16
"""

from __future__ import annotations

__version__ = "0.1.0"


def load_ipython_extension(ipython: object) -> None:
    """Register DOLFINx MCP magics with the IPython shell.

    Called automatically by ``%load_ext dolfinx_mcp_jupyter``.
    """
    from .magics import DOLFINxMagics

    ipython.register_magics(DOLFINxMagics)  # type: ignore[attr-defined]
