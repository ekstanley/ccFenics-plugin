"""Result rendering for DOLFINx MCP tool results in Jupyter notebooks."""

from __future__ import annotations

import base64
from typing import Any


def render_result(items: list[dict[str, Any]], tool_name: str = "") -> None:
    """Render parsed MCP tool result items in the notebook output cell.

    Each item is a dict with ``type`` (``"text"``, ``"image"``, ``"error"``)
    and associated data.  Dispatches to the appropriate IPython display
    function.
    """
    if not isinstance(items, list):
        msg = f"Expected list of result items, got {type(items).__name__}"
        raise TypeError(msg)

    from IPython.display import HTML, Image, display

    for item in items:
        item_type = item.get("type", "text")

        if item_type == "error":
            _render_error(item.get("message", "Unknown error"))

        elif item_type == "image":
            img_bytes = base64.b64decode(item["data"])
            display(Image(data=img_bytes, format="png"))

        elif item_type == "text":
            data = item.get("data")
            if isinstance(data, dict):
                _render_dict(data)
            else:
                display(HTML(f"<pre>{data}</pre>"))


def _render_dict(data: dict[str, Any]) -> None:
    """Render a dict result as a compact HTML table."""
    from IPython.display import HTML, display

    rows = []
    for key, val in data.items():
        val_str = str(val)
        if len(val_str) > 120:
            val_str = val_str[:117] + "..."
        rows.append(
            f"<tr><td style='padding:2px 8px;font-weight:bold;'>{key}</td>"
            f"<td style='padding:2px 8px;'><code>{val_str}</code></td></tr>"
        )
    html = (
        "<table style='border-collapse:collapse;font-size:13px;'>"
        + "".join(rows)
        + "</table>"
    )
    display(HTML(html))


def _render_error(message: str) -> None:
    """Render an error message with visual emphasis."""
    from IPython.display import HTML, display

    html = (
        "<div style='border-left:4px solid #dc3545;padding:8px 16px;"
        "margin:8px 0;background:#fff5f5;'>"
        f"<strong style='color:#dc3545;'>Error</strong><br>"
        f"<span>{message}</span></div>"
    )
    display(HTML(html))


def render_tools_table(tools: dict[str, dict[str, Any]]) -> None:
    """Render a table of available MCP tools with descriptions."""
    from IPython.display import HTML, display

    rows = []
    for name, info in sorted(tools.items()):
        desc = (info.get("description") or "")[:100]
        rows.append(
            f"<tr><td style='padding:4px 8px;'><code>{name}</code></td>"
            f"<td style='padding:4px 8px;'>{desc}</td></tr>"
        )

    html = (
        "<table style='border-collapse:collapse;width:100%;font-size:13px;'>"
        "<thead><tr style='background:#f0f0f0;'>"
        "<th style='text-align:left;padding:4px 8px;'>Tool</th>"
        "<th style='text-align:left;padding:4px 8px;'>Description</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    display(HTML(html))
