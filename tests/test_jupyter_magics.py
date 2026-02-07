"""Tests for dolfinx_mcp_jupyter.magics -- kwarg parsing and utilities."""

from __future__ import annotations

from dolfinx_mcp_jupyter.magics import _parse_kwargs


class TestParseKwargsEmpty:
    """Empty or no-op inputs."""

    def test_empty_list(self):
        assert _parse_kwargs([]) == {}

    def test_no_equals(self):
        assert _parse_kwargs(["foobar"]) == {}

    def test_multiple_no_equals(self):
        assert _parse_kwargs(["foo", "bar", "baz"]) == {}


class TestParseKwargsTypes:
    """Type coercion for different value types."""

    def test_string_value(self):
        assert _parse_kwargs(["name=mesh1"]) == {"name": "mesh1"}

    def test_int_value(self):
        assert _parse_kwargs(["nx=16"]) == {"nx": 16}

    def test_negative_int(self):
        assert _parse_kwargs(["offset=-5"]) == {"offset": -5}

    def test_float_value(self):
        assert _parse_kwargs(["rtol=1e-10"]) == {"rtol": 1e-10}

    def test_float_with_decimal(self):
        assert _parse_kwargs(["value=3.14"]) == {"value": 3.14}

    def test_bool_true(self):
        assert _parse_kwargs(["inline=true"]) == {"inline": True}

    def test_bool_false(self):
        assert _parse_kwargs(["show_mesh=false"]) == {"show_mesh": False}

    def test_bool_case_insensitive(self):
        assert _parse_kwargs(["flag=True"]) == {"flag": True}
        assert _parse_kwargs(["flag=FALSE"]) == {"flag": False}


class TestParseKwargsJSON:
    """JSON values (lists, dicts)."""

    def test_json_list(self):
        result = _parse_kwargs(['points=[[0.5,0.5]]'])
        assert result == {"points": [[0.5, 0.5]]}

    def test_json_dict(self):
        result = _parse_kwargs(['opts={"a":1}'])
        assert result == {"opts": {"a": 1}}


class TestParseKwargsExpressions:
    """UFL/math expression strings should remain as strings."""

    def test_ufl_expression(self):
        result = _parse_kwargs(['value=2*pi**2*sin(pi*x[0])'])
        assert result == {"value": "2*pi**2*sin(pi*x[0])"}

    def test_inner_grad_expression(self):
        result = _parse_kwargs(['bilinear=inner(grad(u), grad(v)) * dx'])
        assert result == {"bilinear": "inner(grad(u), grad(v)) * dx"}

    def test_boundary_expression(self):
        result = _parse_kwargs(['boundary=np.isclose(x[0], 0.0)'])
        assert result == {"boundary": "np.isclose(x[0], 0.0)"}


class TestParseKwargsMultiple:
    """Multiple key=value pairs."""

    def test_mixed_types(self):
        result = _parse_kwargs([
            "name=mesh1",
            "nx=16",
            "ny=16",
            "cell_type=triangle",
        ])
        assert result == {
            "name": "mesh1",
            "nx": 16,
            "ny": 16,
            "cell_type": "triangle",
        }

    def test_mixed_with_bool_and_float(self):
        result = _parse_kwargs([
            "show_mesh=true",
            "colormap=viridis",
            "scale=0.5",
        ])
        assert result == {
            "show_mesh": True,
            "colormap": "viridis",
            "scale": 0.5,
        }
