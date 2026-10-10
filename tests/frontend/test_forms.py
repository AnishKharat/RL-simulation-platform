import pytest

from rcl.frontend.app import field_errors
from rcl.frontend.forms import field_spec, parse_array

DEFS = {"Mode": {"enum": ["a", "b"], "type": "string", "title": "Mode"}}


def spec(prop):
    return field_spec("p", prop, DEFS)


def test_field_spec_covers_the_registry_subset():
    assert spec({"type": "number", "default": 1.0, "exclusiveMinimum": 0}) | {} == {
        "name": "p", "title": "p", "description": None, "default": 1.0,
        "kind": "float", "min": None, "max": None,
    }  # fmt: skip
    assert spec({"type": "integer", "minimum": 1, "maximum": 9})["max"] == 9
    assert spec({"enum": ["cpu", "cuda"], "default": "cpu"})["options"] == ["cpu", "cuda"]
    ref = spec({"$ref": "#/$defs/Mode", "default": "b"})
    assert ref["kind"] == "enum" and ref["default"] == "b" and ref["options"] == ["a", "b"]
    opt = spec({"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None})
    assert opt["kind"] == "optional" and opt["inner"]["kind"] == "int"
    tup = spec({"type": "array", "prefixItems": [{"type": "number"}] * 3, "minItems": 3})
    assert (tup["kind"], tup["item_type"], tup["length"]) == ("array", "number", 3)
    var = spec({"type": "array", "items": {"type": "string"}})
    assert (var["item_type"], var["length"]) == ("string", None)
    with pytest.raises(ValueError):
        spec({"type": "object"})


def test_parse_array():
    assert parse_array("1, 2.5", "number", 2) == [1.0, 2.5]
    assert parse_array("a,b", "string", None) == ["a", "b"]
    with pytest.raises(ValueError, match="expected 3"):
        parse_array("1,2", "number", 3)
    with pytest.raises(ValueError, match="type number"):
        parse_array("x", "number", None)


def test_field_errors_map_params_to_widgets():
    detail = [
        {"loc": ["body", "controller", "params", "kp"], "msg": "bad kp", "type": "t"},
        {"loc": ["body", "controller", "id"], "msg": "not compatible", "type": "incompatible"},
        {"loc": ["body", "training"], "msg": "missing", "type": "missing"},
    ]
    per, general = field_errors(detail)
    assert per == {"environment": {}, "controller": {"kp": "bad kp"}}
    assert general == ["controller.id: not compatible", "training: missing"]
