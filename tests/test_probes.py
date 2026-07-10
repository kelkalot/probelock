import json
from pathlib import Path

import jsonschema
import pytest

from probelock.probes import (
    _number_value,
    derive_probes,
    synth_all_args,
    synth_args,
    synth_value,
    tools_fingerprint,
)

CONSTRAINED_SCHEMA = {
    "type": "object",
    "$defs": {"Priority": {"type": "string", "enum": ["low", "high"]}},
    "properties": {
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "mode": {"const": "fast"},
        "priority": {"$ref": "#/$defs/Priority"},
        "count": {"type": "integer", "minimum": 5},
        "ratio": {"type": "number", "exclusiveMinimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "code": {"type": "string", "minLength": 4},
        "when": {"type": "string", "format": "date-time"},
    },
    "required": ["unit", "mode", "priority", "count", "ratio", "tags", "code", "when"],
}

ROOT = Path(__file__).resolve().parents[1]
TOOLS = json.loads((ROOT / "examples" / "agent_tools.json").read_text())


def test_probe_counts():
    caps = {}
    for p in derive_probes(TOOLS):
        caps[p.capability] = caps.get(p.capability, 0) + 1
    assert caps == {
        "tool_selection": 3,
        "tool_discrimination": 3,
        "needle_in_tools": 3,
        "arity_robustness": 3,
        "tool_permission": 3,
        "no_hallucinated_tool": 3,
        "arg_validity": 3,
        "required_args": 3,
        "structured_output": 3,
        "tool_restraint": 3,
        "format_adherence": 2,
    }


def test_ids_stable_and_unique():
    a = [p.id for p in derive_probes(TOOLS)]
    b = [p.id for p in derive_probes(TOOLS)]
    assert a == b  # derivation is deterministic
    assert len(a) == len(set(a))  # ids are unique


def test_synth_args_valid_against_schema():
    for tool in TOOLS:
        schema = tool["function"]["parameters"]
        jsonschema.validate(synth_args(schema), schema)  # must not raise


def test_synth_args_valid_against_CONSTRAINED_schema():
    # enum / const / $ref / minimum / exclusiveMinimum / minItems / minLength —
    # the cases the old type-only synth_value silently violated.
    jsonschema.validate(synth_args(CONSTRAINED_SCHEMA), CONSTRAINED_SCHEMA)


def test_synth_value_respects_maximum_with_multiple_of():
    # multipleOf must not overshoot an upper bound (was 6 for max=5,multipleOf=2).
    for schema in (
        {"type": "integer", "maximum": 5, "multipleOf": 2},
        {"type": "integer", "maximum": 95, "multipleOf": 10},
        {"type": "integer", "maximum": -20, "multipleOf": 3},
        {"type": "integer", "minimum": 4, "multipleOf": 2},  # lower-bound case still valid
    ):
        jsonschema.validate(synth_value(schema), schema)


def test_synth_value_resolves_ref_and_const_and_enum():
    root = {"$defs": {"P": {"const": "x"}}}
    assert synth_value({"$ref": "#/$defs/P"}, root) == "x"
    assert synth_value({"const": 7}) == 7
    assert synth_value({"type": "string", "enum": ["a", "b"]}) == "a"


def test_synth_value_merges_allOf_branches():
    # allOf requires satisfying every branch simultaneously; taking only the first
    # branch (as anyOf/oneOf correctly do) used to drop a later branch's required
    # property, producing a reference value that fails its own schema.
    schema = {
        "allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
        ]
    }
    value = synth_value(schema)
    jsonschema.validate(value, schema)  # must not raise
    assert "a" in value and "b" in value


def test_synth_args_handles_nested_allOf_property():
    # The realistic trigger: allOf extending a $ref'd base schema on a NESTED property.
    schema = {
        "type": "object",
        "properties": {
            "x": {
                "allOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                    {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
                ]
            }
        },
        "required": ["x"],
    }
    jsonschema.validate(synth_args(schema), schema)


def test_number_value_never_undershoots_minimum_for_unsatisfiable_multiple_of():
    # No multiple of 10 exists in [1, 9]; the synthesized value must still respect the
    # minimum rather than silently flooring below it (there's no value that can satisfy
    # both bounds, but underflowing the minimum is strictly worse than overflowing the
    # maximum since minimum violations are the more common real-world constraint).
    value = _number_value({"minimum": 1, "maximum": 9, "multipleOf": 10}, integer=True)
    assert value >= 1


def test_synth_all_args_fills_nested_optional_properties():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "settings": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            },
        },
        "required": ["name"],
    }
    full = synth_all_args(schema)
    assert "b" in full["settings"]  # optional nested property is filled too, not just "a"
    jsonschema.validate(full, schema)


def test_duplicate_tool_names_raise():
    dup = TOOLS[:1] + TOOLS[:1]  # same tool twice
    with pytest.raises(ValueError, match="duplicate tool name"):
        derive_probes(dup)


def test_fingerprint_changes_with_toolset():
    assert tools_fingerprint(TOOLS) != tools_fingerprint(TOOLS[:2])
    assert tools_fingerprint(TOOLS) == tools_fingerprint(TOOLS)


def test_fingerprint_is_order_invariant():
    # Reordering the tools file must not change the fingerprint, because the
    # derived probe battery is identical (derive_probes sorts by name).
    assert tools_fingerprint(TOOLS) == tools_fingerprint(list(reversed(TOOLS)))


def test_json_mode_probes_are_opt_in():
    without = [p for p in derive_probes(TOOLS) if p.capability == "json_mode"]
    assert without == []  # off by default
    with_jm = [p for p in derive_probes(TOOLS, json_mode=True) if p.capability == "json_mode"]
    assert len(with_jm) == len(TOOLS)
    p = with_jm[0]
    assert p.tools == []  # no tools offered; native structured-output path
    assert p.response_format["type"] == "json_schema"
    assert p.response_format["json_schema"]["schema"] == p.schema
    assert p.id.startswith("json_mode::")


def test_synth_value_terminates_on_recursive_ref():
    # a self-referential $ref (Pydantic tree/linked-list model) must not RecursionError
    schema = {"type": "object", "required": ["root"],
              "properties": {"root": {"$ref": "#/$defs/Node"}},
              "$defs": {"Node": {"type": "object", "required": ["child"],
                                 "properties": {"child": {"$ref": "#/$defs/Node"}}}}}
    value = synth_value(schema)  # bounded depth -> returns, does not blow the stack
    assert isinstance(value, dict)


def test_derive_probes_handles_recursive_tool_schema():
    tools = [{"type": "function", "function": {"name": "build_tree", "description": "t",
              "parameters": {"type": "object", "required": ["root"],
                             "properties": {"root": {"$ref": "#/$defs/Node"}},
                             "$defs": {"Node": {"type": "object", "required": ["child"],
                                                "properties": {"child": {"$ref": "#/$defs/Node"}}}}}}}]
    assert derive_probes(tools)  # no RecursionError
