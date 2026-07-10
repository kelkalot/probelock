"""Tests for probelock doctor: toolset health + probe/tool drift."""

from probelock.doctor import drift, toolset_health


def _tool(name, props=None, required=None):
    fn = {"name": name, "description": "d",
          "parameters": {"type": "object", "properties": props or {}}}
    if required:
        fn["parameters"]["required"] = required
    return {"type": "function", "function": fn}


def _codes(findings):
    return sorted(f.code for f in findings)


# --- toolset health -------------------------------------------------------------


def test_health_flags_duplicate_tool_names():
    tools = [_tool("a", {"x": {"type": "string"}}, ["x"]),
             _tool("a", {"y": {"type": "string"}}, ["y"]),
             _tool("b", {"z": {"type": "string"}}, ["z"])]
    findings = toolset_health(tools)
    dup = [f for f in findings if f.code == "duplicate-tool"]
    assert dup and dup[0].level == "error"


def test_health_flags_too_few_tools():
    findings = toolset_health([_tool("a", {"x": {"type": "string"}}, ["x"])])
    assert "few-tools" in _codes(findings)


def test_health_flags_no_args_and_unconstrained_args():
    no_args = toolset_health([_tool("a"), _tool("b"), _tool("c")])
    assert _codes(no_args).count("no-args") == 3

    # genuinely unconstrained: properties present but carrying nothing the scorer
    # enforces (no type, no keyword) — arg_validity/required_args are vacuous
    unconstrained = toolset_health([
        _tool("a", {"x": {"description": "d"}}),
        _tool("b", {"y": {}}),
        _tool("c", {"z": {"description": "d"}})])
    assert _codes(unconstrained).count("unconstrained-args") == 3


def test_health_typed_args_are_constrained_incl_nested_and_array():
    # a bare `type` IS enforced (a wrong-typed value fails), and nested-object /
    # constrained-array-item properties count too — none of these warn
    tools = [
        _tool("a", {"city": {"type": "string"}}),                         # bare type
        _tool("b", {"cfg": {"type": "object",
                            "properties": {"mode": {"enum": ["fast", "slow"]}}}}),  # nested enum
        _tool("c", {"tags": {"type": "array", "items": {"type": "string"}}})]       # array items
    assert [f for f in toolset_health(tools) if f.code == "unconstrained-args"] == []


def test_health_format_only_property_is_not_credited_as_constrained():
    # `format` is NOT enforced by default jsonschema — a format-only (no type) property
    # is genuinely unconstrained and must be flagged, not given false confidence
    tools = [_tool("a", {"when": {"format": "date-time"}}),
             _tool("b", {"when": {"format": "date-time"}}),
             _tool("c", {"when": {"format": "date-time"}})]
    assert _codes(toolset_health(tools)).count("unconstrained-args") == 3


def test_health_clean_toolset_has_no_findings():
    tools = [_tool("a", {"x": {"type": "string", "enum": ["p", "q"]}}, ["x"]),
             _tool("b", {"y": {"type": "integer", "minimum": 0}}, ["y"]),
             _tool("c", {"z": {"type": "string"}}, ["z"])]  # bare type is enough
    assert toolset_health(tools) == []


# --- drift ----------------------------------------------------------------------


def test_drift_flags_removed_tool_as_error():
    live = [_tool("get_forecast", {"city": {"type": "string"}}, ["city"])]
    frozen = [("get_weather", {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]})]
    findings = drift(live, frozen)
    assert len(findings) == 1
    assert findings[0].level == "error" and findings[0].code == "tool-removed"


def test_drift_flags_changed_schema_as_warning():
    live = [_tool("send", {"text": {"type": "string"}, "cc": {"type": "string"}}, ["text"])]
    frozen = [("send", {"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"]})]  # no 'cc' at mint time
    findings = drift(live, frozen)
    assert len(findings) == 1
    assert findings[0].level == "warn" and findings[0].code == "schema-drift"


def test_drift_silent_when_schema_matches():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    live = [{"type": "function", "function": {"name": "get_weather", "parameters": schema}}]
    assert drift(live, [("get_weather", dict(schema))]) == []


def test_drift_ignores_reordered_required_and_enum():
    # required/enum are sets; a different element order is NOT drift
    live = [{"type": "function", "function": {"name": "send", "parameters": {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                       "mode": {"enum": ["x", "y", "z"]}},
        "required": ["a", "b"]}}}]
    frozen = [("send", {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                       "mode": {"enum": ["z", "y", "x"]}},   # reordered enum
        "required": ["b", "a"]})]                              # reordered required
    assert drift(live, frozen) == []


def test_drift_counts_multiple_frozen_probes_per_tool():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    findings = drift([], [("gone", schema), ("gone", schema), ("gone", schema)])
    assert "3 frozen probe(s)" in findings[0].message


# --- round-2 regressions: composition & ordering ------------------------------------


def test_health_composition_and_ref_args_are_constrained():
    # constraints hidden behind $ref / anyOf / allOf (Pydantic emits exactly these)
    # must NOT be flagged unconstrained — doctor stays conservative on composition
    ref_tool = _tool("a", {"color": {"$ref": "#/$defs/Color"}})
    ref_tool["function"]["parameters"]["$defs"] = {"Color": {"enum": ["r", "g", "b"]}}
    tools = [
        ref_tool,
        _tool("b", {"name": {"anyOf": [{"type": "string"}, {"type": "null"}]}}),
        _tool("c", {"mode": {"allOf": [{"enum": ["on", "off"]}]}})]
    assert [f for f in toolset_health(tools) if f.code == "unconstrained-args"] == []


def test_health_object_level_const_is_not_no_args():
    # a whole-arg const has no `properties` but IS constrained — not "no-args"
    t = {"type": "function", "function": {
        "name": "x", "parameters": {"type": "object", "const": {"mode": "on"}}}}
    assert [f for f in toolset_health([t, t, t]) if f.code in ("no-args", "unconstrained-args")] == []


def test_drift_ignores_reordered_type_list_and_composition():
    def tool(type_order, anyof_order):
        return {"type": "function", "function": {"name": "t", "parameters": {
            "type": "object",
            "properties": {"x": {"type": type_order},
                           "y": {"anyOf": anyof_order}}}}}
    live = [tool(["string", "null"], [{"type": "string"}, {"type": "null"}])]
    frozen = [("t", tool(["null", "string"], [{"type": "null"}, {"type": "string"}])["function"]["parameters"])]
    assert drift(live, frozen) == []  # reordered type-list and anyOf are not drift


def test_traces_frozen_tools_covers_structured_output_records():
    # a content-only (no tool_call) trace record mints a structured_output traced probe
    # pinning the matched tool's schema — drift must see it
    from probelock.models import ResponseMessage
    from probelock.traces import TraceRecord
    tools = [{"type": "function", "function": {"name": "emit", "description": "e",
              "parameters": {"type": "object", "required": ["n"],
                             "properties": {"n": {"type": "integer"}}}}}]
    rec = TraceRecord(id="r1", messages=[{"role": "user", "content": "go"}],
                      tools=tools, response=ResponseMessage(content='{"n": 5}', tool_calls=[]))
    from probelock.doctor import traces_frozen_tools
    frozen = traces_frozen_tools([rec])
    assert frozen and frozen[0][0] == "emit"
    # and drift against a toolset without 'emit' flags it
    assert [f for f in drift([], frozen) if f.code == "tool-removed"]


def test_drift_message_counts_only_stale_probes():
    live = [{"type": "function", "function": {"name": "send", "parameters": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
    current = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    stale = {"type": "object", "properties": {"text": {"type": "string"}}}  # no required
    findings = drift(live, [("send", current), ("send", current), ("send", stale)])
    assert len(findings) == 1
    assert "1 of 3" in findings[0].message  # only 1 of the 3 actually drifted
