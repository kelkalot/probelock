import json

from probelock import scoring
from probelock.models import Probe, ResponseMessage, ToolCall

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "start": {"type": "string"}},
    "required": ["title", "start"],
}


def tool_probe(capability):
    return Probe(
        id=f"{capability}::t",
        capability=capability,
        description="",
        messages=[],
        tools=[],
        expected_tool="t",
        schema=SCHEMA,
    )


def test_tool_selection():
    p = tool_probe("tool_selection")
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}")])) == 1.0
    assert scoring.score(p, ResponseMessage(content="I can help with that")) == 0.0
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("other", "{}")])) == 0.0


def test_tool_discrimination():
    p = tool_probe("tool_discrimination")  # expected_tool="t"
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}")])) == 1.0
    # right tool but ALSO a wrong one -> imprecise -> fail
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}"), ToolCall("x", "{}")])) == 0.0
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("x", "{}")])) == 0.0
    assert scoring.score(p, ResponseMessage(content="no call")) == 0.0


def test_tool_restraint():
    p = Probe(id="tool_restraint::x", capability="tool_restraint", description="",
              messages=[], tools=[], expected_tool=None)
    assert scoring.score(p, ResponseMessage(content="Paris.")) == 1.0  # no call -> good
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}")])) == 0.0  # over-trigger


def test_tool_permission():
    p = tool_probe("tool_permission")  # expected_tool="t" is the FORBIDDEN tool here
    assert scoring.score(p, ResponseMessage(content="not permitted")) == 1.0  # didn't call it
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("other", "{}")])) == 1.0  # other tool ok
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}")])) == 0.0  # called forbidden


def test_no_hallucinated_tool():
    offered = [
        {"type": "function", "function": {"name": "a", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "parameters": {}}},
    ]
    p = Probe(id="no_hallucinated_tool::a", capability="no_hallucinated_tool",
              description="", messages=[], tools=offered, expected_tool=None)
    assert scoring.score(p, ResponseMessage(content="no fit")) == 1.0  # declined
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("b", "{}")])) == 1.0  # offered tool
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("ghost", "{}")])) == 0.0  # fabricated


def test_needle_in_tools():
    p = tool_probe("needle_in_tools")  # expected_tool="t"
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}")])) == 1.0
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("distractor", "{}")])) == 0.0
    assert scoring.score(p, ResponseMessage(content="lost")) == 0.0


def test_arity_robustness():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"}},
        "required": ["a"],
    }
    p = Probe(id="arity_robustness::t", capability="arity_robustness", description="",
              messages=[], tools=[], expected_tool="t", schema=schema)
    import json as _json
    full = ResponseMessage(tool_calls=[ToolCall("t", _json.dumps({"a": "x", "b": "y", "c": "z"}))])
    partial = ResponseMessage(tool_calls=[ToolCall("t", _json.dumps({"a": "x", "b": "y"}))])  # missing c
    assert scoring.score(p, full) == 1.0  # every parameter filled
    assert scoring.score(p, partial) == 0.0  # one optional dropped -> fails


def test_arity_robustness_zero_arg_tool_requires_a_call():
    # A 0-property tool must still be CALLED to earn credit; "nothing to fill" is not
    # the same as "the model demonstrated it can fill every parameter."
    schema = {"type": "object", "properties": {}}
    p = Probe(id="arity_robustness::t", capability="arity_robustness", description="",
              messages=[], tools=[], expected_tool="t", schema=schema)
    assert scoring.score(p, ResponseMessage(tool_calls=[ToolCall("t", "{}")])) == 1.0
    assert scoring.score(p, ResponseMessage(content="no call")) == 0.0


def test_arity_robustness_checks_nested_optional_properties():
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
    p = Probe(id="arity_robustness::t", capability="arity_robustness", description="",
              messages=[], tools=[], expected_tool="t", schema=schema)
    missing_nested_optional = ResponseMessage(tool_calls=[ToolCall(
        "t", json.dumps({"name": "x", "settings": {"a": "y"}})  # nested "b" omitted
    )])
    full = ResponseMessage(tool_calls=[ToolCall(
        "t", json.dumps({"name": "x", "settings": {"a": "y", "b": 1}})
    )])
    assert scoring.score(p, missing_nested_optional) == 0.0
    assert scoring.score(p, full) == 1.0


def test_required_args_forced_empty_value_can_pass():
    # A schema whose only valid value for a required property IS empty (maxItems: 0)
    # must not be permanently unwinnable — the emptiness heuristic exists to catch a
    # lazy non-answer, not to reject the value the schema itself demands.
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "maxItems": 0}},
        "required": ["tags"],
    }
    p = Probe(id="required_args::t", capability="required_args", description="",
              messages=[], tools=[], expected_tool="t", schema=schema)
    resp = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"tags": []}))])
    assert scoring.score(p, resp) == 1.0


def test_arity_robustness_forced_empty_value_can_pass():
    schema = {"type": "object", "properties": {"note": {"const": ""}}}
    p = Probe(id="arity_robustness::t", capability="arity_robustness", description="",
              messages=[], tools=[], expected_tool="t", schema=schema)
    resp = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"note": ""}))])
    assert scoring.score(p, resp) == 1.0


def test_arg_validity():
    p = tool_probe("arg_validity")
    good = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"title": "x", "start": "y"}))])
    bad_type = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"title": 123, "start": "y"}))])
    bad_json = ResponseMessage(tool_calls=[ToolCall("t", "{not json")])
    assert scoring.score(p, good) == 1.0
    assert scoring.score(p, bad_type) == 0.0
    assert scoring.score(p, bad_json) == 0.0


def test_required_args():
    p = tool_probe("required_args")
    full = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"title": "x", "start": "y"}))])
    missing = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"title": "x"}))])
    empty = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"title": "", "start": "y"}))])
    assert scoring.score(p, full) == 1.0
    assert scoring.score(p, missing) == 0.0
    assert scoring.score(p, empty) == 0.0


def test_structured_output():
    p = Probe(
        id="structured_output::t",
        capability="structured_output",
        description="",
        messages=[],
        tools=[],
        schema=SCHEMA,
    )
    good = ResponseMessage(content=json.dumps({"title": "x", "start": "y"}))
    fenced = ResponseMessage(content='```json\n{"title":"x","start":"y"}\n```')
    prose = ResponseMessage(content="Sure, here is the JSON: {...}")
    assert scoring.score(p, good) == 1.0
    assert scoring.score(p, fenced) == 0.0
    assert scoring.score(p, prose) == 0.0


def test_format_adherence():
    p = Probe(
        id="format_adherence::t",
        capability="format_adherence",
        description="",
        messages=[],
        tools=[],
        expected_text="ACK",
    )
    assert scoring.score(p, ResponseMessage(content="ACK")) == 1.0
    assert scoring.score(p, ResponseMessage(content=" ACK ")) == 1.0  # surrounding space ok
    assert scoring.score(p, ResponseMessage(content="ack")) == 0.0  # case matters (exact)
    assert scoring.score(p, ResponseMessage(content="ACK!")) == 0.0
    assert scoring.score(p, ResponseMessage(content="ACK — happy to help!")) == 0.0


def test_arg_validity_accepts_dict_arguments():
    # Some OpenAI-compatible servers return tool arguments already parsed as a dict.
    p = tool_probe("arg_validity")
    good = ResponseMessage(tool_calls=[ToolCall("t", {"title": "x", "start": "y"})])
    assert scoring.score(p, good) == 1.0


def test_arg_validity_any_matching_call_counts():
    # A malformed first call then a corrected valid call -> still passes (any-match).
    p = tool_probe("arg_validity")
    resp = ResponseMessage(tool_calls=[
        ToolCall("t", json.dumps({"title": 123, "start": "y"})),  # invalid
        ToolCall("t", json.dumps({"title": "x", "start": "y"})),  # valid
    ])
    assert scoring.score(p, resp) == 1.0


def test_required_args_any_matching_call_counts():
    p = tool_probe("required_args")
    resp = ResponseMessage(tool_calls=[
        ToolCall("t", json.dumps({"title": "x"})),  # missing 'start'
        ToolCall("t", json.dumps({"title": "x", "start": "y"})),  # complete
    ])
    assert scoring.score(p, resp) == 1.0
