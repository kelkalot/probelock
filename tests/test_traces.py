import json
from pathlib import Path

import pytest

from probelock import scoring
from probelock.models import ResponseMessage, ToolCall
from probelock.traces import (
    TraceRecord,
    derive_traced_probes,
    load_trace_records,
    traces_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACES = ROOT / "fixtures" / "sample_traces.json"

TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "start": {"type": "string"}},
    "required": ["title", "start"],
}

TOOLS = [{"type": "function", "function": {"name": "t", "parameters": TOOL_CALL_SCHEMA}}]


def _record(id="r", messages=None, tools=None, content=None, tool_calls=None):
    return TraceRecord(
        id=id,
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        tools=tools if tools is not None else TOOLS,
        response=ResponseMessage(
            content=content,
            tool_calls=[ToolCall(n, a) for n, a in (tool_calls or [])],
        ),
    )


def test_load_trace_records_parses_sample_fixture():
    records = load_trace_records(SAMPLE_TRACES)
    assert len(records) == 3
    ids = {r.id for r in records}
    assert "schedule-followup-after-clarification" in ids
    calling = next(r for r in records if r.id == "schedule-followup-after-clarification")
    assert calling.response.tool_calls[0].name == "create_calendar_event"
    assert len(calling.messages) == 3  # the multi-turn clarification is preserved


def test_load_trace_records_rejects_missing_records_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "records"}))
    with pytest.raises(ValueError):
        load_trace_records(bad)


def test_load_trace_records_rejects_record_without_response(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"records": [{"messages": []}]}))
    with pytest.raises(ValueError):
        load_trace_records(bad)


def test_load_trace_records_defaults_missing_id_to_stable_content_hash(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"records": [
        {"messages": [], "tools": [], "response": {"content": "x", "tool_calls": []}}
    ]}))
    a = load_trace_records(f)[0].id
    b = load_trace_records(f)[0].id
    assert a == b  # deterministic, not e.g. a random uuid


def test_load_trace_records_preserves_an_explicit_falsy_id(tmp_path):
    # 0 (and "", False) are valid author-supplied ids, not a signal to fall back to a
    # content hash — only a genuinely missing/null "id" should trigger that fallback.
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"records": [
        {"id": 0, "messages": [], "tools": [], "response": {"content": "x", "tool_calls": []}}
    ]}))
    assert load_trace_records(f)[0].id == "0"


def test_load_trace_records_rejects_duplicate_ids(tmp_path):
    # Probe ids are f"{capability}::traced::{id}"; duplicate record ids would collide the
    # same way derive_probes() guards against duplicate tool names.
    f = tmp_path / "t.json"
    record = {"id": "dup", "messages": [], "tools": [], "response": {"content": "x", "tool_calls": []}}
    f.write_text(json.dumps({"records": [record, dict(record)]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_trace_records(f)


def test_derive_traced_probes_builds_tool_call_cluster():
    record = _record(tool_calls=[("t", json.dumps({"title": "x", "start": "y"}))])
    probes = derive_traced_probes([record])
    caps = {p.capability for p in probes}
    assert caps == {"tool_selection", "tool_discrimination", "arg_validity", "required_args"}
    for p in probes:
        assert p.id == f"{p.capability}::traced::r"
        assert p.expected_tool == "t"
        assert p.messages == record.messages
        assert p.tools == TOOLS
        assert p.schema == TOOL_CALL_SCHEMA
        assert p.reference == {"valid_args": {"title": "x", "start": "y"}}


def test_derive_traced_probes_excludes_arity_robustness():
    # arity_robustness needs its own "fill EVERY parameter, including optional ones"
    # instruction to mean anything -- a real conversation was never asked for that, so it
    # doesn't transfer to passive replay the way tool_selection/arg_validity/etc. do.
    record = _record(tool_calls=[("t", json.dumps({"title": "x", "start": "y"}))])
    caps = {p.capability for p in derive_traced_probes([record])}
    assert "arity_robustness" not in caps


def test_derive_traced_probes_skips_call_to_a_tool_not_in_its_own_tools_list():
    # A malformed/incomplete record: the tool actually called isn't offered in this
    # record's own `tools` — replaying it couldn't meaningfully test tool_selection.
    record = _record(tools=[], tool_calls=[("ghost", "{}")])
    assert derive_traced_probes([record]) == []


def test_derive_traced_probes_handles_a_tool_with_no_recorded_schema():
    # The call matches a tool name that IS offered, but that tool has no "parameters"
    # recorded (e.g. a genuinely zero-arg tool, or an incomplete export). All capabilities
    # still get generated; scoring already treats an empty/missing schema as a trivial
    # pass, same as it does for a synthetic zero-property tool — no crash, no vacuous risk.
    bare_tools = [{"type": "function", "function": {"name": "t"}}]  # no "parameters"
    record = _record(tools=bare_tools, tool_calls=[("t", "{}")])
    probes = derive_traced_probes([record])
    caps = {p.capability for p in probes}
    assert caps == {"tool_selection", "tool_discrimination", "arg_validity", "required_args"}
    for p in probes:
        assert p.schema == {}  # the tool was found, but it recorded no "parameters"


def test_load_trace_records_handles_explicit_null_name_and_arguments(tmp_path):
    # `.get(key, default)` only applies the default when the key is missing, not when
    # it's present-but-null — a naive fix would turn a null name into the string "None".
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"records": [{
        "messages": [], "tools": [],
        "response": {"content": None, "tool_calls": [{"name": None, "arguments": None}]},
    }]}))
    call = load_trace_records(f)[0].response.tool_calls[0]
    assert call.name == ""
    assert call.arguments == "{}"


def test_derive_traced_probes_structured_output_skips_schema_less_tool():
    # A tool with no "parameters" validates ANY JSON trivially — if it's scanned first,
    # it would mask a later tool's real, more specific schema.
    bare_tool = {"type": "function", "function": {"name": "ping"}}  # no "parameters"
    record = _record(
        tools=[bare_tool, TOOLS[0]],
        content=json.dumps({"title": "x", "start": "y"}),
        tool_calls=[],
    )
    probes = derive_traced_probes([record])
    assert len(probes) == 1
    assert probes[0].schema == TOOL_CALL_SCHEMA  # matched "t", not the schema-less "ping"


def test_derive_traced_probes_structured_output_survives_a_malformed_schema():
    # jsonschema.validate() raises SchemaError (not ValidationError) for a schema that
    # is itself invalid per the JSON Schema meta-schema — a plausible outcome of an
    # automated/lossy OTel-export conversion. Must not crash derivation.
    bad_tool = {
        "type": "function",
        "function": {"name": "t", "parameters": {"type": "not-a-real-json-schema-type"}},
    }
    record = _record(tools=[bad_tool], content=json.dumps({"a": 1}), tool_calls=[])
    assert derive_traced_probes([record]) == []


def test_derive_traced_probes_structured_output_from_text_response():
    record = _record(
        tools=TOOLS,
        content=json.dumps({"title": "x", "start": "y"}),
        tool_calls=[],
    )
    probes = derive_traced_probes([record])
    assert len(probes) == 1
    p = probes[0]
    assert p.capability == "structured_output"
    assert p.expected_tool is None
    assert p.schema == TOOL_CALL_SCHEMA


def test_derive_traced_probes_no_call_and_non_json_content_yields_nothing():
    record = _record(content="just some prose", tool_calls=[])
    assert derive_traced_probes([record]) == []


def test_traced_probes_score_against_the_recorded_behavior():
    # The whole point: a candidate that reproduces the recorded call scores 1.0 through
    # the SAME scorers used for synthetic probes; a divergent one scores 0.0.
    record = _record(tool_calls=[("t", json.dumps({"title": "x", "start": "y"}))])
    probe = next(p for p in derive_traced_probes([record]) if p.capability == "tool_selection")
    good = ResponseMessage(tool_calls=[ToolCall("t", json.dumps({"title": "x", "start": "y"}))])
    bad = ResponseMessage(content="I can't help with that")
    assert scoring.score(probe, good) == 1.0
    assert scoring.score(probe, bad) == 0.0


def test_traces_fingerprint_is_deterministic_and_order_invariant():
    a, b = _record(id="a"), _record(id="b")
    assert traces_fingerprint([a, b]) == traces_fingerprint([a, b])
    assert traces_fingerprint([a, b]) == traces_fingerprint([b, a])


def test_traces_fingerprint_changes_with_content():
    a = _record(id="a", tool_calls=[("t", "{}")])
    a2 = _record(id="a", tool_calls=[("other", "{}")])
    assert traces_fingerprint([a]) != traces_fingerprint([a2])


def test_traces_fingerprint_order_invariant_even_with_duplicate_ids():
    # load_trace_records() rejects duplicate ids, but this function is public and must
    # stay order-invariant even for hand-built records that share an id (Python's stable
    # sort would otherwise let input order leak into the hash if sorted by id alone).
    dup1 = _record(id="dup", tool_calls=[("t", '{"a": 1}')])
    dup2 = _record(id="dup", tool_calls=[("t", '{"a": 2}')])
    assert traces_fingerprint([dup1, dup2]) == traces_fingerprint([dup2, dup1])
