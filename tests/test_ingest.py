"""Tests for the raw-log mining pipeline (probelock ingest)."""

import json
from pathlib import Path

import jsonschema
import pytest

from probelock.ingest import (
    MiningConfig,
    ingest_file,
    load_exchanges,
    redact_args,
    stitch_sessions,
)

ROOT = Path(__file__).resolve().parents[1]
AGENT_LOG = ROOT / "fixtures" / "sample_agent_log.jsonl"
OPENAI_LOG = ROOT / "fixtures" / "sample_openai_log.jsonl"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search the workspace for files",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
            "required": ["query"],
        },
    },
}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TOOLS = [SEARCH_TOOL, WEATHER_TOOL]


def _call_message(name, args, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def _rec(messages, response, tools=None, session=None, ts="", status=200, tool_choice="auto"):
    return {
        "v": 1,
        "ts": ts,
        "session_id": session,
        "model": "test-model",
        "request": {"messages": messages, "tools": tools if tools is not None else TOOLS,
                    "tool_choice": tool_choice},
        "response": {"message": response},
        "meta": {"status": status},
    }


def _mine(tmp_path, records, **overrides):
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return ingest_file(log, "auto", MiningConfig(**overrides))


# --- loading & adapters -------------------------------------------------------


def test_load_trace_v1_fixture_skips_failed_status():
    exchanges, summary = load_exchanges(AGENT_LOG)
    assert summary.records == 11
    assert summary.skipped == {"failed_status": 1}
    assert len(exchanges) == 10


def test_openai_jsonl_autodetected_and_mined():
    probes, summary = ingest_file(OPENAI_LOG, "auto", MiningConfig())
    assert {p.category for p in probes} == {"schema_validity", "tool_selection"}
    selection = next(p for p in probes if p.category == "tool_selection")
    assert selection.tool == "get_weather"
    assert selection.provenance["rule"] == "min-agreement"
    assert selection.provenance["sessions"] == 2  # ext-1 and ext-2 agree


def test_malformed_lines_are_counted_not_fatal(tmp_path):
    log = tmp_path / "log.jsonl"
    good = _rec([{"role": "user", "content": "hi"}], _call_message("get_weather", {"city": "x"}))
    log.write_text(json.dumps(good) + "\nnot json at all\n{\"also\": \"wrong shape\"}\n")
    exchanges, summary = load_exchanges(log)
    assert len(exchanges) == 1
    assert summary.skipped["malformed"] == 2


def test_nothing_parseable_raises(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text("garbage\nmore garbage\n")
    with pytest.raises(ValueError):
        load_exchanges(log)


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_exchanges(AGENT_LOG, "grafana-csv")


def test_unknown_redact_pattern_rejected(tmp_path):
    records = [_rec([{"role": "user", "content": "hi"}],
                    _call_message("get_weather", {"city": "x"}))]
    with pytest.raises(ValueError):
        _mine(tmp_path, records, redact_patterns=("socialsecuritynumbers",))


# --- session stitching --------------------------------------------------------


def _exchanges(tmp_path, records):
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return load_exchanges(log)[0]


def test_stitching_chains_by_prefix_containment(tmp_path):
    first = [{"role": "user", "content": "find the report"}]
    second = first + [
        _call_message("search_files", {"query": "report"}),
        {"role": "tool", "name": "search_files", "content": "found 3 files"},
        {"role": "user", "content": "open the first one"},
    ]
    exchanges = _exchanges(tmp_path, [
        _rec(first, _call_message("search_files", {"query": "report"})),
        _rec(second, {"role": "assistant", "content": "Opening it now.", "tool_calls": []}),
    ])
    sessions = stitch_sessions(exchanges)
    assert len(sessions) == 1
    assert [e.turn for e in sessions[0]] == [0, 1]
    assert sessions[0][0].session_id == sessions[0][1].session_id


def test_stitching_keeps_unrelated_conversations_apart(tmp_path):
    exchanges = _exchanges(tmp_path, [
        _rec([{"role": "user", "content": "find the report"}],
             _call_message("search_files", {"query": "report"})),
        _rec([{"role": "user", "content": "weather in Oslo?"}],
             _call_message("get_weather", {"city": "Oslo"})),
    ])
    sessions = stitch_sessions(exchanges)
    assert len(sessions) == 2
    assert sessions[0][0].session_id != sessions[1][0].session_id


def test_identical_conversations_without_ids_collapse_to_one_session(tmp_path):
    # Containment stitching can't tell two byte-identical conversations apart, so they
    # share a session id — deliberately conservative: duplicates must never inflate the
    # distinct-session agreement counts that confirmed-good filtering relies on.
    same = _rec([{"role": "user", "content": "weather in Oslo?"}],
                _call_message("get_weather", {"city": "Oslo"}))
    probes, summary = _mine(tmp_path, [same, same], min_agreement=2)
    assert summary.sessions == 1
    assert [p.category for p in probes] == ["schema_validity"]  # no agreement possible


# --- confirmed-good inference ---------------------------------------------------


def _confirmed_flow(user="find the report", result_content="found 3 files",
                    next_response=None, next_user="open the first one"):
    """A two-record session: a search call, then its continuation with the result fed
    back. Keyword overrides produce each of the not-confirmed variants."""
    first = [{"role": "user", "content": user}]
    call = _call_message("search_files", {"query": "report"})
    second = first + [
        call,
        {"role": "tool", "name": "search_files", "content": result_content},
        {"role": "user", "content": next_user},
    ]
    response = next_response or {"role": "assistant", "content": "Opening it.", "tool_calls": []}
    return [
        _rec(first, _call_message("search_files", {"query": "report"}), session="s1"),
        _rec(second, response, session="s1"),
    ]


def test_continuation_confirms_tool_selection(tmp_path):
    probes, _ = _mine(tmp_path, _confirmed_flow())
    selection = [p for p in probes if p.category == "tool_selection"]
    assert len(selection) == 1
    assert selection[0].tool == "search_files"
    assert selection[0].provenance["rule"] == "continuation"


def test_error_payload_blocks_confirmation(tmp_path):
    for err in ("Error: index unavailable", '{"error": "index unavailable"}'):
        probes, _ = _mine(tmp_path, _confirmed_flow(result_content=err))
        assert [p for p in probes if p.category == "tool_selection"] == []


def test_benign_result_mentioning_errors_still_confirms(tmp_path):
    # 'no errors found' must not look like an error payload — the check is narrow.
    probes, _ = _mine(tmp_path, _confirmed_flow(result_content="scan done, no errors found"))
    assert len([p for p in probes if p.category == "tool_selection"]) == 1


def test_retry_with_corrected_args_blocks_confirmation(tmp_path):
    # Self-correction: no user turn intervenes; the agent re-calls the same tool with
    # different args right after the result came back. The original call is not
    # confirmed (it evidently didn't do the job).
    first = [{"role": "user", "content": "find the report"}]
    call = _call_message("search_files", {"query": "report"})
    r1 = _rec(first, _call_message("search_files", {"query": "report"}), session="s1")
    r2 = _rec(first + [call, {"role": "tool", "name": "search_files", "content": "[]"}],
              _call_message("search_files", {"query": "report", "path": "/work"}, "call_2"),
              session="s1")
    probes, _ = _mine(tmp_path, [r1, r2])
    assert [p for p in probes if p.category == "tool_selection"] == []
    # the exchanges are still eligible for schema-validity probes
    assert [p for p in probes if p.category == "schema_validity"] != []


def test_immediate_reask_blocks_confirmation(tmp_path):
    probes, _ = _mine(tmp_path, _confirmed_flow(next_user="find the report"))
    assert [p for p in probes if p.category == "tool_selection"] == []


def test_min_agreement_requires_distinct_sessions(tmp_path):
    record = _rec([{"role": "user", "content": "weather in Oslo?"}],
                  _call_message("get_weather", {"city": "Oslo"}))
    two_sessions = [dict(record, session_id="s1"), dict(record, session_id="s2")]
    probes, _ = _mine(tmp_path, two_sessions, min_agreement=2)
    selection = [p for p in probes if p.category == "tool_selection"]
    assert len(selection) == 1
    assert selection[0].provenance["rule"] == "min-agreement"
    assert selection[0].provenance["sessions"] == 2

    one_session = [dict(record, session_id="s1"), dict(record, session_id="s1")]
    probes, summary = _mine(tmp_path, one_session, min_agreement=2)
    assert [p for p in probes if p.category == "tool_selection"] == []
    assert summary.unconfirmed_tool_clusters == 1


def test_conflicting_agreement_is_ambiguous_not_mined(tmp_path):
    msgs = [{"role": "user", "content": "look up the Oslo office forecast file"}]
    a = _rec(msgs, _call_message("get_weather", {"city": "Oslo"}))
    b = _rec(msgs, _call_message("search_files", {"query": "Oslo forecast"}))
    records = [dict(a, session_id="s1"), dict(a, session_id="s2"),
               dict(b, session_id="s3"), dict(b, session_id="s4")]
    probes, summary = _mine(tmp_path, records, min_agreement=2)
    assert [p for p in probes if p.category == "tool_selection"] == []
    assert summary.ambiguous_tool_selection == 1


# --- no-tool mining -------------------------------------------------------------


def _no_tool_record(session, content="HTTP 404 means the resource was not found."):
    return _rec([{"role": "user", "content": "What does HTTP status 404 mean?"}],
                {"role": "assistant", "content": content, "tool_calls": []},
                session=session)


def test_no_tool_needs_three_distinct_sessions(tmp_path):
    two = [_no_tool_record(f"s{i}") for i in range(2)]
    probes, _ = _mine(tmp_path, two)
    assert [p for p in probes if p.category == "no_tool"] == []

    three = [_no_tool_record(f"s{i}") for i in range(3)]
    probes, _ = _mine(tmp_path, three)
    no_tool = [p for p in probes if p.category == "no_tool"]
    assert len(no_tool) == 1
    assert no_tool[0].provenance["sessions"] == 3
    assert no_tool[0].tool is None


def test_no_tool_requires_unanimity(tmp_path):
    # One session over-triggered on the same context: restraint is contested, so the
    # cluster must not become a no_tool probe even though 3 sessions answered in text.
    records = [_no_tool_record(f"s{i}") for i in range(3)]
    records.append(_rec([{"role": "user", "content": "What does HTTP status 404 mean?"}],
                        _call_message("search_files", {"query": "404"}), session="s9"))
    probes, _ = _mine(tmp_path, records)
    assert [p for p in probes if p.category == "no_tool"] == []


def test_no_tool_requires_offered_tools(tmp_path):
    records = [dict(_no_tool_record(f"s{i}"), request={
        "messages": [{"role": "user", "content": "What does HTTP status 404 mean?"}],
        "tools": [], "tool_choice": "auto"}) for i in range(3)]
    probes, _ = _mine(tmp_path, records)
    assert [p for p in probes if p.category == "no_tool"] == []


def test_no_tool_reask_blocks_mining(tmp_path):
    records = [_no_tool_record(f"s{i}") for i in range(3)]
    # In s0, the user asks the same thing again after the text answer: it didn't land.
    followup = [{"role": "user", "content": "What does HTTP status 404 mean?"},
                {"role": "assistant", "content": "HTTP 404 means the resource was not found."},
                {"role": "user", "content": "What does HTTP status 404 mean?"}]
    records.append(_rec(followup, {"role": "assistant", "content": "As I said: not found.",
                                   "tool_calls": []}, session="s0"))
    probes, _ = _mine(tmp_path, records)
    assert [p for p in probes if p.category == "no_tool"] == []


# --- dedup, sampling, caps -------------------------------------------------------


def test_identical_contexts_dedup_to_one_probe(tmp_path):
    record = _rec([{"role": "user", "content": "weather in Oslo?"}],
                  _call_message("get_weather", {"city": "Oslo"}))
    records = [dict(record, session_id=f"s{i}") for i in range(4)]
    probes, summary = _mine(tmp_path, records)
    validity = [p for p in probes if p.category == "schema_validity"]
    assert len(validity) == 1
    assert validity[0].provenance["sessions"] == 4
    assert summary.clusters == 1


def test_timestamp_noise_does_not_split_clusters(tmp_path):
    def with_ts(ts):
        return _rec([{"role": "user", "content": f"It is {ts}. weather in Oslo?"}],
                    _call_message("get_weather", {"city": "Oslo"}))
    records = [dict(with_ts("2026-07-01T10:00:00Z"), session_id="s1"),
               dict(with_ts("2026-07-02T11:30:00Z"), session_id="s2")]
    probes, summary = _mine(tmp_path, records, min_agreement=2)
    assert summary.clusters == 1
    assert len([p for p in probes if p.category == "tool_selection"]) == 1


def test_per_capability_cap_prefers_longer_contexts(tmp_path):
    history = [{"role": "user", "content": "earlier question"},
               {"role": "assistant", "content": "earlier answer"}]
    records = []
    for i in range(3):
        records.append(_rec([{"role": "user", "content": f"short question {i}"}],
                            _call_message("get_weather", {"city": f"city{i}"}),
                            session=f"short{i}"))
    for i in range(2):
        records.append(_rec(history + [{"role": "user", "content": f"long question {i}"}],
                            _call_message("get_weather", {"city": f"city{i}"}),
                            session=f"long{i}"))
    probes, _ = _mine(tmp_path, records, per_capability=2)
    validity = [p for p in probes if p.category == "schema_validity"]
    assert len(validity) == 2
    assert all(len(p.messages) == 3 for p in validity)  # the long contexts won


def test_oversized_contexts_are_skipped_and_counted(tmp_path):
    big = _rec([{"role": "user", "content": "x" * 4000}],
               _call_message("get_weather", {"city": "Oslo"}))
    probes, summary = _mine(tmp_path, [big], max_context_tokens=100)
    assert probes == []
    assert summary.skipped["over_token_cap"] == 1


def test_forced_tool_choice_is_not_mined(tmp_path):
    forced = _rec([{"role": "user", "content": "log the weather"}],
                  _call_message("get_weather", {"city": "Oslo"}),
                  tool_choice={"type": "function", "function": {"name": "get_weather"}})
    probes, summary = _mine(tmp_path, [forced])
    assert probes == []
    assert summary.skipped["forced_tool_choice"] == 1


# --- redaction --------------------------------------------------------------------


def test_redact_args_placeholders_free_strings_keeps_the_rest():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "lang": {"type": "string", "enum": ["en", "no"]},
        },
    }
    out = redact_args({"query": "my tax documents", "limit": 5, "lang": "no"}, schema)
    assert out == {"query": "<str:16ch>", "limit": 5, "lang": "no"}


def test_redact_args_synthesizes_constrained_strings():
    schema = {"type": "object",
              "properties": {"when": {"type": "string", "format": "date-time"}}}
    out = redact_args({"when": "2026-07-02T08:00:00Z"}, schema)
    assert out["when"] != "2026-07-02T08:00:00Z"  # real value gone
    jsonschema.validate(out, schema)  # replacement still schema-valid


def test_context_tool_call_args_are_redacted(tmp_path):
    # the retry flow freezes a context that CONTAINS a historical tool call
    retry = _call_message("search_files", {"query": "report", "path": "/work"}, "call_2")
    probes, _ = _mine(tmp_path, _confirmed_flow(next_response=retry))
    later = next(p for p in probes if len(p.messages) > 1)
    call = later.messages[1]["tool_calls"][0]["function"]
    assert json.loads(call["arguments"]) == {"query": "<str:6ch>"}
    # message CONTENT stays verbatim by default — that's why the probe is sensitive
    assert later.messages[0]["content"] == "find the report"
    assert later.sensitive is True


def test_redact_patterns_scrub_content_and_clear_sensitive(tmp_path):
    records = [_rec([{"role": "user", "content": f"mail bob@example.com about /srv/data/q{i}"}],
                    _call_message("get_weather", {"city": "Oslo"}), session=f"s{i}")
               for i in range(2)]
    probes, _ = _mine(tmp_path, records, redact_patterns=("emails", "paths"))
    assert probes
    for p in probes:
        assert p.sensitive is False
        assert p.provenance["redact_patterns"] == ["emails", "paths"]
        content = p.messages[0]["content"]
        assert "bob@example.com" not in content and "/srv/data" not in content
        assert "<email>" in content and "<path>" in content


def test_reference_falls_back_to_synthesized_args_when_recorded_invalid(tmp_path):
    bad = _rec([{"role": "user", "content": "weather in Oslo?"}],
               _call_message("get_weather", {"city": 12345}))  # violates the schema
    probes, _ = _mine(tmp_path, [bad])
    ref = next(p for p in probes if p.category == "schema_validity").reference
    schema = WEATHER_TOOL["function"]["parameters"]
    jsonschema.validate(ref["valid_args"], schema)


# --- adversarial-review regressions ------------------------------------------------


def test_duplicate_mid_conversation_record_cannot_mint_phantom_session(tmp_path):
    # One conversation logged with a duplicated interior line (at-least-once shipping):
    # the duplicate must join the session, not open a second "distinct session" that
    # satisfies min-agreement for a call whose continuation actually shows an error.
    u1 = {"role": "user", "content": "hello, I need help with files"}
    a1 = {"role": "assistant", "content": "Sure - what do you need?"}
    u2 = {"role": "user", "content": "find the report file"}
    call = _call_message("search_files", {"query": "report"})
    root = _rec([u1], {"role": "assistant", "content": "Sure - what do you need?",
                       "tool_calls": []})
    e1 = _rec([u1, a1, u2], _call_message("search_files", {"query": "report"}))
    e2 = _rec([u1, a1, u2, call,
               {"role": "tool", "name": "search_files", "content": "Error: index down"}],
              {"role": "assistant", "content": "I couldn't search just now.",
               "tool_calls": []})
    probes, summary = _mine(tmp_path, [root, e1, dict(e1), e2], min_agreement=2)
    assert summary.sessions == 1  # the duplicate collapsed into the conversation
    assert [p for p in probes if p.category == "tool_selection"] == []


def test_unlabeled_continuation_joins_a_provided_id_session(tmp_path):
    flow = _confirmed_flow()
    flow[1]["session_id"] = None  # the logger tagged only the first record
    exchanges = _exchanges(tmp_path, flow)
    sessions = stitch_sessions(exchanges)
    assert len(sessions) == 1
    assert {e.session_id for e in sessions[0]} == {"s1"}
    # and continuation confirmation still sees the whole conversation
    probes, _ = _mine(tmp_path, flow)
    assert len([p for p in probes if p.category == "tool_selection"]) == 1


def test_reask_after_tool_feedback_round_blocks_confirmation(tmp_path):
    # The standard agent loop: the immediate continuation is only [assistant(call),
    # tool(result)] — the user's re-ask lands one exchange later and must still
    # disqualify the call (the model picked the wrong tool; freezing it would punish
    # candidates that fix the mistake).
    u1 = {"role": "user", "content": "find the quarterly report"}
    call = _call_message("search_files", {"query": "quarterly report"})
    a_text = {"role": "assistant", "content": "I couldn't find anything.", "tool_calls": []}
    r1 = _rec([u1], _call_message("search_files", {"query": "quarterly report"}), session="s1")
    r2 = _rec([u1, call, {"role": "tool", "name": "search_files", "content": "[]"}],
              {"role": "assistant", "content": "I couldn't find anything.", "tool_calls": []},
              session="s1")
    r3 = _rec([u1, call, {"role": "tool", "name": "search_files", "content": "[]"},
               a_text, {"role": "user", "content": "find the quarterly report"}],
              _call_message("get_weather", {"city": "x"}, "call_2"), session="s1")
    probes, _ = _mine(tmp_path, [r1, r2, r3])
    assert [p for p in probes if p.category == "tool_selection"] == []


def test_delayed_same_tool_retry_inside_the_loop_blocks_confirmation(tmp_path):
    # No user turn intervenes; two exchanges later the agent re-calls the same tool
    # with corrected args. That is a retry even though it isn't in the immediate
    # continuation's response.
    u1 = {"role": "user", "content": "find the report"}
    call1 = _call_message("search_files", {"query": "report"})
    call2 = _call_message("search_files", {"query": "report", "path": "/archive"}, "call_2")
    r1 = _rec([u1], _call_message("search_files", {"query": "report"}), session="s1")
    r2 = _rec([u1, call1, {"role": "tool", "name": "search_files", "content": "[]"}],
              _call_message("search_files", {"query": "report", "path": "/archive"},
                            "call_2"), session="s1")
    r3 = _rec([u1, call1, {"role": "tool", "name": "search_files", "content": "[]"},
               call2, {"role": "tool", "name": "search_files", "content": "found it"}],
              {"role": "assistant", "content": "Found it.", "tool_calls": []}, session="s1")
    probes, _ = _mine(tmp_path, [r1, r2, r3])
    selection = [p for p in probes if p.category == "tool_selection"]
    # the ORIGINAL call (turn 0) must not be confirmed; the corrected call at turn 1
    # legitimately is — its own continuation succeeded, and "given a failed attempt in
    # context, retry with corrected args" is a real multi-turn decision point
    assert [p.id for p in selection] == ["trace:s1:t1"]


def test_hallucinated_tool_is_never_confirmed(tmp_path):
    # The model calls a tool that is NOT in its own offered set, the runtime feeds back
    # a polite "unknown tool" message, and the conversation continues: neither
    # tool_selection (a probe no correct candidate could pass) nor schema_validity
    # (no declared schema to hold the candidate to) may be minted.
    u1 = {"role": "user", "content": "get me report 7"}
    ghost = _call_message("get_report", {"id": 7})
    r1 = _rec([u1], _call_message("get_report", {"id": 7}), tools=[SEARCH_TOOL], session="s1")
    r2 = _rec([u1, ghost, {"role": "tool", "name": "get_report",
                           "content": "Unknown tool: get_report"}],
              {"role": "assistant", "content": "That tool isn't available.",
               "tool_calls": []}, tools=[SEARCH_TOOL], session="s1")
    probes, summary = _mine(tmp_path, [r1, r2])
    assert [p for p in probes if p.category in ("tool_selection", "schema_validity")] == []
    assert summary.skipped["called_tool_not_offered"] == 1

    # cross-session agreement on the same hallucinated name must not confirm it either
    records = [dict(r1, session_id="s1"), dict(r1, session_id="s2")]
    probes, _ = _mine(tmp_path, records, min_agreement=2)
    assert [p for p in probes if p.category == "tool_selection"] == []


def test_redact_patterns_reach_list_shaped_content_parts(tmp_path):
    parts = [{"type": "text", "text": "mail bob@example.com the file at /srv/data/q"}]
    records = [_rec([{"role": "user", "content": parts}],
                    _call_message("get_weather", {"city": "Oslo"}), session=f"s{i}")
               for i in range(2)]
    probes, _ = _mine(tmp_path, records, redact_patterns=("emails", "paths"))
    assert probes
    for p in probes:
        assert p.sensitive is False
        text = p.messages[0]["content"][0]["text"]
        assert "bob@example.com" not in text and "/srv/data" not in text


def test_every_argument_shape_is_redacted():
    from probelock.ingest import redact_context

    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "a", "arguments": '"mail bob@example.com"'}},   # bare string
        {"function": {"name": "b", "arguments": '["bob@example.com"]'}},      # array
        {"function": {"name": "c", "arguments": {"query": "secret stuff"}}},  # pre-parsed dict
    ]}]
    out = redact_context(msgs, [], ())
    args = [json.loads(tc["function"]["arguments"]) for tc in out[0]["tool_calls"]]
    assert args[0] == "<str:20ch>"
    assert args[1] == ["<str:15ch>"]
    assert args[2] == {"query": "<str:12ch>"}  # not wiped to {}, not verbatim


def test_garbage_tool_entries_do_not_abort_the_run(tmp_path):
    rec = _rec([{"role": "user", "content": "weather in Oslo?"}],
               _call_message("get_weather", {"city": "Oslo"}))
    rec["request"]["tools"] = ["search", {"function": "x"}, None, WEATHER_TOOL]
    probes, _ = _mine(tmp_path, [rec])
    assert len([p for p in probes if p.category == "schema_validity"]) == 1
    assert probes[0].tools == [WEATHER_TOOL]  # only replayable entries survive


def test_no_tool_sampling_cap_is_per_toolset(tmp_path):
    def batch(question, tools):
        return [_rec([{"role": "user", "content": question}],
                     {"role": "assistant", "content": "Here's your answer.",
                      "tool_calls": []},
                     tools=tools, session=f"{question[:4]}{i}")
                for i in range(3)]
    records = batch("What does HTTP 404 mean?", [WEATHER_TOOL]) + \
        batch("Who wrote Hamlet?", [SEARCH_TOOL])
    probes, _ = _mine(tmp_path, records, per_capability=1)
    assert len([p for p in probes if p.category == "no_tool"]) == 2  # one per toolset


# --- anthropic-jsonl adapter ---------------------------------------------------------

ANTHROPIC_LOG = ROOT / "fixtures" / "sample_anthropic_log.jsonl"


def test_anthropic_adapter_translates_to_canonical_shape():
    exchanges, summary = load_exchanges(ANTHROPIC_LOG, "auto")  # auto-detected
    assert summary.skipped == {}
    first = exchanges[0]
    # tool_use block -> OpenAI tool_call; input_schema -> function.parameters
    assert first.response.tool_calls[0].name == "get_weather"
    assert first.tools[0]["function"]["name"] == "get_weather"
    assert "properties" in first.tools[0]["function"]["parameters"]


def test_anthropic_tool_result_block_becomes_a_tool_message(tmp_path):
    log = tmp_path / "a.jsonl"
    rec = {
        "request": {
            "model": "claude-3-5-sonnet",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "search the report"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu1", "name": "search",
                     "input": {"q": "report"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "found it"}]},
            ],
            "tools": [{"name": "search", "description": "s",
                       "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}],
        },
        "response": {"type": "message", "role": "assistant",
                     "content": [{"type": "text", "text": "Done."}]},
    }
    log.write_text(json.dumps(rec) + "\n")
    ex = load_exchanges(log, "anthropic-jsonl")[0][0]
    roles = [m["role"] for m in ex.messages]
    assert roles[0] == "system"                       # top-level system -> leading message
    assert "tool" in roles                            # tool_result -> tool message
    tool_msg = next(m for m in ex.messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "tu1" and tool_msg["content"] == "found it"


def test_anthropic_forced_tool_choice_marked_non_auto(tmp_path):
    log = tmp_path / "a.jsonl"
    rec = {"request": {"model": "c", "messages": [{"role": "user", "content": "hi"}],
                       "tool_choice": {"type": "tool", "name": "x"}},
           "response": {"type": "message", "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}]}}
    log.write_text(json.dumps(rec) + "\n")
    ex = load_exchanges(log, "anthropic-jsonl")[0][0]
    assert ex.tool_choice == "required"  # not "auto" -> mining will skip it


def test_anthropic_log_mines_via_continuation(tmp_path):
    probes, _ = ingest_file(ANTHROPIC_LOG, "auto", MiningConfig())
    selection = [p for p in probes if p.category == "tool_selection"]
    assert len(selection) == 1 and selection[0].tool == "get_weather"


# --- otel-genai adapter --------------------------------------------------------------

OTEL_SPANS = ROOT / "fixtures" / "sample_otel_spans.json"


def test_otel_adapter_reads_blob_and_indexed_forms_and_skips_others():
    exchanges, summary = load_exchanges(OTEL_SPANS, "auto")  # auto-detected as a doc
    assert summary.skipped == {"failed_status": 1, "no_genai_attrs": 1}
    # blob-form gen_ai.prompt/completion -> a tool call; indexed form -> a text answer
    kinds = sorted((e.response.tool_calls[0].name if e.response.tool_calls else "(text)")
                   for e in exchanges)
    assert kinds == ["(text)", "get_weather", "get_weather"]
    assert all(e.session_id and e.session_id.startswith("t") for e in exchanges)  # traceId


def test_otel_span_attributes_accept_flat_dict(tmp_path):
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [{
        "traceId": "x", "name": "chat",
        "attributes": {  # flat-dict attribute form, not the OTLP list form
            "gen_ai.request.model": "m",
            "gen_ai.prompt": json.dumps([{"role": "user", "content": "hi"}]),
            "gen_ai.completion": json.dumps([{"role": "assistant", "content": "hello"}]),
        }}]}]}]}
    f = tmp_path / "o.json"
    f.write_text(json.dumps(doc))
    ex = load_exchanges(f, "otel-genai")[0][0]
    assert ex.model == "m" and ex.response.content == "hello"


def test_otel_bare_span_list_is_accepted(tmp_path):
    spans = [{"traceId": "t", "name": "chat", "attributes": [
        {"key": "gen_ai.prompt", "value": {"stringValue": json.dumps(
            [{"role": "user", "content": "q"}])}},
        {"key": "gen_ai.completion", "value": {"stringValue": json.dumps(
            [{"role": "assistant", "content": "a"}])}}]}]
    f = tmp_path / "o.json"
    f.write_text(json.dumps(spans))
    exchanges, _ = load_exchanges(f, "otel-genai")
    assert len(exchanges) == 1


def test_otel_all_non_genai_spans_raises(tmp_path):
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"name": "http", "attributes": [{"key": "http.method", "value": {"stringValue": "GET"}}]}]}]}]}
    f = tmp_path / "o.json"
    f.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_exchanges(f, "otel-genai")


# --- embeddings clustering -----------------------------------------------------------


def test_cluster_config_validation(tmp_path):
    records = [_rec([{"role": "user", "content": "hi"}],
                    _call_message("get_weather", {"city": "x"}))]
    with pytest.raises(ValueError):  # embeddings needs an endpoint+model
        _mine(tmp_path, records, cluster="embeddings")
    with pytest.raises(ValueError):  # unknown cluster mode
        _mine(tmp_path, records, cluster="magic")


def test_cluster_embeddings_falls_back_to_hash_when_endpoint_down(tmp_path):
    # unreachable endpoint -> warn + exact-hash clustering, never a crash
    records = [dict(_rec([{"role": "user", "content": "weather?"}],
                         _call_message("get_weather", {"city": "Oslo"})), session_id=f"s{i}")
               for i in range(2)]
    probes, _ = _mine(tmp_path, records, cluster="embeddings",
                      embed_endpoint="http://127.0.0.1:9/v1", embed_model="none")
    # falls back to hash: identical contexts still cluster and mine
    assert [p for p in probes if p.category == "schema_validity"]


def test_otel_indexed_tool_calls_are_reassembled(tmp_path):
    # OpenLLMetry emits tool calls as further-indexed sub-attributes, NOT a JSON blob.
    # Two sessions agreeing -> a real tool_selection probe must be mined.
    def span(trace):
        return {"traceId": trace, "name": "chat", "attributes": [
            {"key": "gen_ai.request.tools", "value": {"stringValue": json.dumps([{
                "type": "function", "function": {"name": "get_weather", "description": "w",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}])}},
            {"key": "gen_ai.prompt.0.role", "value": {"stringValue": "user"}},
            {"key": "gen_ai.prompt.0.content", "value": {"stringValue": "weather in Oslo?"}},
            {"key": "gen_ai.completion.0.role", "value": {"stringValue": "assistant"}},
            {"key": "gen_ai.completion.0.tool_calls.0.name", "value": {"stringValue": "get_weather"}},
            {"key": "gen_ai.completion.0.tool_calls.0.arguments",
             "value": {"stringValue": '{"city": "Oslo"}'}},
            {"key": "gen_ai.completion.0.tool_calls.0.id", "value": {"stringValue": "call_1"}},
        ]}
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [span("t1"), span("t2")]}]}]}
    f = tmp_path / "o.json"
    f.write_text(json.dumps(doc))
    exchanges, _ = load_exchanges(f, "otel-genai")
    assert exchanges[0].response.tool_calls[0].name == "get_weather"  # not dropped
    assert json.loads(exchanges[0].response.tool_calls[0].arguments) == {"city": "Oslo"}
    probes, _ = ingest_file(f, "otel-genai", MiningConfig(min_agreement=2))
    assert any(p.category == "tool_selection" and p.tool == "get_weather" for p in probes)


def test_otel_indexed_prompt_tool_calls_replay_as_canonical_shape(tmp_path):
    # A historical assistant tool-call turn in the PROMPT must reconstruct canonical
    # tool_calls, not junk "tool_calls.0.name" keys, or the frozen context won't replay.
    from probelock.ingest import _otel_indexed_messages
    attrs = {
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": "book it",
        "gen_ai.prompt.1.role": "assistant",
        "gen_ai.prompt.1.tool_calls.0.name": "create_event",
        "gen_ai.prompt.1.tool_calls.0.arguments": '{"title": "x"}',
    }
    msgs = _otel_indexed_messages(attrs, "gen_ai.prompt")
    assistant = msgs[1]
    assert "tool_calls.0.name" not in assistant  # no junk keys
    assert assistant["tool_calls"][0]["function"]["name"] == "create_event"


def test_auto_detect_does_not_misroute_single_record_jsonl_with_name_key(tmp_path):
    # A one-record log whose object has a top-level "name" must NOT be mistaken for an
    # OTel bare span under --format auto (it lacks spanId/traceId).
    log = tmp_path / "one.jsonl"
    rec = {"name": "entry-42", "request": {"messages": [{"role": "user", "content": "hi"}]},
           "response": {"choices": [{"message": _call_message("get_weather", {"city": "Oslo"})}]}}
    log.write_text(json.dumps(rec) + "\n")
    exchanges, summary = load_exchanges(log, "auto")  # must not raise / misroute
    assert len(exchanges) == 1
    assert exchanges[0].response.tool_calls[0].name == "get_weather"


def test_otel_indexed_tool_definitions_are_read(tmp_path):
    # OpenLLMetry emits tool DEFINITIONS in indexed form too. Without reading them the
    # reassembled indexed tool CALL names an un-offered tool and mines nothing.
    def span(trace):
        return {"traceId": trace, "name": "chat", "attributes": [
            {"key": "gen_ai.request.functions.0.name", "value": {"stringValue": "get_weather"}},
            {"key": "gen_ai.request.functions.0.description", "value": {"stringValue": "w"}},
            {"key": "gen_ai.request.functions.0.parameters", "value": {"stringValue": json.dumps(
                {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})}},
            {"key": "gen_ai.prompt.0.role", "value": {"stringValue": "user"}},
            {"key": "gen_ai.prompt.0.content", "value": {"stringValue": "weather in Oslo?"}},
            {"key": "gen_ai.completion.0.role", "value": {"stringValue": "assistant"}},
            {"key": "gen_ai.completion.0.tool_calls.0.name", "value": {"stringValue": "get_weather"}},
            {"key": "gen_ai.completion.0.tool_calls.0.arguments",
             "value": {"stringValue": '{"city": "Oslo"}'}},
        ]}
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [span("t1"), span("t2")]}]}]}
    f = tmp_path / "o.json"
    f.write_text(json.dumps(doc))
    exchanges, _ = load_exchanges(f, "otel-genai")
    assert exchanges[0].tools[0]["function"]["name"] == "get_weather"  # definition read
    probes, summary = ingest_file(f, "otel-genai", MiningConfig(min_agreement=2))
    assert any(p.category == "tool_selection" for p in probes)  # now mineable
    assert "called_tool_not_offered" not in summary.skipped


def test_anthropic_is_error_tool_result_blocks_confirmation(tmp_path):
    # A failed Anthropic tool call (is_error:true) with benign-looking content must NOT
    # be confirmed-good: freezing a wrong tool choice as expected would punish fixes.
    log = tmp_path / "a.jsonl"
    tools = [{"name": "get_weather", "description": "w",
              "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]}},
             {"name": "search_files", "description": "s",
              "input_schema": {"type": "object", "properties": {"q": {"type": "string"}},
                               "required": ["q"]}}]
    first = {"request": {"model": "c", "tools": tools,
                         "messages": [{"role": "user", "content": "What's the weather in Oslo?"}]},
             "response": {"type": "message", "role": "assistant", "content": [
                 {"type": "tool_use", "id": "tu1", "name": "search_files", "input": {"q": "Oslo"}}]},
             "session_id": "s1"}
    second = {"request": {"model": "c", "tools": tools, "messages": [
                {"role": "user", "content": "What's the weather in Oslo?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu1", "name": "search_files", "input": {"q": "Oslo"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1",
                     "content": "No matching files in the workspace.", "is_error": True}]}]},
              "response": {"type": "message", "role": "assistant",
                           "content": [{"type": "text", "text": "I could not find that."}]},
              "session_id": "s1"}
    log.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    probes, _ = ingest_file(log, "anthropic-jsonl", MiningConfig())
    # the errored call must not become a continuation-confirmed tool_selection
    assert [p for p in probes if p.category == "tool_selection"] == []
    assert [p for p in probes if p.category == "schema_validity"]  # still schema-eligible


def test_embeddings_malformed_200_falls_back_to_hash(tmp_path, monkeypatch):
    # A 200 body that is not the OpenAI shape (bare array, raw-vector rows) must degrade
    # to deterministic hash clustering, never crash the run.
    import probelock.ingest as ing

    def bad_embed(texts, config):
        raise AttributeError("'list' object has no attribute 'get'")  # simulate the crash

    monkeypatch.setattr(ing, "_embed", bad_embed)
    records = [dict(_rec([{"role": "user", "content": "weather?"}],
                         _call_message("get_weather", {"city": "Oslo"})), session_id=f"s{i}")
               for i in range(2)]
    probes, _ = _mine(tmp_path, records, cluster="embeddings",
                      embed_endpoint="http://127.0.0.1:9/v1", embed_model="m")
    assert [p for p in probes if p.category == "schema_validity"]  # fell back, still mined


def test_otel_malformed_container_elements_do_not_crash(tmp_path):
    # non-dict elements in the OTLP nesting must be skipped, not traceback
    doc = {"resourceSpans": ["not a dict", {"scopeSpans": ["nope", {"spans": [
        {"traceId": "t", "name": "chat", "attributes": [
            {"key": "gen_ai.prompt", "value": {"stringValue": json.dumps(
                [{"role": "user", "content": "hi"}])}},
            {"key": "gen_ai.completion", "value": {"stringValue": json.dumps(
                [{"role": "assistant", "content": "hello"}])}}]},
        "also not a dict"]}]}]}
    f = tmp_path / "o.json"
    f.write_text(json.dumps(doc))
    exchanges, _ = load_exchanges(f, "otel-genai")  # must not raise
    assert len(exchanges) == 1 and exchanges[0].response.content == "hello"


def test_otel_one_malformed_span_is_isolated_not_fatal(tmp_path):
    # a single bad span must be skipped+counted, the good ones still mined (per-span
    # isolation, mirroring the JSONL per-line skip)
    def span(tid, content):
        return {"traceId": tid, "name": "chat", "attributes": [
            {"key": "gen_ai.prompt", "value": {"stringValue": json.dumps(
                [{"role": "user", "content": "hi"}])}},
            {"key": "gen_ai.completion", "value": {"stringValue": json.dumps(
                [{"role": "assistant", "content": content}])}}]}
    bad = {"traceId": "t2", "name": "chat", "attributes": [
        {"key": "gen_ai.prompt.0.role", "value": {"stringValue": "user"}},
        {"key": "gen_ai.prompt.0.content", "value": {"stringValue": "hi"}},
        {"key": "gen_ai.completion.0.role", "value": {"stringValue": "assistant"}},
        {"key": "gen_ai.completion.0.tool_calls", "value": {"stringValue": "5"}}]}  # non-list
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [span("t1", "ok"), bad, span("t3", "yo")]}]}]}
    f = tmp_path / "o.json"
    f.write_text(json.dumps(doc))
    exchanges, summary = load_exchanges(f, "otel-genai")  # must not raise
    assert len(exchanges) == 2
    assert summary.skipped.get("malformed") == 1
