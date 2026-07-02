"""Tests for the frozen mined-probe format, review transitions, and Probe conversion."""

import json

import pytest

from probelock.mined import (
    MinedProbe,
    auto_accept,
    edit_expected_tool,
    load_mined,
    mined_fingerprint,
    save_mined,
    to_probe,
)

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


def _probe(id="trace:abc:t0", category="tool_selection", tool="get_weather", **kw):
    return MinedProbe(
        id=id, category=category,
        messages=[{"role": "user", "content": "weather in Oslo?"}],
        tools=TOOLS, tool=tool, **kw,
    )


# --- file round trip -----------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "mined.json"
    original = _probe(status="pending", provenance={"sessions": 2, "rule": "min-agreement"},
                      sensitive=True, reference={"tool": "get_weather", "valid_args": {}})
    save_mined([original], path, header={"source": "log.jsonl"})
    loaded = load_mined(path)
    assert len(loaded) == 1
    assert loaded[0] == original
    raw = json.loads(path.read_text())
    assert raw["version"] == 1
    assert raw["source"] == "log.jsonl"
    assert raw["probes"][0]["check"] == {"type": "calls_tool", "tool": "get_weather"}


def test_load_rejects_bad_shapes(tmp_path):
    path = tmp_path / "mined.json"
    for bad in (
        {"not": "probes"},
        {"probes": [{"id": "x", "category": "mind_reading",
                     "context": {"messages": []}}]},
        {"probes": [{"id": "x", "category": "no_tool", "status": "maybe",
                     "context": {"messages": []}}]},
        {"probes": [{"id": "x", "category": "no_tool", "context": "not an object"}]},
        {"probes": [{"category": "no_tool", "context": {"messages": []}}]},  # no id
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_mined(path)


def test_load_rejects_duplicate_id_category_pair_but_allows_shared_id(tmp_path):
    path = tmp_path / "mined.json"
    entry = {"id": "trace:a:t0", "category": "schema_validity",
             "context": {"messages": [], "tools": []}}
    path.write_text(json.dumps({"probes": [entry, dict(entry)]}))
    with pytest.raises(ValueError):
        load_mined(path)
    # the same exchange may legitimately yield one probe per category
    path.write_text(json.dumps({"probes": [entry, dict(entry, category="no_tool")]}))
    assert len(load_mined(path)) == 2


def test_missing_sensitive_flag_defaults_to_sensitive(tmp_path):
    path = tmp_path / "mined.json"
    path.write_text(json.dumps({"probes": [
        {"id": "x", "category": "no_tool", "context": {"messages": [], "tools": []}}
    ]}))
    assert load_mined(path)[0].sensitive is True


# --- fingerprint -----------------------------------------------------------------


def test_fingerprint_is_order_invariant_and_ignores_review_metadata():
    a, b = _probe(id="trace:a:t0"), _probe(id="trace:b:t0", category="schema_validity")
    fp = mined_fingerprint([a, b])
    assert mined_fingerprint([b, a]) == fp
    a.status, a.provenance = "accepted", {"review": "accepted"}
    assert mined_fingerprint([a, b]) == fp  # review churn doesn't move the fingerprint
    a.tool = "search_files"
    assert mined_fingerprint([a, b]) != fp  # the check itself does


# --- conversion to Probe ----------------------------------------------------------


def test_to_probe_tool_selection():
    probe = to_probe(_probe(provenance={"sessions": 2, "rule": "min-agreement"}))
    assert probe.capability == "traced_tool_selection"
    assert probe.id == "traced_tool_selection::trace:abc:t0"
    assert probe.expected_tool == "get_weather"
    assert probe.schema == TOOLS[0]["function"]["parameters"]
    assert "min-agreement" in probe.description


def test_to_probe_schema_validity_pins_no_expected_tool():
    probe = to_probe(_probe(category="schema_validity"))
    assert probe.capability == "traced_schema_validity"
    assert probe.expected_tool is None  # any offered tool may be called
    assert probe.schema == TOOLS[0]["function"]["parameters"]  # recorded tool, for the simulator


def test_to_probe_no_tool():
    probe = to_probe(_probe(category="no_tool", tool=None))
    assert probe.capability == "traced_no_tool"
    assert probe.expected_tool is None
    assert probe.schema is None
    assert probe.tools == TOOLS  # restraint replays with the temptation present


# --- review transitions -------------------------------------------------------------


def test_auto_accept_safe_category_only():
    probes = [_probe(id="a", category="schema_validity"),
              _probe(id="b", category="tool_selection"),
              _probe(id="c", category="no_tool", tool=None)]
    counts = auto_accept(probes, frozenset({"schema_validity"}))
    assert counts == {"schema_validity": 1}
    assert [p.status for p in probes] == ["accepted", "pending", "pending"]
    assert probes[0].provenance["review"] == "auto-accepted"


def test_auto_accept_all_never_touches_no_tool():
    probes = [_probe(id="a", category="schema_validity"),
              _probe(id="b", category="tool_selection"),
              _probe(id="c", category="no_tool", tool=None)]
    counts = auto_accept(probes, frozenset(), accept_all=True)
    assert counts == {"schema_validity": 1, "tool_selection": 1, "no_tool_skipped": 1}
    assert probes[2].status == "pending"


def test_auto_accept_leaves_reviewed_probes_alone():
    probes = [_probe(id="a", category="schema_validity", status="rejected")]
    assert auto_accept(probes, frozenset(), accept_all=True) == {}
    assert probes[0].status == "rejected"


def test_edit_expected_tool_validates_against_offered_tools():
    mp = _probe()
    edit_expected_tool(mp, "search_files")
    assert mp.tool == "search_files"
    assert mp.status == "accepted"
    assert mp.provenance["edited_from"] == "get_weather"
    assert mp.check() == {"type": "calls_tool", "tool": "search_files"}

    with pytest.raises(ValueError):
        edit_expected_tool(_probe(), "not_a_real_tool")
    with pytest.raises(ValueError):
        edit_expected_tool(_probe(category="no_tool", tool=None), "get_weather")


def test_load_rejects_tool_selection_without_expected_tool(tmp_path):
    # expected_tool=None would make the scorer match ANY call — a probe that can
    # never fail must be rejected at load time.
    path = tmp_path / "mined.json"
    path.write_text(json.dumps({"probes": [
        {"id": "x", "category": "tool_selection", "context": {"messages": [], "tools": []}}
    ]}))
    with pytest.raises(ValueError):
        load_mined(path)


def test_load_rejects_non_dict_tool_entries(tmp_path):
    path = tmp_path / "mined.json"
    path.write_text(json.dumps({"probes": [
        {"id": "x", "category": "no_tool",
         "context": {"messages": [], "tools": ["not a tool object"]}}
    ]}))
    with pytest.raises(ValueError):
        load_mined(path)
