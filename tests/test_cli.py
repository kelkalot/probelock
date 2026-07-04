"""CLI-level tests for diff/gate exit codes and the within-model guards."""

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from probelock.cli import app

runner = CliRunner()
_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = _ROOT / "examples" / "agent_tools.json"
_SIM_Q8 = _ROOT / "fixtures" / "profile_q8.json"
_SAMPLE_TRACES = _ROOT / "fixtures" / "sample_traces.json"


def _write_lock(path, model, caps):
    path.write_text(json.dumps({
        "model": model, "quant": "", "runtime": "ollama",
        "tools_fingerprint": "fp", "capabilities": caps, "results": [], "n_probes": 0,
    }))


def test_gate_passes_when_identical(tmp_path):
    p = tmp_path / "a.lock"
    _write_lock(p, "m", {"tool_selection": 1.0, "format_adherence": 0.5})
    result = runner.invoke(app, ["gate", "-b", str(p), "-c", str(p)])
    assert result.exit_code == 0, result.output


def test_gate_fails_on_regression_exit_1(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0})
    _write_lock(c, "m", {"tool_selection": 0.5})
    result = runner.invoke(app, ["gate", "-b", str(b), "-c", str(c)])
    assert result.exit_code == 1, result.output


def test_gate_fails_on_removed_capability(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0, "structured_output": 1.0})
    _write_lock(c, "m", {"tool_selection": 1.0})
    result = runner.invoke(app, ["gate", "-b", str(b), "-c", str(c)])
    assert result.exit_code == 1, result.output


def test_gate_missing_lockfile_is_exit_2(tmp_path):
    p = tmp_path / "a.lock"
    _write_lock(p, "m", {"tool_selection": 1.0})
    result = runner.invoke(app, ["gate", "-b", str(tmp_path / "nope.lock"), "-c", str(p)])
    assert result.exit_code == 2, result.output  # invalid input, not a regression


def test_gate_require_same_model_blocks_cross_model_exit_2(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "big", {"tool_selection": 1.0})
    _write_lock(c, "small", {"tool_selection": 1.0})
    result = runner.invoke(app, ["gate", "-b", str(b), "-c", str(c), "--require-same-model"])
    assert result.exit_code == 2, result.output


def test_diff_warns_on_cross_model(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "big", {"tool_selection": 1.0})
    _write_lock(c, "small", {"tool_selection": 1.0})
    result = runner.invoke(app, ["diff", str(b), str(c)])
    assert result.exit_code == 0, result.output
    assert "different models" in result.output


def test_diff_table_preserves_brackets_in_labels(tmp_path):
    # The rich table/notes must not let model labels with brackets get eaten by
    # rich markup (the escaping discipline behind the _err fix, extended to stdout).
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m[anyllm]", {"tool_selection": 1.0})
    _write_lock(c, "m[litellm]", {"tool_selection": 1.0})
    result = runner.invoke(app, ["diff", str(b), str(c)])  # table format -> stdout
    assert result.exit_code == 0, result.output
    assert "m[anyllm]" in result.output and "m[litellm]" in result.output


def test_diff_markdown_format(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0})
    _write_lock(c, "m", {"tool_selection": 0.0})
    result = runner.invoke(app, ["diff", str(b), str(c), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    assert "| Capability | Baseline | Candidate" in result.output
    assert "REGRESSION" in result.output


def test_diff_html_format(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0})
    _write_lock(c, "m", {"tool_selection": 0.0})
    result = runner.invoke(app, ["diff", str(b), str(c), "--format", "html"])
    assert result.exit_code == 0, result.output
    assert result.output.lstrip().startswith("<!doctype html>")
    assert "REGRESSION" in result.output and "</html>" in result.output


def test_diff_html_header_does_not_duplicate_candidate_column(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0})
    _write_lock(c, "m", {"tool_selection": 1.0})
    result = runner.invoke(app, ["diff", str(b), str(c), "--format", "html"])
    assert result.exit_code == 0, result.output
    assert result.output.count("<th>Candidate</th>") == 1


def test_diff_json_format(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0})
    _write_lock(c, "m", {"tool_selection": 0.0})
    result = runner.invoke(app, ["diff", str(b), str(c), "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["regressed"] is True
    assert any(r["capability"] == "tool_selection" for r in payload["rows"])


def test_gate_invalid_confidence_is_exit_2(tmp_path):
    p = tmp_path / "a.lock"
    _write_lock(p, "m", {"tool_selection": 1.0})
    for bad in ("1.5", "0", "1"):
        result = runner.invoke(app, ["gate", "-b", str(p), "-c", str(p), "--confidence", bad])
        assert result.exit_code == 2, f"confidence={bad}: {result.output}"


def test_probe_refuses_to_write_when_all_probes_error(tmp_path, monkeypatch):
    # Regression guard (round 4): a universally-failing run must Exit(2) and write
    # nothing, even though tool_restraint scores 1.0 on error.
    from probelock.clients import HttpClient, ProbeError

    def boom(self, probe):
        raise ProbeError("request timed out")

    monkeypatch.setattr(HttpClient, "_request", boom)
    out = tmp_path / "poison.lock"
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH),
        "--endpoint", "http://localhost:9/v1", "--model", "fake", "-o", str(out),
    ])
    assert result.exit_code == 2, result.output
    assert not out.exists()


def test_probe_bad_tools_file_is_exit_2(tmp_path):
    # Invalid input must exit 2 (not 1, the regression code), with no traceback.
    bad = tmp_path / "bad.json"
    bad.write_text('[{"function": {"description": "no name"}}]')
    result = runner.invoke(app, ["probe", "--tools", str(bad), "--simulate", str(_SIM_Q8)])
    assert result.exit_code == 2, result.output


def test_probe_missing_tools_file_is_exit_2(tmp_path):
    result = runner.invoke(
        app, ["probe", "--tools", str(tmp_path / "nope.json"), "--simulate", str(_SIM_Q8)]
    )
    assert result.exit_code == 2, result.output


def test_probe_bad_simulate_profile_is_exit_2(tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    result = runner.invoke(app, ["probe", "--tools", str(TOOLS_PATH), "--simulate", str(junk)])
    assert result.exit_code == 2, result.output


def test_derive_bad_tools_file_is_exit_2(tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    result = runner.invoke(app, ["derive", "--tools", str(junk)])
    assert result.exit_code == 2, result.output


def test_probe_unknown_via_is_exit_2():
    result = runner.invoke(app, ["probe", "--tools", str(TOOLS_PATH), "--via", "bogus", "--model", "m"])
    assert result.exit_code == 2, result.output


def test_probe_via_anyllm_missing_package_is_exit_2():
    if importlib.util.find_spec("any_llm") is not None:
        pytest.skip("any_llm installed")
    result = runner.invoke(
        app, ["probe", "--tools", str(TOOLS_PATH), "--via", "anyllm", "--model", "anthropic/claude"]
    )
    assert result.exit_code == 2, result.output  # clean error, not a traceback/exit 1


def test_init_scaffolds_files(tmp_path):
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "probelock.tools.json").exists()
    assert (tmp_path / ".github" / "workflows" / "probelock.yml").exists()
    # idempotent: second run skips without --force
    again = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert "skipped" in again.output


def test_init_force_overwrites_existing_file(tmp_path):
    runner.invoke(app, ["init", "--path", str(tmp_path)])
    target = tmp_path / "probelock.tools.json"
    target.write_text("MODIFIED-BY-TEST")
    result = runner.invoke(app, ["init", "--path", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert target.read_text() != "MODIFIED-BY-TEST"
    assert "created" in result.output


def test_probe_happy_path_writes_correct_lockfile(tmp_path):
    out = tmp_path / "q8.lock"
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
        "--label", "test-label", "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "test-label" in result.output
    data = json.loads(out.read_text())
    assert data["label"] == "test-label"
    assert data["n_probes"] > 0
    assert all(v == 1.0 for v in data["capabilities"].values())


def test_probe_success_banner_escapes_label_with_brackets(tmp_path):
    # Rich markup treats [..] specially; an unescaped label used to silently drop
    # everything from the first bracket onward.
    out = tmp_path / "q8.lock"
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
        "--label", "llama-8b[Q4_K_M]", "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "llama-8b[Q4_K_M]" in result.output


def test_derive_happy_path_lists_probes():
    result = runner.invoke(app, ["derive", "--tools", str(TOOLS_PATH)])
    assert result.exit_code == 0, result.output
    assert "probes derived" in result.output
    assert "tool_selection" in result.output


def test_derive_with_traces_includes_traced_probes():
    without = runner.invoke(app, ["derive", "--tools", str(TOOLS_PATH)])
    with_traces = runner.invoke(
        app, ["derive", "--tools", str(TOOLS_PATH), "--traces", str(_SAMPLE_TRACES)]
    )
    assert with_traces.exit_code == 0, with_traces.output
    assert "::traced::" in with_traces.output
    assert "::traced::" not in without.output


def test_probe_with_traces_blends_probes_and_warns(tmp_path):
    out = tmp_path / "blend.lock"
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
        "--traces", str(_SAMPLE_TRACES), "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.split())
    assert "derived from real trace content" in flattened
    assert "review before committing" in flattened
    data = json.loads(out.read_text())
    assert data["traces_fingerprint"] is not None
    probe_ids = {r["probe_id"] for r in data["results"]}
    assert any("::traced::" in pid for pid in probe_ids)
    assert any("::traced::" not in pid for pid in probe_ids)  # synthetic probes still present


def test_probe_without_traces_leaves_traces_fingerprint_none(tmp_path):
    out = tmp_path / "no_traces.lock"
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8), "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["traces_fingerprint"] is None


def test_probe_bad_traces_file_is_exit_2(tmp_path):
    bad = tmp_path / "bad_traces.json"
    bad.write_text(json.dumps({"not": "records"}))
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8), "--traces", str(bad),
    ])
    assert result.exit_code == 2, result.output


def test_diff_notes_flag_differing_trace_inputs(tmp_path):
    def write(path, traces_fp):
        path.write_text(json.dumps({
            "model": "m", "quant": "", "runtime": "ollama", "tools_fingerprint": "fp",
            "traces_fingerprint": traces_fp,
            "capabilities": {"tool_selection": 1.0}, "results": [], "n_probes": 0,
        }))

    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    write(b, "trace-fp-1")
    write(c, "trace-fp-2")
    result = runner.invoke(app, ["diff", str(b), str(c)])
    assert result.exit_code == 0, result.output
    assert "trace inputs differ" in result.output


def test_version_prints_version_string():
    from probelock import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_probe_tools_file_wrong_top_level_type_is_exit_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    result = runner.invoke(app, ["probe", "--tools", str(bad), "--simulate", str(_SIM_Q8)])
    assert result.exit_code == 2, result.output
    assert "JSON array" in result.output


def test_probe_tools_path_is_a_directory_is_exit_2(tmp_path):
    # A directory (an easy typo'd path) must exit cleanly, not crash with a raw
    # IsADirectoryError traceback.
    result = runner.invoke(app, ["probe", "--tools", str(tmp_path), "--simulate", str(_SIM_Q8)])
    assert result.exit_code == 2, result.output
    assert "Could not read tools file" in result.output


def test_diff_unknown_format_is_exit_2(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    _write_lock(b, "m", {"tool_selection": 1.0})
    _write_lock(c, "m", {"tool_selection": 1.0})
    result = runner.invoke(app, ["diff", str(b), str(c), "--format", "bogus"])
    assert result.exit_code == 2, result.output


def test_diff_notes_warn_on_differing_sample_counts(tmp_path):
    def write(path, samples):
        path.write_text(json.dumps({
            "model": "m", "quant": "", "runtime": "ollama", "tools_fingerprint": "fp",
            "capabilities": {"tool_selection": 1.0}, "results": [], "n_probes": 0,
            "samples": samples,
        }))

    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    write(b, 1)
    write(c, 5)
    result = runner.invoke(app, ["diff", str(b), str(c)])
    assert result.exit_code == 0, result.output
    assert "sample counts differ" in result.output


def test_diff_notes_flag_error_derived_negative_capability(tmp_path):
    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    b.write_text(json.dumps({
        "model": "m", "quant": "", "runtime": "ollama", "tools_fingerprint": "fp",
        "capabilities": {"tool_restraint": 1.0},
        "results": [{"probe_id": "tool_restraint::a", "capability": "tool_restraint",
                      "score": 1.0, "error": "model does not support tools"}],
        "n_probes": 1,
    }))
    c.write_text(json.dumps({
        "model": "m", "quant": "", "runtime": "ollama", "tools_fingerprint": "fp",
        "capabilities": {"tool_restraint": 1.0},
        "results": [{"probe_id": "tool_restraint::a", "capability": "tool_restraint", "score": 1.0}],
        "n_probes": 1,
    }))
    result = runner.invoke(app, ["diff", str(b), str(c)])
    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.split())  # the rich table wraps at terminal width
    assert "errored at the API level" in flattened
    assert "low-confidence" in flattened


def test_gate_confidence_marks_noisy_drop_and_still_passes(tmp_path):
    def write(path, results):
        path.write_text(json.dumps({
            "model": "m", "quant": "", "runtime": "ollama", "tools_fingerprint": "fp",
            "capabilities": {"tool_selection": sum(r["score"] for r in results) / len(results)},
            "results": results, "n_probes": len(results), "samples": 1,
        }))

    b, c = tmp_path / "b.lock", tmp_path / "c.lock"
    # 3 probes x 1 sample -> a drop past max_drop that isn't statistically significant.
    write(b, [{"probe_id": f"tool_selection::{i}", "capability": "tool_selection", "score": 1.0}
              for i in range(3)])
    write(c, [{"probe_id": "tool_selection::0", "capability": "tool_selection", "score": 1.0},
              {"probe_id": "tool_selection::1", "capability": "tool_selection", "score": 1.0},
              {"probe_id": "tool_selection::2", "capability": "tool_selection", "score": 0.0}])
    result = runner.invoke(app, ["gate", "-b", str(b), "-c", str(c), "--confidence", "0.95"])
    assert result.exit_code == 0, result.output
    assert "noisy" in result.output


# --- trace ingestion: ingest -> review -> replay ---------------------------------

_AGENT_LOG = _ROOT / "fixtures" / "sample_agent_log.jsonl"


def _ingest(tmp_path, *extra):
    out = tmp_path / "mined.json"
    result = runner.invoke(app, ["ingest", str(_AGENT_LOG), "--out", str(out), *extra])
    assert result.exit_code == 0, result.output
    return out


def test_ingest_mines_the_fixture_log_all_pending(tmp_path):
    out = _ingest(tmp_path)
    data = json.loads(out.read_text())
    assert {p["status"] for p in data["probes"]} == {"pending"}
    cats = [p["category"] for p in data["probes"]]
    assert cats.count("schema_validity") == 5
    assert cats.count("tool_selection") == 2  # one continuation- one agreement-confirmed
    assert cats.count("no_tool") == 1
    assert all(p["id"].startswith("trace:") for p in data["probes"])


def test_ingest_reports_skips_and_next_step(tmp_path):
    out = tmp_path / "mined.json"
    result = runner.invoke(app, ["ingest", str(_AGENT_LOG), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "failed_status" in result.output
    assert "forced_tool_choice" in result.output
    assert "traces review" in result.output


def test_ingest_bad_inputs_exit_2(tmp_path):
    out = tmp_path / "mined.json"
    for args in (
        ["ingest", str(tmp_path / "nope.jsonl"), "--out", str(out)],
        ["ingest", str(_AGENT_LOG), "--out", str(out), "--format", "csv"],
        ["ingest", str(_AGENT_LOG), "--out", str(out), "--redact-patterns", "ssn"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, result.output


def test_review_auto_accept_covers_only_schema_validity(tmp_path):
    out = _ingest(tmp_path)
    result = runner.invoke(app, ["traces", "review", str(out), "--auto-accept", "schema-validity"])
    assert result.exit_code == 0, result.output
    by_status = {}
    for p in json.loads(out.read_text())["probes"]:
        by_status.setdefault(p["status"], []).append(p["category"])
    assert sorted(by_status["accepted"]) == ["schema_validity"] * 5
    assert len(by_status["pending"]) == 3

    result = runner.invoke(app, ["traces", "review", str(out), "--auto-accept", "tool_selection"])
    assert result.exit_code == 2, result.output  # inferred category: review or the footgun pair


def test_review_auto_accept_all_needs_acknowledgement_and_skips_no_tool(tmp_path):
    out = _ingest(tmp_path)
    result = runner.invoke(app, ["traces", "review", str(out), "--auto-accept-all"])
    assert result.exit_code == 2, result.output

    result = runner.invoke(
        app, ["traces", "review", str(out), "--auto-accept-all", "--i-know-what-im-doing"])
    assert result.exit_code == 0, result.output
    assert "no auto-accept path" in result.output
    statuses = {p["category"]: p["status"] for p in json.loads(out.read_text())["probes"]}
    assert statuses["schema_validity"] == "accepted"
    assert statuses["tool_selection"] == "accepted"
    assert statuses["no_tool"] == "pending"  # never batch-accepted


def test_review_interactive_accept_reject_quit(tmp_path):
    out = _ingest(tmp_path)
    result = runner.invoke(app, ["traces", "review", str(out)], input="y\nn\nq\n")
    assert result.exit_code == 0, result.output
    statuses = [p["status"] for p in json.loads(out.read_text())["probes"]]
    assert statuses.count("accepted") == 1
    assert statuses.count("rejected") == 1
    assert statuses.count("pending") == 6  # progress saved, rest untouched


def test_review_interactive_refuses_accept_all_for_no_tool(tmp_path):
    out = _ingest(tmp_path)
    # probes are sorted no_tool first: 'a' on it must refuse and re-prompt
    result = runner.invoke(app, ["traces", "review", str(out)], input="a\nq\n")
    assert result.exit_code == 0, result.output
    assert "no accept-all path" in result.output
    assert all(p["status"] == "pending" for p in json.loads(out.read_text())["probes"])


def test_review_interactive_edit_expected_tool(tmp_path):
    out = _ingest(tmp_path)
    # skip the no_tool + 5 schema_validity probes; edit the first tool_selection probe
    result = runner.invoke(app, ["traces", "review", str(out)],
                           input="s\n" * 6 + "e\nget_weather\nq\n")
    assert result.exit_code == 0, result.output
    edited = [p for p in json.loads(out.read_text())["probes"]
              if p["category"] == "tool_selection" and p["status"] == "accepted"]
    assert len(edited) == 1
    assert edited[0]["check"] == {"type": "calls_tool", "tool": "get_weather"}
    assert edited[0]["provenance"]["edited_from"] == "convert_currency"


def test_probe_mined_excludes_sensitive_from_lockfile_unless_allowed(tmp_path):
    out = _ingest(tmp_path)
    runner.invoke(app, ["traces", "review", str(out), "--auto-accept", "schema_validity"])
    lock = tmp_path / "cand.lock"

    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
        "--mined", str(out), "-o", str(lock)])
    assert result.exit_code == 0, result.output
    assert "sensitive" in result.output
    caps = json.loads(lock.read_text())["capabilities"]
    assert not any(c.startswith("traced_") for c in caps)

    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
        "--mined", str(out), "--allow-sensitive", "-o", str(lock)])
    assert result.exit_code == 0, result.output
    data = json.loads(lock.read_text())
    assert "traced_schema_validity" in data["capabilities"]  # reported separately
    assert "tool_selection" in data["capabilities"]  # synthetic battery still intact
    assert data["traces_fingerprint"]


def test_probe_mined_with_nothing_accepted_warns_about_review(tmp_path):
    out = _ingest(tmp_path)  # everything still pending
    lock = tmp_path / "cand.lock"
    result = runner.invoke(app, [
        "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
        "--mined", str(out), "-o", str(lock)])
    assert result.exit_code == 0, result.output
    assert "pending review" in result.output
    assert not any(c.startswith("traced_")
                   for c in json.loads(lock.read_text())["capabilities"])


def test_derive_shows_accepted_mined_probes(tmp_path):
    out = _ingest(tmp_path)
    runner.invoke(app, ["traces", "review", str(out), "--auto-accept", "schema_validity"])
    result = runner.invoke(app, ["derive", "--tools", str(TOOLS_PATH), "--mined", str(out)])
    assert result.exit_code == 0, result.output
    assert "traced_schema_validity" in result.output


def test_probe_combines_traces_and_mined_fingerprints(tmp_path):
    out = _ingest(tmp_path)
    runner.invoke(app, ["traces", "review", str(out), "--auto-accept", "schema_validity"])

    def fingerprint(*extra):
        lock = tmp_path / "fp.lock"
        result = runner.invoke(app, [
            "probe", "--tools", str(TOOLS_PATH), "--simulate", str(_SIM_Q8),
            "--allow-sensitive", "-o", str(lock), *extra])
        assert result.exit_code == 0, result.output
        return json.loads(lock.read_text())["traces_fingerprint"]

    traces_only = fingerprint("--traces", str(_SAMPLE_TRACES))
    mined_only = fingerprint("--mined", str(out))
    both = fingerprint("--traces", str(_SAMPLE_TRACES), "--mined", str(out))
    assert traces_only and mined_only and both
    # each source contributes: the combined print differs from either alone, so a
    # diff's traces_changed note fires when EITHER trace input changes
    assert len({traces_only, mined_only, both}) == 3


def test_proxy_flag_validation_exits_2(tmp_path):
    out = tmp_path / "t.jsonl"
    for args in (
        ["proxy", "--upstream", "http://127.0.0.1:9", "--out", str(out), "--listen", "nope"],
        ["proxy", "--upstream", "ftp://bad", "--out", str(out), "--listen", "127.0.0.1:0"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, result.output


def test_ingest_accepts_multiple_logs_as_one_corpus(tmp_path):
    # Rotated proxy segments must stitch as one corpus: split the fixture log in two
    # and expect the exact same probes as ingesting the whole file.
    lines = _AGENT_LOG.read_text().splitlines()
    a, b = tmp_path / "part-a.jsonl", tmp_path / "part-b.jsonl"
    a.write_text(lines[0] + "\n")               # session A's turn 0 ...
    b.write_text("\n".join(lines[1:]) + "\n")  # ... and its confirming turn 1, apart
    out = tmp_path / "mined.json"
    result = runner.invoke(app, ["ingest", str(a), str(b), "--out", str(out)])
    assert result.exit_code == 0, result.output
    cats = sorted(p["category"] for p in json.loads(out.read_text())["probes"])
    assert cats.count("tool_selection") == 2  # incl. the continuation across files
    assert cats.count("schema_validity") == 5
    assert cats.count("no_tool") == 1


# --- --tools optional when traces/mined supply probes (validation rough edge) -------


def test_probe_traces_only_needs_no_tools_file(tmp_path):
    lock = tmp_path / "traced.lock"
    result = runner.invoke(app, [
        "probe", "--traces", str(_SAMPLE_TRACES), "--simulate", str(_SIM_Q8),
        "-o", str(lock)])
    assert result.exit_code == 0, result.output
    data = json.loads(lock.read_text())
    assert data["tools_fingerprint"] == ""  # no schema battery ran
    assert data["traces_fingerprint"]
    assert "tool_selection" in data["capabilities"]  # traced probes replayed
    assert data["n_probes"] > 0


def test_derive_traces_only_needs_no_tools_file():
    result = runner.invoke(app, ["derive", "--traces", str(_SAMPLE_TRACES)])
    assert result.exit_code == 0, result.output
    assert "::traced::" in result.output


def test_probe_mined_only_needs_no_tools_file(tmp_path):
    mined = _ingest(tmp_path)
    runner.invoke(app, ["traces", "review", str(mined), "--auto-accept", "schema_validity"])
    lock = tmp_path / "mined-only.lock"
    result = runner.invoke(app, [
        "probe", "--mined", str(mined), "--simulate", str(_SIM_Q8),
        "--allow-sensitive", "-o", str(lock)])
    assert result.exit_code == 0, result.output
    caps = json.loads(lock.read_text())["capabilities"]
    assert list(caps) == ["traced_schema_validity"]


def test_probe_without_any_source_exits_2():
    result = runner.invoke(app, ["probe", "--simulate", str(_SIM_Q8)])
    assert result.exit_code == 2, result.output
    assert "--tools, --traces, or --mined" in result.output
    result = runner.invoke(app, ["derive"])
    assert result.exit_code == 2, result.output


def test_probe_with_empty_battery_exits_2(tmp_path):
    # a mined file with nothing accepted and no --tools: the battery is empty and a
    # zero-probe lockfile can gate nothing
    mined = _ingest(tmp_path)  # everything still pending
    result = runner.invoke(app, [
        "probe", "--mined", str(mined), "--simulate", str(_SIM_Q8)])
    assert result.exit_code == 2, result.output
    assert "empty" in result.output
