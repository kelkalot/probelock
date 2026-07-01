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
