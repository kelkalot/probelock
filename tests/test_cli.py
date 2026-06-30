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
